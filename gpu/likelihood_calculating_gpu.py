"""
GPU port of LikelihoodCalculator static methods.

Standalone functions first; GPULikelihoodCalculator is a one-liner dispatch shim.

Precision notes
---------------
* dh_weights: max ~4e24  — fits in float32
* hh_weights: max ~4e48  — EXCEEDS float32; hh einsum stays in complex128
* h_impb:     max ~4e-20 — fits in float32
* response:   max ~1     — fits in float32
* timeshift:  max ~1     — fits in float32

For get_dh_by_mode the combined product dh_weights * h_conj ~ 1e5 (fits).
For get_hh_by_mode hh_weights alone overflows float32; the einsum result
hh_weights * h * h_conj ~ 1e10 is safe to cast back to float32.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from dot_pe.likelihood_calculating import LikelihoodCalculator
from gpu.gpu_constants import COMPLEX_DTYPE, DEVICE, REAL_DTYPE


def _to_c64(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr, dtype=np.complex64)).to(
        DEVICE, dtype=COMPLEX_DTYPE, non_blocking=True
    )


def _to_c128(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr, dtype=np.complex128)).to(
        DEVICE, dtype=torch.complex128, non_blocking=True
    )


def _to_f32(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(
        DEVICE, dtype=REAL_DTYPE, non_blocking=True
    )


# ---------------------------------------------------------------------------
# get_dh_by_mode
# ---------------------------------------------------------------------------

def get_dh_by_mode_gpu(
    dh_weights_dmpb, h_impb, response_dpe, timeshift_dbe, asd_drift_d
):
    """
    GPU port of LikelihoodCalculator.get_dh_by_mode.

    dh_weights max ~4e24 fits in float32. All operations run in complex64.
    Replaces the per-mode np.dot loop with a single batched torch.matmul.

    Returns dh_iem : numpy.ndarray, shape (i, e, m)
    """
    i, m, p, b = h_impb.shape
    d, _, e = response_dpe.shape
    x = d * p * b

    # Build (m, i, x) product on CPU — result max ~1e5, safe in float32
    dh_weights_mx = np.moveaxis(dh_weights_dmpb, 0, 1).reshape(m, x)
    h_conj_mix = (
        np.moveaxis(h_impb.conj(), 0, 1)[:, :, None, ...]
        .repeat(d, axis=2)
        .reshape(m, i, x)
    )
    product_mix_np = dh_weights_mx[:, None, :] * h_conj_mix  # (m, i, x), max ~1e5

    # Build (x, e) extrinsic tensor on CPU — max ~1
    response_drift_dpe = response_dpe * asd_drift_d[:, None, None] ** -2
    response_drift_xe  = (
        response_drift_dpe[:, :, None, :].repeat(b, axis=2).reshape(x, e)
    )
    timeshift_xe = np.reshape(
        timeshift_dbe.conj()[:, None, :, :].repeat(p, axis=1), (x, e)
    )
    ext_tensor_np = timeshift_xe * response_drift_xe  # (x, e), max ~1

    # Upload and batched matmul: (m, i, x) @ (x, e) → (m, i, e)
    product_g    = _to_c64(product_mix_np)   # (m, i, x)
    ext_tensor_g = _to_c64(ext_tensor_np)    # (x, e)
    dh_mie_g     = torch.matmul(product_g, ext_tensor_g)

    torch.cuda.synchronize()
    dh_mie = dh_mie_g.cpu().numpy()
    return np.moveaxis(dh_mie, (0, 1), (2, 0))   # (i, e, m)


# ---------------------------------------------------------------------------
# get_hh_by_mode
# ---------------------------------------------------------------------------

def get_hh_by_mode_gpu(
    h_impb,
    response_dpe,
    hh_weights_dmppb,
    asd_drift_d,
    m_inds,
    mprime_inds,
):
    """
    GPU port of LikelihoodCalculator.get_hh_by_mode.

    hh_weights max ~4e48 overflows float32 → einsum in complex128.
    Result hh_weights * h * h_conj ~ 1e10 is cast back to complex64.

    Returns hh_iem : numpy.ndarray, shape (i, e, m)
    """
    m_inds_list  = list(m_inds)
    mp_inds_list = list(mprime_inds)

    # Upload: hh_weights in c128 (overflow-safe), rest in c64
    h_g      = _to_c64(h_impb)                                   # (i,m,p,b)
    resp_g   = _to_c64(response_dpe)                             # (d,p,e)
    hh_w_f64 = _to_c128(hh_weights_dmppb)                        # (d,m,p,P,b) c128
    drift_g  = _to_f32(asd_drift_d.astype(np.float32))           # (d,)

    h_mode_g      = h_g[:, m_inds_list,  :, :]                   # (i,m,p,b)
    h_mode_conj_g = h_g.conj()[:, mp_inds_list, :, :]            # (i,m,P,b)

    # einsum in complex128 to avoid overflow in hh_weights
    # result ~ 1e10 → safe to cast to complex64
    hh_idmpP_g = torch.einsum(
        "dmpPb, impb, imPb -> idmpP",
        hh_w_f64,
        h_mode_g.to(torch.complex128),
        h_mode_conj_g.to(torch.complex128),
    ).to(COMPLEX_DTYPE)                                           # (i,d,m,p,P) c64

    # ff_dppe: response * response * asd_drift^-2  — all small, c64 fine
    ff_dppe_g = torch.einsum(
        "dpe, dPe, d -> dpPe",
        resp_g,
        resp_g,
        drift_g.to(COMPLEX_DTYPE) ** -2,
    )                                                             # (d,p,P,e)

    hh_iem_g = torch.einsum("idmpP, dpPe -> iem", hh_idmpP_g, ff_dppe_g)

    torch.cuda.synchronize()
    return hh_iem_g.cpu().numpy()


# ---------------------------------------------------------------------------
# get_dh_hh_phi_grid
# ---------------------------------------------------------------------------

def get_dh_hh_phi_grid_gpu(dh_iem, hh_iem, m_arr, m_inds, mprime_inds, n_phi):
    """
    GPU port of LikelihoodCalculator.get_dh_hh_phi_grid.
    All inputs are small-magnitude outputs of dh/hh computations — c64 fine.

    Returns dh_ieo, hh_ieo : numpy.ndarray, shape (i, e, n_phi)
    """
    phi_grid  = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dh_phasor = np.exp(-1j * np.outer(m_arr, phi_grid))
    hh_phasor = np.exp(
        1j * np.outer(
            m_arr[list(m_inds)] - m_arr[list(mprime_inds)], phi_grid
        )
    )

    dh_iem_g    = _to_c64(dh_iem)
    hh_iem_g    = _to_c64(hh_iem)
    dh_phasor_g = _to_c64(dh_phasor)
    hh_phasor_g = _to_c64(hh_phasor)

    dh_ieo_g = torch.einsum("iem, mo -> ieo", dh_iem_g, dh_phasor_g).real
    hh_ieo_g = torch.einsum("iem, mo -> ieo", hh_iem_g, hh_phasor_g).real

    torch.cuda.synchronize()
    return dh_ieo_g.cpu().numpy(), hh_ieo_g.cpu().numpy()


# ---------------------------------------------------------------------------
# Drop-in subclass
# ---------------------------------------------------------------------------

class GPULikelihoodCalculator(LikelihoodCalculator):
    """Routes the three hot methods to GPU implementations."""

    get_dh_by_mode = staticmethod(get_dh_by_mode_gpu)
    get_hh_by_mode = staticmethod(get_hh_by_mode_gpu)

    def get_dh_hh_phi_grid(self, dh_iem, hh_iem):
        return get_dh_hh_phi_grid_gpu(
            dh_iem,
            hh_iem,
            self.m_arr,
            self.m_inds,
            self.mprime_inds,
            self.n_phi,
        )
