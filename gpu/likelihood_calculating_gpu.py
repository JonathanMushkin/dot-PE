"""
GPU port of LikelihoodCalculator static methods.

Standalone functions first; GPULikelihoodCalculator is a one-liner dispatch shim.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from dot_pe.likelihood_calculating import LikelihoodCalculator
from gpu.gpu_constants import COMPLEX_DTYPE, DEVICE, REAL_DTYPE


def _to_cuda(arr: np.ndarray, dtype: torch.dtype = COMPLEX_DTYPE) -> torch.Tensor:
    np_dtype = np.complex64 if dtype == COMPLEX_DTYPE else np.float32
    return torch.from_numpy(np.asarray(arr, dtype=np_dtype)).to(DEVICE, dtype=dtype, non_blocking=True)


# ---------------------------------------------------------------------------
# get_dh_by_mode
# ---------------------------------------------------------------------------

def get_dh_by_mode_gpu(
    dh_weights_dmpb, h_impb, response_dpe, timeshift_dbe, asd_drift_d
):
    """
    GPU port of LikelihoodCalculator.get_dh_by_mode.
    Replaces the per-mode np.dot loop with a single batched torch.matmul.

    Returns
    -------
    dh_iem : numpy.ndarray, shape (i, e, m)
    """
    i, m, p, b = h_impb.shape
    d, _, e = response_dpe.shape
    x = d * p * b

    # Build the (m, i, x) "left" factor
    dh_weights_mx = np.moveaxis(dh_weights_dmpb, 0, 1).reshape(m, x)      # (m, x)
    h_conj_mix = (
        np.moveaxis(h_impb.conj(), 0, 1)[:, :, None, ...]                  # (m, i, 1, p, b)
        .repeat(d, axis=2)                                                  # (m, i, d, p, b)
        .reshape(m, i, x)                                                   # (m, i, x)
    )
    # product_mix[m, i, x] = dh_weights_mx[m, x] * h_conj_mix[m, i, x]
    product_mix_np = dh_weights_mx[:, None, :] * h_conj_mix                # (m, i, x)

    # Build the (x, e) "right" factor
    response_drift_dpe = response_dpe * asd_drift_d[:, None, None] ** -2   # (d, p, e)
    response_drift_xe = (
        response_drift_dpe[:, :, None, :].repeat(b, axis=2).reshape(x, e)  # (x, e)
    )
    timeshift_xe = np.reshape(
        timeshift_dbe.conj()[:, None, :, :].repeat(p, axis=1), (x, e)
    )                                                                        # (x, e)
    ext_tensor_np = timeshift_xe * response_drift_xe                        # (x, e)

    # Upload to GPU
    product_mix_g = _to_cuda(product_mix_np, COMPLEX_DTYPE)                # (m, i, x)
    ext_tensor_g = _to_cuda(ext_tensor_np, COMPLEX_DTYPE)                  # (x, e)

    # Batched matmul: (m, i, x) @ (x, e) → (m, i, e)
    dh_mie_g = torch.matmul(product_mix_g, ext_tensor_g)                   # (m, i, e)

    torch.cuda.synchronize()
    dh_mie = dh_mie_g.cpu().numpy()
    # (m, i, e) → (i, e, m)
    return np.moveaxis(dh_mie, (0, 1), (2, 0))


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
    Uses three torch.einsum calls with the same subscripts as the CPU version.

    Returns
    -------
    hh_iem : numpy.ndarray, shape (i, e, m)
    """
    m_inds_list = list(m_inds)
    mp_inds_list = list(mprime_inds)

    # Upload to GPU
    h_g = _to_cuda(h_impb, COMPLEX_DTYPE)                                  # (i,m,p,b)
    resp_g = _to_cuda(response_dpe, COMPLEX_DTYPE)                         # (d,p,e)
    hh_w_g = _to_cuda(hh_weights_dmppb, COMPLEX_DTYPE)                     # (d,m,p,P,b)
    drift_g = _to_cuda(asd_drift_d.astype(np.float32), REAL_DTYPE)         # (d,)

    h_mode_g = h_g[:, m_inds_list, :, :]                                   # (i,m,p,b)
    h_mode_conj_g = h_g.conj()[:, mp_inds_list, :, :]                      # (i,m,P,b)

    # "dmpPb, impb, imPb -> idmpP"
    hh_idmpP_g = torch.einsum(
        "dmpPb, impb, imPb -> idmpP",
        hh_w_g,
        h_mode_g,
        h_mode_conj_g,
    )                                                                        # (i,d,m,p,P)

    # "dpe, dPe, d -> dpPe"
    ff_dppe_g = torch.einsum(
        "dpe, dPe, d -> dpPe",
        resp_g,
        resp_g,
        drift_g.to(COMPLEX_DTYPE) ** -2,
    )                                                                        # (d,p,P,e)

    # "idmpP, dpPe -> iem"
    hh_iem_g = torch.einsum("idmpP, dpPe -> iem", hh_idmpP_g, ff_dppe_g)  # (i,e,m)

    torch.cuda.synchronize()
    return hh_iem_g.cpu().numpy()


# ---------------------------------------------------------------------------
# get_dh_hh_phi_grid
# ---------------------------------------------------------------------------

def get_dh_hh_phi_grid_gpu(dh_iem, hh_iem, m_arr, m_inds, mprime_inds, n_phi):
    """
    GPU port of LikelihoodCalculator.get_dh_hh_phi_grid.

    Returns
    -------
    dh_ieo : numpy.ndarray, shape (i, e, n_phi)
    hh_ieo : numpy.ndarray, shape (i, e, n_phi)
    """
    phi_grid = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dh_phasor = np.exp(-1j * np.outer(m_arr, phi_grid))           # (m, o)
    hh_phasor = np.exp(
        1j * np.outer(m_arr[list(m_inds)] - m_arr[list(mprime_inds)], phi_grid)
    )                                                              # (m, o)

    dh_iem_g = _to_cuda(dh_iem, COMPLEX_DTYPE)                    # (i, e, m)
    hh_iem_g = _to_cuda(hh_iem, COMPLEX_DTYPE)                    # (i, e, m)
    dh_phasor_g = _to_cuda(dh_phasor, COMPLEX_DTYPE)              # (m, o)
    hh_phasor_g = _to_cuda(hh_phasor, COMPLEX_DTYPE)              # (m, o)

    # "iem, mo -> ieo"
    dh_ieo_g = torch.einsum("iem, mo -> ieo", dh_iem_g, dh_phasor_g).real  # (i,e,o)
    hh_ieo_g = torch.einsum("iem, mo -> ieo", hh_iem_g, hh_phasor_g).real  # (i,e,o)

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
