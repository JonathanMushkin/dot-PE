"""
GPU port of SingleDetectorProcessor.get_response_over_distance_and_lnlike.

All heavy math is in the standalone function get_response_over_distance_and_lnlike_gpu.
The GPUSingleDetectorProcessor subclass is just a one-line dispatch shim.

Precision notes
---------------
* dh_weights: max ~4e24  — fits comfortably in float32 (max ~3.4e38)
* hh_weights: max ~3e48  — EXCEEDS float32 range; kept in float64 on GPU
* h_impb:     max ~6e-20 — fits in float32
* timeshift:  max ~1     — fits in float32

Strategy:
  - dh path: complex64 throughout (fast)
  - hh einsum: complex128 (hh_weights * h * h_conj → result ~1e10, safe to
    cast back to complex64 afterwards)
  - 2×2 inversion: float64 (catastrophic cancellation when det is small)
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from dot_pe.single_detector import SingleDetectorProcessor
from gpu.gpu_constants import COMPLEX_DTYPE, DEVICE, REAL_DTYPE


def _to_c64(arr) -> torch.Tensor:
    """Upload a numpy array (or move a tensor) to GPU as complex64."""
    if isinstance(arr, torch.Tensor):
        return arr.to(DEVICE, dtype=COMPLEX_DTYPE, non_blocking=True)
    return torch.from_numpy(np.asarray(arr, dtype=np.complex64)).to(
        DEVICE, dtype=COMPLEX_DTYPE, non_blocking=True
    )


def _to_c128(arr: np.ndarray) -> torch.Tensor:
    """Upload a numpy array to GPU as complex128 (no down-cast)."""
    return torch.from_numpy(np.asarray(arr, dtype=np.complex128)).to(
        DEVICE, dtype=torch.complex128, non_blocking=True
    )


def get_response_over_distance_and_lnlike_gpu(
    dh_weights_dmpb,
    hh_weights_dmppb,
    h_impb,
    timeshift_dbt,
    asd_drift_d,
    n_phi,
    m_arr,
):
    """
    GPU port of SingleDetectorProcessor.get_response_over_distance_and_lnlike.

    Inputs and outputs are identical to the CPU version (numpy arrays).

    Returns
    -------
    r_iotp : numpy.ndarray, shape (i, n_phi, t, p)
    lnlike_iot : numpy.ndarray, shape (i, n_phi, t)
    """
    from dot_pe.single_detector import _get_phasors

    i, m, p, b = h_impb.shape
    t = timeshift_dbt.shape[-1]
    x = i * m * p
    y = i * t * p

    drift_inv2 = float(asd_drift_d[0] ** -2)

    dh_phasor_mo_np, hh_phasor_Mo_np = _get_phasors(n_phi, m_arr)
    dh_phasor_mo_np = np.ascontiguousarray(dh_phasor_mo_np)
    hh_phasor_Mo_np = np.ascontiguousarray(hh_phasor_Mo_np)

    m_inds_list, mprime_inds_list = zip(
        *itertools.combinations_with_replacement(range(m), 2)
    )
    m_inds_t  = list(m_inds_list)
    mp_inds_t = list(mprime_inds_list)

    # ---- Upload inputs (dh path → c64, hh weights → c128) ----
    h_g        = _to_c64(h_impb)                  # (i,m,p,b)
    ts_g       = _to_c64(timeshift_dbt[0])         # (b,t)
    dh_w_g     = _to_c64(dh_weights_dmpb[0])       # (m,p,b)
    hh_w_f64   = _to_c128(hh_weights_dmppb[0])     # (M,p,P,b) — overflow-safe
    dh_phasor_g = _to_c64(dh_phasor_mo_np)         # (m,o)
    hh_phasor_g = _to_c64(hh_phasor_Mo_np)         # (M,o)

    #########
    # <d|h>  — all float32
    #########
    h_conj_g   = h_g.conj()
    dh_impb_g  = dh_w_g * drift_inv2 * h_conj_g   # (i,m,p,b)

    dh_xb_g       = dh_impb_g.reshape(x, b).contiguous()
    ts_conj_bt_g  = ts_g.conj().contiguous()
    dh_xt_g       = torch.matmul(dh_xb_g, ts_conj_bt_g)   # (x,t)

    dh_impt_g  = dh_xt_g.reshape(i, m, p, t)
    dh_itpm_g  = dh_impt_g.permute(0, 3, 2, 1)            # (i,t,p,m)
    dh_ym_g    = dh_itpm_g.reshape(y, m).contiguous()
    dh_yo_g    = torch.matmul(dh_ym_g, dh_phasor_g)       # (y,o)
    dh_itpo_g  = dh_yo_g.real.reshape(i, t, p, n_phi)
    dh_iotp_g  = dh_itpo_g.permute(0, 3, 1, 2)            # (i,o,t,p)

    #########
    # <h|h>  — einsum in float64, result cast back to float32
    #########
    # hh_weights max ~3e48 overflows float32; h max ~6e-20.
    # Product hh_weights * h * h_conj ~ 1e10 — safe to cast back to float32.
    hh_wts_drift_f64 = hh_w_f64 * drift_inv2                         # (M,p,P,b) c128
    h_iMpb_f64      = h_g[:, m_inds_t,  :, :].to(torch.complex128)  # (i,M,p,b)
    h_iMpb_conj_f64 = h_conj_g[:, mp_inds_t, :, :].to(torch.complex128)

    hh_iMpp_g = torch.einsum(
        "MpPb, iMpb, iMPb -> iMpP",
        hh_wts_drift_f64,
        h_iMpb_f64,
        h_iMpb_conj_f64,
    ).to(COMPLEX_DTYPE)                                               # (i,M,p,P) → c64

    # (i,p,P,M) @ (M,o) → (i,p,P,o)
    hh_ippM_g  = hh_iMpp_g.permute(0, 2, 3, 1).contiguous()
    hh_ippo_g  = torch.matmul(hh_ippM_g, hh_phasor_g)
    hh_iopp_g  = hh_ippo_g.real.permute(0, 3, 1, 2)                  # (i,o,p,P) f32

    # 2×2 inverse — float64 to avoid catastrophic cancellation when det is small
    # (near-face-on binary: h+ ≈ hx → det ≈ 0; float32 gives ~1% rel error at
    #  det/diag ≈ 1e-5; float64 keeps it below 1e-10)
    hh_f64        = hh_iopp_g.double()
    det_recip_f64 = 1.0 / (
        hh_f64[..., 0, 0] * hh_f64[..., 1, 1]
        - hh_f64[..., 0, 1] * hh_f64[..., 1, 0]
    )
    hh_inv_f64 = torch.empty_like(hh_f64)
    hh_inv_f64[..., 0, 0] =  hh_f64[..., 1, 1] * det_recip_f64
    hh_inv_f64[..., 0, 1] = -hh_f64[..., 0, 1] * det_recip_f64
    hh_inv_f64[..., 1, 0] = -hh_f64[..., 1, 0] * det_recip_f64
    hh_inv_f64[..., 1, 1] =  hh_f64[..., 0, 0] * det_recip_f64
    hh_inv_g = hh_inv_f64.float()                                     # (i,o,p,P) f32

    ######################
    # Optimal solution
    ######################
    r_iotp_g    = torch.einsum("iopP, iotP -> iotp", hh_inv_g, dh_iotp_g)
    lnlike_iot_g = 0.5 * (r_iotp_g * dh_iotp_g).sum(dim=-1)

    torch.cuda.synchronize()
    return r_iotp_g.cpu().numpy(), lnlike_iot_g.cpu().numpy()


class GPUSingleDetectorProcessor(SingleDetectorProcessor):
    """Drop-in replacement: routes the hot static method to GPU."""

    @classmethod
    def get_response_over_distance_and_lnlike(
        cls,
        dh_weights_dmpb,
        hh_weights_dmppb,
        h_impb,
        timeshift_dbt,
        asd_drift_d,
        n_phi,
        m_arr,
    ):
        return get_response_over_distance_and_lnlike_gpu(
            dh_weights_dmpb,
            hh_weights_dmppb,
            h_impb,
            timeshift_dbt,
            asd_drift_d,
            n_phi,
            m_arr,
        )
