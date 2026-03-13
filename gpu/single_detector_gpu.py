"""
GPU port of SingleDetectorProcessor.get_response_over_distance_and_lnlike.

All heavy math is in the standalone function get_response_over_distance_and_lnlike_gpu.
The GPUSingleDetectorProcessor subclass is just a one-line dispatch shim.
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Allow running from repo root or gpu/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dot_pe.single_detector import SingleDetectorProcessor
from gpu.gpu_constants import COMPLEX_DTYPE, DEVICE, REAL_DTYPE


def _np_to_cuda(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr, dtype=np.complex64 if dtype == COMPLEX_DTYPE else np.float32)).to(DEVICE, dtype=dtype, non_blocking=True)


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
    Computation is performed in torch.complex64 on CUDA.

    Returns
    -------
    r_iotp : numpy.ndarray, shape (i, n_phi, t, p)
    lnlike_iot : numpy.ndarray, shape (i, n_phi, t)
    """
    import itertools
    from dot_pe.single_detector import _get_phasors  # cached phasor helper

    i, m, p, b = h_impb.shape
    t = timeshift_dbt.shape[-1]
    x = i * m * p
    y = i * t * p

    drift_inv2 = float(asd_drift_d[0] ** -2)

    # Phasors (numpy → CUDA)
    dh_phasor_mo_np, hh_phasor_Mo_np = _get_phasors(n_phi, m_arr)
    dh_phasor_mo_np = np.ascontiguousarray(dh_phasor_mo_np)
    hh_phasor_Mo_np = np.ascontiguousarray(hh_phasor_Mo_np)

    m_inds_list, mprime_inds_list = zip(*itertools.combinations_with_replacement(range(m), 2))
    M = len(m_inds_list)

    # Upload inputs to GPU
    h_g = _np_to_cuda(h_impb, COMPLEX_DTYPE)                    # (i,m,p,b)
    ts_g = _np_to_cuda(timeshift_dbt[0], COMPLEX_DTYPE)          # (b,t)
    dh_w_g = _np_to_cuda(dh_weights_dmpb[0], COMPLEX_DTYPE)      # (m,p,b)
    hh_w_g = _np_to_cuda(hh_weights_dmppb[0], COMPLEX_DTYPE)     # (M,p,P,b)  (M = mode-pairs)
    dh_phasor_g = _np_to_cuda(dh_phasor_mo_np, COMPLEX_DTYPE)    # (m,o)
    hh_phasor_g = _np_to_cuda(hh_phasor_Mo_np, COMPLEX_DTYPE)    # (M,o)

    #########
    # <d|h>
    #########
    h_conj_g = h_g.conj()
    dh_impb_g = dh_w_g * drift_inv2 * h_conj_g                  # (i,m,p,b)

    # (x, b) @ (b, t) → (x, t)
    dh_xb_g = dh_impb_g.reshape(x, b).contiguous()
    ts_conj_bt_g = ts_g.conj().contiguous()                       # (b,t)
    dh_xt_g = torch.matmul(dh_xb_g, ts_conj_bt_g)               # (x,t)

    dh_impt_g = dh_xt_g.reshape(i, m, p, t)
    # move (m, t) → (t, m): need shape (i,t,p,m) for the next matmul
    dh_itpm_g = dh_impt_g.permute(0, 3, 2, 1)                    # (i,t,p,m)

    # (y, m) @ (m, o) → (y, o)
    dh_ym_g = dh_itpm_g.reshape(y, m).contiguous()
    dh_yo_g = torch.matmul(dh_ym_g, dh_phasor_g)                # (y,o)
    dh_itpo_g = dh_yo_g.real.reshape(i, t, p, n_phi)
    dh_iotp_g = dh_itpo_g.permute(0, 3, 1, 2)                   # (i,o,t,p)

    #########
    # <h|h>
    #########
    hh_weights_drift_Mppb_g = hh_w_g * drift_inv2               # (M,p,P,b)

    # Index into mode pairs
    m_inds_t = list(m_inds_list)
    mp_inds_t = list(mprime_inds_list)

    h_iMpb_g = h_g[:, m_inds_t, :, :]                           # (i,M,p,b)
    h_iMpb_conj_g = h_conj_g[:, mp_inds_t, :, :]                # (i,M,P,b)

    # einsum "MpPb, iMpb, iMPb -> iMpP"
    hh_iMpp_g = torch.einsum(
        "MpPb, iMpb, iMPb -> iMpP",
        hh_weights_drift_Mppb_g,
        h_iMpb_g,
        h_iMpb_conj_g,
    )                                                             # (i,M,p,P)

    # (i,p,p,M) @ (M,o) → (i,p,p,o)
    hh_ippM_g = hh_iMpp_g.permute(0, 2, 3, 1).contiguous()      # (i,p,P,M)
    hh_ippo_g = torch.matmul(hh_ippM_g, hh_phasor_g)            # (i,p,P,o)
    hh_iopp_g = hh_ippo_g.real.permute(0, 3, 1, 2)              # (i,o,p,P)

    # 2×2 inverse per (i, o) — use float64 to avoid catastrophic cancellation.
    # The determinant is det(A) = a00*a11 - a01^2.  When the hh matrix is
    # nearly rank-1 (face-on binary, h+ ≈ hx), both terms are nearly equal
    # and their float32 difference has a relative error of up to ~1% at
    # det/diag ≈ 1e-5.  Upcasting to float64 costs a trivial amount of
    # memory and time (matrices are (i, o, 2, 2)) but preserves the same
    # accuracy as the CPU numpy path.
    hh_iopp_f64 = hh_iopp_g.double()
    det_recip_f64 = 1.0 / (
        hh_iopp_f64[..., 0, 0] * hh_iopp_f64[..., 1, 1]
        - hh_iopp_f64[..., 0, 1] * hh_iopp_f64[..., 1, 0]
    )
    hh_inv_f64 = torch.empty_like(hh_iopp_f64)
    hh_inv_f64[..., 0, 0] = hh_iopp_f64[..., 1, 1] * det_recip_f64
    hh_inv_f64[..., 0, 1] = -hh_iopp_f64[..., 0, 1] * det_recip_f64
    hh_inv_f64[..., 1, 0] = -hh_iopp_f64[..., 1, 0] * det_recip_f64
    hh_inv_f64[..., 1, 1] = hh_iopp_f64[..., 0, 0] * det_recip_f64
    # Cast back to float32 for the subsequent einsum
    hh_inv_g = hh_inv_f64.float()

    ########################
    # Optimal solutions
    ########################
    # r[i,o,t,p] = sum_P hh_inv[i,o,p,P] * dh[i,o,t,P]
    r_iotp_g = torch.einsum("iopP, iotP -> iotp", hh_inv_g, dh_iotp_g)
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
