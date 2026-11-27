"""Incoherent (single-detector) stage of dot-PE on the GPU.

Device counterpart of ``dot_pe.inference.run_for_single_detector`` over all detectors, plus the
linear-free time shift of ``IntrinsicSampleProcessor.get_relative_linfree_dt_from_waveform``:

1. h = amp exp(i phase); the time shift dt_i is the slope of a weighted linear fit of the
   unwrapped phase of h(2, 2, +) / h_ref against frequency, dt = -slope / (2 pi), with numpy
   ``Polynomial.fit`` weights w = polyfit_weights sqrt(|ratio|) (least squares in w * residual);
   the phase gets + 2 pi dt f.
2. Per detector: <d|h> on a grid of n_t arrival times (above the detector), and <h|h> per
   polarization pair, on the phase grid; per (phase, time) the 2x2 polarization solve
   r = <h|h>^-1 <d|h>, lnl = r . <d|h> / 2, masked where det <h|h> <= 0 or |r| > 1
   (best-fit distance below 1 Mpc); the template's lnlike is the maximum over (phase, time).

The CPU setup objects (one ``SingleDetectorProcessor`` per detector, built once) supply the
relative-binning weights; nothing per template runs on the CPU.
"""

from __future__ import annotations

import itertools

import numpy as np

_KERNEL_SRC = r"""
#define NM 4
#define NMM 10
#define INF_F __int_as_float(0x7f800000)

// One thread per (template i, phase o): max over the time grid of the polarization-solved lnlike.
extern "C" __global__ void k_single_det(
        const float2* __restrict__ dh,   // (i, m, p, t)
        const float2* __restrict__ hh,   // (i, M, p, P)
        const float2* __restrict__ dph,  // (m, o)
        const float2* __restrict__ hph,  // (M, o)
        int n_i, int nphi, int nt, float* __restrict__ out_io) {
    long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= (long long)n_i * nphi) return;
    int i = (int)(k / nphi), o = (int)(k % nphi);
    float h[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
    #pragma unroll
    for (int M = 0; M < NMM; ++M) {
        float2 f = hph[M * nphi + o];
        #pragma unroll
        for (int p = 0; p < 2; ++p)
            #pragma unroll
            for (int P = 0; P < 2; ++P) {
                float2 v = hh[((i * NMM + M) * 2 + p) * 2 + P];
                h[p][P] += v.x * f.x - v.y * f.y;
            }
    }
    // determinant and inverse in double: near-singular <h|h> is the case the mask exists for
    double det = (double)h[0][0] * h[1][1] - (double)h[0][1] * h[1][0];
    float best = -INF_F;
    if (det > 0.0) {
        float inv00 = (float)(h[1][1] / det), inv01 = (float)(-h[0][1] / det);
        float inv10 = (float)(-h[1][0] / det), inv11 = (float)(h[0][0] / det);
        float2 f[NM];
        #pragma unroll
        for (int m = 0; m < NM; ++m) f[m] = dph[m * nphi + o];
        const float2* d0 = dh + (long long)i * NM * 2 * nt;
        for (int t = 0; t < nt; ++t) {
            float a0 = 0.f, a1 = 0.f;
            #pragma unroll
            for (int m = 0; m < NM; ++m) {
                float2 v0 = d0[(m * 2 + 0) * nt + t], v1 = d0[(m * 2 + 1) * nt + t];
                a0 += v0.x * f[m].x - v0.y * f[m].y;
                a1 += v1.x * f[m].x - v1.y * f[m].y;
            }
            float r0 = inv00 * a0 + inv01 * a1, r1 = inv10 * a0 + inv11 * a1;
            float lnl = 0.5f * (r0 * a0 + r1 * a1);
            if (r0 * r0 + r1 * r1 <= 1.f) best = fmaxf(best, lnl);  // d_best = 1/|r| >= 1 Mpc
        }
    }
    out_io[k] = best;
}
"""


def linfree_dt(xp, h2plus, h2plus0, polyfit_w, fbin):
    """Time shift [s] per template: -slope / (2 pi) of numpy ``Polynomial.fit(fbin, dphase, 1, w)``,
    with dphase the unwrapped phase of h2plus / h2plus0 and w = polyfit_w sqrt(|ratio|).
    ``fit`` minimizes sum (w residual)^2, hence the weights w^2 below. xp: numpy or cupy."""
    ratio = h2plus / h2plus0[None]
    dphase = xp.unwrap(xp.angle(ratio), axis=-1)
    w2 = (polyfit_w[None] * xp.sqrt(xp.abs(ratio))) ** 2
    x = fbin[None]
    sw = w2.sum(axis=1, keepdims=True)
    xm = (w2 * x).sum(axis=1, keepdims=True) / sw
    ym = (w2 * dphase).sum(axis=1, keepdims=True) / sw
    slope = (w2 * (x - xm) * (dphase - ym)).sum(axis=1) / (w2 * (x - xm) ** 2).sum(axis=1)
    return -slope / (2 * np.pi)


def single_detector_timeshifts(sdp, par_dic_0, n_t):
    """timeshifts_dbt of ``run_for_single_detector`` (sky position above the detector)."""
    from cogwheel import skyloc_angles
    from cogwheel.gw_utils import DETECTORS, get_geocenter_delays
    det_name = sdp.likelihood.event_data.detector_names[0]
    tgps = sdp.likelihood.event_data.tgps
    lat, lon = skyloc_angles.cart3d_to_latlon(skyloc_angles.normalize(DETECTORS[det_name].location))
    par_dic = sdp.transform_par_dic_by_sky_poisition(det_name, par_dic_0, lon, lat, tgps)
    delay = get_geocenter_delays(det_name, par_dic["lat"], par_dic["lon"])[0]
    tcoarse = sdp.likelihood.event_data.tcoarse
    t_grid = (np.arange(n_t) - n_t // 2) * (sdp.likelihood.event_data.times[1])
    t_grid += par_dic["t_geocenter"] + tcoarse + delay
    return np.exp(-2j * np.pi * t_grid[None, None, :] * sdp.likelihood.fbin[None, :, None])


class IncoherentGPU:
    """Per-detector device constants; ``run_batch`` scores a batch of templates on all detectors.

    Parameters
    ----------
    sdps : list of ``SingleDetectorProcessor``, one per detector, in the event's detector order
        (``dot_pe.inference._create_single_detector_processor``).
    par_dic_0, n_t, n_phi, m_arr : as in ``run_for_single_detector``.
    dt_likelihood : likelihood whose ``_h2plus0_fbin`` and ``_polyfit_weights`` define the
        linear-free time shift (default: the first detector's, as the CPU incoherent stage).
    """

    THREADS = 256

    def __init__(self, sdps, par_dic_0, n_t, n_phi, m_arr, dt_likelihood=None):
        import cupy as cp
        self.cp = cp
        self.m_arr = np.asarray(m_arr)
        if len(self.m_arr) != 4:
            raise NotImplementedError("kernels are compiled for 4 modes")
        self.n_t, self.n_phi = int(n_t), int(n_phi)
        lk = dt_likelihood or sdps[0].likelihood
        self.fbin_h = np.asarray(lk.fbin, float)
        self.fbin = cp.asarray(self.fbin_h)
        self.h2plus0 = cp.asarray(np.asarray(lk._h2plus0_fbin, np.complex128))
        self.polyfit_w = cp.asarray(np.asarray(lk._polyfit_weights, float))
        self.m2 = list(self.m_arr).index(2)
        self.m_inds, self.mprime_inds = map(
            list, zip(*itertools.combinations_with_replacement(range(4), 2)))
        phi = np.linspace(0, 2 * np.pi, self.n_phi, endpoint=False)
        self.dph = cp.asarray(np.exp(-1j * np.outer(self.m_arr, phi)).astype(np.complex64))
        self.hph = cp.asarray(np.exp(1j * np.outer(self.m_arr[self.m_inds] - self.m_arr[self.mprime_inds],
                                                   phi)).astype(np.complex64))
        self.dets = []
        for sdp in sdps:
            drift_inv2 = float(sdp.likelihood.asd_drift[0]) ** -2
            ts = single_detector_timeshifts(sdp, par_dic_0, self.n_t)[0]           # (b, t)
            self.dets.append({
                "dh_w": cp.asarray(np.asarray(sdp.dh_weights_dmpb[0]) * drift_inv2),   # (m, p, b) c128
                "hh_w": cp.asarray(np.asarray(sdp.hh_weights_dmppb[0]) * drift_inv2),  # (M, p, P, b)
                "tsc": cp.asarray(np.conj(ts).astype(np.complex64)),                   # (b, t)
            })
        self.kernel = cp.RawModule(code=_KERNEL_SRC, options=("--use_fast_math",)).get_function("k_single_det")

    def linfree_dt(self, h):
        """dt_i [s] of the linear-free time shift, from raw h (i, m, p, b) complex128."""
        return linfree_dt(self.cp, h[:, self.m2, 0], self.h2plus0, self.polyfit_w, self.fbin)

    def run_batch(self, amp, phase, dt=None, return_raw=False):
        """lnlike_di (d, i) on the host, dt_i (device), and h (i, m, p, b) complex64: with dt
        applied, or (``return_raw``) without it."""
        cp = self.cp
        h = cp.asarray(amp) * cp.exp(1j * cp.asarray(phase))
        h_raw = h.astype(cp.complex64) if return_raw else None
        if dt is None:
            dt = self.linfree_dt(h)
        else:
            dt = cp.asarray(dt)
        h = h * cp.exp(2j * np.pi * dt[:, None, None, None] * self.fbin[None, None, None, :])
        n_i = h.shape[0]
        hc = cp.conj(h)
        out = np.empty((len(self.dets), n_i))
        res = cp.empty(n_i * self.n_phi, cp.float32)
        grid = ((n_i * self.n_phi + self.THREADS - 1) // self.THREADS,)
        for d, det in enumerate(self.dets):
            x = (det["dh_w"][None] * hc).reshape(n_i * 8, -1).astype(cp.complex64)       # (i m p, b)
            dh = cp.ascontiguousarray(cp.matmul(x, det["tsc"]))                           # (i m p, t)
            hh = cp.einsum("MpPb, iMpb, iMPb -> iMpP", det["hh_w"], h[:, self.m_inds],
                           hc[:, self.mprime_inds]).astype(cp.complex64)
            hh = cp.ascontiguousarray(hh)
            self.kernel(grid, (self.THREADS,), (dh, hh, self.dph, self.hph, np.int32(n_i),
                                                np.int32(self.n_phi), np.int32(self.n_t), res))
            out[d] = res.reshape(n_i, self.n_phi).max(axis=1).get()
        return out, dt, (h_raw if return_raw else h.astype(cp.complex64))


def single_det_lnlike_numpy(dh_w, hh_w, h, tsc, n_phi, m_arr, dtype=np.float32):
    """Numpy mirror of ``k_single_det`` + the max over phase, for CPU tests.

    dh_w (m, p, b) and hh_w (M, p, P, b): weights with the ASD drift applied; h (i, m, p, b)
    complex128 with dt applied; tsc (b, t) = conj(timeshifts). Products in complex128, cast to
    ``dtype``'s complex type as on the device; determinant and inverse in float64.
    """
    m_arr = np.asarray(m_arr)
    ctype = np.complex64 if dtype == np.float32 else np.complex128
    m_inds, mp_inds = map(list, zip(*itertools.combinations_with_replacement(range(len(m_arr)), 2)))
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dph = np.exp(-1j * np.outer(m_arr, phi)).astype(ctype)
    hph = np.exp(1j * np.outer(m_arr[m_inds] - m_arr[mp_inds], phi)).astype(ctype)
    n_i = h.shape[0]
    x = (dh_w[None] * np.conj(h)).reshape(n_i * len(m_arr) * 2, -1).astype(ctype)
    dh = (x @ tsc.astype(ctype)).reshape(n_i, len(m_arr), 2, -1)                    # (i, m, p, t)
    hh = np.einsum("MpPb, iMpb, iMPb -> iMpP", hh_w, h[:, m_inds], np.conj(h[:, mp_inds])).astype(ctype)
    a = np.einsum("impt, mo -> iotp", dh, dph).real.astype(dtype)                    # (i, o, t, p)
    H = np.einsum("iMpP, Mo -> iopP", hh, hph).real.astype(dtype)                     # (i, o, p, P)
    det = H[..., 0, 0].astype(float) * H[..., 1, 1] - H[..., 0, 1].astype(float) * H[..., 1, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.stack([np.stack([H[..., 1, 1] / det, -H[..., 0, 1] / det], -1),
                        np.stack([-H[..., 1, 0] / det, H[..., 0, 0] / det], -1)], -2).astype(dtype)
    r = np.einsum("iopP, iotP -> iotp", inv, a)
    lnl = 0.5 * (r * a).sum(-1)
    ok = ((r ** 2).sum(-1) <= 1) & (det[..., None] > 0)
    return np.where(ok, lnl, -np.inf).max(axis=(1, 2))
