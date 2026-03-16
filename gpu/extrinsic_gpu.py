"""
GPU-accelerated implementations of draw_extrinsic_samples hotspots.

Track H: GPU acceleration of draw_extrinsic_samples.

H1: np.bincount replacing scipy sparse in MarginalizationInfoSamplerFree.__post_init__
H2: GPU matmuls in CoherentScoreSamplerFree._get_dh_hh_qo
H3: GPU batched matmuls in MarginalizationExtrinsicSamplerFreeLikelihood._get_many_dh_hh
"""

import numpy as np


# ---------------------------------------------------------------------------
# H1: Fast __post_init__ — np.bincount replacing scipy sparse scatter-add
# ---------------------------------------------------------------------------

def _fast_post_init(self):
    """
    Drop-in replacement for MarginalizationInfoSamplerFree.__post_init__.

    Replaces two scipy.sparse.coo_array scatter-adds with np.bincount,
    eliminating sparse matrix construction overhead (~10s at 32k bank).
    All other logic is identical to the original.
    """
    from scipy.special import logsumexp
    from cogwheel.utils import exp_normalize, n_effective

    self.n_qmc = sum(self.proposals_n_qmc)

    if self.q_inds.size == 0:
        self.weights_q = np.array([])
        self.weights = np.array([])
        self.n_effective = 0.0
        self.lnl_marginalized = -np.inf
        self.n_effective_prior = 0.0
        self.prior_weights_q = np.array([])
        self.prior_weights = np.array([])
        return

    denominators = np.zeros(len(self.q_inds))
    total_n_qmc = sum(self.proposals_n_qmc)
    for n_qmc, proposal, w in zip(
        self.proposals_n_qmc, self.proposals, self.proposals_weights
    ):
        denominators += (
            w
            * (n_qmc / total_n_qmc)
            * np.prod(
                np.take_along_axis(proposal, self.tdet_inds, axis=1),
                axis=0,
            )
        )

    ln_weights = self.ln_numerators - np.log(denominators)
    self.weights = exp_normalize(ln_weights)

    # Scatter-add: sum weights per unique q index (replaces scipy sparse)
    weights_q_full = np.bincount(
        self.q_inds, weights=self.weights, minlength=self.n_qmc
    )
    self.weights_q = weights_q_full[weights_q_full > 0]

    self.n_effective = n_effective(self.weights_q)
    self.lnl_marginalized = logsumexp(ln_weights) - np.log(total_n_qmc)

    ln_prior_weights = self.ln_numerators_prior - np.log(denominators)
    self.prior_weights = np.exp(ln_prior_weights)

    prior_weights_q_full = np.bincount(
        self.q_inds, weights=self.prior_weights, minlength=self.n_qmc
    )
    self.prior_weights_q = prior_weights_q_full[prior_weights_q_full > 0]
    self.n_effective_prior = n_effective(self.prior_weights_q)


# ---------------------------------------------------------------------------
# H2: GPU _get_dh_hh_qo — phasor matmuls on GPU
# ---------------------------------------------------------------------------

def _get_dh_hh_qo_gpu(self, sky_inds, q_inds, t_first_det, times, dh_mptd, hh_mppd):
    """
    GPU-accelerated replacement for CoherentScoreSamplerFree._get_dh_hh_qo.

    CPU operations (spline interpolation, fplus/fcross antenna patterns) are
    unchanged.  The following matmuls are moved to GPU:
      - bmm for dh_qm (batched over QMC samples q)
      - outer-product einsum + matmul for hh_qm
      - two complex matmuls for dh_qo, hh_qo (phasor phase integration)

    _dh_phasor and _hh_phasor are uploaded once and cached on self.
    """
    import torch
    from gpu.gpu_constants import DEVICE

    # ---- CPU: antenna patterns + spline interpolation (unchanged) ----
    fplus_fcross = self._get_fplus_fcross(sky_inds, q_inds)  # (n_q, n_d, n_p)

    t_det = np.vstack((
        t_first_det,
        t_first_det + self.sky_dict.delays[:, sky_inds],
    ))
    n_det = len(self.sky_dict.detector_names)
    dh_dmpq = np.array(
        [self._interp_locally(times, dh_mptd[..., i_det], t_det[i_det])
         for i_det in range(n_det)],
        dtype=np.complex64,
    )  # (n_d, n_m, n_p, n_q)

    n_d, n_m, n_p, n_q = dh_dmpq.shape
    n_mm = hh_mppd.shape[0]

    # ---- Cache phasors on GPU (uploaded once per CoherentScore instance) ----
    if not hasattr(self, '_dh_phasor_gpu'):
        self._dh_phasor_gpu = torch.from_numpy(
            np.asarray(self._dh_phasor, dtype=np.complex64)
        ).to(DEVICE)  # (n_m, n_phi)
        self._hh_phasor_gpu = torch.from_numpy(
            np.asarray(self._hh_phasor, dtype=np.complex64)
        ).to(DEVICE)  # (n_mm, n_phi)

    # ---- GPU: dh_qm = batched matmul ----
    # moveaxis(dh_dmpq, (3,1), (0,1)) → (n_q, n_m, n_d, n_p) → (n_q, n_m, n_d*n_p)
    dh_qmdp = np.ascontiguousarray(
        np.moveaxis(dh_dmpq, (3, 1), (0, 1)).reshape(n_q, n_m, n_d * n_p)
    )
    dh_t = torch.from_numpy(dh_qmdp).to(DEVICE)  # (n_q, n_m, n_d*n_p) complex64

    # fplus_fcross (n_q, n_d, n_p) → (n_q, n_d*n_p, 1), cast to complex64 (imag=0)
    fpc_t = torch.from_numpy(
        np.ascontiguousarray(fplus_fcross.astype(np.float32).reshape(n_q, n_d * n_p, 1))
    ).to(DEVICE).to(torch.complex64)

    dh_qm_t = torch.bmm(dh_t, fpc_t).squeeze(-1)  # (n_q, n_m)

    # ---- GPU: hh_qm = f_f @ hh_mppd ----
    # f_f: real outer product (n_q, n_p, n_P, n_d) from fplus_fcross
    fpc_full = torch.from_numpy(
        np.ascontiguousarray(fplus_fcross.astype(np.float32))
    ).to(DEVICE)  # (n_q, n_d, n_p) real
    f_f_t = torch.einsum('qdp,qdP->qpPd', fpc_full, fpc_full)  # (n_q, n_p, n_P, n_d)

    hh_t = torch.from_numpy(
        np.ascontiguousarray(np.asarray(hh_mppd, dtype=np.complex64).reshape(n_mm, -1))
    ).to(DEVICE)  # (n_mm, n_p*n_P*n_d) complex64

    # Cast real f_f to complex64 for the matmul with complex hh_t
    hh_qm_t = f_f_t.reshape(n_q, -1).to(torch.complex64) @ hh_t.T  # (n_q, n_mm)

    # ---- GPU: phase integration via complex matmul ----
    # (a @ b).real == real_matmul(a, b) for complex64 a, b
    dh_qo = (dh_qm_t @ self._dh_phasor_gpu).real.cpu().numpy()  # (n_q, n_phi)
    hh_qo = (hh_qm_t @ self._hh_phasor_gpu).real.cpu().numpy()  # (n_q, n_phi)

    return dh_qo, hh_qo


# ---------------------------------------------------------------------------
# H3: GPU _get_many_dh_hh — batched matmuls replacing 8-iteration loop
# ---------------------------------------------------------------------------

def _get_many_dh_hh_gpu(
    self,
    amp_impb,
    phase_impb,
    d_h_weights,
    h_h_weights,
    m_inds,
    mprime_inds,
    asd_drift,
):
    """
    GPU-accelerated replacement for
    MarginalizationExtrinsicSamplerFreeLikelihood._get_many_dh_hh.

    Replaces the n_m×n_p loop with a single batched torch.bmm, and the
    numpy einsum for hh_imppd with torch.einsum.
    """
    import torch
    from gpu.gpu_constants import DEVICE

    h_impb = amp_impb * np.exp(1j * phase_impb)
    h_mpbi = np.moveaxis(h_impb, 0, -1).astype(np.complex64)  # (n_m, n_p, n_b, n_i)
    n_m, n_t, n_d, n_b = d_h_weights.shape
    n_i = amp_impb.shape[0]
    n_p = 2
    n_td = n_t * n_d

    d_h_w = d_h_weights.reshape((n_m, n_td, n_b))  # (n_m, n_td, n_b)

    # ---- GPU: batched dh_mptdi ----
    # Goal: dh_mptdi[m, p] = d_h_w[m] @ h_mpbi[m, p].conj()  → (n_td, n_i)
    # Batched: (n_m*n_p, n_td, n_b) @ (n_m*n_p, n_b, n_i) → (n_m*n_p, n_td, n_i)
    d_h_w_t = torch.from_numpy(
        np.ascontiguousarray(d_h_w.astype(np.complex64))
    ).to(DEVICE)  # (n_m, n_td, n_b)

    h_mpbi_conj_t = torch.from_numpy(
        np.ascontiguousarray(h_mpbi.conj())
    ).to(DEVICE)  # (n_m, n_p, n_b, n_i)

    # Expand d_h_w to (n_m, n_p, n_td, n_b) then reshape for bmm
    d_h_w_expanded = d_h_w_t.unsqueeze(1).expand(n_m, n_p, n_td, n_b)

    dh_t = torch.bmm(
        d_h_w_expanded.reshape(n_m * n_p, n_td, n_b),
        h_mpbi_conj_t.reshape(n_m * n_p, n_b, n_i),
    ).reshape(n_m, n_p, n_td, n_i)  # (n_m, n_p, n_td, n_i)

    dh_mptdi = dh_t.cpu().numpy()  # (n_m, n_p, n_td, n_i)
    dh_imptd = np.moveaxis(dh_mptdi, -1, 0).reshape(n_i, n_m, n_p, n_t, n_d)

    # ---- GPU: hh_imppd via torch.einsum ----
    # 'mdb,mpbi,mPbi->impPd'
    # h_h_weights: (n_mm, n_d, n_b), h_mpbi[m_inds]: (n_mm, n_p, n_b, n_i)
    m_inds_l = list(m_inds)
    mprime_inds_l = list(mprime_inds)

    # h_h_weights can reach ~3e48, exceeding float32 max (3.4e38).
    # Compute einsum in complex128, then cast result back to complex64.
    h_h_w_t = torch.from_numpy(
        np.ascontiguousarray(np.asarray(h_h_weights))  # preserve complex128
    ).to(DEVICE)  # (n_mm, n_d, n_b) cdouble

    h_m_t = torch.from_numpy(
        np.ascontiguousarray(h_mpbi[m_inds_l].astype(np.complex128))
    ).to(DEVICE)  # (n_mm, n_p, n_b, n_i)

    h_mp_conj_t = torch.from_numpy(
        np.ascontiguousarray(h_mpbi.conj()[mprime_inds_l].astype(np.complex128))
    ).to(DEVICE)  # (n_mm, n_p, n_b, n_i)

    hh_imppd_t = torch.einsum(
        'mdb,mpbi,mPbi->impPd', h_h_w_t, h_m_t, h_mp_conj_t
    )  # (n_i, n_mm, n_p, n_P, n_d) cdouble
    hh_imppd = hh_imppd_t.to(torch.complex64).cpu().numpy()

    asd_drift_correction = asd_drift.astype(np.float32) ** -2  # (n_d,)
    return dh_imptd * asd_drift_correction, hh_imppd * asd_drift_correction
