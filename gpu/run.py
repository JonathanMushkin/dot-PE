"""
GPU-accelerated entry point for inference.run() / inference.run_and_profile().

Monkey-patches GPU classes + batched distance sampling into dot_pe before
delegating, so all downstream code picks up the GPU implementations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Batch helpers for precessing spin inverse transforms (Track D, Track E)
# ---------------------------------------------------------------------------

def _disk_to_square_batch(u, v):
    """
    Batch version of TwoSquircularMapping.disk_to_square.

    Maps (u, v) on the unit disk to (x, y) on the unit square using
    the 2-squircular mapping.  Handles u=0 or v=0 without branching.
    """
    import numpy as np
    uv = u * v
    zero_mask = (uv == 0)
    safe_uv = np.where(zero_mask, 1.0, uv)
    inv_f = np.sqrt(
        np.maximum(0.5 - np.sqrt(np.maximum(0.25 - safe_uv**2, 0.0)), 0.0)
    ) / np.abs(safe_uv)
    inv_f = np.where(zero_mask, 1.0, inv_f)
    return inv_f * u, inv_f * v


def _spin_wvf2pe_batch(iota, s1x_n, s1y_n, s1z, s2x_n, s2y_n, s2z):
    """
    Batch numpy replacement for lalsimulation.SimInspiralTransformPrecessingWvf2PE.

    Valid because |L| >> |S| by ~1e9 for all GW sources at f_ref ≥ 20 Hz
    (Newtonian orbital AM dominates), so J ≈ L and the LAL computation
    reduces to pure Cartesian-to-polar conversions:

        theta_jn ≈ iota   (to |S|/|L| ~ 1e-9 relative error)
        phi_jl   = 0      (LAL convention when phiRef=0 and J ≈ L)
        chi1     = |S1|   (exact)
        tilt1    = arctan2(|S1_perp|, s1z)  (exact)
        phi12    = azimuth(S2_perp) - azimuth(S1_perp)  (exact)

    Returns
    -------
    theta_jn, phi_jl, tilt1, tilt2, phi12, chi1, chi2 — all numpy arrays.
    """
    import numpy as np
    chi1 = np.sqrt(s1x_n**2 + s1y_n**2 + s1z**2)
    chi2 = np.sqrt(s2x_n**2 + s2y_n**2 + s2z**2)
    tilt1 = np.arctan2(np.sqrt(s1x_n**2 + s1y_n**2), s1z)
    tilt2 = np.arctan2(np.sqrt(s2x_n**2 + s2y_n**2), s2z)
    phi12 = (np.arctan2(s2y_n, s2x_n) - np.arctan2(s1y_n, s1x_n)) % (2 * np.pi)
    theta_jn = np.asarray(iota, dtype=float).copy()
    phi_jl = np.zeros_like(theta_jn)
    return theta_jn, phi_jl, tilt1, tilt2, phi12, chi1, chi2


def _batch_precessing_spin_inverse(subprior, arrays):
    """
    Batch inverse transform for cogwheel precessing-spin subpriors.

    Handles three concrete types that call LAL's
    SimInspiralTransformPrecessingWvf2PE internally:

      1. _BaseInplaneSpinsInclinationPrior  (polar sampled params)
      2. CartesianUniformDiskInplaneSpinsIsotropicInclinationPrior
      3. _BaseSkyLocationPrior              (polar + sky location)

    Returns dict of sampled params, same as `subprior.inverse_transform(**arrays)`.
    """
    import numpy as np
    from cogwheel.gw_prior.spin import (
        _BaseInplaneSpinsInclinationPrior,
        _BaseSkyLocationPrior,
        CartesianUniformDiskInplaneSpinsIsotropicInclinationPrior,
    )

    iota   = arrays['iota']
    s1x_n  = arrays['s1x_n']
    s1y_n  = arrays['s1y_n']
    s2x_n  = arrays['s2x_n']
    s2y_n  = arrays['s2y_n']
    s1z    = arrays['s1z']
    s2z    = arrays['s2z']
    # m1, m2, f_ref not needed — J ≈ L so SimInspiral reduces to pure trig

    theta_jn, phi_jl, tilt1, tilt2, phi12, chi1, chi2 = (
        _spin_wvf2pe_batch(iota, s1x_n, s1y_n, s1z, s2x_n, s2y_n, s2z)
    )
    costheta_jn = np.cos(theta_jn)
    phi_jl_hat = (phi_jl + np.pi * (costheta_jn < 0)) % (2 * np.pi)

    if isinstance(subprior, CartesianUniformDiskInplaneSpinsIsotropicInclinationPrior):
        polar_prior = subprior._polar_prior
        cums1r_s1z = polar_prior._inverse_spin_transform(chi1, tilt1, s1z)
        cums2r_s2z = polar_prior._inverse_spin_transform(chi2, tilt2, s2z)
        r1 = np.sqrt(cums1r_s1z)
        u1 = r1 * np.cos(phi_jl_hat)
        v1 = r1 * np.sin(phi_jl_hat)
        r2 = np.sqrt(cums2r_s2z)
        u2 = r2 * np.cos(phi12)
        v2 = r2 * np.sin(phi12)
        x1, y1 = _disk_to_square_batch(u1, v1)
        x2, y2 = _disk_to_square_batch(u2, v2)
        return {'costheta_jn': costheta_jn, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

    if isinstance(subprior, _BaseSkyLocationPrior):
        inplane = subprior._inplane_spin_inclination_prior
        cums1r_s1z = inplane._inverse_spin_transform(chi1, tilt1, s1z)
        cums2r_s2z = inplane._inverse_spin_transform(chi2, tilt2, s2z)
        ra  = arrays['ra']
        dec = arrays['dec']
        thetanet, phinet = subprior.skyloc.radec_to_thetaphinet(ra, dec)
        costhetanet = np.cos(thetanet)
        phinet_hat = (phinet + np.pi * (costheta_jn > 0)) % (2 * np.pi)
        return {
            'costheta_jn':  costheta_jn,
            'phi_jl_hat':   phi_jl_hat,
            'phi12':        phi12,
            'cums1r_s1z':   cums1r_s1z,
            'cums2r_s2z':   cums2r_s2z,
            'costhetanet':  costhetanet,
            'phinet_hat':   phinet_hat,
        }

    # _BaseInplaneSpinsInclinationPrior (polar)
    cums1r_s1z = subprior._inverse_spin_transform(chi1, tilt1, s1z)
    cums2r_s2z = subprior._inverse_spin_transform(chi2, tilt2, s2z)
    return {
        'costheta_jn': costheta_jn,
        'phi_jl_hat':  phi_jl_hat,
        'phi12':       phi12,
        'cums1r_s1z':  cums1r_s1z,
        'cums2r_s2z':  cums2r_s2z,
    }


def _is_precessing_spin_subprior(subprior):
    """Return True if subprior calls SimInspiralTransformPrecessingWvf2PE."""
    from cogwheel.gw_prior.spin import (
        _BaseInplaneSpinsInclinationPrior,
        _BaseSkyLocationPrior,
        CartesianUniformDiskInplaneSpinsIsotropicInclinationPrior,
    )
    return isinstance(subprior, (
        _BaseInplaneSpinsInclinationPrior,
        _BaseSkyLocationPrior,
        CartesianUniformDiskInplaneSpinsIsotropicInclinationPrior,
    ))


# ---------------------------------------------------------------------------
# Batch helpers for extrinsic LAL subpriors (Track F)
# ---------------------------------------------------------------------------

def _time_delay_batch(det_location, ra, dec, tgps):
    """
    Batch replacement for lal.TimeDelayFromEarthCenter(det_location, ra, dec, tgps).

    Computes the light-travel time delay from Earth's center to the detector
    for a source at (ra, dec) at GPS time tgps.

    Formula (verified against LAL to < 4e-18 s):
        t_delay = -dot(det_location, n_hat) / c
    where n_hat = (cos(dec)*cos(gmst-ra), -cos(dec)*sin(gmst-ra), sin(dec))
    in ECEF coordinates.
    """
    import numpy as np
    import lal
    c = lal.C_SI
    gmst = lal.GreenwichMeanSiderealTime(tgps)
    gha = gmst - ra
    n_hat = np.column_stack([
        np.cos(dec) * np.cos(gha),
        -np.cos(dec) * np.sin(gha),
        np.sin(dec),
    ])
    return -n_hat @ det_location / c  # (N,)


def _fplus_fcross_batch(det_response, ra, dec, psi, tgps):
    """
    Batch replacement for lal.ComputeDetAMResponse (F+, Fx per sample).

    Verified against lal.ComputeDetAMResponse to < 1e-14 relative error.

    Parameters
    ----------
    det_response : (3, 3) ndarray — detector tensor (symmetric, traceless).
    ra, dec, psi : (N,) float64 arrays — sky position and polarization angle.
    tgps : float — GPS time (sets Earth orientation via GMST).

    Returns
    -------
    fp, fc : (N,) float64 arrays — antenna patterns F+, Fx.
    """
    import numpy as np
    import lal
    gmst = lal.GreenwichMeanSiderealTime(tgps)
    gha = gmst - ra
    sin_gha = np.sin(gha); cos_gha = np.cos(gha)
    sin_dec = np.sin(dec); cos_dec = np.cos(dec)

    # Orthonormal sky-plane basis (East and North in ECEF):
    # e_East: tangent vector in the direction of increasing RA (East)
    # e_North: tangent vector in the direction of increasing Dec (North)
    # gha = gmst - ra, so increasing ra → decreasing gha
    e_East  = np.column_stack([ sin_gha, cos_gha,    np.zeros(len(ra))])
    e_North = np.column_stack([-sin_dec * cos_gha, sin_dec * sin_gha, cos_dec])

    # Polarization basis (plus/cross at angle psi):
    # m = cos(psi)*e_East - sin(psi)*e_North
    # n = sin(psi)*e_East + cos(psi)*e_North
    cos_psi = np.cos(psi); sin_psi = np.sin(psi)
    m = cos_psi[:, None] * e_East - sin_psi[:, None] * e_North
    n = sin_psi[:, None] * e_East + cos_psi[:, None] * e_North

    # F+ = m^T D m - n^T D n
    # Fx = -(m^T D n + n^T D m) = -2*(m^T D n) since D is symmetric
    Dm = m @ det_response
    Dn = n @ det_response
    fp = np.einsum('ni,ni->n', Dm, m) - np.einsum('ni,ni->n', Dn, n)
    fc = -(np.einsum('ni,ni->n', Dm, n) + np.einsum('ni,ni->n', Dn, m))
    return fp, fc


def _geometric_factor_batch(det_response, ra, dec, psi, iota, tgps):
    """
    Batch replacement for ReferenceDetectorMixin.geometric_factor_refdet.

    Returns complex array of shape (N,):
        R = (1 + cos^2(iota)) / 2 * F+ - i * cos(iota) * Fx
    """
    import numpy as np
    fp, fc = _fplus_fcross_batch(det_response, ra, dec, psi, tgps)
    cosiota = np.cos(iota)
    return (1 + cosiota**2) / 2 * fp - 1j * cosiota * fc


def _batch_extrinsic_subprior(subprior, arrays):
    """
    Batch inverse transform for extrinsic subpriors that call LAL scalars.

    Handles:
      - UniformPolarizationPrior  (LRU-cache blocks arrays)
      - UniformTimePrior          (lal.TimeDelayFromEarthCenter)
      - UniformPhasePrior         (TimeDelayFromEarthCenter + ComputeDetAMResponse)
      - UniformLuminosityVolumePrior  (ComputeDetAMResponse + mchirp)

    Returns dict of sampled params (same as `subprior.inverse_transform(**arrays)`).
    """
    import numpy as np
    from cogwheel.gw_prior.extrinsic import (
        UniformPolarizationPrior,
        UniformTimePrior,
        UniformPhasePrior,
        UniformLuminosityVolumePrior,
    )
    from cogwheel import gw_utils

    if isinstance(subprior, UniformPolarizationPrior):
        # IdentityTransformMixin: inverse_transform == transform (LRU-cached).
        # For psi, the identity just returns psi itself.
        psi = arrays['psi']
        from cogwheel import utils
        start = subprior.range_dic['psi'][0]
        return {'psi': utils.mod(psi, start=start)}

    det_loc = subprior.ref_det_location   # (3,) ECEF meters
    det_resp = gw_utils.DETECTORS[subprior.ref_det_name].response  # (3,3)
    tgps = subprior.tgps
    ra  = arrays['ra']
    dec = arrays['dec']

    if isinstance(subprior, UniformTimePrior):
        t_geocenter = arrays['t_geocenter']
        t_delay = _time_delay_batch(det_loc, ra, dec, tgps)
        return {'t_refdet': t_geocenter + t_delay}

    if isinstance(subprior, UniformPhasePrior):
        phi_ref     = arrays['phi_ref']
        iota        = arrays['iota']
        psi         = arrays['psi']
        t_geocenter = arrays['t_geocenter']

        t_delay  = _time_delay_batch(det_loc, ra, dec, tgps)
        t_refdet = t_geocenter + t_delay
        geom_factor = _geometric_factor_batch(det_resp, ra, dec, psi, iota, tgps)
        # _phase_refdet is called with phi_ref=0 in inverse_transform:
        phase_refdet = (
            np.angle(geom_factor)
            - 2 * np.pi * subprior.f_avg * t_refdet
        ) % (2 * np.pi)
        phase_refdet_0 = subprior._phase_refdet_0
        from cogwheel import utils
        phi_ref_hat = utils.mod(
            phi_ref + (phase_refdet - phase_refdet_0) / 2,
            start=subprior.range_dic['phi_ref_hat'][0],
        )
        return {'phi_ref_hat': phi_ref_hat}

    if isinstance(subprior, UniformLuminosityVolumePrior):
        d_luminosity = arrays['d_luminosity']
        psi  = arrays['psi']
        iota = arrays['iota']
        m1   = arrays['m1']
        m2   = arrays['m2']

        mchirp   = gw_utils.m1m2_to_mchirp(m1, m2)
        geom_abs = np.abs(_geometric_factor_batch(det_resp, ra, dec, psi, iota, tgps))
        d_hat = d_luminosity / (mchirp ** (5 / 6) * geom_abs)
        return {'d_hat': d_hat}

    # Unknown subprior type — should not reach here
    raise TypeError(f"_batch_extrinsic_subprior: unhandled type {type(subprior).__name__}")


def _is_extrinsic_lal_subprior(subprior):
    """Return True if subprior hits the LAL/LRU-cache fallback."""
    from cogwheel.gw_prior.extrinsic import (
        UniformPolarizationPrior,
        UniformTimePrior,
        UniformPhasePrior,
        UniformLuminosityVolumePrior,
    )
    return isinstance(subprior, (
        UniformPolarizationPrior,
        UniformTimePrior,
        UniformPhasePrior,
        UniformLuminosityVolumePrior,
    ))


# ---------------------------------------------------------------------------
# Batch inverse-transform for prior subpriors (Track D)
# ---------------------------------------------------------------------------

def _pr_inverse_transform_batch(pr, samples):
    """
    Batch replacement for pr.inverse_transform_samples.

    Iterates over pr.subpriors in dependency order.  For each subprior,
    attempts a single vectorized call with full numpy arrays (batch-safe path).
    If the call fails — e.g. because a LAL C-function requires scalar inputs —
    uses a numpy batch replacement for known precessing-spin subpriors, or
    falls back to a per-sample loop for all others.

    Modifies `samples` in-place, adding all sampled_params columns.

    Parameters
    ----------
    pr : cogwheel CombinedPrior
        Must expose `pr.subpriors` (list of Prior instances in dependency
        order, as built by CombinedPrior.__init_subclass__).
    samples : pandas.DataFrame
        Must already contain all standard_params and conditioned_on values
        needed by each subprior.  Updated in-place with sampled_params.
    """
    import numpy as np

    from cogwheel.prior import FixedPrior

    n = len(samples)

    for subprior in pr.subpriors:
        # FixedPrior has no sampled_params (returns {}); its equality check
        # breaks on arrays.  Skip — nothing to write, validation not needed.
        if isinstance(subprior, FixedPrior):
            continue

        input_params = subprior.standard_params + subprior.conditioned_on
        input_arrays = {p: samples[p].values for p in input_params}

        # --- Attempt batch call ---
        try:
            result = subprior.inverse_transform(**input_arrays)
            # Validate: every output value must be array-valued (not scalar)
            # when n > 1.  A scalar return signals the subprior is not
            # batch-safe (it returned a single value instead of N).
            for key, val in result.items():
                arr = np.asarray(val)
                if n > 1 and arr.ndim == 0:
                    raise ValueError(
                        f"{type(subprior).__name__}.inverse_transform returned "
                        f"scalar for {key!r} with n={n} — not batch-safe"
                    )
            for key, val in result.items():
                samples[key] = np.asarray(val)

        except Exception:
            # --- Fast numpy batch path for known non-batch-safe subpriors ---
            if _is_precessing_spin_subprior(subprior):
                result = _batch_precessing_spin_inverse(subprior, input_arrays)
                for key, val in result.items():
                    samples[key] = val
                continue

            if _is_extrinsic_lal_subprior(subprior):
                result = _batch_extrinsic_subprior(subprior, input_arrays)
                for key, val in result.items():
                    samples[key] = val
                continue

            # --- Per-sample fallback for any remaining scalar-only subpriors ---
            def _call_one(i, _sp=subprior, _ip=input_params, _ia=input_arrays):
                return _sp.inverse_transform(
                    **{p: float(_ia[p][i]) for p in _ip}
                )

            result_list = [_call_one(i) for i in range(n)]

            for key in result_list[0]:
                samples[key] = np.array([row[key] for row in result_list])


# ---------------------------------------------------------------------------
# Replacement for inference.standardize_samples
# ---------------------------------------------------------------------------

def _standardize_samples_gpu(
    cached_dt_linfree_relative,
    lookup_table,
    pr,
    prob_samples,
    intrinsic_samples,
    extrinsic_samples,
    n_phi,
    tgps,
):
    """
    Drop-in replacement for dot_pe.inference.standardize_samples that uses
    sample_distance_batched_gpu instead of np.vectorize(lookup_table.sample_distance).

    All DataFrame manipulation is identical to the original; only the distance
    sampling block (lines 546-552) is replaced.
    """
    import numpy as np
    import pandas as pd
    from lal import GreenwichMeanSiderealTime
    from cogwheel import skyloc_angles
    from cogwheel.utils import exp_normalize
    from gpu.distance_sampling_gpu import sample_distance_batched_gpu

    if "bank_id" not in prob_samples.columns:
        raise ValueError("prob_samples must have 'bank_id' column")

    if not isinstance(intrinsic_samples, dict) or not isinstance(
        cached_dt_linfree_relative, dict
    ):
        raise ValueError(
            "intrinsic_samples and cached_dt_linfree_relative "
            "must be dicts mapping bank_id to values"
        )

    if isinstance(extrinsic_samples, (str, Path)):
        extrinsic_samples = pd.read_feather(extrinsic_samples)

    combined_samples_list = []
    for bank_id, group in prob_samples.groupby("bank_id"):
        bank_intrinsic = intrinsic_samples[bank_id]
        bank_cache = cached_dt_linfree_relative[bank_id]

        group_combined = pd.concat(
            [
                bank_intrinsic.iloc[group["i"].values].reset_index(drop=True),
                extrinsic_samples.iloc[group["e"].values].reset_index(drop=True),
            ],
            axis=1,
        )

        group_combined.drop(
            columns=["weights", "log_prior_weights", "original_index"],
            inplace=True,
            errors="ignore",
        )

        group_combined = pd.concat(
            [group_combined, group.reset_index(drop=True)], axis=1
        )

        group_combined["phi"] = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)[
            group_combined["o"].values
        ]

        group_combined.rename(
            columns={
                "t_geocenter": "t_geocenter_linfree",
                "phi": "phi_ref_linfree",
            },
            inplace=True,
        )

        unique_i_bank = np.unique(group["i"].values)
        u_i_bank = np.searchsorted(unique_i_bank, group["i"].values)

        dt_linfree_u_bank = np.array([bank_cache[i] for i in unique_i_bank])
        dphi_linfree_u_bank = np.zeros_like(dt_linfree_u_bank)

        group_combined["t_geocenter"] = (
            group_combined["t_geocenter_linfree"] - dt_linfree_u_bank[u_i_bank]
        )
        group_combined["phi_ref"] = (
            group_combined["phi_ref_linfree"] - dphi_linfree_u_bank[u_i_bank]
        )

        combined_samples_list.append(group_combined)

    combined_samples = pd.concat(combined_samples_list, ignore_index=True)
    combined_samples = combined_samples.reindex(prob_samples.index)
    combined_samples.reset_index(drop=True, inplace=True)

    combined_samples["ra"] = skyloc_angles.lon_to_ra(
        combined_samples["lon"], GreenwichMeanSiderealTime(tgps)
    )
    combined_samples["dec"] = combined_samples["lat"]

    # --- GPU batched distance sampling (replaces np.vectorize loop) ---
    combined_samples["d_luminosity"] = sample_distance_batched_gpu(
        combined_samples["d_h_1Mpc"].values,
        combined_samples["h_h_1Mpc"].values,
        lookup_table,
        resolution=32,
    )

    combined_samples["bestfit_d_luminosity"] = (
        combined_samples["h_h_1Mpc"] / combined_samples["d_h_1Mpc"]
    )
    combined_samples["lnl"] = (
        combined_samples["d_h_1Mpc"] / combined_samples["d_luminosity"]
        - 0.5 * combined_samples["h_h_1Mpc"] / combined_samples["d_luminosity"] ** 2
    )

    if "weights" not in combined_samples.columns:
        combined_samples["weights"] = exp_normalize(
            combined_samples["ln_posterior"].values
        )

    _pr_inverse_transform_batch(pr, combined_samples)

    return combined_samples


# ---------------------------------------------------------------------------
# Patch
# ---------------------------------------------------------------------------

def _patch():
    """Replace CPU classes with GPU subclasses in the dot_pe namespace."""
    from gpu.single_detector_gpu import GPUSingleDetectorProcessor
    from gpu.likelihood_calculating_gpu import GPULikelihoodCalculator

    import dot_pe.single_detector as _sd
    import dot_pe.coherent_processing as _cp
    import dot_pe.likelihood_calculating as _lc
    import dot_pe.inference as _inf

    _sd.SingleDetectorProcessor = GPUSingleDetectorProcessor
    _lc.LikelihoodCalculator = GPULikelihoodCalculator
    # coherent_processing imports LikelihoodCalculator at class-body level
    _cp.likelihood_calculating.LikelihoodCalculator = GPULikelihoodCalculator

    # inference.py binds SingleDetectorProcessor via `from .single_detector import ...`
    # so the module-level patch above doesn't reach it — patch the local binding too
    _inf.SingleDetectorProcessor = GPUSingleDetectorProcessor

    # Replace per-sample np.vectorize distance loop with batched GPU call
    _inf.standardize_samples = _standardize_samples_gpu


def run(**kwargs):
    """GPU-accelerated inference.run()."""
    _patch()
    from dot_pe import inference
    return inference.run(**kwargs)


def run_and_profile(**kwargs):
    """GPU-accelerated inference.run_and_profile()."""
    _patch()
    from dot_pe import inference
    return inference.run_and_profile(**kwargs)
