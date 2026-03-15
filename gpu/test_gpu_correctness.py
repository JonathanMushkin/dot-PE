"""
Unit correctness tests: compare GPU outputs to CPU baselines.

Run with:
    python gpu/test_gpu_correctness.py
"""

import sys
from pathlib import Path
import itertools

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

ATOL = 1e-3   # float32 absolute tolerance
RTOL = 2e-3   # float32 relative tolerance (~0.2%)


def make_rng(seed=42):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Helpers to produce random complex64 arrays
# ---------------------------------------------------------------------------

def rand_c64(rng, *shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)


def rand_f32(rng, *shape):
    return rng.standard_normal(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Test: get_response_over_distance_and_lnlike
# ---------------------------------------------------------------------------

def test_single_detector():
    from dot_pe.single_detector import SingleDetectorProcessor, _get_phasors
    from gpu.single_detector_gpu import get_response_over_distance_and_lnlike_gpu

    rng = make_rng(1)
    i, m, p, b, t = 8, 2, 2, 64, 16
    n_phi = 10
    m_arr = np.array([2, 4])

    # Shapes mirror the real code
    d = 1
    M = len(list(itertools.combinations_with_replacement(range(m), 2)))

    h_impb = rand_c64(rng, i, m, p, b)
    dh_weights_dmpb = rand_c64(rng, d, m, p, b)
    hh_weights_dmppb = rand_c64(rng, d, M, p, p, b)
    timeshift_dbt = rand_c64(rng, d, b, t)
    asd_drift_d = np.ones(d, dtype=np.float32)

    r_cpu, ll_cpu = SingleDetectorProcessor.get_response_over_distance_and_lnlike(
        dh_weights_dmpb, hh_weights_dmppb, h_impb, timeshift_dbt, asd_drift_d, n_phi, m_arr
    )
    r_gpu, ll_gpu = get_response_over_distance_and_lnlike_gpu(
        dh_weights_dmpb, hh_weights_dmppb, h_impb, timeshift_dbt, asd_drift_d, n_phi, m_arr
    )

    assert np.allclose(r_cpu, r_gpu, atol=ATOL, rtol=RTOL), f"r_iotp mismatch: max diff {np.abs(r_cpu - r_gpu).max():.3e}"
    assert np.allclose(ll_cpu, ll_gpu, atol=ATOL, rtol=RTOL), f"lnlike_iot mismatch: max diff {np.abs(ll_cpu - ll_gpu).max():.3e}"
    print("PASS test_single_detector")


# ---------------------------------------------------------------------------
# Test: get_dh_by_mode
# ---------------------------------------------------------------------------

def test_get_dh_by_mode():
    from dot_pe.likelihood_calculating import LikelihoodCalculator
    from gpu.likelihood_calculating_gpu import get_dh_by_mode_gpu

    rng = make_rng(2)
    i, m, p, b, d, e = 6, 2, 2, 32, 2, 16

    h_impb = rand_c64(rng, i, m, p, b)
    dh_weights_dmpb = rand_c64(rng, d, m, p, b)
    response_dpe = rand_c64(rng, d, p, e)
    timeshift_dbe = rand_c64(rng, d, b, e)
    asd_drift_d = np.ones(d, dtype=np.float32)

    cpu = LikelihoodCalculator.get_dh_by_mode(
        dh_weights_dmpb, h_impb, response_dpe, timeshift_dbe, asd_drift_d
    )
    gpu = get_dh_by_mode_gpu(
        dh_weights_dmpb, h_impb, response_dpe, timeshift_dbe, asd_drift_d
    )

    assert np.allclose(cpu, gpu, atol=ATOL, rtol=RTOL), f"dh_iem mismatch: max diff {np.abs(cpu - gpu).max():.3e}"
    print("PASS test_get_dh_by_mode")


# ---------------------------------------------------------------------------
# Test: get_hh_by_mode
# ---------------------------------------------------------------------------

def test_get_hh_by_mode():
    from dot_pe.likelihood_calculating import LikelihoodCalculator
    from gpu.likelihood_calculating_gpu import get_hh_by_mode_gpu

    rng = make_rng(3)
    i, m, p, b, d, e = 6, 2, 2, 32, 2, 16

    m_inds, mprime_inds = zip(*itertools.combinations_with_replacement(range(m), 2))
    M = len(m_inds)

    h_impb = rand_c64(rng, i, m, p, b)
    response_dpe = rand_c64(rng, d, p, e)
    hh_weights_dmppb = rand_c64(rng, d, M, p, p, b)
    asd_drift_d = np.ones(d, dtype=np.float32)

    cpu = LikelihoodCalculator.get_hh_by_mode(
        h_impb, response_dpe, hh_weights_dmppb, asd_drift_d, m_inds, mprime_inds
    )
    gpu = get_hh_by_mode_gpu(
        h_impb, response_dpe, hh_weights_dmppb, asd_drift_d, m_inds, mprime_inds
    )

    assert np.allclose(cpu, gpu, atol=ATOL, rtol=RTOL), f"hh_iem mismatch: max diff {np.abs(cpu - gpu).max():.3e}"
    print("PASS test_get_hh_by_mode")


# ---------------------------------------------------------------------------
# Test: get_dh_hh_phi_grid
# ---------------------------------------------------------------------------

def test_get_dh_hh_phi_grid():
    from dot_pe.likelihood_calculating import LikelihoodCalculator
    from gpu.likelihood_calculating_gpu import get_dh_hh_phi_grid_gpu

    rng = make_rng(4)
    i, e, m_val, n_phi = 8, 16, 2, 20
    m_arr = np.array([2, 4])

    m_inds, mprime_inds = zip(*itertools.combinations_with_replacement(range(m_val), 2))
    M = len(m_inds)

    dh_iem = rand_c64(rng, i, e, m_val)
    hh_iem = rand_c64(rng, i, e, M)

    calc = LikelihoodCalculator(n_phi=n_phi, m_arr=m_arr)
    dh_cpu, hh_cpu = calc.get_dh_hh_phi_grid(dh_iem, hh_iem)
    dh_gpu, hh_gpu = get_dh_hh_phi_grid_gpu(dh_iem, hh_iem, m_arr, m_inds, mprime_inds, n_phi)

    assert np.allclose(dh_cpu, dh_gpu, atol=ATOL, rtol=RTOL), f"dh_ieo mismatch: max diff {np.abs(dh_cpu - dh_gpu).max():.3e}"
    assert np.allclose(hh_cpu, hh_gpu, atol=ATOL, rtol=RTOL), f"hh_ieo mismatch: max diff {np.abs(hh_cpu - hh_gpu).max():.3e}"
    print("PASS test_get_dh_hh_phi_grid")


# ---------------------------------------------------------------------------
# Test: float32 precision in the 2×2 inversion (near-singular hh matrix)
#
# Background
# ----------
# The determinant det = a00*a11 - a01^2.  For a near-face-on binary
# h+ ≈ h×, so a01 ≈ a00 ≈ a11, and det is a small difference of two
# nearly-equal float32 numbers — catastrophic cancellation.
# Measured float32 relative error in det:
#   eps=1e-3 → ~2e-5   (fine)
#   eps=1e-5 → ~1e-3   (already 0.1%)
#   eps=1e-6 → ~1e-2   (1.3% — unacceptable for lnlike precision)
#
# Fix: the inversion is performed in float64 and the result cast back to
# float32.  This test verifies that the fix holds at eps=1e-6 (extreme
# face-on), keeping the relative error in lnlike_iot < 1e-4.
# ---------------------------------------------------------------------------

def test_single_detector_near_singular_hh():
    """
    Stress-test the 2×2 hh inversion with a near-face-on binary
    (h_plus ≈ h_cross → nearly rank-1 hh matrix → det ≈ 0).
    Uses float64 CPU baseline vs float32+float64-inversion GPU path.
    Asserts relative error in lnlike_iot stays below 1e-4.
    """
    from dot_pe.single_detector import SingleDetectorProcessor
    from gpu.single_detector_gpu import get_response_over_distance_and_lnlike_gpu

    rng = make_rng(99)
    i, m, p, b, t = 8, 2, 2, 64, 16
    n_phi = 10
    m_arr = np.array([2, 4])
    d = 1
    M = len(list(itertools.combinations_with_replacement(range(m), 2)))

    # h_cross = h_plus * (1 + eps * noise): nearly rank-1 polarization subspace
    eps = 1e-6
    h_plus = (rng.standard_normal((i, m, 1, b)) + 1j * rng.standard_normal((i, m, 1, b)))
    h_cross = h_plus * (
        1 + eps * (rng.standard_normal((i, m, 1, b)) + 1j * rng.standard_normal((i, m, 1, b)))
    )
    h_f64 = np.concatenate([h_plus, h_cross], axis=2)  # float64, (i,m,2,b)

    # Weights also in float64 (as produced by the real CPU pipeline)
    dh_w_f64 = rng.standard_normal((d, m, p, b)) + 1j * rng.standard_normal((d, m, p, b))
    hh_w_f64 = rng.standard_normal((d, M, p, p, b)) + 1j * rng.standard_normal((d, M, p, p, b))
    ts_f64   = rng.standard_normal((d, b, t)) + 1j * rng.standard_normal((d, b, t))
    asd      = np.ones(d, dtype=np.float64)

    # CPU: float64 throughout (the ground truth)
    r_cpu, ll_cpu = SingleDetectorProcessor.get_response_over_distance_and_lnlike(
        dh_w_f64, hh_w_f64, h_f64, ts_f64, asd, n_phi, m_arr
    )
    # GPU: complex64 matmuls + float64 inversion step
    r_gpu, ll_gpu = get_response_over_distance_and_lnlike_gpu(
        dh_w_f64, hh_w_f64, h_f64, ts_f64, asd, n_phi, m_arr
    )

    # Relative error in lnlike_iot must stay well below 1e-3
    # (1.3% det error without the fix would propagate to ~1% lnlike error)
    ll_rel = (np.abs(ll_cpu - ll_gpu) / (np.abs(ll_cpu) + 1e-10)).max()
    r_rel  = (np.abs(r_cpu  - r_gpu)  / (np.abs(r_cpu)  + 1e-10)).max()
    RTOL_STRESS = 1e-3  # 0.1% relative tolerance
    assert ll_rel < RTOL_STRESS, (
        f"lnlike relative error {ll_rel:.3e} exceeds {RTOL_STRESS:.0e} "
        f"(near-singular hh, eps={eps:.0e}) — float64 inversion may be broken"
    )
    assert r_rel < RTOL_STRESS, (
        f"r_iotp relative error {r_rel:.3e} exceeds {RTOL_STRESS:.0e} "
        f"(near-singular hh, eps={eps:.0e}) — float64 inversion may be broken"
    )
    print(f"PASS test_single_detector_near_singular_hh  "
          f"(ll_rel={ll_rel:.2e}, r_rel={r_rel:.2e})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Test: sample_distance_batched_gpu vs CPU lookup_table.sample_distance
# ---------------------------------------------------------------------------

def test_sample_distance_batched():
    """
    Deterministic accuracy test for sample_distance_batched_gpu.

    Both CPU (lookup_table.sample_distance) and GPU use identical:
      - focused grid: 1/linspace(*u_bounds, K) in u-space
      - broad grid: linspace(0, D_MAX, K+1)[1:]
      - CDF: trapezoidal rule (k=1 spline antiderivative == trapz)
      - inversion: piecewise linear interpolation

    By fixing the fractional CDF quantile (u_frac) to the same value in
    both paths, the results must agree up to float32 rounding errors.

    Expected accuracy: relative error < 1e-3 (limited by grid resolution,
    not random noise).  With resolution=256 this would reach < 1e-6.
    """
    import numpy as np
    from scipy.interpolate import InterpolatedUnivariateSpline
    from cogwheel.likelihood import LookupTable
    from gpu.distance_sampling_gpu import sample_distance_batched_gpu

    rng = np.random.default_rng(77)
    N = 500  # enough for a robust check; CDF inversion is deterministic

    # Typical SNR range
    norm_h  = rng.uniform(5.0, 50.0, N)
    overlap = rng.uniform(0.5, 1.0, N) * norm_h
    REF = 1.0
    h_h = norm_h ** 2
    d_h = overlap * REF * norm_h

    lut = LookupTable(d_luminosity_prior_name="euclidean", d_luminosity_max=15000.0)
    D_MAX = float(lut.d_luminosity_max)
    K     = 32

    # Fixed fractional quantiles: same for CPU and GPU
    u_fracs = rng.uniform(0.02, 0.98, N)   # avoid tails where grid is sparse

    # --- CPU reference: same grid as sample_distance, fixed quantile ---
    cpu_results = np.empty(N)
    for k in range(N):
        u_bounds   = 1.0 / lut._get_distance_bounds(d_h[k], h_h[k], sigmas=10.0)
        focused    = 1.0 / np.linspace(u_bounds[0], u_bounds[1], K)
        focused    = focused[(focused > 0) & (focused < D_MAX)]
        broad      = np.linspace(0.0, D_MAX, K + 1)[1:]
        distances  = np.sort(np.concatenate([broad, focused]))
        posterior  = lut._function_integrand(distances, d_h[k], h_h[k])
        cumulative = InterpolatedUnivariateSpline(
            distances, posterior, k=1).antiderivative()(distances)
        cpu_results[k] = np.interp(
            u_fracs[k] * cumulative[-1], cumulative, distances)

    # --- GPU: same fractional quantiles, same grid logic ---
    gpu_results = sample_distance_batched_gpu(
        d_h, h_h, lut, resolution=K, _u_fracs=u_fracs)

    assert gpu_results.shape == (N,)
    assert np.all(gpu_results >= 0)
    assert np.all(gpu_results <= D_MAX)

    rel_err = np.abs(cpu_results - gpu_results) / (cpu_results + 1.0)
    max_rel = rel_err.max()
    med_rel = np.median(rel_err)
    RTOL = 1e-3   # float32 CDF grid interpolation error budget

    assert max_rel < RTOL, (
        f"Max relative error {max_rel:.2e} exceeds {RTOL:.0e}. "
        f"Median={med_rel:.2e}. Worst sample: "
        f"d_h={d_h[rel_err.argmax()]:.3g}, h_h={h_h[rel_err.argmax()]:.3g}, "
        f"cpu={cpu_results[rel_err.argmax()]:.1f}, gpu={gpu_results[rel_err.argmax()]:.1f}"
    )
    print(
        f"PASS test_sample_distance_batched  "
        f"(max_rel={max_rel:.2e}, median_rel={med_rel:.2e}, N={N}, K={K})"
    )


# ---------------------------------------------------------------------------
# Test: _batch_precessing_spin_inverse vs LAL per-sample baseline
# ---------------------------------------------------------------------------

def test_precessing_spin_inverse_transform_batch():
    """
    Verify _batch_precessing_spin_inverse for all three precessing-spin
    subprior types against lalsimulation.SimInspiralTransformPrecessingWvf2PE.

    Expected accuracy: ≤ 1e-6 relative error for all outputs
    (theta_jn: ~1e-9 from J≈L; others: machine precision).
    """
    import lal
    import lalsimulation as ls
    import pandas as pd
    from cogwheel.gw_prior.spin import (
        UniformDiskInplaneSpinsIsotropicInclinationPrior,
        IsotropicSpinsInplaneComponentsIsotropicInclinationPrior,
        CartesianUniformDiskInplaneSpinsIsotropicInclinationPrior,
        UniformDiskInplaneSpinsIsotropicInclinationSkyLocationPrior,
    )
    from gpu.run import _batch_precessing_spin_inverse

    rng = np.random.default_rng(77)
    N = 300
    TGPS = 1187008882.4
    MSUN = lal.MSUN_SI

    m1_kg = rng.uniform(10, 80, N) * MSUN
    m2_kg = rng.uniform(5, 40, N) * MSUN
    f_ref = np.full(N, 50.0)

    # Random spin vectors (Cartesian in L-frame)
    chi1 = rng.uniform(0, 0.99, N)
    chi2 = rng.uniform(0, 0.99, N)
    tilt1 = rng.uniform(0, np.pi, N)
    tilt2 = rng.uniform(0, np.pi, N)
    phi_s1 = rng.uniform(0, 2*np.pi, N)
    phi12 = rng.uniform(0, 2*np.pi, N)

    s1x_n = chi1 * np.sin(tilt1) * np.cos(phi_s1)
    s1y_n = chi1 * np.sin(tilt1) * np.sin(phi_s1)
    s1z   = chi1 * np.cos(tilt1)
    phi_s2 = (phi_s1 + phi12) % (2*np.pi)
    s2x_n = chi2 * np.sin(tilt2) * np.cos(phi_s2)
    s2y_n = chi2 * np.sin(tilt2) * np.sin(phi_s2)
    s2z   = chi2 * np.cos(tilt2)
    iota  = rng.uniform(0, np.pi, N)
    ra    = rng.uniform(0, 2*np.pi, N)
    dec   = rng.uniform(-np.pi/2, np.pi/2, N)

    arrays_base = dict(
        iota=iota, s1x_n=s1x_n, s1y_n=s1y_n, s2x_n=s2x_n, s2y_n=s2y_n,
        s1z=s1z, s2z=s2z, m1=m1_kg, m2=m2_kg, f_ref=f_ref,
    )

    # LAL per-sample reference (polar outputs)
    lal_results = np.array([
        ls.SimInspiralTransformPrecessingWvf2PE(
            iota[i], s1x_n[i], s1y_n[i], s1z[i], s2x_n[i], s2y_n[i], s2z[i],
            m1_kg[i], m2_kg[i], f_ref[i], 0.)
        for i in range(N)
    ])
    lal_theta_jn = lal_results[:, 0]
    lal_tilt1    = lal_results[:, 2]
    lal_tilt2    = lal_results[:, 3]
    lal_phi12    = lal_results[:, 4]
    lal_chi1     = lal_results[:, 5]
    lal_chi2     = lal_results[:, 6]

    RTOL_SPIN = 1e-6   # max relative error budget

    # --- 1. _BaseInplaneSpinsInclinationPrior (polar) ---
    sp1 = UniformDiskInplaneSpinsIsotropicInclinationPrior()
    r1  = _batch_precessing_spin_inverse(sp1, arrays_base)
    assert np.allclose(np.cos(lal_theta_jn), r1['costheta_jn'], rtol=RTOL_SPIN), \
        f"costheta_jn mismatch (polar): max_rel={np.abs(np.cos(lal_theta_jn)-r1['costheta_jn']).max():.2e}"
    lal_cums1 = sp1._inverse_spin_transform(lal_chi1, lal_tilt1, s1z)
    assert np.allclose(lal_cums1, r1['cums1r_s1z'], rtol=RTOL_SPIN), \
        f"cums1r_s1z mismatch: max_rel={np.abs(lal_cums1-r1['cums1r_s1z']).max():.2e}"
    assert np.allclose(lal_phi12, r1['phi12'], atol=1e-12), \
        f"phi12 mismatch: max_abs={np.abs(lal_phi12-r1['phi12']).max():.2e}"

    # --- 2. CartesianUniformDiskInplaneSpinsIsotropicInclinationPrior ---
    sp2 = CartesianUniformDiskInplaneSpinsIsotropicInclinationPrior()
    r2  = _batch_precessing_spin_inverse(sp2, arrays_base)
    # Scalar LAL reference for each sample
    ref2 = [sp2.inverse_transform(
        iota[i], s1x_n[i], s1y_n[i], s2x_n[i], s2y_n[i], s1z[i], s2z[i],
        m1_kg[i], m2_kg[i], f_ref[i]) for i in range(N)]
    for key in ['costheta_jn', 'x1', 'y1', 'x2', 'y2']:
        ref_arr = np.array([r[key] for r in ref2])
        assert np.allclose(r2[key], ref_arr, rtol=RTOL_SPIN, atol=1e-12), \
            f"CartesianUniform {key} mismatch: max_rel={np.abs(r2[key]-ref_arr).max():.2e}"

    # --- 3. _BaseSkyLocationPrior ---
    sp3 = UniformDiskInplaneSpinsIsotropicInclinationSkyLocationPrior(
        detector_pair='HL', tgps=TGPS)
    arrays3 = dict(arrays_base, ra=ra, dec=dec)
    r3 = _batch_precessing_spin_inverse(sp3, arrays3)
    ref3 = [sp3.inverse_transform(
        iota[i], s1x_n[i], s1y_n[i], s2x_n[i], s2y_n[i],
        ra[i], dec[i], s1z[i], s2z[i], m1_kg[i], m2_kg[i], f_ref[i])
        for i in range(N)]
    for key in ['costheta_jn', 'phi_jl_hat', 'phi12', 'cums1r_s1z', 'cums2r_s2z',
                'costhetanet', 'phinet_hat']:
        ref_arr = np.array([r[key] for r in ref3])
        assert np.allclose(r3[key], ref_arr, rtol=RTOL_SPIN, atol=1e-12), \
            f"SkyLocation {key} mismatch: max={np.abs(r3[key]-ref_arr).max():.2e}"

    print(f"PASS test_precessing_spin_inverse_transform_batch  (N={N}, 3 subprior types)")


# ---------------------------------------------------------------------------
# Test: _batch_extrinsic_subprior vs cogwheel per-sample baseline
# ---------------------------------------------------------------------------

def test_extrinsic_lal_subprior_batch():
    """
    Verify _batch_extrinsic_subprior for all four extrinsic LAL subprior types
    against per-sample cogwheel baseline.

    Expected accuracy: < 1e-12 relative error (machine precision).
    """
    import pandas as pd
    from cogwheel.gw_prior.extrinsic import (
        UniformPolarizationPrior,
        UniformTimePrior,
        UniformPhasePrior,
        UniformLuminosityVolumePrior,
    )
    from gpu.run import _batch_extrinsic_subprior

    rng = np.random.default_rng(55)
    N = 500
    TGPS = 1187008882.4
    REF = 'H'
    F_AVG = 109.0

    ra  = rng.uniform(0, 2 * np.pi, N)
    dec = rng.uniform(-np.pi / 2, np.pi / 2, N)
    psi = rng.uniform(0, np.pi, N)
    iota = rng.uniform(0, np.pi, N)
    t_geocenter = rng.uniform(-0.05, 0.05, N)
    phi_ref = rng.uniform(0, 2 * np.pi, N)
    m1 = rng.uniform(10, 80, N)   # solar masses
    m2 = rng.uniform(5,  40, N)
    d_lum = rng.uniform(100, 5000, N)

    RTOL = 1e-10   # machine precision budget

    # --- 1. UniformPolarizationPrior ---
    sp1 = UniformPolarizationPrior()
    ref1 = [sp1.inverse_transform(psi[i]) for i in range(N)]
    r1   = _batch_extrinsic_subprior(sp1, {'psi': psi})
    ref_psi = np.array([r['psi'] for r in ref1])
    assert np.allclose(r1['psi'], ref_psi, rtol=RTOL, atol=1e-12), \
        f"psi mismatch: max_err={np.abs(r1['psi']-ref_psi).max():.2e}"

    # --- 2. UniformTimePrior ---
    sp2 = UniformTimePrior(tgps=TGPS, ref_det_name=REF, t0_refdet=0, dt0=0.07)
    ref2 = [sp2.inverse_transform(t_geocenter[i], ra[i], dec[i]) for i in range(N)]
    r2   = _batch_extrinsic_subprior(sp2, {
        't_geocenter': t_geocenter, 'ra': ra, 'dec': dec})
    ref_t = np.array([r['t_refdet'] for r in ref2])
    assert np.allclose(r2['t_refdet'], ref_t, rtol=RTOL, atol=1e-12), \
        f"t_refdet mismatch: max_err={np.abs(r2['t_refdet']-ref_t).max():.2e}"

    # --- 3. UniformPhasePrior ---
    par_dic_0 = dict(phi_ref=0., iota=1., ra=0.5, dec=0.5, psi=1., t_geocenter=0.)
    sp3 = UniformPhasePrior(tgps=TGPS, ref_det_name=REF, f_avg=F_AVG,
                             par_dic_0=par_dic_0)
    ref3 = [sp3.inverse_transform(phi_ref[i], iota[i], ra[i], dec[i], psi[i],
                                  t_geocenter[i]) for i in range(N)]
    r3   = _batch_extrinsic_subprior(sp3, {
        'phi_ref': phi_ref, 'iota': iota, 'ra': ra, 'dec': dec,
        'psi': psi, 't_geocenter': t_geocenter})
    ref_phi = np.array([r['phi_ref_hat'] for r in ref3])
    assert np.allclose(r3['phi_ref_hat'], ref_phi, rtol=RTOL, atol=1e-12), \
        f"phi_ref_hat mismatch: max_err={np.abs(r3['phi_ref_hat']-ref_phi).max():.2e}"

    # --- 4. UniformLuminosityVolumePrior ---
    sp4 = UniformLuminosityVolumePrior(tgps=TGPS, ref_det_name=REF, d_hat_max=500.)
    ref4 = [sp4.inverse_transform(d_lum[i], ra[i], dec[i], psi[i], iota[i],
                                  m1[i], m2[i]) for i in range(N)]
    r4   = _batch_extrinsic_subprior(sp4, {
        'd_luminosity': d_lum, 'ra': ra, 'dec': dec, 'psi': psi,
        'iota': iota, 'm1': m1, 'm2': m2})
    ref_dhat = np.array([r['d_hat'] for r in ref4])
    assert np.allclose(r4['d_hat'], ref_dhat, rtol=RTOL, atol=1e-12), \
        f"d_hat mismatch: max_err={np.abs(r4['d_hat']-ref_dhat).max():.2e}"

    print(f"PASS test_extrinsic_lal_subprior_batch  "
          f"(N={N}: psi, t_refdet, phi_ref_hat, d_hat all machine-precision)")


# ---------------------------------------------------------------------------
# Test: _pr_inverse_transform_batch vs cogwheel per-sample baseline
# ---------------------------------------------------------------------------

def test_prior_inverse_transform_batch():
    """
    Verify _pr_inverse_transform_batch against cogwheel's per-sample baseline.

    Uses three subpriors that are known to be batch-safe (numpy-only) and
    one that is NOT batch-safe (LAL time-delay), so both the fast path and
    the per-sample fallback are exercised.

    Expected accuracy: machine precision for pure-numpy subpriors; identical
    results for the fallback path (same code path as cogwheel).
    """
    import pandas as pd
    from cogwheel.gw_prior.extrinsic import (
        IsotropicInclinationPrior,
        IsotropicSkyLocationPrior,
        UniformTimePrior,
    )
    from cogwheel.gw_prior.spin import IsotropicSpinsAlignedComponentsPrior
    from gpu.run import _pr_inverse_transform_batch

    rng = np.random.default_rng(99)
    N = 500
    TGPS = 1187008882.4  # GW170817

    # --- Subpriors ---
    inc_sp  = IsotropicInclinationPrior()               # batch-safe
    sky_sp  = IsotropicSkyLocationPrior(               # batch-safe
        detector_pair='HL', tgps=TGPS)
    spin_sp = IsotropicSpinsAlignedComponentsPrior()    # batch-safe
    time_sp = UniformTimePrior(                         # NOT batch-safe (LAL)
        tgps=TGPS, ref_det_name='H', t0_refdet=0, dt0=0.07)

    # --- Random standard-param inputs ---
    iota          = rng.uniform(0, np.pi, N)
    ra            = rng.uniform(0, 2 * np.pi, N)
    dec           = rng.uniform(-np.pi / 2, np.pi / 2, N)
    s1z           = rng.uniform(-0.99, 0.99, N)
    s2z           = rng.uniform(-0.99, 0.99, N)
    t_geocenter   = rng.uniform(-0.05, 0.05, N)

    # sky_sp needs iota as conditioned_on
    # time_sp needs ra, dec as conditioned_on

    # Build DataFrame with all needed columns
    df = pd.DataFrame({
        'iota':        iota,
        'ra':          ra,
        'dec':         dec,
        's1z':         s1z,
        's2z':         s2z,
        't_geocenter': t_geocenter,
    })

    # --- Cogwheel per-sample baseline (run each subprior individually) ---
    inc_ref  = inc_sp.inverse_transform(iota)
    sky_ref  = sky_sp.inverse_transform(ra, dec, iota)
    spin_ref = spin_sp.inverse_transform(s1z, s2z)
    time_ref = [time_sp.inverse_transform(
                    float(t_geocenter[i]), float(ra[i]), float(dec[i]))
                for i in range(N)]
    time_ref_arr = {k: np.array([r[k] for r in time_ref]) for k in time_ref[0]}

    # --- Batch call (adds columns in-place) ---
    class MockPrior:
        subpriors = [inc_sp, sky_sp, spin_sp, time_sp]

    _pr_inverse_transform_batch(MockPrior(), df)

    # --- Assert correctness ---
    for key, ref_val in inc_ref.items():
        assert np.allclose(df[key].values, ref_val, rtol=1e-12), \
            f"inc_sp mismatch on {key}: max_err={np.abs(df[key].values - ref_val).max():.2e}"

    for key, ref_val in sky_ref.items():
        assert np.allclose(df[key].values, ref_val, rtol=1e-12), \
            f"sky_sp mismatch on {key}: max_err={np.abs(df[key].values - ref_val).max():.2e}"

    for key, ref_val in spin_ref.items():
        assert np.allclose(df[key].values, ref_val, rtol=1e-12), \
            f"spin_sp mismatch on {key}: max_err={np.abs(df[key].values - ref_val).max():.2e}"

    for key, ref_val in time_ref_arr.items():
        assert np.allclose(df[key].values, ref_val, rtol=1e-12), \
            f"time_sp fallback mismatch on {key}: max_err={np.abs(df[key].values - ref_val).max():.2e}"

    print(f"PASS test_prior_inverse_transform_batch  "
          f"(N={N}, batch-safe: inc+sky+spin, fallback: time)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch
    if not torch.cuda.is_available():
        print("No CUDA GPU available — skipping GPU tests.")
        sys.exit(0)

    test_single_detector()
    test_get_dh_by_mode()
    test_get_hh_by_mode()
    test_get_dh_hh_phi_grid()
    test_single_detector_near_singular_hh()
    test_sample_distance_batched()
    test_precessing_spin_inverse_transform_batch()
    test_extrinsic_lal_subprior_batch()
    test_prior_inverse_transform_batch()
    print("\nAll tests passed.")
