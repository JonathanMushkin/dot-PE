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
    print("\nAll tests passed.")
