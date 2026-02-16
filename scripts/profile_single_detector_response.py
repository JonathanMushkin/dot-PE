#!/usr/bin/env python3
"""
Compare original vs optimized get_response_over_distance_and_lnlike: same inputs,
runtime of each, line-profile both, allclose comparison. Single timestamped .txt
has core info + runtimes + comparison + line profiles for both.

Run from repo root:  python scripts/profile_single_detector_response.py
Requires: line_profiler (pip install line_profiler).

Dimension subscripts:
  i = 2048  (intrinsic samples)
  e = 2048  (extrinsic; not used in this function)
  o = 100   (n_phi, orbital phase samples)
  t = 128   (timeshift samples)
  m = 4     (harmonic modes)
  M = 10    (mode pairs: m*(m+1)//2)
  d = 2     (detectors; only index 0 is used in the function)
  b = 378   (frequency bins)
  p = 2     (polarizations: plus, cross)
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Thread env var names (set in loop for 1, 2, 4, 8 cores)
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
# Set default before importing numpy
for _v in _THREAD_VARS:
    os.environ.setdefault(_v, "1")

# Ensure project root is on path when run from scripts/ or repo root
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
from dot_pe.single_detector import SingleDetectorProcessor


def _get_core_info():
    """Return string with cores/threads visible to Python and NumPy."""
    import multiprocessing

    n_cpu = os.cpu_count()
    n_mp = multiprocessing.cpu_count()
    thread_vars = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    )
    lines = [
        "Cores / threads for Python and NumPy:",
        f"  os.cpu_count():              {n_cpu}",
        f"  multiprocessing.cpu_count(): {n_mp}",
    ]
    for v in thread_vars:
        val = os.environ.get(v)
        lines.append(f"  {v}: {val if val is not None else '(not set)'}")
    try:
        for key in ("openblas_info", "blas_info"):
            info = np.__config__.get_info(key)
            if info and isinstance(info, dict):
                lib = info.get("library", key)
                lines.append(f"  NumPy {key}: {lib}")
                break
    except Exception:
        pass
    return "\n".join(lines) + "\n\n"


def _print_core_info():
    """Print cores/threads visible to Python and NumPy."""
    print(_get_core_info())


# Dimension sizes (subscripts have meaning as above)
i = 4096
e = 2048
o = 100  # n_phi
t = 128
m = 4
M = 10  # mode pairs
d = 2
b = 378
p = 2


def _get_dimensions():
    """Return string listing all relevant dimensions for the run."""
    return (
        "Dimensions:\n"
        f"  i = {i}   (intrinsic samples)\n"
        f"  e = {e}   (extrinsic)\n"
        f"  o = {o}   (n_phi, orbital phase)\n"
        f"  t = {t}   (timeshift samples)\n"
        f"  m = {m}   (harmonic modes)\n"
        f"  M = {M}   (mode pairs)\n"
        f"  d = {d}   (detectors)\n"
        f"  b = {b}   (frequency bins)\n"
        f"  p = {p}   (polarizations)\n\n"
    )


# RNG for reproducible random arrays
rng = np.random.default_rng(42)


def make_mock_inputs():
    """Build random arrays with correct shapes and dtypes."""
    # dh_weights_dmpb: (d, m, p, b), complex128
    dh_weights_dmpb = rng.standard_normal((d, m, p, b)) + 1j * rng.standard_normal(
        (d, m, p, b)
    )
    dh_weights_dmpb = dh_weights_dmpb.astype(np.complex128)

    # hh_weights_dmppb: (d, M, p, p, b), complex128
    hh_weights_dmppb = rng.standard_normal((d, M, p, p, b)) + 1j * rng.standard_normal(
        (d, M, p, p, b)
    )
    hh_weights_dmppb = hh_weights_dmppb.astype(np.complex128)

    # h_impb: (i, m, p, b), complex128
    h_impb = rng.standard_normal((i, m, p, b)) + 1j * rng.standard_normal((i, m, p, b))
    h_impb = h_impb.astype(np.complex128)

    # timeshift_dbt: (d, b, t), complex128 (phase factors)
    timeshift_dbt = rng.standard_normal((d, b, t)) + 1j * rng.standard_normal((d, b, t))
    timeshift_dbt = timeshift_dbt.astype(np.complex128)

    # asd_drift_d: (d,), float64
    asd_drift_d = np.abs(rng.standard_normal(d)) + 0.5
    asd_drift_d = asd_drift_d.astype(np.float64)

    # n_phi: int
    n_phi = o

    # m_arr: (m,), float64 (e.g. mode numbers for orbital phase)
    m_arr = np.array([2.0, 2.0, 3.0, 3.0], dtype=np.float64)  # typical l,m mode values

    return (
        dh_weights_dmpb,
        hh_weights_dmppb,
        h_impb,
        timeshift_dbt,
        asd_drift_d,
        n_phi,
        m_arr,
    )


def run_both_and_compare(inputs):
    """
    Run original and optimized with same inputs; check allclose and report timings.
    Returns (comparison_text, t_orig, t_opt).
    """
    import time

    # Warmup
    SingleDetectorProcessor.get_response_over_distance_and_lnlike(*inputs)
    SingleDetectorProcessor.get_response_over_distance_and_lnlike_optimized(*inputs)

    n_run = 3
    t0 = time.perf_counter()
    for _ in range(n_run):
        r_ref, ln_ref = SingleDetectorProcessor.get_response_over_distance_and_lnlike(
            *inputs
        )
    t_orig = (time.perf_counter() - t0) / n_run

    t0 = time.perf_counter()
    for _ in range(n_run):
        r_opt, ln_opt = (
            SingleDetectorProcessor.get_response_over_distance_and_lnlike_optimized(
                *inputs
            )
        )
    t_opt = (time.perf_counter() - t0) / n_run

    r_ok = np.allclose(r_ref, r_opt, rtol=1e-10, atol=1e-10)
    ln_ok = np.allclose(ln_ref, ln_opt, rtol=1e-10, atol=1e-10)

    lines = [
        "Comparison (original vs optimized):",
        f"  r_iotp allclose: {r_ok}",
        f"  lnlike_iot allclose: {ln_ok}",
    ]
    if not r_ok:
        lines.append(f"  r_iotp max|diff|: {np.abs(r_ref - r_opt).max()}")
    if not ln_ok:
        lines.append(f"  lnlike_iot max|diff|: {np.abs(ln_ref - ln_opt).max()}")
    lines.append(f"  Original  mean time ({n_run} runs): {t_orig:.4f} s")
    lines.append(f"  Optimized mean time ({n_run} runs): {t_opt:.4f} s")
    lines.append("")
    comparison_text = "\n".join(lines)
    return comparison_text, t_orig, t_opt


if __name__ == "__main__":
    import line_profiler

    CORES_LIST = [1, 2, 4, 8]
    inputs = make_mock_inputs()
    dimensions = _get_dimensions()
    core_info = _get_core_info()

    script_dir = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = script_dir / f"profile_single_detector_response_{timestamp}.txt"
    runtime_summary = []

    with open(txt_path, "w") as f:
        f.write(dimensions)
        f.write(core_info)
        for n_cores in CORES_LIST:
            for _v in _THREAD_VARS:
                os.environ[_v] = str(n_cores)
            section = f"\n{'=' * 80}\n  N_CORES = {n_cores}\n{'=' * 80}\n\n"
            f.write(section)
            comparison_text, t_orig, t_opt = run_both_and_compare(inputs)
            runtime_summary.append((n_cores, t_orig, t_opt))
            f.write(comparison_text)
            prof = line_profiler.LineProfiler()
            prof.add_function(
                SingleDetectorProcessor.get_response_over_distance_and_lnlike
            )
            prof.add_function(
                SingleDetectorProcessor.get_response_over_distance_and_lnlike_optimized
            )
            prof.enable()
            SingleDetectorProcessor.get_response_over_distance_and_lnlike(*inputs)
            SingleDetectorProcessor.get_response_over_distance_and_lnlike_optimized(
                *inputs
            )
            prof.disable()
            f.write("Line profiles (original and optimized):\n")
            f.write("=" * 80 + "\n")
            prof.print_stats(stream=f)
            f.write("\n")
        summary_lines = [
            "",
            "=" * 80,
            "Runtime summary (mean time in s, 3 runs per method)",
            "=" * 80,
            f"  {'cores':>6}  {'original':>10}  {'optimized':>10}",
            "-" * 80,
        ]
        for n_cores, t_orig, t_opt in runtime_summary:
            summary_lines.append(f"  {n_cores:>6}  {t_orig:>10.4f}  {t_opt:>10.4f}")
        summary_lines.append("")
        summary_text = "\n".join(summary_lines)
        f.write(summary_text)
