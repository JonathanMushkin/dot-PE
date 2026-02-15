#!/usr/bin/env python3
"""
Mock script that runs single_detector.get_response_over_distance_and_lnlike()
with random matrices of the correct shapes and dtypes, and performs line-profiling.

Run from repo root with the project env (e.g. conda activate dot-pe):
  python scripts/mock_profile_get_response_over_distance_and_lnlike.py
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

# Limit NumPy/BLAS/OpenMP to this many threads (set before importing numpy)
ENV_NCORES = "1"
for _v in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_v, ENV_NCORES)

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
i = 2048
e = 2048
o = 100  # n_phi
t = 128
m = 4
M = 10  # mode pairs
d = 2
b = 378
p = 2

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


def run_get_response_over_distance_and_lnlike():
    """Call the target function with mock inputs."""
    (
        dh_weights_dmpb,
        hh_weights_dmppb,
        h_impb,
        timeshift_dbt,
        asd_drift_d,
        n_phi,
        m_arr,
    ) = make_mock_inputs()

    r_iotp, lnlike_iot = SingleDetectorProcessor.get_response_over_distance_and_lnlike(
        dh_weights_dmpb,
        hh_weights_dmppb,
        h_impb,
        timeshift_dbt,
        asd_drift_d,
        n_phi,
        m_arr,
    )

    assert r_iotp.shape == (i, n_phi, t, p), r_iotp.shape
    assert lnlike_iot.shape == (i, n_phi, t), lnlike_iot.shape
    return r_iotp, lnlike_iot


if __name__ == "__main__":
    _print_core_info()

    import line_profiler

    prof = line_profiler.LineProfiler()
    prof.add_function(SingleDetectorProcessor.get_response_over_distance_and_lnlike)

    prof.enable()
    run_get_response_over_distance_and_lnlike()
    prof.disable()

    print(
        "Line profile for SingleDetectorProcessor.get_response_over_distance_and_lnlike:"
    )
    print("=" * 80)
    prof.print_stats()

    script_dir = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = (
        script_dir
        / f"mock_profile_get_response_over_distance_and_lnlike_{timestamp}.txt"
    )
    with open(txt_path, "w") as f:
        f.write(_get_core_info())
        prof.print_stats(stream=f)
    print(f"\nStats written to: {txt_path.resolve()}")
