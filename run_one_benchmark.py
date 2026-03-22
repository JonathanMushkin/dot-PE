"""Run one benchmark configuration.

Usage:
    python run_one_benchmark.py <module> <n_workers> <label> <omp_threads>

    module     : inference | mp_inference | dask_inference
    n_workers  : number of workers (ignored for inference)
    label      : output subdirectory name under gpu/artifacts/benchmark_scaling/
    omp_threads: OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS

Example:
    python run_one_benchmark.py dask_inference 16 benchmark_dask_16w 1

Must be invoked as a fresh subprocess (env vars are set before numpy loads).
The if __name__ == '__main__' guard is required: Dask uses spawn on Python 3.13
and re-imports __main__ during worker bootstrap — without the guard it would
try to re-run the whole script in every worker.
"""

import sys


def main():
    import os

    module_name = sys.argv[1]
    n_workers   = int(sys.argv[2])   # 0 = serial (no n_workers kwarg)
    label       = sys.argv[3]
    omp_threads = int(sys.argv[4])

    os.environ["OMP_NUM_THREADS"]      = str(omp_threads)
    os.environ["MKL_NUM_THREADS"]      = str(omp_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(omp_threads)
    os.environ["NUMEXPR_NUM_THREADS"]  = str(omp_threads)

    # ── imports (after env vars) ───────────────────────────────────────
    import importlib
    import io
    import json
    import re
    import time
    import warnings
    from pathlib import Path

    warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

    REPO_ROOT   = Path(__file__).resolve().parent
    EVENT_PATH  = REPO_ROOT / "gpu/artifacts/nb04/nb04_event.npz"
    BANK_FOLDER = REPO_ROOT / "gpu/artifacts/profile_run/test_bank_1048576"
    OUTPUT_ROOT = REPO_ROOT / "gpu/artifacts/benchmark_scaling"

    rundir = OUTPUT_ROOT / label
    rundir.mkdir(parents=True, exist_ok=True)

    assert EVENT_PATH.exists(),  f"Event not found: {EVENT_PATH}"
    assert BANK_FOLDER.exists(), f"Bank not found:  {BANK_FOLDER}"

    # ── tee: write to terminal AND buffer simultaneously ───────────────
    class _Tee:
        def __init__(self, original):
            self.original = original
            self.buf = io.StringIO()
        def write(self, text):
            self.original.write(text)
            self.original.flush()
            self.buf.write(text)
        def flush(self):
            self.original.flush()
        def getvalue(self):
            return self.buf.getvalue()

    tee = _Tee(sys.stdout)
    sys.stdout = tee

    # ── load module ────────────────────────────────────────────────────
    mod = importlib.import_module(f"dot_pe.{module_name}")

    # ── run kwargs ─────────────────────────────────────────────────────
    kwargs = dict(
        event       = str(EVENT_PATH),
        bank_folder = str(BANK_FOLDER),
        rundir      = str(rundir),
        n_ext       = 512,
        n_phi       = 32,
        n_t         = 64,
        blocksize                  = 2048,
        single_detector_blocksize  = 2048,
        mchirp_guess               = None,
        seed                       = 42,
        max_incoherent_lnlike_drop = 20.0,
    )
    if n_workers > 0:
        kwargs["n_workers"] = n_workers

    # ── run (use run_timed() if available for per-stage output) ────────
    run_fn = getattr(mod, "run_timed", mod.run)
    t0 = time.perf_counter()
    run_fn(**kwargs)
    total = time.perf_counter() - t0

    sys.stdout = tee.original
    captured = tee.getvalue()

    # ── save log ───────────────────────────────────────────────────────
    log_path = rundir / "benchmark_run.log"
    with open(log_path, "w") as f:
        f.write(captured)

    # ── parse stage timings ────────────────────────────────────────────
    stage_re = re.compile(r"Stage (\d)\s+\((\w+)\)\s+([\d.]+)\s+s")
    timings = {}
    for m in stage_re.finditer(captured):
        key = f"{m.group(1)}_{m.group(2)}"
        timings[key] = float(m.group(3))

    timings["total"]       = round(total, 3)
    timings["label"]       = label
    timings["module"]      = module_name
    timings["n_workers"]   = n_workers
    timings["omp_threads"] = omp_threads

    timing_path = rundir / "benchmark_timings.json"
    with open(timing_path, "w") as f:
        json.dump(timings, f, indent=2)

    print(f"\n[benchmark] {label}: total={total:.1f}s  timings -> {timing_path}")


if __name__ == "__main__":
    main()
