#!/usr/bin/env python3
"""
Submit a benchmarking experiment as a timestamped LSF job.

Usage examples:

    # Smoke test — serial, small bank
    python experiments/run_experiment.py --mode serial --bank small \
        --n-int 256 --n-ext 128 --queue physics-short --dry-run

    # Phase B — MP, 8 workers
    python experiments/run_experiment.py --mode mp --bank small \
        --n-ext 512 --n-workers 8

    # Phase D — swarm, large bank
    python experiments/run_experiment.py --mode swarm --bank large \
        --n-ext 2048
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
BANKS_DIR = ARTIFACTS / "banks"
EXPERIMENTS_DIR = ARTIFACTS / "experiments"

# ── queue defaults ─────────────────────────────────────────────────────────────

_DEFAULT_QUEUE = {
    "serial": "physics-medium",
    "mp":     "physics-medium",
    "swarm":  "physics-short",
}

_WALL_LIMITS = {
    ("serial", "small"): 120,
    ("serial", "large"): 480,
    ("mp",     "small"): 120,
    ("mp",     "large"): 240,
    ("swarm",  "small"): 240,
    ("swarm",  "large"): 240,
}


def _mem_per_slot_mb(mode, bank, n_ext, n_slots=1, n_ext_workers=1):
    """Memory per LSF slot in MB.

    After thin-worker refactor + malloc_trim + incremental Stage-5 merge:
      - Coherent workers (swarm array job): ~1–2 GB each — unchanged.
      - Swarm orchestrator: ctx ~10 GB + precompute spike freed via malloc_trim
        + Stage 5 incremental merge (size_limit samples at a time).
      - MP: precompute spike freed via malloc_trim before Pool fork;
        workers inherit ctx via COW read-only + ~5 GB own new pages each.
      - Parallel extrinsic workers: ~200 MB new pages each (small per-batch alloc).

    Calibrated observations (n_ext has negligible effect):
      serial/small: 9206-9270 MB  -> 13000 MB
      serial/large: 10473 MB      -> 15000 MB
      swarm/small orchestrator:  ~17 GB peak -> 22000 MB
      swarm/large orchestrator:  ~30 GB peak (malloc_trim helps) -> 40000 MB
      mp workers: ~5 GB new pages/worker (COW-shared pages not double-counted)
    """
    if mode == "serial":
        total = 13000 if bank == "small" else 15000
    elif mode == "swarm":
        # Orchestrator: ctx ~10 GB + Stage 2.5 precompute spike freed via
        # malloc_trim + Stage 5 incremental merge (≤ size_limit samples).
        # With parallel extrinsic: add ~200 MB per ext worker (small new pages).
        base = 22000 if bank == "small" else 40000
        total = base + max(0, n_ext_workers - 1) * 200
    else:  # mp — precompute spike freed via malloc_trim before fork;
           # workers (COW) only pay for new pages (~7 GB each for large bank).
        base = 15000 if bank == "small" else 20000
        per_worker = 4000 if bank == "small" else 8000
        total = base + n_slots * per_worker + max(0, n_ext_workers - 1) * 200
    return max(1024, total // n_slots)


def _make_rundir(mode, bank, n_ext, n_workers=None, n_cores=None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if mode == "serial" and n_cores and n_cores > 1:
        suffix = f"_c{n_cores}"
    elif n_workers is not None:
        suffix = f"_w{n_workers}"
    else:
        suffix = ""
    name = f"{ts}_{mode}{suffix}_{bank}_next{n_ext}"
    d = EXPERIMENTS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _conda_header():
    return (
        'source "$(conda info --base)/etc/profile.d/conda.sh"\n'
        "conda activate dot-pe\n"
        f"export PYTHONPATH={ROOT}:$PYTHONPATH\n"
    )


def _build_script_serial(args, rundir, bank_path, event_path, log_path):
    wall = _WALL_LIMITS.get(("serial", args.bank), 120)
    queue = args.queue or _DEFAULT_QUEUE["serial"]
    n_cores = args.n_cores or 1
    n_int_flag = f"--n_int {args.n_int}" if args.n_int else ""
    ext_flag = f"--extrinsic_samples {args.extrinsic_samples}" if args.extrinsic_samples else ""
    mem = _mem_per_slot_mb("serial", args.bank, args.n_ext, n_slots=n_cores)
    span_line = '#BSUB -R "span[hosts=1]"\n' if n_cores > 1 else ""

    script = f"""\
#!/bin/bash
#BSUB -J serial_{args.bank}_next{args.n_ext}
#BSUB -q {queue}
#BSUB -n {n_cores}
{span_line}#BSUB -R "rusage[mem={mem}]"
#BSUB -W {wall}
#BSUB -o {log_path}
#BSUB -e {log_path}.err

{_conda_header()}
export OMP_NUM_THREADS={n_cores}
export MKL_NUM_THREADS={n_cores}
export OPENBLAS_NUM_THREADS={n_cores}
export NUMEXPR_NUM_THREADS={n_cores}

cd {ROOT} && python -m dot_pe.inference \\
    --event {event_path} \\
    --bank_folder {bank_path} \\
    --rundir {rundir} \\
    --n_ext {args.n_ext} \\
    --seed {args.seed} \\
    {n_int_flag} \\
    {ext_flag}
"""
    return script


_DEFAULT_MP_WORKERS = 8
_DEFAULT_MAX_CONCURRENT = 20

def _build_script_mp(args, rundir, bank_path, event_path, log_path):
    wall = _WALL_LIMITS.get(("mp", args.bank), 120)
    queue = args.queue or _DEFAULT_QUEUE["mp"]
    n_workers = args.n_workers or _DEFAULT_MP_WORKERS
    n_ext_workers = args.n_ext_workers or 1
    n_int_flag = f"--n-int {args.n_int}" if args.n_int else ""
    n_ext_workers_flag = f"--n-ext-workers {n_ext_workers}" if n_ext_workers > 1 else ""
    mem = _mem_per_slot_mb("mp", args.bank, args.n_ext, n_slots=n_workers,
                           n_ext_workers=n_ext_workers)

    script = f"""\
#!/bin/bash
#BSUB -J mp_{args.bank}_w{n_workers}_next{args.n_ext}
#BSUB -q {queue}
#BSUB -n {n_workers}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem={mem}]"
#BSUB -W {wall}
#BSUB -o {log_path}
#BSUB -e {log_path}.err

{_conda_header()}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python {ROOT}/MP/run_mp.py \\
    --event {event_path} \\
    --bank {bank_path} \\
    --rundir {rundir} \\
    --n-ext {args.n_ext} \\
    --n-workers {n_workers} \\
    --seed {args.seed} \\
    {n_ext_workers_flag} \\
    {n_int_flag}
"""
    return script


def _build_script_swarm(args, rundir, bank_path, event_path, log_path):
    wall = _WALL_LIMITS.get(("swarm", args.bank), 240)
    queue = args.queue or _DEFAULT_QUEUE["swarm"]
    max_concurrent = args.max_concurrent or _DEFAULT_MAX_CONCURRENT
    n_ext_workers = args.n_ext_workers or 1
    n_int_flag = f"--n-int {args.n_int}" if args.n_int else ""
    n_ext_workers_flag = f"--n-ext-workers {n_ext_workers}" if n_ext_workers > 1 else ""
    mem = _mem_per_slot_mb("swarm", args.bank, args.n_ext, n_slots=n_ext_workers,
                           n_ext_workers=n_ext_workers)
    # When n_ext_workers > 1, request multiple cores on one host for the parallel fork pool
    n_cores = n_ext_workers if n_ext_workers > 1 else 1
    span_line = '#BSUB -R "span[hosts=1]"\n' if n_ext_workers > 1 else ""

    script = f"""\
#!/bin/bash
#BSUB -J swarm_{args.bank}_next{args.n_ext}
#BSUB -q {queue}
#BSUB -n {n_cores}
{span_line}#BSUB -R "rusage[mem={mem}]"
#BSUB -W {wall}
#BSUB -o {log_path}
#BSUB -e {log_path}.err

{_conda_header()}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python {ROOT}/lsf_swarm/run_swarm.py \\
    --event {event_path} \\
    --bank {bank_path} \\
    --rundir {rundir} \\
    --n-ext {args.n_ext} \\
    --seed {args.seed} \\
    --max-concurrent {max_concurrent} \\
    {n_ext_workers_flag} \\
    {n_int_flag}
"""
    return script


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", required=True, choices=["serial", "mp", "swarm"])
    p.add_argument("--bank", required=True, choices=["small", "large"])
    p.add_argument("--n-ext", type=int, default=512)
    p.add_argument("--n-int", type=int, default=None,
                   help="Limit intrinsic samples (default: full bank)")
    p.add_argument("--n-workers", type=int, default=None,
                   help="MP workers (default 8)")
    p.add_argument("--n-cores", type=int, default=None,
                   help="CPU cores for serial mode; enables multi-threaded BLAS (default: 1)")
    p.add_argument("--max-concurrent", type=int, default=None,
                   help=f"Swarm: max simultaneously running worker tasks (default: {_DEFAULT_MAX_CONCURRENT})")
    p.add_argument("--n-ext-workers", type=int, default=1,
                   help="Workers for parallel extrinsic MI collection in mp/swarm (default: 1 = serial)")
    p.add_argument("--extrinsic-samples", default=None,
                   help="Path to cached extrinsic samples pkl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--queue", default=None,
                   help="Override LSF queue")
    p.add_argument("--dry-run", action="store_true",
                   help="Print bsub script without submitting")
    args = p.parse_args()

    bank_path = BANKS_DIR / f"bank_{args.bank}"
    event_path = BANKS_DIR / "event" / "tutorial_event.npz"

    if not bank_path.exists():
        print(f"ERROR: bank not found: {bank_path}", file=sys.stderr)
        print("Run: python test_data/setup.py --base-dir artifacts/banks", file=sys.stderr)
        sys.exit(1)
    if not event_path.exists():
        print(f"ERROR: event not found: {event_path}", file=sys.stderr)
        sys.exit(1)

    if args.mode == "mp":
        n_workers_for_dir = args.n_workers or _DEFAULT_MP_WORKERS
    elif args.mode == "swarm":
        n_workers_for_dir = args.max_concurrent or _DEFAULT_MAX_CONCURRENT
    else:
        n_workers_for_dir = None
    n_cores_for_dir = (args.n_cores or 1) if args.mode == "serial" else None
    rundir = _make_rundir(args.mode, args.bank, args.n_ext,
                          n_workers=n_workers_for_dir, n_cores=n_cores_for_dir)
    log_path = rundir / "run.log"

    builders = {
        "serial": _build_script_serial,
        "mp":     _build_script_mp,
        "swarm":  _build_script_swarm,
    }
    script = builders[args.mode](args, rundir, bank_path, event_path, log_path)

    if args.dry_run:
        print("=== DRY RUN — bsub script ===")
        print(script)
        print(f"rundir: {rundir}")
        return

    result = subprocess.run(
        ["bsub"],
        input=script,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("bsub FAILED:")
        print(result.stderr)
        sys.exit(1)

    print(result.stdout.strip())
    print(f"rundir : {rundir}")
    print(f"log    : {log_path}")

    # Append to PROGRESS.md
    progress_path = Path(__file__).parent / "PROGRESS.md"
    import re
    m = re.search(r"Job <(\d+)>", result.stdout)
    job_id = m.group(1) if m else "unknown"
    with open(progress_path, "a") as f:
        from datetime import datetime as _dt
        ts = _dt.now().strftime("%Y-%m-%d %H:%M")
        f.write(
            f"\n| {ts} | {job_id} | {args.mode} | {args.bank} | "
            f"{args.n_ext} | {args.n_int or 'full'} | "
            f"{args.n_workers or '-'} | pending | — |\n"
        )


if __name__ == "__main__":
    main()
