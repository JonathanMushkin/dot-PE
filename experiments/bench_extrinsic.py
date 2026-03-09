#!/usr/bin/env python3
"""
Isolated benchmark for the extrinsic sampling stage.

Loads pre-saved intrinsic_samples.npz (from a prior serial run) to skip
incoherent selection, then benchmarks inference.draw_extrinsic_samples()
(n_workers=1) vs draw_extrinsic_samples_parallel() for each requested n_workers.

Usage
-----
    python experiments/bench_extrinsic.py \\
        --source-rundir artifacts/experiments/20260304_222558_serial_large_next2048 \\
        --bank         artifacts/banks/bank_large \\
        --event        artifacts/banks/event/tutorial_event.npz \\
        --n-workers-list 1 2 4 8 \\
        --n-ext 2048

    # Profile step1 (_get_many_dh_hh) vs step2 (get_marginalization_info) split:
    python experiments/bench_extrinsic.py \\
        --source-rundir artifacts/experiments/20260304_222558_serial_large_next2048 \\
        --bank         artifacts/banks/bank_large \\
        --event        artifacts/banks/event/tutorial_event.npz \\
        --n-workers-list 1 \\
        --n-ext 2048 \\
        --profile-split

Results are written under artifacts/bench_extrinsic/<timestamp>/ and a
summary table is printed to stdout.
"""

import argparse
import cProfile
import io
import os
import pstats
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dot_pe import inference
from dot_pe.parallel_extrinsic import draw_extrinsic_samples_parallel

ARTIFACTS = ROOT / "artifacts"
BENCH_DIR = ARTIFACTS / "bench_extrinsic"


def _build_ctx(bank_path, event_path, n_ext, seed):
    """Build the minimal inference context needed for extrinsic sampling."""
    return inference.prepare_run_objects(
        event=str(event_path),
        bank_folder=str(bank_path),
        n_int=None,
        n_ext=n_ext,
        n_phi=100,
        n_t=128,
        blocksize=512,
        single_detector_blocksize=512,
        i_int_start=0,
        seed=seed,
        load_inds=False,
        inds_path=None,
        size_limit=10**7,
        draw_subset=True,
        n_draws=None,
        event_dir=None,
        rundir=str(BENCH_DIR / "_ctx_tmp"),
        coherent_score_min_n_effective_prior=100,
        max_incoherent_lnlike_drop=20,
        max_bestfit_lnlike_diff=20,
        mchirp_guess=None,
        extrinsic_samples=None,
        n_phi_incoherent=None,
        preselected_indices=None,
        bank_logw_override=None,
        coherent_posterior_kwargs={},
    )


def _run_one(n_workers, ctx, selected_inds_by_bank, n_ext, seed, run_out_dir):
    """Run one extrinsic sampling trial; return wall time in seconds."""
    run_out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if n_workers == 1:
        inference.draw_extrinsic_samples(
            banks=ctx["banks"],
            event_data=ctx["event_data"],
            par_dic_0=ctx["par_dic_0"],
            fbin=ctx["fbin"],
            approximant=ctx["approximant"],
            selected_inds_by_bank=selected_inds_by_bank,
            coherent_score_kwargs=ctx["coherent_score_kwargs"],
            seed=seed,
            n_ext=n_ext,
            rundir=run_out_dir,
            extrinsic_samples=None,
        )
    else:
        draw_extrinsic_samples_parallel(
            banks=ctx["banks"],
            event_data=ctx["event_data"],
            par_dic_0=ctx["par_dic_0"],
            fbin=ctx["fbin"],
            approximant=ctx["approximant"],
            selected_inds_by_bank=selected_inds_by_bank,
            coherent_score_kwargs=ctx["coherent_score_kwargs"],
            seed=seed,
            n_ext=n_ext,
            rundir=run_out_dir,
            n_workers=n_workers,
        )
    return time.time() - t0


def _run_one_profiled(ctx, selected_inds_by_bank, n_ext, seed, run_out_dir):
    """Run serial extrinsic sampling under cProfile; return (wall_s, pstats.Stats)."""
    run_out_dir.mkdir(parents=True, exist_ok=True)
    pr = cProfile.Profile()
    t0 = time.time()
    pr.enable()
    inference.draw_extrinsic_samples(
        banks=ctx["banks"],
        event_data=ctx["event_data"],
        par_dic_0=ctx["par_dic_0"],
        fbin=ctx["fbin"],
        approximant=ctx["approximant"],
        selected_inds_by_bank=selected_inds_by_bank,
        coherent_score_kwargs=ctx["coherent_score_kwargs"],
        seed=seed,
        n_ext=n_ext,
        rundir=run_out_dir,
        extrinsic_samples=None,
    )
    pr.disable()
    wall = time.time() - t0
    pr.dump_stats(str(run_out_dir / "profile_split.prof"))
    sio = io.StringIO()
    ps = pstats.Stats(pr, stream=sio).sort_stats("tottime")
    return wall, ps


def _print_split_report(wall_s, ps):
    """Extract and print step1 vs step2 timing from a pstats.Stats object."""
    # Pull tottime for the two key functions by name substring match.
    # pstats stores keys as (filename, lineno, funcname).
    stats = ps.stats  # dict: (file, line, func) -> (cc, nc, tt, ct, callers)
    targets = {
        "_get_many_dh_hh": 0.0,
        "get_marginalization_info": 0.0,
    }
    for (fname, lineno, funcname), (cc, nc, tt, ct, callers) in stats.items():
        for key in targets:
            if funcname == key:
                targets[key] += tt

    step1 = targets["_get_many_dh_hh"]
    step2 = targets["get_marginalization_info"]
    other = wall_s - step1 - step2

    print("\n" + "=" * 60)
    print("Step 1/2 split (serial, n_workers=1)")
    print("-" * 60)
    print(f"  Total wall time              : {wall_s:>8.1f} s  (100%)")
    print(f"  Step 1 — _get_many_dh_hh    : {step1:>8.1f} s  ({100*step1/wall_s:.1f}%)")
    print(f"  Step 2 — get_marg_info loop  : {step2:>8.1f} s  ({100*step2/wall_s:.1f}%)")
    print(f"  Other (filter, I/O, merge)   : {other:>8.1f} s  ({100*other/wall_s:.1f}%)")
    print("=" * 60)
    print()
    print("Interpretation:")
    if step2 / wall_s >= 0.5:
        print(f"  Step 2 dominates ({100*step2/wall_s:.0f}%). Option A viable:")
        print("  parallelize get_marginalization_info, main process does _get_many_dh_hh.")
    else:
        print(f"  Step 1 dominates ({100*step1/wall_s:.0f}%). Option A won't help.")
        print("  Proceed to Option B (shared memory) or Option C (thin-worker redesign).")

    # Also print top-20 by tottime for full context
    print("\n--- Top 20 functions by tottime ---")
    sio = io.StringIO()
    ps.stream = sio
    ps.sort_stats("tottime").print_stats(20)
    print(sio.getvalue())


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source-rundir", required=True,
        help="Path to a prior serial run whose banks/bank_0/intrinsic_samples.npz is reused",
    )
    p.add_argument("--bank",  required=True, help="Path to bank folder")
    p.add_argument("--event", required=True, help="Path to event .npz file")
    p.add_argument(
        "--n-workers-list", nargs="+", type=int, default=[1, 2, 4, 8],
        metavar="N",
        help="List of n_workers values to benchmark (default: 1 2 4 8)",
    )
    p.add_argument("--n-ext", type=int, default=2048)
    p.add_argument("--n-repeat", type=int, default=1,
                   help="Repeat each trial N times and report mean wall time")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--profile-split", action="store_true",
        help=(
            "Run n_workers=1 under cProfile and print step1 (_get_many_dh_hh) "
            "vs step2 (get_marginalization_info) timing split. "
            "Ignores --n-workers-list and --n-repeat."
        ),
    )
    args = p.parse_args()

    source_rundir = Path(args.source_rundir)
    bank_path     = Path(args.bank)
    event_path    = Path(args.event)

    # Load pre-saved intrinsic indices (bank_0 only — single-bank runs)
    inds_path = source_rundir / "banks" / "bank_0" / "intrinsic_samples.npz"
    if not inds_path.exists():
        print(f"ERROR: intrinsic_samples.npz not found at {inds_path}", file=sys.stderr)
        print("Provide a --source-rundir from a completed serial/mp/swarm run.", file=sys.stderr)
        sys.exit(1)

    d = np.load(inds_path)
    inds = d["inds"]
    print(f"Loaded {len(inds)} intrinsic indices from {inds_path}")

    # Build inference context (once; reused for all trials)
    print("\nBuilding inference context (prepare_run_objects)...")
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    ctx = _build_ctx(bank_path, event_path, args.n_ext, args.seed)

    # Determine bank_id from context
    bank_ids = list(ctx["banks"].keys())
    if len(bank_ids) != 1:
        print(f"ERROR: bench_extrinsic supports single-bank only; got {bank_ids}", file=sys.stderr)
        sys.exit(1)
    bank_id = bank_ids[0]
    selected_inds_by_bank = {bank_id: inds}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = BENCH_DIR / ts
    session_dir.mkdir(parents=True, exist_ok=True)

    # Profile-split mode: single serial run under cProfile, report step1 vs step2
    if args.profile_split:
        print("\n--- profile-split mode (n_workers=1, cProfile) ---")
        run_out = session_dir / "profile_split_w1"
        wall, ps = _run_one_profiled(
            ctx, selected_inds_by_bank, args.n_ext, args.seed, run_out
        )
        _print_split_report(wall, ps)
        print(f"Full profile saved to: {run_out / 'profile_split.prof'}")
        return

    # Run benchmarks
    results = []
    for n_workers in args.n_workers_list:
        wall_times = []
        for rep in range(args.n_repeat):
            run_out = session_dir / f"w{n_workers}_rep{rep}"
            print(f"\n--- n_workers={n_workers}, rep={rep+1}/{args.n_repeat} ---")
            wall = _run_one(
                n_workers, ctx, selected_inds_by_bank, args.n_ext, args.seed, run_out
            )
            wall_times.append(wall)
            print(f"  wall: {wall:.1f} s")

        mean_wall = sum(wall_times) / len(wall_times)
        results.append((n_workers, mean_wall, wall_times))

    # Print summary table
    serial_wall = next((w for nw, w, _ in results if nw == 1), None)
    print("\n" + "=" * 60)
    print(f"{'n_workers':>10} {'wall_s':>10} {'speedup':>10}")
    print("-" * 60)
    for n_workers, mean_wall, _ in results:
        speedup = f"{serial_wall / mean_wall:.2f}x" if serial_wall else "—"
        print(f"{n_workers:>10} {mean_wall:>10.1f} {speedup:>10}")
    print("=" * 60)
    print(f"\nResults written to: {session_dir}")


if __name__ == "__main__":
    main()
