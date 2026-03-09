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

    # Memory profile: map the 8.4 GB generator footprint
    python experiments/bench_extrinsic.py \\
        --source-rundir artifacts/experiments/20260304_222558_serial_large_next2048 \\
        --bank         artifacts/banks/bank_large \\
        --event        artifacts/banks/event/tutorial_event.npz \\
        --n-ext 2048 \\
        --memory-profile

Results are written under artifacts/bench_extrinsic/<timestamp>/ and a
summary table is printed to stdout.
"""

import argparse
import cProfile
import io
import os
import pstats
import sys
import tracemalloc
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


# ---------------------------------------------------------------------------
# Memory profiling helpers
# ---------------------------------------------------------------------------

def _collect_ndarrays(obj, path="root", seen=None, max_depth=8):
    """
    Recursively walk an object's attribute tree and return a list of
    (nbytes, path, shape, dtype) for every numpy ndarray found.

    Avoids revisiting the same object (by id) and respects max_depth.
    Does not walk into very large sequences (list/tuple > 200 items) to
    avoid spending minutes iterating over the sky_dict index lists.
    """
    if seen is None:
        seen = set()

    results = []
    obj_id = id(obj)
    if obj_id in seen or max_depth <= 0:
        return results
    seen.add(obj_id)

    if isinstance(obj, np.ndarray):
        results.append((obj.nbytes, path, obj.shape, str(obj.dtype)))
        # Still descend into structured array fields if any (rare), but not
        # into the data itself — the ndarray is already recorded.
        return results

    if hasattr(obj, "__dict__"):
        for attr, val in obj.__dict__.items():
            results.extend(
                _collect_ndarrays(val, f"{path}.{attr}", seen, max_depth - 1)
            )

    if isinstance(obj, dict):
        for k, val in list(obj.items()):
            results.extend(
                _collect_ndarrays(val, f"{path}[{k!r}]", seen, max_depth - 1)
            )

    if isinstance(obj, (list, tuple)) and len(obj) <= 200:
        for i, val in enumerate(obj):
            results.extend(
                _collect_ndarrays(val, f"{path}[{i}]", seen, max_depth - 1)
            )

    return results


def _run_memory_profile(bank_path, event_path, n_ext, seed):
    """
    Build the CoherentExtrinsicSamplesGenerator, then:
      1. Print a numpy-array inventory (object-tree walk) sorted by size.
      2. Print a tracemalloc top-20 report for Python-level allocations.
    """
    import pandas as pd
    from cogwheel.waveform import WaveformGenerator
    from dot_pe.coherent_processing import CoherentExtrinsicSamplesGenerator
    from dot_pe.marginalization import MarginalizationExtrinsicSamplerFreeLikelihood

    # Build the inference context (same as bench_extrinsic normal mode)
    print("\nBuilding inference context (prepare_run_objects)...")
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    ctx = _build_ctx(bank_path, event_path, n_ext, seed)
    bank_ids = list(ctx["banks"].keys())
    bank_id = bank_ids[0]
    first_bank_path = ctx["banks"][bank_id]

    # ---- Build generator under tracemalloc --------------------------------
    print("Starting tracemalloc + building CoherentExtrinsicSamplesGenerator...")
    tracemalloc.start()

    wfg = WaveformGenerator.from_event_data(ctx["event_data"], ctx["approximant"])
    marg_ext_like = MarginalizationExtrinsicSamplerFreeLikelihood(
        ctx["event_data"], wfg, ctx["par_dic_0"], ctx["fbin"],
        coherent_score=ctx["coherent_score_kwargs"],
    )
    ext_generator = CoherentExtrinsicSamplesGenerator(
        likelihood=marg_ext_like,
        intrinsic_bank_file=first_bank_path / "intrinsic_sample_bank.feather",
        waveform_dir=first_bank_path / "waveforms",
        seed=seed,
    )

    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # ---- 1. Numpy array inventory -----------------------------------------
    print("\n" + "=" * 72)
    print("NUMPY ARRAY INVENTORY (object-tree walk of ext_generator)")
    print("=" * 72)
    arrays = _collect_ndarrays(ext_generator)
    arrays.sort(key=lambda x: x[0], reverse=True)
    total_bytes = sum(b for b, *_ in arrays)
    print(f"  Total numpy data found : {total_bytes / 1e9:.3f} GB  ({len(arrays)} arrays)\n")
    print(f"  {'Size (MB)':>10}  {'Shape':>30}  {'Dtype':>10}  Path")
    print(f"  {'-'*10}  {'-'*30}  {'-'*10}  {'-'*40}")
    for nbytes, path, shape, dtype in arrays[:40]:
        mb = nbytes / 1e6
        print(f"  {mb:>10.1f}  {str(shape):>30}  {dtype:>10}  {path}")
    if len(arrays) > 40:
        print(f"  ... ({len(arrays) - 40} more arrays, all smaller than "
              f"{arrays[39][0]/1e6:.1f} MB)")

    # ---- 2. tracemalloc top-20 by size ------------------------------------
    print("\n" + "=" * 72)
    print("TRACEMALLOC TOP 20 (Python-level allocations, grouped by line)")
    print("=" * 72)
    stats = snapshot.statistics("lineno")
    total_tm = sum(s.size for s in stats)
    print(f"  Total tracked by tracemalloc: {total_tm / 1e9:.3f} GB\n")
    for stat in stats[:20]:
        print(f"  {stat.size / 1e6:>8.1f} MB  {stat}")

    print("\n" + "=" * 72)
    print("NOTES")
    print("  * tracemalloc may undercount large numpy arrays allocated via mmap.")
    print("  * Object-tree walk only finds arrays reachable from ext_generator.")
    print("  * If total numpy < 8.4 GB, some arrays live in sub-objects not")
    print("    reachable at max_depth=8 (increase _collect_ndarrays max_depth).")
    print("=" * 72)


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
    p.add_argument(
        "--memory-profile", action="store_true",
        help=(
            "Build CoherentExtrinsicSamplesGenerator then print a full memory "
            "breakdown: numpy-array inventory (object-tree walk) + tracemalloc "
            "top-20. Does not run any extrinsic sampling. "
            "Ignores --n-workers-list and --n-repeat."
        ),
    )
    args = p.parse_args()

    source_rundir = Path(args.source_rundir)
    bank_path     = Path(args.bank)
    event_path    = Path(args.event)

    # Memory-profile mode: no intrinsic indices needed
    if args.memory_profile:
        _run_memory_profile(bank_path, event_path, args.n_ext, args.seed)
        return

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
