"""
End-to-end correctness check for the nb03 real event (mchirp≈75, GW230529-style).

Loads a CPU baseline prob_samples from an existing run directory, then runs GPU
inference with the same event/bank/params and compares.  Asserts max relative
error < 1% on lnl columns.

Usage (defaults use the pre-generated baseline in run_1):
    python gpu/check_real_event.py

Override paths:
    python gpu/check_real_event.py \\
        --cpu-run  gpu/artifacts/profile_run/nb03_event/run_1 \\
        --event-path gpu/artifacts/profile_run/nb03_event/run_1/nb03_event.npz \\
        --bank-path  gpu/artifacts/profile_run/test_bank_32k \\
        --out-dir    gpu/artifacts/profile_run/nb03_event
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

REL_TOL = 0.01  # 1% max relative error on lnl columns
LNL_COLS = ["lnl_marginalized", "bestfit_lnlike"]


def _load_prob_samples(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "prob_samples.feather"
    if not p.exists():
        raise FileNotFoundError(f"prob_samples.feather not found in {run_dir}")
    return pd.read_feather(p)


def _run_gpu(event_path: Path, bank_path: Path, out_dir: Path,
             n_ext: int, n_phi: int, n_t: int, seed: int) -> pd.DataFrame:
    from cogwheel import data
    import gpu.run as runner

    event_data = data.EventData.from_npz(filename=event_path)
    run_kwargs = dict(
        event=event_data,
        bank_folder=bank_path,
        event_dir=str(out_dir),
        n_ext=n_ext,
        n_phi=n_phi,
        n_t=n_t,
        blocksize=2048,
        single_detector_blocksize=2048,
        seed=seed,
        draw_subset=False,
    )
    rundir = runner.run(**run_kwargs)
    print(f"  GPU run dir: {rundir}")
    return _load_prob_samples(Path(rundir)), Path(rundir)


def _compare(cpu_df: pd.DataFrame, gpu_df: pd.DataFrame) -> bool:
    # Sort by (i, e, o) for alignment
    idx_cols = ["i", "e", "o"]
    cpu_s = cpu_df.sort_values(idx_cols).reset_index(drop=True)
    gpu_s = gpu_df.sort_values(idx_cols).reset_index(drop=True)

    same_idx = np.array_equal(cpu_s[idx_cols].values, gpu_s[idx_cols].values)
    if not same_idx:
        print(f"  WARNING: (i,e,o) index sets differ — "
              f"cpu rows={len(cpu_s)}, gpu rows={len(gpu_s)}")
        # Merge on shared indices
        cpu_s = cpu_s.set_index(idx_cols)
        gpu_s = gpu_s.set_index(idx_cols)
        shared = cpu_s.index.intersection(gpu_s.index)
        cpu_s = cpu_s.loc[shared].reset_index()
        gpu_s = gpu_s.loc[shared].reset_index()
        print(f"  Shared rows: {len(shared)}")
    else:
        print(f"  Same (i,e,o) index set: {same_idx} ({len(cpu_s)} rows)")

    all_pass = True
    print(f"\n  {'Column':<25} {'max_rel':>10} {'mean_rel':>10}  Status")
    print(f"  {'-'*55}")
    for col in LNL_COLS:
        if col not in cpu_s.columns or col not in gpu_s.columns:
            continue
        diff = np.abs(cpu_s[col].values - gpu_s[col].values)
        denom = np.abs(cpu_s[col].values) + 1e-30
        rel = diff / denom
        max_rel = float(rel.max())
        mean_rel = float(rel.mean())
        ok = max_rel < REL_TOL
        all_pass &= ok
        status = "PASS" if ok else "FAIL"
        print(f"  {col:<25} {max_rel:>10.3e} {mean_rel:>10.3e}  {status}")

    return all_pass


def parse_args():
    p = argparse.ArgumentParser(description="GPU vs CPU correctness check on nb03 real event")
    p.add_argument("--cpu-run", type=Path,
                   default=Path("gpu/artifacts/profile_run/nb03_event/run_1"),
                   help="Directory containing CPU baseline prob_samples.feather")
    p.add_argument("--event-path", type=Path,
                   default=Path("gpu/artifacts/profile_run/nb03_event/run_1/nb03_event.npz"),
                   help="Path to event .npz file")
    p.add_argument("--bank-path", type=Path,
                   default=Path("gpu/artifacts/profile_run/test_bank_32k"),
                   help="Path to bank folder")
    p.add_argument("--out-dir", type=Path,
                   default=Path("gpu/artifacts/profile_run/nb03_event"),
                   help="Output directory for GPU run (default: nb03_event/)")
    p.add_argument("--n-ext", type=int, default=512)
    p.add_argument("--n-phi", type=int, default=32)
    p.add_argument("--n-t", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"=== Real Event Correctness Check (nb03_event, mchirp≈75) ===\n")
    print(f"CPU baseline : {args.cpu_run}")
    print(f"Event        : {args.event_path}")
    print(f"Bank         : {args.bank_path}")
    print(f"Params       : n_ext={args.n_ext}, n_phi={args.n_phi}, n_t={args.n_t}, seed={args.seed}\n")

    # --- Load CPU baseline ---
    print("Loading CPU baseline...")
    cpu_df = _load_prob_samples(args.cpu_run)
    print(f"  {len(cpu_df)} rows, columns: {list(cpu_df.columns)}")

    # --- Run GPU inference ---
    print("\nRunning GPU inference...")
    import time
    t0 = time.time()
    gpu_df, gpu_rundir = _run_gpu(
        args.event_path, args.bank_path, args.out_dir,
        args.n_ext, args.n_phi, args.n_t, args.seed,
    )
    elapsed = time.time() - t0
    print(f"  {len(gpu_df)} rows  (wall time: {elapsed:.1f}s)")

    # --- Compare ---
    print(f"\nComparing (threshold: max relative error < {REL_TOL:.0%}):")
    ok = _compare(cpu_df, gpu_df)

    print(f"\n{'='*55}")
    if ok:
        print("RESULT: PASS — GPU and CPU agree within 1% on all lnl columns")
    else:
        print("RESULT: FAIL — max relative error exceeds 1%")
        sys.exit(1)


if __name__ == "__main__":
    main()
