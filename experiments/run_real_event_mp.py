#!/usr/bin/env python3
"""
Submit / compare a real GW event PE run using the optimised MP pipeline.

Sub-commands
------------
submit (default)
    Read run_kwargs.json from a reference serial rundir and submit a new
    bsub job for run_mp.py with --n-workers / --n-ext-workers parallelism.

compare
    Compare wall time and ln_evidence of a completed run against the
    reference serial run.

Usage examples
--------------
    # Dry-run — print bsub script without submitting
    python experiments/run_real_event_mp.py \\
        --source-rundir .../pe_runs/GW230605_065343/run_0 \\
        --output-dir artifacts/pe_real_runs/GW230605_065343/mp_w8_ext4 \\
        --n-workers 8 --n-ext-workers 4 --dry-run

    # Submit
    python experiments/run_real_event_mp.py \\
        --source-rundir .../pe_runs/GW230605_065343/run_0 \\
        --output-dir artifacts/pe_real_runs/GW230605_065343/mp_w8_ext4 \\
        --n-workers 8 --n-ext-workers 4

    # Compare after the run completes
    python experiments/run_real_event_mp.py --compare \\
        --output-dir artifacts/pe_real_runs/GW230605_065343/mp_w8_ext4
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ── helpers ────────────────────────────────────────────────────────────────────

def _conda_header() -> str:
    return (
        'source "$(conda info --base)/etc/profile.d/conda.sh"\n'
        "conda activate dot-pe\n"
        f"export PYTHONPATH={ROOT}:$PYTHONPATH\n"
    )


def _parse_wall_s_from_profile(rundir: Path) -> float | None:
    """Extract total wall time from profile_output.txt (first line)."""
    profile = rundir / "profile_output.txt"
    if not profile.exists():
        return None
    first = profile.read_text().splitlines()[0]
    m = re.search(r"in\s+([\d.]+)\s+seconds", first)
    return float(m.group(1)) if m else None


# ── submit ─────────────────────────────────────────────────────────────────────

def _build_and_submit(args):
    source_rundir = Path(args.source_rundir).resolve()
    output_dir    = Path(args.output_dir).resolve()

    # 1. Load reference run_kwargs
    kw_path = source_rundir / "run_kwargs.json"
    if not kw_path.exists():
        print(f"ERROR: run_kwargs.json not found in {source_rundir}", file=sys.stderr)
        sys.exit(1)
    with open(kw_path) as f:
        kw = json.load(f)

    # 2. Extract parameters
    event_path  = Path(kw["event"]).resolve()
    # bank_folder can be a list (multi-bank) or a str
    bank_folders = kw["bank_folder"]
    if isinstance(bank_folders, list):
        if len(bank_folders) != 1:
            print(
                f"ERROR: multi-bank runs not supported ({len(bank_folders)} banks found)",
                file=sys.stderr,
            )
            sys.exit(1)
        bank_path = Path(bank_folders[0]).resolve()
    else:
        bank_path = Path(bank_folders).resolve()

    n_ext         = int(kw.get("n_ext", 2048))
    n_phi         = int(kw.get("n_phi", 100))
    n_t           = int(kw.get("n_t", 128))
    blocksize     = args.blocksize if args.blocksize is not None else int(kw.get("blocksize", 2048))
    sd_blocksize  = int(kw.get("single_detector_blocksize", 2048))
    seed          = kw.get("seed")
    mchirp_guess  = kw.get("mchirp_guess")
    max_inc_drop  = float(kw.get("max_incoherent_lnlike_drop", 20.0))
    max_bf_diff   = float(kw.get("max_bestfit_lnlike_diff", 20.0))

    # 3. Create output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. Write source_reference.json
    ref_summary_path = source_rundir / "summary_results.json"
    ref_summary = {}
    if ref_summary_path.exists():
        with open(ref_summary_path) as f:
            ref_summary = json.load(f)
    ref_wall_s = _parse_wall_s_from_profile(source_rundir)
    source_ref = {
        "source_rundir": str(source_rundir),
        "reference_ln_evidence": ref_summary.get("ln_evidence"),
        "reference_wall_s": ref_wall_s,
        "reference_n_effective": ref_summary.get("n_effective"),
    }
    ref_out = output_dir / "source_reference.json"
    with open(ref_out, "w") as f:
        json.dump(source_ref, f, indent=4)

    # 5. Build run_mp.py command
    n_workers     = args.n_workers
    n_ext_workers = args.n_ext_workers

    mp_cmd_parts = [
        f"python {ROOT}/MP/run_mp.py",
        f"    --event {event_path}",
        f"    --bank {bank_path}",
        f"    --rundir {output_dir}",
        f"    --n-ext {n_ext}",
        f"    --n-phi {n_phi}",
        f"    --n-t {n_t}",
        f"    --blocksize {blocksize}",
        f"    --single-detector-blocksize {sd_blocksize}",
        f"    --n-workers {n_workers}",
        f"    --n-ext-workers {n_ext_workers}",
        f"    --max-incoherent-lnlike-drop {max_inc_drop}",
        f"    --max-bestfit-lnlike-diff {max_bf_diff}",
    ]
    if seed is not None:
        mp_cmd_parts.append(f"    --seed {seed}")
    if mchirp_guess is not None:
        mp_cmd_parts.append(f"    --mchirp-guess {mchirp_guess}")

    mp_cmd = " \\\n".join(mp_cmd_parts)

    # 6. Extract event name for job naming
    event_name = event_path.stem  # e.g. GW230605_065343

    # 7. Build bsub script
    n_slots      = args.n_slots
    mem_per_slot = args.mem_per_slot_mb
    queue        = args.queue
    wall_limit   = 180  # minutes — 3 h ceiling (real event ≈ 1250 s expected)

    script = f"""\
#!/bin/bash
#BSUB -J {event_name}_mp_w{n_workers}
#BSUB -q {queue}
#BSUB -n {n_slots}
#BSUB -R "rusage[mem={mem_per_slot}] span[hosts=1]"
#BSUB -W {wall_limit}
#BSUB -o {output_dir}/lsf_%J.out
#BSUB -e {output_dir}/lsf_%J.err

{_conda_header()}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

{mp_cmd}
"""

    if args.dry_run:
        print("=== DRY RUN — bsub script ===")
        print(script)
        print(f"output_dir : {output_dir}")
        print(f"source_reference.json written to: {ref_out}")
        return

    # 8. Submit
    result = subprocess.run(
        ["bsub"],
        input=script,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("bsub FAILED:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    print(result.stdout.strip())
    m = re.search(r"Job <(\d+)>", result.stdout)
    job_id = m.group(1) if m else "unknown"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"submitted  : {ts}")
    print(f"job_id     : {job_id}")
    print(f"output_dir : {output_dir}")
    print(f"log        : {output_dir}/lsf_{job_id}.out")
    print(f"Monitor    : tail -f {output_dir}/lsf_{job_id}.out")


# ── compare ────────────────────────────────────────────────────────────────────

def _compare(args):
    output_dir = Path(args.output_dir).resolve()

    ref_path = output_dir / "source_reference.json"
    if not ref_path.exists():
        print(f"ERROR: {ref_path} not found — run submit first", file=sys.stderr)
        sys.exit(1)
    with open(ref_path) as f:
        ref = json.load(f)

    new_path = output_dir / "summary_results.json"
    if not new_path.exists():
        print(f"ERROR: {new_path} not found — run has not completed yet", file=sys.stderr)
        sys.exit(1)
    with open(new_path) as f:
        new = json.load(f)

    # Parse wall time from LSF log (look for run_mp.py timing banner)
    new_wall_s = None
    lsf_logs = sorted(output_dir.glob("lsf_*.out"))
    for log in lsf_logs:
        text = log.read_text()
        # run_mp.py prints "Total wall time: X.Xs"
        m = re.search(r"Total wall time:\s*([\d.]+)\s*s", text)
        if m:
            new_wall_s = float(m.group(1))
            break

    ref_wall     = ref.get("reference_wall_s")
    ref_ln_ev    = ref.get("reference_ln_evidence")
    ref_n_eff    = ref.get("reference_n_effective")
    new_ln_ev    = new.get("ln_evidence")
    new_n_eff    = new.get("n_effective")

    speedup = (ref_wall / new_wall_s) if (ref_wall and new_wall_s) else None
    ln_diff = (new_ln_ev - ref_ln_ev) if (new_ln_ev is not None and ref_ln_ev is not None) else None

    print("\n=== Comparison: reference serial vs new MP ===")
    print(f"  Reference rundir : {ref.get('source_rundir')}")
    print(f"  New rundir       : {output_dir}")
    print()
    header = f"{'Metric':<28} {'Reference':>14} {'New MP':>14}"
    print(header)
    print("-" * len(header))

    def _row(label, ref_val, new_val, fmt="{:.3f}"):
        rv = fmt.format(ref_val) if ref_val is not None else "—"
        nv = fmt.format(new_val) if new_val is not None else "—"
        print(f"  {label:<26} {rv:>14} {nv:>14}")

    _row("wall_s", ref_wall, new_wall_s, "{:.1f}")
    if speedup is not None:
        print(f"  {'speedup':<26} {'':>14} {speedup:>13.1f}×")
    _row("ln_evidence", ref_ln_ev, new_ln_ev, "{:.5f}")
    if ln_diff is not None:
        print(f"  {'Δln_evidence':<26} {'':>14} {ln_diff:>+14.5f}")
    _row("n_effective", ref_n_eff, new_n_eff, "{:.0f}")

    print()
    if ln_diff is not None and abs(ln_diff) > 0.01:
        print("WARNING: |Δln_evidence| > 0.01 — check seed / code changes")
    else:
        print("ln_evidence matches within tolerance (±0.01) ✓")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--compare", action="store_true",
                   help="Compare completed run against reference (no submission)")
    p.add_argument("--source-rundir", default=None,
                   help="Path to reference rundir containing run_kwargs.json")
    p.add_argument("--output-dir", required=True,
                   help="Directory for new run outputs (created if absent)")
    p.add_argument("--n-workers", type=int, default=8,
                   help="Coherent+incoherent MP workers (default: 8)")
    p.add_argument("--n-ext-workers", type=int, default=1,
                   help="Parallel extrinsic MI workers (default: 1 = serial)")
    p.add_argument("--queue", default="physics-short",
                   help="LSF queue (default: physics-short)")
    p.add_argument("--mem-per-slot-mb", type=int, default=10000,
                   help="Memory per LSF slot in MB (default: 10000)")
    p.add_argument("--n-slots", type=int, default=9,
                   help="Number of LSF slots (default: 9)")
    p.add_argument("--blocksize", type=int, default=None,
                   help="Intrinsic block size for coherent workers "
                        "(default: read from run_kwargs.json). "
                        "The reference serial run uses 2048 (gives dh_ieo ≈ 6.7 GB/worker); "
                        "use 512 for parallel runs to keep per-worker dh_ieo ≈ 1.7 GB.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print bsub script but do not submit")
    args = p.parse_args()

    if args.compare:
        _compare(args)
    else:
        if args.source_rundir is None:
            p.error("--source-rundir is required for submission")
        _build_and_submit(args)


if __name__ == "__main__":
    main()
