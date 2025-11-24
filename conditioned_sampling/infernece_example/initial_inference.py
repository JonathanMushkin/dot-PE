#!/usr/bin/env python3
"""
Run a simple waveform inference using a precomputed bank and an injected event.

Example: Initial inference run
    python run_inference.py \
        --bank-dir output/bank \
        --event-npz injection/toy_event.npz \
        --n-ext 2048 \
        --blocksize 2048 \
        --n-phi 100 \
        --n-phi-incoherent 100

"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dot_pe import inference
from cogwheel import data, utils


def resolve_path(path_str: str, base: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (base / path).resolve()


def run_inference(
    bank_dir: Path,
    event_npz: Path,
    output_dir: Path,
    n_ext: int,
    blocksize: int,
    n_phi: int,
    n_phi_incoherent: int,
    seed: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    event_data = data.EventData.from_npz(filename=event_npz)
    event_dir = output_dir / event_data.eventname
    event_dir.mkdir(parents=True, exist_ok=True)
    bank_dir = Path(bank_dir)

    bank = pd.read_feather(bank_dir / "intrinsic_sample_bank.feather")
    n_int = len(bank)

    print(f"Running inference for {event_data.eventname}")
    print(f"Using bank: {bank_dir} ({n_int} samples)")
    print(f"Output directory: {event_dir}")

    rundir = inference.run(
        event_dir=str(event_dir),
        event=event_data,
        bank_folder=str(bank_dir),
        n_int=n_int,
        n_ext=n_ext,
        n_phi=n_phi,
        n_phi_incoherent=n_phi_incoherent,
        n_t=n_phi,  # match n_phi for simplicity
        i_int_start=0,
        blocksize=min(n_int, blocksize),
        single_detector_blocksize=min(n_int, blocksize),
        seed=seed,
        size_limit=10**6,
        draw_subset=False,
        n_draws=None,
        preselected_indices=None,
    )

    print(f"Inference complete. Results stored in: {rundir}")

    summary_file = Path(rundir) / "summary_results.json"
    if summary_file.exists():
        summary = utils.read_json(summary_file)
        print("\nSummary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    return Path(rundir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dot-pe inference on a toy event.")
    parser.add_argument("--bank-dir", default="bank")
    parser.add_argument("--event-npz", default="injection/toy_event.npz")
    parser.add_argument("--output-dir", default="inference_runs")
    parser.add_argument("--n-ext", type=int, default=1024)
    parser.add_argument("--blocksize", type=int, default=1024)
    parser.add_argument("--n-phi", type=int, default=100)
    parser.add_argument("--n-phi-incoherent", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    bank_dir = resolve_path(args.bank_dir, base)
    event_npz = resolve_path(args.event_npz, base)
    output_dir = resolve_path(args.output_dir, base)

    run_inference(
        bank_dir=bank_dir,
        event_npz=event_npz,
        output_dir=output_dir,
        n_ext=args.n_ext,
        blocksize=args.blocksize,
        n_phi=args.n_phi,
        n_phi_incoherent=args.n_phi_incoherent,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
