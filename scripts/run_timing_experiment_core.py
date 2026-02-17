#!/usr/bin/env python3
"""
Script to run a single inference timing experiment with a specified number of cores.
Called by run_timing_experiment.sh and run_timing_experiment_lsf.py
"""

import argparse
import os
import sys
from pathlib import Path
import warnings

# Suppress SWIG LAL warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

# Add dot-pe to path if needed (adjust based on your setup)
# dot_pe_path = "/path/to/dot-pe"
# sys.path.append(dot_pe_path)

from cogwheel.data import EventData
from dot_pe import inference, utils


def run_timing_experiment(
    event_name: str,
    bank_names: list,
    n_cores: int,
    data_dir: str = "data/artifacts",
    output_base_dir: str = "pe/artifacts/timing_experiments",
    n_ext: int = 2**11,
    n_phi: int = 100,
    n_t: int = 128,
    blocksize: int = 2**12,
    single_detector_blocksize: int = 2**12,
    seed: int = None,
    size_limit: int = 10**7,
    draw_subset: bool = True,
    n_draws: int = None,
    mchirp_guess: float = None,
    max_incoherent_lnlike_drop: float = 20.0,
    max_bestfit_lnlike_diff: float = 20.0,
    load_inds: bool = False,
    n_int: int = None,
    i_int_start: int = 0,
):
    """
    Run a single inference timing experiment with specified core count.

    Parameters
    ----------
    event_name : str
        Name of the event (used to load data/artifacts/{event_name}.npz)
    bank_names : list
        List of bank names or single bank name
    n_cores : int
        Number of cores to use (for logging purposes)
    data_dir : str
        Base directory for event data (default: data/artifacts)
    output_base_dir : str
        Base directory for output (default: pe/artifacts/timing_experiments)
    n_ext : int
        Number of extrinsic samples
    n_phi : int
        Number of phi_ref samples for coherent evaluation
    n_t : int
        Number of time samples in single-detector time-grid
    blocksize : int
        Block size for coherent likelihood evaluation
    single_detector_blocksize : int
        Block size for single detector likelihood evaluation
    seed : int
        Random seed
    size_limit : int
        Size limit for processing
    draw_subset : bool
        Whether to draw subset
    n_draws : int
        Number of draws
    mchirp_guess : float
        Chirp mass guess
    max_incoherent_lnlike_drop : float
        Maximum incoherent log-likelihood drop
    max_bestfit_lnlike_diff : float
        Maximum bestfit log-likelihood difference
    load_inds : bool
        Whether to load indices
    n_int : int
        Number of intrinsic samples (None = use full bank)
    i_int_start : int
        Starting index for intrinsic samples
    """
    # Load event data
    event_data_path = Path(data_dir) / f"{event_name}.npz"
    if not event_data_path.exists():
        raise FileNotFoundError(f"Event data not found: {event_data_path}")

    print(f"Loading event data from: {event_data_path}")
    event_data = EventData.from_npz(filename=str(event_data_path))

    # Prepare bank folders
    if isinstance(bank_names, str):
        bank_names = [bank_names]

    bank_folders = [Path(data_dir) / bank_name for bank_name in bank_names]

    # Validate bank folders exist
    for bank_folder in bank_folders:
        if not bank_folder.exists():
            raise FileNotFoundError(f"Bank folder not found: {bank_folder}")

    # If single bank, use Path directly; if multiple, use list
    if len(bank_folders) == 1:
        bank_folder = bank_folders[0]
    else:
        bank_folder = bank_folders

    # Determine n_int if not specified
    if n_int is None:
        if isinstance(bank_folder, Path):
            bank_config_path = bank_folder / "bank_config.json"
            if bank_config_path.exists():
                bank_config = utils.read_json(bank_config_path)
                n_int = bank_config.get("bank_size", None)
        if n_int is None:
            # Try to read from feather file
            if isinstance(bank_folder, Path):
                bank_df_path = bank_folder / "intrinsic_sample_bank.feather"
                if bank_df_path.exists():
                    import pandas as pd

                    bank_df = pd.read_feather(bank_df_path)
                    n_int = len(bank_df)

    # Set up output directory
    output_dir = Path(output_base_dir) / event_name / f"cores_{n_cores}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Running timing experiment:")
    print(f"  Event: {event_name}")
    print(f"  Banks: {bank_names}")
    print(f"  Cores: {n_cores}")
    print(f"  Output: {output_dir}")
    print(f"  n_ext: {n_ext}, n_phi: {n_phi}, n_t: {n_t}")
    print(
        f"  blocksize: {blocksize}, single_detector_blocksize: {single_detector_blocksize}"
    )
    if n_int is not None:
        print(f"  n_int: {n_int}")
    print(f"{'=' * 60}\n")

    # Run inference with profiling
    try:
        final_rundir = inference.run_and_profile(
            event=event_data,
            bank_folder=bank_folder,
            n_int=n_int,
            n_ext=n_ext,
            n_phi=n_phi,
            n_t=n_t,
            blocksize=blocksize,
            single_detector_blocksize=single_detector_blocksize,
            i_int_start=i_int_start,
            seed=seed,
            load_inds=load_inds,
            size_limit=size_limit,
            draw_subset=draw_subset,
            n_draws=n_draws,
            rundir=str(output_dir),
            mchirp_guess=mchirp_guess,
            max_incoherent_lnlike_drop=max_incoherent_lnlike_drop,
            max_bestfit_lnlike_diff=max_bestfit_lnlike_diff,
        )

        print(f"\n{'=' * 60}")
        print(f"Timing experiment completed successfully!")
        print(f"Results saved to: {final_rundir}")
        print(f"{'=' * 60}\n")

        return final_rundir

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"Error running timing experiment: {e}")
        print(f"{'=' * 60}\n")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run a single inference timing experiment with specified core count"
    )

    # Required arguments
    parser.add_argument(
        "--event-name",
        type=str,
        required=True,
        help="Name of the event (used to load data/artifacts/{event_name}.npz)",
    )
    parser.add_argument(
        "--bank-names",
        type=str,
        nargs="+",
        required=True,
        help="Bank name(s) - can specify multiple banks",
    )
    parser.add_argument(
        "--cores",
        type=int,
        required=True,
        help="Number of cores (for logging/output directory naming)",
    )

    # Path arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/artifacts",
        help="Base directory for event data (default: data/artifacts)",
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default="pe/artifacts/timing_experiments",
        help="Base directory for output (default: pe/artifacts/timing_experiments)",
    )

    # Inference parameters
    parser.add_argument(
        "--n-ext",
        type=int,
        default=2**11,
        help="Number of extrinsic samples (default: 2048)",
    )
    parser.add_argument(
        "--n-phi",
        type=int,
        default=100,
        help="Number of phi_ref samples for coherent evaluation (default: 100)",
    )
    parser.add_argument(
        "--n-t",
        type=int,
        default=128,
        help="Number of time samples in single-detector time-grid (default: 128)",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=2**12,
        help="Block size for coherent likelihood evaluation (default: 4096)",
    )
    parser.add_argument(
        "--single-detector-blocksize",
        type=int,
        default=2**12,
        help="Block size for single detector likelihood evaluation (default: 4096)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--size-limit",
        type=int,
        default=10**7,
        help="Size limit for processing (default: 10000000)",
    )
    parser.add_argument(
        "--no-draw-subset",
        action="store_false",
        dest="draw_subset",
        help="Disable drawing subset",
    )
    parser.add_argument("--n-draws", type=int, default=None, help="Number of draws")
    parser.add_argument(
        "--mchirp-guess", type=float, default=None, help="Chirp mass guess"
    )
    parser.add_argument(
        "--max-incoherent-lnlike-drop",
        type=float,
        default=20.0,
        help="Maximum incoherent log-likelihood drop (default: 20.0)",
    )
    parser.add_argument(
        "--max-bestfit-lnlike-diff",
        type=float,
        default=20.0,
        help="Maximum bestfit log-likelihood difference (default: 20.0)",
    )
    parser.add_argument(
        "--load-inds", action="store_true", help="Load indices from file"
    )
    parser.add_argument(
        "--n-int",
        type=int,
        default=None,
        help="Number of intrinsic samples (None = use full bank)",
    )
    parser.add_argument(
        "--i-int-start",
        type=int,
        default=0,
        help="Starting index for intrinsic samples (default: 0)",
    )

    args = parser.parse_args()

    try:
        run_timing_experiment(
            event_name=args.event_name,
            bank_names=args.bank_names,
            n_cores=args.cores,
            data_dir=args.data_dir,
            output_base_dir=args.output_base_dir,
            n_ext=args.n_ext,
            n_phi=args.n_phi,
            n_t=args.n_t,
            blocksize=args.blocksize,
            single_detector_blocksize=args.single_detector_blocksize,
            seed=args.seed,
            size_limit=args.size_limit,
            draw_subset=args.draw_subset,
            n_draws=args.n_draws,
            mchirp_guess=args.mchirp_guess,
            max_incoherent_lnlike_drop=args.max_incoherent_lnlike_drop,
            max_bestfit_lnlike_diff=args.max_bestfit_lnlike_diff,
            load_inds=args.load_inds,
            n_int=args.n_int,
            i_int_start=args.i_int_start,
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
