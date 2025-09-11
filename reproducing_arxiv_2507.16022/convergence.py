#!/usr/bin/env python3
"""
Convergence testing script for reproducing arXiv:2507.16022 experiments.

This script performs convergence tests for gravitational wave parameter estimation
using the dot-PE method. It supports both reference runs (dense banks) and
convergence runs (regular banks with varying parameters).

Based on the paper requirements:
- Reference runs: Dense banks, single run per event with optimal parameters
- Convergence runs: Regular banks, 20 runs per event with varying intrinsic/extrinsic parameters
  - Either: vary intrinsic samples 2^6 to 2^16, fix extrinsic to 2^10
  - Or: fix intrinsic to 2^16, vary extrinsic 2 to 2^10
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import packages
from cogwheel import data, utils
from dot_pe import inference

# Event data configuration
EVENT_DATA_KWARGS = {
    "detector_names": "HLV",
    "duration": 120.0,
    "asd_funcs": ["asd_H_O3", "asd_L_O3", "asd_V_O3"],
    "tgps": 0.0,
    "fmax": 1600.0,
}

# Common fixed parameters for all runs
COMMON_FIXED_RUN_KWARGS = dict(
    n_t=128,
    n_phi=100,
    blocksize=1024,
    size_limit=10**7,
)

# Reference parameters for dense banks (optimal parameters)
REFERENCE_PARAMS = dict(
    n_int=2**16,  # 65536
    n_ext=2**10,  # 1024
)

# Convergence test configurations based on paper requirements
# Test 1: Vary intrinsic samples, fix extrinsic
INTRINSIC_CONVERGENCE = [
    dict(n_int=2**i, n_ext=2**10)
    for i in range(6, 17)  # 2^6 to 2^16
]

# Test 2: Fix intrinsic samples, vary extrinsic
EXTRINSIC_CONVERGENCE = [
    dict(n_int=2**16, n_ext=2**i)
    for i in range(1, 11)  # 2^1 to 2^10
]


class PathEncoder(json.JSONEncoder):
    """JSON encoder that handles Path objects."""

    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def str_to_int(s: str) -> int:
    """Convert string to integer deterministically."""
    return int.from_bytes(str(s).encode("utf-8"), "big") % (2**31)


def kwargs_to_seed(base: int = 0, kwargs: Dict = {}) -> int:
    """Generate deterministic seed from kwargs."""
    keys = list(kwargs.keys())
    values = list(kwargs.values())
    if kwargs:
        s = sum([str_to_int(str(k)) + str_to_int(str(v)) for k, v in zip(keys, values)])
    else:
        s = 0
    return (s + base) % (2**31)


def select_starting_point(rv: int, bank_size: int, n_int: int) -> int:
    """Select starting point in bank ensuring we don't exceed boundaries."""
    j = rv % bank_size
    if j + n_int >= bank_size:
        j = bank_size - n_int
    if j + n_int >= bank_size:
        j = 0
    return j


def event_name_from_idx(inj_file: Path, events_common_name: str, idx: int) -> str:
    """Generate event name from injection index."""
    inj_file = Path(inj_file)
    n_digits = len(str(len(pd.read_feather(inj_file))))
    event_name = "_".join([events_common_name, str(idx).zfill(n_digits)])
    return event_name


def check_run_exists(run_kwargs: Dict, event_dir: Path) -> bool:
    """Check if a run with these parameters already exists."""
    run_kwargs = run_kwargs.copy()

    # Standardize paths to strings for comparison
    run_kwargs_std = {}
    for k, v in run_kwargs.items():
        if isinstance(v, Path):
            run_kwargs_std[k] = str(v)
        else:
            run_kwargs_std[k] = v

    run_dirs = list(event_dir.glob("run_*"))
    for run_dir in run_dirs:
        kwargs_file = run_dir / "run_kwargs.json"
        if kwargs_file.exists():
            try:
                with open(kwargs_file, "r") as f:
                    old_kwargs = json.load(f)
                    if all(old_kwargs.get(k) == v for k, v in run_kwargs_std.items()):
                        return True
            except (json.JSONDecodeError, KeyError):
                continue
    return False


def run_inference(**run_kwargs) -> Path:
    """Run inference with given parameters and save results."""
    # Filter kwargs to only include valid parameters
    run_param_names = inference.run.__annotations__.keys()
    filtered_kwargs = {k: v for k, v in run_kwargs.items() if k in run_param_names}

    try:
        rundir = inference.run_and_profile(**filtered_kwargs)

        # Save run parameters
        with open(rundir / "run_kwargs.json", "w") as f:
            json.dump(run_kwargs, f, indent=4, cls=PathEncoder)

        logger.info(f"Completed run: {rundir}")
        return rundir
    except Exception as e:
        logger.error(
            f"Error during inference run with kwargs: {run_kwargs}", exc_info=True
        )
        raise e


def make_event_data(
    bank_folder: Path,
    injection_file: Path,
    inj_idx: int,
    approximant: str = "IMRPhenomXODE",
    event_data_kwargs: dict = EVENT_DATA_KWARGS,
    event_name: str = "",
) -> data.EventData:
    """Create event data with injected signal."""
    injections = pd.read_feather(injection_file)
    event_data = data.EventData.gaussian_noise(
        event_name,
        **event_data_kwargs,
        seed=str_to_int(event_name),
    )
    event_data.inject_signal(
        injections.iloc[inj_idx].to_dict(),
        approximant=approximant,
    )
    return event_data


def run_convergence_tests(
    bank_folder: Path,
    events_homedir: Path,
    inj_idx: int,
    run_type: str = "convergence",
    convergence_type: str = "intrinsic",
    repeats: int = 20,
    skip_existing: bool = True,
    base_seed: int = 0,
) -> List[Path]:
    """
    Run convergence tests for a single injection.

    Args:
        bank_folder: Path to bank folder
        events_homedir: Path to events directory
        inj_idx: Injection index
        run_type: "reference" or "convergence"
        convergence_type: "intrinsic" or "extrinsic" (only for convergence runs)
        repeats: Number of repeats (only for convergence runs)
        skip_existing: Skip existing runs
        base_seed: Base seed for reproducibility
    """
    # Setup event
    injection_file = events_homedir / "injections.feather"
    events_common_name = f"injection_{bank_folder.name}"
    event_name = event_name_from_idx(injection_file, events_common_name, inj_idx)
    event_dir = events_homedir / event_name

    if not event_dir.exists():
        utils.mkdirs(event_dir)

    # Create event data if needed
    event_data_path = event_dir / (event_name + ".npz")
    if not event_data_path.exists():
        event_data = make_event_data(
            bank_folder=bank_folder,
            injection_file=injection_file,
            inj_idx=inj_idx,
            event_name=event_name,
        )
        event_data.to_npz(filename=event_data_path)

    # Load bank config
    with open(bank_folder / "bank_config.json", "r") as f:
        bank_config = json.load(f)
    bank_size = bank_config["bank_size"]

    # Setup base kwargs
    fixed_kwargs = COMMON_FIXED_RUN_KWARGS.copy()
    fixed_kwargs.update(
        {
            "bank_folder": bank_folder,
            "event": event_data_path,
            "event_dir": event_dir,
        }
    )

    rundirs = []

    if run_type == "reference":
        # Single run with optimal parameters for dense banks
        run_kwargs = fixed_kwargs.copy()
        run_kwargs.update(REFERENCE_PARAMS)

        seed = kwargs_to_seed(base_seed, run_kwargs)
        i_int_start = select_starting_point(seed, bank_size, run_kwargs["n_int"])
        run_kwargs.update({"seed": seed, "i_int_start": i_int_start})

        if skip_existing and check_run_exists(run_kwargs, event_dir):
            logger.info(f"Skipping existing reference run for {event_name}")
        else:
            rundir = run_inference(**run_kwargs)
            rundirs.append(rundir)

    elif run_type == "convergence":
        # Multiple runs with varying parameters for regular banks
        if convergence_type == "intrinsic":
            param_list = INTRINSIC_CONVERGENCE
        elif convergence_type == "extrinsic":
            param_list = EXTRINSIC_CONVERGENCE
        else:
            raise ValueError(f"Unknown convergence_type: {convergence_type}")

        for params in param_list:
            for repeat in range(repeats):
                run_kwargs = fixed_kwargs.copy()
                run_kwargs.update(params)

                seed = kwargs_to_seed(base_seed + repeat, run_kwargs)
                i_int_start = select_starting_point(
                    seed, bank_size, run_kwargs["n_int"]
                )
                run_kwargs.update({"seed": seed, "i_int_start": i_int_start})

                if skip_existing and check_run_exists(run_kwargs, event_dir):
                    logger.info(f"Skipping existing run: {params}, repeat {repeat}")
                    continue

                rundir = run_inference(**run_kwargs)
                rundirs.append(rundir)

    logger.info(f"Completed {len(rundirs)} runs for {event_name}")
    return rundirs


def main():
    """Main function for running convergence experiments."""
    parser = argparse.ArgumentParser(
        description="Run convergence experiments for arXiv:2507.16022 reproduction"
    )
    parser.add_argument("bank_folder", type=Path, help="Path to bank folder")
    parser.add_argument("events_homedir", type=Path, help="Path to events directory")
    parser.add_argument("inj_idx", type=int, help="Injection index to process")
    parser.add_argument(
        "--run_type",
        choices=["reference", "convergence"],
        default="convergence",
        help="Type of run: reference (dense, single) or convergence (regular, multiple)",
    )
    parser.add_argument(
        "--convergence_type",
        choices=["intrinsic", "extrinsic"],
        default="intrinsic",
        help="Convergence test type: vary intrinsic or extrinsic parameters",
    )
    parser.add_argument(
        "--repeats", type=int, default=20, help="Number of repeats for convergence runs"
    )
    parser.add_argument(
        "--base_seed", type=int, default=0, help="Base seed for reproducibility"
    )
    parser.add_argument(
        "--no_skip", action="store_true", help="Don't skip existing runs"
    )

    args = parser.parse_args()

    logger.info(f"Starting {args.run_type} run for injection {args.inj_idx}")
    logger.info(f"Bank: {args.bank_folder}")
    logger.info(f"Events dir: {args.events_homedir}")

    if args.run_type == "convergence":
        logger.info(f"Convergence type: {args.convergence_type}")
        logger.info(f"Repeats: {args.repeats}")

    rundirs = run_convergence_tests(
        bank_folder=args.bank_folder,
        events_homedir=args.events_homedir,
        inj_idx=args.inj_idx,
        run_type=args.run_type,
        convergence_type=args.convergence_type,
        repeats=args.repeats,
        skip_existing=not args.no_skip,
        base_seed=args.base_seed,
    )

    logger.info(f"Completed all runs. Total: {len(rundirs)}")


if __name__ == "__main__":
    main()
