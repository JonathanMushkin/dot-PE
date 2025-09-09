#!/usr/bin/env python3
"""
Script to generate injections for all mass ranges used in arXiv:2507.16022 experiments.

This script generates injections for each mass range (mchirp 3,5,10,20,50,100).
The injections are independent of bank density - the same injections will be analyzed
by both regular and dense banks for each mass range.

Based on the original generate_injections.py from the experiments.
"""

import numpy as np
from pathlib import Path
import sys
import pandas as pd
import json
from tqdm import tqdm
import logging
import argparse
from typing import Dict, Tuple

# Add cogwheel to path
cogwheel_path = "/home/projects/barakz/jonatahm/GW/cogwheel-private"
sys.path.insert(0, cogwheel_path)

from cogwheel import data
from cogwheel.likelihood import CBCLikelihood
from cogwheel.waveform import WaveformGenerator
from cogwheel.validation import generate_injections
from cogwheel.sampler_free import sample_generation

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants from original script
EVENT_DATA_KWARGS = {
    "detector_names": "HLV",
    "duration": 120.0,
    "asd_funcs": ["asd_H_O3", "asd_L_O3", "asd_V_O3"],
    "tgps": 0.0,
    "fmax": 1600.0,
}
D_LUMINOSITY_MAX = 15e3
H_H_MIN = 70
H_H_MAX = 200

# Default parameters
DEFAULT_N_INJECTIONS = 1024
DEFAULT_BATCH_SIZE = 10240  # 10 * N_INJECTIONS from original script
DEFAULT_N_CORES = 32


def create_a_batch_of_samples(
    q_min: float,
    min_mchirp: float,
    max_mchirp: float,
    f_ref: float,
    n_samples: int,
) -> pd.DataFrame:
    """Create a batch of samples with specified parameters."""
    isg = sample_generation.IntrinsicSamplesGenerator()
    int_samples = isg.draw_physical_prior_samples(
        q_min,
        min_mchirp,
        max_mchirp,
        f_ref,
        n_samples=n_samples,
        draw_method="mc",
    )
    int_samples["l1"] = 0
    int_samples["l2"] = 0
    int_samples["f_ref"] = f_ref

    ra = np.random.uniform(0, 1, n_samples) * np.pi * 2
    dec = np.arcsin(np.random.uniform(-1, 1, n_samples))
    psi = np.random.uniform(0, 1, n_samples) * np.pi
    dimensionless_volume = np.random.uniform(0, 1, n_samples)
    dimensionless_distance = dimensionless_volume ** (1 / 3)
    phi_ref = np.random.uniform(0, np.pi * 2, n_samples)
    t_geocenter = np.zeros(n_samples)

    batch = pd.concat(
        [
            int_samples,
            pd.DataFrame(
                dict(
                    ra=ra,
                    dec=dec,
                    psi=psi,
                    dimensionless_distance=dimensionless_distance,
                    phi_ref=phi_ref,
                    t_geocenter=t_geocenter,
                )
            ),
        ],
        axis=1,
    )
    return batch


def _batch_of_injections_in_hh_range(
    q_min: float,
    min_mchirp: float,
    max_mchirp: float,
    f_ref: float,
    batch_size: int,
    cbc_likelihood: CBCLikelihood,
    n_cores: int = 1,
    d_luminosity_max: float = D_LUMINOSITY_MAX,
    h_h_min: float = H_H_MIN,
    h_h_max: float = H_H_MAX,
) -> pd.DataFrame:
    """Generate a batch of injections within a specified h_h range."""
    batch = create_a_batch_of_samples(
        q_min, min_mchirp, max_mchirp, f_ref, n_samples=batch_size
    )

    h_h_1mpc = generate_injections._compute_h_h_1mpc(batch, cbc_likelihood, n_cores)
    d_ref = np.sqrt(np.max(h_h_1mpc) / h_h_min)

    batch["d_luminosity"] = batch["dimensionless_distance"] * d_ref
    del batch["dimensionless_distance"]

    batch["h_h"] = h_h_1mpc / batch["d_luminosity"] ** 2
    injections_in_range = batch[
        (batch["h_h"] > h_h_min)
        & (batch["d_luminosity"] < d_luminosity_max)
        & (batch["h_h"] < h_h_max)
    ].reset_index(drop=True)

    logger.info(f"Accepted {len(injections_in_range)} samples out of {batch_size}")
    return injections_in_range


def generate_injections_in_hh_range(
    q_min: float,
    min_mchirp: float,
    max_mchirp: float,
    f_ref: float,
    n_injections: int,
    batch_size: int,
    n_cores: int,
    approximant: str,
    event_data_kwargs: dict = EVENT_DATA_KWARGS,
    d_luminosity_max: float = D_LUMINOSITY_MAX,
    h_h_min: float = H_H_MIN,
    h_h_max: float = H_H_MAX,
) -> pd.DataFrame:
    """Generate injections within a specified h_h range until the desired number is reached."""
    event_data = data.EventData.gaussian_noise("", **event_data_kwargs)
    waveform_generator = WaveformGenerator.from_event_data(event_data, approximant)
    cbc_likelihood = CBCLikelihood(event_data, waveform_generator)
    cbc_likelihood.asd_drift = None

    iteration = 0
    injs_above_threshold = pd.DataFrame()

    with tqdm(total=n_injections, desc=f"mchirp {min_mchirp}-{max_mchirp}") as pbar:
        while len(injs_above_threshold) < n_injections:
            iteration += 1
            batch = _batch_of_injections_in_hh_range(
                q_min,
                min_mchirp,
                max_mchirp,
                f_ref,
                batch_size,
                cbc_likelihood,
                n_cores,
                d_luminosity_max=d_luminosity_max,
                h_h_min=h_h_min,
                h_h_max=h_h_max,
            )
            length_added = len(batch)
            injs_above_threshold = pd.concat(
                (injs_above_threshold, batch), ignore_index=True
            )
            pbar.update(length_added)
            current_length = min(n_injections, len(injs_above_threshold))
            pbar.set_description(
                f"mchirp {min_mchirp}-{max_mchirp} | Iteration: {iteration}, Injections: {current_length}"
            )

    return injs_above_threshold[:n_injections]


def generate_injections_from_bank_config(
    bank_folder: Path,
    n_injections: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_cores: int = DEFAULT_N_CORES,
    event_data_kwargs: dict = EVENT_DATA_KWARGS,
    d_luminosity_max: float = D_LUMINOSITY_MAX,
    h_h_min: float = H_H_MIN,
    h_h_max: float = H_H_MAX,
    save_path: Path = None,
) -> pd.DataFrame:
    """Generate injections from a bank configuration and optionally save them."""
    bank_folder = Path(bank_folder)

    # Read bank configuration
    with open(bank_folder / "bank_config.json", "r") as fp:
        bank_config: dict = json.load(fp)

    q_min: float = bank_config["q_min"]
    min_mchirp: float = bank_config["min_mchirp"]
    max_mchirp: float = bank_config["max_mchirp"]
    f_ref: float = bank_config["f_ref"]
    approximant: str = bank_config["approximant"]

    logger.info(
        f"Generating {n_injections} injections for mass range {min_mchirp}-{max_mchirp}"
    )

    injections: pd.DataFrame = generate_injections_in_hh_range(
        q_min=q_min,
        min_mchirp=min_mchirp,
        max_mchirp=max_mchirp,
        f_ref=f_ref,
        n_injections=n_injections,
        batch_size=batch_size,
        n_cores=n_cores,
        approximant=approximant,
        event_data_kwargs=event_data_kwargs,
        d_luminosity_max=d_luminosity_max,
        h_h_min=h_h_min,
        h_h_max=h_h_max,
    )

    if save_path:
        save_path = Path(save_path)
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        injections.to_feather(save_path)
        logger.info(f"Injections saved to {save_path}")

    return injections


def main():
    """Generate injections for all mass ranges used in the paper."""
    parser = argparse.ArgumentParser(
        description="Generate injections for arXiv:2507.16022 reproduction"
    )
    parser.add_argument(
        "--n_injections",
        type=int,
        default=DEFAULT_N_INJECTIONS,
        help=f"Number of injections per mass range (default: {DEFAULT_N_INJECTIONS})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for injection generation (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--n_cores",
        type=int,
        default=DEFAULT_N_CORES,
        help=f"Number of cores to use (default: {DEFAULT_N_CORES})",
    )
    parser.add_argument(
        "--banks_dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing the bank folders (default: current directory)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to save injection files (default: current directory)",
    )
    parser.add_argument(
        "--mass_ranges",
        nargs="+",
        type=int,
        default=[3, 5, 10, 20, 50, 100],
        help="Mass ranges to generate injections for (default: 3 5 10 20 50 100)",
    )

    args = parser.parse_args()

    # Mass range configurations (same as in create_banks.py)
    bank_configs = {
        3: (3.0, 3.2),
        5: (5.0, 6.0),
        10: (10.0, 12.0),
        20: (20.0, 30.0),
        50: (50.0, 60.0),
        100: (100.0, 300.0),
    }

    logger.info(f"Starting injection generation for mass ranges: {args.mass_ranges}")
    logger.info(f"Banks directory: {args.banks_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Injections per mass range: {args.n_injections}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Cores: {args.n_cores}")

    for mchirp_val in args.mass_ranges:
        if mchirp_val not in bank_configs:
            logger.warning(f"Unknown mass range {mchirp_val}, skipping")
            continue

        logger.info(f"Processing mass range {mchirp_val}")

        # Use the regular bank (not dense) as the reference for configuration
        bank_folder = args.banks_dir / f"bank_mchirp_{mchirp_val}"

        if not bank_folder.exists():
            logger.error(f"Bank folder not found: {bank_folder}")
            continue

        if not (bank_folder / "bank_config.json").exists():
            logger.error(f"Bank config not found: {bank_folder / 'bank_config.json'}")
            continue

        # Create output directory for this mass range
        output_subdir = args.output_dir / f"bank_mchirp_{mchirp_val}"
        save_path = output_subdir / "injections.feather"

        try:
            generate_injections_from_bank_config(
                bank_folder=bank_folder,
                n_injections=args.n_injections,
                batch_size=args.batch_size,
                n_cores=args.n_cores,
                save_path=save_path,
            )
            logger.info(
                f"Successfully generated injections for mass range {mchirp_val}"
            )
        except Exception as e:
            logger.error(
                f"Failed to generate injections for mass range {mchirp_val}: {e}"
            )
            continue

    logger.info("Injection generation completed!")


if __name__ == "__main__":
    main()
