#!/usr/bin/env python3
"""
Utility script to build a PowerLawIntrinsicIASPrior-based waveform bank.

# Run the bank creation script
python create_full_bank.py \
    --bank-dir bank \
    --bank-size 262144 \
    --n-pool 6 \
    --blocksize 4096 \
    --approximant IMRPhenomXPHM \
    --seed 5114
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, "../..")
from cogwheel import gw_utils
from dot_pe import waveform_banks, config
from dot_pe.power_law_mass_prior import PowerLawIntrinsicIASPrior

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")


def create_full_bank(
    bank_dir="output/bank",
    bank_size=2**18,
    mchirp_min=15.0,
    mchirp_max=65.0,
    q_min=0.2,
    f_ref=50.0,
    blocksize=4096,
    n_pool=7,
    approximant="IMRPhenomXODE",
    seed=5114,
):
    """
    Create complete waveform bank from PowerLawIntrinsicIASPrior.

    Parameters
    ----------
    bank_dir : str
        Directory to save the bank
    bank_size : int
        Number of samples (default: 2^18)
    mchirp_min, mchirp_max : float
        Chirp mass range
    q_min : float
        Minimum mass ratio
    f_ref : float
        Reference frequency
    blocksize : int
        Block size for waveform generation
    n_pool : int
        Number of parallel processes
    approximant : str
        Waveform approximant
    seed : int
        Random seed
    """

    bank_path = Path(bank_dir)
    if not bank_path.is_absolute():
        bank_path = Path(__file__).resolve().parent / bank_path
    bank_path.mkdir(parents=True, exist_ok=True)
    waveform_dir = bank_path / "waveforms"
    waveform_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("CREATING POWERLAW WAVEFORM BANK")
    print("=" * 80)
    print(f"Bank directory: {bank_dir}")
    print(f"Bank size: {bank_size:,}")
    print(f"Chirp mass range: [{mchirp_min}, {mchirp_max}]")
    print(f"q_min: {q_min}")
    print(f"f_ref: {f_ref}")
    print(f"Blocksize: {blocksize}")
    print(f"Processes: {n_pool}")
    print(f"Approximant: {approximant}")
    print()

    # Create bank directory
    bank_path = Path(bank_dir)
    bank_path.mkdir(parents=True, exist_ok=True)
    waveform_dir = bank_path / "waveforms"
    waveform_dir.mkdir(exist_ok=True)

    # ===================================================================
    # STEP 1: Generate samples from PowerLawIntrinsicIASPrior
    # ===================================================================
    print("=" * 80)
    print("STEP 1: Generating Prior Samples")
    print("=" * 80)

    print("Initializing PowerLawIntrinsicIASPrior...")
    powerlaw_prior = PowerLawIntrinsicIASPrior(
        mchirp_range=(mchirp_min, mchirp_max),
        q_min=q_min,
        f_ref=f_ref,
    )

    print(f"\nGenerating {bank_size:,} samples...")
    powerlaw_samples = powerlaw_prior.generate_random_samples(
        bank_size, seed=seed, return_lnz=False
    )

    print(f"Generated {len(powerlaw_samples):,} samples")

    # ===================================================================
    # STEP 2: Add derived quantities and calculate weights
    # ===================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Calculating Weights and Derived Quantities")
    print("=" * 80)

    powerlaw_samples["mchirp"] = gw_utils.m1m2_to_mchirp(
        powerlaw_samples["m1"], powerlaw_samples["m2"]
    )
    powerlaw_samples["lnq"] = np.log(powerlaw_samples["m2"] / powerlaw_samples["m1"])
    powerlaw_samples["mtot"] = powerlaw_samples["m2"] + powerlaw_samples["m1"]
    powerlaw_samples["chieff"] = gw_utils.chieff(
        *powerlaw_samples[["m1", "m2", "s1z", "s2z"]].values.T
    )

    print("Calculating importance weights (IAS/PowerLaw ratio)...")
    print(f"Computing for {len(powerlaw_samples):,} samples...")

    # Compute weight for the mass prior ratio only (other priors cancel)
    # Simplified: log(P_IAS) - log(P_PowerLaw) = log(mchirp) - (-0.7 * log(mchirp)) = 1.7 * log(mchirp)
    mchirp = powerlaw_samples["mchirp"].values
    powerlaw_samples["log_prior_weights"] = 1.7 * np.log(mchirp)
    powerlaw_samples["weights"] = np.exp(powerlaw_samples["log_prior_weights"])

    print(
        f"Weight range: [{powerlaw_samples['weights'].min():.3e}, {powerlaw_samples['weights'].max():.3e}]"
    )
    eff_n = (
        powerlaw_samples["weights"].sum() ** 2
        / (powerlaw_samples["weights"] ** 2).sum()
    )
    print(f"Effective sample size: {eff_n:.0f} ({eff_n / bank_size * 100:.1f}%)")

    # ===================================================================
    # STEP 3: Save samples and configuration
    # ===================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Saving Samples and Configuration")
    print("=" * 80)

    # Save samples to feather format (only necessary columns)
    # Columns needed for waveform generation
    columns_to_save = [
        "m1",
        "m2",
        "s1z",
        "s1x_n",
        "s1y_n",
        "s2z",
        "s2x_n",
        "s2y_n",
        "iota",
        "log_prior_weights",
    ]

    samples_path = bank_path / "intrinsic_sample_bank.feather"
    print(f"Saving samples to: {samples_path}")
    powerlaw_samples[columns_to_save].to_feather(samples_path)

    # Create bank configuration
    bank_config = {
        "bank_size": bank_size,
        "mchirp_min": mchirp_min,
        "mchirp_max": mchirp_max,
        "q_min": q_min,
        "f_ref": f_ref,
        "fbin": config.DEFAULT_FBIN.tolist(),
        "blocksize": blocksize,
        "prior_class": "PowerLawIntrinsicIASPrior",
        "prior_description": "P(M_c) ∝ M_c^{-1.7}, intrinsic parameters only",
        "importance_weighted_to": "IntrinsicIASPrior",
        "seed": seed,
    }

    bank_config_path = bank_path / "bank_config.json"
    print(f"Saving configuration to: {bank_config_path}")
    with open(bank_config_path, "w") as f:
        json.dump(bank_config, f, indent=4)

    print("\nSample statistics:")
    print(
        f"  m1: [{powerlaw_samples['m1'].min():.2f}, {powerlaw_samples['m1'].max():.2f}]"
    )
    print(
        f"  m2: [{powerlaw_samples['m2'].min():.2f}, {powerlaw_samples['m2'].max():.2f}]"
    )
    print(
        f"  mchirp: [{powerlaw_samples['mchirp'].min():.2f}, {powerlaw_samples['mchirp'].max():.2f}]"
    )
    print(
        f"  chieff: [{powerlaw_samples['chieff'].min():.3f}, {powerlaw_samples['chieff'].max():.3f}]"
    )

    # ===================================================================
    # STEP 4: Generate waveforms
    # ===================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Generating Waveforms")
    print("=" * 80)
    print(f"Waveform directory: {waveform_dir}")
    print(f"Approximant: {approximant}")
    print(f"Using {n_pool} parallel processes")
    print(f"Blocksize: {blocksize}")
    n_blocks = (bank_size + blocksize - 1) // blocksize
    print(f"Total blocks: {n_blocks}")
    print()
    print("This will take a while...")
    print()

    waveform_banks.create_waveform_bank_from_samples(
        samples_path=samples_path,
        bank_config_path=bank_config_path,
        waveform_dir=waveform_dir,
        n_pool=n_pool,
        blocksize=blocksize,
        approximant=approximant,
    )

    print("\n" + "=" * 80)
    print("BANK CREATION COMPLETE!")
    print("=" * 80)
    print(f"Bank directory: {bank_path.absolute()}")
    print(f"  - intrinsic_sample_bank.feather ({bank_size:,} samples)")
    print("  - bank_config.json")
    print(f"  - waveforms/ ({n_blocks} blocks)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Create complete chieff waveform bank")

    parser.add_argument(
        "--bank-dir",
        type=str,
        default="output/bank",
        help="Directory (relative to this folder) to save the bank.",
    )
    parser.add_argument(
        "--bank-size",
        type=int,
        default=2**18,
        help="Number of samples to draw (default: 2^18 = 262144)",
    )
    parser.add_argument(
        "--n-pool",
        type=int,
        default=7,
        help="Number of parallel processes (default: 7)",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=4096,
        help="Block size (default: 4096)",
    )
    parser.add_argument(
        "--approximant",
        type=str,
        default="IMRPhenomXODE",
        help="Waveform approximant (default: IMRPhenomXODE)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--mchirp-min",
        type=float,
        default=15.0,
        help="Minimum chirp mass in M_sun (default: 15.0)",
    )
    parser.add_argument(
        "--mchirp-max",
        type=float,
        default=65.0,
        help="Maximum chirp mass in M_sun (default: 65.0)",
    )
    parser.add_argument(
        "--q-min",
        type=float,
        default=0.2,
        help="Minimum mass ratio (default: 0.2)",
    )
    parser.add_argument(
        "--f-ref",
        type=float,
        default=50.0,
        help="Reference frequency in Hz (default: 50.0)",
    )

    args = parser.parse_args()

    try:
        create_full_bank(
            bank_dir=args.bank_dir,
            bank_size=args.bank_size,
            mchirp_min=args.mchirp_min,
            mchirp_max=args.mchirp_max,
            q_min=args.q_min,
            f_ref=args.f_ref,
            n_pool=args.n_pool,
            blocksize=args.blocksize,
            approximant=args.approximant,
            seed=args.seed,
        )
    except Exception as e:
        print(f"\nError creating bank: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
