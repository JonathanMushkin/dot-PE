#!/usr/bin/env python3
"""
Example script demonstrating conditional sampling from PowerLawIntrinsicIASPrior.

This script runs the examples from the conditional_sampler.py module comments.

Usage example:
python conditional_sampling_example.py
or
python conditional_sampling_example.py > conditional_sample_example.txt
"""

import sys
from pathlib import Path

# Add parent directory to path to import conditioned_sampling
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from conditioned_sampling import ConditionalPriorSampler
from cogwheel.gw_utils import m1m2_to_mchirp, chieff as gw_chieff

FIXED_STANDARD_PARAMS = {"mchirp", "lnq", "chieff", "f_ref"}


def compute_mchirp(m1: float, m2: float) -> float:
    return float(m1m2_to_mchirp(m1, m2))


def compute_lnq(m1: float, m2: float) -> float:
    return float(np.log(m2 / m1))


def compute_chieff(m1: float, m2: float, s1z: float, s2z: float) -> float:
    return float(gw_chieff(m1, m2, s1z, s2z))


def print_sampled_dataframe_stats(df: pd.DataFrame) -> None:
    print("\nSampled parameter statistics:")
    reported = False
    for col in df.columns:
        if col in FIXED_STANDARD_PARAMS:
            continue
        series = df[col]
        print(
            f"  {col}: mean={series.mean():.3f}, std={series.std():.3f}, "
            f"range=[{series.min():.3f}, {series.max():.3f}]"
        )
        reported = True
    if not reported:
        print("  (no sampled parameters to report)")


def print_sampled_array_stats(samples_dict) -> None:
    print("\nSampled parameter statistics:")
    reported = False
    for name, values in samples_dict.items():
        if name in FIXED_STANDARD_PARAMS:
            continue
        series = pd.Series(values)
        print(
            f"  {name}: mean={series.mean():.3f}, std={series.std():.3f}, "
            f"range=[{series.min():.3f}, {series.max():.3f}]"
        )
        reported = True
    if not reported:
        print("  (no sampled parameters to report)")


def example_1_basic_conditional_sampling():
    """Example 1: Basic conditional sampling."""
    print("=" * 70)
    print("Example 1: Basic conditional sampling")
    print("=" * 70)

    # Create sampler with prior configuration
    sampler = ConditionalPriorSampler(
        mchirp_range=(10, 20),
        q_min=0.3,
        f_ref=50,
        seed=42,
    )

    # Sample remaining parameters conditioned on fixed values
    # Convert from m1=15, m2=12, s1z=0.5, s2z=-0.3 to well-measured parameters
    m1_example = 15.0
    m2_example = 12.0
    s1z_example = 0.5
    s2z_example = -0.3

    mchirp_example = compute_mchirp(m1_example, m2_example)
    lnq_example = compute_lnq(m1_example, m2_example)
    chieff_example = compute_chieff(m1_example, m2_example, s1z_example, s2z_example)

    print("\nSampling 1000 samples with fixed parameters:")
    print(f"  mchirp = {mchirp_example:.3f} M_sun")
    print(f"  lnq = {lnq_example:.3f}")
    print(f"  chieff = {chieff_example:.3f}")
    print("  f_ref = 50 Hz")

    samples = sampler.sample(
        n_samples=1000,
        mchirp=mchirp_example,
        lnq=lnq_example,
        chieff=chieff_example,
        method="qmc",
    )

    print(f"\nGenerated {len(samples)} samples")
    print(f"\nColumns in output: {list(samples.columns)}")

    # Check that fixed parameters are constant
    print("\nVerifying fixed parameters are constant:")
    mchirp_derived = samples.apply(
        lambda row: compute_mchirp(row["m1"], row["m2"]), axis=1
    )
    lnq_derived = samples.apply(lambda row: compute_lnq(row["m1"], row["m2"]), axis=1)
    chieff_derived = samples.apply(
        lambda row: compute_chieff(row["m1"], row["m2"], row["s1z"], row["s2z"]), axis=1
    )
    print(
        f"  mchirp constant: {mchirp_derived.nunique() == 1} (value: {mchirp_derived.iloc[0]:.3f})"
    )
    print(
        f"  lnq constant: {lnq_derived.nunique() == 1} (value: {lnq_derived.iloc[0]:.3f})"
    )
    print(
        f"  chieff constant: {chieff_derived.nunique() == 1} (value: {chieff_derived.iloc[0]:.3f})"
    )
    print(
        f"  f_ref constant: {samples['f_ref'].nunique() == 1} (value: {samples['f_ref'].iloc[0]:.1f})"
    )

    print_sampled_dataframe_stats(samples)

    print("\n" + "=" * 70 + "\n")


def example_2_sample_array():
    """Example 2: Using sample_array for array output."""
    print("=" * 70)
    print("Example 2: Using sample_array for array output")
    print("=" * 70)

    sampler = ConditionalPriorSampler(
        mchirp_range=(10, 20),
        q_min=0.3,
        f_ref=50,
    )

    print("\nSampling 100 samples and returning as arrays...")

    # Convert from m1=15, m2=12, s1z=0.5, s2z=-0.3 to well-measured parameters
    m1_example = 15.0
    m2_example = 12.0
    s1z_example = 0.5
    s2z_example = -0.3

    mchirp_example = compute_mchirp(m1_example, m2_example)
    lnq_example = compute_lnq(m1_example, m2_example)
    chieff_example = compute_chieff(m1_example, m2_example, s1z_example, s2z_example)

    # Get samples as arrays
    samples_dict = sampler.sample_array(
        n_samples=100,
        mchirp=mchirp_example,
        lnq=lnq_example,
        chieff=chieff_example,
    )

    print(f"\nReturned dictionary with keys: {list(samples_dict.keys())}")
    first_key = next(iter(samples_dict))
    print(f"All arrays have shape: {samples_dict[first_key].shape}")
    print_sampled_array_stats(samples_dict)

    print("\n" + "=" * 70 + "\n")


def example_3_monte_carlo():
    """Example 3: Monte Carlo sampling."""
    print("=" * 70)
    print("Example 3: Monte Carlo sampling")
    print("=" * 70)

    sampler = ConditionalPriorSampler(
        mchirp_range=(10, 20),
        q_min=0.3,
        f_ref=50,
        seed=123,
    )

    print("\nUsing Monte Carlo method instead of QMC...")

    # Convert from m1=15, m2=12, s1z=0.5, s2z=-0.3 to well-measured parameters
    m1_example = 15.0
    m2_example = 12.0
    s1z_example = 0.5
    s2z_example = -0.3

    mchirp_example = compute_mchirp(m1_example, m2_example)
    lnq_example = compute_lnq(m1_example, m2_example)
    chieff_example = compute_chieff(m1_example, m2_example, s1z_example, s2z_example)

    # Use Monte Carlo instead of QMC
    samples = sampler.sample(
        n_samples=1000,
        mchirp=mchirp_example,
        lnq=lnq_example,
        chieff=chieff_example,
        method="mc",
    )

    print(f"Generated {len(samples)} samples using Monte Carlo")
    print_sampled_dataframe_stats(samples)

    print("\n" + "=" * 70 + "\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Conditional Sampling Examples")
    print("=" * 70)
    print("\nThis script demonstrates conditional sampling from")
    print("PowerLawIntrinsicIASPrior with fixed mchirp, lnq, chieff, f_ref.\n")

    try:
        example_1_basic_conditional_sampling()
        example_2_sample_array()
        example_3_monte_carlo()

        print("=" * 70)
        print("All examples completed successfully!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
