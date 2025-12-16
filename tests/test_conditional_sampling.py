#!/usr/bin/env python3
"""Test script for conditional sampling with both precessing and aligned spin modes.

Usage:
    Run this script directly to test the conditional sampling functionality:

    python tests/test_conditional_sampling.py

    Or make it executable and run:

    chmod +x tests/test_conditional_sampling.py
    ./tests/test_conditional_sampling.py

    The script will test:
    - Precessing mode (default aligned_spin=False)
    - Aligned spin mode (aligned_spin=True)
    - JSON serialization/deserialization
    - Vectorized sampling with multiple parameter sets
    - Zoomer integration with importance sampling weights

    All tests must pass for the implementation to be considered correct.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

sys.path.insert(0, str(Path(__file__).parent.parent))

from dot_pe.zoom.conditional_sampling import ConditionalPriorSampler
from dot_pe.zoom.zoom_iteration import draw_from_zoomer
from dot_pe.zoom.zoom import Zoomer
from cogwheel import gw_utils, gw_plotting
from cogwheel.gw_prior import IntrinsicIASPrior


def test_precessing_mode():
    """Test precessing mode (default)."""
    print("Testing precessing mode...")

    sampler = ConditionalPriorSampler(
        mchirp_range=(20.0, 40.0),
        q_min=0.3,
        f_ref=50.0,
        aligned_spin=False,
        seed=42,
    )

    # Sample with fixed mchirp, lnq, chieff
    mchirp = 30.0
    lnq = np.log(0.5)
    chieff = 0.2

    samples = sampler.sample(
        n_samples=10,
        mchirp=mchirp,
        lnq=lnq,
        chieff=chieff,
        method="mc",
    )

    print(f"  Generated {len(samples)} samples")
    print(f"  Columns: {list(samples.columns)}")

    # Compute mchirp, lnq, chieff from m1, m2, s1z, s2z to verify they are constant
    mchirp_computed = gw_utils.m1m2_to_mchirp(
        samples["m1"].values, samples["m2"].values
    )
    lnq_computed = np.log(samples["m2"].values / samples["m1"].values)
    chieff_computed = gw_utils.chieff(
        samples["m1"].values,
        samples["m2"].values,
        samples["s1z"].values,
        samples["s2z"].values,
    )

    # Check that fixed parameters are constant
    assert np.allclose(mchirp_computed, mchirp_computed[0]), "mchirp should be constant"
    assert np.allclose(lnq_computed, lnq_computed[0]), "lnq should be constant"
    assert np.allclose(chieff_computed, chieff_computed[0]), "chieff should be constant"

    # Check that in-plane spins vary
    assert samples["s1x_n"].nunique() > 1, "s1x_n should vary"
    assert samples["s1y_n"].nunique() > 1, "s1y_n should vary"

    # Check that iota varies
    assert samples["iota"].nunique() > 1, "iota should vary"

    print("  Precessing mode tests passed\n")


def test_aligned_spin_mode():
    """Test aligned spin mode."""
    print("Testing aligned spin mode...")

    sampler = ConditionalPriorSampler(
        mchirp_range=(20.0, 40.0),
        q_min=0.3,
        f_ref=50.0,
        aligned_spin=True,
        seed=42,
    )

    # Sample with fixed mchirp, lnq, chieff
    mchirp = 30.0
    lnq = np.log(0.5)
    chieff = 0.2

    samples = sampler.sample(
        n_samples=10,
        mchirp=mchirp,
        lnq=lnq,
        chieff=chieff,
        method="mc",
    )

    print(f"  Generated {len(samples)} samples")
    print(f"  Columns: {list(samples.columns)}")

    # Compute mchirp, lnq, chieff from m1, m2, s1z, s2z to verify they are constant
    mchirp_computed = gw_utils.m1m2_to_mchirp(
        samples["m1"].values, samples["m2"].values
    )
    lnq_computed = np.log(samples["m2"].values / samples["m1"].values)
    chieff_computed = gw_utils.chieff(
        samples["m1"].values,
        samples["m2"].values,
        samples["s1z"].values,
        samples["s2z"].values,
    )

    # Check that fixed parameters are constant
    assert np.allclose(mchirp_computed, mchirp_computed[0]), "mchirp should be constant"
    assert np.allclose(lnq_computed, lnq_computed[0]), "lnq should be constant"
    assert np.allclose(chieff_computed, chieff_computed[0]), "chieff should be constant"

    # Check that in-plane spins are zero
    assert np.allclose(samples["s1x_n"], 0.0), "s1x_n should be zero"
    assert np.allclose(samples["s1y_n"], 0.0), "s1y_n should be zero"
    assert np.allclose(samples["s2x_n"], 0.0), "s2x_n should be zero"
    assert np.allclose(samples["s2y_n"], 0.0), "s2y_n should be zero"

    # Check that iota varies
    assert samples["iota"].nunique() > 1, "iota should vary"

    print("  Aligned spin mode tests passed\n")


def test_json_serialization():
    """Test JSON serialization/deserialization."""
    print("Testing JSON serialization...")

    sampler1 = ConditionalPriorSampler(
        mchirp_range=(20.0, 40.0),
        q_min=0.3,
        f_ref=50.0,
        aligned_spin=True,
        seed=42,
    )

    test_file = Path("/tmp/test_conditional_sampler.json")
    sampler1.to_json(test_file)

    sampler2 = ConditionalPriorSampler.from_json(test_file)

    assert sampler2.mchirp_range == sampler1.mchirp_range
    assert sampler2.q_min == sampler1.q_min
    assert sampler2.f_ref == sampler1.f_ref
    assert sampler2.aligned_spin == sampler1.aligned_spin
    assert sampler2.seed == sampler1.seed

    test_file.unlink()

    print("  JSON serialization tests passed\n")


def test_vectorized_sampling():
    """Test vectorized sampling with multiple fixed parameter sets."""
    print("Testing vectorized sampling...")

    sampler = ConditionalPriorSampler(
        mchirp_range=(20.0, 40.0),
        q_min=0.3,
        f_ref=50.0,
        aligned_spin=False,
        seed=42,
    )

    n_samples = 10
    mchirp = np.linspace(25.0, 35.0, n_samples)
    lnq = np.full(n_samples, np.log(0.5))
    chieff = np.linspace(-0.5, 0.5, n_samples)

    samples = sampler.sample_vectorized(mchirp, lnq, chieff, method="mc")

    assert len(samples) == n_samples

    # Compute mchirp, lnq, chieff from m1, m2, s1z, s2z to verify they match input
    mchirp_computed = gw_utils.m1m2_to_mchirp(
        samples["m1"].values, samples["m2"].values
    )
    lnq_computed = np.log(samples["m2"].values / samples["m1"].values)
    chieff_computed = gw_utils.chieff(
        samples["m1"].values,
        samples["m2"].values,
        samples["s1z"].values,
        samples["s2z"].values,
    )

    assert np.allclose(mchirp_computed, mchirp)
    assert np.allclose(lnq_computed, lnq)
    assert np.allclose(chieff_computed, chieff)

    print(f"  Generated {len(samples)} samples with varying fixed parameters")
    print("  Vectorized sampling tests passed\n")


def test_zoomer_with_weights():
    """Test zoomer + conditional sampler integration with visualization."""
    print("Testing zoomer integration with weights...")

    # Create conditional sampler
    cond_sampler = ConditionalPriorSampler(
        mchirp_range=(25.0, 35.0),
        q_min=0.3,
        f_ref=50.0,
        aligned_spin=False,
        seed=42,
    )

    # Create zoomer with fixed mean and covariance
    mean = np.array([30.0, np.log(0.5), 0.5])

    # Define desired standard deviations for each dimension
    sigma = np.array([1.0, 0.1, 0.1])  # For mchirp, lnq, chieff

    # Start with diagonal matrix (identity)
    cov = np.eye(3)

    # Apply rotations using random angles (in cartesian coordinate style)
    rng = np.random.default_rng(42)
    # Two rotation angles for 3D rotations
    theta1 = rng.uniform(0, 2 * np.pi)  # Rotation in x-y plane
    theta2 = rng.uniform(0, 2 * np.pi)  # Rotation in y-z plane

    # Create rotation matrices
    R1 = np.array(
        [
            [np.cos(theta1), -np.sin(theta1), 0],
            [np.sin(theta1), np.cos(theta1), 0],
            [0, 0, 1],
        ]
    )
    R2 = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta2), -np.sin(theta2)],
            [0, np.sin(theta2), np.cos(theta2)],
        ]
    )

    # Apply rotations: R2 @ R1 @ cov @ R1.T @ R2.T
    cov = R2 @ R1 @ cov @ R1.T @ R2.T

    # Scale each element by sigma_i * sigma_j
    for i in range(3):
        for j in range(3):
            cov[i, j] = cov[i, j] * sigma[i] * sigma[j]

    # Create zoomer and manually set mean and covariance
    zoomer = Zoomer(engine_seed=42)
    zoomer.mean = mean
    zoomer.cov = cov
    # Create distribution object

    zoomer.distribution = multivariate_normal(
        mean=zoomer.mean, cov=zoomer.cov, allow_singular=True
    )

    # Set bounds
    bounds = {
        0: (25.0, 35.0),  # mchirp
        1: (np.log(0.3), 0.0),  # lnq
        2: (-1.0, 1.0),  # chieff
    }

    # Draw samples
    n_samples = 4000
    samples = draw_from_zoomer(
        zoomer=zoomer,
        cond_sampler=cond_sampler,
        bounds=bounds,
        n_samples=n_samples,
        seed=42,
    )

    print(f"  Generated {len(samples)} samples")
    print("  Weight statistics:")
    weights = np.exp(samples["log_prior_weights"].values)
    print(f"    Min weight: {weights.min():.6f}")
    print(f"    Max weight: {weights.max():.6f}")
    print(f"    Mean weight: {weights.mean():.6f}")
    print(f"    Median weight: {np.median(weights):.6f}")
    print(f"    Effective samples: {np.sum(weights) ** 2 / np.sum(weights**2):.1f}")

    # Add computed columns for plotting
    samples["weights"] = weights
    samples_unweighted = samples.copy()
    samples_unweighted["weights"] = 1.0 / len(samples_unweighted)

    # Generate samples directly from IntrinsicIASPrior for comparison
    intrinsic_prior = IntrinsicIASPrior(
        mchirp_range=cond_sampler.mchirp_range,
        q_min=cond_sampler.q_min,
        f_ref=cond_sampler.f_ref,
    )

    # Generate random samples from the prior
    prior_samples_df = intrinsic_prior.generate_random_samples(
        n_samples=n_samples, seed=42
    )

    # Transform to get standard parameters, then compute mchirp, lnq, chieff
    prior_samples_df["mchirp"] = gw_utils.m1m2_to_mchirp(
        prior_samples_df["m1"].values, prior_samples_df["m2"].values
    )
    prior_samples_df["lnq"] = np.log(
        prior_samples_df["m2"].values / prior_samples_df["m1"].values
    )
    prior_samples_df["chieff"] = gw_utils.chieff(
        prior_samples_df["m1"].values,
        prior_samples_df["m2"].values,
        prior_samples_df["s1z"].values,
        prior_samples_df["s2z"].values,
    )
    prior_samples_df["weights"] = 1.0 / len(prior_samples_df)

    # Create plots comparing unweighted, weighted, and direct prior samples
    try:
        mcp = gw_plotting.MultiCornerPlot(
            [samples_unweighted, samples, prior_samples_df],
            params=["mchirp", "lnq", "chieff", "m1", "m2"],
            weights_col="weights",
            labels=["Unweighted Proposal", "Weighted Proposal", "Direct Prior"],
            smooth=1,
        )
        mcp.plot()
        output_dir = Path(__file__).parent / "artifacts/cond_sampling/test_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_file = output_dir / "zoomer_test_corner.png"
        fig = plt.gcf()
        fig.savefig(plot_file, dpi=150, bbox_inches="tight")
        print(f"  Saved corner plot to {plot_file!s}")
    except Exception as e:
        print(f"  Warning: Could not create corner plot: {e}")

    print("  Zoomer integration test passed\n")


if __name__ == "__main__":
    print("Running conditional sampling tests...\n")

    try:
        test_precessing_mode()
        test_aligned_spin_mode()
        test_json_serialization()
        test_vectorized_sampling()
        test_zoomer_with_weights()

        print("All tests passed!")

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
