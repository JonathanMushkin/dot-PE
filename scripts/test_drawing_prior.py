#!/usr/bin/env python3
"""
Test script to compare samples from the new drawing prior class
with the original IntrinsicSamplesGenerator.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add the parent directory to the path to import dot_pe
sys.path.insert(0, str(Path(__file__).parent.parent))

from dot_pe import sample_banks_flexible
from dot_pe import sample_banks
from cogwheel import gw_plotting, gw_utils, utils


def test_drawing_prior():
    """Generate samples from both methods and compare them."""

    # Parameters for testing
    n_samples = 10000
    q_min = 1 / 6
    m_min = 50
    m_max = 100
    inc_faceon_factor = 1.0
    f_ref = 50.0
    seed = 42

    print("Generating samples from original IntrinsicSamplesGenerator...")

    # Generate samples using original method
    generator_original = sample_banks.IntrinsicSamplesGenerator()
    q_max = 1.0
    samples_original = (
        generator_original.draw_intrinsic_samples_uniform_in_lnmchrip_lnq(
            n_samples,
            q_min,
            q_max,
            m_min,
            m_max,
            inc_faceon_factor,
            f_ref,
            seed,
        )
    )

    # Remove weights for comparison (we want to compare the raw samples)
    samples_original_no_weights = samples_original.drop(columns=["log_prior_weights"])

    print("Generating samples from new drawing prior...")

    # Generate samples using new drawing prior
    drawing_prior = sample_banks_flexible.LogUniformMassBiasedInclinationDrawingPrior(
        m_min=m_min, m_max=m_max, q_min=q_min, inc_faceon_factor=inc_faceon_factor
    )

    # Generate random samples using the drawing prior
    u = np.random.RandomState(seed).rand(9, n_samples).T

    # Transform the random samples to get the actual parameter values
    samples_new_list = []
    for i in range(n_samples):
        # Use the same logic as the original draw_lnmchirp_lnq_uniform method
        logmchirp = np.log(m_min) + u[i, 0] * np.log(m_max / m_min)
        logq = np.log(q_min) + u[i, 1] * np.log(q_max / q_min)

        # Use the same biased inclination sampling as the original code
        inc_factor = (inc_faceon_factor - 1) / (inc_faceon_factor + 1)
        theta_jn_range = np.linspace(0, np.pi, 10**4)
        theta_jn_cdf = (
            theta_jn_range + inc_factor / 2 * np.sin(2 * theta_jn_range)
        ) / np.pi
        theta_jn = theta_jn_range[np.searchsorted(theta_jn_cdf, u[i, 4])]
        costheta_jn = np.cos(theta_jn)

        sample_dict = drawing_prior.transform(
            logmchirp=logmchirp,
            logq=logq,
            chieff=-1 + 2 * u[i, 2],
            cumchidiff=u[i, 3],
            costheta_jn=costheta_jn,
            phi_jl_hat=u[i, 5] * 2 * np.pi,
            phi12=u[i, 6] * 2 * np.pi,
            cums1r_s1z=u[i, 7],
            cums2r_s2z=u[i, 8],
            f_ref=f_ref,
        )
        samples_new_list.append(sample_dict)

    samples_new = pd.DataFrame(samples_new_list)

    print(f"Original samples shape: {samples_original_no_weights.shape}")
    print(f"New samples shape: {samples_new.shape}")

    # Add derived parameters for comparison
    samples_original_no_weights["mchirp"] = gw_utils.m1m2_to_mchirp(
        samples_original_no_weights["m1"], samples_original_no_weights["m2"]
    )
    samples_original_no_weights["chieff"] = gw_utils.chieff(
        *samples_original_no_weights[["m1", "m2", "s1z", "s2z"]].values.T
    )

    samples_new["mchirp"] = gw_utils.m1m2_to_mchirp(
        samples_new["m1"], samples_new["m2"]
    )
    samples_new["chieff"] = gw_utils.chieff(
        *samples_new[["m1", "m2", "s1z", "s2z"]].values.T
    )

    # Create corner plot comparison
    print("Creating corner plot comparison...")

    # Select parameters to plot
    plot_params = ["m1", "m2", "mchirp", "chieff", "s1z", "s2z", "iota"]

    # Create the corner plot
    mcp = gw_plotting.MultiCornerPlot(
        [samples_original_no_weights, samples_new],
        smooth=1,
        params=plot_params,
        labels=["Original", "New Drawing Prior"],
    )

    # Save the plot
    output_path = Path("test_drawing_prior_comparison.png")
    mcp.plot()
    fig = plt.gcf()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Corner plot saved to: {output_path}")

    # Print some statistics for comparison
    print("\nSample statistics comparison:")
    print("Original samples:")
    print(samples_original_no_weights[plot_params].describe())
    print("\nNew samples:")
    print(samples_new[plot_params].describe())

    return samples_original_no_weights, samples_new


if __name__ == "__main__":
    test_drawing_prior()
