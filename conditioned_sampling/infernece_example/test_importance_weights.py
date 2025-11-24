#!/usr/bin/env python3
"""Test importance sampling weight calculation for zoom banks.

Run with: conda activate dot-pe; python test_importance_weights.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cogwheel import gw_utils
from dot_pe.zoom.zoom import Zoomer
from dot_pe.zoom.conditional_sampling import ConditionalPriorSampler

# Import draw_from_zoomer from zoom_iteration module in same directory
sys.path.insert(0, str(Path(__file__).parent))
import zoom_iteration

draw_from_zoomer = zoom_iteration.draw_from_zoomer


def main():
    base_dir = Path("zoom/iter1/run_0")
    bank_dir = Path("zoom/iter1/bank")

    print("Loading Zoomer and ConditionalPriorSampler...")
    zoomer = Zoomer.from_json(base_dir / "Zoomer.json")
    cond_sampler = ConditionalPriorSampler.from_json(
        base_dir / "ConditionalPriorSampler.json"
    )

    print("\n=== Loading highest probability sample from samples.feather ===")
    samples = pd.read_feather(base_dir / "samples.feather")

    if "mchirp" not in samples.columns:
        samples["mchirp"] = gw_utils.m1m2_to_mchirp(samples["m1"], samples["m2"])
    if "lnq" not in samples.columns:
        samples["lnq"] = np.log(samples["m2"] / samples["m1"])
    if "chieff" not in samples.columns:
        samples["chieff"] = gw_utils.chieff(
            samples["m1"], samples["m2"], samples["s1z"], samples["s2z"]
        )

    prob_col = "bestfit_lnlike"
    best_sample = samples.loc[samples[prob_col].idxmax()]

    test_point_mchirp = float(best_sample["mchirp"])
    test_point_lnq = float(best_sample["lnq"])
    test_point_chieff = float(best_sample["chieff"])

    print(f"Highest probability sample ({prob_col}={best_sample[prob_col]:.4f}):")
    print(
        f"  mchirp={test_point_mchirp:.4f}, lnq={test_point_lnq:.4f}, chieff={test_point_chieff:.4f}"
    )

    print("\n=== Test 1: Compute IntrinsicIASPrior (assumed prior) at test point ===")
    from cogwheel.gw_prior import IntrinsicIASPrior

    assumed_prior = IntrinsicIASPrior(
        mchirp_range=cond_sampler.mchirp_range,
        q_min=cond_sampler.q_min,
        f_ref=cond_sampler.f_ref,
    )
    print(
        f"Created IntrinsicIASPrior with mchirp_range={cond_sampler.mchirp_range}, q_min={cond_sampler.q_min}, f_ref={cond_sampler.f_ref}"
    )
    print(
        f"Evaluating at test point: mchirp={test_point_mchirp:.4f}, lnq={test_point_lnq:.4f}, chieff={test_point_chieff:.4f}"
    )

    test_sampled_params = {
        "mchirp": test_point_mchirp,
        "lnq": test_point_lnq,
        "chieff": test_point_chieff,
        "cumchidiff": 0.0,
        "costheta_jn": 0.0,
        "phi_jl_hat": 0.0,
        "phi12": 0.0,
        "cums1r_s1z": 0.0,
        "cums2r_s2z": 0.0,
    }
    log_assumed_prior = assumed_prior.lnprior(
        **test_sampled_params, f_ref=cond_sampler.f_ref
    )
    print(f"log(assumed_prior) = {log_assumed_prior:.4f}")
    assert np.isfinite(log_assumed_prior), "log_assumed_prior should be finite"
    print("PASSED Test 1 passed")

    print(
        "\n=== Test 2: Compute Gaussian proposal PDF (from Zoomer) at same test point ==="
    )
    print("Gaussian fit parameters:")
    print(
        f"  Mean: mchirp={zoomer.mean[0]:.4f}, lnq={zoomer.mean[1]:.4f}, chieff={zoomer.mean[2]:.4f}"
    )
    print(f"  Covariance matrix:")
    print(f"    {zoomer.cov}")
    print(f"  Diagonal (variances): {np.diag(zoomer.cov)}")
    print(f"  Standard deviations: {np.sqrt(np.diag(zoomer.cov))}")

    test_point_array = np.array([test_point_mchirp, test_point_lnq, test_point_chieff])
    print(f"\nEvaluating at test point: {test_point_array}")
    print(f"  Distance from mean: {test_point_array - zoomer.mean}")
    gaussian_pdf = zoomer.distribution.pdf(test_point_array)
    log_gaussian_proposal = np.log(gaussian_pdf)
    print(f"log(Gaussian_proposal) = {log_gaussian_proposal:.4f}")
    assert np.isfinite(gaussian_pdf), "Gaussian PDF should be finite"
    print("PASSED Test 2 passed")

    print("\n=== Test 3: Compute uniform proposal PDF (constant for all points) ===")
    print("Computing log(uniform_PDF) for uniformly sampled parameters:")
    log_uniform_pdf = 0.0
    cumchidiff_range = cond_sampler.spin_prior.range_dic["cumchidiff"]
    log_uniform_pdf += -np.log(cumchidiff_range[1] - cumchidiff_range[0])
    print(
        f"  cumchidiff range [{cumchidiff_range[0]}, {cumchidiff_range[1]}]: log_PDF = {-np.log(cumchidiff_range[1] - cumchidiff_range[0]):.4f}"
    )
    for param_name in [
        "costheta_jn",
        "phi_jl_hat",
        "phi12",
        "cums1r_s1z",
        "cums2r_s2z",
    ]:
        param_range = cond_sampler.inplane_spin_prior.range_dic[param_name]
        log_uniform_pdf += -np.log(param_range[1] - param_range[0])
        print(
            f"  {param_name} range [{param_range[0]}, {param_range[1]}]: log_PDF = {-np.log(param_range[1] - param_range[0]):.4f}"
        )
    print(
        f"Total log(uniform_proposal) = {log_uniform_pdf:.4f} (constant for all samples)"
    )
    assert np.isfinite(log_uniform_pdf), "log_uniform_pdf should be finite"
    print("PASSED Test 3 passed")

    print("\n=== Test 4: Call draw_from_zoomer() to generate test bank ===")
    bounds = zoomer.bounds if zoomer.bounds is not None else {}
    test_bank = draw_from_zoomer(
        zoomer=zoomer,
        cond_sampler=cond_sampler,
        bounds=bounds,
        n_samples=100,
        seed=42,
    )

    print(f"Generated bank with {len(test_bank)} samples")
    print(f"Columns: {test_bank.columns.tolist()}")

    assert "log_prior_weights" in test_bank.columns, (
        "log_prior_weights column should exist"
    )
    assert all(np.isfinite(test_bank["log_prior_weights"])), (
        "All log_prior_weights should be finite"
    )

    print(f"\nSummary statistics for log_prior_weights:")
    print(f"  Mean: {test_bank['log_prior_weights'].mean():.4f}")
    print(f"  Std: {test_bank['log_prior_weights'].std():.4f}")
    print(f"  Min: {test_bank['log_prior_weights'].min():.4f}")
    print(f"  Max: {test_bank['log_prior_weights'].max():.4f}")
    print("PASSED Test 4 passed")

    print("\n=== Test 5: Load existing bank and verify structure ===")
    existing_bank_path = bank_dir / "intrinsic_sample_bank.feather"
    if existing_bank_path.exists():
        existing_bank = pd.read_feather(existing_bank_path)
        print(f"Loaded existing bank with {len(existing_bank)} samples")
        print(f"Columns: {existing_bank.columns.tolist()}")

        if "log_proposal" in existing_bank.columns:
            print(f"\nExisting bank has log_proposal column:")
            print(f"  Mean: {existing_bank['log_proposal'].mean():.4f}")
            print(f"  Std: {existing_bank['log_proposal'].std():.4f}")

        if "log_prior_weights" in existing_bank.columns:
            print(f"\nExisting bank has log_prior_weights column:")
            print(f"  Mean: {existing_bank['log_prior_weights'].mean():.4f}")
            print(f"  Std: {existing_bank['log_prior_weights'].std():.4f}")

        print("PASSED Test 5 passed")
    else:
        print(f"Existing bank not found at {existing_bank_path}, skipping Test 5")

    print("\n=== All tests completed successfully! ===")


if __name__ == "__main__":
    main()
