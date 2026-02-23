#!/usr/bin/env python3
"""Perform a single zoom iteration using previous inference samples.

The script does three core operations:
1. Fit a Gaussian zoomer from previous run
2. Create a new bank using the zoomer + conditional sampler
3. Run inference with the new bank
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path for CLI execution
if __name__ == "__main__":
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from cogwheel import data
from cogwheel.gw_prior.spin import UniformEffectiveSpinPrior
from dot_pe import inference, waveform_banks
from dot_pe.mass_prior import get_mass_prior
from dot_pe.zoom.conditional_sampling import ConditionalPriorSampler
from dot_pe.zoom.zoom import Zoomer
from dot_pe.utils import load_intrinsic_samples_from_rundir


# ============================================================================
# Helper functions
# ============================================================================


def fit_zoomer(
    df: pd.DataFrame,
    weights: np.ndarray,
    prior_kwargs: dict,
    seed: int,
    n_sig: float = None,
    aligned_spin: bool = False,
) -> tuple[Zoomer, ConditionalPriorSampler, dict]:
    """Fit Gaussian zoomer to weighted samples."""
    mchirp = df["mchirp"].values
    lnq = df["lnq"].values
    chieff = df["chieff"].values
    data = np.column_stack([mchirp, lnq, chieff])

    mchirp_range = prior_kwargs["mchirp_range"]
    q_min = prior_kwargs["q_min"]
    q_max = prior_kwargs.get("q_max", 1.0)
    bounds = {
        0: (mchirp_range[0], mchirp_range[1]),
        1: (np.log(q_min), np.log(q_max)),
        2: (-1.0, 1.0),
    }

    zoomer = Zoomer(engine_seed=seed)
    zoomer.fit(data, weights, n_sig)
    cond_sampler = ConditionalPriorSampler(
        **prior_kwargs, aligned_spin=aligned_spin, seed=seed
    )

    return zoomer, cond_sampler, bounds


def hellinger_distance(xs, ys, logp, logq):
    """Estimate Hellinger distance H = sqrt(1 - BC)."""
    lp_x, lq_x = logp(xs), logq(xs)
    logden_x = np.logaddexp(lp_x, lq_x)
    w_x = np.exp(np.log(2.0) + 0.5 * (lp_x + lq_x) - logden_x)
    bc_x = np.mean(w_x[np.isfinite(w_x)])

    lp_y, lq_y = logp(ys), logq(ys)
    logden_y = np.logaddexp(lp_y, lq_y)
    w_y = np.exp(np.log(2.0) + 0.5 * (lp_y + lq_y) - logden_y)
    bc_y = np.mean(w_y[np.isfinite(w_y)])

    BC = 0.5 * (bc_x + bc_y)
    BC = float(np.clip(BC, 0.0, 1.0))
    return float(np.sqrt(1.0 - BC))


def draw_from_zoomer(
    zoomer: Zoomer,
    cond_sampler: ConditionalPriorSampler,
    bounds: dict,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Draw samples from zoomer + conditional sampler with importance weights."""
    samples, gaussian_pdfs = zoomer.sample(n_samples, bounds=bounds)
    mchirp, lnq, chieff = samples.T

    df = cond_sampler.sample_vectorized(mchirp, lnq, chieff)

    df["mchirp"] = mchirp
    df["lnq"] = lnq
    df["chieff"] = chieff

    q_max = getattr(cond_sampler, "q_max", 1.0)
    mass_prior = get_mass_prior(
        mchirp_range=cond_sampler.mchirp_range,
        q_min=cond_sampler.q_min,
        q_max=q_max,
    )
    aligned_spin_prior = UniformEffectiveSpinPrior()

    log_assumed_prior = np.vectorize(mass_prior.lnprior)(
        df["mchirp"].values, df["lnq"].values
    ) + np.vectorize(aligned_spin_prior.lnprior)(
        df["chieff"].values,
        df["cumchidiff"].values,
        df["m1"].values,
        df["m2"].values,
    )
    log_gaussian = np.log(gaussian_pdfs)

    remaining_sampled_params = [
        "cumchidiff",
        "costheta_jn",
        "phi_jl_hat",
        "phi12",
        "cums1r_s1z",
        "cums2r_s2z",
    ]
    log_uniform = 0.0
    for param_name in remaining_sampled_params:
        if param_name == "cumchidiff":
            param_range = cond_sampler.aligned_spin_prior.range_dic[param_name]
        else:
            param_range = cond_sampler.inplane_spin_prior.range_dic[param_name]
        log_uniform -= np.log(param_range[1] - param_range[0])

    log_proposal = log_gaussian + log_uniform
    df["log_prior_weights"] = log_assumed_prior - log_proposal

    return df


# ============================================================================
# Main script
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Perform zoom iteration")
    parser.add_argument("--prev-run-dir", required=True, help="Previous inference run")
    parser.add_argument("--event-file", required=True, help="Event data file (.npz)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--n-samples", type=int, default=4096, help="Number of bank samples"
    )
    parser.add_argument(
        "--approximant", default="IMRPhenomXODE", help="Waveform approximant"
    )
    parser.add_argument(
        "--bank-blocksize", type=int, default=4096, help="Waveform generation blocksize"
    )
    parser.add_argument("--n-pool", type=int, default=None, help="Number of processes")
    parser.add_argument(
        "--n-ext", type=int, default=None, help="Number of extrinsic samples"
    )
    parser.add_argument(
        "--extrinsic-samples",
        type=str,
        default=None,
        help="Path to extrinsic samples file (.feather)",
    )
    parser.add_argument(
        "--n-phi", type=int, default=100, help="Number of phi_ref samples"
    )
    parser.add_argument(
        "--n-phi-incoherent",
        type=int,
        default=None,
        help="Number of incoherent phi_ref samples (default: same as n-phi)",
    )
    parser.add_argument("--n-t", type=int, default=128, help="Number of time samples")
    parser.add_argument(
        "--inference-blocksize", type=int, default=512, help="Inference blocksize"
    )
    parser.add_argument(
        "--single-detector-blocksize",
        type=int,
        default=None,
        help="Single detector blocksize (default: same as inference-blocksize)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n-sig",
        type=float,
        default=None,
        help="Number of standard deviations for zoomer fitting (default: None)",
    )
    return parser.parse_args()


def main() -> None:
    """Main workflow: fit zoomer -> create bank -> run inference."""
    args = parse_args()

    prev_run_dir = Path(args.prev_run_dir).expanduser().resolve()
    event_file = Path(args.event_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_kwargs = json.load(open(prev_run_dir / "run_kwargs.json"))
    bank_dir = Path(run_kwargs["bank_folder"])
    bank_config = json.load(open(bank_dir / "bank_config.json"))

    mchirp_range = bank_config.get("mchirp_range")
    if mchirp_range is None:
        mchirp_range = (bank_config["mchirp_min"], bank_config["mchirp_max"])
    prior_kwargs = {
        "mchirp_range": tuple(mchirp_range),
        "q_min": bank_config.get("q_min"),
        "q_max": bank_config.get("q_max", 1.0),
        "f_ref": bank_config.get("f_ref"),
    }

    print("Fitting zoomer from previous run...")
    weighted_df = load_intrinsic_samples_from_rundir(prev_run_dir)
    zoomer, cond_sampler, bounds = fit_zoomer(
        weighted_df, weighted_df["weights"].values, prior_kwargs, args.seed, args.n_sig
    )

    zoomer.to_json(output_dir / "Zoomer.json")
    cond_sampler.to_json(output_dir / "ConditionalPriorSampler.json")
    print(f"Saved zoomer to {output_dir / 'Zoomer.json'}")

    print(f"Creating bank with {args.n_samples} samples...")
    bank_output_dir = output_dir / "bank"
    bank_output_dir.mkdir(parents=True, exist_ok=True)

    new_bank = draw_from_zoomer(zoomer, cond_sampler, bounds, args.n_samples, args.seed)

    samples_path = bank_output_dir / "intrinsic_sample_bank.feather"
    new_bank.to_feather(samples_path)

    bank_config.setdefault("q_max", 1.0)
    bank_config.update({"bank_size": len(new_bank)})
    json.dump(bank_config, open(bank_output_dir / "bank_config.json", "w"), indent=2)
    print(f"Created bank at {samples_path}")

    print("Generating waveforms...")
    n_pool = args.n_pool if args.n_pool is not None else max(1, os.cpu_count() or 1)
    waveform_banks.create_waveform_bank_from_samples(
        samples_path=samples_path,
        bank_config_path=bank_output_dir / "bank_config.json",
        waveform_dir=bank_output_dir / "waveforms",
        n_pool=n_pool,
        blocksize=args.bank_blocksize,
        approximant=args.approximant,
    )
    print(f"Generated waveforms in {bank_output_dir / 'waveforms'}")

    print("Running inference...")
    event_data = data.EventData.from_npz(filename=event_file)

    extrinsic_samples_path = None
    if args.extrinsic_samples is not None:
        extrinsic_samples_path = Path(args.extrinsic_samples).expanduser().resolve()
        if not extrinsic_samples_path.exists():
            raise ValueError(
                f"Extrinsic samples file not found: {extrinsic_samples_path}"
            )
        extrinsic_df = pd.read_feather(extrinsic_samples_path)
        n_ext = len(extrinsic_df)
    elif args.n_ext is None:
        if (prev_run_dir / "extrinsic_samples.feather").exists():
            extrinsic_samples_path = prev_run_dir / "extrinsic_samples.feather"
            extrinsic_df = pd.read_feather(extrinsic_samples_path)
            n_ext = len(extrinsic_df)
        else:
            raise ValueError(
                "n_ext is required unless --extrinsic-samples is provided or extrinsic_samples.feather exists in prev_run_dir"
            )
    else:
        n_ext = args.n_ext

    inference_kwargs = {
        "event_dir": str(output_dir),
        "event": event_data,
        "bank_folder": str(bank_output_dir),
        "n_int": len(new_bank),
        "n_ext": n_ext,
        "n_phi": args.n_phi,
        "n_phi_incoherent": args.n_phi_incoherent,
        "n_t": args.n_t,
        "blocksize": args.inference_blocksize,
        "single_detector_blocksize": args.single_detector_blocksize
        if args.single_detector_blocksize is not None
        else args.inference_blocksize,
        "seed": args.seed,
    }

    if extrinsic_samples_path is not None:
        inference_kwargs["extrinsic_samples"] = str(extrinsic_samples_path)

    rundir = inference.run(**inference_kwargs)
    rundir = Path(rundir)

    kwargs = json.load(open(rundir / "run_kwargs.json"))
    kwargs["prev_run_dir"] = str(prev_run_dir)
    json.dump(kwargs, open(rundir / "run_kwargs.json", "w"), indent=2)

    prev_zoomer_paths = [
        prev_run_dir.parent / "Zoomer.json",
        prev_run_dir / "Zoomer.json",
    ]
    prev_zoomer_path = None
    for path in prev_zoomer_paths:
        if path.exists():
            prev_zoomer_path = path
            break

    if prev_zoomer_path is not None:
        prev_zoomer = Zoomer.from_json(prev_zoomer_path)
        xs_old, _ = prev_zoomer.sample(1000, bounds=bounds)
        xs_new, _ = zoomer.sample(1000, bounds=bounds)
        hellinger = hellinger_distance(
            xs_old, xs_new, prev_zoomer.distribution.logpdf, zoomer.distribution.logpdf
        )
        summary_path = rundir / "summary_results.json"
        if summary_path.exists():
            summary = json.load(open(summary_path))
            summary["hellinger_distance"] = hellinger
            json.dump(summary, open(summary_path, "w"), indent=4)

    print(f"Inference complete. Results in {rundir}")


if __name__ == "__main__":
    main()
