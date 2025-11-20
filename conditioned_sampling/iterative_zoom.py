#!/usr/bin/env python3
"""Iterative zoom experiment that reuses PowerLawIntrinsicIASPrior.

Run as a script:
    python iterative_zoom.py --iterations 2 --n-zoom 2000
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import sys
import shutil
from textwrap import wrap

sys.path.append("/Users/jonatahm/Work/GW/dot-pe-future/")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cogwheel import gw_plotting
from cogwheel.gw_utils import m1m2_to_mchirp, chieff as gw_chieff

from dot_pe.power_law_mass_prior import PowerLawIntrinsicIASPrior
from dot_pe.zoom.zoom import Zoomer

from conditioned_sampling.conditional_sampler import ConditionalPriorSampler


@dataclass
class LikelihoodSummary:
    loglikes: np.ndarray
    weights: np.ndarray
    n_effective: float


@dataclass(frozen=True)
class GaussianParameter:
    mean: float
    sigma: float

    def log_likelihood(self, value: np.ndarray | float) -> np.ndarray | float:
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")
        norm = -0.5 * np.log(2 * np.pi * self.sigma**2)
        resid = (value - self.mean) / self.sigma
        return norm - 0.5 * resid**2


@dataclass
class ZoomConfig:
    weight_mass: float = 0.9
    n_zoom_samples: int = 10**3


REQUIRED_SAMPLE_COLS = {
    "mchirp",
    "lnq",
    "chieff",
    "loglike",
    "weights",
}
RANGE_COLUMNS = ["m1", "m2", "s1z", "s2z", "mchirp", "lnq", "chieff"]


TARGETS: Dict[str, GaussianParameter] = {
    "mchirp": GaussianParameter(30.0, 0.5),
    "chieff": GaussianParameter(0.3, 0.01),
    "lnq": GaussianParameter(-0.5, 0.1),
    "iota": GaussianParameter(0.7, 0.5),
    "chip": GaussianParameter(0.3, 0.1),
}

PRIOR_KWARGS = dict(mchirp_range=(5.0, 100.0), q_min=0.1, f_ref=50.0)
PLOT_PARAMS = ["mchirp", "lnq", "chieff", "chip", "iota"]


def compute_mchirp(m1: float, m2: float) -> float:
    return float(m1m2_to_mchirp(m1, m2))


def compute_lnq(m1: float, m2: float) -> float:
    return float(np.log(m2 / m1))


def compute_chieff(m1: float, m2: float, s1z: float, s2z: float) -> float:
    return float(gw_chieff(m1, m2, s1z, s2z))


def compute_chip(
    m1: float,
    m2: float,
    s1x_n: float,
    s1y_n: float,
    s2x_n: float,
    s2y_n: float,
) -> float:
    q = m2 / m1
    a1 = 2 + 1.5 * q
    a2 = 2 + 1.5 / q
    s1p = np.hypot(s1x_n, s1y_n)
    s2p = np.hypot(s2x_n, s2y_n)
    term1 = (a1 / a2) * q ** (-2) * s1p
    term2 = s2p
    return float(max(term1, term2))


def ensure_output_dir(subdir: str, root: Path | None = None) -> Path:
    base = (
        Path(root) if root is not None else Path(__file__).resolve().parent / "output"
    )
    base.mkdir(exist_ok=True)
    out = base / subdir
    out.mkdir(exist_ok=True)
    return out


def init_logger(output_dir: Path) -> callable:
    log_path = output_dir / "zoom_log.txt"
    log_path.write_text("")  # Fresh log per run

    def log(message: str) -> None:
        text = str(message)
        lines = text.splitlines() or [text]
        with log_path.open("a") as handle:
            for raw_line in lines:
                if not raw_line:
                    handle.write("\n")
                    continue
                wrapped = wrap(
                    raw_line, width=72, break_long_words=False, drop_whitespace=False
                )
                if not wrapped:
                    handle.write("\n")
                    continue
                for line in wrapped:
                    handle.write(line.rstrip() + "\n")

    return log


def format_ranges(df: pd.DataFrame, cols: List[str]) -> str:
    stats = df[cols].agg(["min", "max"]).T
    return stats.to_string(float_format=lambda v: f"{v:8.3f}")


def log_sample_snapshot(
    logger: callable,
    title: str,
    samples: pd.DataFrame,
    *,
    summary: LikelihoodSummary | None = None,
) -> None:
    if summary is not None:
        logger(
            f"{title} effective samples: {summary.n_effective:.2f} "
            f"out of {len(samples)}"
        )
    else:
        logger(f"{title} sample count: {len(samples)}")

    range_cols = [col for col in RANGE_COLUMNS if col in samples.columns]
    if range_cols:
        logger("Ranges:\n" + format_ranges(samples, range_cols))

    if "loglike" in samples.columns:
        sorted_samples = samples.sort_values("loglike", ascending=False)
        logger(
            "Top 3 by loglike:\n"
            + sorted_samples[PLOT_PARAMS + ["loglike"]]
            .head(3)
            .to_string(index=False, float_format=lambda v: f"{v:8.4f}")
        )
        logger(
            "Bottom 3 by loglike:\n"
            + sorted_samples[PLOT_PARAMS + ["loglike"]]
            .tail(3)
            .to_string(index=False, float_format=lambda v: f"{v:8.4f}")
        )
        logger(
            "Loglike range: min={:.4f} max={:.4f}".format(
                samples["loglike"].min(), samples["loglike"].max()
            )
        )
    logger("")


def summarize_iteration_lines(
    iteration: int, samples: pd.DataFrame, summary: LikelihoodSummary
) -> List[str]:
    lines = [
        f"iter{iteration}:",
        f"  mchirp range [{samples['mchirp'].min():.3f}, {samples['mchirp'].max():.3f}]",
        f"  lnq range    [{samples['lnq'].min():.3f}, {samples['lnq'].max():.3f}]",
        f"  chieff range [{samples['chieff'].min():.3f}, {samples['chieff'].max():.3f}]",
        f"  n_eff = {summary.n_effective:.2f}",
    ]
    return lines


def compute_log_prior_from_standard(
    prior: PowerLawIntrinsicIASPrior, row: pd.Series
) -> float:
    std = {name: row.get(name, 0.0) for name in prior.standard_params}
    std["f_ref"] = PRIOR_KWARGS["f_ref"]
    sampled = prior.inverse_transform(**std)
    return prior.lnprior(**sampled, f_ref=PRIOR_KWARGS["f_ref"])


def sample_from_prior(
    prior: PowerLawIntrinsicIASPrior,
    n_samples: int,
    *,
    seed: int | None = None,
    ranges: Dict[str, Tuple[float, float]] | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampled_params = prior.sampled_params
    rows: List[Dict[str, float]] = []

    for _ in range(n_samples):
        sampled = {}
        for name in sampled_params:
            low, high = (
                ranges[name] if ranges and name in ranges else prior.range_dic[name]
            )
            sampled[name] = rng.uniform(low, high)

        log_prior, std = prior.lnprior_and_transform(
            f_ref=PRIOR_KWARGS["f_ref"], **sampled
        )
        m1 = std["m1"]
        m2 = std["m2"]
        row = dict(std)
        row["mchirp"] = compute_mchirp(m1, m2)
        row["q"] = m2 / m1
        row["lnq"] = compute_lnq(m1, m2)
        row["chieff"] = compute_chieff(m1, m2, std["s1z"], std["s2z"])
        row["chip"] = compute_chip(
            m1, m2, std["s1x_n"], std["s1y_n"], std["s2x_n"], std["s2y_n"]
        )
        row["log_prior"] = log_prior
        for name, value in sampled.items():
            row[f"sampled_{name}"] = value
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_likelihoods(samples: pd.DataFrame) -> LikelihoodSummary:
    derived = {
        name: samples[name].to_numpy()
        for name in ["mchirp", "lnq", "chieff", "iota", "chip"]
    }

    loglikes = (
        TARGETS["mchirp"].log_likelihood(derived["mchirp"])
        + TARGETS["lnq"].log_likelihood(derived["lnq"])
        + TARGETS["chieff"].log_likelihood(derived["chieff"])
        + TARGETS["iota"].log_likelihood(derived["iota"])
        + TARGETS["chip"].log_likelihood(derived["chip"])
    )
    loglikes -= loglikes.max()
    weights = np.exp(loglikes)
    weights /= weights.sum()
    n_effective = 1.0 / np.sum(weights**2)
    return LikelihoodSummary(
        loglikes=loglikes, weights=weights, n_effective=n_effective
    )


def draw_initial_samples(
    prior: PowerLawIntrinsicIASPrior,
    n_samples: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, LikelihoodSummary]:
    raw = sample_from_prior(prior, n_samples, seed=rng.integers(0, 2**32))
    summary = evaluate_likelihoods(raw)
    samples = raw.assign(loglike=summary.loglikes, weights=summary.weights)
    return samples, summary


def select_zoom_seed(
    samples: pd.DataFrame,
    weights: np.ndarray,
    weight_mass: float,
    n_effective: float | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    order = np.argsort(weights)[::-1]
    sorted_weights = weights[order]
    cumsum = np.cumsum(sorted_weights)
    n_enough_mass = int(np.searchsorted(cumsum, weight_mass, side="left")) + 1
    n_eff_req = 0 if n_effective is None else int(np.ceil(5 * n_effective))
    base_min = 20
    n_select = min(len(samples), max(n_enough_mass, n_eff_req, base_min))
    info = {
        "n_select": n_select,
        "mass_target_count": n_enough_mass,
        "n_eff_requirement": n_eff_req,
        "base_min": base_min,
        "achieved_mass": float(cumsum[n_select - 1]),
    }
    top_idx = order[:n_select]
    return samples.iloc[top_idx].reset_index(drop=True), info


def normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    log_weights = log_weights - np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= weights.sum()
    return weights


def zoom_once(
    samples: pd.DataFrame,
    summary: LikelihoodSummary,
    *,
    prior: PowerLawIntrinsicIASPrior,
    cond_sampler: ConditionalPriorSampler,
    rng: np.random.Generator,
    logger: callable,
    config: ZoomConfig,
) -> Tuple[pd.DataFrame, LikelihoodSummary]:
    missing = REQUIRED_SAMPLE_COLS - set(samples.columns)
    if missing:
        raise ValueError(f"samples missing required columns: {sorted(missing)}")

    base_cols = ["mchirp", "lnq", "chieff"]
    zoom_seed, seed_info = select_zoom_seed(
        samples,
        samples["weights"].to_numpy(),
        config.weight_mass,
        summary.n_effective,
    )
    logger(
        "Zoom seed keeps {n} samples (mass target count={target_count}, "
        "5*n_eff≈{n_eff_req}, floor={base_min}, achieved mass={mass:.3f}).".format(
            n=seed_info["n_select"],
            target_count=seed_info["mass_target_count"],
            n_eff_req=seed_info["n_eff_requirement"],
            base_min=seed_info["base_min"],
            mass=seed_info["achieved_mass"],
        )
    )
    logger(
        "Zoom seed ranges:\n"
        + format_ranges(
            zoom_seed,
            [col for col in RANGE_COLUMNS if col in zoom_seed.columns],
        )
    )
    logger("")

    zoomer = Zoomer(engine_seed=int(rng.integers(0, 2**32)))
    zoomer.fit(zoom_seed[base_cols].to_numpy(), weights=zoom_seed["weights"].to_numpy())
    logger("Zoomer mean: " + np.array2string(zoomer.mean, precision=4))
    logger(
        "Zoomer std diag: " + np.array2string(np.sqrt(np.diag(zoomer.cov)), precision=4)
    )

    bounds = {
        0: ("mchirp", (samples["mchirp"].min(), samples["mchirp"].max())),
        1: ("lnq", (samples["lnq"].min(), samples["lnq"].max())),
        2: ("chieff", (samples["chieff"].min(), samples["chieff"].max())),
    }
    logger("Zoom bounds (Gaussian over [mchirp, lnq, chieff]):")
    for axis, (name, range_tuple) in bounds.items():
        logger(f"  axis {axis} ({name}) in {range_tuple}")
    numeric_bounds = {axis: range_tuple for axis, (_, range_tuple) in bounds.items()}

    base_samples, base_pdf = zoomer.sample(config.n_zoom_samples, bounds=numeric_bounds)
    base_df = pd.DataFrame(base_samples, columns=base_cols)
    logger(
        "Zoom Gaussian pdf stats at proposal draws: "
        "min={:.3e} max={:.3e} mean={:.3e}".format(
            base_pdf.min(), base_pdf.max(), base_pdf.mean()
        )
    )
    logger("")
    log_proposal = np.log(np.clip(base_pdf, 1e-300, None))

    zoom_rows: List[Dict[str, float]] = []
    for idx in range(base_df.shape[0]):
        mchirp_val, lnq_val, chieff_val = base_df.iloc[idx]

        cond_sampler.seed = int(rng.integers(0, 2**63 - 1))
        cond_row = (
            cond_sampler.sample(
                1,
                mchirp=mchirp_val,
                lnq=lnq_val,
                chieff=chieff_val,
                method="mc",
            )
            .iloc[0]
            .to_dict()
        )
        cond_row.update(
            {
                "mchirp": mchirp_val,
                "lnq": lnq_val,
                "chieff": chieff_val,
                "chip": compute_chip(
                    cond_row["m1"],
                    cond_row["m2"],
                    cond_row["s1x_n"],
                    cond_row["s1y_n"],
                    cond_row["s2x_n"],
                    cond_row["s2y_n"],
                ),
                "log_proposal": log_proposal[idx],
            }
        )
        zoom_rows.append(cond_row)

    zoom = pd.DataFrame(zoom_rows)
    zoom_summary = evaluate_likelihoods(zoom)
    zoom_log_prior = zoom.apply(
        lambda r: compute_log_prior_from_standard(prior, r), axis=1
    )
    log_weights = zoom_summary.loglikes + zoom_log_prior.to_numpy() - log_proposal
    zoom_weights = normalize_log_weights(log_weights)
    zoom = zoom.assign(
        loglike=zoom_summary.loglikes,
        log_prior=zoom_log_prior,
        log_proposal=log_proposal,
        weights=zoom_weights,
    )
    log_sample_snapshot(logger, "Zoom samples", zoom, summary=zoom_summary)
    return zoom, zoom_summary


def save_samples_csv(samples: pd.DataFrame, output_dir: Path, name: str) -> None:
    samples.to_csv(output_dir / name, index=False)


def save_corner_plot(
    dataframes: List[pd.DataFrame],
    labels: List[str],
    output_path: Path,
) -> None:
    plot_params = PLOT_PARAMS + ["loglike"]
    try:
        cp = gw_plotting.MultiCornerPlot(
            dataframes,
            params=plot_params,
            labels=labels,
            tail_probability=1e-2,
            weights_col="weights",
        )
        cp.plot()
        fig = cp.figure if hasattr(cp, "figure") else plt.gcf()
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
    except Exception as err:
        print(f"Warning: failed to create corner plot ({err}). Skipping.")


def run_zoom_pipeline(
    *,
    n_initial_samples: int = 10**5,
    n_iterations: int = 1,
    seed: int | None = 1234,
    config: ZoomConfig | None = None,
    output_root: str | Path | None = None,
) -> None:
    config = config or ZoomConfig()
    rng = np.random.default_rng(seed)
    prior = PowerLawIntrinsicIASPrior(**PRIOR_KWARGS)
    cond_sampler = ConditionalPriorSampler(
        mchirp_range=PRIOR_KWARGS["mchirp_range"],
        q_min=PRIOR_KWARGS["q_min"],
        f_ref=PRIOR_KWARGS["f_ref"],
        seed=None,
    )

    root = (
        Path(output_root)
        if output_root is not None
        else Path(__file__).resolve().parent / "output"
    )
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    samples, summary = draw_initial_samples(prior, n_initial_samples, rng)

    lines: List[str] = []
    lines.append("Likelihood targets (mean ± sigma):")
    for name, target in TARGETS.items():
        lines.append(f"  {name}: {target.mean:.3f} ± {target.sigma:.3f}")
    lines.append("")

    for iteration in range(0, n_iterations + 1):
        subdir = ensure_output_dir(f"iter{iteration}", root=root)
        logger = init_logger(subdir)

        if iteration == 0:
            logger("=== Iteration 0: initial prior draw ===")
            log_sample_snapshot(logger, "Initial samples", samples, summary=summary)
            save_samples_csv(samples, subdir, "samples.csv")
            lines.extend(summarize_iteration_lines(0, samples, summary))
            continue

        logger(f"=== Iteration {iteration}: zoom ===")
        log_sample_snapshot(
            logger,
            "Input samples",
            samples,
            summary=summary,
        )
        prev_samples = samples
        prev_summary = summary
        samples, summary = zoom_once(
            prev_samples,
            prev_summary,
            prior=prior,
            cond_sampler=cond_sampler,
            rng=rng,
            logger=logger,
            config=config,
        )
        save_samples_csv(prev_samples, subdir, "input_samples.csv")
        save_samples_csv(samples, subdir, "samples.csv")
        logger(
            f"Iteration {iteration} effective samples: {summary.n_effective:.2f} "
            f"out of {len(samples)}"
        )
        save_corner_plot(
            [prev_samples, samples],
            [
                f"Iteration {iteration - 1}",
                f"Iteration {iteration}",
            ],
            subdir / "multi_corner.jpg",
        )
        lines.extend(summarize_iteration_lines(iteration, samples, summary))

    increment_path = root / "increment_summary.txt"
    increment_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run iterative zoom sampling pipeline."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of zoom iterations to perform (default: 1).",
    )
    parser.add_argument(
        "--n-initial",
        type=int,
        default=10**5,
        help="Number of prior samples drawn for iteration 0 (default: 100000).",
    )
    parser.add_argument(
        "--n-zoom",
        type=int,
        default=10**3,
        help="Number of samples drawn from each zoom Gaussian (default: 1000).",
    )
    parser.add_argument(
        "--weight-mass",
        type=float,
        default=0.9,
        help="Fraction of weight retained when selecting seeds (default: 0.9).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility (default: 1234).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional output root directory (default: conditioned_sampling/output).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_zoom_pipeline(
        n_initial_samples=args.n_initial,
        n_iterations=args.iterations,
        seed=args.seed,
        config=ZoomConfig(
            weight_mass=args.weight_mass,
            n_zoom_samples=args.n_zoom,
        ),
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
