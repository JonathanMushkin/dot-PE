#!/usr/bin/env python
"""E3 budget sweep: P_LVK, quantile shifts and n_effective_i against the row budget, per rule.

Design: research/weight-thinning/design.md (design note of 2026-11-23). Env o4a-pe.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "artifacts/research/weight-thinning"
SCALES = [0.015625, 0.0625, 0.25, 1.0, 4.0]
RULES = {"d": ("rule 1: current, top rows by likelihood", "#d62728"),
         "b": ("rule 2: thinning of rows", "#2ca02c"),
         "e": ("rule 3: LVK-protected thinning of rows", "#9467bd"),
         "f": ("rule 4: thinning of phase-marginalized pairs", "#1f77b4")}
QS = [("chieff", 5), ("chieff", 50), ("chieff", 95), ("mchirp", 5), ("mchirp", 50),
      ("mchirp", 95)]
N_SEEDS = 30


def load() -> pd.DataFrame:
    parts = []
    for r in RULES:
        for s in SCALES:
            f = OUT / (f"{r}.csv" if s == 1.0 else f"{r}_s{s:g}.csv")
            d = pd.read_csv(f)
            d["scale"], d["rule"] = s, r
            parts.append(d.head(N_SEEDS))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ref = json.loads((OUT / "reference.json").read_text())
    d = load()
    width = {p: ref[f"{p}_q95"] - ref[f"{p}_q05"] for p in ("chieff", "mchirp")}
    for p, q in QS:
        d[f"shift_{p}_q{q:02d}"] = (d[f"{p}_q{q:02d}"] - ref[f"{p}_q{q:02d}"]) / width[p]
    d["err_P_LVK"] = d["P_LVK"] - ref["P_LVK"]
    d["size_limit_equiv"] = 1e7 * d["scale"]
    d.to_csv(OUT / "sweep.csv", index=False, float_format="%.6g")
    cols = ["err_P_LVK", *[f"shift_{p}_q{q:02d}" for p, q in QS], "neff_i", "b_min",
            "n_kept"]
    g = d.groupby(["rule", "scale"])
    summ = pd.concat({"mean": g[cols].mean(), "rms": g[cols].agg(lambda x: np.sqrt(np.mean(
        np.square(x)))), "sd": g[cols].std(ddof=1)}, axis=1)
    summ.to_csv(OUT / "sweep_summary.csv", float_format="%.5g")
    pd.set_option("display.width", 250)
    for c in ["err_P_LVK", "shift_chieff_q05", "shift_mchirp_q05", "neff_i", "b_min", "n_kept"]:
        print(c)
        print(pd.DataFrame({k: summ[(k, c)] for k in ("mean", "rms", "sd")}).unstack(0)
              .round(5).to_string())
    print("T:", {k: ref[k] for k in ("P_LVK", "neff_i", "chieff_q05", "chieff_q95",
                                     "mchirp_q05", "mchirp_q95", "n_T", "k")})
    print("90 % widths:", width)
    figures(d, ref)


def _x(d, r, n):
    rng = np.random.default_rng(n)
    off = {"d": -0.09, "b": -0.03, "e": 0.03, "f": 0.09}[r]
    return d["size_limit_equiv"] * 10 ** (off + rng.uniform(-0.015, 0.015, len(d)))


def figures(d, ref) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rev = HERE / "review"
    rev.mkdir(exist_ok=True)
    import thin

    T = np.load(OUT / "T_hists.npz")
    bc = 0.5 * (thin.B_BINS[1:] + thin.B_BINS[:-1])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.step(bc, T["b_counts"], where="mid", color="k", lw=2, label="all rows above the cut (T)")
    for r, (lab, col) in RULES.items():
        h = np.load(OUT / f"{r}_seed0.npz")["b_counts"].astype(float)
        h[h == 0] = np.nan
        ax.step(bc, h, where="mid", color=col, lw=1, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("distance-maximized log-likelihood b of a stored row")
    ax.set_ylabel("stored rows per 0.25 bin")
    ax.set_title("production budget (equivalent size_limit 1e7), one draw", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(rev / "fig1_b_kept.png", dpi=150)
    plt.close(fig)

    cc = 0.5 * (thin.CHI_BINS[1:] + thin.CHI_BINS[:-1])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.step(cc, T["chi_lvk"], where="mid", color="k", lw=2, label="all rows above the cut (T)")
    for r, (lab, col) in RULES.items():
        ax.step(cc, np.load(OUT / f"{r}_seed0.npz")["chi_lvk"], where="mid", color=col, lw=1,
                label=lab)
    ax.axvline(0.4, color="0.5", ls=":")
    ax.set_xlim(-0.2, 1)
    ax.set_xlabel("chi_eff")
    ax.set_ylabel("posterior mass per bin, LVK spin prior")
    ax.set_title("production budget (equivalent size_limit 1e7), one draw", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(rev / "fig2_chieff_lvk.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for n, (r, (lab, col)) in enumerate(RULES.items()):
        x = d[d.rule == r]
        ax.scatter(_x(x, r, n), x["P_LVK"], s=7, color=col, label=lab)
    ax.axhline(ref["P_LVK"], color="k", lw=1.2, label="all rows above the cut (T)")
    ax.axvline(1e7, color="0.6", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("row budget (equivalent size_limit)")
    ax.set_ylabel("P_LVK: posterior weight at chi_eff < 0.4,\nLVK spin prior")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(rev / "fig3_P_LVK_vs_budget.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for ax, (p, q) in zip(axes.ravel(), QS):
        for n, (r, (lab, col)) in enumerate(RULES.items()):
            x = d[d.rule == r]
            ax.scatter(_x(x, r, n), x[f"shift_{p}_q{q:02d}"], s=5, color=col, label=lab)
        ax.axhline(0, color="k", lw=1)
        for y in (-0.1, 0.1):
            ax.axhline(y, color="0.6", ls="--", lw=0.8)
        ax.axvline(1e7, color="0.6", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_title(f"{'chi_eff' if p == 'chieff' else 'chirp mass'}, {q} % quantile",
                     fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("row budget (equivalent size_limit)")
    for ax in axes[:, 0]:
        ax.set_ylabel("shift from T, in units of T's\n90 % interval width")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(rev / "fig4_quantile_shift_vs_budget.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for n, (r, (lab, col)) in enumerate(RULES.items()):
        x = d[d.rule == r]
        ax.scatter(_x(x, r, n), x["neff_i"], s=7, color=col, label=lab)
    ax.axhline(ref["neff_i"], color="k", lw=1.2, label="all rows above the cut (T)")
    ax.axhline(150, color="0.4", ls="--", lw=1, label="paper-repo minimum, 150")
    ax.axvline(1e7, color="0.6", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("row budget (equivalent size_limit)")
    ax.set_ylabel("n_effective_i, IAS prior")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(rev / "fig5_neff_i_vs_budget.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(HERE))
    main()
