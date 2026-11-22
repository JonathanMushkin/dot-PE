#!/usr/bin/env python
"""E3 analysis: bias, spread and the accept/reject test per rule; figures 1 to 4.

Design: research/weight-thinning/design.md. Reads artifacts/research/weight-thinning/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "artifacts/research/weight-thinning"
RAND = ["a", "b", "e", "f"]
DET = ["d", "c"]
PARS = ("chieff", "mchirp", "q")
QCOLS = [f"{p}_q{q:02d}" for p in PARS for q in (5, 50, 95)]
MEASURES = ["evidence_ratio", *QCOLS, "P_LVK"]
NAMES = {"d": "(d) current: top k by b", "c": "(c) top k by weight",
         "a": "(a) priority, one stage", "b": "(b) priority, three stages",
         "e": "(e) priority, defensive key", "f": "(f) phase-marginalized priority"}


def main() -> None:
    import thin

    ref = json.loads((OUT / "reference.json").read_text())
    runs = {r: pd.read_csv(OUT / f"{r}.csv") for r in RAND + DET}
    ceil = {"evidence_ratio": 0.05, "P_LVK": 0.02}
    for p in PARS:
        for q in (5, 50, 95):
            ceil[f"{p}_q{q:02d}"] = 0.1 * ref[f"{p}_std"]
    rows = []
    for r, df in runs.items():
        n = len(df)
        for m in MEASURES + ["b_min", "neff_i", "neff_e", "n_kept"]:
            mean, sd = df[m].mean(), df[m].std(ddof=1) if n > 1 else 0.0
            refv = 1.0 if m == "evidence_ratio" else ref.get(m, np.nan)
            row = {"rule": r, "measure": m, "reference": refv, "mean": mean, "bias": mean - refv,
                   "spread": sd}
            if r in RAND and m in MEASURES:
                row["bias_ok"] = abs(mean - refv) <= 2 * sd / np.sqrt(n) + 0.1 * sd
                row["spread_ok"] = sd <= ceil[m]
            rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "measures.csv", index=False, float_format="%.5g")
    neff_d = runs["d"]["neff_i"].mean()
    verdict = {}
    for r in RAND:
        t = tab[(tab.rule == r) & tab.measure.isin(MEASURES)]
        ok_b, ok_s = bool(t.bias_ok.all()), bool(t.spread_ok.all())
        ok_n = bool(runs[r]["neff_i"].mean() >= neff_d)
        verdict[r] = {"bias_ok": ok_b, "spread_ok": ok_s, "neff_i_ok": ok_n,
                      "pass": ok_b and ok_s and ok_n,
                      "P_LVK_spread": float(runs[r]["P_LVK"].std(ddof=1)),
                      "failed": t.loc[~(t.bias_ok & t.spread_ok), "measure"].tolist()}
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))
    show = tab[tab.measure.isin(["evidence_ratio", "chieff_q05", "chieff_q50", "mchirp_q50",
                                 "q_q50", "P_LVK", "b_min", "neff_i", "neff_e"])]
    print(show.pivot(index="measure", columns="rule", values="mean").round(4).to_string())
    print(show.pivot(index="measure", columns="rule", values="spread").round(4).to_string())
    figures(ref, runs, tab, thin)


def figures(ref, runs, tab, thin) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rev = HERE / "review"
    rev.mkdir(exist_ok=True)
    T = np.load(OUT / "T_hists.npz")
    cols = {"d": "#d62728", "a": "#1f77b4", "b": "#2ca02c", "e": "#9467bd", "f": "#ff7f0e",
            "c": "0.5"}
    bc = 0.5 * (thin.B_BINS[1:] + thin.B_BINS[:-1])
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.step(bc, T["b_counts"], where="mid", color="k", lw=2, label="T (all rows ≥ c)")
    for r in ["d", "a", "b", "e", "f"]:
        h = np.load(OUT / f"{r}_seed0.npz")["b_counts"]
        ax.step(bc, np.maximum(h, 0.5), where="mid", color=cols[r], lw=1, label=NAMES[r])
    ax.set_yscale("log")
    ax.set_xlabel("distance-maximized log-likelihood b of a kept row")
    ax.set_ylabel("rows per 0.25 bin (seed 0)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(rev / "fig1_b_kept.png", dpi=150)
    plt.close(fig)

    cc = 0.5 * (thin.CHI_BINS[1:] + thin.CHI_BINS[:-1])
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.step(cc, T["chi_lvk"], where="mid", color="k", lw=2, label="T")
    for r in ["d", "a", "b", "e", "f"]:
        h = np.load(OUT / f"{r}_seed0.npz")["chi_lvk"]
        ax.step(cc, h, where="mid", color=cols[r], lw=1, label=NAMES[r])
    ax.axvline(0.4, color="0.5", ls=":")
    ax.set_xlim(-0.2, 1)
    ax.set_xlabel("chi_eff")
    ax.set_ylabel("posterior mass per bin, LVK spin prior (seed 0)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(rev / "fig2_chieff_lvk.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for n, r in enumerate(RAND):
        v = runs[r]["P_LVK"].to_numpy()
        ax.scatter(n + np.random.default_rng(0).uniform(-0.2, 0.2, len(v)), v, s=6,
                   color=cols[r])
    for r in DET:
        ax.axhline(runs[r]["P_LVK"].iloc[0], color=cols[r], ls="--", lw=1, label=NAMES[r])
    ax.axhline(ref["P_LVK"], color="k", lw=1.5, label="T")
    ax.set_xticks(range(len(RAND)), [NAMES[r] for r in RAND], fontsize=6)
    ax.set_ylabel("P_LVK, weight at chi_eff < 0.4 (LVK spin prior)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(rev / "fig3_P_LVK_seeds.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(HERE))
    main()
