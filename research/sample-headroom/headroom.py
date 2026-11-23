#!/usr/bin/env python
"""E2: headroom H in n_effective_i and n_effective_e for every production run.

Design: research/sample-headroom/design.md. Env o4a-pe; login node (reads JSON summaries only).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

O4B = Path("/home/projects/barakz/Collaboration-gw/mushkin/dot-pe/o4b-pe-paper")
O4A = O4B.parent / "o4a-pe-paper"
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "artifacts/research/sample-headroom"
MIN_NEFF_I, MIN_NEFF_E = 150.0, 100.0
STALE_BEFORE = 1783900800.0  # 2026-07-13 00:00 IST, dot-PE 2403731 (cross-bank n_effective fix)


def root_of(event: str) -> Path:
    return O4A if event[:8] < "GW240117" else O4B


def main() -> None:
    rl = pd.read_csv(O4B / "dev/row_limit_check/row_limit.csv")
    rows = []
    for e in rl.itertuples():
        rd = root_of(e.event) / "artifacts/pe/dotpe" / e.event / e.run
        r = {"event": e.event, "run": e.run, "best": e.bestfit_lnlike_max}
        ias = json.loads((rd / "summary_results.json").read_text())
        r["summary_stale"] = (rd / "summary_results.json").stat().st_mtime < STALE_BEFORE
        r["n_ext"] = json.loads((rd / "timing.json").read_text()).get("n_ext")
        r["neff_i_IAS"], r["neff_e_IAS"] = ias["n_effective_i"], ias["n_effective_e"]
        f = rd / "summary_results_lvc.json"
        if f.exists():
            lvk = json.loads(f.read_text())
            r["neff_i_LVK"], r["neff_e_LVK"] = lvk["n_effective_i"], lvk["n_effective_e"]
        rows.append(r)
    df = pd.DataFrame(rows)
    df["lvk_missing"] = df["neff_i_LVK"].isna()
    # minima under the IAS prior only, the production criterion (design note 2026-11-23 06:52)
    hi = df["neff_i_IAS"] / MIN_NEFF_I
    he = df["neff_e_IAS"] / MIN_NEFF_E
    df["head_i"], df["head_e"] = hi, he
    df["H"] = hi.clip(lower=1) * he.clip(lower=1)
    df["fails_min"] = (hi < 1) | (he < 1)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "headroom.csv", index=False, float_format="%.4g")
    q = df["H"].quantile([0, 0.25, 0.5, 0.75, 1]).round(2).tolist()
    print(f"runs {len(df)}; LVK summary missing {int(df.lvk_missing.sum())}; "
          f"H quantiles 0/25/50/75/100 %: {q}; runs with H = 1: {int((df.H == 1).sum())}; "
          f"failing a minimum already {int(df.fails_min.sum())}; stale summaries "
          f"{int(df.summary_stale.sum())}")
    print("R needed (from E1) is compared in the report.")
    print(df[df.event == "GW240525_201644"].to_string())
    figures(df)


def figures(df) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rev = HERE / "review"
    rev.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    f = df.fails_min
    ax.scatter(df.head_i[~f], df.head_e[~f], s=10, c="k", label="runs")
    ax.scatter(df.head_i[f], df.head_e[f], s=12, c="0.6", marker="x", label="fails a minimum")
    p = df[df.event == "GW240525_201644"]
    ax.scatter(p.head_i, p.head_e, s=60, marker="*", c="#d62728", label="GW240525_201644")
    ax.axvline(1, color="0.6", lw=0.8)
    ax.axhline(1, color="0.6", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n_effective_i / 150, IAS prior")
    ax.set_ylabel("n_effective_e / 100, IAS prior")
    ax.set_title("headroom per production run", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(rev / "fig1_headroom_i_e.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    f = df.fails_min
    ax.scatter(df.best[~f], df.H[~f], s=10, c="k", label="runs")
    ax.scatter(df.best[f], df.H[f], s=12, c="0.6", marker="x", label="fails a minimum")
    ax.scatter(p.best, p.H, s=60, marker="*", c="#d62728", label="GW240525_201644")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("best-fit log-likelihood (summary value)")
    ax.set_ylabel("H, row reduction allowed by the minima")
    ax.axhline(74, color="#d62728", ls=":", lw=1, label="R = 74, needed by GW240525_201644")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(rev / "fig2_H_vs_best.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
