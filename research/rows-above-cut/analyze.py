#!/usr/bin/env python
"""E1 analysis: N̂(x), rows of GW240525_201644 run_2 with b >= x, from the sampled i-blocks.

Design: research/rows-above-cut/design.md (c = 16.379, design note of 2026-11-22 10:27).
Env dot-pe or o4a-pe; reads the harness outputs; a few minutes on one core (LSF).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "artifacts/research/rows-above-cut"
BEST_ROWS = 36.379370782586555  # max bestfit_lnlike over run_2's prob_samples.feather
C = BEST_ROWS - 20.0
FLOOR = 24.35
SIZE_LIMIT = 1e7
XS = np.round(np.arange(C, BEST_ROWS + 0.25, 0.25), 3)
SEED, N_BOOT = 20261122, 2000


def main() -> None:
    plan = pd.read_csv(OUT / "plan.csv")
    counts = pd.concat([pd.read_csv(f) for f in sorted((OUT / "counts").glob("pos_*.csv"))])
    assert len(counts) == len(plan), f"{len(counts)} of {len(plan)} blocks done"
    assert counts.validation_ok.all(), counts[~counts.validation_ok]
    # N_j(x) per block on the x grid
    nj = np.zeros((len(counts), len(XS)), dtype=np.int64)
    bmax = np.zeros(len(counts))
    for r, c in enumerate(counts.itertuples()):
        b = pd.read_feather(OUT / f"blocks/{c.bank}_{c.block}.feather",
                           columns=["bestfit_lnlike"])["bestfit_lnlike"].to_numpy(np.float64)
        b = np.sort(b)
        nj[r] = len(b) - np.searchsorted(b, XS, side="left")
        bmax[r] = b[-1] if len(b) else np.nan
    counts = counts.reset_index(drop=True)
    counts["N_c"], counts["b_max_rows"] = nj[:, 0], bmax
    banks = sorted(counts.bank.unique())
    idx = {b: np.flatnonzero(counts.bank.to_numpy() == b) for b in banks}
    Bk = {b: int(counts.B_k[idx[b][0]]) for b in banks}

    def est(sel: dict) -> np.ndarray:
        return sum(Bk[b] / len(sel[b]) * nj[sel[b]].sum(axis=0) for b in banks)

    n_hat = est(idx)
    rng = np.random.default_rng(SEED)
    boot = np.array([est({b: rng.choice(idx[b], len(idx[b])) for b in banks})
                     for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [16, 84], axis=0)
    df = pd.DataFrame({"x": XS, "N_hat": n_hat, "N_lo": lo, "N_hi": hi})
    df.to_csv(OUT / "n_hat.csv", index=False, float_format="%.5g")
    counts.to_csv(OUT / "per_block.csv", index=False)
    j_floor = int(np.argmin(np.abs(XS - FLOOR)))
    res = {"c": C, "N_hat_c": n_hat[0], "N_lo_c": lo[0], "N_hi_c": hi[0],
           "R_c": n_hat[0] / SIZE_LIMIT, "x_near_floor": float(XS[j_floor]),
           "N_hat_floor": n_hat[j_floor], "N_lo_floor": lo[j_floor], "N_hi_floor": hi[j_floor],
           "per_bank_N_hat_c": {b: Bk[b] / len(idx[b]) * float(nj[idx[b], 0].sum())
                                for b in banks},
           "n_blocks": len(counts), "rows_in_T": int(nj[:, 0].sum()),
           "wall_s_median": float(counts.wall_s.median()),
           "maxrss_gb_max": float(counts.maxrss_gb.max())}
    (OUT / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    figures(counts, nj, df, bmax)


def figures(counts, nj, df, bmax) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rev = HERE / "review"
    rev.mkdir(exist_ok=True)
    c0 = counts.iloc[0]
    b = pd.read_feather(OUT / f"blocks/{c0.bank}_{c0.block}.feather",
                        columns=["bestfit_lnlike"])["bestfit_lnlike"].to_numpy()
    b = b[b >= C]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(b, np.arange(C, b.max() + 0.25, 0.25), histtype="step", color="k")
    ax.axvline(C, color="0.3", ls=":", label=f"nominal cut c = {C:.3f}")
    ax.axvline(FLOOR, color="0.3", ls="--", label=f"run_2 floor {FLOOR}")
    ax.set_yscale("log")
    ax.set_xlabel("distance-maximized log-likelihood b of a row")
    ax.set_ylabel("rows per 0.25 bin")
    ax.set_title(f"one i-block: {c0.bank} block {c0.block} (512 templates)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(rev / "fig1_block_hist.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for bank, col in zip(sorted(counts.bank.unique()), ("k", "#1f77b4", "#d62728")):
        m = counts.bank == bank
        ax.scatter(bmax[m], np.maximum(nj[m.to_numpy(), 0], 0.5), s=12, c=col, label=bank)
    ax.set_yscale("log")
    ax.set_xlabel("largest b in the block")
    ax.set_ylabel("N_j(c), rows with b ≥ c (0 shown at 0.5)")
    ax.set_title("sampled i-blocks", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(rev / "fig2_block_counts.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.fill_between(df.x, df.N_lo, df.N_hi, color="0.8", label="68 % interval")
    ax.plot(df.x, df.N_hat, "k", label="N̂(x)")
    ax.axhline(SIZE_LIMIT, color="#d62728", lw=1, label="size_limit 1e7")
    ax.axvline(FLOOR, color="0.3", ls="--", label=f"run_2 floor {FLOOR}")
    ax.axvline(C, color="0.3", ls=":", label=f"nominal cut c = {C:.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("threshold x on b")
    ax.set_ylabel("rows of run_2 with b ≥ x")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(rev / "fig3_n_hat.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
