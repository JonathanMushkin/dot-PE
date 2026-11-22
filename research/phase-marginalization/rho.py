#!/usr/bin/env python
"""E4: rows per distinct (intrinsic, extrinsic) pair in T, as a function of the threshold on b.

Design: research/phase-marginalization/design.md. Reads E1's block files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
E1 = REPO / "artifacts/research/rows-above-cut"
OUT = REPO / "artifacts/research/phase-marginalization"
FLOOR = 24.35
SEED, N_BOOT = 20261122, 2000


def main() -> None:
    res = json.loads((E1 / "result.json").read_text())
    c = res["c"]
    xs = np.round(np.arange(c, c + 20.25, 0.5), 3)
    plan = pd.read_csv(E1 / "plan.csv")
    rows = np.zeros((len(plan), len(xs)))
    pairs = np.zeros((len(plan), len(xs)))
    mult_hist = np.zeros(101, dtype=np.int64)
    for g, p in enumerate(plan.itertuples()):
        d = pd.read_feather(E1 / f"blocks/{p.bank}_{p.block}.feather",
                            columns=["i", "e", "bestfit_lnlike"])
        b = d["bestfit_lnlike"].to_numpy(np.float64)
        m = b >= c
        key = d["i"].to_numpy(np.int64)[m] * 20000 + d["e"].to_numpy(np.int64)[m]
        b = b[m]
        uk, inv = np.unique(key, return_inverse=True)
        bmax_pair = np.full(len(uk), -np.inf)
        np.maximum.at(bmax_pair, inv, b)
        mult_hist += np.bincount(np.bincount(inv), minlength=101)[:101]
        bs, ps = np.sort(b), np.sort(bmax_pair)
        rows[g] = len(bs) - np.searchsorted(bs, xs)
        pairs[g] = len(ps) - np.searchsorted(ps, xs)
    banks = plan.bank.to_numpy()
    Bk = plan.groupby("bank").B_k.first().to_dict()
    idx = {b: np.flatnonzero(banks == b) for b in Bk}

    def rho(sel):
        r = sum(Bk[b] / len(sel[b]) * rows[sel[b]].sum(0) for b in Bk)
        q = sum(Bk[b] / len(sel[b]) * pairs[sel[b]].sum(0) for b in Bk)
        return r / np.maximum(q, 1)

    est = rho(idx)
    rng = np.random.default_rng(SEED)
    boot = np.array([rho({b: rng.choice(idx[b], len(idx[b])) for b in Bk})
                     for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [16, 84], axis=0)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x": xs, "rho": est, "lo": lo, "hi": hi}).to_csv(
        OUT / "rho.csv", index=False, float_format="%.4g")
    jf = int(np.argmin(np.abs(xs - FLOOR)))
    out = {"c": c, "rho_c": est[0], "rho_lo_c": lo[0], "rho_hi_c": hi[0],
           "x_floor": float(xs[jf]), "rho_floor": est[jf], "rho_lo_floor": lo[jf],
           "rho_hi_floor": hi[jf]}
    (OUT / "result.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rev = HERE / "review"
    rev.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(np.arange(1, 101), mult_hist[1:], width=1, color="0.3")
    ax.set_yscale("log")
    ax.set_xlabel("rows with b ≥ c per (intrinsic, extrinsic) pair")
    ax.set_ylabel("pairs (in T)")
    fig.tight_layout()
    fig.savefig(rev / "fig1_phases_per_pair.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    npairs = pairs.sum(0)
    ok = npairs >= 100
    ax.fill_between(xs[ok], lo[ok], hi[ok], color="0.8", label="68 % interval")
    ax.plot(xs[ok], est[ok], "k", label="ρ(x)")
    ax2 = ax.twinx()
    ax2.plot(xs, np.maximum(npairs, 0.5), color="#1f77b4", lw=0.8)
    ax2.set_yscale("log")
    ax2.set_ylabel("pairs in T with b ≥ x", color="#1f77b4")
    ax.axvline(FLOOR, color="0.3", ls="--", label=f"run_2 floor {FLOOR}")
    ax.set_xlabel("threshold x on b")
    ax.set_ylabel("rows per pair, ρ(x)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(rev / "fig2_rho.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
