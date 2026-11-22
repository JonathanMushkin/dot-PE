#!/usr/bin/env python
"""E0 test: posterior mass below the production floor, dot-PE against cogwheel, per event.

Design: research/row-limit/design.md. The solution experiment runs this script unchanged and
passes its own dot-PE rundirs with --dotpe EVENT=RUNDIR; F and the cogwheel runs stay the
production ones.

Env: o4a-pe (imports o4b-pe-paper's dev/highlights helpers and pe_lib.lvk_reweight read-only).
Usage: python test_mass_below_floor.py [--dotpe EVENT=RUNDIR ...] [--tag NAME]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

O4B = Path("/home/projects/barakz/Collaboration-gw/mushkin/dot-pe/o4b-pe-paper")
sys.path.insert(0, str(O4B / "src"))
sys.path.insert(0, str(O4B / "dev/highlights"))
from compute import load_cogwheel, load_cogwheel_lvk  # noqa: E402
from reweighting_check import root_of, rundirs  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "artifacts/research/row-limit"
CONTROL = "GW240621_195059"
EVENTS = [CONTROL, "GW240525_201644", "GW230902_150325", "GW231113_150041"]
# Production floors F, o4b-pe-paper dev/row_limit_check/row_limit.csv (2026-11-17); fixed here so
# the solution experiment evaluates the same F.
FLOOR = {"GW240621_195059": 385.0, "GW240525_201644": 24.35, "GW230902_150325": 33.21,
         "GW231113_150041": 32.18}


def b_of(s: pd.DataFrame) -> np.ndarray:
    """Distance-maximized log-likelihood: dot-PE column, or from cogwheel's lnl and h_h."""
    if "bestfit_lnlike" in s:
        return s["bestfit_lnlike"].to_numpy(float)
    lnl, hh = s["lnl"].to_numpy(float), s["h_h"].to_numpy(float)
    dh = lnl + hh / 2
    return np.where(dh > 0, dh**2 / (2 * hh), 0.0)


def mass_below(b, w, floor) -> tuple[float, float, float]:
    w = np.asarray(w, float)
    m = float(w[b < floor].sum() / w.sum())
    n_eff = float(w.sum() ** 2 / (w**2).sum())
    return m, float(np.sqrt(m * (1 - m) / n_eff)), n_eff


def lvk_cog(event: str, cog: Path):
    lvk = next((root_of(event) / "artifacts/dev/highlights_cogwheel_lvk" / event)
               .glob("*/*/run_*"), None)
    if lvk is not None and (lvk / "samples.feather").exists():
        return (*load_cogwheel_lvk(lvk), "direct")
    return (*load_cogwheel(cog), "reweighted")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dotpe", nargs="*", default=[], help="EVENT=RUNDIR overrides")
    ap.add_argument("--tag", default="production")
    a = ap.parse_args()
    over = dict(x.split("=", 1) for x in a.dotpe)
    rows, hist = [], {}
    for ev in EVENTS:
        dot, cog = rundirs(ev)
        dot = Path(over.get(ev, dot))
        F = FLOOR[ev]
        ds = pd.read_feather(dot / "samples.feather")
        cs = pd.read_feather(cog / "samples.feather")
        cw = cs["weights"].to_numpy(float) if "weights" in cs else np.ones(len(cs))
        dl = pd.read_feather(dot / "samples_lvc.feather")
        cl, clw, src = lvk_cog(ev, cog)
        r = {"event": ev, "dotpe_rundir": str(dot), "cog_rundir": str(cog), "F": F,
             "dot_best": float(b_of(pd.read_feather(dot / "prob_samples.feather",
                                                    columns=["bestfit_lnlike"])).max()),
             "lvk_source": src}
        r["cog_best"] = float(b_of(cs).max())
        r["offset"] = r["dot_best"] - r["cog_best"]
        for prior, (sd, wd, sc, wc) in {
            "IAS": (ds, np.ones(len(ds)), cs, cw),
            "LVK": (dl, dl["weights"].to_numpy(float), cl, clw),
        }.items():
            md, ed, nd = mass_below(b_of(sd), wd, F)
            mc, ec, nc = mass_below(b_of(sc), wc, F)
            r.update({f"M_dot_{prior}": md, f"M_cog_{prior}": mc, f"neff_dot_{prior}": nd,
                      f"neff_cog_{prior}": nc, f"dW_{prior}": mc - md,
                      f"dW_err_{prior}": float(np.hypot(ed, ec))})
        hist[ev] = (b_of(ds), b_of(cs), cw, F, r["dot_best"] - 20.0)
        rows.append(r)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"mass_below_floor_{a.tag}.csv", index=False, float_format="%.5g")
    print(df[["event", "F", "dot_best", "M_dot_IAS", "M_cog_IAS", "dW_IAS", "dW_err_IAS",
              "M_dot_LVK", "M_cog_LVK", "dW_LVK", "dW_err_LVK", "offset", "lvk_source"]].to_string())
    figures(df, hist, a.tag)


def figures(df, hist, tag) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rev = HERE / "review"
    rev.mkdir(exist_ok=True)
    def panel(ax, ev):
        bd, bc, cw, F, cut = hist[ev]
        lo = min(cut, F) - 4
        hi = max(bd.max(), bc.max()) + 1
        bins = np.linspace(lo, hi, 60)
        ax.hist(bc, bins, weights=cw / cw.sum(), histtype="step", color="k", label="cogwheel")
        ax.hist(bd, bins, weights=np.full(len(bd), 1 / len(bd)), histtype="step",
                color="#d62728", label="dot-PE")
        ax.axvline(F, color="0.3", ls="--", label=f"floor F = {F}")
        ax.axvline(cut, color="0.3", ls=":", label=f"nominal cut {cut:.2f}")
        ax.set_xlabel("distance-maximized log-likelihood b")
        ax.set_title(f"{ev} ({tag})", fontsize=8)
        ax.legend(fontsize=6)

    for n, evs in ((1, EVENTS[:1]), (2, EVENTS[1:])):
        fig, axes = plt.subplots(1, len(evs), figsize=(5 * len(evs), 3.6), squeeze=False)
        for ax, ev in zip(axes[0], evs):
            panel(ax, ev)
        axes[0][0].set_ylabel("posterior mass per bin (IAS prior)")
        fig.tight_layout()
        fig.savefig(rev / f"fig{n}_b_hist_{tag}.png", dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for prior, c, dx in (("IAS", "k", 0.98), ("LVK", "#1f77b4", 1.02)):
        ax.errorbar(df.dot_best * dx, df[f"dW_{prior}"], df[f"dW_err_{prior}"], fmt="o",
                    color=c, ms=4, capsize=2, label=f"{prior} prior")
    for n, r in enumerate(df.itertuples()):
        ax.annotate(r.event, (r.dot_best, r.dW_IAS), fontsize=6,
                    xytext=(5, -12 if n % 2 else 5), textcoords="offset points")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("dot-PE best-fit log-likelihood")
    ax.set_ylabel("ΔW = M_cog(F) − M_dot(F)")
    ax.set_title(f"posterior mass below the production floor ({tag})", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(rev / f"fig3_dW_{tag}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
