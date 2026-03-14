#!/usr/bin/env python3
"""
Compare posterior samples from serial reference run vs MP runs.

Produces:
  - corner_all.pdf      : MultiCornerPlot of all runs (key GW params)
  - histograms.pdf      : 1-D marginals for every plotted param
  - ks_table.txt        : KS test p-values + JS divergences per param

Usage:
  python experiments/compare_posteriors.py
  python experiments/compare_posteriors.py --outdir /tmp/compare
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import rel_entr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent

# ── run definitions ────────────────────────────────────────────────────────────
REFERENCE = {
    "label": "serial (ref)",
    "path": Path("/home/projects/barakz/Collaboration-gw/mushkin/dot-pe/o4a-pe"
                 "/pe/artifacts/pe_runs/GW230605_065343/run_0/samples.feather"),
}

MP_RUNS = [
    {"label": "MP w4",           "path": ROOT / "artifacts/pe_real_runs/GW230605_065343/mp_w4_ext4_v4/samples.feather"},
    {"label": "MP w8",           "path": ROOT / "artifacts/pe_real_runs/GW230605_065343/mp_w8_ext8_v4/samples.feather"},
    {"label": "MP w20",          "path": ROOT / "artifacts/pe_real_runs/GW230605_065343/mp_w20_ext16_v4/samples.feather"},
    {"label": "MP w8 fixedext",  "path": ROOT / "artifacts/pe_real_runs/GW230605_065343/mp_w8_fixedext_v6/samples.feather"},
]

# Key GW parameters for the corner plot
CORNER_PARAMS = ["mchirp", "lnq", "chieff", "d_luminosity", "costheta_jn", "ra", "dec"]

# All 1-D params for histograms (superset of corner)
HIST_PARAMS = ["mchirp", "lnq", "chieff", "cumchidiff", "d_luminosity",
               "costheta_jn", "ra", "dec", "psi", "t_geocenter"]

COLORS = ["#333333", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


# ── helpers ────────────────────────────────────────────────────────────────────

def load(path: Path) -> pd.DataFrame:
    df = pd.read_feather(path)
    df["weights"] = df["weights"] / df["weights"].sum()   # normalise
    return df


def weighted_ks(x1, w1, x2, w2, n_bins=500):
    """KS statistic between two weighted 1-D distributions via CDF on a shared grid."""
    lo, hi = min(x1.min(), x2.min()), max(x1.max(), x2.max())
    grid = np.linspace(lo, hi, n_bins + 1)
    cdf1 = np.array([(w1[x1 <= g]).sum() for g in grid])
    cdf2 = np.array([(w2[x2 <= g]).sum() for g in grid])
    ks = np.abs(cdf1 - cdf2).max()
    return ks


def js_divergence(x1, w1, x2, w2, n_bins=50):
    """Jensen-Shannon divergence between two weighted 1-D distributions."""
    lo = min(np.quantile(x1, 0.001), np.quantile(x2, 0.001))
    hi = max(np.quantile(x1, 0.999), np.quantile(x2, 0.999))
    bins = np.linspace(lo, hi, n_bins + 1)
    h1, _ = np.histogram(x1, bins=bins, weights=w1, density=False)
    h2, _ = np.histogram(x2, bins=bins, weights=w2, density=False)
    h1 = h1 / h1.sum() + 1e-12
    h2 = h2 / h2.sum() + 1e-12
    m = 0.5 * (h1 + h2)
    js = 0.5 * (rel_entr(h1, m).sum() + rel_entr(h2, m).sum())
    return js


def weighted_median_std(x, w):
    idx = np.argsort(x)
    xs, ws = x[idx], w[idx]
    cdf = np.cumsum(ws)
    med = xs[np.searchsorted(cdf, 0.5)]
    mean = (xs * ws).sum()
    std = np.sqrt(((xs - mean) ** 2 * ws).sum())
    return med, std


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(ROOT / "experiments" / "posterior_comparison"))
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading samples...")
    ref_df = load(REFERENCE["path"])
    mp_dfs  = [load(r["path"]) for r in MP_RUNS]

    all_labels = [REFERENCE["label"]] + [r["label"] for r in MP_RUNS]
    all_dfs    = [ref_df] + mp_dfs

    # ── 1. Corner plot ─────────────────────────────────────────────────────────
    print("Making corner plot...")
    try:
        import cogwheel.gw_plotting as gp
        dataframes_dict = {lbl: df for lbl, df in zip(all_labels, all_dfs)}
        cp = gp.MultiCornerPlot(dataframes_dict, params=CORNER_PARAMS)
        cp.plot()
        fig = cp.corner_plots[0].fig
        fig.savefig(outdir / "corner_all.pdf", bbox_inches="tight")
        fig.savefig(outdir / "corner_all.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved corner_all.pdf/png")
    except Exception as e:
        print(f"  cogwheel corner plot failed ({e}), skipping")

    # ── 2. Histogram grid ──────────────────────────────────────────────────────
    print("Making histograms...")
    ncols = 5
    nrows = int(np.ceil(len(HIST_PARAMS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.ravel()

    for ax_idx, param in enumerate(HIST_PARAMS):
        ax = axes[ax_idx]
        for i, (lbl, df) in enumerate(zip(all_labels, all_dfs)):
            if param not in df.columns:
                continue
            x = df[param].values
            w = df["weights"].values
            lo, hi = np.quantile(x, [0.001, 0.999])
            bins = np.linspace(lo, hi, 40)
            ax.hist(x, bins=bins, weights=w, histtype="step",
                    color=COLORS[i % len(COLORS)],
                    linewidth=1.8 if i == 0 else 1.2,
                    linestyle="-" if i == 0 else ["--", ":", "-.", (0, (3,1,1,1))][min(i-1,3)],
                    label=lbl, density=True)
        ax.set_xlabel(param, fontsize=9)
        ax.set_yticks([])
        if ax_idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    for ax in axes[len(HIST_PARAMS):]:
        ax.set_visible(False)

    fig.suptitle("GW230605_065343  —  serial reference vs MP runs", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "histograms.pdf", bbox_inches="tight")
    fig.savefig(outdir / "histograms.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved histograms.pdf/png")

    # ── 3. Quantitative table ──────────────────────────────────────────────────
    print("\nQuantitative comparison vs serial reference:")
    print(f"{'param':<18}", end="")
    for r in MP_RUNS:
        short = r["label"].replace("MP ", "")
        print(f"  {'KS_'+short:<12} {'JS_'+short:<12}", end="")
    print()
    print("-" * (18 + len(MP_RUNS) * 26))

    table_rows = []
    for param in HIST_PARAMS:
        if param not in ref_df.columns:
            continue
        x_ref = ref_df[param].values
        w_ref = ref_df["weights"].values
        row = [param]
        line = f"{param:<18}"
        for mp_df in mp_dfs:
            if param not in mp_df.columns:
                line += f"  {'—':<12} {'—':<12}"
                row += [None, None]
                continue
            x_mp = mp_df[param].values
            w_mp = mp_df["weights"].values
            ks = weighted_ks(x_ref, w_ref, x_mp, w_mp)
            js = js_divergence(x_ref, w_ref, x_mp, w_mp)
            line += f"  {ks:<12.4f} {js:<12.4f}"
            row += [ks, js]
        print(line)
        table_rows.append(row)

    # Save table
    header = ["param"]
    for r in MP_RUNS:
        s = r["label"].replace("MP ", "")
        header += [f"KS_{s}", f"JS_{s}"]
    with open(outdir / "ks_table.txt", "w") as f:
        f.write("\t".join(header) + "\n")
        for row in table_rows:
            f.write("\t".join(str(x) if x is not None else "—" for x in row) + "\n")

    # ── 4. Summary statistics table ────────────────────────────────────────────
    print("\nParameter medians (serial | w4 | w8 | w20 | w8-fixedext):")
    print(f"{'param':<18}", end="")
    for lbl in all_labels:
        short = lbl.replace("serial (ref)", "ref").replace("MP ", "")
        print(f"  {short:<20}", end="")
    print()
    print("-" * (18 + len(all_dfs) * 22))
    for param in ["mchirp", "lnq", "chieff", "d_luminosity", "costheta_jn"]:
        if param not in ref_df.columns:
            continue
        line = f"{param:<18}"
        for df in all_dfs:
            x = df[param].values
            w = df["weights"].values
            med, std = weighted_median_std(x, w)
            line += f"  {med:>8.3f}±{std:<8.3f}"
        print(line)

    print(f"\nOutputs written to: {outdir}/")


if __name__ == "__main__":
    main()
