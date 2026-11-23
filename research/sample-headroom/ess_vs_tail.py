#!/usr/bin/env python
"""Effective sample sizes against the posterior mass dot-PE loses below its floor, per event.

x: W_IAS, the fraction of cogwheel's posterior mass (IAS prior) whose distance-maximized
log-likelihood lies below dot-PE's floor (o4b-pe-paper dev/row_limit_check/below_floor.csv).
y: n_effective_i and n_effective_e of the production run, IAS prior (headroom.csv, E2).
Env o4a-pe; login node, seconds.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "artifacts/research/sample-headroom"
BF = Path("/home/projects/barakz/Collaboration-gw/mushkin/dot-pe/o4b-pe-paper"
          "/dev/row_limit_check/below_floor.csv")
INK, MUTED, ACCENT = "#1f2933", "#9aa5b1", "#c2410c"


def main() -> None:
    h = pd.read_csv(OUT / "headroom.csv")
    b = pd.read_csv(BF)[["event", "W_IAS", "offset"]]
    d = h.merge(b, on="event", how="inner")
    ok = d["offset"].abs() <= 2.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ax, col, lim, lab in ((axes[0], "neff_i_IAS", 150, "n_effective_i (templates)"),
                              (axes[1], "neff_e_IAS", 100, "n_effective_e (extrinsic samples)")):
        ax.scatter(d.W_IAS[ok], d[col][ok], s=14, color=INK, label="dot-PE and cogwheel best fits within 2")
        ax.scatter(d.W_IAS[~ok], d[col][~ok], s=14, facecolors="none", edgecolors=MUTED,
                   label="best fits differ by more than 2")
        p = d[d.event == "GW240525_201644"]
        ax.scatter(p.W_IAS, p[col], s=90, marker="*", color=ACCENT, label="GW240525_201644")
        ax.axhline(lim, color=MUTED, ls="--", lw=1)
        ax.text(0.99, lim * 1.06, f"minimum {lim}", ha="right", va="bottom", fontsize=8,
                color="#52606d", transform=ax.get_yaxis_transform())
        ax.set_yscale("log")
        ax.set_ylabel(f"{lab}, IAS prior")
        ax.set_xlabel("W_IAS: fraction of cogwheel's posterior mass below dot-PE's floor")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].legend(fontsize=7, loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "review" / "fig3_ess_vs_mass_below_floor.png", dpi=150)
    print(f"events {len(d)}, reliable {int(ok.sum())}")
    for lo, hi in ((0, 0.05), (0.05, 0.5), (0.5, 1.01)):
        m = ok & (d.W_IAS >= lo) & (d.W_IAS < hi)
        print(f"W_IAS {lo}-{hi}: n {int(m.sum())}, median neff_i {d.neff_i_IAS[m].median():.0f}, "
              f"median neff_e {d.neff_e_IAS[m].median():.0f}, H>=74: {int((d.H[m] >= 74).sum())}")


if __name__ == "__main__":
    main()
