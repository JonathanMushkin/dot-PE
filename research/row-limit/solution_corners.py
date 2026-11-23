#!/usr/bin/env python
"""E5 corner plots: production dot-PE, the rerun with row_selection="phase_marginalized_priority"
and cogwheel, overlaid, under the IAS prior and under the LVK prior.

Style and parameters from o4b-pe-paper's event gallery (results_lib.gallery.CORNER_PARAMS,
corner_style.plot_style / combined_lims / weighted_quantiles), plus bestfit_lnlike (b), the
distance-maximized log-likelihood that the floor is defined on. Diagonal titles: the rerun's
median and its 10th and 90th percentiles. Colours: production tab:blue (gallery's dot-PE colour),
cogwheel tab:orange (gallery's), rerun tab:red.
Env o4a-pe. Usage: python solution_corners.py --rerun EVENT=RUNDIR [...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

O4B = Path("/home/projects/barakz/Collaboration-gw/mushkin/dot-pe/o4b-pe-paper")
sys.path.insert(0, str(O4B / "src"))
sys.path.insert(0, str(O4B / "dev/highlights"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from o4a_pe_paper.results_lib import corner_style as cs  # noqa: E402
from o4a_pe_paper.results_lib.gallery import CORNER_PARAMS, add_derived  # noqa: E402
from reweighting_check import rundirs  # noqa: E402
from test_mass_below_floor import FLOOR, b_of, lvk_cog  # noqa: E402

HERE = Path(__file__).resolve().parent
PARAMS = (*CORNER_PARAMS, "bestfit_lnlike")
RED = "#d62728"


def prep(s: pd.DataFrame, w=None) -> pd.DataFrame:
    s = add_derived(s)
    s["bestfit_lnlike"] = b_of(s)
    if w is not None:
        s["weights"] = np.asarray(w, float)
    return s


def kish(w) -> float:
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w**2).sum())


def figure(event: str, prior: str, sets: list[tuple[str, str, pd.DataFrame]], out: Path) -> None:
    from cogwheel.gw_plotting import MultiCornerPlot

    params = [p for p in PARAMS if all(p in d.columns for _, _, d in sets)]
    dfs = []
    for _, _, d in sets:
        df = d[params].copy()
        df["weights"] = d["weights"] if "weights" in d else 1.0
        dfs.append(df)
    mcp = MultiCornerPlot(dfs, params=params, labels=[lab for lab, _, _ in sets],
                          plotstyles=[cs.plot_style(c) for _, c, _ in sets], weights_col="weights")
    mcp.plot(max_figsize=8, title=f"{event}, {prior} prior")
    mcp.set_lims(**cs.combined_lims(dfs, params))
    cs._annotate_titles(mcp, dfs[1], params)  # rerun's quantiles
    fig = plt.gcf()
    # production floor on the b panels
    axes = mcp.corner_plots[0].axes if hasattr(mcp, "corner_plots") else mcp.axes
    j = params.index("bestfit_lnlike")
    for ax in axes[j, : j + 1]:
        ax.axvline(FLOOR[event], color="k", ls=":", lw=1) if ax is axes[j, j] else None
    for ax in axes[j, :j]:
        ax.axhline(FLOOR[event], color="k", ls=":", lw=1)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rerun", nargs="+", required=True)
    a = ap.parse_args()
    for ev, rd in (x.split("=", 1) for x in a.rerun):
        rd = Path(rd)
        prod, cog = rundirs(ev)
        p_ias = prep(pd.read_feather(prod / "samples.feather"))
        r_ias = prep(pd.read_feather(rd / "samples.feather"))
        c_ias = prep(pd.read_feather(cog / "samples.feather"))
        p_lvk = prep(pd.read_feather(prod / "samples_lvc.feather"))
        r_lvk = prep(pd.read_feather(rd / "samples_lvc.feather"))
        c_s, c_w, src = lvk_cog(ev, cog)
        c_lvk = prep(c_s, c_w)
        for prior, (p, r, c), cog_note in (
            ("IAS", (p_ias, r_ias, c_ias), "cogwheel"),
            ("LVK", (p_lvk, r_lvk, c_lvk),
             "cogwheel, LVK-prior run" if src == "direct" else "cogwheel, reweighted to LVK prior"),
        ):
            sets = [
                (f"dot-PE production {prod.name} (sample ESS {kish(p['weights']):.0f})", cs.COLOR_DOTPE, p),
                (f"dot-PE rerun, phase marginalization + priority sampling "
                 f"(sample ESS {kish(r['weights']):.0f})", RED, r),
                (f"{cog_note} {cog.name} (sample ESS {kish(c['weights']) if 'weights' in c else len(c):.0f})",
                 cs.COLOR_COGWHEEL, c),
            ]
            figure(ev, prior, sets, HERE / "review" / f"fig5_corner_{ev}_{prior}.png")


if __name__ == "__main__":
    main()
