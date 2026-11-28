"""Per-event corner plots for P5 (design.md, revision 3, item 4).

For each event of kwargs/ps/index.json, two figures of the 1-D and 2-D marginals of PARAMS:
1. <EVENT>_vs_reference.pdf: reference run (o4b priority-sampling rerun), recycled GPU run
   (seed + 0) and fresh GPU run;
2. <EVENT>_vs_production.pdf: production run, production-recycled GPU run and fresh GPU run.
Runs whose samples.feather is missing are left out of the figure (and listed in its title).
Written to artifacts/research/acceptance/ps/corner/.

Usage: python research/acceptance/corner_plots.py [EVENT ...]   (default: all events)
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from cogwheel import gw_plotting  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "artifacts/research/acceptance/ps"
INDEX = REPO / "research/acceptance/kwargs/ps/index.json"
OUT = RUNS / "corner"
PARAMS = ["mchirp", "lnq", "chieff", "d_luminosity", "costheta_jn", "ra", "dec"]


def load(rd):
    f = Path(rd) / "samples.feather"
    if not f.exists():
        return None
    d = pd.read_feather(f)
    if "weights" not in d:
        d["weights"] = 1.0
    return d[PARAMS + ["weights"]]


def figure(ev, num, runs, name):
    dfs, labels, missing = [], [], []
    for label, rd in runs:
        d = load(rd)
        if d is None:
            missing.append(label)
        else:
            dfs.append(d)
            labels.append(label)
    if len(dfs) < 2:
        print(f"{num} {ev} {name}: fewer than 2 runs, skipped (missing: {', '.join(missing)})")
        return
    cp = gw_plotting.MultiCornerPlot(dfs, params=PARAMS, labels=labels)
    title = f"{num}  {ev}" + (f"  (missing: {', '.join(missing)})" if missing else "")
    cp.plot(title=title)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plt.savefig(OUT / f"{ev}_{name}.{ext}", bbox_inches="tight", dpi=110)
    plt.close("all")
    print(f"{num} {ev} {name}: {', '.join(labels)}" + (f"; missing {', '.join(missing)}" if missing else ""))


def main(events=None):
    for ev, v in json.load(open(INDEX)).items():
        if events and ev not in events:
            continue
        num = v["number"]
        figure(ev, num, [("reference (CPU priority rerun)", v["reference_rundir"]),
                         ("GPU, recycled, seed + 0", RUNS / f"{ev}_rec_s0"),
                         ("GPU, fresh", RUNS / f"{ev}_fresh")], "vs_reference")
        figure(ev, num, [("production (CPU, bestfit)", v["production_rundir"]),
                         ("GPU, production-recycled (bestfit)", RUNS / f"{ev}_prod"),
                         ("GPU, fresh", RUNS / f"{ev}_fresh")], "vs_production")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
