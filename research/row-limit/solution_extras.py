#!/usr/bin/env python
"""E5 figure 4 and table: chi_eff under the LVK prior, handoff acceptance numbers and cost.

Design: research/row-limit/design_solution.md. Env o4a-pe. Reads production and rerun rundirs.
Usage: python solution_extras.py --rerun EVENT=RUNDIR [--rerun ...] --lsf EVENT=LOGFILE [...]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

O4B = Path("/home/projects/barakz/Collaboration-gw/mushkin/dot-pe/o4b-pe-paper")
sys.path.insert(0, str(O4B / "src"))
sys.path.insert(0, str(O4B / "dev/highlights"))
from compute import load_cogwheel_lvk  # noqa: E402
from reweighting_check import root_of, rundirs  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "artifacts/research/row-limit"
POSTER = "GW240525_201644"


def chieff(s):
    if "chieff" in s:
        return s["chieff"].to_numpy(float)
    return ((s.m1 * s.s1z + s.m2 * s.s2z) / (s.m1 + s.m2)).to_numpy(float)


def tail(s, w=None):
    w = np.ones(len(s)) if w is None else np.asarray(w, float)
    return float(w[chieff(s) < 0.4].sum() / w.sum())


def lsf_numbers(log: Path) -> dict:
    t = log.read_text()
    mem = re.search(r"Max Memory :\s+([\d.]+) MB", t)
    run = re.search(r"Run time :\s+([\d.]+) sec", t)
    return {"peak_mem_gb": float(mem.group(1)) / 1e3 if mem else np.nan,
            "lsf_run_h": float(run.group(1)) / 3600 if run else np.nan}


def row(label, rd: Path, lsf: dict | None = None) -> dict:
    ias = json.loads((rd / "summary_results.json").read_text())
    lvk = json.loads((rd / "summary_results_lvc.json").read_text())
    ps = pd.read_feather(rd / "prob_samples.feather", columns=["bestfit_lnlike"])
    sl = pd.read_feather(rd / "samples_lvc.feather")
    timing = rd / "timing.json"
    rr = rd / "rerun_timing.json"
    wall = (json.loads(rr.read_text())["wall_seconds"] if rr.exists()
            else json.loads(timing.read_text())["wall_seconds"]) / 3600
    r = {"run": label, "rows": len(ps), "lowest_kept_b": float(ps.bestfit_lnlike.min()),
         "best_fit_rows": float(ps.bestfit_lnlike.max()),
         "P_chieff_lt_0.4_LVK": tail(sl, sl["weights"]),
         "neff_i_IAS": ias["n_effective_i"], "neff_e_IAS": ias["n_effective_e"],
         "neff_i_LVK": lvk["n_effective_i"], "neff_e_LVK": lvk["n_effective_e"],
         "wall_h": wall}
    r.update(lsf or {})
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rerun", nargs="+", required=True)
    ap.add_argument("--lsf", nargs="*", default=[])
    ap.add_argument("--prod-lsf", nargs="*", default=[], help="EVENT=peak_GB (production)")
    a = ap.parse_args()
    reruns = dict(x.split("=", 1) for x in a.rerun)
    logs = dict(x.split("=", 1) for x in a.lsf)
    prod_mem = dict(x.split("=", 1) for x in a.prod_lsf)
    rows = []
    for ev, rd in reruns.items():
        prod, cog = rundirs(ev)
        rows.append({"event": ev, **row(f"production {prod.name}", prod,
                                        {"peak_mem_gb": float(prod_mem.get(ev, "nan"))})})
        rows.append({"event": ev, **row("rerun, row_thinning=pairs", Path(rd),
                                        lsf_numbers(Path(logs[ev])) if ev in logs else None)})
        lvk = next((root_of(ev) / "artifacts/dev/highlights_cogwheel_lvk" / ev).glob("*/*/run_*"),
                   None)
        if lvk is not None:
            cs, cw = load_cogwheel_lvk(lvk)
            rows.append({"event": ev, "run": "cogwheel, LVK prior",
                         "P_chieff_lt_0.4_LVK": tail(cs, cw)})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "solution_table.csv", index=False, float_format="%.4g")
    print(df.to_string())

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if POSTER in reruns:
        prod, _ = rundirs(POSTER)
        lvk = next((root_of(POSTER) / "artifacts/dev/highlights_cogwheel_lvk" / POSTER)
                   .glob("*/*/run_*"))
        cs, cw = load_cogwheel_lvk(lvk)
        bins = np.linspace(-0.4, 1.0, 57)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for label, s, w, kw in (
            ("cogwheel", cs, cw, dict(color="k", lw=2)),
            (f"dot-PE production {prod.name}", pd.read_feather(prod / "samples_lvc.feather"),
             None, dict(color="#d62728")),
            ("dot-PE rerun, row_thinning=pairs",
             pd.read_feather(Path(reruns[POSTER]) / "samples_lvc.feather"), None,
             dict(color="#1f77b4")),
        ):
            w = s["weights"].to_numpy(float) if w is None else np.asarray(w, float)
            ax.hist(chieff(s), bins, weights=w / w.sum(), histtype="step",
                    label=f"{label}: P(chi_eff < 0.4) = {tail(s, w):.3f}", **kw)
        ax.axvline(0.4, color="0.5", ls=":")
        ax.set_xlabel("chi_eff")
        ax.set_ylabel("posterior mass per bin, LVK prior")
        ax.set_title(POSTER, fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(HERE / "review" / "fig4_chieff_lvk_pairs.png", dpi=150)


if __name__ == "__main__":
    main()
