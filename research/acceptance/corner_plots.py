"""Per-event corner plots for P5 (design.md, revision 3, item 4), in the o4b event-gallery style.

The parameters are the gallery's (`o4a_pe_paper.results_lib.gallery.CORNER_PARAMS`): chirp mass, ln q,
chi_eff, chi_p, iota, ra, dec, luminosity distance and ln L (the log-likelihood ratio at each sample).
Styling, axis limits and the 10/50/90 % titles come from `o4a_pe_paper.results_lib.corner_style`,
imported read-only from o4a-pe-paper. Sample weights are applied.

For each event of kwargs/ps/index.json, two figures:
1. <EVENT>_vs_reference.png: reference run (o4b CPU priority-sampling rerun), recycled GPU run
   (seed + 0) and fresh GPU run;
2. <EVENT>_vs_production.png: production run, production-recycled GPU run and fresh GPU run.
Each figure carries a wall-time box: the CPU runs (cores, host where recorded) against the GPU runs
(GPU model, host and its CPU model, cores). Runs whose samples.feather is missing are left out and
listed. Written to artifacts/research/acceptance/ps/corner/.

Env: o4a-pe (cogwheel and o4a_pe_paper).
Usage: python research/acceptance/corner_plots.py [EVENT ...]   (default: all events)
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from cogwheel.gw_plotting import MultiCornerPlot  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from o4a_pe_paper.results_lib import corner_style as cs  # noqa: E402
from o4a_pe_paper.results_lib.gallery import CORNER_PARAMS, add_derived  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "artifacts/research/acceptance/ps"
LOGS = REPO / "artifacts/research/logs"
INDEX = REPO / "research/acceptance/kwargs/ps/index.json"
OUT = RUNS / "corner"
COLOR_CPU, COLOR_GPU_REC, COLOR_GPU_FRESH = cs.COLOR_DOTPE, "#9467bd", "#d62728"


def load(rd):
    f = Path(rd) / "samples.feather"
    if not f.exists():
        return None
    s = add_derived(pd.read_feather(f))
    if "weights" not in s:
        s["weights"] = 1.0
    return s


def summary(rd):
    return json.load(open(Path(rd) / "summary_results.json"))


def host_model(host):
    out = subprocess.run(["lshosts", "-w", host], capture_output=True, text=True).stdout.splitlines()
    return out[1].split()[2] if len(out) > 1 else None


def gpu_job_host(rd):
    """Host of the last successful LSF job whose log ran this rundir's kwargs file."""
    key = f"/{Path(rd).name}.json"
    best = None
    for f in LOGS.glob("p5r*/*.out"):
        t = f.read_text(errors="ignore")
        if key in t and "Total wall-clock" in t:
            hosts = re.findall(r"executed on host\(s\) <(\d+)\*(\w+)>", t)
            if hosts:
                best = (f.stat().st_mtime, hosts[-1]) if best is None or f.stat().st_mtime > best[0] else best
    return best[1] if best else (None, None)


def cpu_wall(rd):
    rd = Path(rd)
    if (rd / "rerun_info.json").exists():
        info = json.load(open(rd / "rerun_info.json"))
        n = info.get("args", {}).get("n_workers", "7")
        return info["wall_seconds"], f"8 CPU cores ({n} workers); host not recorded"
    t = json.load(open(rd / "timing.json"))
    host = t.get("host", "").split(".")[0]
    model = host_model(host) if host else None
    return t["wall_seconds"], f"{t.get('n_cores', 8)} CPU cores, {host}" + (f" ({model})" if model else "")


def gpu_wall(rd):
    f = Path(rd) / "gpu_timing.json"
    if not f.exists():
        return None, None
    t = json.load(open(f))
    n, host = gpu_job_host(rd)
    model = host_model(host) if host else None
    where = f"1x {t.get('gpu', 'GPU')}" + (f" + {n} CPU cores, {host} ({model})" if host else "")
    return t["wall_seconds"], where


def figure(ev, num, sets, timing_lines, name):
    have = [(lab, c, d) for lab, c, d in sets if d is not None]
    missing = [lab.split("\n")[0] for lab, _, d in sets if d is None]
    if len(have) < 2:
        print(f"{num} {ev} {name}: fewer than 2 runs, skipped (missing: {', '.join(missing)})", flush=True)
        return
    params = [p for p in CORNER_PARAMS if all(p in d for _, _, d in have)]
    dfs = [d[params + ["weights"]].copy() for _, _, d in have]
    mcp = MultiCornerPlot(dfs, params=params, labels=[lab for lab, _, _ in have],
                          plotstyles=[cs.plot_style(c) for _, c, _ in have], weights_col="weights")
    title = f"{num}  {ev}" + (f"   (not yet run: {', '.join(missing)})" if missing else "")
    mcp.plot(max_figsize=11, title=title)
    mcp.set_lims(**cs.combined_lims(dfs, params))
    cs._annotate_titles(mcp, dfs[0], params)
    fig = plt.gcf()
    for ax in fig.get_axes():
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    fig.legend([Line2D([], [], color=c, lw=2) for _, c, _ in have], [lab for lab, _, _ in have],
               loc="upper right", bbox_to_anchor=(0.99, 0.93), frameon=False)
    fig.text(0.47, 0.775, "\n".join(timing_lines), ha="left", va="top", family="monospace", fontsize=8.5,
             bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{ev}_{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"{num} {ev} {name}: {len(have)} runs" + (f"; missing {', '.join(missing)}" if missing else ""), flush=True)


def label(kind, rd):
    if not (Path(rd) / "summary_results.json").exists():
        return kind
    s = summary(rd)
    return (f"{kind}\n(ln Z {s['ln_evidence']:.2f}, n_eff_i {s['n_effective_i']:.0f}, "
            f"n_eff_e {s['n_effective_e']:.0f})")


def main(events=None):
    for ev, v in json.load(open(INDEX)).items():
        if events and ev not in events:
            continue
        num = v["number"]
        ref, prod = Path(v["reference_rundir"]), Path(v["production_rundir"])
        rec, fresh, prodrec = RUNS / f"{ev}_rec_s0", RUNS / f"{ev}_fresh", RUNS / f"{ev}_prod"
        w_ref, where_ref = cpu_wall(ref)
        w_prod, where_prod = cpu_wall(prod)
        w_fresh, where_fresh = gpu_wall(fresh)
        w_rec, _ = gpu_wall(rec)
        w_prodrec, _ = gpu_wall(prodrec)
        target = min(w_ref, w_prod)

        def line(name, w, extra=""):
            return f"{name:<24}{'—' if w is None else f'{w:6.0f}'}   {extra}"

        common = [f"{'Wall-clock time':<24}{'[s]':>6}   hardware",
                  line("CPU priority rerun", w_ref, where_ref),
                  line("CPU production", w_prod, where_prod),
                  line("GPU fresh (timed)", w_fresh, where_fresh or ""),
                  f"{'  GPU / faster CPU':<24}{w_fresh / target:6.3f}" if w_fresh else "  GPU / faster CPU: —"]
        figure(ev, num, [(label("CPU priority-sampling rerun (reference)", ref), COLOR_CPU, load(ref)),
                         (label("GPU, recycled reference inputs, seed + 0", rec), COLOR_GPU_REC, load(rec)),
                         (label("GPU, fresh", fresh), COLOR_GPU_FRESH, load(fresh))],
               common + [line("GPU recycled (untimed)", w_rec, "skips stages 2 and 4")], "vs_reference")
        figure(ev, num, [(label("CPU production", prod), COLOR_CPU, load(prod)),
                         (label("GPU, recycled production inputs (bestfit)", prodrec), COLOR_GPU_REC, load(prodrec)),
                         (label("GPU, fresh", fresh), COLOR_GPU_FRESH, load(fresh))],
               common + [line("GPU prod.-recycled", w_prodrec, "skips stages 2 and 4")], "vs_production")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
