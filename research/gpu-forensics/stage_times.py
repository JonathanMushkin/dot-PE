"""Per-stage wall time of production dot-PE runs, from the LSF logs (P1, measures 1-3).

Reads (never writes) o4a/o4b `artifacts/logs/dotpe_GW*/*.out` and the matching rundirs.
Writes `artifacts/research/gpu-forensics/stage_times.csv` and prints summary tables.

Usage: python research/gpu-forensics/stage_times.py
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/projects/barakz/Collaboration-gw/mushkin/dot-pe")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts/research/gpu-forensics"
STAGES = {1: "setup", 2: "incoherent", 3: "cross_bank", 4: "extrinsic", 5: "coherent",
          6: "postprocess"}
STAGE_RE = re.compile(r"Stage (\d) \([^)]*\)\s+([\d.]+) s")
TOTAL_RE = re.compile(r"Total wall-clock time: ([\d.]+) s")


def parse_log(path):
    txt = Path(path).read_text(errors="ignore")
    if "Stage timings" not in txt:
        return None
    blk = txt[txt.rindex("Stage timings"):]
    st = {STAGES[int(k)]: float(v) for k, v in STAGE_RE.findall(blk)}
    m = TOTAL_RE.search(blk)
    if not m or len(st) < 6:
        return None
    return {"total": float(m.group(1)), **st}


def rundir_index(paper):
    out = []
    for f in glob.glob(str(ROOT / paper / "artifacts/pe/dotpe/*/run_*/timing.json")):
        try:
            t = json.load(open(f))
        except Exception:
            continue
        rd = Path(f).parent
        row = {"rundir": str(rd), "event": rd.parent.name, "wall_seconds": t.get("wall_seconds"),
               "status": t.get("status"), "n_banks": t.get("n_banks"), "n_int": t.get("n_int"),
               "n_ext": t.get("n_ext"), "n_phi": t.get("n_phi"), "blocksize": t.get("blocksize"),
               "n_cores": t.get("n_cores")}
        try:
            s = json.load(open(rd / "summary_results.json"))
            row["n_i_inds_used"] = s.get("n_i_inds_used")
            row["n_dist_marg"] = s.get("n_distance_marginalizations")
        except Exception:
            pass
        out.append(row)
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for paper in ("o4a-pe-paper", "o4b-pe-paper"):
        idx = rundir_index(paper)
        for log in glob.glob(str(ROOT / paper / "artifacts/logs/dotpe_GW*/*.out")):
            p = parse_log(log)
            if p is None:
                continue
            logdir = Path(log).parent.name  # dotpe_<EVENT>[_suffix]
            m = re.match(r"dotpe_(GW\d{6}_\d{6})", logdir)
            event = m.group(1) if m else None
            r = {"paper": paper, "log": log, "logdir": logdir, "event": event, **p}
            cand = idx[(idx.event == event) & idx.wall_seconds.notna()] if len(idx) else idx
            if len(cand):
                d = (cand.wall_seconds - p["total"]).abs()
                if d.min() <= 5:
                    r.update(cand.loc[d.idxmin()].drop(["event"]).to_dict())
            rows.append(r)
    df = pd.DataFrame(rows)
    for s in STAGES.values():
        df[f"frac_{s}"] = df[s] / df["total"]
    df.to_csv(OUT / "stage_times.csv", index=False)

    matched = df[df.rundir.notna()] if "rundir" in df else df.iloc[:0]
    print(f"logs with stage timings: {len(df)}; matched to a rundir: {len(matched)}")
    q = lambda x: f"{np.median(x):.2f} [{np.percentile(x, 10):.2f}, {np.percentile(x, 90):.2f}]"
    print("\nShare of wall time, median [p10, p90], all logs:")
    for s in STAGES.values():
        print(f"  {s:12s} {q(df[f'frac_{s}'])}")
    print(f"  total h      {q(df.total / 3600)}")
    if len(matched):
        g = matched.groupby(["n_banks", "n_ext"], dropna=False)
        print("\nPer configuration (matched runs): n, median total h, median stage seconds")
        cols = ["total", "incoherent", "extrinsic", "coherent", "postprocess", "setup"]
        t = g[cols].median().round(0)
        t.insert(0, "n", g.size())
        t["total_h"] = (t.pop("total") / 3600).round(2)
        print(t.to_string())

    # Amdahl: CPU-side stages kept at production time (setup + cross_bank + extrinsic).
    print("\nAmdahl budget for the GPU stages (incoherent + coherent + postprocess), seconds:")
    med = df.loc[(df.total - df.total.median()).abs().idxmin()]
    ref = df[df.log.str.contains("dotpe_GW240525_201644/") & (df.total.round() == 21022)]
    for name, r in [("median-total run", med)] + ([("GW240525 run_2", ref.iloc[0])] if len(ref) else []):
        cpu_side = r.setup + r.cross_bank + r.extrinsic
        for f in (0.5, 0.1):
            print(f"  {name:18s} target {f:.1f}x = {f * r.total:8.0f} s; CPU-side {cpu_side:6.0f} s; "
                  f"GPU budget {f * r.total - cpu_side:8.0f} s  ({r.logdir}, total {r.total:.0f} s)")


if __name__ == "__main__":
    main()
