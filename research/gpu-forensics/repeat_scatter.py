"""Scatter between repeat production runs of the same event and settings (P1, measure 5).

Pairs: events with >= 2 `status: ok` runs with the same banks, n_int, n_ext, n_phi (from
`stage_times.csv`). Per pair: |d ln_evidence|, n_effective_i and n_effective_e ratios, and the
Jensen-Shannon divergence [bits] of 1-D marginals in samples.feather (weighted if a `weights`
column exists), on 50 bins spanning the union's 0.5-99.5 percentiles.
Writes artifacts/research/gpu-forensics/repeat_scatter.csv.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts/research/gpu-forensics"
PARAMS = ["mchirp", "lnq", "chieff", "d_luminosity", "costheta_jn"]


def marginal(df, col, edges):
    w = df["weights"].to_numpy() if "weights" in df else None
    h, _ = np.histogram(df[col].to_numpy(), bins=edges, weights=w)
    return h / h.sum()


def js_bits(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / np.where(b > 0, b, 1)), 0))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def summary(rd):
    s = json.load(open(Path(rd) / "summary_results.json"))
    return s["ln_evidence"], s["n_effective_i"], s["n_effective_e"]


def main():
    df = pd.read_csv(OUT / "stage_times.csv")
    m = df[df.rundir.notna() & (df.status == "ok")].copy()
    m["banks"] = m.rundir.map(
        lambda rd: tuple(sorted(json.load(open(Path(rd) / "run_kwargs.json"))["bank_folder"])))
    rows = []
    for key, grp in m.groupby(["event", "banks", "n_int", "n_ext", "n_phi"]):
        rds = sorted(set(grp.rundir))
        for a, b in itertools.combinations(rds, 2):
            za, ia, ea = summary(a)
            zb, ib, eb = summary(b)
            sa = pd.read_feather(Path(a) / "samples.feather")
            sb = pd.read_feather(Path(b) / "samples.feather")
            r = {"event": key[0], "n_ext": key[3], "run_a": Path(a).name, "run_b": Path(b).name,
                 "d_ln_evidence": abs(za - zb), "ratio_n_eff_i": ib / ia, "ratio_n_eff_e": eb / ea}
            for c in PARAMS:
                if c in sa and c in sb:
                    lo, hi = np.percentile(np.r_[sa[c], sb[c]], [0.5, 99.5])
                    e = np.linspace(lo, hi, 51)
                    r[f"js_{c}"] = js_bits(marginal(sa, c, e), marginal(sb, c, e))
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "repeat_scatter.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    print("\nmedian / max over pairs:")
    print(out.drop(columns=["event", "run_a", "run_b", "n_ext"]).agg(["median", "max"]).round(4).to_string())


if __name__ == "__main__":
    main()
