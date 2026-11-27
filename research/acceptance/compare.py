"""P5 comparison and pass/fail (criteria of research/acceptance/design.md, coded here).

Usage:
  python research/acceptance/compare.py PROD_RUNDIR GPU_RUNDIR [--seed-runs RD1 RD2] [--json OUT]
--seed-runs: the extra-seed GPU runs of a regime-2 event; seed scatter = largest value over the
three pairs among (GPU_RUNDIR, RD1, RD2).
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PARAMS = ["mchirp", "lnq", "chieff", "d_luminosity", "costheta_jn"]
N_BINS, N_BOOT = 50, 20


def _hist(x, edges):
    h, _ = np.histogram(x, bins=edges)
    tot = h.sum()
    return h / tot if tot > 0 else None


def js_bits(p, q):
    if p is None or q is None:
        return np.nan
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / np.where(b > 0, b, 1)), 0))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def js_pair(xa, xb):
    """JS [bits] of two sample sets and its floor (median over N_BOOT bootstrap pairs of A)."""
    xa, xb = xa[np.isfinite(xa)], xb[np.isfinite(xb)]
    if len(xa) < 2 or len(xb) < 2:
        return np.nan, np.nan
    lo, hi = np.percentile(np.r_[xa, xb], [0.5, 99.5])
    if not hi > lo:
        return 0.0, 0.0
    e = np.linspace(lo, hi, N_BINS + 1)
    js = js_bits(_hist(xa, e), _hist(xb, e))
    rng = np.random.default_rng(0)
    floor = np.median([js_bits(_hist(rng.choice(xa, len(xa)), e), _hist(rng.choice(xa, len(xa)), e))
                       for _ in range(N_BOOT)])
    return float(js), float(floor)


def wall(rd):
    for f in ("gpu_timing.json", "timing.json"):
        if (Path(rd) / f).exists():
            return json.load(open(Path(rd) / f)).get("wall_seconds")
    return None


def measures(a, b):
    a, b = Path(a), Path(b)
    sa_, sb_ = (json.load(open(x / "summary_results.json")) for x in (a, b))
    r = {"d_ln_evidence": abs(sa_["ln_evidence"] - sb_["ln_evidence"]),
         "ratio_n_eff_i": sb_["n_effective_i"] / sa_["n_effective_i"],
         "ratio_n_eff_e": sb_["n_effective_e"] / sa_["n_effective_e"]}
    sa, sb = pd.read_feather(a / "samples.feather"), pd.read_feather(b / "samples.feather")
    r["n_samples_a"], r["n_samples_b"] = len(sa), len(sb)
    for c in PARAMS:
        if c in sa and c in sb:
            r[f"js_{c}"], r[f"js_floor_{c}"] = js_pair(sa[c].to_numpy(float), sb[c].to_numpy(float))
    return r


def same_extrinsic(a, b):
    ea, eb = (pd.read_feather(Path(x) / "extrinsic_samples.feather") for x in (a, b))
    if ea.shape != eb.shape:
        return False
    cols = [c for c in ea.columns if c in eb.columns]
    return bool(np.allclose(ea[cols].to_numpy(float), eb[cols].to_numpy(float), rtol=1e-12, atol=0, equal_nan=True))


def evaluate(prod, gpu, seed_runs=()):
    m = measures(prod, gpu)
    res = {"production": str(prod), "gpu": str(gpu), "measures": m,
           "wall_production": wall(prod), "wall_gpu": wall(gpu)}
    res["wall_ratio"] = res["wall_gpu"] / res["wall_production"] if res["wall_gpu"] and res["wall_production"] else None
    res["pass_speed"] = bool(res["wall_ratio"] is not None and res["wall_ratio"] <= 0.5)
    regime = 1 if same_extrinsic(prod, gpu) else 2
    res["regime"] = regime
    checks = {}
    if regime == 1:
        checks["d_ln_evidence"] = m["d_ln_evidence"] <= 1e-3
        for k in ("ratio_n_eff_i", "ratio_n_eff_e"):
            checks[k] = 0.99 <= m[k] <= 1.01
        for c in PARAMS:
            if f"js_{c}" in m:
                checks[f"js_{c}"] = m[f"js_{c}"] <= max(0.02, 2 * m[f"js_floor_{c}"])
    else:
        if len(seed_runs) < 2:
            res["pass_results"] = None
            res["note"] = "regime 2: seed runs needed (--seed-runs RD1 RD2)"
            return res
        runs = [gpu, *seed_runs]
        sc = [measures(x, y) for x, y in itertools.combinations(runs, 2)]
        scatter = {"d_ln_evidence": max(s["d_ln_evidence"] for s in sc)}
        for k in ("ratio_n_eff_i", "ratio_n_eff_e"):
            vals = [s[k] for s in sc] + [1 / s[k] for s in sc]
            scatter[k] = (min(vals) / 1.25, max(vals) * 1.25)
        for c in PARAMS:
            scatter[f"js_{c}"] = max(s.get(f"js_{c}", 0) for s in sc)
        res["seed_scatter"] = scatter
        checks["d_ln_evidence"] = m["d_ln_evidence"] <= max(0.1, 2 * scatter["d_ln_evidence"])
        for k in ("ratio_n_eff_i", "ratio_n_eff_e"):
            lo, hi = min(0.8, scatter[k][0]), max(1.25, scatter[k][1])
            checks[k] = lo <= m[k] <= hi
        for c in PARAMS:
            if f"js_{c}" in m:
                checks[f"js_{c}"] = m[f"js_{c}"] <= max(0.02, 2 * scatter[f"js_{c}"], 2 * m[f"js_floor_{c}"])
    res["checks"] = {k: bool(v) for k, v in checks.items()}
    res["pass_results"] = bool(all(checks.values()))
    return res


if __name__ == "__main__":
    a = sys.argv
    seeds = a[a.index("--seed-runs") + 1:a.index("--seed-runs") + 3] if "--seed-runs" in a else ()
    out = evaluate(a[1], a[2], seeds)
    print(json.dumps(out, indent=1, default=float))
    if "--json" in a:
        json.dump(out, open(a[a.index("--json") + 1], "w"), indent=1, default=float)
