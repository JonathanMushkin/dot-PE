"""P5 comparison of two dot-PE rundirs (GPU against production, or two seeds).

Usage: python research/acceptance/compare.py RUNDIR_A RUNDIR_B [--json OUT]
Prints |d ln_evidence|, n_effective_i/e ratios (B / A), wall times, and per-marginal
Jensen-Shannon divergences [bits] with their estimator floor (two random halves of A).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpu-forensics"))
from repeat_scatter import PARAMS, js_bits, marginal  # noqa: E402


def wall(rd):
    rd = Path(rd)
    for f, key in (("gpu_timing.json", "wall_seconds"), ("timing.json", "wall_seconds")):
        if (rd / f).exists():
            return json.load(open(rd / f)).get(key)
    return None


def compare(a, b):
    a, b = Path(a), Path(b)
    sa_, sb_ = json.load(open(a / "summary_results.json")), json.load(open(b / "summary_results.json"))
    r = {"rundir_a": str(a), "rundir_b": str(b),
         "d_ln_evidence": abs(sa_["ln_evidence"] - sb_["ln_evidence"]),
         "ratio_n_eff_i": sb_["n_effective_i"] / sa_["n_effective_i"],
         "ratio_n_eff_e": sb_["n_effective_e"] / sa_["n_effective_e"],
         "wall_a": wall(a), "wall_b": wall(b)}
    if r["wall_a"] and r["wall_b"]:
        r["wall_ratio_b_over_a"] = r["wall_b"] / r["wall_a"]
    sa, sb = pd.read_feather(a / "samples.feather"), pd.read_feather(b / "samples.feather")
    r["n_samples_a"], r["n_samples_b"] = len(sa), len(sb)
    for c in PARAMS:
        if c in sa and c in sb:
            lo, hi = np.percentile(np.r_[sa[c], sb[c]], [0.5, 99.5])
            e = np.linspace(lo, hi, 51)
            r[f"js_{c}"] = js_bits(marginal(sa, c, e), marginal(sb, c, e))
            idx = np.random.default_rng(0).permutation(len(sa))
            r[f"js_floor_{c}"] = js_bits(marginal(sa.iloc[idx[: len(sa) // 2]], c, e),
                                         marginal(sa.iloc[idx[len(sa) // 2:]], c, e))
    return r


if __name__ == "__main__":
    res = compare(sys.argv[1], sys.argv[2])
    print(json.dumps(res, indent=1, default=float))
    if "--json" in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1, default=float)
