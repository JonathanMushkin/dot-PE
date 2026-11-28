"""Evaluate every P5 revision-2 event whose GPU runs are complete (design.md, revision 2).

For each event in kwargs/ps/index.json with <EVENT>_fresh and <EVENT>_rec_s{0,1,2} finished
(samples.feather present):
1. compare.evaluate(reference, rec_s0, seed_runs=(rec_s1, rec_s2), timed=fresh, production=prod):
   the results criteria and the speed criterion;
2. compare.measures(reference, fresh) and compare.measures(production, fresh): reported without a
   criterion;
3. revision 3: compare.evaluate(production, <EVENT>_prod) when that run is done (regime 1).
Writes artifacts/research/acceptance/ps/compare_<EVENT>.json and prints one line per event.

Usage: python research/acceptance/evaluate_all.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "artifacts/research/acceptance/ps"
INDEX = REPO / "research/acceptance/kwargs/ps/index.json"


def done(rd):
    return (rd / "samples.feather").exists() and (rd / "summary_results.json").exists()


def main():
    for ev, v in json.load(open(INDEX)).items():
        fresh, recs = RUNS / f"{ev}_fresh", [RUNS / f"{ev}_rec_s{k}" for k in range(3)]
        have = [p.name for p in [fresh, *recs] if done(p)]
        if len(have) < 4:
            print(f"{v['number']} {ev}: incomplete ({', '.join(have) or 'none'})")
            continue
        res = compare.evaluate(v["reference_rundir"], recs[0], seed_runs=recs[1:], timed=fresh,
                               production=v["production_rundir"])
        res["fresh_vs_reference"] = compare.measures(v["reference_rundir"], fresh)
        res["fresh_vs_production"] = compare.measures(v["production_rundir"], fresh)
        prod = RUNS / f"{ev}_prod"  # revision 3: production-recycled run against production (regime 1)
        res["vs_production"] = compare.evaluate(v["production_rundir"], prod) if done(prod) else None
        res["event"], res["number"] = ev, v["number"]
        json.dump(res, open(RUNS / f"compare_{ev}.json", "w"), indent=1, default=float)
        failed = [k for k, ok in res["checks"].items() if not ok]
        print(f"{v['number']} {ev}: speed {res['wall_gpu']:.0f} s / {res['wall_production']:.0f} s = "
              f"{res['wall_ratio']:.3f} ({'pass' if res['pass_speed'] else 'FAIL'}); results "
              f"{'pass' if res['pass_results'] else 'FAIL ' + ','.join(failed)}; "
              f"dlnZ {res['measures']['d_ln_evidence']:.3g} (scatter {res['seed_scatter']['d_ln_evidence']:.3g})")
        vp = res["vs_production"]
        if vp is None:
            print("    vs production: production-recycled run missing")
        else:
            bad = [k for k, ok in vp.get("checks", {}).items() if not ok]
            print(f"    vs production: regime {vp['regime']}; "
                  f"{'pass' if vp['pass_results'] else 'FAIL ' + ','.join(bad)}; "
                  f"dlnZ {vp['measures']['d_ln_evidence']:.3g}")


if __name__ == "__main__":
    main()
