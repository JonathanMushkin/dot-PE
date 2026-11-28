"""Interim P5 summary: every event whose revision-2 runs are complete, from compare_<EVENT>.json
(written by evaluate_all.py). No verdict: that waits for the user's review of the corner plots.

Writes research/acceptance/review/interim_summary.md. Light (reads JSON only).
Usage: python research/acceptance/interim_summary.py
"""

import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "artifacts/research/acceptance/ps"
INDEX = REPO / "research/acceptance/kwargs/ps/index.json"
OUT = REPO / "research/acceptance/review/interim_summary.md"
PARAMS = ("mchirp", "lnq", "chieff", "d_luminosity", "costheta_jn")


def js_max(m):
    """Largest JS divergence over PARAMS [bits], and its parameter."""
    k = max(PARAMS, key=lambda c: m.get(f"js_{c}", 0))
    return m[f"js_{k}"], k


def done(rd):
    return (rd / "samples.feather").exists()


def main():
    idx = json.load(open(INDEX))
    speed, rec, fresh, pending, prodrows = [], [], [], [], []
    for ev, v in idx.items():
        n = v["number"]
        f = RUNS / f"compare_{ev}.json"
        if not f.exists():
            runs = [f"{ev}_{s}" for s in ("fresh", "rec_s0", "rec_s1", "rec_s2", "prod")]
            pending.append(f"{n} {ev}: finished: {', '.join(r for r in runs if done(RUNS / r)) or 'none'}")
            continue
        r = json.load(open(f))
        m, sc, fr, fp = r["measures"], r["seed_scatter"], r["fresh_vs_reference"], r["fresh_vs_production"]
        speed.append(f"| {n} | {ev} | {r['wall_reference_all'][0]:.0f} | {r['wall_reference_all'][1]:.0f} | "
                     f"{r['wall_gpu']:.0f} | {r['wall_ratio']:.3f} | {'pass' if r['pass_speed'] else 'FAIL'} |")
        js, jk = js_max(m)
        thr = {c: max(0.02, 2 * sc[f"js_{c}"], 2 * m[f"js_floor_{c}"]) for c in PARAMS}
        tk = max(PARAMS, key=lambda c: m[f"js_{c}"] / thr[c])  # parameter nearest its threshold
        failed = [k for k, ok in r["checks"].items() if not ok]
        rec.append(f"| {n} | {ev} | {m['d_ln_evidence']:.4f} | {sc['d_ln_evidence']:.2g} | "
                   f"{m['ratio_n_eff_i']:.3f} | {m['ratio_n_eff_e']:.3f} | {js:.4f} ({jk}) | "
                   f"{m[f'js_{tk}']:.4f} / {thr[tk]:.4f} ({tk}) | "
                   f"{'pass' if r['pass_results'] else 'FAIL: ' + ', '.join(failed)} |")
        jr, jrk = js_max(fr)
        jp, jpk = js_max(fp)
        fresh.append(f"| {n} | {ev} | {fr['d_ln_evidence']:.3f} | {fr['ratio_n_eff_i']:.2f} | "
                     f"{fr['ratio_n_eff_e']:.2f} | {jr:.4f} ({jrk}) | {fp['d_ln_evidence']:.3f} | "
                     f"{fp['ratio_n_eff_e']:.2f} | {jp:.4f} ({jpk}) |")
        vp = r.get("vs_production")
        if vp is None:
            pending.append(f"{n} {ev}: production-recycled run (revision 3) not yet run")
        else:
            mp = vp["measures"]
            sa = json.load(open(Path(v["production_rundir"]) / "summary_results.json"))
            sb = json.load(open(RUNS / f"{ev}_prod/summary_results.json"))
            bad = [k for k, ok in vp["checks"].items() if not ok]
            prodrows.append(
                f"| {n} | {ev} | {mp['d_ln_evidence']:.5f} | {mp['ratio_n_eff_i']:.4f} | "
                f"{mp['ratio_n_eff_e']:.4f} | {js_max(mp)[0]:.4f} ({js_max(mp)[1]}) | "
                f"{sb['bestfit_lnlike_max'] - sa['bestfit_lnlike_max']:+.4f} | "
                f"{sb['lnl_marginalized_max'] - sa['lnl_marginalized_max']:+.4f} | "
                f"{'pass' if vp['pass_results'] else 'FAIL: ' + ', '.join(bad)} |")
    L = ["# P5 interim summary",
         "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `research/acceptance/interim_summary.py` from "
         "`compare_<EVENT>.json` (`evaluate_all.py`). Design: `research/acceptance/design.md`, revisions 2 "
         "and 3. Interim: no verdict until the corner plots are reviewed. Corner plots: "
         "[corners.md](corners.md).",
         "",
         "Terms:",
         "1. Reference run: the o4b CPU priority-sampling rerun of the event (7 workers on 8 cores).",
         "2. Production run: the o4a/o4b CPU production run (8 cores).",
         "3. GPU fresh run: the GPU run with nothing recycled, timed on an exclusive NVIDIA L40S with 8 cores.",
         "4. GPU recycled run: the GPU run on the reference's intrinsic selection (and its extrinsic samples "
         "where the reference recycled them), seeds + 0, + 1, + 2.",
         "5. Seed scatter: the largest |Δ ln Z| between pairs of the three GPU recycled runs.",
         "6. JS: Jensen–Shannon divergence of a weighted 1-D marginal [bits]; the largest over "
         f"{', '.join(PARAMS)} is listed.",
         "",
         "## 1. Speed (criterion: GPU fresh wall time ≤ 0.5 × the faster CPU run)",
         "",
         "| # | event | reference [s] | production [s] | GPU fresh [s] | GPU / faster CPU | speed |",
         "|---|---|---|---|---|---|---|", *speed,
         "",
         "## 2. Results: GPU recycled run (seed + 0) against the reference (criteria of revision 2)",
         "",
         "Criteria: |Δ ln Z| ≤ max(0.1, 2 × seed scatter); n_eff ratios within the seed range widened to at "
         "least [0.8, 1.25]; JS ≤ max(0.02 bits, 2 × seed JS, 2 × sampling floor).",
         "",
         "| # | event | Δ ln Z | seed scatter | n_eff_i ratio | n_eff_e ratio | largest JS [bits] "
         "| JS / threshold, nearest [bits] | results |",
         "|---|---|---|---|---|---|---|---|---|", *rec,
         "",
         "## 3. GPU fresh run against the reference and against production (reported, no criterion)",
         "",
         "Production uses another row-selection rule (bestfit), so the second comparison mixes the GPU "
         "difference with the method difference.",
         "",
         "| # | event | Δ ln Z vs ref. | n_eff_i ratio vs ref. | n_eff_e ratio vs ref. | largest JS vs ref. "
         "| Δ ln Z vs prod. | n_eff_e ratio vs prod. | largest JS vs prod. |",
         "|---|---|---|---|---|---|---|---|---|", *fresh,
         "",
         "## 4. GPU production-recycled run against production (revision 3, regime 1)",
         "",
         "Both runs use production's rule (bestfit), intrinsic selection, extrinsic samples, event file and "
         "seed, so they should agree up to numerical precision. Criteria: |Δ ln Z| ≤ 1e-3; n_eff ratios in "
         "[0.99, 1.01]; JS ≤ max(0.02 bits, 2 × sampling floor). Δ max ln L: GPU minus CPU, the largest "
         "best-fit (over phase and distance) and the largest distance-marginalized log-likelihood ratio "
         "over all (intrinsic, extrinsic) pairs.",
         "",
         "| # | event | Δ ln Z | n_eff_i ratio | n_eff_e ratio | largest JS [bits] | Δ max best-fit ln L "
         "| Δ max marginalized ln L | results |",
         "|---|---|---|---|---|---|---|---|---|", *prodrows,
         "",
         "## 5. Points to look at (written 2026-11-28 from the tables above)",
         "",
         "1. Recycled Δ ln Z is far above the seed scatter on events 3 (0.081 against 4e-8), 1 (0.030 "
         "against 3e-6) and R (0.012 against 7e-6). It passes because of the 0.1 floor, but it is a "
         "systematic GPU–CPU difference, not sampling noise. Its cause is not known.",
         "2. Event 5 has a low effective sample size (production n_eff_e 30). Its JS divergences "
         "(0.14 to 0.21 bits) are large in every comparison; the threshold there is set by the sampling "
         "floor (2 × floor = 0.30 bits on chieff). The corner plot is the better judge.",
         "3. Fresh runs draw their own intrinsic selection and extrinsic samples. Against the reference, "
         "|Δ ln Z| reaches 0.61 (event 6) and n_eff_e ratios range from 0.70 to 1.35. There is no CPU "
         "repeat-run scatter in this table to set them against.",
         "4. Section 4 fails on 4 of 7 events, and its Δ ln Z equals the recycled Δ ln Z of section 2 "
         "(event 3: 0.081 in both). Production and the reference agree with each other; the GPU differs "
         "from both by the same amount. On identical inputs the largest marginalized ln L differs by up to "
         "0.070 (event 2) and the largest best-fit ln L by up to 1.08 (event R, whose ln Z nevertheless "
         "agrees to 4e-5). This points to the GPU coherent stage, precision or the best-fit "
         "definition (not yet examined). P2's coherent-kernel tolerance (relative error of the marginalized ln L "
         "below 1e-3, i.e. up to 0.24 at ln L 240) was looser than the 1e-3 absolute ln Z criterion here.",
         "",
         "## 6. Not yet available",
         "",
         *[f"{i}. {p}" for i, p in enumerate(pending, 1)],
         ""]
    OUT.write_text("\n".join(L))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
