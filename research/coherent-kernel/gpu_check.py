"""P2 GPU check: gates 1-4 of research/coherent-kernel/design.md.

Needs the CPU reference (make_reference.py). Writes artifacts/research/coherent-kernel/gpu_check.json.
Usage (GPU job): python research/coherent-kernel/gpu_check.py [--dry-run] [--seeds 20]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

N_IBLOCKS_RUN2 = 1162
GATE_SECONDS = 1879.0


def lse(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return logsumexp(x) if len(x) else -np.inf


def gate1_lookup(cp, lookup):
    from cogwheel.likelihood import LookupTable
    lt = LookupTable()
    rng = np.random.default_rng(0)
    n = 1_000_000
    norm_h = 10 ** rng.uniform(0.5, 7.5, n)
    ov = rng.uniform(-40, 120, n)
    dh, hh = ov * norm_h, norm_h ** 2
    ref = lt.lnlike_marginalized(dh, hh)
    got = lookup.lnlike_marginalized(cp.asarray(dh, cp.float32), cp.asarray(hh, cp.float32)).get()
    err = np.abs(got.astype(float) - ref)
    # float32 inputs: compare against cogwheel at the float32-rounded inputs too
    ref32 = lt.lnlike_marginalized(dh.astype(np.float32).astype(float), hh.astype(np.float32).astype(float))
    err32 = np.abs(got.astype(float) - ref32)
    return {"max_abs_err": float(err.max()), "max_abs_err_f32_inputs": float(err32.max()),
            "p99_abs_err": float(np.percentile(err, 99)), "pass": bool(err32.max() <= 2e-3)}


def compare_bestfit(ref, got, ref_s, got_s):
    key = ["i", "e", "o"]
    m = ref.merge(got, on=key, how="outer", suffixes=("_c", "_g"), indicator=True)
    both = m[m._merge == "both"]
    only_c, only_g = m[m._merge == "left_only"], m[m._merge == "right_only"]
    cut = min(ref.bestfit_lnlike.min(), got.bestfit_lnlike.min()) if len(ref) and len(got) else np.nan
    near = lambda d: bool(len(d) == 0 or (np.abs(d[["bestfit_lnlike_c", "bestfit_lnlike_g"]].max(axis=1) - cut) <= 1e-3).all())
    r = {
        "n_cpu": len(ref), "n_gpu": len(got), "n_common": len(both),
        "frac_common": len(both) / max(1, max(len(ref), len(got))),
        "only_cpu_near_cut": near(only_c), "only_gpu_near_cut": near(only_g),
        "max_d_bestfit": float(np.abs(both.bestfit_lnlike_c - both.bestfit_lnlike_g).max()) if len(both) else 0.0,
        "max_d_lnl": float(np.abs(both.lnl_marginalized_c - both.lnl_marginalized_g).max()) if len(both) else 0.0,
        "max_rel_dh": float((np.abs(both.d_h_1Mpc_c - both.d_h_1Mpc_g) / np.abs(both.d_h_1Mpc_c).clip(1e-30)).max()) if len(both) else 0.0,
        "max_rel_hh": float((np.abs(both.h_h_1Mpc_c - both.h_h_1Mpc_g) / np.abs(both.h_h_1Mpc_c)).max()) if len(both) else 0.0,
        "rel_d_n_dist_marg": abs(ref_s["n_dist_marg"] - got_s["n_dist_marg"]) / max(1, ref_s["n_dist_marg"]),
        "d_lse_post": abs(lse(ref.ln_posterior) - lse(got.ln_posterior)),
        "d_lse_disc": abs(ref_s["logsumexp_disc"] - got_s["logsumexp_disc"]),
        "n_disc_cpu": ref_s["n_disc"], "n_disc_gpu": got_s["n_disc"],
    }
    r["pass"] = bool(r["frac_common"] >= 0.999 and r["only_cpu_near_cut"] and r["only_gpu_near_cut"]
                     and r["max_d_bestfit"] <= 1e-3 and r["max_d_lnl"] <= 2e-3
                     and r["max_rel_dh"] <= 1e-4 and r["max_rel_hh"] <= 1e-4
                     and r["rel_d_n_dist_marg"] <= 1e-4 and r["d_lse_post"] <= 1e-3
                     and r["d_lse_disc"] <= 0.5)
    return r


def block_estimate(df):
    """logsumexp over kept rows of max(ln_phase_marginalized_weight, priority_tau)."""
    return lse(np.maximum(df.ln_phase_marginalized_weight, df.priority_tau))


def phase_draw_test(cp, gpu_by_seed, setup, b, amp, phase, last_by_seed, n_pairs_test=300):
    """Mean probability of the chosen phase against its expectation sum_o p_o^2 (z-score)."""
    from dot_pe.likelihood_calculating import LikelihoodCalculator, invalid_bestfit_mask, incoherent_bound_mask
    last0 = last_by_seed[0]
    nacc = last0["pair_nacc"].get()
    lnw = last0["pair_lnw"].get()
    n_e = len(setup["log_prior_weights_e"])
    cand = np.flatnonzero(nacc >= 2)
    cand = cand[np.argsort(-lnw[cand])[:n_pairs_test]]
    thr = last0["thr"]
    lc = LikelihoodCalculator(setup["n_phi"], setup["m_arr"])
    dt = setup["dt_linfree"][b]
    h = amp * np.exp(1j * (phase + 2 * np.pi * dt[:, None, None, None] * setup["fbin"]))
    probs = {}
    for p in cand:
        i, e = divmod(int(p), n_e)
        dh, hh = lc.get_dh_hh_ieo(setup["dh_weights_dmpb"], h[i:i + 1], setup["response_dpe"][..., [e]],
                                  setup["timeshift_dbe"][..., [e]], setup["hh_weights_dmppb"], setup["asd_drift"])
        dh, hh = dh[0, 0], hh[0, 0]
        bf = np.where(invalid_bestfit_mask(dh, hh), -np.inf, 0.5 * dh ** 2 / hh * (dh > 0))
        if setup.get("incoherent_lnlike_i") is not None:
            bf = np.where(incoherent_bound_mask(bf[None, None], setup["incoherent_lnlike_i"][b[i:i + 1]])[0, 0], -np.inf, bf)
        acc = bf > thr[e // common.BLOCKSIZE]
        lnl = np.full(len(dh), -np.inf)
        lnl[acc] = lc.lookup_table.lnlike_marginalized(dh[acc], hh[acc])
        w = np.exp(lnl - lnl.max())
        probs[p] = w / w.sum()
    chosen_p, expect = [], []
    for last in last_by_seed:
        oo = last["out_o"].get()
        for p, pr in probs.items():
            chosen_p.append(pr[oo[p]])
            expect.append(np.sum(pr ** 2))
    chosen_p, expect = np.array(chosen_p), np.array(expect)
    # variance of p_chosen per draw: sum p^3 - (sum p^2)^2
    var = np.array([np.sum(pr ** 3) - np.sum(pr ** 2) ** 2 for pr in probs.values()] * len(last_by_seed))
    z = (chosen_p.sum() - expect.sum()) / np.sqrt(var.sum())
    return {"n_pairs": len(probs), "n_draws": len(chosen_p), "z": float(z), "pass": bool(abs(z) < 3.29)}


def main(dry_run=False, n_seeds=20):
    out = common.OUT
    ref_dir = out / "ref"
    blocks = {int(p.stem[5:-5]): np.load(p) for p in ref_dir.glob("block*_inds.npy")}
    print("blocks:", {k: len(v) for k, v in blocks.items()}, flush=True)
    if dry_run:
        s = common.setup_for("bestfit", False)
        print("reference files:", sorted(p.name for p in ref_dir.glob("*.json")))
        return
    import cupy as cp
    from dot_pe_gpu.coherent import CoherentGPU
    from dot_pe_gpu.lookup_table import GPULookupTable
    dev = cp.cuda.Device()
    print("device:", cp.cuda.runtime.getDeviceProperties(dev.id)["name"].decode(), flush=True)
    lookup = GPULookupTable()
    res = {"gate1_lookup": gate1_lookup(cp, lookup)}
    print("gate1", res["gate1_lookup"], flush=True)

    res["gate2_bestfit"], res["gate3_priority"], res["timing"] = {}, {}, {}
    for rs, bound in common.CONFIGS:
        s = common.setup_for(rs, bound)
        t_setup = time.perf_counter()
        g = CoherentGPU(s, blocksize_e=common.BLOCKSIZE, lookup=lookup)
        t_setup = time.perf_counter() - t_setup
        tg = common.tag(rs, bound)
        for k, b in blocks.items():
            amp = np.load(ref_dir / f"block{k}_amp.npy")
            phase = np.load(ref_dir / f"block{k}_phase.npy")
            ref = pd.read_feather(ref_dir / f"{tg}_block{k}.feather")
            ref_s = json.load(open(ref_dir / f"{tg}_block{k}.json"))
            g.run_iblock(b, amp, phase)  # warm-up (compilation)
            cp.cuda.Device().synchronize()
            g.profile, g.timings = True, {}
            reps = 3
            t0 = time.perf_counter()
            for _ in range(reps):
                df, n_disc, l1, l2, _, n_dm = g.run_iblock(b, amp, phase)
            t_blk = (time.perf_counter() - t0) / reps
            g.profile = False
            res["timing"][f"{tg}_block{k}"] = {"seconds_per_block": t_blk, "setup_seconds": t_setup,
                                              "stages": {a: v / reps for a, v in g.timings.items()},
                                              "cpu_seconds_per_block": ref_s["seconds"]}
            got_s = {"n_disc": n_disc, "logsumexp_disc": l1, "logsumsqrexp_disc": l2, "n_dist_marg": n_dm}
            if rs == "bestfit":
                r = compare_bestfit(ref, df, ref_s, got_s)
                res["gate2_bestfit"][f"{tg}_block{k}"] = r
            else:
                g.debug = True
                g.run_iblock(b, amp, phase)
                last = g.last
                n_e = g.n_e
                # 3.1: ln W on the CPU-kept pairs
                pos = {int(i): j for j, i in enumerate(b)}
                p_idx = np.array([pos[i] for i in ref.i]) * n_e + ref.e.to_numpy()
                lnw_g = last["pair_lnw"].get()[p_idx]
                d_lnw = float(np.abs(lnw_g - ref.ln_phase_marginalized_weight.to_numpy()).max())
                exact = lse(last["pair_lnw"].get())
                # 3.2: estimates over seeds
                ests, lasts = [], []
                for sd in range(n_seeds):
                    g.seed = sd + 1
                    dfs = g.run_iblock(b, amp, phase)[0]
                    ests.append(block_estimate(dfs))
                    lasts.append({"out_o": g.last["out_o"].copy(), "pair_nacc": g.last["pair_nacc"],
                                  "pair_lnw": g.last["pair_lnw"], "thr": g.last["thr"]})
                g.seed, g.debug = int(s["selection_seed"]), False
                sd_est = float(np.std(ests, ddof=1))
                est_cpu = block_estimate(ref)
                est_gpu = block_estimate(df)
                z_cpu = (est_cpu - exact) / sd_est if sd_est > 0 else 0.0
                z_gpu = (est_gpu - exact) / sd_est if sd_est > 0 else 0.0
                pdt = phase_draw_test(cp, None, s, b, amp, phase, lasts)
                r = {"n_cpu": len(ref), "n_gpu": len(df), "max_d_lnW_on_cpu_pairs": d_lnw,
                     "exact_lse": exact, "est_cpu": est_cpu, "est_gpu": est_gpu,
                     "sd_est_gpu_seeds": sd_est, "z_cpu": z_cpu, "z_gpu": z_gpu,
                     "z_mean_gpu_seeds": float((np.mean(ests) - exact) / (sd_est / np.sqrt(n_seeds))) if sd_est > 0 else 0.0,
                     "rel_d_n_dist_marg": abs(ref_s["n_dist_marg"] - n_dm) / max(1, ref_s["n_dist_marg"]),
                     "phase_draws": pdt}
                r["pass"] = bool(d_lnw <= 2e-3 and abs(z_cpu) <= 3 and abs(z_gpu) <= 3 and pdt["pass"])
                res["gate3_priority"][f"{tg}_block{k}"] = r
            print(tg, k, json.dumps({kk: vv for kk, vv in (res["gate2_bestfit"].get(f"{tg}_block{k}")
                                     or res["gate3_priority"][f"{tg}_block{k}"]).items()}, default=float),
                  "time/block", round(t_blk, 3), flush=True)
        del g
        cp.get_default_memory_pool().free_all_blocks()
    mean_t = float(np.mean([v["seconds_per_block"] for v in res["timing"].values()]))
    res["gate4_speed"] = {"mean_seconds_per_block": mean_t, "projected_run2_seconds": mean_t * N_IBLOCKS_RUN2,
                          "gate_seconds": GATE_SECONDS, "pass": bool(mean_t * N_IBLOCKS_RUN2 <= GATE_SECONDS)}
    res["pass_all"] = bool(res["gate1_lookup"]["pass"] and all(v["pass"] for v in res["gate2_bestfit"].values())
                           and all(v["pass"] for v in res["gate3_priority"].values()) and res["gate4_speed"]["pass"])
    json.dump(res, open(out / "gpu_check.json", "w"), indent=1, default=float)
    print("gate4", res["gate4_speed"], "pass_all", res["pass_all"], flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(dry_run="--dry-run" in a, n_seeds=int(a[a.index("--seeds") + 1]) if "--seeds" in a else 20)
