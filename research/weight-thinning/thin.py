#!/usr/bin/env python
"""E3: thinning rules applied offline to T, the complete rows of E1's sampled i-blocks.

Design: research/weight-thinning/design.md. Env o4a-pe (needs o4a_pe_paper.pe_lib.lvk_reweight).
Usage: python thin.py --rule a|b|c|d|e|f [--seeds 100] [--scale 1.0]
--scale multiplies the row budget: the equivalent `size_limit` is 1e7 x scale.
Writes artifacts/research/weight-thinning/<rule>.csv (one row per seed) and, for seed 0,
<rule>_seed0.npz (kept rows' b, chi_eff and weights) for the figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
E1 = REPO / "artifacts/research/rows-above-cut"
OUT = REPO / "artifacts/research/weight-thinning"
RUN = Path("/home/projects/barakz/Collaboration-gw/mushkin/dot-pe/o4b-pe-paper/"
           "artifacts/pe/dotpe/GW240525_201644/run_2")
SIZE_LIMIT = 10**7
CHI_TAIL = 0.4
QS = (0.05, 0.5, 0.95)
B_BINS = np.arange(16.25, 37.0, 0.25)
CHI_BINS = np.linspace(-1, 1, 81)


def hists(b, chi, w_lvk) -> dict:
    return {"b_counts": np.histogram(b, B_BINS)[0],
            "chi_lvk": np.histogram(chi, CHI_BINS, weights=w_lvk / w_lvk.sum())[0]}


# ── reference table T ───────────────────────────────────────────────────────

def load_T(scale: float = 1.0) -> tuple[pd.DataFrame, dict]:
    res = json.loads((E1 / "result.json").read_text())
    c = res["c"]
    plan = pd.read_csv(E1 / "plan.csv")
    kw = json.loads((RUN / "run_kwargs.json").read_text())
    summ = json.loads((RUN / "summary_results.json").read_text())
    parts = []
    for g, p in enumerate(plan.itertuples()):
        d = pd.read_feather(E1 / f"blocks/{p.bank}_{p.block}.feather",
                            columns=["i", "e", "bestfit_lnlike", "ln_posterior"])
        d = d[d["bestfit_lnlike"].to_numpy() >= c]
        d["blk"] = np.int32(g)
        d["bank"] = np.int8(int(p.bank.split("_")[1]))
        d["ln_w"] = d["ln_posterior"] - np.log(summ["per_bank_results"][p.bank]["N_k"])
        parts.append(d.drop(columns="ln_posterior"))
    T = pd.concat(parts, ignore_index=True)
    from o4a_pe_paper.pe_lib.lvk_reweight import ln_prior_ratio_fast

    for k, folder in enumerate(kw["bank_folder"]):
        m = (T["bank"] == k).to_numpy()
        bank = pd.read_feather(Path(folder) / "intrinsic_sample_bank.feather")
        ui, inv = np.unique(T.loc[m, "i"].to_numpy(), return_inverse=True)
        br = bank.iloc[ui]
        m1, m2 = br["m1"].to_numpy(), br["m2"].to_numpy()
        chi = (m1 * br["s1z"].to_numpy() + m2 * br["s2z"].to_numpy()) / (m1 + m2)
        mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        T.loc[m, "chieff"] = chi[inv]
        T.loc[m, "mchirp"] = mc[inv]
        T.loc[m, "q"] = (m2 / m1)[inv]
        T.loc[m, "ln_r_lvk"] = ln_prior_ratio_fast(br)[inv]
    blk_bank = plan.bank.map(lambda s: int(s.split("_")[1])).to_numpy()
    B_k = plan.groupby("bank").B_k.first()
    size_limit = int(round(SIZE_LIMIT * scale))
    caps = {int(b.split("_")[1]): max(50_000, 2 * size_limit // int(n)) for b, n in B_k.items()}
    k = int(round(len(T) * size_limit / res["N_hat_c"]))
    meta = {"c": c, "k": k, "caps": caps, "blk_bank": blk_bank, "n_T": len(T)}
    return T, meta


# ── staged top-k with thresholds ────────────────────────────────────────────

def top_k_stage(x: np.ndarray, idx: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """Keep the k rows of largest x among idx; return kept indices and the largest dropped x."""
    if len(idx) <= k:
        return idx, -np.inf
    part = np.argpartition(x[idx], len(idx) - k)
    return idx[part[len(idx) - k:]], float(x[idx[part[: len(idx) - k]]].max())


def staged(x: np.ndarray, T: pd.DataFrame, meta: dict, stages: str) -> tuple[np.ndarray, np.ndarray]:
    """stages 'g': one global top-k; 'bkg': block, bank, global as production. Returns kept
    indices and, per kept row, tau = the largest dropped x of any group containing it."""
    k = meta["k"]
    if stages == "g":
        kept, t = top_k_stage(x, np.arange(len(x)), k)
        return kept, np.full(len(kept), t)
    blk = T["blk"].to_numpy()
    order = np.argsort(blk, kind="stable")
    bounds = np.searchsorted(blk[order], np.arange(len(meta["blk_bank"]) + 1))
    kept_l, tau_l = [], []
    for g in range(len(meta["blk_bank"])):
        idx = order[bounds[g]:bounds[g + 1]]
        kk, t = top_k_stage(x, idx, meta["caps"][meta["blk_bank"][g]])
        kept_l.append(kk)
        tau_l.append(np.full(len(kk), t))
    kept, tau = np.concatenate(kept_l), np.concatenate(tau_l)
    bank = T["bank"].to_numpy()[kept]
    k2, t2 = [], []
    for b in np.unique(bank):
        sel = np.flatnonzero(bank == b)
        kk, t = top_k_stage(x, kept[sel], k)
        keep_pos = np.isin(kept[sel], kk)
        k2.append(kept[sel][keep_pos])
        t2.append(np.maximum(tau[sel][keep_pos], t))
    kept, tau = np.concatenate(k2), np.concatenate(t2)
    kk, t = top_k_stage(x, kept, k)
    keep_pos = np.isin(kept, kk)
    return kept[keep_pos], np.maximum(tau[keep_pos], t)


# ── measures ────────────────────────────────────────────────────────────────

def wquantile(v, w, qs):
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    cw /= cw[-1]
    return [float(v[o][np.searchsorted(cw, q)]) for q in qs]


def n_eff_i_e(T, kept, w):
    p = w / w.sum()
    key_i = T["bank"].to_numpy()[kept].astype(np.int64) * 10**8 + T["i"].to_numpy()[kept]
    p_i = np.bincount(np.unique(key_i, return_inverse=True)[1], weights=p)
    p_e = np.bincount(T["e"].to_numpy()[kept].astype(np.int64), weights=p)
    return 1 / np.sum(p_i**2), 1 / np.sum(p_e**2)


def measures(T, kept, w_ias, w_lvk, sum_w_T):
    out = {"n_kept": len(kept), "b_min": float(T["bestfit_lnlike"].to_numpy()[kept].min()),
           "evidence_ratio": float(w_ias.sum() / sum_w_T)}
    for par in ("chieff", "mchirp", "q"):
        v = T[par].to_numpy()[kept]
        for q, val in zip(QS, wquantile(v, w_ias, QS)):
            out[f"{par}_q{int(q * 100):02d}"] = val
    chi = T["chieff"].to_numpy()[kept]
    out["P_LVK"] = float(w_lvk[chi < CHI_TAIL].sum() / w_lvk.sum())
    out["neff_i"], out["neff_e"] = n_eff_i_e(T, kept, w_ias)
    return out


def apply_rule(rule, T, meta, seed, lnw, lnr, ln_def):
    if rule in ("d", "c"):
        x = T["bestfit_lnlike"].to_numpy(np.float64) if rule == "d" else lnw
        kept, _ = staged(x, T, meta, "bkg")
        w = np.exp(lnw[kept])
        return kept, w, w * np.exp(lnr[kept])
    ln_u = np.log1p(-np.random.default_rng(seed).random(len(T)))  # u in (0, 1]
    base = ln_def if rule == "e" else lnw
    kept, tau = staged(base - ln_u, T, meta, "g" if rule == "a" else "bkg")
    ln_hat = np.maximum(base[kept], tau)
    if rule == "e":
        w_ias = np.exp(ln_hat - base[kept] + lnw[kept])
        w_lvk = np.exp(ln_hat - base[kept] + lnw[kept] + lnr[kept])
    else:
        w_ias = np.exp(ln_hat)
        w_lvk = w_ias * np.exp(lnr[kept])
    return kept, w_ias, w_lvk


class Pairs:
    """Phase-marginalized pairs (bank, i, e) of T, for rule (f)."""

    def __init__(self, T: pd.DataFrame, lnw: np.ndarray):
        key = (T["bank"].to_numpy(np.int64) << 48) | (T["i"].to_numpy(np.int64) << 16) \
            | T["e"].to_numpy(np.int64)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        self.order, self.starts = order, starts
        lw = lnw[order]
        mx = np.maximum.reduceat(lw, starts)
        rep = np.repeat(mx, np.diff(np.r_[starts, len(lw)]))
        s = np.add.reduceat(np.exp(lw - rep), starts)
        self.ln_W = mx + np.log(s)
        # within-pair cumulative probability, offset by the pair's index
        pid = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(lw)]))
        c = np.exp(lw - rep) / np.repeat(s, np.diff(np.r_[starts, len(lw)]))
        cs = np.cumsum(c)
        cs -= np.repeat(cs[starts] - c[starts], np.diff(np.r_[starts, len(lw)]))
        self.cum = pid + np.minimum(cs, 1.0)
        first = order[starts]
        self.table = T.iloc[first][["blk", "bank"]].reset_index(drop=True)
        self.first = first

    def draw_rows(self, kept_pairs: np.ndarray, rng) -> np.ndarray:
        """One phase per kept pair, with probability w_o / W; returns row indices into T."""
        target = kept_pairs + rng.random(len(kept_pairs))
        pos = np.searchsorted(self.cum, target, side="left")
        pos = np.minimum(pos, np.r_[self.starts[1:], len(self.cum)][kept_pairs] - 1)
        return self.order[pos]


def apply_pairs(P: Pairs, meta, seed, lnr):
    rng = np.random.default_rng(seed)
    ln_u = np.log1p(-rng.random(len(P.ln_W)))
    kept, tau = staged(P.ln_W - ln_u, P.table, meta, "bkg")
    ln_hat = np.maximum(P.ln_W[kept], tau)
    rows = P.draw_rows(kept, rng)
    w_ias = np.exp(ln_hat)
    return rows, w_ias, w_ias * np.exp(lnr[rows])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rule", required=True, choices=list("abcdef"))
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    T, meta = load_T(a.scale)
    tag = a.rule if a.scale == 1.0 else f"{a.rule}_s{a.scale:g}"
    lnw = T["ln_w"].to_numpy()
    shift = lnw.max()
    lnw = lnw - shift  # common constant; cancels in every measure
    lnr = T["ln_r_lvk"].to_numpy()
    lnr = lnr - lnr.max()
    w_T, w_T_lvk = np.exp(lnw), np.exp(lnw + lnr)
    S, S_lvk = w_T.sum(), w_T_lvk.sum()
    ln_def = np.log((w_T / S + w_T_lvk / S_lvk) / 2)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.rule == "d" and a.scale == 1.0:  # reference values on T, written once
        ref = measures(T, np.arange(len(T)), w_T, w_T_lvk, S)
        for par in ("chieff", "mchirp", "q"):
            v = T[par].to_numpy()
            mu = np.average(v, weights=w_T)
            ref[f"{par}_std"] = float(np.sqrt(np.average((v - mu) ** 2, weights=w_T)))
        ref.update({"k": meta["k"], "n_T": meta["n_T"], "caps": meta["caps"], "c": meta["c"]})
        (OUT / "reference.json").write_text(json.dumps(ref, indent=2, default=float))
        np.savez(OUT / "T_hists.npz", **hists(T["bestfit_lnlike"].to_numpy(),
                                              T["chieff"].to_numpy(), w_T_lvk))
    seeds = [0] if a.rule in ("c", "d") else range(a.seeds)
    P = Pairs(T, lnw) if a.rule == "f" else None
    if P is not None:
        print(json.dumps({"n_pairs": len(P.ln_W), "n_rows": len(T)}), flush=True)
    rows = []
    for s in seeds:
        if P is not None:
            kept, w_ias, w_lvk = apply_pairs(P, meta, s, lnr)
        else:
            kept, w_ias, w_lvk = apply_rule(a.rule, T, meta, s, lnw, lnr, ln_def)
        r = measures(T, kept, w_ias, w_lvk, S)
        r["seed"] = s
        rows.append(r)
        if s == 0:
            np.savez(OUT / f"{tag}_seed0.npz", **hists(T["bestfit_lnlike"].to_numpy()[kept],
                                                          T["chieff"].to_numpy()[kept], w_lvk))
        print(json.dumps(r), flush=True)
    df = pd.DataFrame(rows)
    df["scale"], df["k"] = a.scale, meta["k"]
    df.to_csv(OUT / f"{tag}.csv", index=False)


if __name__ == "__main__":
    main()
