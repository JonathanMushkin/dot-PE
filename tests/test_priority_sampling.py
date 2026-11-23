"""Tests for dot_pe.priority_sampling (phase-marginalized priority sampling of coherent rows)."""

import numpy as np
import pandas as pd
import pytest

from dot_pe import priority_sampling


def _rows(rng, n_pairs=300, max_phases=6):
    i, e, lp = [], [], []
    for p in range(n_pairs):
        n = rng.integers(1, max_phases + 1)
        i += [p // 20] * n
        e += [p % 20] * n
        lp += list(rng.normal(0, 2, n))
    return np.array(i), np.array(e), np.array(lp)


def test_check_mode():
    mode = "phase_marginalized_priority"
    assert priority_sampling.check_mode(mode) == mode
    with pytest.raises(ValueError):
        priority_sampling.check_mode("bogus")


def test_marginalize_phase_weights_and_draw_frequencies():
    rng = np.random.default_rng(0)
    i, e, lp = _rows(rng)
    pos, ln_w = priority_sampling.marginalize_phase(i, e, lp, np.random.default_rng(1))
    key = i * 1000 + e
    assert len(pos) == len(np.unique(key))
    # ln W is the log-sum over the pair's phases
    for k_pos, lw in zip(pos[:50], ln_w[:50]):
        m = key == key[k_pos]
        assert np.isclose(lw, np.log(np.exp(lp[m]).sum()))
    # phase drawn with probability w_o / W
    m = key == key[0]
    idx = np.flatnonzero(m)
    counts = np.zeros(len(idx))
    for s in range(4000):
        pos_s, _ = priority_sampling.marginalize_phase(i, e, lp, np.random.default_rng(100 + s))
        chosen = pos_s[np.isin(pos_s, idx)]
        counts[np.searchsorted(idx, chosen[0])] += 1
    expected = np.exp(lp[idx]) / np.exp(lp[idx]).sum()
    assert np.allclose(counts / 4000, expected, atol=0.03)


def _table(rng, n=2000):
    return pd.DataFrame({
        "ln_phase_marginalized_weight": rng.normal(0, 2, n),
        "ln_posterior": rng.normal(0, 2, n),
        "priority_tau": np.full(n, -np.inf),
    })


def test_top_by_key_order_independent():
    rng = np.random.default_rng(2)
    df = _table(rng)
    df["priority_key"] = df["ln_phase_marginalized_weight"] - np.log1p(-rng.random(len(df)))
    df["id"] = np.arange(len(df))
    k = 300
    results = []
    for perm_seed in range(3):
        perm = np.random.default_rng(perm_seed).permutation(len(df))
        acc, tau = None, -np.inf
        for chunk in np.array_split(df.iloc[perm], 7):
            acc = chunk if acc is None else pd.concat([acc, chunk], ignore_index=True)
            acc, t = priority_sampling.top_by_key(acc, k)
            tau = max(tau, t)
        results.append((set(acc["id"]), tau))
    full, tau_full = priority_sampling.top_by_key(df, k)
    for ids, tau in results:
        assert ids == set(full["id"])
        assert tau == tau_full


def test_staged_estimator_unbiased():
    """Three drops (block, bank, global) with per-row tau keep sums unbiased."""
    rng = np.random.default_rng(3)
    n_blk, per = 8, 600
    lnw = rng.normal(0, 2, n_blk * per) + np.repeat(rng.normal(0, 1, n_blk), per)
    x = rng.normal(size=len(lnw))
    truth = np.sum(np.exp(lnw) * x)
    est = []
    for s in range(1500):
        r = np.random.default_rng(10_000 + s)
        frames = []
        for b in range(n_blk):
            sl = slice(b * per, (b + 1) * per)
            d = pd.DataFrame({"ln_phase_marginalized_weight": lnw[sl], "ln_posterior": lnw[sl],
                              "x": x[sl], "priority_tau": -np.inf, "bank": b % 2})
            d["priority_key"] = d["ln_phase_marginalized_weight"] - np.log1p(-r.random(per))
            d, t = priority_sampling.top_by_key(d, 200)
            frames.append(priority_sampling.raise_tau(d, t))
        banks = []
        for bank in (0, 1):
            d = pd.concat([f for f in frames if f["bank"].iloc[0] == bank], ignore_index=True)
            d, t = priority_sampling.top_by_key(d, 500)
            banks.append(priority_sampling.raise_tau(d, t))
        d, t = priority_sampling.top_by_key(pd.concat(banks, ignore_index=True), 600)
        d = priority_sampling.raise_tau(d, t)
        w = np.exp(d["ln_posterior"] + priority_sampling.ln_priority_weight(d, 0.0))
        est.append(np.sum(w * d["x"]))
    est = np.array(est)
    assert abs(est.mean() - truth) < 4 * est.std() / np.sqrt(len(est))


def test_old_tables_get_zero_correction():
    df = pd.DataFrame({"ln_posterior": [0.0, 1.0]})
    assert np.all(priority_sampling.ln_priority_weight(df, 0.0) == 0)


def test_unthinned_rows_keep_their_pair_weight():
    df = pd.DataFrame({"ln_phase_marginalized_weight": [1.0, -2.0], "ln_posterior": [0.5, -3.0],
                       "priority_tau": [-np.inf, -np.inf]})
    lt = priority_sampling.ln_priority_weight(df, np.log(10.0))
    # weight = W, i.e. ln_posterior - ln N_k + ln_priority_weight = ln_phase_marginalized_weight
    assert np.allclose(df["ln_posterior"] - np.log(10.0) + lt, df["ln_phase_marginalized_weight"])
