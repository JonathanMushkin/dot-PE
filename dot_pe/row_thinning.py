"""
Row thinning by weight: phase-marginalized priority sampling of the coherent rows.

The default row rule of dot-PE (``row_thinning="bestfit"``) keeps the ``size_limit`` rows of
highest ``bestfit_lnlike``; when more rows pass the cut, the posterior is truncated at the
lowest kept value. ``row_thinning="pairs"`` instead

1. groups a worker call's accepted rows by (intrinsic, extrinsic) pair and sums their weights
   over the reference phase, W = sum_o exp(ln_posterior), on the scale common to all banks
   (``ln_pair_weight`` = ln W - ln N_k);
2. keeps one row per pair, its phase drawn with probability w_o / W;
3. keeps, at every drop (per i-block, per bank, across banks), the rows of largest key
   ``thin_key`` = ln_pair_weight - ln u, u ~ U(0, 1] drawn once per pair (priority sampling,
   Duffield, Lund & Thorup 2007, J. ACM 54, 32);
4. records ``thin_tau``, the largest key dropped from any group containing the row, and sets
   ``ln_thinning_weight`` = max(ln_pair_weight, thin_tau) - (ln_posterior - ln N_k).

A row's weight is then exp(ln_posterior + ln_thinning_weight) up to normalization, and sums of
these weights over kept rows are unbiased for the sums over all accepted rows (in expectation
over the phase draw for functions of phase). ``ln_posterior`` stays the row's own physical
log posterior. Validated offline in ``research/weight-thinning/`` (rule (f)).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ROW_THINNING_MODES = ("bestfit", "pairs")
THIN_COLS = ("ln_pair_weight", "thin_key", "thin_tau")


def check_mode(mode: str) -> str:
    if mode not in ROW_THINNING_MODES:
        raise ValueError(f"row_thinning must be one of {ROW_THINNING_MODES}, got {mode!r}")
    return mode


def pair_reduce(i_k, e_k, ln_post_k, rng):
    """One row per (i, e) pair, its phase drawn with probability w_o / W.

    Returns the positions (into the input arrays) of the chosen rows and ln W per chosen row,
    on the scale of ``ln_post_k``.
    """
    n = len(i_k)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    key = (np.asarray(i_k, np.int64) << 32) | np.asarray(e_k, np.int64)
    # Gumbel-max: argmax over the pair of ln w + Gumbel draws with probability w / W.
    g = ln_post_k - np.log(-np.log(1.0 - rng.random(n)))
    order = np.lexsort((g, key))
    ks = key[order]
    last = np.r_[ks[1:] != ks[:-1], True]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    lw = ln_post_k[order]
    mx = np.maximum.reduceat(lw, starts)
    sizes = np.diff(np.r_[starts, n])
    ln_w_pair = mx + np.log(np.add.reduceat(np.exp(lw - np.repeat(mx, sizes)), starts))
    return order[last], ln_w_pair


def top_by_key(df: pd.DataFrame, k: int) -> tuple[pd.DataFrame, float]:
    """Keep the k rows of largest ``thin_key``; return them and the largest dropped key."""
    if len(df) <= k:
        return df, -np.inf
    key = df["thin_key"].to_numpy()
    part = np.argpartition(key, len(df) - k)
    tau = float(key[part[: len(df) - k]].max())
    return df.iloc[part[len(df) - k:]].reset_index(drop=True), tau


def raise_tau(df: pd.DataFrame, tau: float) -> pd.DataFrame:
    if len(df) and np.isfinite(tau):
        df["thin_tau"] = np.maximum(df["thin_tau"].to_numpy(), tau)
    return df


def ln_thinning_weight(df: pd.DataFrame, ln_n_k) -> np.ndarray:
    """max(ln_pair_weight, thin_tau) - (ln_posterior - ln N_k); 0 for rows without the columns."""
    if "ln_pair_weight" not in df.columns:
        return np.zeros(len(df))
    ln_n_k = np.broadcast_to(np.asarray(ln_n_k, float), (len(df),))
    ln_hat = np.maximum(df["ln_pair_weight"].to_numpy(), df["thin_tau"].to_numpy())
    return ln_hat - (df["ln_posterior"].to_numpy() - ln_n_k)
