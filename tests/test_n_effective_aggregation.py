"""
Regression tests for cross-bank n_effective_i / n_effective_e aggregation.

Previously the aggregate step averaged per-bank n_effective values weighted
by prior bank sizes, which could report n_effective_i < 1 (impossible for
(sum w)^2 / sum(w^2)) whenever posterior weight concentrated in a subset of
banks. The pooled statistic is computed by get_n_effective_total_i_e over
the union of samples: intrinsic identity is (bank_id, i) since the intrinsic
index restarts at 0 in every bank; extrinsic samples are shared across
banks, so extrinsic identity is e alone.
"""

import numpy as np
import pandas as pd
import pytest

from dot_pe.base_sampler_free_sampling import get_n_effective_total_i_e


def _make_samples(rows):
    """rows: list of (bank_id, i, e, weight)."""
    return pd.DataFrame(rows, columns=["bank_id", "i", "e", "weights"])


def test_single_surviving_sample_gives_n_eff_i_of_one():
    """One surviving intrinsic sample in one bank -> n_eff_i == 1, not 1/n_banks."""
    samples = _make_samples([("bank_5", 0, 0, 1.0)])
    _, n_eff_i, n_eff_e = get_n_effective_total_i_e(samples)
    assert n_eff_i == pytest.approx(1.0)
    assert n_eff_e == pytest.approx(1.0)


def test_n_eff_at_least_one_when_any_weight_nonzero():
    rng = np.random.default_rng(42)
    rows = [
        (f"bank_{k}", i, e, w)
        for k in range(3)
        for i, e, w in zip(
            rng.integers(0, 10, 20), rng.integers(0, 50, 20), rng.random(20)
        )
    ]
    n_eff, n_eff_i, n_eff_e = get_n_effective_total_i_e(_make_samples(rows))
    assert n_eff >= 1.0
    assert n_eff_i >= 1.0
    assert n_eff_e >= 1.0


def test_bank_split_is_irrelevant():
    """Splitting the same samples into more banks must not change n_eff_i /
    n_eff_e, as long as sample identities are preserved."""
    rng = np.random.default_rng(0)
    n = 100
    i_inds = np.arange(n)
    e_inds = rng.integers(0, 30, n)
    weights = rng.random(n)

    one_bank = _make_samples(
        [("bank_0", i, e, w) for i, e, w in zip(i_inds, e_inds, weights)]
    )
    # Same samples split across two banks; per-bank i indices restart at 0.
    two_banks = _make_samples(
        [(f"bank_{i % 2}", i // 2, e, w) for i, e, w in zip(i_inds, e_inds, weights)]
    )

    assert get_n_effective_total_i_e(one_bank) == pytest.approx(
        get_n_effective_total_i_e(two_banks)
    )


def test_same_i_in_different_banks_are_distinct_samples():
    """i indices are per-bank: identical i in two banks must count as two."""
    samples = _make_samples([("bank_0", 7, 0, 1.0), ("bank_1", 7, 1, 1.0)])
    _, n_eff_i, n_eff_e = get_n_effective_total_i_e(samples)
    assert n_eff_i == pytest.approx(2.0)
    assert n_eff_e == pytest.approx(2.0)


def test_shared_extrinsic_sample_across_banks_counts_once():
    """e indices are global: the same e in two banks is one sample."""
    samples = _make_samples([("bank_0", 0, 3, 1.0), ("bank_1", 0, 3, 1.0)])
    _, _, n_eff_e = get_n_effective_total_i_e(samples)
    assert n_eff_e == pytest.approx(1.0)


def test_no_bank_id_column_preserves_single_bank_behavior():
    samples = pd.DataFrame(
        {"i": [0, 0, 1, 1], "e": [0, 1, 0, 1], "weights": [0.25] * 4}
    )
    n_eff, n_eff_i, n_eff_e = get_n_effective_total_i_e(samples)
    assert n_eff == pytest.approx(4.0)
    assert n_eff_i == pytest.approx(2.0)
    assert n_eff_e == pytest.approx(2.0)


def test_empty_and_zero_weight_inputs():
    empty = pd.DataFrame(columns=["bank_id", "i", "e", "weights"])
    assert get_n_effective_total_i_e(empty) == (0, 0, 0)

    zero = _make_samples([("bank_0", 0, 0, 0.0)])
    assert get_n_effective_total_i_e(zero) == (0, 0, 0)
