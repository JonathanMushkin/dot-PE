"""
Regression tests for the top-N fallback in cross-bank incoherent selection.

The cross-bank stage keeps samples within max_incoherent_lnlike_drop of the
global maximum. A single anomalously high incoherent lnlike (runaway
distance optimum) used to leave ~1 survivor, starving the coherent stage. If fewer
than min_incoherent_survivors samples pass the relative threshold, the
selection now falls back to keeping the top min_incoherent_survivors samples
by incoherent lnlike (capped at the number of candidates).
"""

import inspect
from types import SimpleNamespace

import numpy as np

from dot_pe import inference, mp_inference

EVENT_DATA = SimpleNamespace(detector_names=("H", "L"))


def _select(tmp_path, lnlikes_by_bank, **kwargs):
    banks = {bank_id: tmp_path / bank_id for bank_id in lnlikes_by_bank}
    candidate_inds = {
        bank_id: np.arange(len(lnlikes))
        for bank_id, lnlikes in lnlikes_by_bank.items()
    }
    lnlikes_di = {
        bank_id: np.vstack([lnlikes, lnlikes])
        for bank_id, lnlikes in lnlikes_by_bank.items()
    }
    return inference.select_intrinsic_samples_across_banks_by_incoherent_likelihood(
        banks=banks,
        candidate_inds_by_bank=candidate_inds,
        incoherent_lnlikes_by_bank=lnlikes_by_bank,
        lnlikes_di_by_bank=lnlikes_di,
        banks_dir=tmp_path,
        event_data=EVENT_DATA,
        **kwargs,
    )


def test_poisoned_max_triggers_topn_fallback(tmp_path):
    """One runaway lnlike >> the rest: threshold alone keeps 1 sample,
    the fallback keeps the top min_incoherent_survivors."""
    rng = np.random.default_rng(0)
    lnlikes_a = rng.normal(0, 1, 1000)
    lnlikes_b = rng.normal(0, 1, 1000)
    lnlikes_b[500] = 94254.0
    selected_inds, selected_lnlikes, selected_di = _select(
        tmp_path,
        {"bank_a": lnlikes_a, "bank_b": lnlikes_b},
        max_incoherent_lnlike_drop=20.0,
        min_incoherent_survivors=100,
    )
    n_selected = sum(len(v) for v in selected_inds.values())
    assert n_selected == 100
    # The kept samples are the top 100 of the union.
    all_lnlikes = np.concatenate([lnlikes_a, lnlikes_b])
    expected_threshold = np.sort(all_lnlikes)[-100]
    kept = np.concatenate(list(selected_lnlikes.values()))
    assert kept.min() >= expected_threshold
    # lnlikes_di stays aligned with the selection.
    for bank_id in selected_inds:
        assert selected_di[bank_id].shape == (2, len(selected_inds[bank_id]))


def test_fallback_does_not_fire_when_enough_survivors(tmp_path):
    """All lnlikes within the drop of the max: selection is unchanged."""
    lnlikes = {"bank_a": np.linspace(0, 10, 500)}
    selected_inds, _, _ = _select(
        tmp_path,
        lnlikes,
        max_incoherent_lnlike_drop=20.0,
        min_incoherent_survivors=100,
    )
    assert len(selected_inds["bank_a"]) == 500


def test_fewer_candidates_than_min_keeps_all(tmp_path):
    lnlikes = np.zeros(50)
    lnlikes[0] = 94254.0
    selected_inds, _, _ = _select(
        tmp_path,
        {"bank_a": lnlikes},
        max_incoherent_lnlike_drop=20.0,
        min_incoherent_survivors=100,
    )
    assert len(selected_inds["bank_a"]) == 50


def test_min_incoherent_survivors_exposed_in_entry_points():
    for func in (inference.run, mp_inference.run):
        assert "min_incoherent_survivors" in inspect.signature(func).parameters
