"""Regression tests for preselected_indices vs n_int mismatch handling."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dot_pe import inference, mp_inference


def test_serial_stage2_warns_and_uses_all_preselected_indices(tmp_path, monkeypatch):
    """Serial Stage 2 should warn but still evaluate all preselected indices."""
    banks = {"bank_a": tmp_path / "bank_a"}
    banks_dir = tmp_path / "banks_out"
    banks["bank_a"].mkdir(parents=True)
    banks_dir.mkdir(parents=True)

    captured = {}

    def _fake_collect(**kwargs):
        captured["kwargs"] = kwargs
        inds = np.asarray(kwargs["preselected_indices"], dtype=np.int_)
        lnlikes_di = np.zeros((1, len(inds)), dtype=float)
        incoherent_lnlikes = np.zeros(len(inds), dtype=float)
        return inds, lnlikes_di, incoherent_lnlikes

    monkeypatch.setattr(inference, "collect_int_samples_from_single_detectors", _fake_collect)

    with pytest.warns(UserWarning, match=r"n_int \(2\) is smaller than the number of preselected indices \(3\)"):
        candidate_inds_by_bank, _, _ = inference.select_intrinsic_samples_per_bank_incoherently(
            banks=banks,
            event_data=SimpleNamespace(detector_names=["H1"]),
            par_dic_0={},
            n_int_dict={"bank_a": 2},
            single_detector_blocksize=4,
            n_phi_incoherent=None,
            n_phi=3,
            n_t=2,
            max_incoherent_lnlike_drop=20.0,
            preselected_indices_dict={"bank_a": [1, 4, 7]},
            load_inds=False,
            inds_path_dict=None,
            banks_dir=banks_dir,
        )

    np.testing.assert_array_equal(candidate_inds_by_bank["bank_a"], np.array([1, 4, 7]))
    assert captured["kwargs"]["n_int"] == 2
    assert captured["kwargs"]["preselected_indices"] == [1, 4, 7]


def test_mp_run_warns_when_preselected_longer_than_n_int(tmp_path, monkeypatch):
    """MP run() should emit the same warning for mismatch and keep running."""
    banks = {"bank_a": tmp_path / "bank_a"}
    banks["bank_a"].mkdir(parents=True)
    banks_dir = tmp_path / "banks_out"
    banks_dir.mkdir(parents=True)
    rundir = tmp_path / "run"
    rundir.mkdir(parents=True)

    collect_call = {}

    def _fake_prepare_run_objects(**kwargs):
        return {
            "coherent_posterior": object(),
            "pr": object(),
            "banks": banks,
            "event_data": SimpleNamespace(detector_names=["H1"]),
            "par_dic_0": {},
            "fbin": np.array([1.0]),
            "approximant": "dummy",
            "m_arr": np.array([2]),
            "n_int_dict": {"bank_a": 2},
            "preselected_indices_dict": {"bank_a": [1, 4, 7]},
            "banks_dir": banks_dir,
            "rundir": rundir,
            "coherent_score_kwargs": {},
            "bank_logw_override_dict": None,
        }

    def _fake_collect_incoherent_mp(**kwargs):
        collect_call["kwargs"] = kwargs
        inds = np.asarray(kwargs["preselected_indices"], dtype=np.int_)
        lnlikes_di = np.zeros((1, len(inds)), dtype=float)
        lnlikes = np.zeros(len(inds), dtype=float)
        return inds, lnlikes_di, lnlikes

    def _fake_cross_bank(**kwargs):
        return kwargs["candidate_inds_by_bank"], {}, {}

    def _fake_draw_extrinsic_samples(**kwargs):
        return None, None, None

    def _fake_run_coherent_mp(**kwargs):
        return []

    def _fake_aggregate_and_save_results(**kwargs):
        return Path(kwargs["rundir"]) / "result.feather"

    monkeypatch.setattr(inference, "prepare_run_objects", _fake_prepare_run_objects)
    monkeypatch.setattr(mp_inference, "_collect_incoherent_mp", _fake_collect_incoherent_mp)
    monkeypatch.setattr(
        inference,
        "select_intrinsic_samples_across_banks_by_incoherent_likelihood",
        _fake_cross_bank,
    )
    monkeypatch.setattr(inference, "draw_extrinsic_samples", _fake_draw_extrinsic_samples)
    monkeypatch.setattr(mp_inference, "_run_coherent_mp", _fake_run_coherent_mp)
    monkeypatch.setattr(inference, "aggregate_and_save_results", _fake_aggregate_and_save_results)

    with pytest.warns(UserWarning, match=r"n_int \(2\) is smaller than the number of preselected indices \(3\)"):
        result = mp_inference.run(
            event="dummy_event.npz",
            bank_folder=str(banks["bank_a"]),
            n_ext=2,
            n_phi=2,
            n_t=2,
            n_int=2,
            n_workers=1,
            preselected_indices=[1, 4, 7],
        )

    assert collect_call["kwargs"]["preselected_indices"] == [1, 4, 7]
    assert result.name == "result.feather"
