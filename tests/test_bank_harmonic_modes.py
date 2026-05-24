"""Tests for bank harmonic_modes and backward-compatible m_arr fallback."""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

from cogwheel.waveform import WaveformGenerator

from dot_pe import config, waveform_banks
from dot_pe.utils import (
    normalize_harmonic_modes,
    resolve_bank_modes,
    waveform_generator_from_config,
)


def test_resolve_explicit_harmonic_modes_22_only():
    bank_config = {
        "approximant": "IMRPhenomXODE",
        "harmonic_modes": [[2, 2]],
    }
    harmonic_modes, m_arr = resolve_bank_modes(bank_config)
    assert harmonic_modes == [(2, 2)]
    assert list(m_arr) == [2]


def test_resolve_legacy_m_arr_only_unchanged():
    legacy_m = [2, 1, 3, 4]
    bank_config = {
        "approximant": "IMRPhenomXPHM",
        "m_arr": legacy_m,
    }
    harmonic_modes, m_arr = resolve_bank_modes(bank_config)
    assert list(m_arr) == legacy_m
    assert "harmonic_modes" not in bank_config
    allowed = set(legacy_m)
    for l, m in harmonic_modes:
        assert m in allowed
    assert (3, 3) in harmonic_modes
    assert (3, 2) in harmonic_modes


def test_legacy_harmonic_modes_match_wfg_m_arr():
    bank_config = {
        "approximant": "IMRPhenomXPHM",
        "m_arr": [2, 1, 3, 4],
    }
    harmonic_modes, m_arr = resolve_bank_modes(bank_config)
    from cogwheel.data import EventData

    ed = EventData.gaussian_noise("", **config.EVENT_DATA_KWARGS)
    wfg = WaveformGenerator.from_event_data(
        ed, bank_config["approximant"], harmonic_modes=harmonic_modes
    )
    assert np.array_equal(wfg.m_arr, m_arr)


def test_both_fields_must_be_consistent():
    bank_config = {
        "approximant": "IMRPhenomXODE",
        "harmonic_modes": [[2, 2]],
        "m_arr": [2, 1],
    }
    with pytest.raises(ValueError, match="does not match"):
        resolve_bank_modes(bank_config)


def test_normalize_harmonic_modes():
    assert normalize_harmonic_modes([[2, 2], (2, 1)]) == [(2, 2), (2, 1)]


def _minimal_bank_samples(n=8):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "m1": 30.0 + rng.normal(size=n),
            "m2": 25.0 + rng.normal(size=n),
            "s1z": rng.normal(size=n) * 0.1,
            "s1x_n": np.zeros(n),
            "s1y_n": np.zeros(n),
            "s2z": rng.normal(size=n) * 0.1,
            "s2x_n": np.zeros(n),
            "s2y_n": np.zeros(n),
            "iota": np.pi / 3 + rng.normal(size=n) * 0.05,
            "log_prior_weights": np.zeros(n),
        }
    )


@pytest.mark.integration
def test_single_mode_bank_waveforms_and_sdp(tmp_path):
    """Tiny (2,2)-only bank: waveform shape and SDP weights match one mode."""
    pytest.importorskip("lalsimulation")

    bank_dir = tmp_path / "bank_22"
    bank_dir.mkdir()
    n = 8
    blocksize = 8
    samples_path = bank_dir / "intrinsic_sample_bank.feather"
    _minimal_bank_samples(n).to_feather(samples_path)

    bank_config = {
        "fbin": config.DEFAULT_FBIN.tolist(),
        "f_ref": 50.0,
        "approximant": "IMRPhenomXODE",
        "harmonic_modes": [[2, 2]],
    }
    bank_config_path = bank_dir / "bank_config.json"
    with open(bank_config_path, "w") as f:
        json.dump(bank_config, f)

    waveform_dir = bank_dir / "waveforms"
    waveform_banks.create_waveform_bank_from_samples(
        samples_path=samples_path,
        bank_config_path=bank_config_path,
        waveform_dir=waveform_dir,
        n_pool=1,
        blocksize=blocksize,
        approximant="IMRPhenomXODE",
    )

    with open(bank_config_path) as f:
        saved = json.load(f)
    assert "harmonic_modes" in saved
    assert saved["harmonic_modes"] == [[2, 2]]
    assert saved["m_arr"] == [2]

    amp = np.load(waveform_dir / "amplitudes_block_0.npy")
    assert amp.shape[1] == 1

    from cogwheel.data import EventData
    from cogwheel.posterior import Posterior
    from dot_pe.inference import _create_single_detector_processor

    event_data = EventData.gaussian_noise("", **config.EVENT_DATA_KWARGS)
    posterior = Posterior.from_event(
        event=event_data,
        mchirp_guess=25.0,
        likelihood_kwargs={
            "fbin": np.array(saved["fbin"]),
            "pn_phase_tol": None,
        },
        approximant="IMRPhenomXODE",
        prior_class="CartesianIASPrior",
    )
    par_dic_0 = posterior.likelihood.par_dic_0.copy()
    _, m_arr = resolve_bank_modes(saved)
    sdp = _create_single_detector_processor(
        event_data,
        "H",
        par_dic_0,
        bank_dir,
        np.array(saved["fbin"]),
        "IMRPhenomXODE",
        16,
        m_arr,
        blocksize,
        10**6,
    )
    assert sdp.dh_weights_dmpb.shape[1] == 1
    assert len(sdp.likelihood_calculator.m_arr) == 1


def test_waveform_generator_from_config_matches_cogwheel():
    from dot_pe.utils import default_harmonic_modes_for_approximant

    bank_config = {
        "approximant": "IMRPhenomXPHM",
        "m_arr": [2, 1, 3, 4],
    }
    from cogwheel.data import EventData

    ed = EventData.gaussian_noise("", **config.EVENT_DATA_KWARGS)
    wfg = waveform_generator_from_config(ed, bank_config)
    default_modes = [
        m
        for m in default_harmonic_modes_for_approximant("IMRPhenomXPHM")
        if m[1] in {2, 1, 3, 4}
    ]
    assert list(wfg.harmonic_modes) == default_modes


def test_legacy_xode_m_arr_fallback():
    from dot_pe.utils import default_harmonic_modes_for_approximant

    bank_config = {"approximant": "IMRPhenomXODE", "m_arr": [2, 1, 3, 4]}
    harmonic_modes, m_arr = resolve_bank_modes(bank_config)
    assert list(m_arr) == [2, 1, 3, 4]
    allowed = {2, 1, 3, 4}
    for mode in harmonic_modes:
        assert mode[1] in allowed
    xode_modes = default_harmonic_modes_for_approximant("IMRPhenomXODE")
    n_xode = len([m for m in xode_modes if m[1] in allowed])
    assert len(harmonic_modes) == n_xode


# --- Fast inference smoke tests (small bank + small blocks + capped n_int) ---

SMALL_BANK_SIZE = 64
SMALL_BLOCKSIZE = 64
SMALL_N_INT = 32
SMALL_N_EXT = 4
SMALL_N_PHI = 8
SMALL_N_T = 16
SMALL_APPROXIMANT = "IMRPhenomXODE"


def _create_small_bank(
    tmp_path: Path,
    name: str,
    *,
    legacy_m_arr_only: bool,
    harmonic_modes: Optional[List] = None,
    m_arr: Optional[List] = None,
    bank_size: int = SMALL_BANK_SIZE,
    blocksize: int = SMALL_BLOCKSIZE,
) -> Path:
    """
    Build a tiny bank.

    If legacy_m_arr_only, config has m_arr only (pre-change style).
    """
    bank_dir = tmp_path / name
    bank_dir.mkdir()
    samples_path = bank_dir / "intrinsic_sample_bank.feather"
    _minimal_bank_samples(bank_size).to_feather(samples_path)

    bank_config = {
        "bank_size": bank_size,
        "fbin": config.DEFAULT_FBIN.tolist(),
        "f_ref": 50.0,
        "approximant": SMALL_APPROXIMANT,
        "blocksize": blocksize,
    }
    if legacy_m_arr_only:
        bank_config["m_arr"] = m_arr if m_arr is not None else [2, 1, 3, 4]
    else:
        if harmonic_modes is None:
            raise ValueError(
                "harmonic_modes required when not legacy_m_arr_only"
            )
        bank_config["harmonic_modes"] = harmonic_modes
        if m_arr is not None:
            bank_config["m_arr"] = m_arr

    bank_config_path = bank_dir / "bank_config.json"
    with open(bank_config_path, "w", encoding="utf-8") as f:
        json.dump(bank_config, f, indent=2)

    waveform_banks.create_waveform_bank_from_samples(
        samples_path=samples_path,
        bank_config_path=bank_config_path,
        waveform_dir=bank_dir / "waveforms",
        n_pool=1,
        blocksize=blocksize,
        approximant=SMALL_APPROXIMANT,
        harmonic_modes=None if legacy_m_arr_only else harmonic_modes,
    )
    return bank_dir


def _run_small_inference(
    tmp_path: Path,
    bank_dir: Path,
    rundir_name: str,
) -> dict:
    from cogwheel.data import EventData

    from dot_pe import inference

    event_data = EventData.gaussian_noise("", **config.EVENT_DATA_KWARGS)
    rundir = tmp_path / rundir_name
    inference.run(
        event=event_data,
        bank_folder=bank_dir,
        n_int=SMALL_N_INT,
        n_ext=SMALL_N_EXT,
        n_phi=SMALL_N_PHI,
        n_t=SMALL_N_T,
        blocksize=SMALL_BLOCKSIZE,
        single_detector_blocksize=SMALL_BLOCKSIZE,
        rundir=rundir,
        mchirp_guess=25.0,
        max_incoherent_lnlike_drop=20,
        max_bestfit_lnlike_diff=20,
        coherent_score_min_n_effective_prior=0,
        draw_subset=False,
        seed=42,
    )
    with open(rundir / "summary_results.json", encoding="utf-8") as f:
        summary = json.load(f)
    assert np.isfinite(summary["ln_evidence"]), summary
    assert summary.get("n_effective", 0) >= 0
    return summary


@pytest.mark.integration
def test_inference_legacy_m_arr_only_bank(tmp_path):
    """Legacy bank_config (m_arr only) through full inference.run."""
    pytest.importorskip("lalsimulation")
    bank_dir = _create_small_bank(
        tmp_path, "bank_legacy", legacy_m_arr_only=True, m_arr=[2, 1, 3, 4]
    )
    with open(bank_dir / "bank_config.json", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["m_arr"] == [2, 1, 3, 4]
    assert "harmonic_modes" in saved
    amp = np.load(bank_dir / "waveforms" / "amplitudes_block_0.npy")
    assert amp.shape == (SMALL_BANK_SIZE, 4, 2, len(config.DEFAULT_FBIN))
    _run_small_inference(tmp_path, bank_dir, "run_legacy")


@pytest.mark.integration
def test_inference_explicit_harmonic_modes_bank(tmp_path):
    """New scheme: harmonic_modes in config from the start (full mode set)."""
    pytest.importorskip("lalsimulation")
    from dot_pe.utils import default_harmonic_modes_for_approximant

    default_modes = default_harmonic_modes_for_approximant(SMALL_APPROXIMANT)
    hm = [[l, m] for l, m in default_modes]
    bank_dir = _create_small_bank(
        tmp_path,
        "bank_hm",
        legacy_m_arr_only=False,
        harmonic_modes=hm,
        m_arr=[2, 1, 3, 4],
    )
    with open(bank_dir / "bank_config.json", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["harmonic_modes"] == hm
    assert saved["m_arr"] == [2, 1, 3, 4]
    amp = np.load(bank_dir / "waveforms" / "amplitudes_block_0.npy")
    assert amp.shape[1] == 4
    _run_small_inference(tmp_path, bank_dir, "run_hm")


@pytest.mark.integration
def test_inference_single_mode_harmonic_modes_bank(tmp_path):
    """New scheme: (2,2)-only bank through inference.run."""
    pytest.importorskip("lalsimulation")
    bank_dir = _create_small_bank(
        tmp_path,
        "bank_22",
        legacy_m_arr_only=False,
        harmonic_modes=[[2, 2]],
    )
    with open(bank_dir / "bank_config.json", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["m_arr"] == [2]
    amp = np.load(bank_dir / "waveforms" / "amplitudes_block_0.npy")
    assert amp.shape[1] == 1
    _run_small_inference(tmp_path, bank_dir, "run_22")

