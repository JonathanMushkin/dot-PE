"""Tests for evidence calibration (compute_evidence_on_noise)."""

import pytest

from cogwheel.data import EventData

from dot_pe import config
from dot_pe.evidence_calibration import compute_evidence_on_noise


def test_compute_evidence_on_noise_raises_for_nonexistent_rundir(tmp_path):
    """compute_evidence_on_noise raises FileNotFoundError for non-existent rundir."""
    nonexistent = tmp_path / "nonexistent_rundir"
    event_data = EventData.gaussian_noise("", **config.EVENT_DATA_KWARGS)
    with pytest.raises(FileNotFoundError, match="Rundir not found"):
        compute_evidence_on_noise(nonexistent, event_data)


def test_compute_evidence_on_noise_raises_for_missing_posterior(tmp_path):
    """compute_evidence_on_noise raises FileNotFoundError when Posterior.json is missing."""
    # Create empty rundir (no Posterior.json)
    rundir = tmp_path / "empty_rundir"
    rundir.mkdir()
    event_data = EventData.gaussian_noise("", **config.EVENT_DATA_KWARGS)
    with pytest.raises(FileNotFoundError, match="Posterior.json not found"):
        compute_evidence_on_noise(rundir, event_data)
