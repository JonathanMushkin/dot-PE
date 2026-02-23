"""Tests for q_max parameter in bank creation and priors."""

import pytest

from dot_pe.utils import validate_q_bounds
from dot_pe.mass_prior import get_mass_prior, UniformDetectorFrameMassesPriorWithQMax


def test_validate_q_bounds_valid():
    """validate_q_bounds accepts valid (q_min, q_max) pairs."""
    validate_q_bounds(0.1, 0.5)
    validate_q_bounds(0.2, 1.0)
    validate_q_bounds(0.01, 0.99)


def test_validate_q_bounds_q_min_ge_q_max():
    """validate_q_bounds rejects q_min >= q_max."""
    with pytest.raises(ValueError, match="q_min.*must be strictly less"):
        validate_q_bounds(0.5, 0.5)
    with pytest.raises(ValueError, match="q_min.*must be strictly less"):
        validate_q_bounds(0.6, 0.5)


def test_validate_q_bounds_q_max_gt_1():
    """validate_q_bounds rejects q_max > 1."""
    with pytest.raises(ValueError, match="q_max.*must be <= 1"):
        validate_q_bounds(0.2, 1.1)


def test_get_mass_prior_q_max_1_uses_cogwheel():
    """get_mass_prior returns cogwheel prior when q_max >= 1."""
    from cogwheel.gw_prior.mass import UniformDetectorFrameMassesPrior

    prior = get_mass_prior(
        mchirp_range=(10, 20), q_min=0.2, q_max=1.0
    )
    assert isinstance(prior, UniformDetectorFrameMassesPrior)
    assert not isinstance(prior, UniformDetectorFrameMassesPriorWithQMax)


def test_get_mass_prior_q_max_lt_1_uses_custom():
    """get_mass_prior returns custom prior when q_max < 1."""
    prior = get_mass_prior(
        mchirp_range=(10, 20), q_min=0.2, q_max=0.5
    )
    assert isinstance(prior, UniformDetectorFrameMassesPriorWithQMax)
    assert prior._q_max == 0.5
