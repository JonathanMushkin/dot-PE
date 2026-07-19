"""
Regression tests for the minimal best-fit-distance cut.

The distance-optimized likelihood lnl = <d|h>^2 / (2 <h|h>) is unbounded
in distance: some (phase, polarization, sky) combinations drive <h|h>
toward zero, and the optimum lands at an unphysically small best-fit
distance d_best = <h|h>/<d|h> (Mpc at the 1 Mpc reference) with a huge
lnl. Since selections keep samples within a fixed drop of the maximum,
one such sample used to poison the entire selection (measured case:
GW240630_212937, d_h=3095.37, h_h=50.81, lnl=94287.6, d_best=16 kpc).
Samples with d_best < MIN_D_LUMINOSITY are now masked before any
relative-to-max thresholding, in both coherent selection functions and
in the single-detector (incoherent) stage.
"""

import numpy as np

from dot_pe.likelihood_calculating import (
    MIN_D_LUMINOSITY,
    LikelihoodCalculator,
)
from dot_pe.single_detector import SingleDetectorProcessor

# Measured pathological sample (see module docstring).
PATHOLOGICAL_DH, PATHOLOGICAL_HH = 3095.37, 50.81
# Genuine sample: lnl = dh^2/(2 hh) = 30 at d_best ~ 129 Mpc.
GENUINE_HH = 1.0e6
GENUINE_DH = np.sqrt(2 * 30.0 * GENUINE_HH)


def test_pathological_sample_is_within_measured_regime():
    """The constants above reproduce the documented pathology."""
    assert PATHOLOGICAL_HH / PATHOLOGICAL_DH < MIN_D_LUMINOSITY
    assert PATHOLOGICAL_DH**2 / (2 * PATHOLOGICAL_HH) > 9e4
    assert GENUINE_HH / GENUINE_DH > MIN_D_LUMINOSITY


def _dh_hh_with_pathological():
    """(2, 2, 1) arrays: 3 genuine entries + a pathological one at i=1, e=0, o=0."""
    dh = np.full((2, 2, 1), GENUINE_DH)
    hh = np.full((2, 2, 1), GENUINE_HH)
    dh[1, 0, 0] = PATHOLOGICAL_DH
    hh[1, 0, 0] = PATHOLOGICAL_HH
    return dh, hh


def test_bestfit_selection_masks_pathological_keeps_genuine():
    dh, hh = _dh_hh_with_pathological()
    i_inds, e_inds, o_inds = LikelihoodCalculator.select_ieo_by_bestfit_lnlike(
        dh, hh, cut_threshold=20.0
    )
    selected = set(zip(i_inds, e_inds, o_inds))
    assert (1, 0, 0) not in selected
    assert selected == {(0, 0, 0), (0, 1, 0), (1, 1, 0)}


def test_bestfit_selection_still_excludes_negative_dh():
    dh = np.full((1, 2, 1), GENUINE_DH)
    hh = np.full((1, 2, 1), GENUINE_HH)
    dh[0, 1, 0] = -GENUINE_DH
    i_inds, e_inds, o_inds = LikelihoodCalculator.select_ieo_by_bestfit_lnlike(dh, hh)
    assert set(zip(i_inds, e_inds, o_inds)) == {(0, 0, 0)}


def test_bestfit_selection_all_masked_returns_empty():
    dh = np.full((2, 1, 1), PATHOLOGICAL_DH)
    hh = np.full((2, 1, 1), PATHOLOGICAL_HH)
    i_inds, e_inds, o_inds = LikelihoodCalculator.select_ieo_by_bestfit_lnlike(dh, hh)
    assert len(i_inds) == len(e_inds) == len(o_inds) == 0


def test_dist_marginalized_selection_masks_pathological_keeps_genuine():
    dh, hh = _dh_hh_with_pathological()
    log_w_i = np.zeros(2)
    log_w_e = np.zeros(2)
    i_inds, e_inds, o_inds = (
        LikelihoodCalculator.select_ieo_by_approx_lnlike_dist_marginalized(
            dh, hh, log_w_i, log_w_e, cut_threshold=20.0
        )
    )
    selected = set(zip(i_inds, e_inds, o_inds))
    assert (1, 0, 0) not in selected
    assert selected == {(0, 0, 0), (0, 1, 0), (1, 1, 0)}


def _single_detector_inputs(h_impb):
    """Minimal well-formed inputs: 1 detector, 1 mode, 1 bin, 1 time."""
    n_i = h_impb.shape[0]
    dh_weights_dmpb = np.zeros((1, 1, 2, 1), dtype=complex)
    dh_weights_dmpb[0, 0, 0, 0] = 1.0
    hh_weights_dmppb = np.ones((1, 1, 2, 2, 1), dtype=complex)
    timeshift_dbt = np.ones((1, 1, 1), dtype=complex)
    asd_drift_d = np.ones(1)
    return dict(
        dh_weights_dmpb=dh_weights_dmpb,
        hh_weights_dmppb=hh_weights_dmppb,
        h_impb=h_impb.reshape(n_i, 1, 2, 1),
        timeshift_dbt=timeshift_dbt,
        asd_drift_d=asd_drift_d,
        n_phi=1,
        m_arr=np.array([2]),
    )


def test_single_detector_masks_near_singular_and_close_response():
    """Sample 0: well-conditioned, d_best = hh/dh = 10 Mpc -> finite lnlike.
    Sample 1: near-singular 2x2 <h|h> matrix -> huge response -> -inf.
    Sample 2: well-conditioned but d_best < MIN_D_LUMINOSITY -> -inf."""
    amp = 10.0
    h_impb = np.array(
        [
            [amp, 1j * amp],
            [amp, amp * (1 + 1e-8j)],
            [0.5, 0.5j],
        ],
        dtype=complex,
    )
    r_iotp, lnlike_iot = SingleDetectorProcessor.get_response_over_distance_and_lnlike(
        **_single_detector_inputs(h_impb)
    )

    # hh = amp^2 * I, dh = [amp, 0] -> r = [1/amp, 0], lnlike = 1/2.
    d_best_0 = 1 / np.linalg.norm(r_iotp[0, 0, 0])
    assert d_best_0 >= MIN_D_LUMINOSITY
    assert np.isclose(lnlike_iot[0, 0, 0], 0.5)

    assert np.isneginf(lnlike_iot[1, 0, 0])

    d_best_2 = 1 / np.linalg.norm(r_iotp[2, 0, 0])
    assert d_best_2 < MIN_D_LUMINOSITY
    assert np.isneginf(lnlike_iot[2, 0, 0])


def test_single_detector_masks_nonpositive_determinant():
    """h = [amp, amp] gives a rank-1 Gram matrix amp^2 * [[1, 1], [1, 1]]
    (det = 0): the explicit det > 0 check masks it regardless of what the
    inf/nan-contaminated solve returns."""
    amp = 10.0
    h_impb = np.array([[amp, 1j * amp], [amp, amp]], dtype=complex)
    with np.errstate(invalid="ignore"):
        _, lnlike_iot = SingleDetectorProcessor.get_response_over_distance_and_lnlike(
            **_single_detector_inputs(h_impb)
        )
    assert np.isclose(lnlike_iot[0, 0, 0], 0.5)
    assert np.isneginf(lnlike_iot[1, 0, 0])
