"""Coherent best fit bounded by the template's incoherent log-likelihood.

For one template, the sum over detectors of the per-detector maximum log-likelihood
(Stage 2) bounds the network log-likelihood at any extrinsic point. GW240630_212937 run_5
(on 4516a13, so with `invalid_bestfit_mask`) still put all its weight on a template whose
coherent best fit was 229.6 against an incoherent 46.7: its <h|h> nearly cancelled at that
row (0.06 percent of the same template's largest) without going non-positive or below the
1 Mpc distance cut. Over 162 O4b production runs, genuine rows exceed the bound by at most
13.2 nats (a coarser Stage 2 time grid), so the tolerance is 20.
"""

import inspect

import numpy as np

from dot_pe import coherent_processing, inference, mp_inference, thin_coherent
from dot_pe.likelihood_calculating import (
    INCOHERENT_BOUND_TOL,
    incoherent_bound_mask,
    incoherent_lookup,
)

RUN5_BESTFIT, RUN5_INCOHERENT = 229.558270, 46.695880  # very_low_q_bank_57 index 656668


def test_bound_flags_only_rows_above_tolerance():
    incoherent_i = np.array([RUN5_INCOHERENT, 30.0])
    bestfit = np.full((2, 3, 1), -np.inf)
    bestfit[0, :, 0] = [RUN5_BESTFIT, RUN5_INCOHERENT + 13.2, RUN5_INCOHERENT - 5.0]
    bestfit[1, :, 0] = [30.0 + INCOHERENT_BOUND_TOL + 1e-6, 30.0 + INCOHERENT_BOUND_TOL, 10.0]
    mask = incoherent_bound_mask(bestfit, incoherent_i)
    assert mask.shape == bestfit.shape and mask.dtype == bool
    assert mask[:, :, 0].tolist() == [[True, False, False], [True, False, False]]


def test_lookup_is_bank_indexed_and_unbounded_elsewhere():
    lookup = incoherent_lookup(6, np.array([4, 1]), np.array([40.0, 10.0]))
    assert lookup[1] == 10.0 and lookup[4] == 40.0
    assert np.isinf(lookup[[0, 2, 3, 5]]).all() and (lookup[[0, 2, 3, 5]] > 0).all()


def test_serial_selection_drops_rows_above_the_bound():
    hh = np.full((2, 1, 1), 1.0e6)
    target = np.array([RUN5_BESTFIT, 40.0])  # template 0 above its bound, 1 below
    dh = np.sqrt(2 * target * 1.0e6).reshape(2, 1, 1)
    select = coherent_processing.CoherentLikelihoodProcessor.select_and_get_lnlike
    _, unbounded = select(dh, hh)
    _, bounded = select(dh, hh, incoherent_i=np.array([RUN5_INCOHERENT, 45.0]))
    assert unbounded[:, 0, 0].tolist() == [True, True]
    assert bounded[:, 0, 0].tolist() == [False, True]


def test_both_coherent_selections_apply_the_bound_before_thresholding():
    for func in (
        coherent_processing.CoherentLikelihoodProcessor.select_and_get_lnlike,
        thin_coherent.run_thin_iblock,
    ):
        src = inspect.getsource(func)
        assert "incoherent_bound_mask" in src, f"{func.__qualname__} does not bound"
        assert src.index("incoherent_bound_mask") < src.index("accepted ="), func.__qualname__


def test_stage3_values_reach_both_coherent_stages():
    """The bound is only active if the callers pass Stage 3's incoherent values on."""
    assert "selected_lnlikes_by_bank=selected_lnlikes_by_bank" in inspect.getsource(mp_inference.run)
    assert "incoherent_lnlike_i" in inspect.getsource(mp_inference._run_coherent_mp)
    per_bank = inspect.getsource(inference.run_coherent_inference_per_bank)
    assert "incoherent_lnlikes=" in per_bank
    src = inspect.getsource(inference)
    assert src.count("selected_lnlikes_by_bank=selected_lnlikes_by_bank") >= 2
