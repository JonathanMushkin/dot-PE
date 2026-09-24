"""The corrupted-inner-product mask is shared by both coherent selections.

PR #23 added the mask to `CoherentLikelihoodProcessor.select_and_get_lnlike`, the serial
coherent stage. Production runs go through `mp_inference.run`, which dispatches to
`thin_coherent.run_thin_iblock` — a second, independent selection that had no mask, so
every multiprocessing run stayed exposed: GW240630_212937 run_2 put all of its posterior
weight on two templates with d_best = 16 kpc (best fit 94 288 against 27 expected), and
run_4 crashed in postprocess with 9 819 rows of non-positive <h|h> making every weight NaN.

Both call sites now use `likelihood_calculating.invalid_bestfit_mask`.
"""

import inspect

import numpy as np

from dot_pe import coherent_processing, thin_coherent
from dot_pe.likelihood_calculating import MIN_D_LUMINOSITY, invalid_bestfit_mask

# Measured pathological pair (GW240630_212937 run_2) and a genuine one at d_best ~ 129 Mpc.
PATHOLOGICAL_DH, PATHOLOGICAL_HH = 3095.37, 50.81
GENUINE_HH = 1.0e6
GENUINE_DH = np.sqrt(2 * 30.0 * GENUINE_HH)


def test_mask_flags_nonpositive_hh_and_close_distance_only():
    dh = np.array([GENUINE_DH, GENUINE_DH, PATHOLOGICAL_DH, GENUINE_DH, -GENUINE_DH])
    hh = np.array([GENUINE_HH, -1.0e-3, PATHOLOGICAL_HH, 0.0, GENUINE_HH])
    mask = invalid_bestfit_mask(dh, hh)
    assert list(mask) == [False, True, True, True, False]
    assert PATHOLOGICAL_HH / PATHOLOGICAL_DH < MIN_D_LUMINOSITY
    assert GENUINE_HH / GENUINE_DH > MIN_D_LUMINOSITY


def test_mask_is_finite_and_shape_preserving():
    rng = np.random.default_rng(0)
    dh = rng.normal(size=(3, 4, 2)) * 1e3
    hh = rng.normal(size=(3, 4, 2)) * 1e6
    mask = invalid_bestfit_mask(dh, hh)
    assert mask.shape == dh.shape and mask.dtype == bool
    assert mask[hh <= 0].all()


def test_both_coherent_selections_apply_the_mask():
    """Regression: the multiprocessing path must not diverge from the serial one again."""
    for func in (
        coherent_processing.CoherentLikelihoodProcessor.select_and_get_lnlike,
        thin_coherent.run_thin_iblock,
    ):
        src = inspect.getsource(func)
        assert "invalid_bestfit_mask" in src, f"{func.__qualname__} does not mask"
        # the mask has to be applied before the relative-to-maximum threshold
        # ("accepted = bestfit... > min_bestfit..."), not after it
        assert src.index("invalid_bestfit_mask") < src.index("accepted ="), func.__qualname__
