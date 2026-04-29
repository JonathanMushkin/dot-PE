"""Tests for prob_samples row count vs size_limit in CoherentLikelihoodProcessor."""

import numpy as np
import pandas as pd

from dot_pe.coherent_processing import CoherentLikelihoodProcessor


def _minimal_clp_for_combine():
    """Bare instance with only attributes needed by combine_prob_samples_with_next_block."""
    clp = object.__new__(CoherentLikelihoodProcessor)
    clp.size_limit = 3
    clp.min_bestfit_lnlike_to_keep = -np.inf
    clp.full_log_prior_weights_i = np.zeros(100, dtype=float)
    clp.full_log_prior_weights_e = np.zeros(100, dtype=float)
    clp._intrinsic_logw_sorted_inds = None
    clp.prob_samples = pd.DataFrame(
        columns=[
            "i",
            "e",
            "o",
            "lnl_marginalized",
            "ln_posterior",
            "bestfit_lnlike",
            "d_h_1Mpc",
            "h_h_1Mpc",
        ]
    )
    clp._get_intrinsic_logw = lambda abs_i: np.zeros(np.asarray(abs_i).shape, dtype=float)
    return clp


def test_combine_raises_floor_when_row_count_hits_size_limit():
    """Buffer-full branch must use row count, not DataFrame.size (rows * columns)."""
    clp = _minimal_clp_for_combine()
    n = 3
    clp._next_block = {
        "i_k": np.arange(n),
        "e_k": np.arange(n),
        "o_k": np.arange(n),
        "dh_k": np.ones(n),
        "hh_k": np.ones(n),
        "dist_marg_lnlike_k": np.zeros(n),
        "bestfit_lnlike_k": np.array([10.0, 20.0, 30.0]),
        "bestfit_lnlike_max": 30.0,
        "bank_i_inds_k": np.zeros(n, dtype=int),
        "bank_e_inds_k": np.zeros(n, dtype=int),
    }

    CoherentLikelihoodProcessor.combine_prob_samples_with_next_block(clp)

    assert len(clp.prob_samples) == 3
    assert clp.prob_samples.size == 3 * len(clp.prob_samples.columns)
    # Regression: old bug compared total elements to size_limit
    assert clp.prob_samples.size != clp.size_limit
    assert clp.min_bestfit_lnlike_to_keep == 10.0
