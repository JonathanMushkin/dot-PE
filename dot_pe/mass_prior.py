"""
Mass prior with configurable q_max.

Cogwheel's UniformDetectorFrameMassesPrior uses lnq ∈ [ln(q_min), 0] and does not
accept q_max. This module provides a prior with lnq ∈ [ln(q_min), ln(q_max)]
for q_max < 1.
"""

import numpy as np
from scipy.integrate import dblquad

from cogwheel import utils
from cogwheel.prior import Prior
from cogwheel.gw_prior.mass import UniformDetectorFrameMassesPrior


class UniformDetectorFrameMassesPriorWithQMax(Prior):
    """
    Uniform prior for detector-frame masses with configurable q_max.

    Same functional form as cogwheel's UniformDetectorFrameMassesPrior:
    p(mchirp, lnq) ∝ mchirp * cosh(lnq/2)^0.4
    but with lnq ∈ [ln(q_min), ln(q_max)] instead of [ln(q_min), 0].

    Parameters
    ----------
    mchirp_range : tuple
        (mchirp_min, mchirp_max)
    q_min : float
        Minimum mass ratio m2/m1
    q_max : float
        Maximum mass ratio m2/m1 (must satisfy q_min < q_max <= 1)
    """

    standard_params = ["m1", "m2"]
    range_dic = {"mchirp": None, "lnq": None}
    reflective_params = ["lnq"]

    def __init__(self, *, mchirp_range, q_min=0.05, q_max=1.0, **kwargs):
        lnq_min = np.log(q_min)
        lnq_max = np.log(q_max)
        self.range_dic = {
            "mchirp": mchirp_range,
            "lnq": (lnq_min, lnq_max),
        }
        self._q_min = q_min
        self._q_max = q_max
        super().__init__(**kwargs)

        self.prior_norm = 1
        self.prior_norm = dblquad(
            lambda mchirp, lnq: np.exp(self.lnprior(mchirp, lnq)),
            *self.range_dic["lnq"],
            *self.range_dic["mchirp"],
        )[0]

    @staticmethod
    @utils.lru_cache()
    def transform(mchirp, lnq):
        """(mchirp, lnq) to (m1, m2). Same as cogwheel's UniformDetectorFrameMassesPrior."""
        q = np.exp(-np.abs(lnq))
        m1 = mchirp * (1 + q) ** 0.2 / q**0.6
        return {"m1": m1, "m2": m1 * q}

    @staticmethod
    def inverse_transform(m1, m2):
        """(m1, m2) to (mchirp, lnq)."""
        q = m2 / m1
        return {"mchirp": m1 * q**0.6 / (1 + q) ** 0.2, "lnq": np.log(q)}

    def lnprior(self, mchirp, lnq):
        """Natural log prior density for (mchirp, lnq)."""
        return np.log(mchirp * np.cosh(lnq / 2) ** 0.4 / self.prior_norm)

    def get_init_dict(self, **kwargs):
        """Return keyword arguments to reproduce the class instance."""
        return super().get_init_dict(
            mchirp_range=self.range_dic["mchirp"],
            q_min=self._q_min,
            q_max=self._q_max,
            **kwargs,
        )


def get_mass_prior(mchirp_range, q_min, q_max=1.0):
    """
    Return the appropriate mass prior for the given q bounds.

    Uses cogwheel's UniformDetectorFrameMassesPrior when q_max == 1,
    otherwise UniformDetectorFrameMassesPriorWithQMax.
    """
    if q_max >= 1.0:
        return UniformDetectorFrameMassesPrior(
            mchirp_range=mchirp_range, q_min=q_min
        )
    return UniformDetectorFrameMassesPriorWithQMax(
        mchirp_range=mchirp_range, q_min=q_min, q_max=q_max
    )
