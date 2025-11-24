"""
Custom prior with power-law chirp mass distribution.

Copied locally for the incoherent-search tooling to avoid cross-package imports.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import dblquad

from cogwheel import utils
from cogwheel.prior import Prior, CombinedPrior
from cogwheel.gw_prior.combined import (
    IntrinsicIASPrior,
    RegisteredPriorMixin,
    MarginalizedExtrinsicLikelihoodQAS,
)
from cogwheel.gw_prior.mass import UniformDetectorFrameMassesPrior
from cogwheel.gw_prior.spin import UniformEffectiveSpinPrior
from cogwheel.gw_prior.miscellaneous import (
    ZeroTidalDeformabilityPrior,
    FixedReferenceFrequencyPrior,
)


class PowerLawChirpMassPrior(Prior):
    """
    Power-law prior for detector-frame chirp mass: P(M_c) \propto M_c^{-1.7}

    Sampled variables are mchirp_p = mchirp^{-0.7}, lnq.
    These are transformed to m1, m2.
    The prior integrates to 1 over mchirp_p and lnq ranges.
    """

    standard_params = ["m1", "m2"]
    range_dic = {"mchirp_p": None, "lnq": None}
    reflective_params = ["lnq"]

    ALPHA = -1.7  # P(M_c) ∝ M_c^{ALPHA}
    TRANSFORM_POWER = -0.7  # mchirp_p = mchirp^{TRANSFORM_POWER}

    def __init__(self, *, mchirp_range, q_min=0.05, symmetrize_lnq=False, **kwargs):
        lnq_min = np.log(q_min)

        mchirp_min, mchirp_max = mchirp_range
        mchirp_p_min = min(
            mchirp_min**self.TRANSFORM_POWER, mchirp_max**self.TRANSFORM_POWER
        )
        mchirp_p_max = max(
            mchirp_min**self.TRANSFORM_POWER, mchirp_max**self.TRANSFORM_POWER
        )

        self.range_dic = {
            "mchirp_p": (mchirp_p_min, mchirp_p_max),
            "lnq": (lnq_min, -lnq_min * symmetrize_lnq),
        }

        self._mchirp_range = mchirp_range
        self._q_min = q_min
        self._symmetrize_lnq = symmetrize_lnq

        super().__init__(**kwargs)

        self.prior_norm = 1
        self.prior_norm = dblquad(
            lambda mchirp_p, lnq: np.exp(self.lnprior(mchirp_p, lnq)),
            *self.range_dic["lnq"],
            *self.range_dic["mchirp_p"],
        )[0]

    @staticmethod
    @utils.lru_cache()
    def _mchirp_p_to_mchirp(mchirp_p):
        return mchirp_p ** (-10 / 7)

    @staticmethod
    @utils.lru_cache()
    def transform(mchirp_p, lnq):
        mchirp = PowerLawChirpMassPrior._mchirp_p_to_mchirp(mchirp_p)
        q = np.exp(-np.abs(lnq))
        m1 = mchirp * (1 + q) ** 0.2 / q**0.6
        return {"m1": m1, "m2": m1 * q}

    @staticmethod
    def inverse_transform(m1, m2):
        q = m2 / m1
        mchirp = m1 * q**0.6 / (1 + q) ** 0.2
        mchirp_p = mchirp**PowerLawChirpMassPrior.TRANSFORM_POWER
        return {"mchirp_p": mchirp_p, "lnq": np.log(q)}

    @staticmethod
    def ln_jacobian_determinant(m1, m2):
        ln_jac_mchirp_lnq = -np.log((m1 * m2) ** 2 * (m1 + m2)) / 5
        q = m2 / m1
        mchirp = m1 * q**0.6 / (1 + q) ** 0.2
        ln_jac_mchirp_p = np.log(
            np.abs(PowerLawChirpMassPrior.TRANSFORM_POWER)
        ) - 1.7 * np.log(mchirp)
        return ln_jac_mchirp_lnq + ln_jac_mchirp_p

    @utils.lru_cache()
    def lnprior(self, mchirp_p, lnq):
        lnp = np.log(np.cosh(lnq / 2) ** 0.4) - np.log(self.prior_norm)
        return lnp

    def get_init_dict(self):
        return {
            "mchirp_range": self._mchirp_range,
            "q_min": self._q_min,
            "symmetrize_lnq": self._symmetrize_lnq,
        }


class PowerLawIntrinsicIASPrior(CombinedPrior):
    """
    IntrinsicIASPrior with power-law chirp mass distribution: P(M_c) ∝ M_c^{-1.7}
    """

    default_likelihood_class = IntrinsicIASPrior.default_likelihood_class

    prior_classes = utils.replace(
        IntrinsicIASPrior.prior_classes,
        UniformDetectorFrameMassesPrior,
        PowerLawChirpMassPrior,
    )


class PowerLawMassAlignedSpinIASPrior(RegisteredPriorMixin, CombinedPrior):
    """
    Intrinsic aligned-spin prior with power-law chirp mass distribution.

    Produces intrinsic samples over (m1, m2, s1z, s2z) with no in-plane spin or tides.
    """

    default_likelihood_class = MarginalizedExtrinsicLikelihoodQAS

    prior_classes = [
        PowerLawChirpMassPrior,
        UniformEffectiveSpinPrior,
        ZeroTidalDeformabilityPrior,
        FixedReferenceFrequencyPrior,
    ]
