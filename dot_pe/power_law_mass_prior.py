"""
Custom prior with power-law chirp mass distribution.

Copied locally for the incoherent-search tooling to avoid cross-package imports.
"""

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
    Power-law prior for detector-frame chirp mass: P(M_c) ∝ M_c^alpha.

    Default alpha=-1.7 gives P(M_c) ∝ M_c^{-1.7}. Sampled variables are
    mchirp_p = mchirp^{alpha+1} and lnq, transformed to m1, m2.
    The prior integrates to 1 over mchirp_p and lnq ranges.
    """

    standard_params = ["m1", "m2"]
    range_dic = {"mchirp_p": None, "lnq": None}
    reflective_params = ["lnq"]

    def __init__(
        self,
        *,
        mchirp_range,
        q_min=0.05,
        q_max=1.0,
        symmetrize_lnq=False,
        alpha=-1.7,
        **kwargs,
    ):
        if alpha == -1:
            raise ValueError(
                "alpha=-1 gives transform_power=0; choose alpha != -1."
            )
        self._alpha = float(alpha)
        self._transform_power = self._alpha + 1

        lnq_min = np.log(q_min)
        lnq_max = -lnq_min * symmetrize_lnq if symmetrize_lnq else np.log(q_max)

        mchirp_min, mchirp_max = mchirp_range
        mchirp_p_min = min(
            mchirp_min**self._transform_power, mchirp_max**self._transform_power
        )
        mchirp_p_max = max(
            mchirp_min**self._transform_power, mchirp_max**self._transform_power
        )

        self.range_dic = {
            "mchirp_p": (mchirp_p_min, mchirp_p_max),
            "lnq": (lnq_min, lnq_max),
        }

        self._mchirp_range = mchirp_range
        self._q_min = q_min
        self._q_max = q_max
        self._symmetrize_lnq = symmetrize_lnq

        super().__init__(**kwargs)

        self.prior_norm = 1
        self.prior_norm = dblquad(
            lambda mchirp_p, lnq: np.exp(self.lnprior(mchirp_p, lnq)),
            *self.range_dic["lnq"],
            *self.range_dic["mchirp_p"],
        )[0]

    def _mchirp_p_to_mchirp(self, mchirp_p):
        return mchirp_p ** (1 / self._transform_power)

    def transform(self, mchirp_p, lnq):
        mchirp = self._mchirp_p_to_mchirp(mchirp_p)
        q = np.exp(-np.abs(lnq))
        m1 = mchirp * (1 + q) ** 0.2 / q**0.6
        return {"m1": m1, "m2": m1 * q}

    def inverse_transform(self, m1, m2):
        q = m2 / m1
        mchirp = m1 * q**0.6 / (1 + q) ** 0.2
        mchirp_p = mchirp**self._transform_power
        return {"mchirp_p": mchirp_p, "lnq": np.log(q)}

    def ln_jacobian_determinant(self, m1, m2):
        ln_jac_mchirp_lnq = -np.log((m1 * m2) ** 2 * (m1 + m2)) / 5
        q = m2 / m1
        mchirp = m1 * q**0.6 / (1 + q) ** 0.2
        ln_jac_mchirp_p = np.log(np.abs(self._transform_power)) + self._alpha * np.log(
            mchirp
        )
        return ln_jac_mchirp_lnq + ln_jac_mchirp_p

    @utils.lru_cache()
    def lnprior(self, mchirp_p, lnq):
        lnp = np.log(np.cosh(lnq / 2) ** 0.4) - np.log(self.prior_norm)
        return lnp

    def get_init_dict(self):
        return {
            "mchirp_range": self._mchirp_range,
            "q_min": self._q_min,
            "q_max": self._q_max,
            "symmetrize_lnq": self._symmetrize_lnq,
            "alpha": self._alpha,
        }


class PowerLawIntrinsicIASPrior(CombinedPrior):
    """
    IntrinsicIASPrior with power-law chirp mass: P(M_c) ∝ M_c^alpha (default -1.7).
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
    P(M_c) ∝ M_c^alpha with default alpha=-1.7.
    """

    default_likelihood_class = MarginalizedExtrinsicLikelihoodQAS

    prior_classes = [
        PowerLawChirpMassPrior,
        UniformEffectiveSpinPrior,
        ZeroTidalDeformabilityPrior,
        FixedReferenceFrequencyPrior,
    ]
