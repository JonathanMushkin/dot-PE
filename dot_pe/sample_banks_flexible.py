"""
Generate intrinsic samples and waveforms for a bank.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lal import MSUN_SI
from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions
from scipy.stats import qmc

from cogwheel.gw_prior import UniformDiskInplaneSpinsIsotropicInclinationPrior
from cogwheel.gw_prior import UniformEffectiveSpinPrior
from cogwheel.gw_prior.mass import UniformDetectorFrameMassesPrior
from cogwheel.gw_utils import m1m2_to_mchirp, mchirpeta_to_m1m2, q_to_eta
from cogwheel.utils import NumpyEncoder
from cogwheel.prior import Prior, CombinedPrior

from . import waveform_banks
from .config import DEFAULT_F_REF


class DefaultPhysicalPrior(CombinedPrior):
    """
    Default physical prior combining cogwheel priors:
    - UniformDetectorFrameMassesPrior for masses
    - UniformEffectiveSpinPrior for aligned spins
    - UniformDiskInplaneSpinsIsotropicInclinationPrior for in-plane spins and inclination
    """

    prior_classes = [
        UniformDetectorFrameMassesPrior,
        UniformEffectiveSpinPrior,
        UniformDiskInplaneSpinsIsotropicInclinationPrior,
    ]


class LogUniformMassBiasedInclinationDrawingPrior(Prior):
    """
    Drawing prior that matches the current IntrinsicSamplesGenerator behavior:
    - Masses: uniform in log(m1), log(m2)
    - Spins: UniformEffectiveSpinPrior
    - Inclination: biased towards face-on/back-on
    """

    standard_params = [
        "m1",
        "m2",
        "s1z",
        "s2z",
        "iota",
        "s1x_n",
        "s1y_n",
        "s2x_n",
        "s2y_n",
    ]
    range_dic = {
        "logmchirp": None,  # Will be set in __init__
        "logq": None,  # Will be set in __init__
        "chieff": (-1, 1),
        "cumchidiff": (0, 1),
        "costheta_jn": (-1, 1),
        "phi_jl_hat": (0, 2 * np.pi),
        "phi12": (0, 2 * np.pi),
        "cums1r_s1z": (0, 1),
        "cums2r_s2z": (0, 1),
    }
    periodic_params = ["phi_jl_hat", "phi12"]
    folded_reflected_params = ["costheta_jn"]
    conditioned_on = ["f_ref"]

    def __init__(self, m_min, m_max, q_min, inc_faceon_factor):
        self.m_min = m_min
        self.m_max = m_max
        self.q_min = q_min
        self.inc_faceon_factor = inc_faceon_factor

        # Set mass ranges
        self.range_dic["logmchirp"] = (np.log(m_min), np.log(m_max))
        self.range_dic["logq"] = (np.log(q_min), 0)  # log(q) from log(q_min) to 0

        # Initialize spin prior
        self.spin_prior = UniformEffectiveSpinPrior()
        self.inplane_spin_prior = UniformDiskInplaneSpinsIsotropicInclinationPrior()

        super().__init__()

    def transform(
        self,
        logmchirp,
        logq,
        chieff,
        cumchidiff,
        costheta_jn,
        phi_jl_hat,
        phi12,
        cums1r_s1z,
        cums2r_s2z,
        f_ref,
    ):
        """Transform sampled parameters to standard parameters"""
        # Masses - convert from chirp mass and mass ratio to m1, m2
        mchirp = np.exp(logmchirp)
        q = np.exp(logq)

        # Use the same conversion as the original draw_lnmchirp_lnq_uniform method
        m1, m2 = mchirpeta_to_m1m2(mchirp, q_to_eta(q))

        # Spins
        spin_dict = self.spin_prior.transform(chieff, cumchidiff, m1, m2)
        s1z, s2z = spin_dict["s1z"], spin_dict["s2z"]

        # Inclination and in-plane spins
        theta_jn = np.arccos(costheta_jn)
        chi1, tilt1 = self.inplane_spin_prior._spin_transform(cums1r_s1z, s1z)
        chi2, tilt2 = self.inplane_spin_prior._spin_transform(cums2r_s2z, s2z)

        phi_jl = (phi_jl_hat + np.pi * (costheta_jn < 0)) % (2 * np.pi)

        # Use LALSimulation to get final parameters
        iota, s1x_n, s1y_n, s1z, s2x_n, s2y_n, s2z = (
            SimInspiralTransformPrecessingNewInitialConditions(
                theta_jn,
                phi_jl,
                tilt1,
                tilt2,
                phi12,
                chi1,
                chi2,
                m1 * MSUN_SI,
                m2 * MSUN_SI,
                f_ref,
                phiRef=0.0,
            )
        )

        return {
            "m1": m1,
            "m2": m2,
            "s1z": s1z,
            "s2z": s2z,
            "iota": iota,
            "s1x_n": s1x_n,
            "s1y_n": s1y_n,
            "s2x_n": s2x_n,
            "s2y_n": s2y_n,
        }

    def inverse_transform(
        self, m1, m2, s1z, s2z, iota, s1x_n, s1y_n, s2x_n, s2y_n, f_ref
    ):
        """Transform standard parameters to sampled parameters"""
        # This would be more complex to implement
        # For now, we'll use the drawing prior for sampling, not inverse transform
        raise NotImplementedError("Inverse transform not needed for drawing prior")

    def lnprior(
        self,
        logmchirp,
        logq,
        chieff,
        cumchidiff,
        costheta_jn,
        phi_jl_hat,
        phi12,
        cums1r_s1z,
        cums2r_s2z,
        f_ref,
    ):
        """Log prior density in sampled parameter space"""
        # Masses - convert from chirp mass and mass ratio to m1, m2
        mchirp = np.exp(logmchirp)
        q = np.exp(logq)
        m1, m2 = mchirpeta_to_m1m2(mchirp, q_to_eta(q))

        # Log-uniform mass prior
        log_mass_prior = 0.0  # uniform in log space

        # Uniform spin prior (same as physical)
        log_spin_prior = self.spin_prior.lnprior(chieff, cumchidiff, m1, m2)

        # Biased inclination prior
        inc_factor = (self.inc_faceon_factor - 1) / (self.inc_faceon_factor + 1)
        theta_jn = np.arccos(costheta_jn)
        log_inclination_prior = np.log((1 + inc_factor * np.cos(2 * theta_jn)) / np.pi)

        # Uniform in-plane spin prior
        log_inplane_prior = 0.0  # uniform in cums1r_s1z, cums2r_s2z

        return (
            log_mass_prior + log_spin_prior + log_inclination_prior + log_inplane_prior
        )

    def get_init_dict(self):
        """Return dictionary with keyword arguments to reproduce the class instance."""
        return {
            "m_min": self.m_min,
            "m_max": self.m_max,
            "q_min": self.q_min,
            "inc_faceon_factor": self.inc_faceon_factor,
        }


class IntrinsicSamplesGenerator:
    """
    Class to generate intrinsic samples, with importance sampling.
    The samples are drawn under the provided drawing prior and re-weighted
    according to the physical prior.

    Parameters
    ----------
    drawing_prior : cogwheel.prior.Prior
        The prior used to draw samples (what we actually use to generate samples)
    physical_prior : cogwheel.prior.Prior, optional
        The physical prior (what we want to sample from). If None, uses the same as drawing_prior.
    """

    def __init__(self, drawing_prior, physical_prior=None):
        self.drawing_prior = drawing_prior
        self.physical_prior = (
            physical_prior if physical_prior is not None else drawing_prior
        )

    def draw_lnmchirp_lnq_uniform(self, q_min, mchirp_min, mchirp_max, u=None, n=None):
        """
        Draw (m1,m2) samples from a distribution that is uniform in lnq and
        ln(mchirp), and them with weights relative to the uniform-mass prior
        with the same constraints.

        """
        if u is None:
            if n is None:
                raise ValueError(
                    "pass either 2xn random samples from \
                                 (0-1) or number of samples to draw"
                )
            u = qmc.Halton(d=2).random(n).T

        mass_prior = UniformDetectorFrameMassesPrior(
            mchirp_range=(mchirp_min, mchirp_max), q_min=q_min
        )

        ln_mchirp = np.log(mchirp_min) + u[0] * np.log(mchirp_max / mchirp_min)
        mchirp = np.exp(ln_mchirp)

        lnq = np.log(q_min) + u[1] * np.log(1 / q_min)

        m1 = mchirp * (1 + np.exp(lnq)) ** (1 / 5) / np.exp(lnq) ** (3 / 5)
        m2 = m1 * np.exp(lnq)

        area = np.log(mchirp_max / mchirp_min) * np.log(1 / q_min)
        log_jacobian = -ln_mchirp  # log(det|d(log_mchirp,lnq)/(dmchirp,lnq)|
        # make units match (mchirp, lnq) with mass_prior
        fiducial_log_prior = np.log(1 / area) + log_jacobian
        log_prior = np.vectorize(mass_prior.lnprior, otypes=[float])(mchirp, lnq)
        log_prior_weights = log_prior - fiducial_log_prior

        return m1, m2, log_prior_weights

    def draw_m1m2_loguniform(self, q_min, m_min, m_max, u=None, n=None):
        """
        draw n-samples of component masses m1,m2, uniform in logarithmic
        space, under the constraints (m_min <= m <= m_max) and
        (m2/m1 >= q_min)

        Parameters
        ----------
        q_min : float,
            minimal m2/m1 allowed.
        m_min : float,
            minimal mass for both m1 and m2
        m_max : float,
            maximal mass for both m1 and m2
        u : array-like, optional
            random variable input of shape (2,n) for drawing. default None
        n : int, optional
            number of samples to draw. defalut None

        Returns
        -------
        m1 : 1d-array, masses of first component (in solar mass)
        m2 : 1d-array, masses of second component (in solar mass)
        log_prior_weights : 1d-array, weights for each sample relative
                            to the uniform-mass prior with same
                            constraints

        """
        if u is None:
            if n is None:
                raise ValueError(
                    "pass either 2xn random samples from \
                                 (0-1) or number of samples to draw"
                )
            u = qmc.Halton(d=2).random(n).T

        # m1 is uniform in log space between min and max values
        logm1 = np.log(m_min) + u[0] * np.log(m_max / m_min)
        # m2 is uniform in log space between being m1*q_min and m1
        logm2 = logm1 + (1 - u[1]) * np.log(q_min)
        m1, m2 = np.exp(logm1), np.exp(logm2)

        mchirp = m1m2_to_mchirp(m1, m2)
        lnq = np.log(m2 / m1)
        # the
        mchirp_min = m1m2_to_mchirp(m_min, m_min * q_min)
        mchirp_max = m1m2_to_mchirp(m_max, m_max)
        mass_prior = UniformDetectorFrameMassesPrior(
            mchirp_range=(mchirp_min, mchirp_max), q_min=q_min
        )

        # find weight = pdf(m1,m2) / pdf_pseudo(m1,m2).
        pseudo_prior_pdf = (
            1 / (np.log(m_max / m_min)) * 1 / np.log(1 / q_min) * 1 / m1 * 1 / m2
        )  # normalized expression
        # prior has units of 1/mchirp / lnq,
        # our pseudo prior has units of 1/m1 / m2

        jacobian_det = mchirp / m1 / m2  # det|(dmc, dlnq) / (dm1, dm2)|
        prior_lnpdf = np.vectorize(mass_prior.lnprior)(mchirp, lnq) + np.log(
            jacobian_det
        )  # normalized expression

        log_prior_weights = prior_lnpdf - np.log(pseudo_prior_pdf)

        return m1, m2, log_prior_weights

    def transform_and_unpack_aligned_spins(self, *args):
        """change dictionary output of method transform() for
        vectorization"""
        d = self.physical_prior.transform(*args)
        return d["s1z"], d["s2z"]

    def draw_s1z_s2z(self, m1, m2, u=None):
        """
        draw aligned-spins s1z, s2z, under the UniformEffectiveSpinPrior
        Parameters
        ----------
        m1 : 1d-array, first component masses (in solar mass)
        m2 : 1d-array, second component masses (in solar mass)
        u : 2d-array, optinal,
            values between 0 and 1 used to draw samples. If not passes,
            size in inferred from m1

        Returns
        -------
        s1z, s2z : 1d-arrays : aligned-spins of first and second binary
        components
        """
        if u is None:
            n = len(np.atleast_1d(m1))
            u = qmc.Halton(d=2).random(n).T

        chieff = -1 + 2 * u[0]
        cumchidiff = u[1]

        s1z, s2z = np.vectorize(self.transform_and_unpack_aligned_spins)(
            chieff, cumchidiff, m1, m2
        )

        return s1z, s2z

    def draw_inplane_spins_weighted_inclination(
        self, s1z, s2z, m1, m2, f_ref, u=None, inc_faceon_factor=2
    ):
        """
        Draw inplane spins and inclination under the ??? prior.
        Use prior on iota to prefer face/back-on inclinations, and
        re-weight according the isotropic prior

        Parameters
        ----------
        s1z : 1d-array, first component aligned spins (normalized)
        s2z : 1d-array, second component aligned spins (normalizeD)
        m1 : 1d-array, first component masses (in solar mass)
        m2 : 1d-array, second component masses (in solar mass)
        f_ref : reference frequency
        u :
            values between 0 and 1 used to draw samples. If not passes,
            size in inferred from m1
        inc_faceon_factor :float, optional,
            factor by which the prior (p.d.f) on inclination prefers
            face-on (iota=0) to edge-on (iota=pi/2) orientations.
            Default 2

        Returns
        -------
        iota : 1d-array, line of sight inclination
        s1x_n, s1y_n : 1d-arrays, in-plane spins of first component
                (normalized)
        s1z : same as input
        s2x_n s2y_n : 1d-arrays, in-plane spins of second component
                (normalized)
        s2z : same as input
        log_prior_weights : 1d-arrays, log weights of importance sampling
                (true prior / prior used for drawing)

        """

        if u is None:
            n = len(np.atleast_1d(m1))
            u = qmc.Halton(d=5).random(n).T
        inc_factor = (inc_faceon_factor - 1) / (inc_faceon_factor + 1)
        theta_jn_range = np.linspace(0, np.pi, 10**4)
        theta_jn_cdf = (
            theta_jn_range + inc_factor / 2 * np.sin(2 * theta_jn_range)
        ) / np.pi
        theta_jn_samples = theta_jn_range[np.searchsorted(theta_jn_cdf, u[0])]
        costheta_jn_samples = np.cos(theta_jn_samples)

        phi_jl_hat_samples = u[1] * np.pi * 2
        phi12_samples = u[2] * np.pi * 2
        cums1r_s1z_samples, cums2r_s2z_samples = u[3], u[4]

        chi1, tilt1 = self.inplane_spin_prior._spin_transform(cums1r_s1z_samples, s1z)
        chi2, tilt2 = self.inplane_spin_prior._spin_transform(cums2r_s2z_samples, s2z)
        phi_jl_samples = (phi_jl_hat_samples + np.pi * (costheta_jn_samples < 0)) % (
            2 * np.pi
        )
        theta_jn_samples = np.arccos(costheta_jn_samples)

        (iota, s1x_n, s1y_n, s1z, s2x_n, s2y_n, s2z) = np.array(
            [
                SimInspiralTransformPrecessingNewInitialConditions(
                    theta_jn_samples[i],
                    phi_jl_samples[i],
                    tilt1[i],
                    tilt2[i],
                    phi12_samples[i],
                    chi1[i],
                    chi2[i],
                    m1[i] * MSUN_SI,
                    m2[i] * MSUN_SI,
                    f_ref,
                    phiRef=0.0,
                )
                for i in range(u.shape[1])
            ]
        ).T

        pseudo_prior_pdf = (1 + inc_factor * np.cos(2 * theta_jn_samples)) / np.pi
        # cos(theta_jn_samples) ~ U(-1,1)
        prior_pdf = 1 / 2 * np.sin(theta_jn_samples)
        log_prior_weights = np.log(prior_pdf / pseudo_prior_pdf)
        return iota, s1x_n, s1y_n, s1z, s2x_n, s2y_n, s2z, log_prior_weights

    def draw_intrinsic_samples_uniform_in_ln_masses(
        self,
        n,
        q_min,
        m_min,
        m_max,
        inc_faceon_factor,
        f_ref,
        seed=None,
    ):
        """
        Draw intrinsic samples, with importance sampling.
        The samples are drawn under:
        - log(m1) is uniform between log(m_min) and log(m_max)
        - log(m2) is uniform in log(m1*q_min) and log(m1)
        - theta_jn prefers face-on / back-on orientation
        Samples are re-wegithed according to the Physical prior:
        - m1,m2 from mass.UniformDetectorFrameMassesPrior
        - cos(theta_jn) uniform in (-1, +1)
        Parameters
        ----------
        n : int, number of samples to draw
        q_min : float, minimal value of m2/m1 (q < 1 always)
        m_min : float, minimial mass of m1
        m_max : float, maximal mass of m1

        Returns
        -------
        intrinsic_samples : pandas.DataFrame
        """
        u = qmc.Halton(d=9, seed=seed).random(n).T
        m1, m2, lw_m = self.draw_m1m2_loguniform(q_min, m_min, m_max, u[:2])
        s1z, s2z = self.draw_s1z_s2z(m1, m2, u=u[2:4,])
        iota, s1x_n, s1y_n, s1z, s2x_n, s2y_n, s2z, lw_ipsi = (
            self.draw_inplane_spins_weighted_inclination(
                s1z, s2z, m1, m2, f_ref, u[4:,], inc_faceon_factor
            )
        )

        keys = [
            "m1",
            "m2",
            "s1z",
            "s1x_n",
            "s1y_n",
            "s2z",
            "s2x_n",
            "s2y_n",
            "iota",
            "log_prior_weights",
        ]
        values = [
            m1,
            m2,
            s1z,
            s1x_n,
            s1y_n,
            s2z,
            s2x_n,
            s2y_n,
            iota,
            lw_m + lw_ipsi,
        ]
        intrinsic_samples = pd.DataFrame(data={k: v for k, v in zip(keys, values)})

        return intrinsic_samples

    def draw_intrinsic_samples_uniform_in_lnmchrip_lnq(
        self, n, q_min, m_min, m_max, inc_faceon_factor, f_ref, seed=None
    ):
        """
        Draw intrinsic samples, with importance sampling.
        The samples are drawn under:
        - log(m1) is uniform between log(m_min) and log(m_max)
        - log(m2) is uniform in log(m1*q_min) and log(m1)
        - theta_jn prefers face-on / back-on orientation
        Samples are re-wegithed according to the Physical prior:
        - m1,m2 from mass.UniformDetectorFrameMassesPrior
        - cos(theta_jn) uniform in (-1, +1)
        Parameters
        ----------
        n : int, number of samples to draw
        q_min : float, minimal value of m2/m1 (q < 1 always)
        m_min : float, minimial mass of m1
        m_max : float, maximal mass of m1

        Returns
        -------
        intrinsic_samples : pandas.DataFrame
        """
        u = qmc.Halton(d=9, seed=seed).random(n).T
        m1, m2, lw_m = self.draw_lnmchirp_lnq_uniform(q_min, m_min, m_max, u[:2])
        s1z, s2z = self.draw_s1z_s2z(m1, m2, u=u[2:4])
        iota, s1x_n, s1y_n, s1z, s2x_n, s2y_n, s2z, lw_ipsi = (
            self.draw_inplane_spins_weighted_inclination(
                s1z,
                s2z,
                m1,
                m2,
                f_ref,
                u[4:,],
                inc_faceon_factor,
            )
        )

        keys = [
            "m1",
            "m2",
            "s1z",
            "s1x_n",
            "s1y_n",
            "s2z",
            "s2x_n",
            "s2y_n",
            "iota",
            "log_prior_weights",
        ]
        values = [
            m1,
            m2,
            s1z,
            s1x_n,
            s1y_n,
            s2z,
            s2x_n,
            s2y_n,
            iota,
            lw_m + lw_ipsi,
        ]
        intrinsic_samples = pd.DataFrame(data={k: v for k, v in zip(keys, values)})

        return intrinsic_samples

    def draw_physical_prior_inplane_spins_and_inclination(
        self, s1z, s2z, m1, m2, f_ref, u=None
    ):
        if u is None:
            n = len(np.atleast_1d(m1))
            u = qmc.Halton(d=5).random(n).T

        costheta_jn_samples = -1 + 2 * u[0]
        theta_jn_samples = np.arccos(costheta_jn_samples)

        phi_jl_hat_samples = u[1] * np.pi * 2
        phi12_samples = u[2] * np.pi * 2
        cums1r_s1z_samples, cums2r_s2z_samples = u[3], u[4]

        chi1, tilt1 = self.inplane_spin_prior._spin_transform(cums1r_s1z_samples, s1z)
        chi2, tilt2 = self.inplane_spin_prior._spin_transform(cums2r_s2z_samples, s2z)
        phi_jl_samples = (phi_jl_hat_samples + np.pi * (costheta_jn_samples < 0)) % (
            2 * np.pi
        )

        (iota, s1x_n, s1y_n, s1z, s2x_n, s2y_n, s2z) = np.array(
            [
                SimInspiralTransformPrecessingNewInitialConditions(
                    theta_jn_samples[i],
                    phi_jl_samples[i],
                    tilt1[i],
                    tilt2[i],
                    phi12_samples[i],
                    chi1[i],
                    chi2[i],
                    m1[i] * MSUN_SI,
                    m2[i] * MSUN_SI,
                    f_ref,
                    phiRef=0.0,
                )
                for i in range(u.shape[1])
            ]
        ).T

        return iota, s1x_n, s1y_n, s1z, s2x_n, s2y_n, s2z

    def draw_physical_prior_samples(
        self,
        q_min,
        mchirp_min,
        mchirp_max,
        f_ref,
        *,
        n_samples=None,
        u=None,
        draw_method="mc",
    ):
        mass_prior = UniformDetectorFrameMassesPrior(
            mchirp_range=(mchirp_min, mchirp_max), q_min=q_min
        )

        if u is None and n_samples is not None and draw_method is not None:
            if draw_method == "mc":
                u = np.random.rand(9, n_samples)
            elif draw_method == "qmc":
                u = qmc.Halton(d=9).random(n_samples).T
            else:
                raise ValueError("draw_method must be 'mc' or 'qmc'")
        elif u is None or n_samples is None or draw_method is None:
            raise ValueError(
                "You must pass either 'u' or both 'n_samples' and 'draw_method'"
            )

        mass_samples = mass_prior.generate_random_samples(n_samples)
        m1 = mass_samples["m1"].values
        m2 = mass_samples["m2"].values
        s1z, s2z = self.draw_s1z_s2z(m1, m2)
        iota, s1x_n, s1y_n, s1z, s2x_n, s2y_n, s2z = (
            self.draw_physical_prior_inplane_spins_and_inclination(
                s1z, s2z, m1, m2, f_ref, u
            )
        )
        samples = pd.DataFrame(
            dict(
                m1=m1,
                m2=m2,
                s1x_n=s1x_n,
                s1y_n=s1y_n,
                s1z=s1z,
                s2x_n=s2x_n,
                s2y_n=s2y_n,
                s2z=s2z,
                iota=iota,
                l1=0,
                l2=0,
            )
        )
        return samples

    def draw_samples_with_drawing_prior(
        self, n_samples, seed=None, draw_method="qmc", f_ref=None
    ):
        """
        Draw samples using the drawing prior and calculate importance sampling weights.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        seed : int, optional
            Random seed
        draw_method : str, optional
            Method to use for drawing samples ("qmc" for quasi-Monte Carlo, "mc" for Monte Carlo)
        f_ref : float, optional
            Reference frequency. Must be provided if the drawing prior requires it.

        Returns
        -------
        pd.DataFrame
            DataFrame with samples and importance sampling weights
        """
        if f_ref is None:
            f_ref = DEFAULT_F_REF
        if draw_method == "qmc":
            u = (
                qmc.Halton(d=len(self.drawing_prior.sampled_params), seed=seed)
                .random(n_samples)
                .T
            )
        elif draw_method == "mc":
            np.random.seed(seed)
            u = np.random.rand(len(self.drawing_prior.sampled_params), n_samples)
        else:
            raise ValueError("draw_method must be 'qmc' or 'mc'")

        # Generate samples using drawing prior
        samples_list = []
        log_weights = np.zeros(n_samples)

        for i in range(n_samples):
            # Get sampled parameter values
            sampled_params = {}
            for j, param_name in enumerate(self.drawing_prior.sampled_params):
                param_range = self.drawing_prior.range_dic[param_name]
                if param_name.startswith("log"):
                    # Log-uniform sampling
                    sampled_params[param_name] = param_range[0] + u[j, i] * (
                        param_range[1] - param_range[0]
                    )
                else:
                    # Uniform sampling
                    sampled_params[param_name] = param_range[0] + u[j, i] * (
                        param_range[1] - param_range[0]
                    )

            # Transform to standard parameters
            standard_params = self.drawing_prior.transform(
                **sampled_params, f_ref=f_ref
            )

            # Calculate importance sampling weights
            if self.drawing_prior != self.physical_prior:
                # Calculate log prior ratio
                from cogwheel.prior_ratio import PriorRatio

                if (
                    "f_ref" in self.physical_prior.conditioned_on
                    and "f_ref" not in standard_params
                ):
                    standard_params["f_ref"] = f_ref

                prior_ratio = PriorRatio(self.physical_prior, self.drawing_prior)
                log_weights[i] = prior_ratio.ln_prior_ratio(**standard_params)

            # Add weights to the sample
            standard_params["log_prior_weights"] = log_weights[i]
            samples_list.append(standard_params)

        return pd.DataFrame(samples_list)


def create_physical_prior_bank(
    bank_size,
    q_min,
    m_min,
    m_max,
    f_ref,
    fbin,
    n_blocks=None,
    n_pool=1,
    blocksize=4096,
    i_start=0,
    i_end=None,
    bank_dir=".",
    seed=None,
    approximant="IMRPhenomXODE",
):
    """
    Generate intrinsic samples and waveforms for a given bank.
    """
    print("Generating intrinsic samples")
    bank_dir = Path(bank_dir)
    if not bank_dir.exists():
        bank_dir.mkdir(parents=True)
    # create the generator
    generator = IntrinsicSamplesGenerator(
        LogUniformMassBiasedInclinationDrawingPrior(m_min, m_max, q_min, 2)
    )
    # draw samples
    intrinsic_samples = generator.draw_physical_prior_samples(
        q_min,
        m_min,
        m_max,
        f_ref,
        n_samples=bank_size,
    )

    if isinstance(fbin, (str, Path)):
        fbin = np.load(fbin)
    # save to file
    bank_file_path = bank_dir / "intrinsic_sample_bank.feather"
    bank_config_path = bank_dir / "bank_config.json"
    with open(bank_config_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "q_min": q_min,
                "min_mchirp": m_min,
                "max_mchirp": m_max,
                "f_ref": f_ref,
                "fbin": fbin,
                "seed": seed,
                "bank_size": bank_size,
                "approximant": approximant,
            },
            fp=fp,
            cls=NumpyEncoder,
            indent=4,
        )
    intrinsic_samples.to_feather(bank_file_path)
    print("Saved intrinsic samples to", bank_file_path)

    # create waveforms
    waveform_dir = bank_dir / "waveforms"
    print("Creating waveforms")
    waveform_banks.create_waveform_bank_from_samples(
        samples_path=bank_file_path,
        bank_config_path=bank_config_path,
        waveform_dir=waveform_dir,
        n_pool=n_pool,
        blocksize=blocksize,
        n_blocks=n_blocks,
        i_start=i_start,
        i_end=i_end,
        approximant=approximant,
    )

    print("waveform bank created at", waveform_dir)


def main(
    bank_size,
    q_min,
    m_min,
    m_max,
    inc_faceon_factor,
    f_ref,
    fbin,
    n_blocks=None,
    n_pool=1,
    blocksize=4096,
    i_start=0,
    i_end=None,
    bank_dir=".",
    seed=None,
    approximant="IMRPhenomXODE",
):
    """
    Generate intrinsic samples and waveforms for a given bank.
    """
    print("Generating intrinsic samples")
    bank_dir = Path(bank_dir)
    if not bank_dir.exists():
        bank_dir.mkdir(parents=True)

    # Create drawing prior
    drawing_prior = LogUniformMassBiasedInclinationDrawingPrior(
        m_min=m_min, m_max=m_max, q_min=q_min, inc_faceon_factor=inc_faceon_factor
    )

    # Create physical prior
    mchirp_min = m1m2_to_mchirp(m_min, m_min * q_min)
    mchirp_max = m1m2_to_mchirp(m_max, m_max)

    # Create default physical prior using cogwheel priors
    physical_prior = DefaultPhysicalPrior(
        mchirp_range=(mchirp_min, mchirp_max), q_min=q_min, f_ref=f_ref
    )

    # create the generator
    generator = IntrinsicSamplesGenerator(drawing_prior, physical_prior)

    # draw samples
    intrinsic_samples = generator.draw_samples_with_drawing_prior(
        n_samples=bank_size, seed=seed, draw_method="qmc", f_ref=f_ref
    )

    if isinstance(fbin, (str, Path)):
        fbin = np.load(fbin)
    # save to file
    bank_file_path = bank_dir / "intrinsic_sample_bank.feather"
    bank_config_path = bank_dir / "bank_config.json"
    with open(bank_config_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "q_min": q_min,
                "min_mchirp": m_min,
                "max_mchirp": m_max,
                "f_ref": f_ref,
                "fbin": fbin,
                "seed": seed,
                "bank_size": bank_size,
                "inc_faceon_factor": inc_faceon_factor,
                "approximant": approximant,
            },
            fp=fp,
            cls=NumpyEncoder,
            indent=4,
        )
    intrinsic_samples.to_feather(bank_file_path)
    print("Saved intrinsic samples to", bank_file_path)

    # create waveforms
    waveform_dir = bank_dir / "waveforms"
    print("Creating waveforms")
    waveform_banks.create_waveform_bank_from_samples(
        samples_path=bank_file_path,
        bank_config_path=bank_config_path,
        waveform_dir=waveform_dir,
        n_pool=n_pool,
        blocksize=blocksize,
        n_blocks=n_blocks,
        i_start=i_start,
        i_end=i_end,
        approximant=approximant,
    )

    print("waveform bank created at", waveform_dir)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate intrinsic samples " + "and waveforms for a given bank."
    )
    parser.add_argument(
        "--bank_size",
        type=int,
        required=True,
        help="Number of samples to draw",
    )
    parser.add_argument(
        "--q_min",
        type=float,
        required=True,
        help="Minimal value of m2/m1 (q < 1 always)",
    )
    parser.add_argument("--m_min", type=float, required=True, help="Minimal mass of m1")
    parser.add_argument("--m_max", type=float, required=True, help="Maximal mass of m1")
    parser.add_argument("--inc_faceon_factor", type=float, default=2)
    parser.add_argument(
        "--f_ref", type=float, required=True, help="Reference frequency"
    )
    parser.add_argument(
        "--fbin",
        type=Path,
        required=True,
        help="Path to frequency bins npy file",
    )
    parser.add_argument("--n_blocks", type=int, default=None, help="Number of blocks")
    parser.add_argument("--n_pool", type=int, default=1, help="Number of pools")
    parser.add_argument("--blocksize", type=int, default=4096, help="Block size")
    parser.add_argument("--i_start", type=int, default=0, help="Start index")
    parser.add_argument("--i_end", type=int, default=None, help="End index")
    parser.add_argument("--bank_dir", type=str, default=".", help="Bank directory")
    parser.add_argument(
        "--approximant",
        type=str,
        default="IMRPhenomXODE",
        help="Waveform approximant to use",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    args = parse_args()
    main(**args)
