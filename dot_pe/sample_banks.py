"""
Generate intrinsic samples and waveforms for a bank.
# TDOD: Add m_arr to the `bank_config.json` file.
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
from dot_pe.mass_prior import get_mass_prior
from cogwheel.gw_utils import m1m2_to_mchirp
from cogwheel.utils import NumpyEncoder

from . import waveform_banks
from .utils import validate_q_bounds


class IntrinsicSamplesGenerator:
    """
    Class to generate intrinsic samples, with importance sampling.
    The samples are drawn under:
    - log(m1) is uniform between log(m_min) and log(m_max)
    - log(m2) is uniform in log(m1*q_min) and log(m1)
    - cos(theta_jn) uniform in (-1, +1)
    Samples are re-weighted according to the Physical prior:
    - m1,m2 from mass.UniformDetectorFrameMassesPrior

    """

    def __init__(self):
        self.effective_spin_prior = UniformEffectiveSpinPrior()
        self.inplane_spin_prior = UniformDiskInplaneSpinsIsotropicInclinationPrior()

    def draw_lnmchirp_lnq_uniform(
        self, q_min, q_max, mchirp_min, mchirp_max, u=None, n=None
    ):
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

        mass_prior = get_mass_prior(
            mchirp_range=(mchirp_min, mchirp_max),
            q_min=q_min,
            q_max=q_max,
        )

        ln_mchirp = np.log(mchirp_min) + u[0] * np.log(mchirp_max / mchirp_min)
        mchirp = np.exp(ln_mchirp)

        lnq = np.log(q_min) + u[1] * np.log(q_max / q_min)

        m1 = mchirp * (1 + np.exp(lnq)) ** (1 / 5) / np.exp(lnq) ** (3 / 5)
        m2 = m1 * np.exp(lnq)

        area = np.log(mchirp_max / mchirp_min) * np.log(q_max / q_min)
        log_jacobian = -ln_mchirp  # log(det|d(log_mchirp,lnq)/(dmchirp,lnq)|
        # make units match (mchirp, lnq) with mass_prior
        fiducial_log_prior = np.log(1 / area) + log_jacobian
        log_prior = np.vectorize(mass_prior.lnprior, otypes=[float])(mchirp, lnq)
        log_prior_weights = log_prior - fiducial_log_prior

        return m1, m2, log_prior_weights

    def draw_m1m2_loguniform(
        self, q_min, q_max, m_min, m_max, u=None, n=None
    ):
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
        # m2 is uniform in log space between m1*q_min and m1*q_max
        logm2 = logm1 + np.log(q_min) + u[1] * np.log(q_max / q_min)
        m1, m2 = np.exp(logm1), np.exp(logm2)

        mchirp = m1m2_to_mchirp(m1, m2)
        lnq = np.log(m2 / m1)
        # the
        mchirp_min = m1m2_to_mchirp(m_min, m_min * q_min)
        mchirp_max = m1m2_to_mchirp(m_max, m_max)
        mass_prior = get_mass_prior(
            mchirp_range=(mchirp_min, mchirp_max),
            q_min=q_min,
            q_max=q_max,
        )

        # find weight = pdf(m1,m2) / pdf_pseudo(m1,m2).
        pseudo_prior_pdf = (
            1
            / (np.log(m_max / m_min))
            * 1
            / np.log(q_max / q_min)
            * 1
            / m1
            * 1
            / m2
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
        d = self.effective_spin_prior.transform(*args)
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
        self, s1z, s2z, m1, m2, f_ref, u=None, inc_faceon_factor=None
    ):
        """
        Draw inplane spins and inclination uniform cos_theta_jn, chieff,
        cums1r_s1z, cums1r_s1z, and inplane-spins angles.
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
            Ignored parameter kept for backward compatibility.

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
        if inc_faceon_factor is not None:
            import warnings

            warnings.warn(
                "inc_faceon_factor parameter is deprecated and will be removed in a future version",
                DeprecationWarning,
            )
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

        # No reweighting needed since cosine_jn is drawn uniformly
        log_prior_weights = np.zeros_like(costheta_jn_samples)
        return iota, s1x_n, s1y_n, s1z, s2x_n, s2y_n, s2z, log_prior_weights

    def draw_intrinsic_samples_uniform_in_ln_masses(
        self,
        n,
        q_min,
        q_max,
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
        - cos(theta_jn) uniform in (-1, +1)
        Samples are re-weighted according to the Physical prior:
        - m1,m2 from mass.UniformDetectorFrameMassesPrior
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
        m1, m2, lw_m = self.draw_m1m2_loguniform(
            q_min, q_max, m_min, m_max, u[:2]
        )
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
        self, n, q_min, q_max, m_min, m_max, inc_faceon_factor, f_ref, seed=None
    ):
        """
        Draw intrinsic samples, with importance sampling.
        The samples are drawn under:
        - log(mchirp) is uniform between log(mchirp_min) and log(mchirp_max)
        - log(q) is uniform in log(q_min) and log(1)
        - cos(theta_jn) uniform in (-1, +1)
        Samples are re-weighted according to the Physical prior:
        - m1,m2 from mass.UniformDetectorFrameMassesPrior
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
        m1, m2, lw_m = self.draw_lnmchirp_lnq_uniform(
            q_min, q_max, m_min, m_max, u[:2]
        )
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
    generator = IntrinsicSamplesGenerator()
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


def find_missing_block_indices(waveform_dir, blocksize, n_samples):
    """
    Find all missing block indices based on existing waveform files.

    Parameters
    ----------
    waveform_dir : Path
        Directory containing waveform block files
    blocksize : int
        Number of samples per block
    n_samples : int
        Total number of samples

    Returns
    -------
    list
        List of missing block indices
    """
    if not waveform_dir.exists():
        # If directory doesn't exist, all blocks are missing
        n_blocks = -(n_samples // -blocksize)
        return list(range(n_blocks))

    # Calculate total expected blocks (ceil division)
    n_blocks = -(n_samples // -blocksize)
    missing_blocks = []

    # Find all incomplete blocks
    for i in range(n_blocks):
        amp_file = waveform_dir / f"amplitudes_block_{i}.npy"
        phase_file = waveform_dir / f"phase_block_{i}.npy"
        if not (amp_file.exists() and phase_file.exists()):
            missing_blocks.append(i)

    return missing_blocks


def find_next_block_index(waveform_dir, blocksize, n_samples):
    """
    Find the next block index to start from (for backward compatibility).

    Parameters
    ----------
    waveform_dir : Path
        Directory containing waveform block files
    blocksize : int
        Number of samples per block
    n_samples : int
        Total number of samples

    Returns
    -------
    int
        Next block index to start generation from
    """
    missing_blocks = find_missing_block_indices(waveform_dir, blocksize, n_samples)
    if not missing_blocks:
        # All blocks exist
        n_blocks = -(n_samples // -blocksize)
        return n_blocks
    else:
        # Return the first missing block
        return missing_blocks[0]


def main(
    bank_size,
    q_min,
    m_min,
    m_max,
    q_max=1.0,
    inc_faceon_factor,
    f_ref,
    fbin,
    n_blocks=None,
    n_pool=1,
    blocksize=4096,
    i_start=None,
    i_end=None,
    i_list=None,
    bank_dir=".",
    seed=None,
    approximant="IMRPhenomXODE",
    resume=True,
):
    """
    Generate intrinsic samples and waveforms for a given bank.
    """
    bank_dir = Path(bank_dir)
    if not bank_dir.exists():
        bank_dir.mkdir(parents=True)

    bank_file_path = bank_dir / "intrinsic_sample_bank.feather"
    bank_config_path = bank_dir / "bank_config.json"

    # Check if samples already exist and continue is True
    if resume and bank_file_path.exists():
        print(
            f"Found existing intrinsic samples at {bank_file_path}, skipping sample generation"
        )
        intrinsic_samples = pd.read_feather(bank_file_path)
    else:
        print("Generating intrinsic samples")
        # create the generator
        generator = IntrinsicSamplesGenerator()
        # draw samples
        intrinsic_samples = generator.draw_intrinsic_samples_uniform_in_lnmchrip_lnq(
            bank_size,
            q_min,
            q_max,
            m_min,
            m_max,
            inc_faceon_factor,
            f_ref,
            seed,
        )

        if isinstance(fbin, (str, Path)):
            fbin = np.load(fbin)
        # save to file
        with open(bank_config_path, "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "q_min": q_min,
                    "q_max": q_max,
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

    # Load fbin from config if samples were loaded from file
    if resume and bank_file_path.exists() and bank_config_path.exists():
        with open(bank_config_path, "r", encoding="utf-8") as fp:
            config_dict = json.load(fp)
            if isinstance(config_dict["fbin"], list):
                fbin = np.array(config_dict["fbin"])
            else:
                fbin = config_dict["fbin"]
                if isinstance(fbin, (str, Path)):
                    fbin = np.load(fbin)

    # create waveforms
    waveform_dir = bank_dir / "waveforms"

    # Determine which blocks to generate (priority: i_list > auto-detect > i_start/i_end)
    if i_list is not None:
        print(f"Using provided i_list with {len(i_list)} blocks: {i_list}")
    elif resume and i_start is None:
        # Auto-detect missing blocks when resuming
        missing_blocks = find_missing_block_indices(
            waveform_dir, blocksize, len(intrinsic_samples)
        )
        if missing_blocks:
            i_list = missing_blocks
            print(
                f"Auto-detected {len(missing_blocks)} missing blocks: {missing_blocks[:10]}{'...' if len(missing_blocks) > 10 else ''}"
            )
        else:
            i_list = []
            print("No missing blocks found, nothing to generate")
    else:
        # Use traditional i_start/i_end logic for backward compatibility
        if i_start is None:
            i_start = 0
            print("Starting waveform generation from beginning (resume=False)")

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
        i_list=i_list,
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
    parser.add_argument(
        "--q_max",
        type=float,
        default=1.0,
        help="Maximal value of m2/m1, must satisfy q_min < q_max <= 1.0 (default: 1.0)",
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
    parser.add_argument(
        "--i_start", type=int, default=None, help="Start index (auto-detected if None)"
    )
    parser.add_argument("--i_end", type=int, default=None, help="End index")
    parser.add_argument(
        "--i_list",
        type=int,
        nargs="*",
        default=None,
        help="List of specific block indices to generate (overrides i_start/i_end and auto-detection)",
    )
    parser.add_argument("--bank_dir", type=str, default=".", help="Bank directory")
    parser.add_argument(
        "--approximant",
        type=str,
        default="IMRPhenomXODE",
        help="Waveform approximant to use",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing files if they exist (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Overwrite existing files instead of resuming",
    )

    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    args = parse_args()
    validate_q_bounds(args["q_min"], args["q_max"])
    main(**args)
