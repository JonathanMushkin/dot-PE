"""
A module to perform the sample handling, likelihood and posterior
calculations for the free-sampling method.

TODO: split to separate modules for the samples-handling and the
likelihood calculations.
JM 29.5.24
"""

import itertools
import json
from pathlib import Path

import lalsimulation as lalsim
import numpy as np
import pandas as pd
from lal import CreateDict
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import logsumexp
from scipy.stats.qmc import Halton

from cogwheel import likelihood
from cogwheel.gw_utils import get_fplus_fcross_0, get_geocenter_delays
from cogwheel.likelihood.relative_binning import BaseLinearFree
from cogwheel.waveform import FORCE_NNLO_ANGLES, compute_hplus_hcross_by_mode
from cogwheel.waveform_models.xode import compute_hplus_hcross_by_mode_xode

from . import config
from .sampler_free_utils import safe_logsumexp

lalsimulation_commands = FORCE_NNLO_ANGLES

# The marginalization over distance for the likelihood assumes that
# the maximal-likelihood d_luminosity (=sqrt(<h|h>)/<d|h>) smaller than
# the maximal allowed distance by more than the width of the Gaussian.
# This translated to z > h_norm/(15e3) + (a few)
# We approximate the RHS of this inequality with 4
MIN_Z_FOR_LNL_DIST_MARG_APPROX = 4


def compute_hplus_hcross_safe(
    f, par_dic, approximant, harmonic_modes, harmonic_modes_by_m, lal_dic=None
):
    """
    Compute hplus and hcross for a given frequency, parameters and
    approximant. If the approximant is IMRPhenomXODE, use the
    specialized function for that approximant. Otherwise, use the
    generic function.

    Parameters:
    f: array of frequencies
    par_dic: dictionary of parameters
    approximant: string, name of the approximant. See `waveform.APPROXIMANTS`
    for details.
    harmonic_modes: list of modes to compute
    harmonic_modes_by_m: dictionary of modes by m
    lal_dic: LAL dictionary
    """
    if approximant == "IMRPhenomXODE":
        hplus_hcross_modes = compute_hplus_hcross_by_mode_xode(
            f,
            par_dic,
            approximant=approximant,
            harmonic_modes=harmonic_modes,
            lal_dic=lal_dic,
        )
    else:
        hplus_hcross_modes = compute_hplus_hcross_by_mode(
            f,
            par_dic,
            approximant=approximant,
            harmonic_modes=harmonic_modes,
            lal_dic=lal_dic,
        )

    h_mpf = np.array(
        [
            np.sum([hplus_hcross_modes[mode] for mode in m_modes], axis=0)
            for m_modes in harmonic_modes_by_m.values()
        ]
    )

    return h_mpf


def create_lal_dict():
    """Return a LAL dict object per ``self.lalsimulation_commands``."""
    lal_dic = CreateDict()
    for function_name, value in lalsimulation_commands:
        getattr(lalsim, function_name)(lal_dic, value)
    return lal_dic


def get_shift(f, t):
    """
    unhsifted h is centered at event_data.tcoarse. further shifts are
    due to relative times (t_geonceter + time_delays) = t_refdet
    """
    return np.exp(-2j * np.pi * f * t)


def pick_top_values(x, frac):
    """
    Given array x = log(y), return the indices of the x to keep,
    such that the sum y over the kept indices is  (1-frac) of the
    total integral. Use binary-search-like algorithm to set a bar and
    take all values above it. To reduce runtime, the code:
    1) Iteratively reduce the size of the array by removing
    elements below the lower bar and above the upper bar.
    2) Save the sum of terms above the upper bar.
    Inputs:
    x: array of log-values
    frac: fraction of the total integral to discard
    Output:
    inds: array of indices of the kept values.
    """
    log_sum = logsumexp(x)
    n = len(x)
    target_log_relative_error = np.log(frac)
    # if cannot remove even a single term
    if x.min() - log_sum > target_log_relative_error:
        return np.arange(n)
    number_of_iterations = int(np.log2(n))
    # zero step
    upper_bar = x.max()
    lower_bar = x.min()
    log_sum_above_upper_bar = -np.inf
    xx = x.copy()
    for _ in range(number_of_iterations):
        x_bar = (upper_bar + lower_bar) / 2
        cond = xx > x_bar
        if any(cond):
            log_sum_above_bar = logsumexp(xx[cond])
            # include terms above upper bar
            log_sum_above_bar = np.logaddexp(log_sum_above_bar, log_sum_above_upper_bar)
            log_relative_error = np.log(1 - np.exp(log_sum_above_bar - log_sum))
        else:
            break

        if log_relative_error > target_log_relative_error:
            # lower the upper bar
            upper_bar = x_bar
            log_sum_above_upper_bar = log_sum_above_bar
        else:
            # raise lower bar
            lower_bar = x_bar
        cond = (xx > lower_bar) * (xx <= upper_bar)
        xx = xx[cond]
        if len(xx) == 0:
            # no terms are between lower and upper bars.
            break
    inds = np.where(x > x_bar)[0]
    return inds


class LinearFree(BaseLinearFree):
    """
    Minimal likelihood class for IntrinsicSampleProcessor, namely
    the relative-binning linear-free timeshift summary data.
    """

    def lnlike_and_metadata(self, par_dic):
        """Not implemented, but needed for instantiation."""
        raise NotImplementedError


class Evidence:
    """
    A class that receives as input the components of intrinsic and
    extrinsic factorizations and calculates the likelihood at each
    (int., ext., phi) combination (distance marginalized).
    Indexing and size convention:
        * i: intrinsic parameter
        * m: modes, or combinations of modes
        * p: polarization, plus (0) or cross (1)
        * b: frequency bin (as in relative binning)
        * e: extrinsic parameters
        * d: detector
    """

    def __init__(self, n_phi, m_arr, lookup_table=None):
        """
        Initialize the Evidence object.
        Parameters:
        n_phi : number of points to evaluate phi on,
        m_arr : modes
        lookup_table = lookup table for distance marginalization
        """
        self.n_phi = n_phi
        self.phi_grid = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        self.m_arr = np.asarray(m_arr)
        # Used for terms with 2 modes indices
        self.m_inds, self.mprime_inds = zip(
            *itertools.combinations_with_replacement(range(len(self.m_arr)), 2)
        )

        self.lookup_table = lookup_table or likelihood.LookupTable()

    @staticmethod
    def get_dh_by_mode(
        dh_weights_dmpb, h_impb, response_dpe, timeshift_dbe, asd_drift_d
    ):
        """
        Calculate the inner product <d|h> for each combination of
        intrinsic sample, extrinsic sample and mode.
        Equivalent, but more efficient, than:
        dh_iem = np.einsum('dmpb, impb, dpe, dbe, d -> iem',
                            dh_weights_dmpb, h_impb.conj(), response_dpe,
                            timeshift_dbe.conj(), asd_drift_d**-2,
                            optimize=True)
        Parameters:
        dh_weights_dmpb: array weights for detector, mode, polarization,
        and frequency bin. Assumes it was calculated from the integrand
        integrand = data * h.conj()
        h_impb: array of waveforms.
        response_dpe: array of detector response.
        timeshift_dbe: array of time shifts.
        asd_drift_d: array of ASD drifts.
        Output:
        dh_iem: array of inner products.
        """

        i, m, p, b = h_impb.shape
        d, *_, e = response_dpe.shape
        x = d * p * b  # size of summed dimensions

        # reshape & conjugate arrays
        dh_weights_mx = np.moveaxis(dh_weights_dmpb, 0, 1).reshape(m, x)

        h_conj_mix = np.repeat(
            np.moveaxis(h_impb.conj(), 0, 1)[:, :, None, ...], d, axis=2
        ).reshape(m, i, x)

        response_drift_dpe = response_dpe * asd_drift_d[:, None, None] ** -2
        response_drift_xe = np.repeat(
            response_drift_dpe[:, :, None, :], b, axis=2
        ).reshape(x, e)

        timeshift_xe = np.reshape(
            timeshift_dbe.conj()[:, None, :, :].repeat(p, axis=1), (x, e)
        )
        ext_tensor = timeshift_xe * response_drift_xe
        dh_mie = np.empty((m, i, e), dtype=h_impb.dtype)

        for _m in range(m):
            dh_mie[_m] = np.dot(dh_weights_mx[_m] * h_conj_mix[_m], ext_tensor)

        return np.moveaxis(dh_mie, (0, 1), (2, 0))

    @staticmethod
    def get_hh_by_mode(
        h_impb,
        response_dpe,
        hh_weights_dmppb,
        asd_drift_d,
        m_inds,
        mprime_inds,
    ):
        """
        Calculate the inner priducts <h|h> for each combination of
        intrinsic sample, extrinsic sample and modes-pair.
        Modes m mean unique modes combinations: (2,2), (2,0), ..., (3,3)
        (over all 10 combinations).
        Inputs:
        h_impb: array of waveforms.
        response_dpe: array of detector response.
        hh_weights_dmppb: array of weights for detector. Assumes the
        weights are calculated for integrand h[m]*h[mprime].conj()
        asd_drift_d: array of ASD drifts.
        m_inds: indices of modes.
        mprime_inds: indices of modes.
        Output:
        hh_iem: array of inner products.
        """
        hh_idmpP = np.einsum(
            "dmpPb, impb, imPb -> idmpP",
            hh_weights_dmppb,
            h_impb[:, m_inds, ...],
            h_impb.conj()[:, mprime_inds, ...],
            optimize=True,
        )  # idmpp
        ff_dppe = np.einsum(
            "dpe, dPe, d -> dpPe",
            response_dpe,
            response_dpe,
            asd_drift_d**-2,
            optimize=True,
        )  # dppe
        hh_iem = np.einsum(
            "idmpP, dpPe -> iem", hh_idmpP, ff_dppe, optimize=True
        )  # iem
        return hh_iem

    def get_dh_hh_phi_grid(self, dh_iem, hh_iem):
        """
        Apply orbital phase shifts to <d|h>, <h|h> (by mode) and combine
        the results.

        Phase change phi manifests itself as exp(+j*phi*m) to each mode
        m of h, or as exp(-j*phi*m) to mode m of h.conj

        Parameters:
        dh_iem : array-like, (N_int, N_ext, N_modes)
                inner product of data with waveform, per intrinsic
                sample, extrinsic sample, and mode
        hh_iem : array-like, (N_int, N_ext, N_modes)
                inner product of waveform with itself, per intrinsic
                sample, extrinsic sample, and mode pairs.

        Return:
        dh_ieo: array-like, (N_int, N_ext, N_phi)
                inner product of data with waveform, per intrinsic
                sample, extrinsic sample, orbital phase.
        hh_ieo: array-like, (N_int, N_ext, N_phi)
                inner product of waveform with itself, per intrinsic
                sample, extrinsic sample, orbital phase
        """

        phi_grid = np.linspace(0, 2 * np.pi, self.n_phi, endpoint=False)  # o
        # dh_phasor is the phase shift applied to h[m].conj, hence the
        # minus sign
        dh_phasor = np.exp(-1j * np.outer(self.m_arr, phi_grid))  # mo
        # hh_phasor is applied both to h[m] and h[mprime].conj, hence
        # the subtraction of the two modes
        hh_phasor = np.exp(
            1j
            * np.outer(
                self.m_arr[self.m_inds,] - self.m_arr[self.mprime_inds,],
                phi_grid,
            )
        )  # mo
        dh_ieo = np.einsum(
            "iem, mo -> ieo", dh_iem, dh_phasor, optimize=True
        ).real  # ieo
        hh_ieo = np.einsum(
            "iem, mo -> ieo", hh_iem, hh_phasor, optimize=True
        ).real  # ieo

        return dh_ieo, hh_ieo

    @staticmethod
    def select_ieo_by_approx_lnlike_dist_marginalized(
        dh_ieo,
        hh_ieo,
        log_prior_weights_i,
        log_prior_weights_e,
        cut_threshold=20.0,
    ):
        """
        Return three arrays with intrinsic, extrinsic and phi sample
        indices.
        """
        h_norm = np.sqrt(hh_ieo).astype(np.float32)
        z = dh_ieo / h_norm
        i_inds, e_inds, o_inds = np.where(z > MIN_Z_FOR_LNL_DIST_MARG_APPROX)
        lnl_approx = (
            z[i_inds, e_inds, o_inds] ** 2 / 2
            + 3 * np.log(h_norm[i_inds, e_inds, o_inds])
            - 4 * np.log(z[i_inds, e_inds, o_inds])
            + log_prior_weights_i[i_inds]
            + log_prior_weights_e[e_inds]
        )  # 1d

        flattened_inds = np.where(lnl_approx >= lnl_approx.max() - cut_threshold)[0]

        return (
            i_inds[flattened_inds],
            e_inds[flattened_inds],
            o_inds[flattened_inds],
        )

    @staticmethod
    def select_ieo_by_bestfit_lnlike(dh_ieo, hh_ieo, cut_threshold=20.0):
        """
        Select elements ieo by having distance-fitted lnlike not less
        than cut_threshold below the maximum.
        Return three arrays with intrinsic, extrinsic and phi sample
        indices.
        """
        lnl_approx = 0.5 * (dh_ieo**2) / hh_ieo * (dh_ieo > 0)
        i_inds, e_inds, o_inds = np.where(lnl_approx > lnl_approx.max() - cut_threshold)
        return (i_inds, e_inds, o_inds)

    @staticmethod
    def evaluate_log_evidence(lnlike, log_prior_weights, n_samples):
        """
        Evaluate the logarithm of the evidence (ratio) integral
        """
        return logsumexp(lnlike + log_prior_weights) - np.log(n_samples)

    def calculate_lnlike_and_evidence(
        self,
        dh_weights_dmpb,
        h_impb,
        response_dpe,
        timeshift_dbe,
        hh_weights_dmppb,
        asd_drift_d,
        log_prior_weights_i,
        log_prior_weights_e,
        approx_dist_marg_lnl_drop_threshold=20.0,
        n_samples=None,
        frac_evidence_threshold=0.0,
    ):
        """
        # TODO: Consider removing this code.
        Use stored samples to compute lnl (distance marginalized) on
        grid of intrinsic x extrinsic x phi samples.

        Parameters:

        dh_weights_dmpb: array relative-binning weights for the
        integrand data * h.conj() for each detector, mode, polarization,
        mode, polarization,and frequency bin.
        h_impb: array of waveforms, with shape (n_intrinsic, n_modes,
        n_polarizations, n_fbin)
        response_dpe: array of detector response, with shape
        (n_extrinsic, n_polarizations, n_fbin)
        timeshift_dbe: array of time shifts (complex exponents), with shape
        (n_extrinsic, n_detector, n_fbin)
        hh_weights_dmppb: array of relative-binning weights for the
        integrand h[m]*h[mprime].conj() for each detector, mode pair,
        polarization pair, and frequency bin.
        asd_drift_d: array of ASD drifts, with shape (n_detector)
        log_prior_weights_i: array of intrinsic log-weights due to
        importance sampling, per intrinsic sample.
        log_prior_weights_e: array of extrinsic log-weights due to
        importance sampling, per extrinsic sample.
        approx_dist_marg_lnl_drop_threshold: float, threshold for
        approximated distance marginalized lnlike, below which the
        sample is discarded.
        n_samples: int, overall number of samples (intrinsic x extrinsic
        x phases) used. Needed for normalization of the evidence. Could
        be different from the number of samples used in the calculation,
        due to rejection-sampling. If None, infered from `dh_ieo`.
        frac_evidence_threshold: float, set threshold on log-posterior
        probability for throwing out low-probability samples, such
        that (1-frac_evidence_threshold) of the total probability is
        retained.
        debug_mode: bool, if True, return additional arrays for
        debugging purposes.

        Output:
        dist_marg_lnlike_k: array of lnlike for each combination of
        intrinsic, extrinsic and phi sample.
        ln_evidence: float, ln(evidence) for the given samples.
        inds_i_k: array of intrinsic sample indices.
        inds_e_k: array of extrinsic sample indices.
        inds_o_k: array of phi sample indices.

        """

        dh_ieo, hh_ieo = self.get_dh_hh_ieo(
            dh_weights_dmpb,
            h_impb,
            response_dpe,
            timeshift_dbe,
            hh_weights_dmppb,
            asd_drift_d,
        )

        n_samples = n_samples or dh_ieo.size

        # introduce index k for serial index for (i, e, o) combinations
        # with high enough approximated distance marginalzed likelihood
        # i, e, o = inds_i_k[k], inds_e_k[k], inds_o_k[k]

        inds_i_k, inds_e_k, inds_o_k = (
            self.select_ieo_by_approx_lnlike_dist_marginalized(
                dh_ieo,
                hh_ieo,
                log_prior_weights_i,
                log_prior_weights_e,
                approx_dist_marg_lnl_drop_threshold,
            )
        )

        dist_marg_lnlike_k = self.lookup_table.lnlike_marginalized(
            dh_ieo[inds_i_k, inds_e_k, inds_o_k],
            hh_ieo[inds_i_k, inds_e_k, inds_o_k],
        )

        ln_posterior_k = (
            dist_marg_lnlike_k
            + log_prior_weights_i[inds_i_k]
            + log_prior_weights_e[inds_e_k]
            - np.log(n_samples)
        )

        ln_evidence = logsumexp(ln_posterior_k)

        if frac_evidence_threshold:
            # set a probability threshold frac_evidence_threshold = x
            # use it to throw the n lowest-probability samples,
            # such that the combined probability of the n thrown samples
            # is equal to x.

            log_prior_weights_k = (
                log_prior_weights_e[inds_e_k] + log_prior_weights_i[inds_i_k]
            )

            inds_k = pick_top_values(
                log_prior_weights_k + dist_marg_lnlike_k,
                frac_evidence_threshold,
            )

            dist_marg_lnlike_k = dist_marg_lnlike_k[inds_k]
            inds_i_k = inds_i_k[inds_k]
            inds_e_k = inds_e_k[inds_k]
            inds_o_k = inds_o_k[inds_k]
            log_prior_weights_k = log_prior_weights_k[inds_k]
            ln_evidence = logsumexp(dist_marg_lnlike_k + log_prior_weights_k) - np.log(
                n_samples
            )

        return (
            dist_marg_lnlike_k,
            ln_evidence,
            inds_i_k,
            inds_e_k,
            inds_o_k,
            dh_ieo[inds_i_k, inds_e_k, inds_o_k],
            hh_ieo[inds_i_k, inds_e_k, inds_o_k],
        )

    def calculate_marg_likelihood_ie(
        self,
        dh_weights_dmpb,
        h_impb,
        response_dpe,
        timeshift_dbe,
        hh_weights_dmppb,
        asd_drift_d,
    ):
        """
        Calculate the marginalized likelihood over phi and distnace
        for each combination of intrinsic and extrinsic samples.
        Do not apply and prior-related weights.
        """
        # compute the distance marginalized likelihood for each phi
        dh_ieo, hh_ieo = self.get_dh_hh_ieo(
            dh_weights_dmpb,
            h_impb,
            response_dpe,
            timeshift_dbe,
            hh_weights_dmppb,
            asd_drift_d,
        )
        dist_marg_lnlike_ieo = self.lookup_table.lnlike_marginalized(dh_ieo, hh_ieo)
        # marginalized over phi
        marg_lnlike_ie = logsumexp(dist_marg_lnlike_ieo, axis=-1) - np.log(self.n_phi)
        return marg_lnlike_ie

    def get_dh_hh_ieo(
        self,
        dh_weights_dmpb,
        h_impb,
        response_dpe,
        timeshift_dbe,
        hh_weights_dmppb,
        asd_drift_d,
    ):
        """
        Compute the inner products <d|h> and <h|h> for each combination
        of intrinsic samples (i) extrinsic samples (e) and orbital phase
        (o).

        Parameters:
        dh_weights_dmpb: array-like, complex
        Relative-binning weights for the integrand data * h.conj() for
        each detector, mode, polarization, and frequency bin.
        h_impb: array-like, complex
        response_dpe: array-array, float
        Detector response, with shape for each detector, polarization,
        and extrinsic sample.
        """

        dh_iem = self.get_dh_by_mode(
            dh_weights_dmpb, h_impb, response_dpe, timeshift_dbe, asd_drift_d
        )
        hh_iem = self.get_hh_by_mode(
            h_impb,
            response_dpe,
            hh_weights_dmppb,
            asd_drift_d,
            self.m_inds,
            self.mprime_inds,
        )
        # compute the distance marginalized likelihood for each phi
        dh_ieo, hh_ieo = self.get_dh_hh_phi_grid(dh_iem, hh_iem)
        return dh_ieo, hh_ieo

    def calculate_likelihoods_ieo(
        self,
        dh_weights_dmpb,
        h_impb,
        response_dpe,
        timeshift_dbe,
        hh_weights_dmppb,
        asd_drift_d,
    ):
        """
        Get the distance-marginalized and distance-fitted likelihoods
        for each combination of intrinsic and extrinsic samples.
        Use a single call to get_dh_hh_ieo to avoid redundant
        calculations.
        """
        dh_ieo, hh_ieo = self.get_dh_hh_ieo(
            dh_weights_dmpb,
            h_impb,
            response_dpe,
            timeshift_dbe,
            hh_weights_dmppb,
            asd_drift_d,
        )
        # distance-fitted likelihood
        lnlike_ieo = 1 / 2 * dh_ieo**2 / hh_ieo * (dh_ieo > 0)
        # distance-marginalized likelihood
        dist_marg_lnlike_ieo = self.lookup_table.lnlike_marginalized(dh_ieo, hh_ieo)
        return lnlike_ieo, dist_marg_lnlike_ieo

    def combine_samples(
        self,
        intrinsic,
        extrinsic,
        lnl_k,
        dt_linfree_i_k,
        dphi_linfree_i_k,
        inds_i_k,
        inds_e_k,
        inds_o_k,
        dh_k=None,
        hh_k=None,
    ):
        """
        Combine combinations of intrinsic, extrinsic and orbital phase
        indexed k to a single dataframe. Apply linear-free time and
        phase shifts to the samples.
        """

        combined_samples = pd.concat(
            [
                intrinsic.iloc[inds_i_k].reset_index(drop=True),
                extrinsic.iloc[inds_e_k].reset_index(drop=True),
            ],
            axis=1,
        )
        log_prior_weights_e = np.log(extrinsic["weights"].values)
        log_prior_weights_i = intrinsic["log_prior_weights"].values
        # add the extrinsic (log) weights to the intrinsic (log) weights in
        # the magic samples dataframe
        combined_samples["log_prior_weights"] = (
            log_prior_weights_i[inds_i_k] + log_prior_weights_e[inds_e_k]
        )
        combined_samples["phi"] = np.linspace(0, 2 * np.pi, self.n_phi, endpoint=False)[
            inds_o_k
        ]

        combined_samples["lnl"] = lnl_k
        # save separately the linear free times and phases,
        # and the standard convention
        combined_samples.rename(
            columns={
                "t_geocenter": "t_geocenter_linfree",
                "phi": "phi_ref_linfree",
            },
            inplace=True,
        )
        # see linear_free_timeshifts.py for details about the convention
        combined_samples["t_geocenter"] = (
            combined_samples["t_geocenter_linfree"] + dt_linfree_i_k
        )
        combined_samples["phi_ref"] = (
            combined_samples["phi_ref_linfree"] - dphi_linfree_i_k
        )
        # calculate probabilities per sample
        log_prob_unormalized_k = combined_samples["log_prior_weights"] + lnl_k
        # divide unormalized probabilities by the maximum to avoid overflow
        log_prob_unormalized_k -= log_prob_unormalized_k.max()

        prob_k = np.exp(log_prob_unormalized_k - logsumexp(log_prob_unormalized_k))
        combined_samples["weights"] = prob_k
        if not dh_k is None:
            combined_samples["dh"] = dh_k

        if not hh_k is None:
            combined_samples["hh"] = hh_k

        return combined_samples

    @staticmethod
    def get_effective_ns(inds_i_k, inds_e_k, prob_k):
        """
        Get effective number of samples, both overall, and of the
        intrinsic samples and of the extrinsic samples.

        Parameters:
        inds_i_k: array of intrinsic sample indices per combination k.
        inds_e_k: array of extrinsic sample indices per combination k.
        prob_k: array of probabilities per combination k.
        Output:
        n_eff: effective number of samples.
        n_eff_i: effective number of intrinsic samples.

        """
        n_effective = np.sum(prob_k) ** 2 / np.sum(prob_k**2)
        unique_i = np.unique(inds_i_k)
        unique_e = np.unique(inds_e_k)

        p_i = np.array([np.sum(prob_k[inds_i_k == i]) for i in unique_i])
        p_e = np.array([np.sum(prob_k[inds_e_k == e]) for e in unique_e])
        n_effective_int = np.sum(p_i) ** 2 / np.sum(p_i**2)
        n_effective_ext = np.sum(p_e) ** 2 / np.sum(p_e**2)

        return n_effective, n_effective_int, n_effective_ext


class ExtrinsicSampleProcessor:
    """
    A class to process extirnsic samples and creates the
    relevant high-dimensional arrays to pass to an Evidence class.
    """

    def __init__(self, detector_names):
        self.n_polarizations = 2
        self.lal_dic = create_lal_dict()
        self.detector_names = detector_names

    @staticmethod
    def compute_detector_responses(detector_names, lat, lon, psi):
        """Compute detector response at specific lat, lon and psi"""
        lat, lon, psi = np.atleast_1d(lat, lon, psi)
        fplus_fcross_0 = get_fplus_fcross_0(detector_names, lat, lon)  # edP
        psi_rot = np.array(
            [
                [np.cos(2 * psi), np.sin(2 * psi)],
                [-np.sin(2 * psi), np.cos(2 * psi)],
            ]
        )  # pPe
        return np.einsum(
            "edP, pPe-> edp", fplus_fcross_0, psi_rot, optimize=True
        )  # edp

    def compute_extrinsic_timeshift(self, detector_names, extrinsic_samples, f):
        """
        Compute extrinsic time shift for each detector, related to
        the relative position of the source and the detectors.
        input:
        detector_names: list of detector names
        extrinsic_samples: pandas dataframe with extrinsic parameters
        f: array of frequencies
        output:
        extrinsic_timeshift_exp: timeshift exponentials for each
                                 detector
        """
        # time difference related to the relative positions of the source
        # and the detectors (shape ed)
        geocentric_delays = get_geocenter_delays(
            detector_names,
            extrinsic_samples["lat"].values,
            extrinsic_samples["lon"].values,
        ).T

        # add geocentric delays to the time delays from the source
        total_delays = (
            geocentric_delays + extrinsic_samples["t_geocenter"].values[:, np.newaxis]
        )  # ed

        # timeshift exponentials for each detector (shape edf)
        extrinsic_timeshift_exp = np.exp(
            -2j * np.pi * total_delays[..., np.newaxis] * f[np.newaxis, np.newaxis, :]
        )  # edb
        return extrinsic_timeshift_exp

    def get_components(self, extrinsic_samples, fbin, tcoarse):
        """
        Compute the detector responses and the extrinsic time shifts
        for the extrinsic samples.
        """
        response_dpe = np.moveaxis(
            self.compute_detector_responses(
                self.detector_names,
                *[extrinsic_samples[x] for x in ["lat", "lon", "psi"]],
            ),
            (0, 1, 2),
            (2, 0, 1),
        )

        timeshifts_edb = self.compute_extrinsic_timeshift(
            self.detector_names, extrinsic_samples, fbin
        )

        # apply time shift to the extrinsic components, undo the time
        # shift from the relative binning weights.
        # Equivalent to the timeshift applied in
        # waveform.WaveformGenerator.get_hplus_hcross_at_detectors
        timeshifts_edb *= np.exp(-2j * np.pi * fbin * tcoarse)
        timeshifts_dbe = np.moveaxis(timeshifts_edb, (0, 1, 2), (2, 0, 1))
        return response_dpe, timeshifts_dbe

    def get_lon_lat_grid_and_distributions(
        self, det_name, lon_grid_size=2**10, lat_grid_size=2**10
    ):
        """
        Generate points on (x,y) grid which maps to (lon, lat) points,
        and evaluate p.d.f. and c.d.fs for drawing points.

        Paremeters
        ----------
        det_name : str,
            detector name, 'H','L' or 'V'
        lon_grid_size : int,
            number of grid points in longitude axis. Default 2**13
        lat_grid_size : int,
            number of grid points in latitude axis. Default 2**13.

        Return
        ------
        x : numpy.ndarray,
            grid points of uniform points mapped to lon.
        y : numpy.ndarray
            grid points of uniform points mapped to lat.
        cdf_x : numpy.ndarray
            CDF evaluated at x grid points.
        pdf_y_given_x numpy.ndarray (lon_grid_size, lat_grid_size)
            PDF of y given x, evaluated on grid points.

        """

        x = np.linspace(0, 1, lon_grid_size, endpoint=False)  # -> lon
        y = np.linspace(0, 1, lat_grid_size, endpoint=False)  # -> lat
        x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")

        def unnormalized_pdf_func(d, x, y):
            shape = x.shape
            lon = x.flatten() * np.pi * 2
            lat = np.arcsin(1 - 2 * y.flatten())
            response_mag_cubed = np.sum(
                self.compute_detector_responses(d, lat, lon, 0).take(0, 1) ** 2,
                axis=-1,
            ) ** (3 / 2)
            return response_mag_cubed.reshape(shape)

        # pdf define on the mesh-grid X, Y
        pdf_values = unnormalized_pdf_func(det_name, x_mesh, y_mesh)
        # normalize
        normalization = trapezoid(trapezoid(pdf_values, x=x, axis=0), x=y)
        pdf_values /= normalization

        pdf_x_values = trapezoid(pdf_values, y, axis=1)
        # pdf_y_values = trapezoid(pdf_values, x, axis=0)
        cdf_x = cumulative_trapezoid(pdf_x_values, x, initial=0)
        # pdf_y_given_x has dimensions (x,y), like all 2-d distributions
        # here
        pdf_y_given_x = pdf_values / pdf_x_values.reshape(-1, 1)

        return x, y, cdf_x, pdf_y_given_x, normalization

    def uniform_samples_to_lon_lat_psi(self, u, x, y, cdf_x, pdf_y_given_x):
        """
        Draw samples using (random) numbers u.
        Parameters
        ----------
        u - numpy.ndarray, (2, n_samples)
            numbers from the (0,1) interval to be mapped to lat and lon.
        x - numpy.ndarray, (lon_grid_size,)
            grid points on the (0,1) interval, mapped to lon.
        y - numpy.ndarray, (lat_grid_size)
            grid points on the (0,1) interval, mapped to lat.
        cdf_x - numpy.ndarray, (lon_grid_size,)
            correspond to the cumulative distribution of x
        pdf_y_given_x - numpy.ndarray( lon_grid_size, lat_grid_size)
            distribution fucntion of y given x.

        Return
        ------
        lon, lat, psi - numpy.ndarrays, (n_samples, ) each.

        """
        x_inds = np.searchsorted(cdf_x, u[0])
        x_samples = x[x_inds]
        cdf_y_given_x = cumulative_trapezoid(pdf_y_given_x, y, axis=1, initial=0)

        y_inds = np.zeros(len(u[1]), dtype=int)

        for i in np.unique(x_inds):
            cond = i == x_inds
            temp_cdf = cdf_y_given_x[i, :]
            y_inds[cond] = np.searchsorted(temp_cdf, u[1, cond])
        y_samples = y[y_inds]

        lon = np.pi * 2 * x_samples
        lat = np.arcsin(1 - 2 * y_samples)
        psi = u[2] * np.pi * 2
        return lon, lat, psi

    def create_extrinsic_data_for_detector(
        self, det_name, n_samples=512, lon_grid_size=2**13, lat_grid_size=2**13
    ):
        """
        Draw random samples from the detector response pattern.
        Sky position (lon, lat) are drawn form the cubed-detector
        reponse magnitude, (Fp^2 + Fc^2)^(3/2).
        Polaraization angle is drawn uniformly on (0, pi).
        Time is not explicitly drawn, but u_t (uniform on (0,1)) are
        provided. Given a CDF (c) defined on time array (t), samples
        can be drawn using t_samples = t[np.searchsorted(u_t, c)]

        Parameters:
        ----------
        det_name : str,
            'H','L', or 'V'.
        n_samples : int,
            number of samples to draw.
        log_grid_size : int,
            number of grid points on the pdf / cdf of longitude (x).
        lat_grid_size : int,
            number of grid points on the pdf, cdf of latitude (y).

        Returns:
        -------
        lon, lat, psi : numpy.ndarrays of floats,
            samples of longitude, latitude, and polarization angle.
        u_t: numpy.ndarray of floats,
            random samples on the unit intervals, to be mapped to time
            samples given a cumulative distribution.
        detector_response : numpy.ndarray
             detector response of detector `det_name` evaluated on
             `lon`, `lat` and `psi`.

        """

        (x, y, cdf_x, pdf_y_given_x, normalization) = (
            self.get_lon_lat_grid_and_distributions(
                det_name=det_name,
                lon_grid_size=lon_grid_size,
                lat_grid_size=lat_grid_size,
            )
        )

        u = Halton(4).random(n_samples).T
        lon, lat, psi = self.uniform_samples_to_lon_lat_psi(
            u[:3], x, y, cdf_x, pdf_y_given_x
        )
        u_t = u[3]  # given cumulative distribution of t,
        # draw t samples using t_samples = t[np.searchsorted()]

        detector_response = self.compute_detector_responses(
            detector_names=det_name, lat=lat, lon=lon, psi=psi
        )
        return lon, lat, psi, u_t, detector_response


class IntrinsicSampleProcessor:
    """
    A class to load and process intrinsic samples and creates the
    relevant high-dimensional arrays to pass to an Evidence class.
    """

    def __init__(self, like, waveform_dir=None):
        """
        Parameters:

        like: Relative-binning likelihood object, serves as a
        place holder for event_data, waveform_generator, distance
        marginalization lookup-table, and used to
        calculate relative-binning weights.
        """

        self.n_polarizations = 2
        self.likelihood = like
        self.m_arr = like.waveform_generator.m_arr
        self.m_inds, self.mprime_inds = zip(
            *itertools.combinations_with_replacement(range(len(self.m_arr)), 2)
        )
        self.lal_dic = create_lal_dict()
        self.waveform_dir = waveform_dir
        self.cached_dt_linfree_relative = {}

    @property
    def n_intrinsic(self):
        """Number of intrinsic samples."""
        return getattr(getattr(self, "intrinsic_samples", None), "__len__", 0)

    @property
    def n_fbin(self):
        """Number of relative-binning frequency bins."""
        return len(self.likelihood.fbin)

    @property
    def n_modes(self):
        """Number of harminic modes l."""
        return len(self.likelihood.waveform_generator._harmonic_modes_by_m.values())

    def cache_dt_linfree_relative(self, inds, dts):
        """
        Store relative-linear-free timeshifts
        """
        is_not_cached = np.logical_not(
            np.isin(
                inds,
                np.fromiter(self.cached_dt_linfree_relative.keys(), dtype=int),
            )
        )

        for i, dt in zip(inds[is_not_cached], dts[is_not_cached]):
            self.cached_dt_linfree_relative[i] = dt

    @staticmethod
    def load_bank(sample_bank_path, indices=None):
        """
        load intrinsic samples from a file
        """
        sample_bank_path = Path(sample_bank_path)
        bank_config_path = sample_bank_path.with_name("bank_config.json")
        if bank_config_path.exists():
            with open(bank_config_path, "r", encoding="utf-8") as fp:
                bank_config = json.load(fp)
                f_ref = bank_config["f_ref"]
        else:
            f_ref = config.DEFAULT_F_REF

        bank = pd.read_feather(sample_bank_path)
        bank["log_prior_weights"] = (
            bank["log_prior_weights"].values
            - safe_logsumexp(bank["log_prior_weights"].values)
            + np.log(bank.shape[0])
        )
        if indices is None:
            indices = range(bank.shape[0])
        if max(indices) > bank.shape[0]:
            raise ValueError("Indices out of range")

        bank = bank.iloc[indices]

        if "original_index" not in bank.columns:
            bank["original_index"] = bank.index.values

        bank.index = range(len(indices))

        # add columns with default values
        bank["l1"] = config.DEFAULT_PARAMS_DICT["l1"]
        bank["l2"] = config.DEFAULT_PARAMS_DICT["l2"]
        bank["f_ref"] = f_ref

        # samples['log_prior_weights'] is stored as the ratio of
        # physical prior over fiducial prior used in creation of the
        # bank. They are un-normalized. When proparly normalized, their
        # empirical average has mean value 1.

        return bank

    @staticmethod
    def _load_waveforms(waveform_dir, indices):
        """
        load amplitude and phases from directory
        file name format is
        {x}_block_{y}_samples_{low}_{high}.npy
        with x being 'amplitudes', 'phase'.
        Phases are in the linear-free convention.
        Per mode, the linear-free nad standard phases are related by:
        Phase(m,f) [linear free] = (Phase(m,f) [standard]
                                  -2*pi*dt_linear_free*f
                                  - m*dphi_linear_free)
        t (standard) = t (linear free) - dt_linear_free
        phi (standard) = phi (linear free) + dphi_linear_free
        """

        amplitudes, phases = IntrinsicSampleProcessor._load_amp_and_phase(
            waveform_dir, indices
        )
        dt_linfree, dphi_linfree = IntrinsicSampleProcessor._load_linfree_dt_and_dphi(
            waveform_dir, indices
        )

        return amplitudes, phases, dt_linfree, dphi_linfree

    @staticmethod
    def _load_amp_and_phase(waveform_dir, indices):
        """
        Load amplitudes and phases from directory.
        h = amplitudes * np.exp(1j * phases)
        all arrays have shape (n_inds, n_modes, n_pol, n_fbin).
        """
        waveform_dir = Path(waveform_dir)
        bank_config_path = waveform_dir.parent / "bank_config.json"
        with open(bank_config_path, "r", encoding="utf-8") as f:
            bank_config = json.load(f)
            block_size = bank_config["blocksize"]
            n_fbin = len(bank_config["fbin"])
            m_arr = np.asarray(bank_config["m_arr"])
        indices = np.array(indices)

        def get_sorted_files(prefix):
            return sorted(
                waveform_dir.glob(f"{prefix}_block_*.npy"),
                key=lambda file: int(
                    file.stem.removeprefix(prefix + "_block").split("_")[1]
                ),
            )

        amp_files = get_sorted_files("amplitudes")
        phase_files = get_sorted_files("phase")
        n_inds = len(indices)
        n_modes = len(m_arr)  # number of modes
        n_pol = 2  # number of polarizations

        amplitudes = np.zeros((n_inds, n_modes, n_pol, n_fbin))
        phases = np.zeros((n_inds, n_modes, n_pol, n_fbin))

        # Calculate the file index for each requested index
        file_indices = indices // block_size
        indices_in_file = indices % block_size
        unique_file_indices = np.unique(file_indices)

        # Load data from relevant files
        for i in unique_file_indices:
            w = np.where(file_indices == i)[0]
            a = np.load(amp_files[i])
            p = np.load(phase_files[i])
            amplitudes[w] = a[indices_in_file[w]]
            phases[w] = p[indices_in_file[w]]

        return amplitudes, phases

    @staticmethod
    def _load_linfree_dt_and_dphi(waveform_dir, indices):
        """
        load amplitude and phases from directory
        file name format is
        {x}_block_{y}_samples_{low}_{high}.npy
        with x being 'amplitudes', 'phase'.
        Phases are in the linear-free convention.
        Per mode, the linear-free nad standard phases are related by:
        Phase(m,f) [linear free] = (Phase(m,f) [standard]
                                  -2*pi*dt_linear_free*f
                                  - m*dphi_linear_free)
        t (standard) = t (linear free) - dt_linear_free
        phi (standard) = phi (linear free) + dphi_linear_free

        If waveform_dir holds no linear_free files, return zeros.
        """
        # define file names
        waveform_dir = Path(waveform_dir)
        bank_config_path = waveform_dir.parent / "bank_config.json"
        with open(bank_config_path, "r", encoding="utf-8") as f:
            bank_config = json.load(f)
            block_size = bank_config["blocksize"]

        indices = np.array(indices)

        def get_sorted_files(prefix):
            return sorted(
                waveform_dir.glob(f"{prefix}_block_*.npy"),
                key=lambda file: int(
                    file.stem.removeprefix(prefix + "_block").split("_")[1]
                ),
            )

        dt_files = get_sorted_files("dt_linfree")
        dphi_files = get_sorted_files("dphi_linfree")
        # if the bank does not hold linear-free waveform, return empty arrays
        if len(dt_files) == 0:
            dt_linfree = np.zeros(len(indices), dtype=float)
            dphi_linfree = np.zeros(len(indices), dtype=float)
            return dt_linfree, dphi_linfree

        # define dimnesions of arrays
        n_inds = len(indices)
        dt_linfree = np.zeros(n_inds)
        dphi_linfree = np.zeros(n_inds)

        # Calculate the file index for each requested index
        file_indices = indices // block_size
        indices_in_file = indices % block_size
        unique_file_indices = np.unique(file_indices)

        # load waveform & supplementary linfree data
        for i in unique_file_indices:
            w = np.where(file_indices == i)[0]
            dt = np.load(dt_files[i])
            dphi = np.load(dphi_files[i])
            dt_linfree[w] = dt[indices_in_file[w]]
            dphi_linfree[w] = dphi[indices_in_file[w]]

        return dt_linfree, dphi_linfree

    def load_amp_and_phase(self, waveform_dir, indices):
        """
        load relative binning waveform with timeshift correction
        relative to the reference waveform.
        """

        amplitudes, phases = self._load_amp_and_phase(waveform_dir, indices)
        relative_timeshifts = np.zeros(len(indices))

        is_cached = np.isin(
            indices, np.fromiter(self.cached_dt_linfree_relative, dtype=int)
        )
        if np.any(is_cached):
            relative_timeshifts[is_cached] = np.array(
                [self.cached_dt_linfree_relative[i] for i in indices[is_cached]]
            )

            phases[is_cached] += (
                2
                * np.pi
                * relative_timeshifts[is_cached, None, None, None]
                * self.likelihood.fbin[None, None, None, :]
            )

        phases[~is_cached], relative_timeshifts[~is_cached] = (
            self.get_relative_linfree_dt_from_waveform(
                amplitudes[~is_cached], phases[~is_cached]
            )
        )

        self.cache_dt_linfree_relative(indices, relative_timeshifts)

        return amplitudes, phases

    def get_relative_linfree_dt_from_waveform(self, amp_impb, phase_impb):
        """
        Find the relative-binning timeshifts and apply them to the phase.
        """
        m2_index = list(self.likelihood.waveform_generator.m_arr).index(2)
        h2plus_fbin = amp_impb[:, m2_index, 0] * np.exp(
            1j * phase_impb[:, m2_index, 0]
        )  # ib

        hplus_ratio = h2plus_fbin / self.likelihood._h2plus0_fbin  # ib
        dphase = np.unwrap(np.angle(hplus_ratio), axis=-1)  # ib
        weights = self.likelihood._polyfit_weights * np.sqrt(np.abs(hplus_ratio))
        relative_timeshift_i = np.zeros(len(dphase))

        for i, (d, w) in enumerate(zip(dphase, weights)):
            fit = np.polynomial.Polynomial.fit(self.likelihood.fbin, d, deg=1, w=w)

            relative_timeshift_i[i] = -fit.deriv()(0) / (2 * np.pi)
        phase_impb = phase_impb + (
            2
            * np.pi
            * relative_timeshift_i[:, None, None, None]
            * self.likelihood.fbin[None, None, None, :]
        )

        return phase_impb, relative_timeshift_i

    def load_linfree_dt_and_dphi(self, waveform_dir, indices):
        """
        # dt_linfree is the projection of linear term, subtracted form
        # the LAL-convention phase. To correct (from linear free to
        LAL convention), we add it.
        # dt_linfree_relative minus the projection of linear term, added
        # to the LAL-convention phase. To correct, we subtract it.

        """
        if isinstance(indices, int):
            indices = [
                indices,
            ]
        if isinstance(indices, list):
            indices = np.array(indices)

        # load from files
        dt_linfree, dphi_linfree = self._load_linfree_dt_and_dphi(waveform_dir, indices)

        # add the relative timeshifts
        dt_linfree_relative = np.zeros_like(dt_linfree)
        is_not_cached = np.logical_not(
            np.isin(
                indices,
                np.fromiter(self.cached_dt_linfree_relative, dtype=int),
            )
        )

        if is_not_cached.any():
            # by loading the amplitude and phase, we fill the cahced
            # linear free dt dictionary.
            _ = self.load_amp_and_phase(waveform_dir, indices[is_not_cached])

        dt_linfree_relative = np.array(
            [self.cached_dt_linfree_relative[i] for i in indices]
        )

        return dt_linfree - dt_linfree_relative, dphi_linfree

    def load_waveforms(self, waveform_dir, indices):
        """
        Same as parent method, but applies timeshift relative to the
        relative-binning reference waveform as well.
        """
        amplitudes, phases = self.load_amp_and_phase(waveform_dir, indices)

        dt_linfree, dphi_linfree = self.load_linfree_dt_and_dphi(waveform_dir, indices)

        relative_timeshifts = np.array(
            [self.cached_dt_linfree_relative[i] for i in indices]
        )
        return (
            amplitudes,
            phases,
            -relative_timeshifts + dt_linfree,
            dphi_linfree,
        )

    def get_hplus_hcross_0(self, par_dic, f=None, force_fslice=False, fslice=None):
        """
        create (n modes x 2 polarizations x n frequencies) array
        using d_luminosity = 1Mpc and phi_ref = 0
        shifted to center for lk.event_data.times, without t_refdet
        shifts or linear-free time shifts
        """
        if f is None:
            f = self.likelihood.fbin
        if force_fslice and (fslice is None):
            fslice = self.likelihood.event_data.fslice
        elif not force_fslice:
            fslice = slice(0, len(f), 1)

        slow_par_vals = np.array(
            [par_dic[par] for par in self.likelihood.waveform_generator.slow_params]
        )
        # Compute the waveform mode by mode

        # force d_luminosity=1, phi_ref=0
        waveform_par_dic_0 = dict(
            zip(self.likelihood.waveform_generator.slow_params, slow_par_vals),
            d_luminosity=config.DEFAULT_PARAMS_DICT["d_luminosity"],
            phi_ref=config.DEFAULT_PARAMS_DICT["phi_ref"],
        )

        # hplus_hcross_0 is a (n_m x 2 x n_frequencies) array with
        # sum_l (hlm+, hlmx), at phi_ref=0, d_luminosity=1Mpc.

        n_freq = len(f)
        shape = (self.n_modes, self.n_polarizations, n_freq)
        hplus_hcross_0_mpf = np.zeros(shape, np.complex128)
        hplus_hcross_0_mpf[..., fslice] = compute_hplus_hcross_safe(
            f[fslice],
            waveform_par_dic_0,
            self.likelihood.waveform_generator.approximant,
            self.likelihood.waveform_generator.harmonic_modes,
            self.likelihood.waveform_generator._harmonic_modes_by_m,
            self.lal_dic,
        )
        # tcoarse_time_shift
        shift = get_shift(f, self.likelihood.event_data.tcoarse)
        return hplus_hcross_0_mpf * shift

    def get_summary(self):
        """
        Get the relative-binning summary statistics for the likelihood
        calculations. The weights are calculated per detector (d),
        harmonic mode (m), polarization (p) and frequency bin (b).
        For <h,h>, the weights are cacluated per mode pair (m and
        m') and pair polarization pair (p and p'). For the modes we use
        a single index (m) to represent all unique combinations of
        modes, and for the polarizations we calcualted the 4 p-p
        combinations.

        Output:
        dh_weights_dmpb: array of weights for the integrand
        data * h.conj(), for each detector, mode, polarization, and
        frequency bin.
        hh_weights_dmppb: array of weights for the integrand
        h*h.conj(), for each detector, mode pairs, polarization pair,
        and

        """
        # impose zero orbital phase and distance 1Mpc
        par_dic = self.likelihood.par_dic_0 | getattr(self.likelihood, "_ref_dic", {})

        h0_mpf = self.get_hplus_hcross_0(
            par_dic,
            f=self.likelihood.event_data.frequencies,
            force_fslice=True,
        )
        h0_mpb = self.get_hplus_hcross_0(par_dic, f=self.likelihood.fbin)

        # Temporarily undo big time shift so waveform is smooth at
        # high frequencies:
        shift_f = np.exp(
            -2j
            * np.pi
            * self.likelihood.event_data.frequencies
            * self.likelihood.event_data.tcoarse
        )
        shift_fbin = np.exp(
            -2j * np.pi * self.likelihood.fbin * self.likelihood.event_data.tcoarse
        )
        h0_mpf *= shift_f.conj()
        h0_mpb *= shift_fbin.conj()
        # Ensure reference waveform is smooth and !=0 at high frequency:
        self.likelihood._stall_ringdown(h0_mpf, h0_mpb)
        # Reapply big time shift:
        h0_mpf *= shift_f
        h0_mpb *= shift_fbin

        d_h0_dmpf = (
            self.likelihood.event_data.blued_strain[:, np.newaxis, np.newaxis, :]
            * h0_mpf.conj()
        )

        dh_weights_dmpb = (
            self.likelihood._get_summary_weights(d_h0_dmpf) / h0_mpb.conj()
        )

        whitened_h0_dmpf = (
            h0_mpf[np.newaxis, ...]
            * self.likelihood.event_data.wht_filter[:, np.newaxis, np.newaxis, :]
        )
        h0_h0_dmppf = np.einsum(
            "dmpf, dmPf-> dmpPf",
            whitened_h0_dmpf[:, self.m_inds, ...],
            whitened_h0_dmpf[:, self.mprime_inds, ...].conj(),
            optimize=True,
        )

        hh_weights_dmppb = self.likelihood._get_summary_weights(h0_h0_dmppf)

        hh_weights_dmppb = np.einsum(
            "mpb, mPb, dmpPb -> dmpPb",
            h0_mpb[self.m_inds, ...] ** (-1),
            h0_mpb[self.mprime_inds, ...].conj() ** (-1),
            hh_weights_dmppb,
            optimize=True,
        )
        # Count off-diagonoal terms twice:
        hh_weights_dmppb[:, ~np.equal(self.m_inds, self.mprime_inds)] *= 2

        return dh_weights_dmpb, hh_weights_dmppb
