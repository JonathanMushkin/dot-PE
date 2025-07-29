"""
Likelihood calculation engine for the free-sampling method.

This module contains the core computational classes and utilities for
performing likelihood evaluations through efficient matrix multiplications
and tensor operations.
"""

import itertools

import lalsimulation as lalsim
import numpy as np
import pandas as pd
import torch
from lal import CreateDict
from scipy.special import logsumexp

from .device_manager import get_device_manager

from cogwheel import likelihood
from cogwheel.likelihood.relative_binning import BaseLinearFree
from cogwheel.waveform import FORCE_NNLO_ANGLES, compute_hplus_hcross_by_mode
from cogwheel.waveform_models.xode import compute_hplus_hcross_by_mode_xode

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


class LikelihoodCalculator:
    """
    A class that receives as input the components of intrinsic and
    extrinsic factorizations and calculates the likelihood at each
    (int., ext., phi) combination (distance marginalized).

    This is the atomic unit for evaluating likelihoods through efficient
    matrix multiplications and tensor operations.

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
        Initialize the LikelihoodCalculator object.
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

        # Convert inputs to tensors on the correct device
        device_manager = get_device_manager()
        dh_weights_dmpb = device_manager.to_tensor(dh_weights_dmpb)
        h_impb = device_manager.to_tensor(h_impb)
        response_dpe = device_manager.to_tensor(response_dpe)
        timeshift_dbe = device_manager.to_tensor(timeshift_dbe)
        asd_drift_d = device_manager.to_tensor(asd_drift_d)

        i, m, p, b = h_impb.shape
        d, *_, e = response_dpe.shape
        x = d * p * b  # size of summed dimensions

        # reshape & conjugate arrays
        dh_weights_mx = torch.moveaxis(dh_weights_dmpb, 0, 1).reshape(m, x)

        h_conj_mix = torch.repeat_interleave(
            torch.moveaxis(h_impb.conj(), 0, 1)[:, :, None, ...], d, dim=2
        ).reshape(m, i, x)

        response_drift_dpe = response_dpe * asd_drift_d[:, None, None] ** -2
        response_drift_xe = torch.repeat_interleave(
            response_drift_dpe[:, :, None, :], b, dim=2
        ).reshape(x, e)

        timeshift_xe = torch.reshape(
            timeshift_dbe.conj()[:, None, :, :].repeat(1, p, 1, 1), (x, e)
        )
        ext_tensor = timeshift_xe * response_drift_xe
        dh_mie = torch.empty((m, i, e), dtype=h_impb.dtype, device=h_impb.device)

        for _m in range(m):
            dh_mie[_m] = torch.matmul(dh_weights_mx[_m] * h_conj_mix[_m], ext_tensor)

        result = torch.moveaxis(dh_mie, (0, 1), (2, 0))

        return result

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
        # Convert inputs to tensors on the correct device
        device_manager = get_device_manager()
        h_impb = device_manager.to_tensor(h_impb)
        response_dpe = device_manager.to_tensor(response_dpe)
        hh_weights_dmppb = device_manager.to_tensor(hh_weights_dmppb)
        asd_drift_d = device_manager.to_tensor(asd_drift_d)
        m_inds = device_manager.to_tensor(m_inds)
        mprime_inds = device_manager.to_tensor(mprime_inds)

        hh_idmpP = torch.einsum(
            "dmpPb, impb, imPb -> idmpP",
            hh_weights_dmppb,
            h_impb[:, m_inds, ...],
            h_impb.conj()[:, mprime_inds, ...],
        )  # idmpp
        ff_dppe = torch.einsum(
            "dpe, dPe, d -> dpPe",
            response_dpe,
            response_dpe,
            asd_drift_d**-2,
        ).to(dtype=hh_idmpP.dtype)  # dppe
        hh_iem = torch.einsum("idmpP, dpPe -> iem", hh_idmpP, ff_dppe)  # iem

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
        # Convert inputs to tensors on the correct device
        device_manager = get_device_manager()
        dh_iem = device_manager.to_tensor(dh_iem)
        hh_iem = device_manager.to_tensor(hh_iem)

        # Convert m_arr to tensor if it isn't already
        if not hasattr(self, "_m_arr_tensor"):
            self._m_arr_tensor = device_manager.to_tensor(self.m_arr)
            self._m_inds_tensor = device_manager.to_tensor(self.m_inds)
            self._mprime_inds_tensor = device_manager.to_tensor(self.mprime_inds)

        phi_grid = torch.linspace(
            0, 2 * torch.pi, self.n_phi + 1, device=dh_iem.device, dtype=dh_iem.dtype
        )[:-1]  # o
        # dh_phasor is the phase shift applied to h[m].conj, hence the
        # minus sign
        dh_phasor = torch.exp(-1j * torch.outer(self._m_arr_tensor, phi_grid))  # mo
        # hh_phasor is applied both to h[m] and h[mprime].conj, hence
        # the subtraction of the two modes
        hh_phasor = torch.exp(
            1j
            * torch.outer(
                self._m_arr_tensor[self._m_inds_tensor]
                - self._m_arr_tensor[self._mprime_inds_tensor],
                phi_grid,
            )
        )  # mo
        dh_ieo = torch.einsum("iem, mo -> ieo", dh_iem, dh_phasor).real  # ieo
        hh_ieo = torch.einsum("iem, mo -> ieo", hh_iem, hh_phasor).real  # ieo

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
