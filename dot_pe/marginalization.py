"""
Holds the modified likelihood.marginalization objects
for sampler-free integration.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.special import logsumexp

from cogwheel.likelihood import MarginalizedExtrinsicLikelihood
from cogwheel.likelihood.marginalization import CoherentScoreHM
from cogwheel.likelihood.marginalization.base import MarginalizationInfoHM
from cogwheel.likelihood.marginalization.coherent_score_hm import _flip_psi
from cogwheel.utils import exp_normalize, n_effective


class CoherentScoreSamplerFree(CoherentScoreHM):
    """
    CoherentScoreHM with modifications for the sampler-free methods.
    """

    def __init__(self, *args, **kwargs):
        self.min_n_effective_prior = kwargs.pop("min_n_effective_prior", 0)
        super().__init__(*args, **kwargs)

    def get_marginalization_info(
        self, d_h_timeseries, h_h, times, lnl_marginalized_threshold=-np.inf
    ):
        """
        Return a MarginalizationInfo object with extrinsic parameter
        integration results, ensuring that one of three conditions
        regarding the effective sample size holds:
            * n_effective >= .min_n_effective; or
            * n_qmc == 2 ** .max_log2n_qmc; or
            * lnl_marginalized < lnl_marginalized_threshold
        """
        self._switch_qmc_sequence()

        # Resample to match sky_dict's dt:
        d_h_timeseries, times = self.sky_dict.resample_timeseries(
            d_h_timeseries, times, axis=-2
        )

        t_arrival_lnprob = self._incoherent_t_arrival_lnprob(d_h_timeseries, h_h)  # dt
        self.sky_dict.apply_tdet_prior(t_arrival_lnprob)
        t_arrival_prob = exp_normalize(t_arrival_lnprob, axis=1)

        i_chunk = 0
        marginalization_info = self._get_marginalization_info_chunk(
            d_h_timeseries, h_h, times, t_arrival_prob, i_chunk
        )

        while (marginalization_info.n_effective < self.min_n_effective) or (
            marginalization_info.n_effective_prior < self.min_n_effective_prior
        ):
            # Perform adaptive mixture importance sampling:
            i_chunk += 1
            if i_chunk == len(self._qmc_ind_chunks):
                logging.warning("Maximum QMC resolution reached.")
                break

            if marginalization_info.n_effective == 0:  # Unphysical point
                break

            if marginalization_info.n_effective > 2 and (
                marginalization_info.lnl_marginalized < lnl_marginalized_threshold
            ):  # Worthless point
                break

            # Hybridize with KDE of the weighted samples as next proposal:
            t_arrival_prob = 0.5 * (
                self._kde_t_arrival_prob(marginalization_info, times) + t_arrival_prob
            )

            marginalization_info.update(
                self._get_marginalization_info_chunk(
                    d_h_timeseries, h_h, times, t_arrival_prob, i_chunk
                )
            )

        return marginalization_info

    def _get_marginalization_info_chunk(
        self, d_h_timeseries, h_h, times, t_arrival_prob, i_chunk
    ):
        """
        Evaluate inner products (d|h) and (h|h) at integration points
        over a chunk of a QMmarC sequence of extrinsic parameters, given
        timeseries of (d|h) and value of (h|h) by mode `m`, polarization
        `p` and detector `d`.

        Parameters
        ----------
        d_h_timeseries: (n_m, 2, n_t, n_d) complex array
            Timeseries of complex (d|h), inner product of data against a
            waveform at reference distance and phase.
            Decomposed by mode, polarization, time, detector.

        h_h: (n_mm, 2, 2, n_d) complex array
            Complex (h|h) inner product of a waveform with itself,
            decomposed by mode, polarization and detector.

        times: (n_t,) float array
            Timestamps of the timeseries (s).

        t_arrival_prob: (n_d, n_t) float array
            Proposal probability of time of arrival at each detector,
            normalized to sum to 1 along the time axis.

        i_chunk: int
            Index to ``._qmc_ind_chunks``.

        Return
        ------
        Instance of ``MarginalizationInfoHM`` with several fields, see
        its documentation.
        """
        if d_h_timeseries.shape[0] != self.m_arr.size:
            raise ValueError("Incorrect number of harmonic modes.")

        q_inds = self._qmc_ind_chunks[i_chunk]  # Will update along the way
        n_qmc = len(q_inds)
        tdet_inds = self._get_tdet_inds(t_arrival_prob, q_inds)

        sky_inds, sky_prior, physical_mask = self.sky_dict.get_sky_inds_and_prior(
            tdet_inds[1:] - tdet_inds[0]
        )  # q, q, q

        # Apply physical mask (sensible time delays):
        q_inds = q_inds[physical_mask]
        tdet_inds = tdet_inds[:, physical_mask]

        if not any(physical_mask):
            return MarginalizationInfoSamplerFree(
                qmc_sequence_id=self._current_qmc_sequence_id,
                ln_numerators=np.array([]),
                ln_numerators_prior=np.array([]),
                q_inds=np.array([], int),
                o_inds=np.array([], int),
                sky_inds=np.array([], int),
                t_first_det=np.array([]),
                d_h=np.array([]),
                h_h=np.array([]),
                tdet_inds=tdet_inds,
                proposals_n_qmc=[n_qmc],
                proposals_weights=[1],
                proposals=[t_arrival_prob],
                flip_psi=np.array([], bool),
            )

        t_first_det = times[tdet_inds[0]] + self._qmc_sequence["t_fine"][q_inds]

        dh_qo, hh_qo = self._get_dh_hh_qo(
            sky_inds, q_inds, t_first_det, times, d_h_timeseries, h_h
        )  # qo, qo

        ln_numerators_prior, ln_numerators_posterior, important, flip_psi = (
            self._get_lnnumerators_important_flippsi(dh_qo, hh_qo, sky_prior)
        )

        # Keep important samples (lnl above threshold):
        q_inds = q_inds[important[0]]
        sky_inds = sky_inds[important[0]]
        t_first_det = t_first_det[important[0]]
        tdet_inds = tdet_inds[:, important[0]]

        return MarginalizationInfoSamplerFree(
            qmc_sequence_id=self._current_qmc_sequence_id,
            ln_numerators_prior=ln_numerators_prior,
            ln_numerators=ln_numerators_posterior,
            q_inds=q_inds,
            o_inds=important[1],
            sky_inds=sky_inds,
            t_first_det=t_first_det,
            d_h=dh_qo[important],
            h_h=hh_qo[important],
            tdet_inds=tdet_inds,
            proposals_n_qmc=[n_qmc],
            proposals_weights=[1],
            proposals=[t_arrival_prob],
            flip_psi=flip_psi,
        )

    def _get_lnnumerators_important_flippsi(self, dh_qo, hh_qo, sky_prior):
        """
        Parameters
        ----------
        dh_qo: (n_physical, n_phi) float array
            ⟨d|h⟩ real inner product between data and waveform at
            ``self.lookup_table.REFERENCE_DISTANCE``.

        hh_qo: (n_physical, n_phi) float array
            ⟨h|h⟩ real inner product of a waveform at
            ``self.lookup_table.REFERENCE_DISTANCE`` with itself.

        sky_prior: (n_physical,) float array
            Prior weights of the QMC sequence.

        Return
        ------
        ln_numerators: float array of length n_important
            Natural log of the weights of the QMC samples, including the
            likelihood and prior but excluding the importance sampling
            weights.

        important: (array of ints, array of ints) of lengths n_important
            The first array contains indices between 0 and n_physical-1
            corresponding to (physical) QMC samples.
            The second array contains indices between 0 and n_phi-1
            corresponding to orbital phases.
            They correspond to samples with sufficiently high maximum
            likelihood over distance to be included in the integral.

        """
        flip_psi = np.signbit(dh_qo)  # qo
        max_over_distance_lnl = 0.5 * dh_qo**2 / hh_qo  # qo
        threshold = np.max(max_over_distance_lnl) - self.DLNL_THRESHOLD
        important = np.where(max_over_distance_lnl > threshold)

        ln_numerators_prior = +np.log(sky_prior)[important[0]] - np.log(self._nphi)  # i

        ln_numerators_posterior = (
            ln_numerators_prior
            + self.lookup_table.lnlike_marginalized(dh_qo[important], hh_qo[important])
        )  # i

        return (
            ln_numerators_prior,
            ln_numerators_posterior,
            important,
            flip_psi[important],
        )

    def gen_samples_from_marg_info(self, marg_info, num=None):
        """
        Generate requested number of extrinsic parameter samples.

        Parameters
        ----------
        marg_info: MarginalizationInfoHM or None
            Normally, output of ``.get_marginalization_info``.
            If ``None``, assume that the sampled parameters were unphysical
            and return samples full of nans.

        num: int, optional
            Number of samples to generate, ``None`` makes a single sample.

        Return
        ------
        samples: dict
            Values are scalar if `num` is ``None``, else numpy arrays.
            If ``marg_info`` correspond to an unphysical sample (i.e.,
            a realization of matched-filtering timeseries in the
            detectors incompatible with a real signal) the values will
            be NaN.
            '_psi' and '_phi_ref' are returned only for diagnostic
            purposes. They are not used as samples
        """
        if marg_info is None or marg_info.q_inds.size == 0:
            # Order and dtype must match that of regular output
            out = dict.fromkeys(
                ["dec", "lon", "_phi_ref", "psi", "t_geocenter", "weights"],
                np.full(num, np.nan)[()],
            )
            return out

        self._switch_qmc_sequence(marg_info.qmc_sequence_id)
        random_ids = self._rng.choice(len(marg_info.q_inds), size=num)

        q_ids = marg_info.q_inds[random_ids]
        o_ids = marg_info.o_inds[random_ids]
        sky_ids = marg_info.sky_inds[random_ids]
        t_geocenter = (
            marg_info.t_first_det[random_ids]
            - self.sky_dict.geocenter_delay_first_det[sky_ids]
        )

        psi = _flip_psi(
            self._qmc_sequence["psi"][q_ids],
            marg_info.d_h[random_ids],
            marg_info.flip_psi[random_ids],
        )[0]

        return {
            "dec": self.sky_dict.sky_samples["lat"][sky_ids],
            "lon": self.sky_dict.sky_samples["lon"][sky_ids],
            "_phi_ref": self._phi_ref[o_ids],
            "psi": psi,
            "t_geocenter": t_geocenter,
            "weights": marg_info.prior_weights[random_ids],
        }


@dataclass
class MarginalizationInfoSamplerFree(MarginalizationInfoHM):
    """
    This dataclass separate the prior-weights and posterior weights.
    The weights are computed using single intrinsic set of
    parameters. They used to determine n_effective and the marginalized
    likelihood, and are principally a diagnostic tool, in the context of
    sampler free sampling and integration.
    The prior weights are used for drawing samples.
    """

    ln_numerators_prior: np.ndarray
    prior_weights: np.ndarray = field(init=False)
    prior_weights_q: np.ndarray = field(init=False)
    n_effective_prior: float = field(init=False)
    proposals_weights: list

    def __post_init__(self):
        """Set derived attributes."""
        self.n_qmc = sum(self.proposals_n_qmc)

        if self.q_inds.size == 0:
            self.weights_q = np.array([])
            self.weights = np.array([])
            self.n_effective = 0.0
            self.lnl_marginalized = -np.inf
            self.n_effective_prior = 0.0
            self.prior_weights_q = np.array([])
            self.prior_weights = np.array([])
            return

        denominators = np.zeros(len(self.q_inds))
        total_n_qmc = sum(self.proposals_n_qmc)
        for n_qmc, proposal, w in zip(
            self.proposals_n_qmc, self.proposals, self.proposals_weights
        ):
            denominators += (
                w
                * (n_qmc / total_n_qmc)
                * np.prod(
                    np.take_along_axis(proposal, self.tdet_inds, axis=1),
                    axis=0,
                )
            )  # q

        ln_weights = self.ln_numerators - np.log(denominators)
        self.weights = exp_normalize(ln_weights)

        weights_q = sparse.coo_array(
            (self.weights, (np.zeros_like(self.q_inds), self.q_inds))
        ).toarray()[0]  # Repeated q_inds get summed

        self.weights_q = weights_q[weights_q > 0]

        self.n_effective = n_effective(self.weights_q)
        self.lnl_marginalized = logsumexp(ln_weights) - np.log(total_n_qmc)

        ln_prior_weights = self.ln_numerators_prior - np.log(denominators)

        self.prior_weights = np.exp(ln_prior_weights)

        prior_weights_q = sparse.coo_array(
            (self.prior_weights, (np.zeros_like(self.q_inds), self.q_inds))
        ).toarray()[0]

        self.prior_weights_q = prior_weights_q[prior_weights_q > 0]
        self.n_effective_prior = n_effective(self.prior_weights_q)

    def update(self, other):
        """
        Update entries of this instance of MarginalizationInfo to
        include information from another instance. The intended use is
        to extend the QMC sequence if it has too low ``.n_effective``.

        Parameters
        ----------
        other: MarginalizationInfoSamplerFree
            Typically ``self`` will be the first half of the extended
            QMC sequence and ``other`` would be the second half.
        """
        if self.qmc_sequence_id != other.qmc_sequence_id:
            raise ValueError("Cannot use different QMC sequences.")

        self.proposals_n_qmc += other.proposals_n_qmc
        self.proposals += other.proposals
        self.proposals_weights += other.proposals_weights
        for attr in (
            "ln_numerators",
            "ln_numerators_prior",
            "q_inds",
            "o_inds",
            "flip_psi",
            "sky_inds",
            "t_first_det",
            "d_h",
            "h_h",
            "tdet_inds",
        ):
            updated = np.concatenate(
                [getattr(self, attr), getattr(other, attr)], axis=-1
            )
            setattr(self, attr, updated)

        self.__post_init__()  # Update derived attributes

    def update_with_list(self, other_list):
        """
        Update entries of this instance of MarginalizationInfo to
        include information from a list of instances. This is meant to
        reduce the compuational time of calling self.update() many time,
        due to repeated self.__post_init__() calls.

        Parameters
        ----------
        other_list: list of MarginalizationInfo
        """

        for other in other_list:
            if self.qmc_sequence_id != other.qmc_sequence_id:
                raise ValueError("Cannot use different QMC sequences.")

        # treat lists
        for other in other_list:
            self.proposals_n_qmc += other.proposals_n_qmc
            self.proposals += other.proposals
            self.proposals_weights += other.proposals_weights

        # treat numpy arrays
        for attr in (
            "ln_numerators",
            "ln_numerators_prior",
            "q_inds",
            "o_inds",
            "flip_psi",
            "sky_inds",
            "t_first_det",
            "d_h",
            "h_h",
            "tdet_inds",
        ):
            from_list = np.concatenate(
                [getattr(other, attr) for other in other_list], axis=-1
            )

            updated = np.concatenate([getattr(self, attr), from_list], axis=-1)

            setattr(self, attr, updated)

        self.__post_init__()  # Update derived attributes


class MarginalizationExtrinsicSamplerFreeLikelihood(MarginalizedExtrinsicLikelihood):
    """
    MarginalizedExtrinsicLikelihood for sampler-free methods.
    """

    def _create_coherent_score(self, sky_dict, m_arr, **kwargs):
        return CoherentScoreSamplerFree(sky_dict, m_arr=m_arr, **kwargs)

    def _get_many_dh_hh(
        self,
        amp_impb,
        phase_impb,
        d_h_weights,
        h_h_weights,
        m_inds,
        mprime_inds,
        asd_drift,
    ):
        """
        A modified version of
        `MarginalizedExtrinsicLikelihood.get_many_dh_hh`
        that loads the waveforms from a bank, already in the
        linear-free convention.

        Parameters
        ----------
        amp_impb: array of floats
            amplitude of the waveform at the reference extrinsic
            parameters, decomposed by intrinsic samples, mode,
            polarization, and fbin.
        phase_impb: array of floats,
            phase of the waveform at the reference extrinsic parameters,
            decomposed by intrinsic samples, mode, polarization, and
            fbin.
        d_h_weights: (n_m, n_t, n_d, n_b) complex array
            Complex weights of the waveform decomposition.
            Decomposed by mode, time, detector, bank.
        h_h_weights: (n_mm, n_d, n_d) complex array
            Complex weights of the waveform decomposition.
            Decomposed by mode, detector, detector.
        asd_drift: (n_d) float array
            ASD drift correction.

        return
        ------
        dh_nmptd: (n_n, n_m, n_p, n_t, n_d) complex array
            Complex (d|h) inner product of a waveform with the data,
            decomposed by mode, polarization, time and detector.
        hh_nmppd: (n_n, n_m, n_m, n_p, n_p, n_d) complex array
            Complex (h|h) inner product of a waveform with itself,
            decomposed by mode, polarization and detector.

        """

        h_impb = amp_impb * np.exp(1j * phase_impb)
        h_mpbi = np.moveaxis(h_impb, 0, -1).astype(np.complex64)
        n_m, n_t, n_d, n_b = d_h_weights.shape
        n_i = amp_impb.shape[0]
        n_p = 2
        d_h_weights = d_h_weights.reshape((n_m, n_t * n_d, n_b))  # m(td)b

        # Loop instead of broadcasting, to save memory:
        dh_mptdi = np.zeros((n_m, n_p, n_t * n_d, n_i), np.complex64)
        for i_m, i_p in np.ndindex(n_m, n_p):
            dh_mptdi[i_m, i_p] = d_h_weights[i_m] @ h_mpbi[i_m, i_p].conj()

        dh_imptd = np.moveaxis(dh_mptdi, -1, 0).reshape(n_i, n_m, n_p, n_t, n_d)

        m_inds = list(m_inds)
        mprime_inds = list(mprime_inds)
        hh_imppd = np.einsum(
            "mdb,mpbi,mPbi->impPd",
            h_h_weights,
            h_mpbi[m_inds],
            h_mpbi.conj()[mprime_inds],
        )
        asd_drift_correction = asd_drift.astype(np.float32) ** -2  # d
        return dh_imptd * asd_drift_correction, hh_imppd * asd_drift_correction

    def _incoherent_t_arrival_lnprob(self, d_h_timeseries, h_h):
        """
        Log likelihood maximized over distance and phase, approximating
        that different modes and polarizations are all orthogonal and
        have independent phases.

        Parameters
        ----------
        self: MarginalizedExtrinsicLikelihood object

        d_h_timeseries: (n_i, n_m, 2, n_t, n_d) complex array
            Timeseries of complex (d|h), inner product of data against a
            waveform at reference distance and phase.
            Decomposed by mode, polarization, time, detector.

        h_h: (n_i, n_mm, 2, 2, n_d) complex array
            Complex (h|h) inner product of a waveform with itself,
            decomposed by mode, polarization and detector.
        """

        hh_mpdiagonal = h_h[:, np.equal(self.m_inds, self.mprime_inds)][
            :, (0, 1), (0, 1)
        ].real  # mpd
        chi_squared = np.einsum("imptd->tid", np.abs(d_h_timeseries)) ** 2 / np.einsum(
            "impd->id", hh_mpdiagonal
        )  # tid

        return np.moveaxis(
            self.beta_temperature / 2 * chi_squared, (0, 1, 2), (2, 0, 1)
        )  # idt
