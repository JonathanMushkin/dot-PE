"""
Sampler free sampling implementation.
"""

import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from cogwheel.utils import JSONMixin, DIR_PERMISSIONS, FILE_PERMISSIONS, exp_normalize

from . import config, likelihood_calculating, sample_processing
from .base_sampler_free_sampling import (
    get_top_n_indices_two_pointer,
    Loggable,
)
from .utils import (
    safe_logsumexp,
)


class CoherentLikelihoodProcessor(JSONMixin, Loggable):
    """
    A class that orchestrates the main multi-detector coherent likelihood
    evaluation workflow. Takes intrinsic and extrinsic samples, performs
    block likelihood evaluations, and combines them into `prob_samples`
    (which contain only indices and probabilistic information).

    This is the primary workflow coordinator for coherent analysis.
    """

    DEFAULT_SIZE_LIMIT = 10**6

    PROB_SAMPLES_COLS = [
        "i",
        "e",
        "o",
        "lnl_marginalized",
        "ln_posterior",
        "bestfit_lnlike",
        "d_h_1Mpc",
        "h_h_1Mpc",
        "weights",
    ]

    PROB_SAMPLES_COLS_DTYPES = [int, int, int, float, float, float, float, float, float]

    def __init__(
        self,
        intrinsic_bank_file,
        waveform_dir,
        n_phi,
        m_arr,
        likelihood,
        seed=None,
        max_bestfit_lnlike_diff=20,
        size_limit=None,
        int_block_size=512,
        ext_block_size=512,
        min_bestfit_lnlike_to_keep=None,
        dir_permissions=DIR_PERMISSIONS,
        file_permissions=FILE_PERMISSIONS,
        full_intrinsic_indices=None,
        renormalize_log_prior_weights_i: bool = False,
        intrinsic_logw_lookup=None,
        n_samples_discarded: int = 0,
        logsumexp_discarded_ln_posterior: float = -np.inf,
        logsumsqrexp_discarded_ln_posterior: float = -np.inf,
        n_samples_accepted: int = 0,
        logsumexp_accepted_ln_posterior: float = -np.inf,
        logsumsqrexp_accepted_ln_posterior: float = -np.inf,
        n_distance_marginalizations: int = 0,
    ):
        """
        Initialization of the SamplerFreeSampler.

        Parameters
        ----------
        intrinsic_bank_file : str or Path
            The file containing the intrinsic sample bank.
        waveform_dir : str or Path
            The directory containing the waveform data.
        n_phi : int
            The number of orbital phase samples.
        m_arr : np.ndarray
            The array of mass samples.
        likelihood : MarginalizationExtrinsicSamplerFreeLikelihood
            The likelihood object.
        seed : int, optional
            The seed for the random number generator.
        max_bestfit_lnlike_diff : float, optional
            The maximum difference between the bestfit likelihood of
            a sample and the maximum bestfit likelihood to keep the
            sample.
        size_limit : int, optional
            The size limit of the light samples dataframe.
        int_block_size : int, optional
            The size of the intrinsic block.
        ext_block_size : int, optional
            The size of the extrinsic block.
        min_bestfit_lnlike_to_keep : float, optional
            The minimum bestfit likelihood to keep a sample.
        dir_permissions : int, optional
            The permissions for the directories.
        file_permissions : int, optional
            The permissions for the files.
        full_bank_indices : np.ndarray, optional
            The indices of the full intrinsic bank. If None, all samples
            are used.
        """

        self.dir_permissions = dir_permissions
        self.file_permissions = file_permissions
        self.intrinsic_bank_file = Path(intrinsic_bank_file)
        self.waveform_dir = Path(waveform_dir)
        self.likelihood = likelihood
        self.intrinsic_sample_processor = sample_processing.IntrinsicSampleProcessor(
            self.likelihood, self.waveform_dir
        )

        self.extrinsic_sample_processor = sample_processing.ExtrinsicSampleProcessor(
            self.likelihood.event_data.detector_names
        )

        self.likelihood_calculator = likelihood_calculating.LikelihoodCalculator(
            n_phi=n_phi, m_arr=np.array(m_arr)
        )

        self.dh_weights_dmpb, self.hh_weights_dmppb = (
            self.intrinsic_sample_processor.get_summary()
        )

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.cur_rundir = None  # current rundir

        # this is the raw intrinsic sample bank, used to draw samples
        self.intrinsic_sample_bank = self.intrinsic_sample_processor.load_bank(
            intrinsic_bank_file,
            full_intrinsic_indices,
            renormalize_log_prior_weights=renormalize_log_prior_weights_i,
        )

        if full_intrinsic_indices is None:
            self.full_intrinsic_indices = np.arange(len(self.intrinsic_sample_bank))
        else:
            self.full_intrinsic_indices = full_intrinsic_indices

        self.int_block_size = int_block_size
        self.ext_block_size = ext_block_size

        self.size_limit = size_limit if size_limit else self.DEFAULT_SIZE_LIMIT
        self.full_log_prior_weights_i = self.intrinsic_sample_bank[
            "log_prior_weights"
        ].values
        if renormalize_log_prior_weights_i:
            self.full_log_prior_weights_i = (
                self.full_log_prior_weights_i
                - safe_logsumexp(self.full_log_prior_weights_i)
                + np.log(len(self.full_log_prior_weights_i))
            )  # mean weight = 1

        self._intrinsic_logw_sorted_inds = None
        self._intrinsic_logw_sorted_vals = None
        if intrinsic_logw_lookup is not None:
            inds, logw = intrinsic_logw_lookup
            inds = np.asarray(inds, dtype=int)
            logw = np.asarray(logw, dtype=float)
            if inds.shape != logw.shape:
                raise ValueError(
                    "intrinsic_logw_lookup inds and logw must have same shape"
                )
            order = np.argsort(inds)
            self._intrinsic_logw_sorted_inds = inds[order]
            self._intrinsic_logw_sorted_vals = logw[order]

        self.full_log_prior_weights_e = None
        self.full_response_dpe = None
        self.full_timeshifts_dbe = None

        self.max_bestfit_lnlike_diff = max_bestfit_lnlike_diff
        # minimal bestfit lnlike to keep is updated throughout the run
        if min_bestfit_lnlike_to_keep is None:
            self.min_bestfit_lnlike_to_keep = -np.inf
        else:
            self.min_bestfit_lnlike_to_keep = min_bestfit_lnlike_to_keep

        # record diagnostic quantities for the accepted and discarded
        # samples: number of samples, sum and sum-of-squares of
        # (unnormalized) posterior.
        # From these, the error in evidence and effective number of
        # samples can be calculated.
        self.n_samples_discarded = (
            0 if n_samples_discarded is None else n_samples_discarded
        )
        self.logsumexp_discarded_ln_posterior = (
            logsumexp_discarded_ln_posterior
            if logsumexp_discarded_ln_posterior is not None
            else -np.inf
        )  # unnormalized
        self.logsumsqrexp_discarded_ln_posterior = (
            logsumsqrexp_discarded_ln_posterior
            if logsumsqrexp_discarded_ln_posterior is not None
            else -np.inf
        )  # unnormalized

        self.n_samples_accepted = (
            n_samples_accepted if n_samples_accepted is not None else 0
        )
        self.logsumexp_accepted_ln_posterior = (
            logsumexp_accepted_ln_posterior
            if logsumexp_accepted_ln_posterior is not None
            else -np.inf
        )  # unnormalized
        self.logsumsqrexp_accepted_ln_posterior = (
            logsumsqrexp_accepted_ln_posterior
            if logsumsqrexp_accepted_ln_posterior is not None
            else -np.inf
        )  # unnormalized

        # counter of number of distance marginalization call performed
        self.n_distance_marginalizations = n_distance_marginalizations

        self.block_list = []
        self.initialize_prob_samples()
        self._next_block = None
        self.setup_logger()

    def _get_intrinsic_logw(self, abs_i):
        if self._intrinsic_logw_sorted_inds is None:
            return self.full_log_prior_weights_i[abs_i]
        abs_i = np.asarray(abs_i, dtype=int)
        pos = np.searchsorted(self._intrinsic_logw_sorted_inds, abs_i)
        if np.any(pos < 0) or np.any(pos >= self._intrinsic_logw_sorted_inds.size):
            raise ValueError("Intrinsic index out of range of intrinsic_logw_lookup")
        if np.any(self._intrinsic_logw_sorted_inds[pos] != abs_i):
            raise ValueError("Intrinsic index missing from intrinsic_logw_lookup")
        return self._intrinsic_logw_sorted_vals[pos]

    def get_init_dict(self):
        """Return init dict.
        Used for to_json and load_json methods"""
        init_dict = dict(
            intrinsic_bank_file=self.intrinsic_bank_file.as_posix(),
            waveform_dir=self.waveform_dir.as_posix(),
            n_phi=self.likelihood_calculator.n_phi,
            m_arr=self.likelihood_calculator.m_arr,
            likelihood=self.likelihood,
            seed=self.seed,
            max_bestfit_lnlike_diff=self.max_bestfit_lnlike_diff,
            size_limit=self.size_limit,
            int_block_size=self.int_block_size,
            ext_block_size=self.ext_block_size,
            min_bestfit_lnlike_to_keep=self.min_bestfit_lnlike_to_keep,
            dir_permissions=self.dir_permissions,
            file_permissions=self.file_permissions,
            full_intrinsic_indices=self.full_intrinsic_indices,
            n_samples_discarded=self.n_samples_discarded,
            logsumexp_discarded_ln_posterior=self.logsumexp_discarded_ln_posterior,
            logsumsqrexp_discarded_ln_posterior=self.logsumsqrexp_discarded_ln_posterior,
            n_samples_accepted=self.n_samples_accepted,
            logsumexp_accepted_ln_posterior=self.logsumexp_accepted_ln_posterior,
            logsumsqrexp_accepted_ln_posterior=self.logsumsqrexp_accepted_ln_posterior,
            n_distance_marginalizations=self.n_distance_marginalizations,
        )

        return init_dict

    def load_extrinsic_samples_data(self, folder_path):
        """
        Load the extrinsic sampels and derived quantities from files.
        Save as attributes to `self`.
        """

        folder_path = Path(folder_path)
        extrinsic_samples_path = folder_path / "extrinsic_samples.feather"
        detector_response_path = folder_path / "response_dpe.npy"
        timeshift_path = folder_path / "timeshift_dbe.npy"

        self.full_extrinsic_samples = pd.read_feather(extrinsic_samples_path)

        if detector_response_path.exists() and timeshift_path.exists():
            self.full_response_dpe = np.load(detector_response_path)
            self.full_timeshift_dbe = np.load(timeshift_path)
        else:
            (self.full_response_dpe, self.full_timeshift_dbe) = (
                self.extrinsic_sample_processor.get_components(
                    self.full_extrinsic_samples,
                    self.likelihood.fbin,
                    self.likelihood.event_data.tcoarse,
                )
            )
        self.full_log_prior_weights_e = (
            self.full_extrinsic_samples.log_prior_weights.values
        )

    def initialize_prob_samples(self):
        """
        Initialize the prob_samples dataframe.
        """
        self.prob_samples = pd.DataFrame(columns=self.PROB_SAMPLES_COLS)
        self.prob_samples.astype(
            {
                k: v
                for k, v in zip(self.PROB_SAMPLES_COLS, self.PROB_SAMPLES_COLS_DTYPES)
            }
        )

    @staticmethod
    def select_and_get_lnlike(dh_ieo, hh_ieo, min_bestfit_lnlike_to_keep=-np.inf):
        """
        Select elements ieo by having distance-fitted lnlike not less
        than cut_threshold below the maximum.
        Return three arrays with intrinsic, extrinsic and phi sample.
        """

        bestfit_lnlike = 0.5 * (dh_ieo**2) / hh_ieo * (dh_ieo > 0)

        # in case of too low / non existing minimal besfit-likelihood
        # to be accepted into

        accepted = bestfit_lnlike > min_bestfit_lnlike_to_keep

        bestfit_lnlike_k = bestfit_lnlike[accepted]

        # account for discarded samples
        # consider calculating explicitly the marginalized likelihoods
        # and priors and add to the logsumexp_discarded_samples and
        # logsumsqrexp_discarded_samples

        return (bestfit_lnlike_k, accepted)

    def create_a_likelihood_block(
        self,
        h_impb,
        response_dpe,
        timeshift_dbe,
        bank_i_inds,
        bank_e_inds,
    ):
        """
        Create a single likelihood block, and save it to a file.
        """

        dh_ieo, hh_ieo = self.likelihood_calculator.get_dh_hh_ieo(
            self.dh_weights_dmpb,
            h_impb,
            response_dpe,
            timeshift_dbe,
            self.hh_weights_dmppb,
            self.likelihood.asd_drift,
        )

        (bestfit_lnlike_k, accepted) = self.select_and_get_lnlike(
            dh_ieo, hh_ieo, self.min_bestfit_lnlike_to_keep
        )

        self.n_distance_marginalizations += np.sum(accepted)

        # Accepted Samples:
        # Find their marginalized likelihood & store to file
        if np.any(accepted):
            # i_k, e_k are indices within the block, not the full bank
            i_k, e_k, o_k = np.where(accepted)

            dh_k = dh_ieo[accepted]
            hh_k = hh_ieo[accepted]
            # indices to be used in the full banks / waveform loading
            bank_i_inds_k = bank_i_inds[i_k]
            bank_e_inds_k = bank_e_inds[e_k]
            bestfit_lnlike_max = bestfit_lnlike_k.max()
            # update minimal likelihood values to allow
            if self.min_bestfit_lnlike_to_keep < (
                bestfit_lnlike_max - self.max_bestfit_lnlike_diff
            ):
                self.min_bestfit_lnlike_to_keep = (
                    bestfit_lnlike_max - self.max_bestfit_lnlike_diff
                )

            dist_marg_lnlike_k = (
                self.likelihood_calculator.lookup_table.lnlike_marginalized(dh_k, hh_k)
            )
            # sort the samples by bestfit_lnlike_k
            sort_inds = np.argsort(bestfit_lnlike_k)
            i_k = i_k[sort_inds]
            e_k = e_k[sort_inds]
            o_k = o_k[sort_inds]
            dh_k = dh_k[sort_inds]
            hh_k = hh_k[sort_inds]
            bank_i_inds_k = bank_i_inds_k[sort_inds]
            bank_e_inds_k = bank_e_inds_k[sort_inds]
            dist_marg_lnlike_k = dist_marg_lnlike_k[sort_inds]
            bestfit_lnlike_k = bestfit_lnlike_k[sort_inds]

            # Saving & loading the block step by step could be
            # RAM efficient, but more likely just time consuming.
            # Consider just passing the data frames or storing it in "self"

            self._next_block = dict(
                i_k=i_k,
                e_k=e_k,
                o_k=o_k,
                dh_k=dh_k,
                hh_k=hh_k,
                dist_marg_lnlike_k=dist_marg_lnlike_k,
                bestfit_lnlike_k=bestfit_lnlike_k,
                bestfit_lnlike_max=bestfit_lnlike_max,
                bank_i_inds_k=bank_i_inds_k,
                bank_e_inds_k=bank_e_inds_k,
            )

        # Discarded samples:
        # account for number, ln_posterior and 2 * ln_posterior
        # of discarded samples
        discarded = ~accepted

        if np.any(discarded):
            i_k, e_k, o_k = np.where(discarded)
            bank_i_inds_k = bank_i_inds[i_k]
            bank_e_inds_k = bank_e_inds[e_k]
            # choose a subset
            n_discarded_samples = len(i_k)
            # subset_size = np.min((1000, len(i_k)))
            if n_discarded_samples < 1000:
                subset_size = len(i_k)
            elif n_discarded_samples < 10**8:
                subset_size = np.sqrt(n_discarded_samples).astype(int)
            else:
                subset_size = 10**4  # maximal allowed size

            subset = self.rng.choice(len(i_k), subset_size)

            # create a block with the subset
            i_k = i_k[subset]
            e_k = e_k[subset]
            o_k = o_k[subset]
            bank_i_inds_k = bank_i_inds_k[subset]
            bank_e_inds_k = bank_e_inds_k[subset]
            dh_k = dh_ieo[(i_k, e_k, o_k)]
            hh_k = hh_ieo[(i_k, e_k, o_k)]
            dist_marg_lnlike_k = np.zeros_like(dh_k)
            dist_marg_lnlike_k[dh_k > 0] = (
                self.likelihood_calculator.lookup_table.lnlike_marginalized(
                    dh_k[dh_k > 0], hh_k[dh_k > 0]
                )
            )
            discarded_block = {
                "i_k": i_k,
                "e_k": e_k,
                "o_k": o_k,
                "dist_marg_lnlike_k": dist_marg_lnlike_k,
                "bank_i_inds_k": bank_i_inds_k,
                "bank_e_inds_k": bank_e_inds_k,
            }

            self.store_discarded_samples_n_ln_posterior_from_block(
                discarded_block,
                discarded_inds=range(len(i_k)),
                n_discarded=n_discarded_samples,
            )

    def create_likelihood_blocks(
        self, tempdir, i_blocks, e_blocks, response_dpe, timeshift_dbe
    ):
        """
        Create likelihood blocks for the given intrinsic and extrinsic blocks.

        Parameters
        ----------
        tempdir : str or Path
            Directory to save the blocks.
        i_blocks : list of np.ndarray
            List of intrinsic blocks, where each block is an array of indices.
        e_blocks : list of np.ndarray
            List of extrinsic blocks, where each block is an array of indices.
        response_dpe : np.ndarray
            Response matrix (dimensions: intrinsic x extrinsic).
        timeshift_dbe : np.ndarray
            Time shift matrix (dimensions: intrinsic x extrinsic).

        Returns
        -------
        blocknames : list of Path
            Names of the created likelihood blocks.
        """
        tempdir = Path(tempdir)
        blocknames = []

        # Calculate total number of blocks for progress bar
        total_blocks = len(i_blocks) * len(e_blocks)

        # Iterate over all intrinsic and extrinsic block pairs with progress bar
        with tqdm(total=total_blocks, desc="Coherent likelihood blocks") as pbar:
            for i_block_idx, i_block in enumerate(i_blocks):
                # Validate i_block as an array of intrinsic indices
                if not isinstance(i_block, np.ndarray) or i_block.dtype != int:
                    raise ValueError(
                        f"Intrinsic block {i_block_idx} must be an integer array."
                    )

                # Load amplitude and phase for the current intrinsic block
                amp_impb, phase_impb = (
                    self.intrinsic_sample_processor.load_amp_and_phase(
                        self.waveform_dir, i_block
                    )
                )
                h_impb = amp_impb * np.exp(1j * phase_impb)

                for e_block_idx, e_block in enumerate(e_blocks):
                    # Validate e_block as an array of extrinsic indices
                    if not isinstance(e_block, np.ndarray) or e_block.dtype != int:
                        raise ValueError(
                            f"Extrinsic block {e_block_idx} must be an integer array."
                        )

                    # Create a filename for the current block pair
                    blockname = f"block_{i_block_idx}_{e_block_idx}.npz"
                    blocknames.append(blockname)

                    response_subset = response_dpe[..., e_block]
                    timeshift_subset = timeshift_dbe[..., e_block]

                    # Create the likelihood block
                    self.create_a_likelihood_block(
                        h_impb,
                        response_subset,
                        timeshift_subset,
                        i_block,
                        e_block,
                    )

                    # Combine with global probability samples if block file exists
                    self.combine_prob_samples_with_next_block()

                    # Update progress bar
                    pbar.update(1)

        return blocknames

    def _block_name(self, i_block, e_block):
        """Return the name of the block file."""
        return f"block_{i_block}_{e_block}.npz"

    def store_discarded_samples_n_ln_posterior_from_prob_samples(self, discarded_inds):
        """
        Collect the (unnormalized) sum of probabilities of the discarded
        samples in the prob_samples, and the sum of their squares.
        """
        if discarded_inds is None or len(discarded_inds) == 0:
            return

        ln_posterior = (
            self.prob_samples["ln_posterior"]
            .astype(np.float64, copy=False)
            .values[discarded_inds]
        )
        logsumexp_ln_posterior = safe_logsumexp(ln_posterior)
        logsumsqrtexp_ln_posterior = safe_logsumexp(2 * ln_posterior)
        n_samples_discarded = len(discarded_inds)

        self.n_samples_discarded += n_samples_discarded
        self.logsumexp_discarded_ln_posterior = safe_logsumexp(
            [logsumexp_ln_posterior, self.logsumexp_discarded_ln_posterior]
        )
        self.logsumsqrexp_discarded_ln_posterior = safe_logsumexp(
            [
                logsumsqrtexp_ln_posterior,
                self.logsumsqrexp_discarded_ln_posterior,
            ]
        )

    def store_discarded_samples_n_ln_posterior_from_block(
        self, block, discarded_inds, n_discarded=None
    ):
        """
        Collect the (unnormalized) sum of probabilities of the discarded
        samples in the block, and the sum of their squares.
        Has option to use a subset of samples, so that the block
        creation (namely `dist_marg_lnlike_k`) is cheaper. When using a
        subset, pass explicitly n_discarded. The two logsums,
        `logsumexp_ln_posterior` and `logsumsqrtexp_ln_posterior` will
        be added `subset_factor` to account for the actual number
        of samples discarded.

        Parameters
        ----------
        block : npz file, npz object or dict
            The block of likelihood evaluations & indices.
        i_start : int
            The starting index of the intrinsic block.
        e_start : int
            The starting index of the extrinsic block.
        discarded_inds : list
            The indices of the discarded samples.
        n_discarded : int, optional
            The number of samples discarded. If None, will be set to
            the length of `discarded_inds`.
        """

        if discarded_inds is None or len(discarded_inds) == 0:
            return

        if isinstance(block, (str, Path)):
            block = np.load(block)

        if n_discarded is None:
            n_discarded = len(discarded_inds)
        subset_factor = np.log(n_discarded / len(discarded_inds))

        discarded_i = block["bank_i_inds_k"][discarded_inds]
        discarded_e = block["bank_e_inds_k"][discarded_inds]

        ln_posterior = (
            block["dist_marg_lnlike_k"][discarded_inds]
            + self._get_intrinsic_logw(discarded_i)
            + self.full_log_prior_weights_e[discarded_e]
        )

        logsumexp_ln_posterior = safe_logsumexp(ln_posterior) + subset_factor
        logsumsqrtexp_ln_posterior = safe_logsumexp(2 * ln_posterior) + subset_factor

        self.logsumexp_discarded_ln_posterior = safe_logsumexp(
            [logsumexp_ln_posterior, self.logsumexp_discarded_ln_posterior]
        )

        self.logsumsqrexp_discarded_ln_posterior = safe_logsumexp(
            [
                logsumsqrtexp_ln_posterior,
                self.logsumsqrexp_discarded_ln_posterior,
            ]
        )

        self.n_samples_discarded += n_discarded

    def combine_block_dict_into_prob_samples(self, block_dict):
        """
        Combine the prob_samples dataframe with a likelihood block given as a dict.

        The block_dict must contain the same keys as _next_block: i_k, e_k, o_k,
        dh_k, hh_k, dist_marg_lnlike_k, bestfit_lnlike_k, bank_i_inds_k, bank_e_inds_k.
        Used by the sequential combiner after HPC parallel block creation.
        """
        if (
            not block_dict
            or not isinstance(block_dict, dict)
            or "bestfit_lnlike_k" not in block_dict
        ):
            self.log("No valid block to combine with prob_samples")
            return

        block = block_dict
        new_bestfit_lnlike_k = block["bestfit_lnlike_k"]

        # Assume prob_samples and the block are sorted by bestfit_lnlike.
        prob_samples_accepted_inds, block_samples_accepted_inds = (
            get_top_n_indices_two_pointer(
                self.prob_samples.bestfit_lnlike.to_numpy(np.float64),
                new_bestfit_lnlike_k,
                self.size_limit,
            )
        )

        # Compute discarded block indices without setdiff1d
        n_block = new_bestfit_lnlike_k.shape[0]
        block_keep_mask = np.zeros(n_block, dtype=bool)
        block_keep_mask[block_samples_accepted_inds] = True
        block_samples_discarded_inds = np.nonzero(~block_keep_mask)[0]

        if block_samples_discarded_inds.size:
            self.store_discarded_samples_n_ln_posterior_from_block(
                block, block_samples_discarded_inds
            )

        # Compute discarded prob_samples indices without setdiff1d
        n_prob = self.prob_samples.shape[0]
        prob_keep_mask = np.zeros(n_prob, dtype=bool)
        prob_keep_mask[prob_samples_accepted_inds] = True
        prob_samples_discarded_inds = np.nonzero(~prob_keep_mask)[0]

        if prob_samples_discarded_inds.size:
            self.store_discarded_samples_n_ln_posterior_from_prob_samples(
                prob_samples_discarded_inds
            )

        # if any new samples are accepted, concatenate them to self.prob_samples
        if block_samples_accepted_inds.size:
            new_inds_o = block["o_k"][block_samples_accepted_inds]
            new_dh_k = block["dh_k"][block_samples_accepted_inds]
            new_hh_k = block["hh_k"][block_samples_accepted_inds]
            new_lnl_marginalized = block["dist_marg_lnlike_k"][
                block_samples_accepted_inds
            ]
            new_bestfit_lnlike = block["bestfit_lnlike_k"][block_samples_accepted_inds]
            new_inds_i = block["bank_i_inds_k"][block_samples_accepted_inds]
            new_inds_e = block["bank_e_inds_k"][block_samples_accepted_inds]
            new_ln_posterior = (
                new_lnl_marginalized
                + self._get_intrinsic_logw(new_inds_i)
                + self.full_log_prior_weights_e[new_inds_e]
            )

            new_samples_df = pd.DataFrame(
                {
                    "i": new_inds_i,
                    "e": new_inds_e,
                    "o": new_inds_o,
                    "lnl_marginalized": new_lnl_marginalized,
                    "ln_posterior": new_ln_posterior,
                    "bestfit_lnlike": new_bestfit_lnlike,
                    "d_h_1Mpc": new_dh_k,
                    "h_h_1Mpc": new_hh_k,
                }
            )

            # Concatenate `self.prob_samples` and `new_samples_df` in a safe manner.
            if self.prob_samples.size:
                self.prob_samples = pd.concat(
                    [
                        self.prob_samples.iloc[prob_samples_accepted_inds],
                        new_samples_df,
                    ],
                    ignore_index=True,
                )
            else:
                self.prob_samples = new_samples_df

        self.prob_samples.sort_values(by="bestfit_lnlike", inplace=True)
        self.prob_samples.reset_index(drop=True, inplace=True)
        # if prob_samples is at maximum allowed size, accept only
        # samples with likelihood higher than its minimum likelihood.
        if self.prob_samples.size == self.size_limit:
            self.min_bestfit_lnlike_to_keep = np.max(
                (
                    self.prob_samples["bestfit_lnlike"].values.min(),
                    self.min_bestfit_lnlike_to_keep,
                )
            )

    def combine_prob_samples_with_next_block(self):
        """
        Combine the prob_samples dataframe with a likelihood block.
        """
        if (
            not self._next_block
            or not isinstance(self._next_block, dict)
            or "bestfit_lnlike_k" not in self._next_block
        ):
            self.log("No valid block to combine with prob_samples")
            return
        self.combine_block_dict_into_prob_samples(self._next_block)
        self._next_block = {}

    def get_bestfit_and_marginalized_lnlike(self, par_dic):
        """
        Get the distance marginalized likelihood for a given par_dic.
        """
        par_dic = par_dic | {
            "d_luminosity": self.likelihood_calculator.lookup_table.REFERENCE_DISTANCE
        }

        h_f = self.likelihood._get_h_f(par_dic)
        d_h = np.sum(self.likelihood._compute_d_h(h_f).real)
        h_h = np.sum(self.likelihood._compute_h_h(h_f))

        lnl_marginalized = self.likelihood_calculator.lookup_table.lnlike_marginalized(
            d_h, h_h
        )
        bestfit_lnlike = 0.5 * (d_h**2) / h_h * (d_h > 0)

        return bestfit_lnlike, lnl_marginalized

    def combine_samples(
        self, prob_samples, intrinsic_samples, extrinsic_samples, n_phi
    ):
        """
        Combine the intrinsic, extrinsic and prob_samples to create a
        dataframe of samples with both parameters and probabilistic
        values.
        """

        combined_samples = pd.concat(
            [
                intrinsic_samples.iloc[prob_samples["i"].values].reset_index(drop=True),
                extrinsic_samples.iloc[prob_samples["e"].values].reset_index(drop=True),
            ],
            axis=1,
        )

        combined_samples.drop(
            columns=["weights", "log_prior_weights", "original_index"],
            inplace=True,
        )

        combined_samples = pd.concat([combined_samples, prob_samples], axis=1)

        combined_samples["phi"] = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)[
            combined_samples["o"].values
        ]

        combined_samples["weights"] = exp_normalize(
            combined_samples["ln_posterior"].values
        )

        # Apply changes for linear free timeshifts.
        # See linear_free_timeshifts.py for details about the convention.
        # suffix _u represent unique intrinsic indices
        combined_samples.rename(
            columns={
                "t_geocenter": "t_geocenter_linfree",
                "phi": "phi_ref_linfree",
            },
            inplace=True,
        )
        # load timeshifts and phaseshifts of the intrinsic samples.
        # suffix _u represent unique intrinsic indices
        unique_i = np.unique(prob_samples["i"].values)
        u_i = np.searchsorted(unique_i, prob_samples["i"].values)
        # prob_samples['i'].values == unique_i[u_i]

        dt_linfree_u, dphi_linfree_u = (
            self.intrinsic_sample_processor.load_linfree_dt_and_dphi(
                self.waveform_dir, unique_i
            )
        )

        combined_samples["t_geocenter"] = (
            combined_samples["t_geocenter_linfree"] + dt_linfree_u[u_i]
        )

        combined_samples["phi_ref"] = (
            combined_samples["phi_ref_linfree"] - dphi_linfree_u[u_i]
        )

        return combined_samples


class CoherentExtrinsicSamplesGenerator(JSONMixin, Loggable):
    """
    A class capable of using Extrinsic Marginalization classes and
    methods to draw extrinsic samples from the data.
    Relatively slow to initialize and use.
    """

    DEFAULT_GET_MARG_INFO_KWARGS = {"n_combine": 16, "save_marg_info": True}

    def __init__(
        self,
        likelihood,
        intrinsic_bank_file,
        waveform_dir,
        seed=None,
        full_intrinsic_indices=None,
    ):
        """
        Initialization of the CoherentExtrinsicSamplesGenerator.

        Parameters
        ----------
        likelihood : Likelihood
            The likelihood object. Must contain a CoherentScore object.
        intrinsic_bank_file : str or Path
            Path to the intrinsic sample bank file.
        waveform_dir : str or Path
            Directory containing waveform data.
        seed : int, optional
            Random seed for reproducibility.
        full_intrinsic_indices : np.ndarray, optional
            Indices of intrinsic samples to use. If None, all samples are used.
        """

        self.likelihood = likelihood
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.setup_logger()

        self.waveform_dir = waveform_dir
        self.intrinsic_bank_file = intrinsic_bank_file

        self.extrinsic_sample_processor = sample_processing.ExtrinsicSampleProcessor(
            self.likelihood.event_data.detector_names
        )
        self.intrinsic_sample_processor = sample_processing.IntrinsicSampleProcessor(
            self.likelihood, self.waveform_dir
        )

        self.intrinsic_sample_bank = self.intrinsic_sample_processor.load_bank(
            intrinsic_bank_file, full_intrinsic_indices
        )

        self.full_intrinsic_indices = (
            full_intrinsic_indices
            if full_intrinsic_indices
            else np.arange(len(self.intrinsic_sample_bank))
        )

    def get_marg_info_batch(
        self,
        batch_idx,
        bank,
        waveform_dir=None,
        min_marg_lnlike_for_sampling=0.0,
        single_marg_info_min_n_effective_prior=0.0,
    ):
        """
        Process a batch of indices:
        - Filter indices based on the marginalized likelihood threshold.
        - Load waveforms.
        - Compute marginalization info for each valid sample.

        Returns a list of tuples (marg_info_object, index).
        """
        # Filter indices using the likelihood threshold
        valid_ids = [
            int(i)
            for i in batch_idx
            if self.likelihood.lnlike(
                bank.iloc[i].to_dict() | config.DEFAULT_PARAMS_DICT
            )
            > min_marg_lnlike_for_sampling
        ]
        if not valid_ids:
            return [], []

        # Load waveforms for the valid indices
        amp_impb, phase_impb = self.intrinsic_sample_processor.load_amp_and_phase(
            waveform_dir, np.array(valid_ids)
        )

        # Process waveforms to obtain dh and hh batches
        dh_batch, hh_batch = self.likelihood._get_many_dh_hh(
            amp_impb,
            phase_impb,
            self.likelihood._d_h_weights,
            self.likelihood._h_h_weights,
            self.intrinsic_sample_processor.m_inds,
            self.intrinsic_sample_processor.mprime_inds,
            self.likelihood.asd_drift,
        )

        marg_info_batch = []
        used_indices_batch = []

        # Iterate over each sample in the batch
        for dh, hh, idx in zip(dh_batch, hh_batch, valid_ids):
            self.likelihood.coherent_score._switch_qmc_sequence(0)
            mi = self.likelihood.coherent_score.get_marginalization_info(
                dh, hh, self.likelihood._times
            )

            if mi.n_effective_prior > single_marg_info_min_n_effective_prior:
                marg_info_batch.append(mi)
                used_indices_batch.append(idx)
        return marg_info_batch, used_indices_batch

    def get_marg_info_batch_multibank(
        self,
        batch_sample_idx,
        batch_bank_idx,
        banks,
        waveform_dirs,
        min_marg_lnlike_for_sampling,
        single_marg_info_min_n_effective_prior,
    ):
        """
        Process a batch of (bank_idx, sample_idx) pairs for multibank extrinsic sampling.
        Filters in batch order, loads waveforms grouped by bank but preserves shuffled order,
        then builds MarginalizationInfo objects.

        Parameters
        ----------
        batch_sample_idx : np.ndarray
            1D array of iloc positions within the relevant bank dataframe.
        batch_bank_idx : np.ndarray
            1D array of int bank indices (0..n_banks-1), same length as batch_sample_idx.
        banks : list[pd.DataFrame]
            List of bank DataFrames, indexed by bank idx.
        waveform_dirs : list[Path]
            List of waveform directories, indexed by bank idx.
        min_marg_lnlike_for_sampling : float
            Minimum marginalized likelihood threshold.
        single_marg_info_min_n_effective_prior : float
            Minimum n_effective_prior threshold for accepting MarginalizationInfo objects.

        Returns
        -------
        marg_info_batch : list
            List of accepted MarginalizationInfo objects.
        used_bank_idx_batch : list[int]
            List of bank indices for accepted objects.
        used_sample_idx_batch : list[int]
            List of sample indices for accepted objects.
        """
        # Filter in batch order (preserves shuffled order)
        valid_mask = np.array(
            [
                self.likelihood.lnlike(
                    banks[int(bank_idx)].iloc[int(sample_idx)].to_dict()
                    | config.DEFAULT_PARAMS_DICT
                )
                > min_marg_lnlike_for_sampling
                for sample_idx, bank_idx in zip(batch_sample_idx, batch_bank_idx)
            ],
            dtype=bool,
        )
        if not np.any(valid_mask):
            return [], [], []

        valid_sample_idx = np.asarray(batch_sample_idx, dtype=int)[valid_mask]
        valid_bank_idx = np.asarray(batch_bank_idx, dtype=int)[valid_mask]

        # Load waveforms per bank, scatter back preserving valid order
        # Cannot initialize arrays upfront: shape depends on waveform data (n_modes, n_pol, n_fbin)
        # which we only know after loading the first batch
        amp_impb = None
        phase_impb = None

        for bank_idx in np.unique(valid_bank_idx):
            pos = np.where(valid_bank_idx == int(bank_idx))[0]
            ids = valid_sample_idx[pos]

            amp_bank, phase_bank = self.intrinsic_sample_processor.load_amp_and_phase(
                Path(waveform_dirs[int(bank_idx)]), np.asarray(ids, dtype=int)
            )

            if amp_impb is None:
                amp_impb = np.empty(
                    (len(valid_sample_idx),) + amp_bank.shape[1:], dtype=amp_bank.dtype
                )
                phase_impb = np.empty(
                    (len(valid_sample_idx),) + phase_bank.shape[1:],
                    dtype=phase_bank.dtype,
                )

            amp_impb[pos] = amp_bank
            phase_impb[pos] = phase_bank

        # Process waveforms to obtain dh and hh batches
        dh_batch, hh_batch = self.likelihood._get_many_dh_hh(
            amp_impb,
            phase_impb,
            self.likelihood._d_h_weights,
            self.likelihood._h_h_weights,
            self.intrinsic_sample_processor.m_inds,
            self.intrinsic_sample_processor.mprime_inds,
            self.likelihood.asd_drift,
        )

        marg_info_batch = []
        used_bank_idx_batch = []
        used_sample_idx_batch = []

        # Iterate in valid (shuffled) order
        for dh, hh, bank_idx, sample_idx in zip(
            dh_batch, hh_batch, valid_bank_idx, valid_sample_idx
        ):
            self.likelihood.coherent_score._switch_qmc_sequence(0)
            mi = self.likelihood.coherent_score.get_marginalization_info(
                dh, hh, self.likelihood._times
            )

            if mi.n_effective_prior > single_marg_info_min_n_effective_prior:
                marg_info_batch.append(mi)
                used_bank_idx_batch.append(int(bank_idx))
                used_sample_idx_batch.append(int(sample_idx))

        return marg_info_batch, used_bank_idx_batch, used_sample_idx_batch

    def get_marg_info(
        self,
        n_combine,
        indices=None,
        min_marg_lnlike_for_sampling=0.0,
        single_marg_info_min_n_effective_prior=0.0,
        save_marg_info=True,
        save_marg_info_dir=None,
        waveform_dir=None,
    ):
        """
        Create a MarginalizationInfo object from a set of indices.

        Parameters
        ----------
        n_combine : int
            The number of MarginalizationInfo objects to combine.
        indices : np.ndarray, optional
            The indices of the intrinsic samples to use. If None, all
            samples are used. Default is None.
        min_marg_lnlike_for_sampling : float, optional
            The minimum marginalized likelihood for a sample to be
            used in the MarginalizationInfo object.
        single_marg_info_min_n_effective_prior : float, optional
            The minimum effective number of samples for a single
            MarginalizationInfo object to be used.
        save_marg_info : bool, optional
            If True, save the MarginalizationInfo object to a file.
        save_marg_info_dir: str, optional
            Folder path to save the MarginalizationInfo into.
        waveform_dir: str, optional
            Path for waveforms. Default to None, which uses value from
            self.waveform_dir.

        """

        self.log(f"Getting {n_combine} MarginalizationInfo objects.")
        bank = pd.read_feather(self.intrinsic_bank_file)
        if waveform_dir is None:
            waveform_dir = self.waveform_dir
        if indices is None:
            indices = self.full_intrinsic_indices

        if save_marg_info and (save_marg_info_dir is None):
            save_marg_info_dir = Path(".")

        # Set the batch size and shuffle the indices
        batch_size = min(n_combine * 2, len(indices))
        indices = self.rng.permutation(indices)
        batches = np.array_split(indices, np.ceil(len(indices) / batch_size))
        marg_info_i = []
        used_indices = []
        total_batches = len(batches)

        # Progress bar for marginalization object collection with batch info
        with tqdm(total=n_combine, desc="Marginalization objects", unit="obj") as pbar:
            # Collect enough MarginalizationInfo objects
            for batch_idx, batch in enumerate(batches):
                marg_info_batch, used_indices_batch = self.get_marg_info_batch(
                    batch,
                    bank,
                    waveform_dir,
                    min_marg_lnlike_for_sampling,
                    single_marg_info_min_n_effective_prior,
                )
                for mi, idx in zip(marg_info_batch, used_indices_batch):
                    marg_info_i.append(mi)
                    used_indices.append(idx)
                    pbar.update(1)
                    if len(marg_info_i) >= n_combine:
                        break

                # Update postfix to show batches processed
                pbar.set_postfix(
                    {
                        "batches": f"{batch_idx + 1}/{total_batches}",
                        "accepted": f"{len(marg_info_i)}/{n_combine}",
                    }
                )

                if len(marg_info_i) >= n_combine:
                    break

        if not marg_info_i:
            raise ValueError(
                f"No valid MarginalizationInfo objects found. No file saved to {save_marg_info_dir}"
            )

        # Merge collected objects into one
        marg_info = deepcopy(marg_info_i[0])
        if len(marg_info_i) > 1:
            marg_info.update_with_list([mi for mi in marg_info_i[1:]])
        self.log("MarginalizationInfo object created!")

        # Optionally save the resulting MarginalizationInfo object and indices
        if save_marg_info:
            save_marg_info_dir = Path(save_marg_info_dir)
            with open(save_marg_info_dir / "marg_info.pkl", "wb") as f:
                pickle.dump(marg_info, f)
            np.save(
                save_marg_info_dir / "indices_used_for_marg_info.npy",
                used_indices,
            )
            self.log(f"Saving MarginalizationInfo object to {save_marg_info_dir}.")

        return marg_info

    def get_marg_info_multibank(
        self,
        n_combine,
        banks,
        waveform_dirs,
        sample_idx_arr,
        bank_idx_arr,
        min_marg_lnlike_for_sampling=0.0,
        single_marg_info_min_n_effective_prior=0.0,
        save_marg_info=True,
        save_marg_info_dir=None,
    ):
        """
        Create a MarginalizationInfo object from globally shuffled (bank_idx, sample_idx) pairs
        across multiple banks, without replacement.

        Parameters
        ----------
        n_combine : int
            The number of MarginalizationInfo objects to combine.
        banks : list[pd.DataFrame]
            List of bank DataFrames, indexed by bank idx.
        waveform_dirs : list[Path]
            List of waveform directories, indexed by bank idx.
        sample_idx_arr : np.ndarray
            1D array of sample indices (iloc positions), already globally shuffled.
        bank_idx_arr : np.ndarray
            1D array of bank indices, same length as sample_idx_arr, already globally shuffled.
        min_marg_lnlike_for_sampling : float, optional
            The minimum marginalized likelihood for a sample to be used.
        single_marg_info_min_n_effective_prior : float, optional
            The minimum effective number of samples for a single MarginalizationInfo object.
        save_marg_info : bool, optional
            If True, save the MarginalizationInfo object to a file.
        save_marg_info_dir : str or Path, optional
            Folder path to save the MarginalizationInfo into.

        Returns
        -------
        marg_info : MarginalizationInfo
            Merged MarginalizationInfo object.
        """
        # Ensure arrays
        sample_idx_arr = np.asarray(sample_idx_arr, dtype=int)
        bank_idx_arr = np.asarray(bank_idx_arr, dtype=int)
        assert sample_idx_arr.shape == bank_idx_arr.shape, (
            f"sample_idx_arr and bank_idx_arr must have same shape, "
            f"got {sample_idx_arr.shape} vs {bank_idx_arr.shape}"
        )

        # Disable dt cache for multibank extrinsic usage
        use_cached_dt_prev = self.intrinsic_sample_processor.use_cached_dt
        update_cached_dt_prev = self.intrinsic_sample_processor.update_cached_dt
        self.intrinsic_sample_processor.use_cached_dt = False
        self.intrinsic_sample_processor.update_cached_dt = False

        try:
            self.log(
                f"Getting {n_combine} MarginalizationInfo objects from {len(banks)} banks."
            )

            if save_marg_info and (save_marg_info_dir is None):
                save_marg_info_dir = Path(".")
            elif save_marg_info_dir is not None:
                save_marg_info_dir = Path(save_marg_info_dir)

            # Choose a safe batch size
            batch_size = max(1024, 2 * int(n_combine))

            # Split into sequential batches (no replacement - iterate through shuffled arrays)
            n_total = len(sample_idx_arr)
            n_batches = int(np.ceil(n_total / batch_size))

            marg_info_i = []
            used_bank_idx = []
            used_sample_idx = []

            with tqdm(
                total=n_combine, desc="Marginalization objects", unit="obj"
            ) as pbar:
                for bi in range(n_batches):
                    start = bi * batch_size
                    end = min((bi + 1) * batch_size, n_total)

                    batch_samples = sample_idx_arr[start:end]
                    batch_banks = bank_idx_arr[start:end]

                    mi_batch, used_b_batch, used_s_batch = (
                        self.get_marg_info_batch_multibank(
                            batch_samples,
                            batch_banks,
                            banks,
                            waveform_dirs,
                            min_marg_lnlike_for_sampling,
                            single_marg_info_min_n_effective_prior,
                        )
                    )

                    for mi, b, s in zip(mi_batch, used_b_batch, used_s_batch):
                        marg_info_i.append(mi)
                        used_bank_idx.append(b)
                        used_sample_idx.append(s)
                        pbar.update(1)
                        if len(marg_info_i) >= n_combine:
                            break

                    pbar.set_postfix(
                        {
                            "batches": f"{bi + 1}/{n_batches}",
                            "accepted": f"{len(marg_info_i)}/{n_combine}",
                        }
                    )

                    if len(marg_info_i) >= n_combine:
                        break

            if len(marg_info_i) == 0:
                raise RuntimeError(
                    "No marginalization objects accepted. Try lowering thresholds or increasing candidate pool."
                )

            # Merge collected objects into one
            marg_info = deepcopy(marg_info_i[0])
            if len(marg_info_i) > 1:
                marg_info.update_with_list(marg_info_i[1:])
            self.log("MarginalizationInfo object created!")

            # Optionally save the resulting MarginalizationInfo object and indices
            if save_marg_info:
                save_marg_info_dir.mkdir(parents=True, exist_ok=True)
                with open(save_marg_info_dir / "marg_info.pkl", "wb") as f:
                    pickle.dump(marg_info, f)
                np.save(
                    save_marg_info_dir / "used_bank_idx.npy",
                    np.asarray(used_bank_idx, dtype=int),
                )
                np.save(
                    save_marg_info_dir / "used_sample_idx.npy",
                    np.asarray(used_sample_idx, dtype=int),
                )
                self.log(f"Saving MarginalizationInfo object to {save_marg_info_dir}.")

            return marg_info

        finally:
            # Restore cache flags
            self.intrinsic_sample_processor.use_cached_dt = use_cached_dt_prev
            self.intrinsic_sample_processor.update_cached_dt = update_cached_dt_prev

    def draw_extrinsic_samples_from_indices(
        self,
        n_ext,
        marg_info=None,
        get_marg_info_kwargs=None,
    ):
        """
        Use the marginalization_info object to draw extrinsic samples.
        Either use passed marg_info, or create a new one.
        """
        # Load or create marg_info
        if marg_info is None:
            if get_marg_info_kwargs is None:
                get_marg_info_kwargs = self.DEFAULT_GET_MARG_INFO_KWARGS
            marg_info = self.get_marg_info(**get_marg_info_kwargs)
        elif isinstance(marg_info, (str, Path)):
            marg_info_path = Path(marg_info)
            with open(marg_info_path, "rb") as f:
                marg_info = pickle.load(f)
        extrinsic_samples = pd.DataFrame(
            self.likelihood.coherent_score.gen_samples_from_marg_info(
                marg_info, num=n_ext
            )
        )

        extrinsic_samples["log_prior_weights"] = np.log(extrinsic_samples.weights)
        extrinsic_samples.rename(columns={"dec": "lat"}, inplace=True)

        (response_dpe, timeshift_dbe) = self.extrinsic_sample_processor.get_components(
            extrinsic_samples,
            self.likelihood.fbin,
            self.likelihood.event_data.tcoarse,
        )

        return (
            extrinsic_samples,
            response_dpe,
            timeshift_dbe,
        )
