"""
Classes and methods for single detector usecases:
1. Vetoing.
2. Finding best-fit waveforms for the purpose of vetoing.
3. Selection of intrinsic samples consistent with a single detector,
   before performing multi-detector integration.
"""

from pathlib import Path
from collections import defaultdict
import itertools
import numpy as np
import pandas as pd
from lal import GreenwichMeanSiderealTime
from scipy.stats import chi2
import matplotlib.pyplot as plt

# add cogwheel path

from cogwheel import skyloc_angles
from cogwheel import utils
from cogwheel import gw_utils
from cogwheel.sampler_free.base_sampler_free_sampling import (
    get_top_n_indices_two_pointer,
    Loggable,
)
from cogwheel.sampler_free import evidence_calculator


class BlockLikelihood(utils.JSONMixin, Loggable):
    """
    A class with the ability to combine intrinsic and extrinsic samples
    into likelihood evaluations, with optimization over distance and
    a grid of orbital phases.

    This method is made for runtime optimization in the single detector
    case in mind.
    """

    DEFAULT_SIZE_LIMIT = 10**6

    LIGHT_SAMPLES_COLS = [
        "i",
        "e",
        "o",
        "bestfit_lnlike",
        "d_h_1Mpc",
        "h_h_1Mpc",
    ]

    LIGHT_SAMPLES_COLS_DTYPES = [
        int,
        int,
        int,
        float,
        float,
        float,
    ]

    def __init__(
        self,
        intrinsic_bank_file,
        waveform_dir,
        n_phi,
        m_arr,
        likelihood,
        max_bestfit_lnlike_diff=20,
        size_limit=None,
        int_block_size=512,
        ext_block_size=512,
        min_bestfit_lnlike_to_keep=None,
        dir_permissions=utils.DIR_PERMISSIONS,
        file_permissions=utils.FILE_PERMISSIONS,
        full_intrinsic_indices=None,
    ):
        """
        Initialization of the BlockLikelihood.

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
        likelihood : Likelihood
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
        self.intrinsic_sample_processor = evidence_calculator.IntrinsicSampleProcessor(
            self.likelihood, self.waveform_dir
        )
        self.gmst = GreenwichMeanSiderealTime(self.likelihood.event_data.tgps)
        self.extrinsic_sample_processor = evidence_calculator.ExtrinsicSampleProcessor(
            self.likelihood.event_data.detector_names
        )

        self.evidence = evidence_calculator.Evidence(n_phi=n_phi, m_arr=np.array(m_arr))

        self.dh_weights_dmpb, self.hh_weights_dmppb = (
            self.intrinsic_sample_processor.get_summary()
        )

        self.cur_rundir = None  # current rundir

        # this is the raw intrinsic sample bank, used to draw samples
        self.intrinsic_sample_bank = self.intrinsic_sample_processor.load_bank(
            intrinsic_bank_file, full_intrinsic_indices
        )

        self.full_intrinsic_indices = (
            full_intrinsic_indices
            if full_intrinsic_indices
            else np.arange(len(self.intrinsic_sample_bank))
        )

        self.int_block_size = int_block_size
        self.ext_block_size = ext_block_size

        self.size_limit = size_limit if size_limit else self.DEFAULT_SIZE_LIMIT
        self.full_response_dpe = None
        self.full_timeshifts_dbe = None

        # set the stop conditions for the run
        self.block_file_list = []  # list of block files created.
        self.max_bestfit_lnlike_diff = max_bestfit_lnlike_diff
        # minimal bestfit lnlike to keep is updated throughout the run
        if min_bestfit_lnlike_to_keep is None:
            self.min_bestfit_lnlike_to_keep = -np.inf
        else:
            self.min_bestfit_lnlike_to_keep = min_bestfit_lnlike_to_keep

        self.block_file_list = []
        self.initialize_light_samples()

        self.h_impb = None  # set by loading with
        # intrinsic_sample_processor or explicitly

        self.setup_logger()

    def initialize_light_samples(self):
        """
        Initialize the light_samples dataframe.
        """
        self.light_samples = pd.DataFrame(columns=self.LIGHT_SAMPLES_COLS)
        self.light_samples.astype(
            dict(zip(self.LIGHT_SAMPLES_COLS, self.LIGHT_SAMPLES_COLS_DTYPES))
        )

    def select_and_get_lnlike(
        self,
        dh_ieo,
        hh_ieo,
    ):
        """
        Select elements ieo by having distance-fitted lnlike not less
        than cut_threshold below the maximum.
        Return three arrays with intrinsic, extrinsic and phi sample.
        """

        bestfit_lnlike = 0.5 * (dh_ieo**2) / hh_ieo * (dh_ieo > 0)

        if (
            self.max_bestfit_lnlike_diff is not None
            and self.min_bestfit_lnlike_to_keep is not None
        ):
            proposal_min = bestfit_lnlike.max() - self.max_bestfit_lnlike_diff
            if proposal_min > self.min_bestfit_lnlike_to_keep:
                self.min_bestfit_lnlike_to_keep = proposal_min

        accepted = bestfit_lnlike > self.min_bestfit_lnlike_to_keep
        bestfit_lnlike_k = bestfit_lnlike[accepted]

        return bestfit_lnlike_k, accepted

    def create_a_likelihood_block(
        self,
        filename,
        h_impb,
        response_dpe,
        timeshift_dbe,
        bank_i_inds,
        bank_e_inds,
    ):
        """
        Create a single likelihood block, and save it to a file.
        """

        dh_ieo, hh_ieo = self.evidence.get_dh_hh_ieo(
            self.dh_weights_dmpb,
            h_impb,
            response_dpe,
            timeshift_dbe,
            self.hh_weights_dmppb,
            self.likelihood.asd_drift,
        )

        (bestfit_lnlike_k, accepted) = self.select_and_get_lnlike(
            dh_ieo,
            hh_ieo,
        )
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
            proposal_min = bestfit_lnlike_max - self.max_bestfit_lnlike_diff
            if self.min_bestfit_lnlike_to_keep < proposal_min:
                self.min_bestfit_lnlike_to_keep = proposal_min

            # sort the samples by bestfit_lnlike_k
            sort_inds = np.argsort(bestfit_lnlike_k)
            i_k = i_k[sort_inds]
            e_k = e_k[sort_inds]
            o_k = o_k[sort_inds]
            dh_k = dh_k[sort_inds]
            hh_k = hh_k[sort_inds]
            bank_i_inds_k = bank_i_inds_k[sort_inds]
            bank_e_inds_k = bank_e_inds_k[sort_inds]
            bestfit_lnlike_k = bestfit_lnlike_k[sort_inds]

            np.savez(
                filename,
                i_k=i_k,
                e_k=e_k,
                o_k=o_k,
                dh_k=dh_k,
                hh_k=hh_k,
                bestfit_lnlike_k=bestfit_lnlike_k,
                bestfit_lnlike_max=bestfit_lnlike_max,
                bank_i_inds_k=bank_i_inds_k,
                bank_e_inds_k=bank_e_inds_k,
            )

    def create_likelihood_blocks(
        self, tempdir, i_blocks, e_blocks, response_dpe, timeshift_dbe
    ):
        """
        Create likelihood blocks for the given intrinsic and extrinsic
        blocks. Combine them with the light samples, and delete the block.
        Return the filenames of the created blocks.

        Parameters
        ----------
        tempdir : str or Path
            The directory to save the blocks.
        i_blocks : list
            The list of intrinsic blocks.
        e_blocks : list
            The list of extrinsic blocks.
        response_dpe : np.ndarray
            The response matrix.
        timeshift_dbe : np.ndarray
            The time shift matrix.
        """
        tempdir = Path(tempdir)
        filenames = []  # list of filenames of created blocks
        # The subset of intrinsic indices in the block
        # do not have to be a simple range, e.g. if some
        # rejection sampling is applied. We use int_indices to specify
        # the indices allowed.

        if self.h_impb is None:
            waveforms_to_load = (max(i_blocks) + 1) * self.int_block_size
            amp, phase = self.intrinsic_sample_processor.load_amp_and_phase(
                self.waveform_dir, waveforms_to_load
            )
            self.h_impb = amp * np.exp(1j * phase)
        # i_to_e_dict is a dictionary tha maps intrinsic block indices
        # to extrinsic block indices, of files to be created.
        # e.g. if need to create block (i=3, e=4), then
        # i_to_e_dict[3] = [4,]
        i_to_e_dict = defaultdict(list)
        # fill the dictionary
        for i in i_blocks:
            for e in e_blocks:
                # make sure file not already used.
                if (tempdir / self._block_name(i, e)) not in self.block_file_list:
                    i_to_e_dict[i].append(e)

        # iterate over blocks:
        for i_block, es in i_to_e_dict.items():
            i_start = i_block * self.int_block_size
            i_end = (i_block + 1) * self.int_block_size

            bank_i_inds_in_block = np.arange(i_start, i_end)

            h_impb = self.get_h_impb(bank_i_inds_in_block)

            for e_block in es:
                filename = tempdir / self._block_name(i_block, e_block)
                filenames.append(filename)
                self.log(f"starting {self._block_name(i_block, e_block)}")

                e_start = e_block * self.ext_block_size
                e_end = (e_block + 1) * self.ext_block_size

                # Unlike intrinsic samples, all the extrinsic components
                # are passed to this function, therefor there is no need
                # to keep track of the indices using other arrays.
                range_e = np.array(range(e_start, e_end))

                self.create_a_likelihood_block(
                    filename,
                    h_impb,
                    response_dpe[..., range_e],
                    timeshift_dbe[..., range_e],
                    bank_i_inds_in_block,
                    range_e,
                )
                # combine the block with the light samples.
                # If no samples are accepted, the file is not created.
                if filename.exists():
                    self.combine_light_samples_with_block(filename)

                    # Delete the block file to free space
                    if filename.is_file():
                        filename.unlink()

        return filenames

    def get_h_impb(self, inds):
        """
        Access attribute h_impb or return None if it is None
        """
        if self.h_impb is not None:
            return self.h_impb[inds]
        return None

    def _block_name(self, i_block, e_block):
        """Return the name of the block file."""
        return f"block_{i_block}_{e_block}.npz"

    def combine_light_samples_with_block(self, block_file):
        """
        Combine the light_samples dataframe with a likelihood block.
        """

        block = np.load(block_file)
        new_bestfit_lnlike_k = block["bestfit_lnlike_k"]

        # Assume light_samples and the block are sorted by
        # bestfit_lnlike.
        light_samples_accepted_inds, block_samples_accepted_inds = (
            get_top_n_indices_two_pointer(
                self.light_samples.bestfit_lnlike.values,
                new_bestfit_lnlike_k,
                self.size_limit,
            )
        )

        # if any new samples are accepted, concatenate them to
        # `self.light_samples`
        if len(block_samples_accepted_inds) > 0:
            new_inds_o = block["o_k"][block_samples_accepted_inds]
            new_dh_k = block["dh_k"][block_samples_accepted_inds]
            new_hh_k = block["hh_k"][block_samples_accepted_inds]
            new_bestfit_lnlike = block["bestfit_lnlike_k"][block_samples_accepted_inds]
            new_inds_i = block["bank_i_inds_k"][block_samples_accepted_inds]
            new_inds_e = block["bank_e_inds_k"][block_samples_accepted_inds]
            # get unnormalized ln_posterior for the new samples

            new_samples_df = pd.DataFrame(
                {
                    "i": new_inds_i,
                    "e": new_inds_e,
                    "o": new_inds_o,
                    "bestfit_lnlike": new_bestfit_lnlike,
                    "d_h_1Mpc": new_dh_k,
                    "h_h_1Mpc": new_hh_k,
                }
            )

            # Concatenate `self.light_samples` and `new_samples_df` in
            # a safe manner.
            if len(self.light_samples) > 0:
                self.light_samples = pd.concat(
                    [
                        self.light_samples.iloc[light_samples_accepted_inds],
                        new_samples_df,
                    ],
                    ignore_index=True,
                )
            else:
                self.light_samples = new_samples_df

        self.light_samples.sort_values(by="bestfit_lnlike", inplace=True)
        self.light_samples.reset_index(drop=True, inplace=True)
        # if light_samples is at maximum allowed size, accept only
        # samples with likelihood higher its minimum likelihood.
        if len(self.light_samples) == self.size_limit:
            self.min_bestfit_lnlike_to_keep = np.max(
                (
                    self.light_samples["bestfit_lnlike"].values.min(),
                    self.min_bestfit_lnlike_to_keep,
                )
            )

    def get_t_grid(self, n=16):
        """return a regular grid with n time points around zero."""
        delta_t = np.diff(self.likelihood.event_data.times[:2])[0]
        t_grid = (np.arange(n) - n // 2) * delta_t

        return t_grid

    def get_psi_grid(self, n=32):
        """return a regular psi grid on (0, pi)"""
        return np.linspace(0, np.pi, n)

    def get_psi_t_grid(self, n_psi=32, n_t=16):
        """Return mesh-grid points of (psi, time), flattened to 1d"""
        t_grid = self.get_t_grid(n_t)
        psi_grid = self.get_psi_grid(n_psi)
        x, y = np.meshgrid(psi_grid, t_grid, indexing="ij")
        psi = x.flatten()  # psi points
        t = y.flatten()  # time points
        return psi, t

    def get_extrinsic_samples(self, n_psi, n_t, t_geocenter, lon, lat):
        """
        Get extrinsic samples with fixed (lon, lat), psi regularly
        sampled on (0, pi) with n_psi points, and t regularly sampled
        with n_t points around t_geocenter, with time resolution set
        by the event data.
        """

        psi, delta_t = self.get_psi_t_grid(n_psi, n_t)
        n_ext = n_psi * n_t
        dummy_extrinsic_samples = pd.DataFrame(
            {
                "lon": lon * np.ones(n_ext),
                "lat": lat * np.ones(n_ext),
                "t_geocenter": t_geocenter + delta_t,
                "psi": psi,
            }
        )
        return dummy_extrinsic_samples

    def get_extrinsic_samples_from_par_dic(self, n_psi, n_t, par_dic):
        """
        Get extrinsic samples with fixed (lon, lat), psi regularly
        sampled on (0, pi) with n_psi points, and t regularly sampled
        with n_t points around t_geocenter, with time resolution set
        by the event data.
        """

        if "lon" in par_dic:
            lon = par_dic["lon"]
        else:
            lon = skyloc_angles.ra_to_lon(par_dic["ra"], self.gmst)
        if "lat" in par_dic:
            lat = par_dic["lat"]
        else:
            lat = par_dic["dec"]
        t_geocenter = par_dic["t_geocenter"]

        return self.get_extrinsic_samples(n_psi, n_t, t_geocenter, lon, lat)

    def get_components(self, dummy_extrinsic_samples):
        """
        Create likelihood-components from dummy extrinsic samples.
        Assume dummy extrinsic samples share sky positions, so the
        rescaling of all responses is the same.

        """
        (response_dpe, timeshifts_dbe) = self.extrinsic_sample_processor.get_components(
            dummy_extrinsic_samples,
            self.likelihood.fbin,
            self.likelihood.event_data.tcoarse,
        )

        response_scale_factor = np.sqrt(np.sum(response_dpe[0, :, 0] ** 2))
        rescaled_response_dpe = response_dpe / response_scale_factor

        return rescaled_response_dpe, timeshifts_dbe, response_scale_factor

    def light_samples_to_bestfit_sample(
        self, extrinsic_samples, intrinsic_samples, response_scale=None
    ):
        light_samples = self.light_samples.copy()
        n_phi = self.evidence.n_phi
        dic = light_samples.iloc[light_samples.bestfit_lnlike.idxmax()].to_dict()
        i, e, o = [int(dic.get(x)) for x in ["i", "e", "o"]]

        bestfit_par_dic = extrinsic_samples.iloc[e].to_dict()
        bestfit_par_dic |= intrinsic_samples.iloc[i].to_dict()
        bestfit_par_dic |= {"phi_ref": np.linspace(0, np.pi * 2, n_phi)[o]}
        bestfit_par_dic["ra"] = skyloc_angles.lon_to_ra(
            bestfit_par_dic["lon"], self.gmst
        )
        bestfit_par_dic["dec"] = bestfit_par_dic["lat"]

        if not response_scale:
            response = self.extrinsic_sample_processor.compute_detector_responses(
                self.extrinsic_sample_processor.detector_names,
                lon=bestfit_par_dic["lon"],
                lat=bestfit_par_dic["lat"],
                psi=0,
            )
            response_scale = np.sum(response**2) ** (1 / 2)

        # under the assumption that the original response had norm 1
        dic["h_h_1Mpc"] *= response_scale**2
        dic["d_h_1Mpc"] *= response_scale

        bestfit_par_dic["d_luminosity"] = dic["h_h_1Mpc"] / dic["d_h_1Mpc"]

        dt_linfree, dphi_linfree = (
            self.intrinsic_sample_processor.load_linfree_dt_and_dphi(
                self.waveform_dir,
                [
                    i,
                ],
            )
        )
        dt_linfree, dphi_linfree = dt_linfree[0], dphi_linfree[0]

        bestfit_par_dic["t_geocenter"] = bestfit_par_dic["t_geocenter"] + dt_linfree

        bestfit_par_dic["phi_ref"] = bestfit_par_dic["phi_ref"] - dphi_linfree

        return bestfit_par_dic

    @classmethod
    def get_response_over_distance_and_lnlike(
        cls,
        dh_weights_dmpb,
        hh_weights_dmppb,
        h_impb,
        timeshift_dbt,
        asd_drift_d,
        n_phi,
        m_arr,
    ):
        """
        Returns
        -------
        r_iotp : numpy.ndarray
            detector_response / distance per intrinsic sample (i),
            orbital phase (o), timeshift (t), and polarization (p).

        lnlike_iot : numpy.ndarray
            Likelihood using specified intrinsic sample (i), orbital
            phase (o), timeshift (t).
        """
        # shapes
        i, m, p, b = h_impb.shape
        t = timeshift_dbt.shape[-1]

        # temporary shapes, applied to improve runtime
        x = i * m * p
        y = i * t * p
        z = i * t * n_phi

        phi_grid_o = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)  # o
        m_inds, mprime_inds = zip(*itertools.combinations_with_replacement(range(m), 2))
        dh_phasor_mo = np.exp(-1j * np.outer(m_arr, phi_grid_o))  # mo
        hh_phasor_Mo = np.exp(
            1j * np.outer(m_arr[m_inds,] - m_arr[mprime_inds,], phi_grid_o)
        )  # mo

        #########
        # <d|h>
        #########
        # complex inner product per i, mode, polarization, timeshift
        h_impb_conj = h_impb.conj()
        dh_impb = dh_weights_dmpb[0] * asd_drift_d[0] ** -2 * h_impb_conj
        dh_xb = dh_impb.reshape(x, b)
        timeshift_conj_bt = timeshift_dbt[0].conj()
        dh_xt = dh_xb @ timeshift_conj_bt
        dh_impt = dh_xt.reshape(i, m, p, t)
        # apply orbital phase, sum over modes
        dh_itpm = np.moveaxis(dh_impt, (1, 3), (3, 1))
        dh_ym = dh_itpm.reshape(y, m)
        dh_yo = dh_ym @ dh_phasor_mo
        dh_itpo = dh_yo.real.reshape(i, t, p, n_phi)
        dh_iotp = np.moveaxis(dh_itpo, 3, 1)

        #########
        # <h|h>
        #########
        # complex inner proct per i, mode-pair, and polarization-pair
        hh_weights_drift_Mppb = hh_weights_dmppb[0] * asd_drift_d[0] ** -2
        h_iMpb = h_impb[:, m_inds]
        h_iMpb_conj = h_impb_conj[:, mprime_inds]
        hh_iMpp = np.einsum(
            "MpPb, iMpb, iMPb -> iMpP",
            hh_weights_drift_Mppb,
            h_iMpb,
            h_iMpb_conj,
            optimize=True,
        )

        # apply orbital phase, and sum over modes
        hh_ippM = np.moveaxis(hh_iMpp, (2, 3, 1), (1, 2, 3))
        hh_ippo = hh_ippM @ hh_phasor_Mo
        hh_iopp = np.moveaxis(hh_ippo.real, (1, 2, 3), (2, 3, 1))

        # inverse the matrix
        hh_det_iopp_reciptocal = 1 / (
            hh_iopp[..., 0, 0] * hh_iopp[..., 1, 1]
            - hh_iopp[..., 0, 1] * hh_iopp[..., 1, 0]
        )

        hh_inv_iopp = np.empty_like(hh_iopp)
        hh_inv_iopp[..., 0, 0] = hh_iopp[..., 1, 1] * hh_det_iopp_reciptocal
        hh_inv_iopp[..., 0, 1] = -hh_iopp[..., 0, 1] * hh_det_iopp_reciptocal
        hh_inv_iopp[..., 1, 0] = -hh_iopp[..., 1, 0] * hh_det_iopp_reciptocal
        hh_inv_iopp[..., 1, 1] = hh_iopp[..., 0, 0] * hh_det_iopp_reciptocal

        ########################
        # Optimal solutions
        ########################

        # r[j,k] is the optimal response/distance of the j-th (i,o,t)
        # tuple.

        dh_zp = dh_iotp.reshape(z, p)
        hh_inv_zpp = np.reshape(
            hh_inv_iopp[:, :, None, ...].repeat(repeats=t, axis=2), (z, p, p)
        )

        r_zp = np.einsum("zpP, zP -> zp", hh_inv_zpp, dh_zp, optimize=True)
        lnlike_z = 0.5 * np.einsum("zp, zp-> z", r_zp, dh_zp, optimize=True)

        # reshape
        r_iotp = r_zp.reshape((i, n_phi, t, p))
        lnlike_iot = lnlike_z.reshape((i, n_phi, t))

        return r_iotp, lnlike_iot

    def get_response_over_distance_and_lnlike_for_bank_samples(
        self, inds, n_t, n_phi, dt_fraction=1.0
    ):
        if isinstance(inds, int):  # single index
            inds = [inds]
        dt = (
            self.likelihood.event_data.times[1] - self.likelihood.event_data.times[0]
        ) * dt_fraction

        t_grid = (np.array(n_t) - n_t // 2) * dt
        par_dic_0 = self.likelihood.par_dic_0
        det_name = self.likelihood.event_data.detector_names[0]
        tgps = self.likelihood.event_data.tgps

        lat, lon = skyloc_angles.cart3d_to_latlon(
            skyloc_angles.normalize(gw_utils.DETECTORS[det_name].location)
        )
        # r0 = gw_utils.get_fplus_fcross_0(det_name, lat, lon).squeeze()

        par_dic = self.transform_par_dic_by_sky_poisition(
            det_name, par_dic_0, lon, lat, tgps
        )

        delay = gw_utils.get_geocenter_delays(det_name, par_dic["lat"], par_dic["lon"])[
            0
        ]
        tcoarse = self.likelihood.event_data.tcoarse
        t_grid = (np.arange(n_t) - n_t // 2) * (
            self.likelihood.event_data.times[1] * dt_fraction
        )
        t_grid += par_dic["t_geocenter"] + tcoarse + delay
        timeshifts_dbt = np.exp(
            -2j * np.pi * t_grid[None, None, :] * self.likelihood.fbin[None, :, None]
        )
        r_iotp, lnlike_iot = self.get_response_over_distance_and_lnlike(
            self.dh_weights_dmpb,
            self.hh_weights_dmppb,
            self.h_impb[inds],
            timeshifts_dbt,
            self.likelihood.asd_drift,
            n_phi,
            self.likelihood.waveform_generator.m_arr,
        )
        return r_iotp, lnlike_iot

    def bestfit_response_to_psi_and_d_luminosity(self, r, det_name):
        """
        Use the combined distance and detector response to find
        distance and polarization angle, assuming signal originates
        from just above the detector.

        Parameters
        ----------
        r : numpy.ndarray, [..., 2]
            detector response over d_luminosity. Last dimension is
            polarizations
        det_name : str
            detector name ('H','L', 'V')

        Retures
        -------

        """
        # use lat, lon of maximal response:
        lat, lon = skyloc_angles.cart3d_to_latlon(
            skyloc_angles.normalize(gw_utils.DETECTORS[det_name].location)
        )

        fpfc0 = gw_utils.get_fplus_fcross_0(det_name, lat, lon).squeeze()
        arg_r0 = np.arctan2(fpfc0[1], fpfc0[0])
        # psi0 satisfy fplus = 1, fcross = 0

        # detector response is approximately 1 at lat, lon

        d_luminosity = 1 / np.sqrt(np.sum(r**2, axis=-1))

        arg_r = np.arctan2(r[1], r[0])

        psi = -(arg_r - arg_r0) / 2

        return psi, d_luminosity

    def transform_par_dic_by_sky_poisition(
        self, det_name, par_dic, lon, lat, tgps=None
    ):
        # work in lon,lat convention unless par_dic has ra/dec in it
        new_par_dic = par_dic.copy()
        new_par_dic["lat"] = lat
        new_par_dic["lon"] = lon
        gmst = None
        if "ra" in par_dic or ("lat" not in par_dic and "ra" in par_dic):
            if tgps is None:
                raise ValueError(
                    "`tgps` must be provided when `par_dic` includes 'ra' or lacks 'lat'."
                )
            gmst = GreenwichMeanSiderealTime(tgps)

        if "ra" in par_dic:
            new_par_dic["ra"] = skyloc_angles.lon_to_ra(new_par_dic["lon"], gmst)
            new_par_dic["dec"] = new_par_dic["lat"]

        if "lat" in par_dic:
            old_skyloc = (par_dic["lat"], par_dic["lon"])
        else:
            old_skyloc = (
                par_dic["dec"],
                skyloc_angles.ra_to_lon(par_dic["ra"], gmst),
            )

        timedelay = gw_utils.get_geocenter_delays(det_name, *old_skyloc)[0]
        new_timedelay = gw_utils.get_geocenter_delays(
            det_name, new_par_dic["lat"], new_par_dic["lon"]
        )[0]

        # data must agree:
        # t_geocenter + timedealy = const between sky positions
        new_par_dic["t_geocenter"] = par_dic["t_geocenter"] + timedelay - new_timedelay

        # polarization response must agree
        response = gw_utils.get_fplus_fcross_0(det_name, *old_skyloc).squeeze()
        new_response = gw_utils.get_fplus_fcross_0(
            det_name, new_par_dic["lat"], new_par_dic["lon"]
        ).squeeze()
        new_par_dic["d_luminosity"] = par_dic["d_luminosity"] * np.sqrt(
            np.sum(new_response**2) / np.sum(response**2)
        )

        old_psi0 = (-np.arctan2(response[1], response[0]) / 2) % (np.pi)
        new_psi0 = (-np.arctan2(new_response[1], new_response[0]) / 2) % (np.pi)

        new_par_dic["psi"] = par_dic["psi"] + old_psi0 - new_psi0

        return new_par_dic
