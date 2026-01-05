"""
Sample handling and creation for the free-sampling method.

This module contains classes for processing intrinsic and extrinsic samples,
loading waveform data, and preparing high-dimensional arrays for likelihood
calculations.
"""

import itertools
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.stats.qmc import Halton

from cogwheel.gw_utils import get_fplus_fcross_0, get_geocenter_delays

from . import config
from .likelihood_calculating import (
    create_lal_dict,
    compute_hplus_hcross_safe,
    get_shift,
)
from .utils import safe_logsumexp


class ExtrinsicSampleProcessor:
    """
    Process extrinsic samples and create high-dimensional arrays for likelihood calculations.

    This class handles detector responses, time shifts, and sky location sampling
    for extrinsic parameters in gravitational wave data analysis.
    """

    def __init__(self, detector_names: List[str]) -> None:
        """
        Initialize the extrinsic sample processor.

        Parameters
        ----------
        detector_names
            Detector names (e.g., ['H', 'L', 'V']).
        """
        self.n_polarizations: int = 2
        self.lal_dic = create_lal_dict()
        self.detector_names: List[str] = detector_names

    @staticmethod
    def compute_detector_responses(
        detector_names: List[str],
        lat: Union[float, NDArray[np.float64]],
        lon: Union[float, NDArray[np.float64]],
        psi: Union[float, NDArray[np.float64]],
    ) -> NDArray[np.complex128]:
        """
        Compute detector response at specific latitude, longitude, and polarization angle.

        Parameters
        ----------
        detector_names
            Detector names.
        lat
            Latitude in radians.
        lon
            Longitude in radians.
        psi
            Polarization angle in radians.

        Returns
        -------
        Detector response array with shape (n_detectors, n_samples, 2).
        """
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

    def compute_extrinsic_timeshift(
        self,
        detector_names: List[str],
        extrinsic_samples: pd.DataFrame,
        f: NDArray[np.float64],
    ) -> NDArray[np.complex128]:
        """
        Compute extrinsic time shift for each detector.

        Parameters
        ----------
        detector_names
            Detector names.
        extrinsic_samples
            DataFrame with extrinsic parameters (must contain 'lat', 'lon', 't_geocenter').
        f
            Frequency array.

        Returns
        -------
        Time shift exponentials with shape (n_samples, n_detectors, n_frequencies).
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

    def get_components(
        self,
        extrinsic_samples: pd.DataFrame,
        fbin: NDArray[np.float64],
        tcoarse: float,
    ) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """
        Compute detector responses and extrinsic time shifts for extrinsic samples.

        Parameters
        ----------
        extrinsic_samples
            DataFrame with extrinsic parameters (must contain 'lat', 'lon', 'psi').
        fbin
            Frequency bins.
        tcoarse
            Coarse time shift.

        Returns
        -------
        Tuple of (response_dpe, timeshifts_dbe) arrays.
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
        self, det_name: str, lon_grid_size: int = 2**10, lat_grid_size: int = 2**10
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        float,
    ]:
        """
        Generate points on (x,y) grid which maps to (lon, lat) points.

        Evaluates PDF and CDF for drawing points from detector response pattern.

        Parameters
        ----------
        det_name
            Detector name, 'H', 'L', or 'V'.
        lon_grid_size
            Number of grid points in longitude axis.
        lat_grid_size
            Number of grid points in latitude axis.

        Returns
        -------
        Tuple of (x, y, cdf_x, pdf_y_given_x, normalization).
            - x: Grid points mapped to longitude.
            - y: Grid points mapped to latitude.
            - cdf_x: CDF evaluated at x grid points.
            - pdf_y_given_x: PDF of y given x, shape (lon_grid_size, lat_grid_size).
            - normalization: Normalization constant.
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

    def uniform_samples_to_lon_lat_psi(
        self,
        u: NDArray[np.float64],
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        cdf_x: NDArray[np.float64],
        pdf_y_given_x: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Map uniform random numbers to longitude, latitude, and polarization angle.

        Parameters
        ----------
        u
            Uniform random numbers, shape (3, n_samples) from (0,1) interval.
        x
            Grid points on (0,1) interval mapped to longitude.
        y
            Grid points on (0,1) interval mapped to latitude.
        cdf_x
            Cumulative distribution of x.
        pdf_y_given_x
            PDF of y given x.

        Returns
        -------
        Tuple of (lon, lat, psi) arrays, each with shape (n_samples,).
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
        self,
        det_name: str,
        n_samples: int = 512,
        lon_grid_size: int = 2**13,
        lat_grid_size: int = 2**13,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.complex128],
    ]:
        """
        Draw random samples from the detector response pattern.

        Sky position (lon, lat) are drawn from the cubed detector response magnitude,
        (Fp^2 + Fc^2)^(3/2). Polarization angle is drawn uniformly on (0, 2π).
        Time is not explicitly drawn, but u_t (uniform on (0,1)) are provided.
        Given a CDF (c) defined on time array (t), samples can be drawn using
        t_samples = t[np.searchsorted(u_t, c)].

        Parameters
        ----------
        det_name
            Detector name, 'H', 'L', or 'V'.
        n_samples
            Number of samples to draw.
        lon_grid_size
            Number of grid points on the PDF/CDF of longitude (x).
        lat_grid_size
            Number of grid points on the PDF/CDF of latitude (y).

        Returns
        -------
        Tuple of (lon, lat, psi, u_t, detector_response).
            - lon, lat, psi: Samples of longitude, latitude, and polarization angle.
            - u_t: Random samples on (0,1) to be mapped to time samples.
            - detector_response: Detector response evaluated on lon, lat, psi.
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
    Load and process intrinsic samples and create high-dimensional arrays for likelihood calculations.

    This class handles loading waveform banks, processing amplitudes and phases,
    and computing relative-binning weights.
    """

    def __init__(
        self,
        like,
        waveform_dir: Optional[Union[str, Path]] = None,
        use_cached_dt: bool = True,
        update_cached_dt: bool = True,
    ) -> None:
        """
        Initialize the intrinsic sample processor.

        Parameters
        ----------
        like
            Relative-binning likelihood object. Serves as a placeholder for event_data,
            waveform_generator, distance marginalization lookup-table, and is used to
            calculate relative-binning weights.
        waveform_dir
            Directory containing waveform data.
        use_cached_dt
            Whether to use cached time shifts.
        update_cached_dt
            Whether to update cached time shifts.
        """

        self.n_polarizations: int = 2
        self.likelihood = like
        self.m_arr: NDArray[np.float64] = like.waveform_generator.m_arr
        self.m_inds, self.mprime_inds = zip(
            *itertools.combinations_with_replacement(range(len(self.m_arr)), 2)
        )
        self.lal_dic = create_lal_dict()
        self.waveform_dir: Optional[Path] = Path(waveform_dir) if waveform_dir else None
        self.cached_dt_linfree_relative: dict[int, float] = {}
        self.use_cached_dt: bool = use_cached_dt
        self.update_cached_dt: bool = update_cached_dt

    @property
    def n_intrinsic(self) -> int:
        """
        Number of intrinsic samples.

        Returns
        -------
        Number of intrinsic samples, or 0 if not set.
        """
        return getattr(getattr(self, "intrinsic_samples", None), "__len__", 0)

    @property
    def n_fbin(self) -> int:
        """
        Number of relative-binning frequency bins.

        Returns
        -------
        Number of frequency bins.
        """
        return len(self.likelihood.fbin)

    @property
    def n_modes(self) -> int:
        """
        Number of harmonic modes.

        Returns
        -------
        Number of harmonic modes.
        """
        return len(self.likelihood.waveform_generator._harmonic_modes_by_m.values())

    def cache_dt_linfree_relative(
        self, inds: NDArray[np.int_], dts: NDArray[np.float64]
    ) -> None:
        """
        Store relative-linear-free time shifts.

        Parameters
        ----------
        inds
            Sample indices.
        dts
            Time shifts to cache.
        """
        is_not_cached = np.logical_not(
            np.isin(
                inds,
                np.fromiter(self.cached_dt_linfree_relative.keys(), dtype=int),
            )
        )

        for i, dt in zip(inds[is_not_cached], dts[is_not_cached]):
            self.cached_dt_linfree_relative[int(i)] = float(dt)

    @staticmethod
    def load_bank(
        sample_bank_path: Union[str, Path],
        indices: Optional[Union[List[int], NDArray[np.int_], range]] = None,
        renormalize_log_prior_weights: bool = False,
    ) -> pd.DataFrame:
        """
        Load intrinsic samples from a file.

        Parameters
        ----------
        sample_bank_path
            Path to the intrinsic sample bank file.
        indices
            Indices of samples to load. If None, all samples are loaded.
        renormalize_log_prior_weights
            Whether to renormalize log prior weights.

        Returns
        -------
        DataFrame containing intrinsic samples.
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
        if renormalize_log_prior_weights:
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
    def _load_waveforms(
        waveform_dir: Union[str, Path], indices: NDArray[np.int_]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Load amplitude and phases from directory.

        Phases are in the linear-free convention. Per mode, the linear-free and standard
        phases are related by:
        Phase(m,f) [linear free] = (Phase(m,f) [standard] - 2*pi*dt_linear_free*f - m*dphi_linear_free)
        t (standard) = t (linear free) - dt_linear_free
        phi (standard) = phi (linear free) + dphi_linear_free

        Parameters
        ----------
        waveform_dir
            Directory containing waveform data.
        indices
            Indices of waveforms to load.

        Returns
        -------
        Tuple of (amplitudes, phases) arrays.
        """

        amplitudes, phases = IntrinsicSampleProcessor._load_amp_and_phase(
            waveform_dir, indices
        )

        return amplitudes, phases

    @staticmethod
    def _load_amp_and_phase(
        waveform_dir: Union[str, Path], indices: NDArray[np.int_]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Load amplitudes and phases from directory.

        Waveforms are represented as h = amplitudes * np.exp(1j * phases).
        All arrays have shape (n_inds, n_modes, n_pol, n_fbin).

        Parameters
        ----------
        waveform_dir
            Directory containing waveform data.
        indices
            Indices of waveforms to load.

        Returns
        -------
        Tuple of (amplitudes, phases) arrays, each with shape (n_inds, n_modes, n_pol, n_fbin).
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

    def load_amp_and_phase(
        self, waveform_dir: Union[str, Path], indices: NDArray[np.int_]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Load relative-binning waveforms with time shift correction relative to reference waveform.

        Parameters
        ----------
        waveform_dir
            Directory containing waveform data.
        indices
            Indices of waveforms to load.

        Returns
        -------
        Tuple of (amplitudes, phases) arrays.
        """

        amplitudes, phases = self._load_amp_and_phase(waveform_dir, indices)
        relative_timeshifts = np.zeros(len(indices))
        if self.use_cached_dt:
            is_cached = np.isin(
                indices, np.fromiter(self.cached_dt_linfree_relative, dtype=int)
            )
        else:
            is_cached = np.zeros_like(indices, dtype=bool)
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
        if self.update_cached_dt:
            self.cache_dt_linfree_relative(
                indices[~is_cached], relative_timeshifts[~is_cached]
            )

        return amplitudes, phases

    def get_relative_linfree_dt_from_waveform(
        self,
        amp_impb: NDArray[np.float64],
        phase_impb: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Find the relative-binning time shifts and apply them to the phase.

        Parameters
        ----------
        amp_impb
            Amplitudes with shape (n_inds, n_modes, n_pol, n_fbin).
        phase_impb
            Phases with shape (n_inds, n_modes, n_pol, n_fbin).

        Returns
        -------
        Tuple of (phase_impb_updated, relative_timeshift_i).
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

    def load_linfree_dt_and_dphi(
        self,
        waveform_dir: Union[str, Path],
        indices: Union[int, List[int], NDArray[np.int_]],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Load linear-free dt and dphi values.

        Since banks no longer store these values, we return zeros.
        The relative time shifts are computed on-the-fly when needed.

        Parameters
        ----------
        waveform_dir
            Directory containing waveform data.
        indices
            Indices of waveforms to process.

        Returns
        -------
        Tuple of (dt_linfree, dphi_linfree) arrays.
        """
        if isinstance(indices, int):
            indices = [
                indices,
            ]
        if isinstance(indices, list):
            indices = np.array(indices)

        # Return zeros since banks no longer store dt/dphi linfree
        dt_linfree = np.zeros(len(indices), dtype=float)
        dphi_linfree = np.zeros(len(indices), dtype=float)

        # add the relative timeshifts
        dt_linfree_relative = np.zeros_like(dt_linfree)
        is_not_cached = np.logical_not(
            np.isin(
                indices,
                np.fromiter(self.cached_dt_linfree_relative, dtype=int),
            )
        )

        if is_not_cached.any():
            # by loading the amplitude and phase, we fill the cached
            # linear free dt dictionary.
            _ = self.load_amp_and_phase(waveform_dir, indices[is_not_cached])

        dt_linfree_relative = np.array(
            [self.cached_dt_linfree_relative[i] for i in indices]
        )

        return dt_linfree - dt_linfree_relative, dphi_linfree

    def load_waveforms(
        self, waveform_dir: Union[str, Path], indices: NDArray[np.int_]
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """
        Load waveforms with time shift correction relative to reference waveform.

        Parameters
        ----------
        waveform_dir
            Directory containing waveform data.
        indices
            Indices of waveforms to load.

        Returns
        -------
        Tuple of (amplitudes, phases, dt_linfree, dphi_linfree).
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

    def get_hplus_hcross_0(
        self,
        par_dic: dict,
        f: Optional[NDArray[np.float64]] = None,
        force_fslice: bool = False,
        fslice: Optional[slice] = None,
    ) -> NDArray[np.complex128]:
        """
        Create (n_modes x 2 polarizations x n_frequencies) array.

        Uses d_luminosity = 1Mpc and phi_ref = 0, shifted to center for event_data.times,
        without t_refdet shifts or linear-free time shifts.

        Parameters
        ----------
        par_dic
            Parameter dictionary.
        f
            Frequency array. If None, uses self.likelihood.fbin.
        force_fslice
            Whether to force frequency slice.
        fslice
            Frequency slice to use.

        Returns
        -------
        Waveform array with shape (n_modes, 2, n_frequencies).
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

    def get_summary(
        self,
    ) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """
        Get the relative-binning summary statistics for likelihood calculations.

        The weights are calculated per detector (d), harmonic mode (m), polarization (p),
        and frequency bin (b). For <h,h>, the weights are calculated per mode pair (m and m')
        and polarization pair (p and p'). For the modes we use a single index (m) to represent
        all unique combinations of modes, and for the polarizations we calculate the 4 p-p
        combinations.

        Returns
        -------
        Tuple of (dh_weights_dmpb, hh_weights_dmppb).
            - dh_weights_dmpb: Weights for integrand data * h.conj(), shape (d, m, p, b).
            - hh_weights_dmppb: Weights for integrand h*h.conj(), shape (d, m, m', p, p', b).
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
