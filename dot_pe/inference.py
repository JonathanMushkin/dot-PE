"""
Code to run the Sampler Free inference run for a given event and
bank folder. It is performed in two steps:
1. Perform single-detector likelihood evaluations and get the indices of
   the intrinsic samples that pass the threshold.
   Alternatively, load the indices from a file.
2. Perform a full coherent multi-detector likelihood evaluation, and get
   samples with probabilistic weigths.


# TODO:
1. save run-summary, corner plot and bestfit waveform on-data as pdfs,
   at the end of the run.
"""

import argparse
import cProfile
import json
import os
import pstats
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from lal import GreenwichMeanSiderealTime
from numpy.typing import NDArray
from tqdm import tqdm

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

from cogwheel import skyloc_angles  # noqa: E402
from cogwheel.data import EventData  # noqa: E402
from cogwheel.gw_utils import DETECTORS, get_geocenter_delays  # noqa: E402
from cogwheel.likelihood import RelativeBinningLikelihood, LookupTable  # noqa: E402
from cogwheel.posterior import Posterior  # noqa: E402
from cogwheel.utils import exp_normalize, get_rundir, mkdirs, read_json  # noqa: E402
from cogwheel.waveform import WaveformGenerator  # noqa: E402
from cogwheel.prior import Prior  # noqa: E402
from .base_sampler_free_sampling import (  # noqa: E402
    get_n_effective_total_i_e,
)
from .likelihood_calculating import LinearFree  # noqa: E402
from .marginalization import MarginalizationExtrinsicSamplerFreeLikelihood  # noqa: E402
from .coherent_processing import (  # noqa: E402
    CoherentLikelihoodProcessor,
    CoherentExtrinsicSamplesGenerator,
)
from .utils import (  # noqa: E402
    extract_single_detector_event_data,
    get_event_data,
    inds_to_blocks,
    parse_bank_folders,
    safe_logsumexp,
    validate_bank_configs,
)
from .single_detector import SingleDetectorProcessor  # noqa: E402
from .sample_processing import (  # noqa: E402
    ExtrinsicSampleProcessor,
    IntrinsicSampleProcessor,
)


def _create_single_detector_processor(
    event_data: Union[str, Path, EventData],
    det_name: str,
    par_dic_0: Dict,
    bank_folder: Union[str, Path],
    fbin: NDArray[np.float64],
    approximant: str,
    n_phi: int,
    m_arr: NDArray[np.int64],
    single_detector_blocksize: int,
    size_limit: int,
) -> SingleDetectorProcessor:
    """Create a SingleDetectorProcessor for a given detector."""
    event_data_1d = extract_single_detector_event_data(event_data, det_name)
    wfg = WaveformGenerator.from_event_data(event_data_1d, approximant)
    likelihood_linfree = LinearFree(event_data_1d, wfg, par_dic_0, fbin)
    bank_file_path = Path(bank_folder) / "intrinsic_sample_bank.feather"
    waveform_dir = Path(bank_folder) / "waveforms"

    return SingleDetectorProcessor(
        bank_file_path,
        waveform_dir,
        n_phi,
        m_arr,
        likelihood_linfree,
        size_limit=size_limit,
        ext_block_size=single_detector_blocksize,
        int_block_size=single_detector_blocksize,
    )


def run_for_single_detector(
    event_data: Union[str, Path, EventData],
    det_name: str,
    par_dic_0: Dict,
    bank_folder: Union[str, Path],
    inds: Union[int, List[int], NDArray[np.int64]],
    fbin: NDArray[np.float64],
    h_impb: NDArray[np.complex128] = None,
    approximant: str = "IMRPhenomXODE",
    n_phi: int = 100,
    single_detector_blocksize: int = 512,
    m_arr: NDArray[np.int64] = np.array([2, 1, 3, 4]),
    n_t: int = 128,
    size_limit: int = 10**7,
    sdp: Optional[SingleDetectorProcessor] = None,
) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.complex128]]]:
    """
    Perform single detector likelihood evaluations and return the likelihoods.
    """
    if isinstance(inds, int):
        inds = np.arange(inds)
    if isinstance(inds, list):
        inds = np.array(inds)

    if sdp is None:
        sdp = _create_single_detector_processor(
            event_data,
            det_name,
            par_dic_0,
            bank_folder,
            fbin,
            approximant,
            n_phi,
            m_arr,
            single_detector_blocksize,
            size_limit,
        )

    waveform_dir = Path(bank_folder) / "waveforms"

    if h_impb is None:
        return_h_impb = True
        amp, phase = sdp.intrinsic_sample_processor.load_amp_and_phase(
            waveform_dir, inds
        )
        h_impb = amp * np.exp(1j * phase)
    else:
        return_h_impb = False

    sdp.h_impb = h_impb

    # fix sky position to above detector
    det_name = sdp.likelihood.event_data.detector_names[0]
    tgps = sdp.likelihood.event_data.tgps
    lat, lon = skyloc_angles.cart3d_to_latlon(
        skyloc_angles.normalize(DETECTORS[det_name].location)
    )

    par_dic = sdp.transform_par_dic_by_sky_poisition(
        det_name, par_dic_0, lon, lat, tgps
    )

    delay = get_geocenter_delays(det_name, par_dic["lat"], par_dic["lon"])[0]
    tcoarse = sdp.likelihood.event_data.tcoarse
    t_grid = (np.arange(n_t) - n_t // 2) * (sdp.likelihood.event_data.times[1])
    t_grid += par_dic["t_geocenter"] + tcoarse + delay

    timeshifts_dbt = np.exp(
        -2j * np.pi * t_grid[None, None, :] * sdp.likelihood.fbin[None, :, None]
    )

    lnlike_iot = sdp.get_response_over_distance_and_lnlike(
        sdp.dh_weights_dmpb,
        sdp.hh_weights_dmppb,
        sdp.h_impb,
        timeshifts_dbt,
        sdp.likelihood.asd_drift,
        sdp.likelihood_calculator.n_phi,
        sdp.likelihood_calculator.m_arr,
    )[1]

    lnlike_i = lnlike_iot.max(axis=(1, 2))

    if return_h_impb:
        return lnlike_i, h_impb
    return lnlike_i


def collect_int_samples_from_single_detectors(
    event_data: EventData,
    par_dic_0: Dict,
    single_detector_blocksize: int,
    n_int: int,
    n_phi: int,
    n_t: int,
    bank_folder: Union[str, Path],
    i_int_start: int = 0,
    max_incoherent_lnlike_drop: float = 20,
    preselected_indices: Union[NDArray[np.int_], List[int], str, Path, None] = None,
    apply_threshold: bool = True,
) -> Tuple[NDArray[np.int_], NDArray[np.float64], NDArray[np.float64]]:
    """
    Perform n_det independent single-detector likelihood evaluations and
    return the indices of the samples that passed a threshold and their
    corresponding single detector log-likelihood values and their incoherent sum.

    Parameters
    ----------
    preselected_indices : Union[NDArray[np.int_], List[int], str, Path, None], optional
        Pre-filtered absolute bank indices to use instead of generating
        np.arange(i_int_start, i_int_start + n_int). Can be:
        - numpy array or list of integers: direct indices
        - str or Path: path to .npy file containing indices
        - None: use standard range generation (default)
        When provided, n_int and i_int_start are ignored for index generation.
    apply_threshold : bool, optional
        If True (default), apply threshold filtering based on max_incoherent_lnlike_drop.
        If False, return all indices and their likelihoods without filtering.
    """
    bank_folder = Path(bank_folder)
    with open(bank_folder / "bank_config.json", "r", encoding="utf-8") as f:
        bank_config = json.load(f)
        fbin = np.array(bank_config["fbin"])
        approximant = bank_config["approximant"]
        m_arr = np.array(bank_config["m_arr"])

    sdp_by_detector = {}
    for det_name in event_data.detector_names:
        sdp_by_detector[det_name] = _create_single_detector_processor(
            event_data,
            det_name,
            par_dic_0,
            bank_folder,
            fbin,
            approximant,
            n_phi,
            m_arr,
            single_detector_blocksize,
            size_limit=10**7,
        )

    if preselected_indices is not None:
        # Handle different input types for preselected_indices
        if isinstance(preselected_indices, (str, Path)):
            # Load from file
            intrinsic_indices = np.load(preselected_indices)
        elif isinstance(preselected_indices, list):
            # Convert list to numpy array
            intrinsic_indices = np.array(preselected_indices, dtype=np.int_)
        elif isinstance(preselected_indices, np.ndarray):
            # Use numpy array directly
            intrinsic_indices = preselected_indices
        else:
            raise TypeError(
                f"preselected_indices must be array, list, str, Path, or None, got {type(preselected_indices)}"
            )
    else:
        intrinsic_indices = np.arange(i_int_start, i_int_start + n_int)

    lnlike_di = np.zeros((len(event_data.detector_names), len(intrinsic_indices)))
    for batch_start in tqdm(
        range(0, len(intrinsic_indices), single_detector_blocksize),
        desc="Processing intrinsic batches",
        total=-(len(intrinsic_indices) // -single_detector_blocksize),
    ):
        batch_end = min(batch_start + single_detector_blocksize, len(intrinsic_indices))
        batch_intrinsic_indices = intrinsic_indices[batch_start:batch_end]
        h_impb = None  # Reset h_impb for each new batch
        for d, det_name in enumerate(event_data.detector_names):
            temp = run_for_single_detector(
                event_data,
                det_name,
                par_dic_0,
                bank_folder,
                batch_intrinsic_indices,
                fbin,
                h_impb,
                approximant,
                n_phi,
                single_detector_blocksize,
                m_arr,
                n_t,
                size_limit=10**7,
                sdp=sdp_by_detector[det_name],
            )
            if h_impb is None:
                lnlike_di[d, batch_start:batch_end] = temp[0]
                h_impb = temp[1]
            else:
                lnlike_di[d, batch_start:batch_end] = temp

    incoherent_lnlikes = np.sum(lnlike_di, axis=0)

    if apply_threshold:
        incoherent_threshold = incoherent_lnlikes.max() - max_incoherent_lnlike_drop
        selected = incoherent_lnlikes >= incoherent_threshold
        inds = intrinsic_indices[selected]
        lnlike_di = lnlike_di[:, selected]
        incoherent_lnlikes = incoherent_lnlikes[selected]
    else:
        inds = intrinsic_indices
    return inds, lnlike_di, incoherent_lnlikes


def run_coherent_inference(
    event_data: EventData,
    bank_rundir: Path,
    top_rundir: Path,
    par_dic_0: Dict,
    bank_folder: Union[str, Path],
    n_total_samples: int,
    inds: NDArray[np.int_],
    n_ext: int,
    n_phi: int,
    m_arr: NDArray[np.int_],
    blocksize: int,
    renormalize_log_prior_weights_i: bool = False,
    intrinsic_logw_lookup=None,
    size_limit: int = 10**7,
    max_bestfit_lnlike_diff: float = 20,
) -> Tuple[float, float, float, float, float, int]:
    """
    Perform the heavy computation phase of coherent inference.
    This function stops after saving prob_samples.feather.

    Parameters
    ----------
    bank_rundir : Path
        Directory for bank-specific outputs (CLP, prob_samples, cache).
    top_rundir : Path
        Top-level rundir where shared extrinsic samples are stored.

    Returns
    -------
    ln_evidence : float
        The log evidence.
    ln_evidence_discarded : float
        The log evidence of discarded samples.
    n_effective : float
        The effective sample size.
    n_effective_i : float
        The effective sample size for intrinsic parameters.
    n_effective_e : float
        The effective sample size for extrinsic parameters.
    n_distance_marginalizations : int
        The number of distance marginalizations performed.
    """
    bank_folder = Path(bank_folder)
    waveform_dir = bank_folder / "waveforms"
    bank_file_path = bank_folder / "intrinsic_sample_bank.feather"
    with open(bank_folder / "bank_config.json", "r", encoding="utf-8") as f:
        bank_config = json.load(f)
        fbin = np.array(bank_config["fbin"])
        approximant = bank_config["approximant"]

    wfg = WaveformGenerator.from_event_data(event_data, approximant)

    likelihood_linfree = LinearFree(event_data, wfg, par_dic_0, fbin)

    clp = CoherentLikelihoodProcessor(
        bank_file_path,
        waveform_dir,
        n_phi,
        m_arr,
        likelihood_linfree,
        size_limit=size_limit,
        max_bestfit_lnlike_diff=max_bestfit_lnlike_diff,
        renormalize_log_prior_weights_i=renormalize_log_prior_weights_i,
        intrinsic_logw_lookup=intrinsic_logw_lookup,
    )

    i_blocks = inds_to_blocks(inds, blocksize)
    e_blocks = inds_to_blocks(np.arange(n_ext), blocksize)
    # Load extrinsic samples data from top-level rundir (shared)
    clp.load_extrinsic_samples_data(top_rundir)
    # Save CLP to bank-specific rundir
    clp.to_json(bank_rundir, overwrite=True)
    # perform the run
    print(f"Creating {len(i_blocks)} x {len(e_blocks)} likelihood blocks...")

    _ = clp.create_likelihood_blocks(
        tempdir=bank_rundir,
        i_blocks=i_blocks,
        e_blocks=e_blocks,
        response_dpe=clp.full_response_dpe,
        timeshift_dbe=clp.full_timeshift_dbe,
    )

    clp.prob_samples["weights"] = exp_normalize(clp.prob_samples["ln_posterior"].values)
    clp.prob_samples.to_feather(bank_rundir / "prob_samples.feather")

    # Save the IntrinsicSampleProcessor cache for post-processing
    cache_path = bank_rundir / "intrinsic_sample_processor_cache.json"
    cache_dict = {
        int(k): float(v)
        for k, v in clp.intrinsic_sample_processor.cached_dt_linfree_relative.items()
    }
    with open(cache_path, "w") as f:
        json.dump(cache_dict, f)

    # Process the results
    # Normalize by total MC samples attempted (before rejection sampling)
    ln_evidence = safe_logsumexp(clp.prob_samples["ln_posterior"].values) - np.log(
        n_total_samples
    )
    ln_evidence_discarded = clp.logsumexp_discarded_ln_posterior - np.log(
        n_total_samples
    )

    n_effective, n_effective_i, n_effective_e = get_n_effective_total_i_e(
        clp.prob_samples, assume_normalized=False
    )

    return (
        ln_evidence,
        ln_evidence_discarded,
        n_effective,
        n_effective_i,
        n_effective_e,
        clp.n_distance_marginalizations,
    )


def standardize_samples(
    cached_dt_linfree_relative: Union[dict, str, Path, Dict[str, dict]],
    lookup_table: LookupTable,
    pr: Prior,
    prob_samples: pd.DataFrame,
    intrinsic_samples: Union[pd.DataFrame, str, Path, Dict[str, pd.DataFrame]],
    extrinsic_samples: Union[pd.DataFrame, str, Path],
    n_phi: int,
    tgps: float,
) -> pd.DataFrame:
    """
    Standardize the samples and calculate the weights.

    Parameters
    ----------
    cached_dt_linfree_relative : Dict[str, dict]
        Cached relative timeshifts from the compute phase, mapping bank_id to cache dict.
    lookup_table : LookupTable
        LookupTable object, for distance marginalization.
    pr : Prior
        Prior object.
    prob_samples : DataFrame
        Samples with indices columns `i`, `e` and `o`, probabilistic information,
        and `bank_id` column.
    intrinsic_samples : Dict[str, pd.DataFrame]
        Intrinsic samples, mapping bank_id to DataFrame.
    extrinsic_samples : DataFrame or str or Path
        Extrinsic samples. Can be a DataFrame or a path to a feather file.
    n_phi : int
        Number of phi_ref samples.
    tgps : float
        GPS time of event.
    """
    if "bank_id" not in prob_samples.columns:
        raise ValueError("prob_samples must have 'bank_id' column")

    if not isinstance(intrinsic_samples, dict) or not isinstance(
        cached_dt_linfree_relative, dict
    ):
        raise ValueError(
            "intrinsic_samples and cached_dt_linfree_relative "
            "must be dicts mapping bank_id to values"
        )

    # Load extrinsic samples from file if needed
    if isinstance(extrinsic_samples, (str, Path)):
        extrinsic_samples = pd.read_feather(extrinsic_samples)

    # Process each bank group separately
    combined_samples_list = []
    for bank_id, group in prob_samples.groupby("bank_id"):
        # Get intrinsic samples for this bank
        bank_intrinsic = intrinsic_samples[bank_id]
        bank_cache = cached_dt_linfree_relative[bank_id]

        # Combine intrinsic and extrinsic samples for this group
        group_combined = pd.concat(
            [
                bank_intrinsic.iloc[group["i"].values].reset_index(drop=True),
                extrinsic_samples.iloc[group["e"].values].reset_index(drop=True),
            ],
            axis=1,
        )

        # Drop unwanted columns
        group_combined.drop(
            columns=["weights", "log_prior_weights", "original_index"],
            inplace=True,
            errors="ignore",  # Only drop columns that exist
        )

        # Add prob_samples columns
        group_combined = pd.concat(
            [group_combined, group.reset_index(drop=True)], axis=1
        )

        # Add phi column
        group_combined["phi"] = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)[
            group_combined["o"].values
        ]

        # Apply linear free timeshift renaming
        group_combined.rename(
            columns={
                "t_geocenter": "t_geocenter_linfree",
                "phi": "phi_ref_linfree",
            },
            inplace=True,
        )

        # Load timeshifts for this bank
        unique_i_bank = np.unique(group["i"].values)
        u_i_bank = np.searchsorted(unique_i_bank, group["i"].values)

        dt_linfree_u_bank = np.array([bank_cache[i] for i in unique_i_bank])
        dphi_linfree_u_bank = np.zeros_like(dt_linfree_u_bank)

        group_combined["t_geocenter"] = (
            group_combined["t_geocenter_linfree"] - dt_linfree_u_bank[u_i_bank]
        )

        group_combined["phi_ref"] = (
            group_combined["phi_ref_linfree"] - dphi_linfree_u_bank[u_i_bank]
        )

        combined_samples_list.append(group_combined)

    # Concatenate all banks in original order
    combined_samples = pd.concat(combined_samples_list, ignore_index=True)
    # Reorder to match original prob_samples order
    combined_samples = combined_samples.reindex(prob_samples.index)
    combined_samples.reset_index(drop=True, inplace=True)

    combined_samples["ra"] = skyloc_angles.lon_to_ra(
        combined_samples["lon"], GreenwichMeanSiderealTime(tgps)
    )
    combined_samples["dec"] = combined_samples["lat"]
    n_cores = int(os.environ.get("LSB_DJOB_NUMPROC", 1))
    if (n_cores > 1) and False:
        combined_samples["d_luminosity"] = sample_distance_multiprocess(
            n_cores,
            lookup_table,
            combined_samples["d_h_1Mpc"].values,
            combined_samples["h_h_1Mpc"].values,
        )
    else:
        combined_samples["d_luminosity"] = np.vectorize(
            lambda x, y: lookup_table.sample_distance(x, y, resolution=32),
            otypes=[np.float64],
        )(
            combined_samples["d_h_1Mpc"].values,
            combined_samples["h_h_1Mpc"].values,
        )

    combined_samples["bestfit_d_luminosity"] = (
        combined_samples["h_h_1Mpc"] / combined_samples["d_h_1Mpc"]
    )
    combined_samples["lnl"] = (
        combined_samples["d_h_1Mpc"] / combined_samples["d_luminosity"]
        - 0.5 * combined_samples["h_h_1Mpc"] / combined_samples["d_luminosity"] ** 2
    )
    # Normalize weights (use existing weights column if present, else recompute)
    if "weights" not in combined_samples.columns:
        combined_samples["weights"] = exp_normalize(
            combined_samples["ln_posterior"].values
        )

    pr.inverse_transform_samples(combined_samples)

    return combined_samples


def _draw_distance(args):
    x, y, lookup_table = args
    return lookup_table.sample_distance(x, y, resolution=32)


def sample_distance_multiprocess(num_cores, lookup_table, dh, hh):
    """Sample distance values in parallel using multiprocessing."""
    import multiprocessing

    # Create argument tuples including lookup_table
    args = ((x, y, lookup_table) for x, y in zip(dh, hh))
    # Use multiprocessing Pool to parallelize the computation
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(_draw_distance, args)
    return np.array(results)


def postprocess(
    event_data: EventData,
    rundir: Path,
    bank_folder: Union[
        str, Path, Dict[str, Path], List[Union[str, Path]], Tuple[Union[str, Path], ...]
    ],
    n_phi: int,
    pr: Prior,
    prob_samples: Union[pd.DataFrame, Path, str] = None,
    n_draws: int = None,
    max_n_draws: int = 10**4,
    draw_subset: bool = False,
    lookup_table=None,
) -> pd.DataFrame:
    """
    Perform the post-processing phase of coherent inference.
    This function starts after prob_samples.feather has been saved.

    Parameters
    ----------
    event_data : EventData
        Event data.
    rundir : Path
        Directory containing the results from the compute phase.
    bank_folder : Union[str, Path, Dict[str, Path], List, Tuple]
        Path(s) to the bank folder(s). Can be a single path, dict mapping bank_id to path,
        or list/tuple of paths. Will be normalized to a dict internally.
    n_phi : int
        Number of phi_ref samples.
    pr : Prior
        Prior object.
    prob_samples : Union[pd.DataFrame, Path, str], optional
        Probabilistic samples. If None, loads from rundir / "prob_samples.feather".
        If Path or str, loads from that path. If DataFrame, uses directly.
    n_draws : int, optional
        Number of samples to draw if draw_subset is True.
    max_n_draws : int, optional
        Maximum number of samples to draw.
    draw_subset : bool, optional
        Whether to draw a subset of samples.
    lookup_table : optional
        Lookup table for distance marginalization. If None, creates new object.
        Otherwise uses the provided table.

    Returns
    -------
    samples : pd.DataFrame
        The standardized samples.
    """
    # Normalize bank_folder to dict format using parse_bank_folders
    # This handles str, Path, List, Tuple, or already-dict inputs
    if isinstance(bank_folder, dict):
        # Already a dict, use as-is
        banks = bank_folder
    else:
        # Use parse_bank_folders to normalize to dict format
        banks = parse_bank_folders(bank_folder)

    # Load prob_samples if needed
    if prob_samples is None:
        prob_samples = pd.read_feather(rundir / "prob_samples.feather")
    elif isinstance(prob_samples, (str, Path)):
        prob_samples = pd.read_feather(prob_samples)
    # If prob_samples is already a DataFrame, use it directly

    # Check if prob_samples has bank_id column (indicates multiple banks)
    is_multi_bank = "bank_id" in prob_samples.columns
    if not is_multi_bank:
        # If bank_id not present, add it based on banks dict
        prob_samples["bank_id"] = list(banks.keys())[0]
        is_multi_bank = True

    # Load extrinsic samples
    extrinsic_samples = pd.read_feather(rundir / "extrinsic_samples.feather")

    if draw_subset:
        # Calculate effective sample size for subsetting
        n_effective, _, _ = get_n_effective_total_i_e(
            prob_samples, assume_normalized=False
        )

        n_draws = n_draws if n_draws else int(max(n_effective // 2, 1))
        if max_n_draws is not None:
            n_draws = min(n_draws, max_n_draws)
        if n_draws > n_effective:
            warnings.warn(
                f"n_draws ({n_draws}) is bigger than n_effective ({n_effective})."
            )

        prob_samples = prob_samples.sample(
            n=n_draws, weights="weights", replace=True
        ).reset_index(drop=True)
        prob_samples["weights"] = 1.0

    print("Standardizing samples...")
    stnd_start_time = time.time()

    # Handle lookup table
    if lookup_table is None:
        # Create new lookup table like IntrinsicSampleProcessor does
        lookup_table = LookupTable()
    # Otherwise, use the provided lookup_table directly

    # Load intrinsic samples per bank
    intrinsic_samples_by_bank = {}
    for bank_id, bank_path in banks.items():
        bank_file_path = Path(bank_path) / "intrinsic_sample_bank.feather"
        intrinsic_samples_by_bank[bank_id] = IntrinsicSampleProcessor.load_bank(
            bank_file_path
        )

    # Load cached timeshifts per bank
    cached_dt_by_bank = {}
    unique_bank_ids = prob_samples["bank_id"].unique()
    for bank_id in unique_bank_ids:
        # Load cache file from banks/<bank_id>/ subdirectory, or rundir if not found there
        cache_path_multi = (
            rundir / "banks" / bank_id / "intrinsic_sample_processor_cache.json"
        )
        cache_path_single = rundir / "intrinsic_sample_processor_cache.json"
        if cache_path_multi.exists():
            cache_path = cache_path_multi
        elif cache_path_single.exists():
            cache_path = cache_path_single
        else:
            warnings.warn(f"Cache file not found for bank {bank_id}, using empty cache")
            cached_dt_by_bank[bank_id] = {}
            continue

        with open(cache_path, "r") as f:
            cached_dt_by_bank[bank_id] = {int(k): v for k, v in json.load(f).items()}

    samples = standardize_samples(
        cached_dt_by_bank,
        lookup_table,
        pr,
        prob_samples,
        intrinsic_samples_by_bank,
        extrinsic_samples,
        n_phi,
        event_data.tgps,
    )

    if draw_subset:
        samples["weights"] = 1.0
    print(
        "Standardizing samples done in "
        + f"{(time.time() - stnd_start_time):.3g} seconds."
    )

    return samples


def prepare_run_objects(
    event: Union[str, Path, EventData],
    bank_folder: Union[str, Path, List[Union[str, Path]], Tuple[Union[str, Path], ...]],
    n_int: Union[int, List[int], Dict[str, int], None],
    n_ext: int,
    n_phi: int,
    n_t: int,
    blocksize: int,
    single_detector_blocksize: int,
    i_int_start: int,
    seed: Optional[int],
    load_inds: bool,
    inds_path: Union[Path, str, Dict[str, Union[Path, str]], None],
    size_limit: int,
    draw_subset: bool,
    n_draws: Optional[int],
    event_dir: Union[str, Path, None],
    rundir: Union[str, Path, None],
    coherent_score_min_n_effective_prior: int,
    max_incoherent_lnlike_drop: float,
    max_bestfit_lnlike_diff: float,
    mchirp_guess: Optional[float],
    extrinsic_samples: Union[str, Path, None],
    n_phi_incoherent: Optional[int],
    preselected_indices: Union[
        NDArray[np.int_],
        List[int],
        str,
        Path,
        Dict[str, Union[NDArray[np.int_], List[int], str, Path]],
        None,
    ],
    bank_logw_override: Union[
        Dict[str, Union[NDArray[np.float64], List[float], pd.Series]],
        NDArray[np.float64],
        List[float],
        pd.Series,
        None,
    ],
    coherent_posterior_kwargs: Dict,
) -> Dict[str, Any]:
    """Prepare shared objects for inference run."""
    print("Setting paths & loading configurations...")

    banks = parse_bank_folders(bank_folder)

    bank_paths = list(banks.values())
    bank_config = validate_bank_configs(bank_paths)

    if event_dir is not None and rundir is not None:
        warnings.warn(
            "Both 'event_dir' and 'rundir' are provided. 'event_dir' will be ignored."
        )
    if rundir is None:
        rundir = get_rundir(event_dir)

    if not Path(rundir).exists():
        mkdirs(rundir)

    # Convert bank_folder to JSON-serializable format preserving structure
    if isinstance(bank_folder, (list, tuple)):
        bank_folder_serializable = [str(Path(p)) for p in bank_folder]
    elif isinstance(bank_folder, dict):
        bank_folder_serializable = {k: str(Path(v)) for k, v in bank_folder.items()}
    else:
        bank_folder_serializable = str(Path(bank_folder))

    with open(rundir / "run_kwargs.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "event_dir": str(event_dir),
                "event": str(event),
                "bank_folder": bank_folder_serializable,
                "n_int": int(n_int) if isinstance(n_int, (int, float)) else str(n_int),
                "n_ext": int(n_ext),
                "n_phi": int(n_phi),
                "n_t": int(n_t),
                "blocksize": int(blocksize),
                "single_detector_blocksize": int(single_detector_blocksize),
                "i_int_start": int(i_int_start),
                "seed": int(seed) if seed is not None else None,
                "load_inds": bool(load_inds),
                "inds_path": str(inds_path) if inds_path is not None else None,
                "size_limit": int(size_limit),
                "draw_subset": bool(draw_subset),
                "n_draws": int(n_draws) if n_draws is not None else None,
                "rundir": str(rundir),
                "coherent_score_min_n_effective_prior": int(
                    coherent_score_min_n_effective_prior
                ),
                "max_incoherent_lnlike_drop": float(max_incoherent_lnlike_drop),
                "mchirp_guess": float(mchirp_guess)
                if mchirp_guess is not None
                else None,
                "extrinsic_samples": str(extrinsic_samples)
                if extrinsic_samples is not None
                else None,
                "n_phi_incoherent": int(n_phi_incoherent)
                if n_phi_incoherent is not None
                else None,
                "preselected_indices": str(preselected_indices)
                if isinstance(preselected_indices, (str, Path))
                else ("array" if preselected_indices is not None else None),
                "bank_logw_override": (
                    "dict" if bank_logw_override is not None else None
                ),
            },
            fp,
            indent=4,
        )
    event_data = get_event_data(event)
    if isinstance(inds_path, str):
        inds_path = Path(inds_path)

    fbin = np.array(bank_config["fbin"])
    approximant = bank_config["approximant"]
    f_ref = bank_config["f_ref"]
    m_arr = np.array(bank_config["m_arr"])

    print("Creating COGWHEEL objects...")
    coherent_posterior_kwargs = (
        coherent_posterior_kwargs if coherent_posterior_kwargs else {}
    )
    posterior_kwargs = {
        "likelihood_class": RelativeBinningLikelihood,
        "approximant": approximant,
        "prior_class": "CartesianIASPrior",
    } | coherent_posterior_kwargs

    likelihood_kwargs = {"fbin": fbin, "pn_phase_tol": None}
    if "likelihood_kwargs" in posterior_kwargs:
        likelihood_kwargs = likelihood_kwargs | posterior_kwargs.pop(
            "likelihood_kwargs"
        )

    ref_wf_finder_kwargs = {"time_range": (-1e-1, +1e-1), "f_ref": f_ref}
    if "ref_wf_finder_kwargs" in posterior_kwargs:
        ref_wf_finder_kwargs = ref_wf_finder_kwargs | posterior_kwargs.pop(
            "ref_wf_finder_kwargs"
        )
    coherent_posterior = Posterior.from_event(
        event=event_data,
        mchirp_guess=mchirp_guess,
        likelihood_kwargs=likelihood_kwargs,
        ref_wf_finder_kwargs=ref_wf_finder_kwargs,
        **posterior_kwargs,
    )
    par_dic_0 = coherent_posterior.likelihood.par_dic_0.copy()

    coherent_posterior.to_json(dirname=rundir)

    pr = coherent_posterior.prior

    coherent_score_kwargs = {
        "min_n_effective_prior": coherent_score_min_n_effective_prior
    }

    print(f"Running inference with {len(banks)} bank(s)")

    banks_dir = rundir / "banks"
    banks_dir.mkdir(exist_ok=True)

    preselected_indices_dict = None
    if preselected_indices is not None:
        if isinstance(preselected_indices, dict):
            preselected_indices_dict = preselected_indices
        else:
            preselected_indices_dict = {
                bank_id: preselected_indices for bank_id in banks
            }

    inds_path_dict = None
    if load_inds and inds_path is not None:
        if isinstance(inds_path, dict):
            inds_path_dict = {k: Path(v) for k, v in inds_path.items()}
        else:
            inds_path_dict = {bank_id: Path(inds_path) for bank_id in banks}

    # Standardize n_int to n_int_dict
    if n_int is None or (isinstance(n_int, list) and len(n_int) == 0):
        n_int_dict = {}
        for bank_id, bank_path in banks.items():
            bank_file_path = Path(bank_path) / "intrinsic_sample_bank.feather"
            bank = IntrinsicSampleProcessor.load_bank(bank_file_path)
            n_int_dict[bank_id] = len(bank)
    elif isinstance(n_int, int) or (isinstance(n_int, list) and len(n_int) == 1):
        value = n_int if isinstance(n_int, int) else n_int[0]
        n_int_dict = {bank_id: value for bank_id in banks}
    elif isinstance(n_int, list):
        bank_ids = list(banks.keys())
        if len(n_int) != len(bank_ids):
            raise ValueError(
                f"n_int list length ({len(n_int)}) must match number of banks ({len(bank_ids)})"
            )
        n_int_dict = {bank_id: n_int_val for bank_id, n_int_val in zip(bank_ids, n_int)}
    elif isinstance(n_int, dict):
        n_int_dict = n_int
        unknown_bank_ids = set(n_int_dict.keys()) - set(banks.keys())
        if unknown_bank_ids:
            raise ValueError(f"n_int contains unknown bank_id(s): {unknown_bank_ids}")

    bank_logw_override_dict = None
    if bank_logw_override is not None:
        if isinstance(bank_logw_override, dict):
            bank_logw_override_dict = bank_logw_override
            unknown_bank_ids = set(bank_logw_override_dict.keys()) - set(banks.keys())
            if unknown_bank_ids:
                raise ValueError(
                    f"bank_logw_override contains unknown bank_id(s): {unknown_bank_ids}"
                )
        else:
            if len(banks) == 1:
                bank_id = list(banks.keys())[0]
                bank_logw_override_dict = {bank_id: bank_logw_override}
            else:
                raise ValueError(
                    "bank_logw_override must be a dict when using multiple banks"
                )

    return {
        "rundir": rundir,
        "banks": banks,
        "event_data": event_data,
        "coherent_posterior": coherent_posterior,
        "pr": pr,
        "coherent_score_kwargs": coherent_score_kwargs,
        "banks_dir": banks_dir,
        "bank_config": bank_config,
        "fbin": fbin,
        "approximant": approximant,
        "f_ref": f_ref,
        "m_arr": m_arr,
        "par_dic_0": par_dic_0,
        "n_int_dict": n_int_dict,
        "preselected_indices_dict": preselected_indices_dict,
        "inds_path_dict": inds_path_dict,
        "bank_logw_override_dict": bank_logw_override_dict,
    }


def select_intrinsic_samples_per_bank_incoherently(
    *,
    banks: Dict[str, Path],
    event_data: EventData,
    par_dic_0: Dict,
    n_int_dict: Dict[str, int],
    single_detector_blocksize: int,
    n_phi_incoherent: Optional[int],
    n_phi: int,
    n_t: int,
    max_incoherent_lnlike_drop: float,
    preselected_indices_dict: Optional[Dict],
    load_inds: bool,
    inds_path_dict: Optional[Dict[str, Path]],
    banks_dir: Path,
) -> Tuple[
    Dict[str, NDArray[np.int_]],
    Optional[Dict[str, NDArray[np.float64]]],
    Optional[Dict[str, NDArray[np.float64]]],
]:
    """Select intrinsic samples per bank using incoherent likelihood."""
    print("\n=== Incoherent selection per bank ===")
    candidate_inds_by_bank = {}
    lnlikes_by_bank = {}
    lnlikes_di_by_bank = {}

    for bank_id, bank_path in banks.items():
        print(f"\nProcessing bank: {bank_id}")
        bank_rundir = banks_dir / bank_id
        bank_rundir.mkdir(exist_ok=True)

        n_int_k = n_int_dict[bank_id]

        if load_inds and inds_path_dict is not None:
            bank_inds_path = inds_path_dict.get(bank_id)
            if bank_inds_path is not None and bank_inds_path.exists():
                print(
                    f"Loading intrinsic samples indices for bank {bank_id} from {bank_inds_path}"
                )
                if bank_inds_path.suffix == ".npz":
                    data = np.load(bank_inds_path)
                    inds = data["inds"]
                    incoherent_lnlikes = data.get("incoherent_lnlikes", None)
                    lnlikes_di = data.get("lnlikes_di", None)
                else:
                    inds = np.load(bank_inds_path)
                    incoherent_lnlikes = None
                    lnlikes_di = None
            else:
                raise FileNotFoundError(
                    f"inds_path for bank {bank_id} not found: {bank_inds_path}"
                )
        else:
            print(f"Collecting intrinsic samples for bank {bank_id}...")
            n_phi_incoherent = (
                n_phi_incoherent if n_phi_incoherent is not None else n_phi
            )
            inds, lnlikes_di, incoherent_lnlikes = (
                collect_int_samples_from_single_detectors(
                    event_data=event_data,
                    par_dic_0=par_dic_0,
                    single_detector_blocksize=single_detector_blocksize,
                    n_int=n_int_k,
                    n_phi=n_phi_incoherent,
                    n_t=n_t,
                    bank_folder=bank_path,
                    i_int_start=0,
                    max_incoherent_lnlike_drop=max_incoherent_lnlike_drop,
                    preselected_indices=preselected_indices_dict.get(bank_id)
                    if preselected_indices_dict
                    else None,
                    apply_threshold=False,
                )
            )

        candidate_inds_by_bank[bank_id] = inds
        lnlikes_by_bank[bank_id] = incoherent_lnlikes
        if lnlikes_di is not None:
            lnlikes_di_by_bank[bank_id] = lnlikes_di
        print(f"Bank {bank_id}: {len(inds)} intrinsic samples evaluated.")

    return candidate_inds_by_bank, lnlikes_by_bank, lnlikes_di_by_bank


def select_intrinsic_samples_across_banks_by_incoherent_likelihood(
    *,
    banks: Dict[str, Path],
    candidate_inds_by_bank: Dict[str, NDArray[np.int_]],
    incoherent_lnlikes_by_bank: Optional[Dict[str, NDArray[np.float64]]],
    lnlikes_di_by_bank: Optional[Dict[str, NDArray[np.float64]]],
    max_incoherent_lnlike_drop: float,
    banks_dir: Path,
    event_data: EventData,
) -> Tuple[
    Dict[str, NDArray[np.int_]],
    Optional[Dict[str, NDArray[np.float64]]],
    Optional[Dict[str, NDArray[np.float64]]],
]:
    """Select intrinsic samples across banks by applying global threshold."""
    print("\n=== Cross-bank threshold selection ===")
    valid_maxima = [
        np.max(lnlikes)
        for lnlikes in incoherent_lnlikes_by_bank.values()
        if lnlikes is not None and len(lnlikes) > 0
    ]

    selected_inds_by_bank = {}
    selected_lnlikes_by_bank = {}
    selected_lnlikes_di_by_bank = {}

    if valid_maxima:
        global_max_lnlike = max(valid_maxima)
        global_threshold = global_max_lnlike - max_incoherent_lnlike_drop
        print(f"Global maximum incoherent lnlike: {global_max_lnlike:.2f}")
        print(f"Global threshold: {global_threshold:.2f}")
    else:
        print(
            "Warning: No banks have valid incoherent likelihoods. Skipping threshold selection."
        )

    for bank_id in banks.keys():
        inds = candidate_inds_by_bank[bank_id]
        incoherent_lnlikes = incoherent_lnlikes_by_bank[bank_id]

        if (
            incoherent_lnlikes is not None
            and len(incoherent_lnlikes) > 0
            and valid_maxima
        ):
            selected_mask = incoherent_lnlikes >= global_threshold
            selected_inds = inds[selected_mask]
            selected_incoherent_lnlikes = incoherent_lnlikes[selected_mask]
            if bank_id in lnlikes_di_by_bank:
                selected_lnlikes_di = lnlikes_di_by_bank[bank_id][:, selected_mask]
            else:
                selected_lnlikes_di = None
        else:
            selected_inds = np.array([], dtype=np.int_)
            selected_incoherent_lnlikes = np.array([], dtype=np.float64)
            if (
                bank_id in lnlikes_di_by_bank
                and lnlikes_di_by_bank[bank_id] is not None
            ):
                selected_lnlikes_di = np.array([], dtype=np.float64).reshape(
                    len(event_data.detector_names), 0
                )
            else:
                selected_lnlikes_di = None

        selected_inds_by_bank[bank_id] = selected_inds
        selected_lnlikes_by_bank[bank_id] = selected_incoherent_lnlikes
        if selected_lnlikes_di is not None:
            selected_lnlikes_di_by_bank[bank_id] = selected_lnlikes_di

        bank_rundir = banks_dir / bank_id
        np.savez(
            bank_rundir / "intrinsic_samples.npz",
            inds=selected_inds,
            lnlikes_di=selected_lnlikes_di,
            incoherent_lnlikes=selected_incoherent_lnlikes,
        )
        print(
            f"Bank {bank_id}: {len(selected_inds)} intrinsic samples "
            + (
                "passed global threshold."
                if incoherent_lnlikes is not None
                and len(incoherent_lnlikes) > 0
                and valid_maxima
                else "(no likelihoods or skipped)."
            )
        )

    total_filtered = sum(len(arr) for arr in selected_inds_by_bank.values())
    if total_filtered == 0:
        raise ValueError(
            "No intrinsic samples remain after incoherent selection; "
            "cannot proceed to extrinsic/coherent steps."
        )

    return selected_inds_by_bank, selected_lnlikes_by_bank, selected_lnlikes_di_by_bank


def draw_extrinsic_samples(
    *,
    banks: Dict[str, Path],
    event_data: EventData,
    par_dic_0: Dict,
    fbin: NDArray[np.float64],
    approximant: str,
    selected_inds_by_bank: Dict[str, NDArray[np.int_]],
    coherent_score_kwargs: Dict,
    seed: Optional[int],
    n_ext: int,
    rundir: Path,
    extrinsic_samples: Union[str, Path, None],
    n_combine: int = 16,
    min_marg_lnlike_for_sampling: float = 0.0,
    single_marg_info_min_n_effective_prior: float = 32.0,
) -> Tuple[pd.DataFrame, NDArray, NDArray]:
    """Draw extrinsic samples from intrinsic candidates or load from file."""
    print("\n=== Extrinsic sample generation ===")
    if extrinsic_samples is not None:
        print("Loading extrinsic samples from file...")
        if isinstance(extrinsic_samples, (str, Path)):
            extrinsic_samples_df = pd.read_feather(extrinsic_samples)
            if "log_prior_weights" not in extrinsic_samples_df.columns:
                if "weights" in extrinsic_samples_df.columns:
                    extrinsic_samples_df["log_prior_weights"] = np.log(
                        extrinsic_samples_df["weights"]
                    )
                else:
                    extrinsic_samples_df["log_prior_weights"] = 0.0
            extrinsic_samples_df.to_feather(rundir / "extrinsic_samples.feather")
            response_dpe_path = rundir / "response_dpe.npy"
            timeshift_dbe_path = rundir / "timeshift_dbe.npy"
            if response_dpe_path.exists() and timeshift_dbe_path.exists():
                response_dpe = np.load(response_dpe_path)
                timeshift_dbe = np.load(timeshift_dbe_path)
            else:
                processor = ExtrinsicSampleProcessor(event_data.detector_names)
                response_dpe, timeshift_dbe = processor.get_components(
                    extrinsic_samples_df, fbin, event_data.tcoarse
                )
                np.save(arr=response_dpe, file=response_dpe_path)
                np.save(arr=timeshift_dbe, file=timeshift_dbe_path)
            return extrinsic_samples_df, response_dpe, timeshift_dbe

    print("Generating extrinsic samples (global, multibank)...")
    first_bank_path = list(banks.values())[0]
    waveform_dir = first_bank_path / "waveforms"
    bank_file_path = first_bank_path / "intrinsic_sample_bank.feather"
    wfg = WaveformGenerator.from_event_data(event_data, approximant)
    marg_ext_like = MarginalizationExtrinsicSamplerFreeLikelihood(
        event_data, wfg, par_dic_0, fbin, coherent_score=coherent_score_kwargs
    )

    # Note: intrinsic_bank_file and waveform_dir are not used in multibank
    # get_marg_info_multibank; choosing first_bank_path is only for compatibility
    ext_sample_generator = CoherentExtrinsicSamplesGenerator(
        likelihood=marg_ext_like,
        intrinsic_bank_file=bank_file_path,
        waveform_dir=waveform_dir,
        seed=seed,
    )

    banks_list = []
    waveform_dirs_list = []
    sample_idx_arrays = []
    bank_idx_arrays = []

    for bank_id, bank_path in banks.items():
        filtered_inds = selected_inds_by_bank[bank_id]
        if len(filtered_inds) == 0:
            continue

        dense_idx = len(banks_list)
        bank_df = pd.read_feather(bank_path / "intrinsic_sample_bank.feather")

        banks_list.append(bank_df)
        waveform_dirs_list.append(bank_path / "waveforms")
        sample_idx_arrays.append(filtered_inds)
        bank_idx_arrays.append(np.full(len(filtered_inds), dense_idx, dtype=int))

    if len(sample_idx_arrays) == 0:
        raise ValueError(
            "No intrinsic samples available for multibank extrinsic sampling "
            "(all banks empty)."
        )

    sample_idx_arr = np.concatenate(sample_idx_arrays)
    bank_idx_arr = np.concatenate(bank_idx_arrays)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(sample_idx_arr))
    sample_idx_arr = sample_idx_arr[perm]
    bank_idx_arr = bank_idx_arr[perm]

    marg_info = ext_sample_generator.get_marg_info_multibank(
        n_combine=n_combine,
        banks=banks_list,
        waveform_dirs=waveform_dirs_list,
        sample_idx_arr=sample_idx_arr,
        bank_idx_arr=bank_idx_arr,
        min_marg_lnlike_for_sampling=min_marg_lnlike_for_sampling,
        single_marg_info_min_n_effective_prior=single_marg_info_min_n_effective_prior,
        save_marg_info=True,
        save_marg_info_dir=rundir,
    )

    (extrinsic_samples_df, response_dpe, timeshift_dbe) = (
        ext_sample_generator.draw_extrinsic_samples_from_indices(
            n_ext, marg_info=marg_info
        )
    )

    extrinsic_samples_df.to_feather(rundir / "extrinsic_samples.feather")
    np.save(arr=response_dpe, file=rundir / "response_dpe.npy")
    np.save(arr=timeshift_dbe, file=rundir / "timeshift_dbe.npy")

    return extrinsic_samples_df, response_dpe, timeshift_dbe


def run_coherent_inference_per_bank(
    *,
    banks: Dict[str, Path],
    event_data: EventData,
    rundir: Path,
    banks_dir: Path,
    par_dic_0: Dict,
    selected_inds_by_bank: Dict[str, NDArray[np.int_]],
    n_int_dict: Dict[str, int],
    n_ext: int,
    n_phi: int,
    m_arr: NDArray[np.int_],
    blocksize: int,
    size_limit: int,
    max_bestfit_lnlike_diff: float,
    bank_logw_override_dict: Optional[Dict],
) -> List[Dict[str, Any]]:
    """Run coherent inference for each bank."""
    print("\n=== Coherent inference per bank ===")
    bank_results = []
    all_prob_samples = []

    for bank_id, bank_path in banks.items():
        print(f"\nProcessing bank: {bank_id}")
        bank_rundir = banks_dir / bank_id

        inds = selected_inds_by_bank[bank_id]
        n_int_k = n_int_dict[bank_id]
        n_total_samples = n_phi * n_ext * n_int_k

        intrinsic_logw_lookup = None
        if bank_logw_override_dict is not None and bank_id in bank_logw_override_dict:
            override_logw_full = np.asarray(bank_logw_override_dict[bank_id])
            override_logw = override_logw_full[inds]
            intrinsic_logw_lookup = (inds, override_logw)

        (
            lnZ_k,
            lnZ_discarded_k,
            n_effective_k,
            n_effective_i_k,
            n_effective_e_k,
            n_distance_marginalizations_k,
        ) = run_coherent_inference(
            event_data=event_data,
            bank_rundir=bank_rundir,
            top_rundir=rundir,
            par_dic_0=par_dic_0,
            bank_folder=bank_path,
            n_total_samples=n_total_samples,
            inds=inds,
            n_ext=n_ext,
            n_phi=n_phi,
            m_arr=m_arr,
            blocksize=blocksize,
            renormalize_log_prior_weights_i=False,
            intrinsic_logw_lookup=intrinsic_logw_lookup,
            size_limit=size_limit,
            max_bestfit_lnlike_diff=max_bestfit_lnlike_diff,
        )

        prob_samples_k = pd.read_feather(bank_rundir / "prob_samples.feather")
        prob_samples_k["bank_id"] = bank_id

        bank_results.append(
            {
                "bank_id": bank_id,
                "lnZ_k": lnZ_k,
                "lnZ_discarded_k": lnZ_discarded_k,
                "N_k": n_total_samples,
                "n_effective_k": n_effective_k,
                "n_effective_i_k": n_effective_i_k,
                "n_effective_e_k": n_effective_e_k,
                "n_distance_marginalizations_k": n_distance_marginalizations_k,
                "n_inds_used": len(inds),
            }
        )
        all_prob_samples.append(prob_samples_k)

    return bank_results


def aggregate_and_save_results(
    *,
    per_bank_results: List[Dict[str, Any]],
    banks: Dict[str, Path],
    event_data: EventData,
    rundir: Path,
    banks_dir: Path,
    n_phi: int,
    pr: Prior,
    n_draws: Optional[int],
    draw_subset: bool,
) -> Path:
    """Aggregate results across banks and save final outputs."""
    print("\n=== Combining results ===")

    N_total = sum(r["N_k"] for r in per_bank_results)
    if N_total <= 0:
        warnings.warn("Total intrinsic normalization N_total <= 0")

    # Assuming disjoint bank supports and properly normalized priors,
    # the total evidence is the sum of sub-bank evidences
    lnZ_values = [r["lnZ_k"] for r in per_bank_results]
    lnZ_total = safe_logsumexp(lnZ_values)

    lnZ_discarded_values = [r["lnZ_discarded_k"] for r in per_bank_results]
    lnZ_discarded_total = safe_logsumexp(lnZ_discarded_values)

    print("\nCombining prob_samples across banks...")
    all_prob_samples = []
    for bank_id in banks.keys():
        bank_rundir = banks_dir / bank_id
        prob_samples_k = pd.read_feather(bank_rundir / "prob_samples.feather")
        prob_samples_k["bank_id"] = bank_id
        all_prob_samples.append(prob_samples_k)

    combined_prob_samples = pd.concat(all_prob_samples, ignore_index=True)

    for r in per_bank_results:
        mask = combined_prob_samples["bank_id"] == r["bank_id"]
        combined_prob_samples.loc[mask, "ln_weight_unnormalized"] = (
            combined_prob_samples.loc[mask, "ln_posterior"] - np.log(r["N_k"])
        )

    combined_prob_samples["weights"] = exp_normalize(
        combined_prob_samples["ln_weight_unnormalized"].values
    )

    combined_prob_samples.to_feather(rundir / "prob_samples.feather")

    samples = postprocess(
        event_data=event_data,
        rundir=rundir,
        bank_folder=banks,
        n_phi=n_phi,
        pr=pr,
        prob_samples=combined_prob_samples,
        n_draws=n_draws,
        max_n_draws=10**4,
        draw_subset=draw_subset,
    )

    print("Saving samples to file...")
    samples_path = Path(rundir) / "samples.feather"
    samples.to_feather(samples_path)
    print(f"Samples saved to:\n {samples_path}")

    total_weight = combined_prob_samples["weights"].sum()
    n_effective_total = 1.0 / (
        (combined_prob_samples["weights"] ** 2).sum() / total_weight
    )

    n_effective_i_total = np.average(
        [r["n_effective_i_k"] for r in per_bank_results],
        weights=[r["N_k"] for r in per_bank_results],
    )
    n_effective_e_total = np.average(
        [r["n_effective_e_k"] for r in per_bank_results],
        weights=[r["N_k"] for r in per_bank_results],
    )

    summary_dict = {
        "n_effective": float(n_effective_total),
        "n_effective_i": float(n_effective_i_total),
        "n_effective_e": float(n_effective_e_total),
        "bestfit_lnlike_max": float(samples["bestfit_lnlike"].max()),
        "lnl_marginalized_max": float(samples["lnl_marginalized"].max()),
        "n_i_inds_used": int(sum(r["n_inds_used"] for r in per_bank_results)),
        "ln_evidence": float(lnZ_total),
        "ln_evidence_discarded": float(lnZ_discarded_total),
        "n_distance_marginalizations": int(
            sum(r["n_distance_marginalizations_k"] for r in per_bank_results)
        ),
        "n_banks": len(banks),
        "per_bank_results": {
            r["bank_id"]: {
                "ln_evidence": float(r["lnZ_k"]),
                "n_effective": float(r["n_effective_k"]),
                "n_inds_used": int(r["n_inds_used"]),
                "N_k": int(r["N_k"]),
            }
            for r in per_bank_results
        },
    }

    if getattr(event_data, "injection", None) is not None:
        first_bank_id = list(banks.keys())[0]
        clp_path = banks_dir / first_bank_id / "CoherentLikelihoodProcessor.json"
        clp = read_json(clp_path)
        inj_par_dic = event_data.injection["par_dic"]
        bestfit_lnlike, lnl_marginalized = clp.get_bestfit_and_marginalized_lnlike(
            inj_par_dic
        )
        summary_dict["injection"] = {
            "bestfit_lnlike": bestfit_lnlike,
            "lnl_marginalized": lnl_marginalized,
        }

    with open(rundir / "summary_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=4)

    return rundir


def run(
    event: Union[str, Path, EventData],
    bank_folder: Union[str, Path, List[Union[str, Path]], Tuple[Union[str, Path], ...]],
    n_ext: int,
    n_phi: int,
    n_t: int,
    n_int: Union[int, List[int], Dict[str, int], None] = None,
    blocksize: int = 512,
    single_detector_blocksize: int = 512,
    i_int_start: int = 0,
    seed: int = None,
    load_inds: bool = False,
    inds_path: Union[Path, str, Dict[str, Union[Path, str]], None] = None,
    size_limit: int = 10**7,
    draw_subset: bool = True,
    n_draws: int = None,
    event_dir: Union[str, Path] = None,
    rundir: Union[str, Path] = None,
    coherent_score_min_n_effective_prior: int = 100,
    max_incoherent_lnlike_drop: float = 20,
    max_bestfit_lnlike_diff: float = 20,
    mchirp_guess: float = None,
    extrinsic_samples: Union[str, Path] = None,
    n_phi_incoherent: int = None,
    preselected_indices: Union[NDArray[np.int_], List[int], str, Path, None] = None,
    bank_logw_override: Union[
        Dict[str, Union[NDArray[np.float64], List[float], pd.Series]],
        NDArray[np.float64],
        List[float],
        pd.Series,
        None,
    ] = None,
    coherent_posterior_kwargs: Dict = {},
) -> Path:
    """Run the magic integral for a given event and bank folder."""
    # Step 1: Prepare shared objects
    ctx = prepare_run_objects(
        event=event,
        bank_folder=bank_folder,
        n_int=n_int,
        n_ext=n_ext,
        n_phi=n_phi,
        n_t=n_t,
        blocksize=blocksize,
        single_detector_blocksize=single_detector_blocksize,
        i_int_start=i_int_start,
        seed=seed,
        load_inds=load_inds,
        inds_path=inds_path,
        size_limit=size_limit,
        draw_subset=draw_subset,
        n_draws=n_draws,
        event_dir=event_dir,
        rundir=rundir,
        coherent_score_min_n_effective_prior=coherent_score_min_n_effective_prior,
        max_incoherent_lnlike_drop=max_incoherent_lnlike_drop,
        max_bestfit_lnlike_diff=max_bestfit_lnlike_diff,
        mchirp_guess=mchirp_guess,
        extrinsic_samples=extrinsic_samples,
        n_phi_incoherent=n_phi_incoherent,
        preselected_indices=preselected_indices,
        bank_logw_override=bank_logw_override,
        coherent_posterior_kwargs=coherent_posterior_kwargs,
    )

    # Step 2: Incoherent selection per bank
    candidate_inds_by_bank, lnlikes_by_bank, lnlikes_di_by_bank = (
        select_intrinsic_samples_per_bank_incoherently(
            banks=ctx["banks"],
            event_data=ctx["event_data"],
            par_dic_0=ctx["par_dic_0"],
            n_int_dict=ctx["n_int_dict"],
            single_detector_blocksize=single_detector_blocksize,
            n_phi_incoherent=n_phi_incoherent,
            n_phi=n_phi,
            n_t=n_t,
            max_incoherent_lnlike_drop=max_incoherent_lnlike_drop,
            preselected_indices_dict=ctx["preselected_indices_dict"],
            load_inds=load_inds,
            inds_path_dict=ctx["inds_path_dict"],
            banks_dir=ctx["banks_dir"],
        )
    )

    # Step 3: Cross-bank selection
    selected_inds_by_bank, selected_lnlikes_by_bank, selected_lnlikes_di_by_bank = (
        select_intrinsic_samples_across_banks_by_incoherent_likelihood(
            banks=ctx["banks"],
            candidate_inds_by_bank=candidate_inds_by_bank,
            incoherent_lnlikes_by_bank=lnlikes_by_bank,
            lnlikes_di_by_bank=lnlikes_di_by_bank,
            max_incoherent_lnlike_drop=max_incoherent_lnlike_drop,
            banks_dir=ctx["banks_dir"],
            event_data=ctx["event_data"],
        )
    )

    # Step 4: Draw extrinsic samples
    extrinsic_samples_df, response_dpe, timeshift_dbe = draw_extrinsic_samples(
        banks=ctx["banks"],
        event_data=ctx["event_data"],
        par_dic_0=ctx["par_dic_0"],
        fbin=ctx["fbin"],
        approximant=ctx["approximant"],
        selected_inds_by_bank=selected_inds_by_bank,
        coherent_score_kwargs=ctx["coherent_score_kwargs"],
        seed=seed,
        n_ext=n_ext,
        rundir=ctx["rundir"],
        extrinsic_samples=extrinsic_samples,
    )

    # Step 5: Coherent inference per bank
    per_bank_results = run_coherent_inference_per_bank(
        banks=ctx["banks"],
        event_data=ctx["event_data"],
        rundir=ctx["rundir"],
        banks_dir=ctx["banks_dir"],
        par_dic_0=ctx["par_dic_0"],
        selected_inds_by_bank=selected_inds_by_bank,
        n_int_dict=ctx["n_int_dict"],
        n_ext=n_ext,
        n_phi=n_phi,
        m_arr=ctx["m_arr"],
        blocksize=blocksize,
        size_limit=size_limit,
        max_bestfit_lnlike_diff=max_bestfit_lnlike_diff,
        bank_logw_override_dict=ctx["bank_logw_override_dict"],
    )

    # Step 6: Aggregate and save
    return aggregate_and_save_results(
        per_bank_results=per_bank_results,
        banks=ctx["banks"],
        event_data=ctx["event_data"],
        rundir=ctx["rundir"],
        banks_dir=ctx["banks_dir"],
        n_phi=n_phi,
        pr=ctx["pr"],
        n_draws=n_draws,
        draw_subset=draw_subset,
    )


def run_and_profile(**kwargs):
    """
    Run the magic integral and profile the run.
    """
    profiler = cProfile.Profile()
    rundir = None
    profiler.enable()
    try:
        rundir = run(**kwargs)
        return rundir
    finally:
        profiler.disable()
        if rundir is not None:
            profile_obj_path = Path(rundir) / "profile_output.prof"
            profile_report_path = Path(rundir) / "profile_output.txt"
            profiler.dump_stats(profile_obj_path)
            with open(profile_report_path, "w", encoding="utf-8") as f:
                ps = pstats.Stats(profiler, stream=f)
                ps.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats()


def parse_arguments() -> Dict:
    """Parser for arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--event",
        type=Path,
        required=True,
        help="Path to the event data.",
    )
    parser.add_argument(
        "--bank_folder",
        type=Path,
        required=True,
        help="Path to the bank folder.",
    )
    parser.add_argument(
        "--n_int",
        type=int,
        nargs="*",
        help=(
            "Number of intrinsic samples. Can be: int (all banks), "
            "list of ints (one per bank in order), dict mapping bank_id to int, "
            "or None/omitted to use full bank size."
        ),
        default=None,
    )
    parser.add_argument(
        "--i_int_start",
        type=int,
        help="Minimal index of intrinsic sample to use.",
    )
    parser.add_argument(
        "--n_ext", type=int, help="Number of extrinsic samples.", default=2**9
    )
    parser.add_argument(
        "--n_phi",
        type=int,
        help="Number of phi_ref samples for coherent evaluation.",
        default=100,
    )
    parser.add_argument(
        "--n_phi_incoherent",
        type=int,
        help="Number of phi_ref samples for single detector evaluation (thresholding). If None, uses n_phi.",
        default=None,
    )
    parser.add_argument(
        "--n_t",
        type=int,
        help="""Number of time samples in the single-detecotor
                        time-grid""",
        default=128,
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        help="Block size for coherent likelihood evaluation.",
        default=512,
    )
    parser.add_argument(
        "--single_detector_blocksize",
        type=int,
        help="Block size for single detector likelihood evaluation.",
        default=512,
    )
    parser.add_argument("--seed", type=int, help="Random seed", default=None)
    parser.add_argument(
        "--load_inds", action="store_true", help="Load indices from file."
    )
    parser.add_argument(
        "--inds_path", type=Path, help="Path to indices file.", default=None
    )
    parser.add_argument(
        "--size_limit",
        type=int,
        default=10**7,
        help="Maximal number of samples generated.",
    )
    parser.add_argument(
        "--event_dir",
        type=Path,
        help="Directory of the event. Ignored if 'rundir' is provided.",
        default=None,
    )
    parser.add_argument(
        "--rundir",
        type=Path,
        help=(
            "Directory to save the results. If not provided, will be inferred "
            "from 'event_dir'."
        ),
        default=None,
    )

    parser.add_argument(
        "--coherent_score_min_n_effective_prior",
        type=int,
        default=100,
        help=(
            "Minimum Effective Sample Size (using prior weights) for the "
            + "coherent score. Too low ESS can bias the weights and "
            + "the evidence calculation."
            + "Too ESS will require many MarginalizationInfo iterations."
        ),
    )
    parser.add_argument(
        "--max_incoherent_lnlike_drop",
        type=float,
        default=20,
        help=(
            "Maximum log-likelihood drop from the best fit for the incoherent "
            "sum of single-detector likelihoods."
        ),
    )
    parser.add_argument(
        "--max_bestfit_lnlike_diff",
        type=float,
        default=20,
        help=(
            "Maximum log-likelihood difference from the best fit for the "
            "coherent likelihood evaluation."
        ),
    )
    parser.add_argument(
        "--mchirp_guess",
        type=float,
        default=None,
        help="Optional: Initial guess for the chirp mass (mchirp).",
    )
    parser.add_argument(
        "--extrinsic_samples",
        type=Path,
        default=None,
        help="Path to extrinsic samples file. If None, extrinsic samples will be drawn as usual.",
    )
    parser.add_argument(
        "--preselected_indices",
        type=str,
        default=None,
        help="Preselected indices for second-stage filtering. Can be a path to .npy file.",
    )

    return vars(parser.parse_args())


if __name__ == "__main__":
    kwargs = parse_arguments()
    run_and_profile(**kwargs)
