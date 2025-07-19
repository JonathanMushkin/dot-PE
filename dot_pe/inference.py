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
import copy
import cProfile
import json
import os
import pstats
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from lal import GreenwichMeanSiderealTime
from numpy.typing import NDArray
from tqdm import tqdm

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

from cogwheel import skyloc_angles
from cogwheel.data import EventData
from cogwheel.gw_utils import DETECTORS, get_geocenter_delays
from cogwheel.likelihood import RelativeBinningLikelihood
from cogwheel.posterior import Posterior
from cogwheel.utils import exp_normalize, get_rundir, mkdirs, read_json
from cogwheel.waveform import WaveformGenerator

from .base_sampler_free_sampling import (
    get_n_effective_total_i_e,
)
from .likelihood_calculator import LinearFree
from .marginalization import MarginalizationExtrinsicSamplerFreeLikelihood
from .coherent_processing import (
    CoherentLikelihoodProcessor,
    CoherentExtrinsicSamplesGenerator,
)
from .utils import get_event_data, safe_logsumexp
from .single_detector import SingleDetectorProcessor


def inds_to_blocks(
    indices: NDArray[np.int_], block_size: int
) -> List[NDArray[np.int_]]:
    """Split the indices into blocks of size blocksize (or less)."""
    return [
        indices[i * block_size : (i + 1) * block_size]
        for i in range(-(len(indices) // -block_size))
    ]


def run_for_single_detector(
    event_data: EventData,
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
) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.complex128]]]:
    """
    Perform single detector likelihood evaluations and return the likelihoods.
    """
    if isinstance(inds, int):
        inds = np.arange(inds)
    if isinstance(inds, list):
        inds = np.array(inds)

    # load and edit event data
    if isinstance(event_data, Path):
        event_data_path = event_data
        event_data_1d = EventData.from_npz(filename=event_data_path)
    else:
        event_data_1d = copy.deepcopy(event_data)
    indices = [event_data_1d.detector_names.index(det) for det in list(det_name)]

    array_attributes = ["strain", "blued_strain", "wht_filter"]
    for attr in array_attributes:
        setattr(event_data_1d, attr, getattr(event_data_1d, attr)[indices])

    tuple_attributes = [
        "detector_names",
    ]
    for attr in tuple_attributes:
        temp = tuple(np.take(getattr(event_data_1d, attr), indices))
        setattr(event_data_1d, attr, temp)
    if getattr(event_data_1d, "injection", None) is not None:
        event_data_1d.injection["h_h"] = np.take(
            event_data_1d.injection["h_h"], indices
        ).tolist()
        event_data_1d.injection["d_h"] = np.take(
            event_data_1d.injection["d_h"], indices
        ).tolist()
    wfg = WaveformGenerator.from_event_data(event_data_1d, approximant)

    likelihood_linfree = LinearFree(event_data_1d, wfg, par_dic_0, fbin)
    bank_file_path = Path(bank_folder) / "intrinsic_sample_bank.feather"
    waveform_dir = Path(bank_folder) / "waveforms"

    sdp = SingleDetectorProcessor(
        bank_file_path,
        waveform_dir,
        n_phi,
        m_arr,
        likelihood_linfree,
        size_limit=size_limit,
        ext_block_size=single_detector_blocksize,
        int_block_size=single_detector_blocksize,
    )

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
    dt_fraction = 1
    lat, lon = skyloc_angles.cart3d_to_latlon(
        skyloc_angles.normalize(DETECTORS[det_name].location)
    )

    par_dic = sdp.transform_par_dic_by_sky_poisition(
        det_name, par_dic_0, lon, lat, tgps
    )

    delay = get_geocenter_delays(det_name, par_dic["lat"], par_dic["lon"])[0]
    tcoarse = sdp.likelihood.event_data.tcoarse
    t_grid = (np.arange(n_t) - n_t // 2) * (
        sdp.likelihood.event_data.times[1] * dt_fraction
    )
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
        sdp.evidence.n_phi,
        sdp.evidence.m_arr,
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
) -> Tuple[NDArray[np.int_], NDArray[np.float64]]:
    """
    Perform n_det independent single-detector likelihood evaluations and
    return the indices of the samples that passed a threshold and their
    corresponding incoherent log-likelihood values.
    """

    with open(bank_folder / "bank_config.json", "r", encoding="utf-8") as f:
        bank_config = json.load(f)
        fbin = np.array(bank_config["fbin"])
        approximant = bank_config["approximant"]
        m_arr = np.array(bank_config["m_arr"])
    # do single detector pe for each detector
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
            )
            if h_impb is None:
                lnlike_di[d, batch_start:batch_end] = temp[0]
                h_impb = temp[1]
            else:
                lnlike_di[d, batch_start:batch_end] = temp

    incoherent_lnlikes = np.sum(lnlike_di, axis=0)

    incoherent_threshold = incoherent_lnlikes.max() - max_incoherent_lnlike_drop

    selected = incoherent_lnlikes >= incoherent_threshold

    inds = intrinsic_indices[selected]
    incoherent_lnlikes = incoherent_lnlikes[inds]
    return inds, incoherent_lnlikes


def run_coherent_inference(
    event_data: EventData,
    rundir: Path,
    par_dic_0: Dict,
    bank_folder: Path,
    n_int: int,
    inds: NDArray[np.int_],
    n_ext: int,
    n_phi: int,
    m_arr: NDArray[np.int_],
    blocksize: int,
    pr,
    n_combine: int = 16,
    seed: int = None,
    size_limit: int = 10**7,
    n_draws: int = None,
    max_n_draws: int = 10**4,
    draw_subset: bool = False,
    single_marg_info_min_n_effective_prior: int = 32,
    max_bestfit_lnlike_diff: float = 20,
    coherent_score_kwargs: Dict = None,
) -> Tuple[pd.DataFrame, float, float, float, float, float, int]:
    """
    TODO: consider split this into a function that prepares and runs the loop,
    and one that produces the samples & integrates
    Perform a full coherent multi-detector likelihood evaluation,
    and get samples with probabilistic weights.

    Parameters
    ----------
    event_data : EventData
        Event data.
    rundir : Path
        Directory to save / load the results.
    par_dic_0 : dict
        Dictionary of reference-waveform parameters.
    bank_folder : Path
        Path to the bank folder.
    n_int : int
        Number of intrinsic samples.
    inds : array
        Indices of the intrinsic samples to include.
    n_ext : int
        Number of extrinsic samples.
    n_phi : int
        Number of phi_ref samples.
    m_arr : array
        Array of harmonic m-modes to include. Default is [2,1,3,4]
    blocksize : int
        Block size for likelihood evaluation.
    pr : Prior
        Prior object. Used to (inverse-) transform the samples.
    seed : int
        Random seed.

    Returns
    -------
    samples : pd.DataFrame
        The standardized samples.
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
    waveform_dir = bank_folder / "waveforms"
    bank_file_path = bank_folder / "intrinsic_sample_bank.feather"
    with open(bank_folder / "bank_config.json", "r", encoding="utf-8") as f:
        bank_config = json.load(f)
        fbin = np.array(bank_config["fbin"])
        approximant = bank_config["approximant"]

    wfg = WaveformGenerator.from_event_data(event_data, approximant)
    marg_ext_like = MarginalizationExtrinsicSamplerFreeLikelihood(
        event_data, wfg, par_dic_0, fbin, coherent_score=coherent_score_kwargs
    )

    ext_sample_generator = CoherentExtrinsicSamplesGenerator(
        likelihood=marg_ext_like,
        intrinsic_bank_file=bank_file_path,
        waveform_dir=waveform_dir,
        seed=seed,
        n_phi=n_phi,
        m_arr=m_arr,
    )

    get_marg_info_kwargs = {
        "save_marg_info": True,
        "save_marg_info_dir": rundir,
        "n_combine": n_combine,
        "indices": inds,
        "single_marg_info_min_n_effective_prior": single_marg_info_min_n_effective_prior,
    }

    (extrinsic_samples, response_dpe, timeshift_dbe) = (
        ext_sample_generator.draw_extrinsic_samples_from_indices(
            n_ext, get_marg_info_kwargs=get_marg_info_kwargs
        )
    )

    # save results to disk
    extrinsic_samples.to_feather(rundir / "extrinsic_samples.feather")
    np.save(arr=response_dpe, file=rundir / "response_dpe.npy")
    np.save(arr=timeshift_dbe, file=rundir / "timeshift_dbe.npy")

    likelihood_linfree = LinearFree(event_data, wfg, par_dic_0, fbin)

    clp = CoherentLikelihoodProcessor(
        bank_file_path,
        waveform_dir,
        n_phi,
        m_arr,
        likelihood_linfree,
        size_limit=size_limit,
        max_bestfit_lnlike_diff=max_bestfit_lnlike_diff,
    )

    i_blocks = inds_to_blocks(inds, blocksize)
    e_blocks = inds_to_blocks(np.arange(n_ext), blocksize)
    clp.load_extrinsic_samples_data(rundir)
    clp.to_json(rundir, overwrite=True)
    # perform the run
    print(f"Creating {len(i_blocks)} x {len(e_blocks)} likelihood blocks...")

    _ = clp.create_likelihood_blocks(
        tempdir=rundir,
        i_blocks=i_blocks,
        e_blocks=e_blocks,
        response_dpe=response_dpe,
        timeshift_dbe=timeshift_dbe,
    )

    clp.prob_samples["weights"] = np.exp(
        clp.prob_samples["ln_posterior"].to_numpy(dtype=np.float64)
    )
    clp.prob_samples["weights"] /= clp.prob_samples["weights"].sum()
    clp.prob_samples.to_feather(rundir / "prob_samples.feather")

    # Process the results
    ln_evidence = safe_logsumexp(clp.prob_samples["ln_posterior"].values) - np.log(
        n_phi * n_ext * n_int
    )
    ln_evidence_discarded = clp.logsumexp_discarded_ln_posterior - np.log(
        n_phi * n_ext * n_int
    )
    intrinsic_samples = clp.intrinsic_sample_processor.load_bank(
        bank_file_path,
    )

    n_effective, n_effective_i, n_effective_e = get_n_effective_total_i_e(
        clp.prob_samples, assume_noramlized=False
    )

    if draw_subset:
        n_draws = n_draws if n_draws else int(max(n_effective // 2, 1))
        if max_n_draws is not None:
            n_draws = min(n_draws, max_n_draws)
        if n_draws > n_effective:
            warnings.warn(
                f"n_draws ({n_draws}) is bigger than n_effective ({n_effective})."
            )

        prob_samples = clp.prob_samples.sample(
            n=n_draws, weights="weights", replace=True
        ).reset_index(drop=True)
        prob_samples["weights"] = 1.0
    else:
        prob_samples = clp.prob_samples

    print("Standardizing samples...")
    stnd_start_time = time.time()
    samples = standardize_samples(
        clp.intrinsic_sample_processor,
        ext_sample_generator.likelihood.coherent_score.lookup_table,
        pr,
        waveform_dir,
        prob_samples,
        intrinsic_samples,
        extrinsic_samples,
        n_phi,
        event_data.tgps,
    )

    if draw_subset:  # could be more elegant. But it is not. JM 10/3/2025
        samples["weights"] = 1.0
    print(
        "Standardizing samples done in "
        + f"{(time.time() - stnd_start_time):.3g} seconds."
    )

    clp.to_json(dirname=rundir, overwrite=True)

    return (
        samples,
        ln_evidence,
        ln_evidence_discarded,
        n_effective,
        n_effective_i,
        n_effective_e,
        clp.n_distance_marginalizations,
    )


def standardize_samples(
    intrinsic_sample_processor,
    lookup_table,
    pr,
    waveform_dir: Path,
    prob_samples: pd.DataFrame,
    intrinsic_samples: pd.DataFrame,
    extrinsic_samples: pd.DataFrame,
    n_phi: int,
    tgps: float,
) -> pd.DataFrame:
    """
    Standardize the samples and calculate the weights.

    Parameters
    ----------
    intrinsic_sample_processor : IntrinsicSampleProcessor
        IntrinsicSampleProcessor object.
    lookup_table : LookupTable
        LookupTable object, for distance marginalization.
    pr : Prior
        Prior object.
    waveform_dir : Path
        Path to the waveform directory.
    prob_samples : DataFrame
        Samples with indices columns `i`, `e` and `o` and probablistic
        information.
    intrinsic_samples : DataFrame
        Intrinsic samples.
    extrinsic_samples : DataFrame
        Extrinsic samples.
    n_phi : int
        Number of phi_ref samples.
    tgps : float
        GPS time of event.
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

    dt_linfree_u, dphi_linfree_u = intrinsic_sample_processor.load_linfree_dt_and_dphi(
        waveform_dir, unique_i
    )

    combined_samples["t_geocenter"] = (
        combined_samples["t_geocenter_linfree"] + dt_linfree_u[u_i]
    )

    combined_samples["phi_ref"] = (
        combined_samples["phi_ref_linfree"] - dphi_linfree_u[u_i]
    )

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
    combined_samples["weights"] = exp_normalize(combined_samples["ln_posterior"].values)

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


def run(
    event: Union[str, Path, EventData],
    bank_folder: Path,
    n_int: int,
    n_ext: int,
    n_phi: int,
    n_t: int,
    blocksize: int = 512,
    single_detector_blocksize: int = 512,
    i_int_start: int = 0,
    seed: int = None,
    load_inds: bool = False,
    inds_path: Path = None,
    size_limit: int = 10**7,
    draw_subset: bool = True,
    n_draws: int = None,
    event_dir: Union[str, Path] = None,
    rundir: Union[str, Path] = None,
    coherent_score_min_n_effective_prior: int = 100,
    max_incoherent_lnlike_drop: float = 20,
    max_bestfit_lnlike_diff: float = 20,
    mchirp_guess: float = None,
) -> Path:
    """Run the magic integral for a given event and bank folder."""

    print("Setting paths & loading configurations...")
    if event_dir is not None and rundir is not None:
        warnings.warn(
            "Both 'event_dir' and 'rundir' are provided. 'event_dir' will be ignored."
        )
    if rundir is None:
        rundir = get_rundir(event_dir)

    if not Path(rundir).exists():
        mkdirs(rundir)
    with open(rundir / "run_kwargs.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "event_dir": str(event_dir),
                "event": str(event),
                "bank_folder": str(bank_folder),
                "n_int": int(n_int),
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
            },
            fp,
            indent=4,
        )
    # set paths and basic configs
    event_data = get_event_data(event)
    bank_config_path = Path(bank_folder) / "bank_config.json"

    with open(bank_config_path, "r", encoding="utf-8") as fp:
        bank_config = json.load(fp)
        fbin = np.array(bank_config["fbin"])
        approximant = bank_config["approximant"]
        f_ref = bank_config["f_ref"]
        m_arr = np.array(bank_config["m_arr"])

    print("Creating COGWHEEL objects...")
    posterior_kwargs = {
        "likelihood_class": RelativeBinningLikelihood,
        "approximant": approximant,
        "prior_class": "CartesianIASPrior",
    }

    likelihood_kwargs = {"fbin": fbin, "pn_phase_tol": None}
    ref_wf_finder_kwargs = {"time_range": (-1e-1, +1e-1), "f_ref": f_ref}

    coherent_posterior = Posterior.from_event(
        event=event_data,
        mchirp_guess=mchirp_guess,
        likelihood_kwargs=likelihood_kwargs,
        ref_wf_finder_kwargs=ref_wf_finder_kwargs,
        **posterior_kwargs,
    )
    par_dic_0 = coherent_posterior.likelihood.par_dic_0.copy()

    coherent_posterior.to_json(dirname=rundir)

    if load_inds and inds_path is not None and Path(inds_path).exists():
        print("Loading intrinsic samples indices")
        if inds_path.suffix == ".npz":
            data = np.load(inds_path)
            inds = data["inds"]
            incoherent_lnlikes = data["incoherent_lnlikes"]
        else:
            # Backward compatibility for old .npy format
            inds = np.load(inds_path)
            incoherent_lnlikes = None
        np.savez(
            rundir / "intrinsic_samples.npz",
            inds=inds,
            incoherent_lnlikes=incoherent_lnlikes,
        )
    else:
        print("Collecting intrinsic samples from individual detectors...")
        inds, incoherent_lnlikes = collect_int_samples_from_single_detectors(
            event_data=event_data,
            par_dic_0=par_dic_0,
            single_detector_blocksize=single_detector_blocksize,
            n_int=n_int,
            n_phi=n_phi,
            n_t=n_t,
            bank_folder=bank_folder,
            i_int_start=i_int_start,
            max_incoherent_lnlike_drop=max_incoherent_lnlike_drop,
        )
        np.savez(
            rundir / "intrinsic_samples.npz",
            inds=inds,
            incoherent_lnlikes=incoherent_lnlikes,
        )
    print(f"{len(inds)} intrinsic samples selected.")

    pr = coherent_posterior.prior

    cohernt_score_kwargs = {
        "min_n_effective_prior": coherent_score_min_n_effective_prior
    }
    print("Running magic integral...")
    (
        samples,
        ln_evidence,
        ln_evidence_discarded,
        n_effective,
        n_effective_i,
        n_effective_e,
        n_distance_marginalizations,
    ) = run_coherent_inference(
        event_data=event_data,
        rundir=rundir,
        par_dic_0=par_dic_0,
        bank_folder=bank_folder,
        n_int=n_int,
        inds=inds,
        n_ext=n_ext,
        n_phi=n_phi,
        m_arr=m_arr,
        blocksize=blocksize,
        pr=pr,
        seed=seed,
        size_limit=size_limit,
        draw_subset=draw_subset,
        n_draws=n_draws,
        coherent_score_kwargs=cohernt_score_kwargs,
        max_bestfit_lnlike_diff=max_bestfit_lnlike_diff,
    )
    print("Saving samples to file...")
    samples_path = Path(rundir) / "samples.feather"
    samples.to_feather(samples_path)
    print(f"Samples saved to:\n {samples_path}")

    summary_dict = {
        "n_effective": float(n_effective),
        "n_effective_i": float(n_effective_i),
        "n_effective_e": float(n_effective_e),
        "bestfit_lnlike_max": float(samples["bestfit_lnlike"].max()),
        "lnl_marginalized_max": float(samples["lnl_marginalized"].max()),
        "n_i_inds_used": int(len(inds)),
        "ln_evidence": float(ln_evidence),
        "ln_evidence_discarded": float(ln_evidence_discarded),
        "n_distance_marginalizations": int(n_distance_marginalizations),
    }

    # for injections, add the likelihood to the summary dict
    if getattr(event_data, "injection", None) is not None:
        clp = read_json(rundir / "CoherentLikelihoodProcessor.json")
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


def run_and_profile(**kwargs):
    """
    Run the magic integral and profile the run.
    """
    profiler = cProfile.Profile()
    profiler.enable()

    rundir = run(**kwargs)

    profiler.disable()

    # Save the printed profiler file & profiler report
    profile_obj_path = Path(rundir) / "profile_output.prof"
    profile_report_path = Path(rundir) / "profile_output.txt"

    profiler.dump_stats(profile_obj_path)

    with open(profile_report_path, "w", encoding="utf-8") as f:
        ps = pstats.Stats(profiler, stream=f)
        ps.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats()

    return rundir


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
        "--n_int", type=int, help="Number of intrinsic samples.", default=2**13
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
        "--n_phi", type=int, help="Number of phi_ref samples.", default=100
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

    return vars(parser.parse_args())


if __name__ == "__main__":
    kwargs = parse_arguments()
    run_and_profile(**kwargs)
