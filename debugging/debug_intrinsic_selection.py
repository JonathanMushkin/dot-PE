"""
Debugging script that replicates inference.py but stops after intrinsic sample selection (line 811).
"""

import argparse
import copy
import cProfile
import json
import os
import pstats
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from lal import GreenwichMeanSiderealTime
from numpy.typing import NDArray
from tqdm import tqdm

# Add parent directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

from cogwheel import skyloc_angles
from cogwheel.data import EventData
from cogwheel.gw_utils import DETECTORS, get_geocenter_delays
from cogwheel.likelihood import RelativeBinningLikelihood, LookupTable
from cogwheel.posterior import Posterior
from cogwheel.utils import exp_normalize, get_rundir, mkdirs, read_json
from cogwheel.waveform import WaveformGenerator
from cogwheel.prior import Prior
from dot_pe.base_sampler_free_sampling import (
    get_n_effective_total_i_e,
)
from dot_pe.likelihood_calculating import LinearFree
from dot_pe.marginalization import MarginalizationExtrinsicSamplerFreeLikelihood
from dot_pe.coherent_processing import (
    CoherentLikelihoodProcessor,
    CoherentExtrinsicSamplesGenerator,
)
from dot_pe.utils import get_event_data, safe_logsumexp
from dot_pe.single_detector import SingleDetectorProcessor
from dot_pe.sample_processing import IntrinsicSampleProcessor, ExtrinsicSampleProcessor


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
    """
    bank_folder = Path(bank_folder)
    with open(bank_folder / "bank_config.json", "r", encoding="utf-8") as f:
        bank_config = json.load(f)
        fbin = np.array(bank_config["fbin"])
        approximant = bank_config["approximant"]
        m_arr = np.array(bank_config["m_arr"])
    # do single detector pe for each detector
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
    lnlike_di = lnlike_di[:, selected]
    incoherent_lnlikes = incoherent_lnlikes[selected]
    return inds, lnlike_di, incoherent_lnlikes


def run(
    event: Union[str, Path, EventData],
    bank_folder: Union[str, Path],
    n_int: int,
    n_ext: int,
    n_phi: int,
    n_t: int,
    blocksize: int = 512,
    single_detector_blocksize: int = 512,
    i_int_start: int = 0,
    seed: int = None,
    load_inds: bool = False,
    inds_path: Union[Path, str] = None,
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
    coherent_posterior_kwargs: Dict = {},
) -> Path:
    """Run the magic integral for a given event and bank folder."""
    bank_folder = Path(bank_folder)
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
                "extrinsic_samples": str(extrinsic_samples)
                if extrinsic_samples is not None
                else None,
                "n_phi_incoherent": int(n_phi_incoherent)
                if n_phi_incoherent is not None
                else None,
                "preselected_indices": str(preselected_indices)
                if isinstance(preselected_indices, (str, Path))
                else ("array" if preselected_indices is not None else None),
            },
            fp,
            indent=4,
        )
    # set paths and basic configs
    event_data = get_event_data(event)
    bank_config_path = Path(bank_folder) / "bank_config.json"
    if isinstance(inds_path, str):
        inds_path = Path(inds_path)
    with open(bank_config_path, "r", encoding="utf-8") as fp:
        bank_config = json.load(fp)
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
        # Use n_phi_incoherent for single detector evaluation (thresholding)
        n_phi_incoherent = n_phi_incoherent if n_phi_incoherent is not None else n_phi
        inds, lnlikes_di, incoherent_lnlikes = (
            collect_int_samples_from_single_detectors(
                event_data=event_data,
                par_dic_0=par_dic_0,
                single_detector_blocksize=single_detector_blocksize,
                n_int=n_int,
                n_phi=n_phi_incoherent,
                n_t=n_t,
                bank_folder=bank_folder,
                i_int_start=i_int_start,
                max_incoherent_lnlike_drop=max_incoherent_lnlike_drop,
                preselected_indices=preselected_indices,
            )
        )
        np.savez(
            rundir / "intrinsic_samples.npz",
            inds=inds,
            lnlikes_di=lnlikes_di,
            incoherent_lnlikes=incoherent_lnlikes,
        )
    print(f"{len(inds)} intrinsic samples selected.")

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
    run(**kwargs)
