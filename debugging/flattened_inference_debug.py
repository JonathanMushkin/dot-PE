"""
Flattened version of inference.py for debugging.

This script inlines all the nested function calls from:
- run()
- collect_int_samples_from_single_detectors()
- run_for_single_detector()

So that all intermediate variables are in a single scope for inspection.
"""

import sys
import copy
import json
import warnings
from pathlib import Path
from typing import Dict, List, Union

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

from cogwheel import skyloc_angles
from cogwheel.data import EventData
from cogwheel.gw_utils import DETECTORS, get_geocenter_delays, get_fplus_fcross_0
from cogwheel.likelihood import RelativeBinningLikelihood
from cogwheel.posterior import Posterior
from cogwheel.utils import get_rundir, mkdirs, read_json
from cogwheel.waveform import WaveformGenerator
from dot_pe import config
from dot_pe.likelihood_calculating import LinearFree, get_shift
from dot_pe.utils import get_event_data
from dot_pe.single_detector import SingleDetectorProcessor


def flattened_run_debug(
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
) -> Dict:
    """
    Flattened version of run() that inlines collect_int_samples_from_single_detectors()
    and run_for_single_detector() so all intermediate variables are accessible.

    Returns a dictionary with all intermediate intrinsic_selection_results for inspection.
    """

    # ===== STEP 1: Paths and configuration setup =====
    print("Setting paths & loading configurations...")
    bank_folder = Path(bank_folder)

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

    # ===== STEP 2: Load event data and bank config =====
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

    # ===== STEP 3: Create COGWHEEL posterior objects =====
    # Set up cache directory for posterior
    cache_dir = Path(__file__).parent / "posterior_cache"
    mkdirs(cache_dir)
    cache_path = cache_dir / "Posterior.json"

    # Try to load from cache
    if cache_path.exists():
        print(f"Loading posterior from cache")
        coherent_posterior = read_json(cache_path)
        par_dic_0 = coherent_posterior.likelihood.par_dic_0.copy()
    else:
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
        # Save to cache directory
        coherent_posterior.to_json(dirname=str(cache_dir))
        print(f"Saved posterior to cache")

    # Also save to rundir for compatibility
    coherent_posterior.to_json(dirname=rundir)

    # ===== STEP 4: Check if loading pre-computed indices =====
    if load_inds and inds_path is not None and Path(inds_path).exists():
        print("Loading intrinsic samples indices")
        if inds_path.suffix == ".npz":
            data = np.load(inds_path)
            inds = data["inds"]
            incoherent_lnlikes = data["incoherent_lnlikes"]
        else:
            inds = np.load(inds_path)
            incoherent_lnlikes = None
        np.savez(
            rundir / "intrinsic_samples.npz",
            inds=inds,
            incoherent_lnlikes=incoherent_lnlikes,
        )
    else:
        # ===== STEP 5: Single detector processing (FLATTENED) =====
        print("Collecting intrinsic samples from individual detectors...")
        n_phi_incoherent = n_phi_incoherent if n_phi_incoherent is not None else n_phi

        # === 5a: Prepare indices ===
        if preselected_indices is not None:
            if isinstance(preselected_indices, (str, Path)):
                intrinsic_indices = np.load(preselected_indices)
            elif isinstance(preselected_indices, list):
                intrinsic_indices = np.array(preselected_indices, dtype=np.int_)
            elif isinstance(preselected_indices, np.ndarray):
                intrinsic_indices = preselected_indices
            else:
                raise TypeError(
                    f"preselected_indices must be array, list, str, Path, or None, got {type(preselected_indices)}"
                )
        else:
            intrinsic_indices = np.arange(i_int_start, i_int_start + n_int)

        # === 5b: Initialize storage for per-detector intrinsic_selection_results ===
        n_det = len(event_data.detector_names)
        lnlike_di = np.zeros((n_det, len(intrinsic_indices)))

        # === 5c: Process in batches (FLATTENED from collect_int_samples_from_single_detectors) ===
        for batch_idx, batch_start in enumerate(
            tqdm(
                range(0, len(intrinsic_indices), single_detector_blocksize),
                desc="Processing intrinsic batches",
                total=-(len(intrinsic_indices) // -single_detector_blocksize),
            )
        ):
            batch_end = min(
                batch_start + single_detector_blocksize, len(intrinsic_indices)
            )
            batch_intrinsic_indices = intrinsic_indices[batch_start:batch_end]

            # Initialize h_impb for this batch (shared across detectors)
            h_impb_batch = None

            # === 5d: Process each detector (FLATTENED from run_for_single_detector) ===
            for det_idx, det_name in enumerate(event_data.detector_names):
                # --- 5d(i): Prepare single-detector event data ---
                if isinstance(event_data, Path):
                    event_data_1d = EventData.from_npz(filename=event_data)
                else:
                    event_data_1d = copy.deepcopy(event_data)

                det_indices = [
                    event_data_1d.detector_names.index(det) for det in list(det_name)
                ]

                # Filter event data to single detector
                array_attributes = ["strain", "blued_strain", "wht_filter"]
                for attr in array_attributes:
                    setattr(
                        event_data_1d, attr, getattr(event_data_1d, attr)[det_indices]
                    )

                tuple_attributes = ["detector_names"]
                for attr in tuple_attributes:
                    temp = tuple(np.take(getattr(event_data_1d, attr), det_indices))
                    setattr(event_data_1d, attr, temp)

                if getattr(event_data_1d, "injection", None) is not None:
                    event_data_1d.injection["h_h"] = np.take(
                        event_data_1d.injection["h_h"], det_indices
                    ).tolist()
                    event_data_1d.injection["d_h"] = np.take(
                        event_data_1d.injection["d_h"], det_indices
                    ).tolist()

                # --- 5d(ii): Create waveform generator and likelihood ---
                wfg_1d = WaveformGenerator.from_event_data(event_data_1d, approximant)
                likelihood_linfree = LinearFree(event_data_1d, wfg_1d, par_dic_0, fbin)

                bank_file_path = Path(bank_folder) / "intrinsic_sample_bank.feather"
                waveform_dir = Path(bank_folder) / "waveforms"

                # --- 5d(iii): Create SingleDetectorProcessor ---
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

                # --- 5d(iv): Load waveforms if needed ---
                if h_impb_batch is None:
                    amp, phase = sdp.intrinsic_sample_processor.load_amp_and_phase(
                        waveform_dir, batch_intrinsic_indices
                    )
                    h_impb_batch = amp * np.exp(1j * phase)

                sdp.h_impb = h_impb_batch

                # --- 5d(v): Fix sky position to above detector ---
                # This is the critical transformation step
                det_name_inner = sdp.likelihood.event_data.detector_names[0]
                tgps_1d = sdp.likelihood.event_data.tgps
                lat, lon = skyloc_angles.cart3d_to_latlon(
                    skyloc_angles.normalize(DETECTORS[det_name_inner].location)
                )

                par_dic_transformed = sdp.transform_par_dic_by_sky_poisition(
                    det_name_inner, par_dic_0, lon, lat, tgps_1d
                )

                # --- 5d(vi): Compute time grid and timeshifts ---
                delay_single = get_geocenter_delays(
                    det_name_inner,
                    par_dic_transformed["lat"],
                    par_dic_transformed["lon"],
                )[0]
                tcoarse_1d = sdp.likelihood.event_data.tcoarse
                dt_sample = sdp.likelihood.event_data.times[1]
                t_grid_single = (np.arange(n_t) - n_t // 2) * dt_sample
                t_grid_single += (
                    par_dic_transformed["t_geocenter"] + tcoarse_1d + delay_single
                )

                timeshifts_dbt_single = np.exp(
                    -2j
                    * np.pi
                    * t_grid_single[None, None, :]
                    * sdp.likelihood.fbin[None, :, None]
                )

                # --- 5d(vii): Compute likelihoods ---
                # THIS IS THE KEY COMPUTATION
                r_iotp, lnlike_iot_single = sdp.get_response_over_distance_and_lnlike(
                    sdp.dh_weights_dmpb,
                    sdp.hh_weights_dmppb,
                    sdp.h_impb,
                    timeshifts_dbt_single,
                    sdp.likelihood.asd_drift,
                    sdp.likelihood_calculator.n_phi,
                    sdp.likelihood_calculator.m_arr,
                )

                # --- 5d(viii): Maximize over o (orbital phase) and t (time) ---
                lnlike_i_single = lnlike_iot_single.max(axis=(1, 2))

                # Store intrinsic_selection_results for this detector and batch
                lnlike_di[det_idx, batch_start:batch_end] = lnlike_i_single

                # Store intermediate intrinsic_selection_results for first batch of first detector
                if batch_idx == 0 and det_idx == 0:
                    debug_intrinsic_selection_results = {
                        "sdp": sdp,
                        "h_impb_batch": h_impb_batch,
                        "par_dic_0": par_dic_0,
                        "par_dic_transformed": par_dic_transformed,
                        "det_name": det_name_inner,
                        "lat": lat,
                        "lon": lon,
                        "t_grid_single": t_grid_single,
                        "timeshifts_dbt_single": timeshifts_dbt_single,
                        "r_iotp": r_iotp,
                        "lnlike_iot_single": lnlike_iot_single,
                        "lnlike_i_single": lnlike_i_single,
                        "batch_intrinsic_indices": batch_intrinsic_indices,
                        "dh_weights_dmpb": sdp.dh_weights_dmpb,
                        "hh_weights_dmppb": sdp.hh_weights_dmppb,
                        "asd_drift": sdp.likelihood.asd_drift,
                        "n_phi_calc": sdp.likelihood_calculator.n_phi,
                        "m_arr_calc": sdp.likelihood_calculator.m_arr,
                    }

        # === 5e: Aggregate intrinsic_selection_results across detectors ===
        incoherent_lnlikes = np.sum(lnlike_di, axis=0)
        incoherent_threshold = incoherent_lnlikes.max() - max_incoherent_lnlike_drop
        selected = incoherent_lnlikes >= incoherent_threshold

        inds = intrinsic_indices[selected]
        lnlike_di_selected = lnlike_di[:, selected]
        incoherent_lnlikes_selected = incoherent_lnlikes[selected]

        np.savez(
            rundir / "intrinsic_samples.npz",
            inds=inds,
            lnlikes_di=lnlike_di_selected,
            incoherent_lnlikes=incoherent_lnlikes_selected,
        )

    print(f"{len(inds)} intrinsic samples selected.")

    # Add intrinsic_selection_results to debug dictionary
    debug_intrinsic_selection_results.update(
        {
            "inds": inds,
            "lnlike_di": lnlike_di_selected if not load_inds else None,
            "incoherent_lnlikes": incoherent_lnlikes_selected
            if not load_inds
            else incoherent_lnlikes,
            "rundir": rundir,
            "event_data": event_data,
            "bank_config": bank_config,
            "coherent_posterior": coherent_posterior,
        }
    )

    return debug_intrinsic_selection_results


if __name__ == "__main__":
    # Example usage
    event = Path("test_event.npz")
    bank_folder = Path("test_bank")
    n_int = 2**12
    n_ext = 512
    n_phi = 32
    n_t = 64
    blocksize = 1024
    single_detector_blocksize = 1024
    i_int_start = 0
    seed = 42
    load_inds = False
    inds_path = None
    size_limit = 10**6
    draw_subset = False
    n_draws = None
    event_dir = Path("test_event")
    rundir = None
    coherent_score_min_n_effective_prior = 100
    max_incoherent_lnlike_drop = 1000
    max_bestfit_lnlike_diff = 20
    mchirp_guess = None
    extrinsic_samples = None
    n_phi_incoherent = 32
    preselected_indices = None
    coherent_posterior_kwargs = {}

    intrinsic_selection_results = flattened_run_debug(
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
        coherent_posterior_kwargs=coherent_posterior_kwargs,
    )

    print("\n" + "=" * 80)
    print("DEBUG RESULTS SUMMARY")
    print("=" * 80)
    print(
        f"Number of selected intrinsic samples: {len(intrinsic_selection_results['inds'])}"
    )
    print(f"First detector processed: {intrinsic_selection_results['det_name']}")
    print(
        f"Transformed t_geocenter: {intrinsic_selection_results['par_dic_transformed']['t_geocenter']:.6f}"
    )
    print(
        f"Transformed psi: {intrinsic_selection_results['par_dic_transformed']['psi']:.6f}"
    )
    print(
        f"Transformed d_luminosity: {intrinsic_selection_results['par_dic_transformed']['d_luminosity']:.6f}"
    )
    print(f"r_iotp shape: {intrinsic_selection_results['r_iotp'].shape}")
    print(
        f"lnlike_iot_single shape: {intrinsic_selection_results['lnlike_iot_single'].shape}"
    )
    print(
        f"First 10 lnlike_i values: {intrinsic_selection_results['lnlike_i_single'][:10]}"
    )
    print(
        f"First 10 incoherent_lnlikes: {intrinsic_selection_results['incoherent_lnlikes'][:10]}"
    )
    print("\nAll intermediate variables are in the returned dictionary!")

    # ===== Comparison with coherent_posterior =====
    print("\n" + "=" * 80)
    print("LIKELIHOOD TRACEABILITY COMPARISON")
    print("=" * 80)

    # Load bank config and intrinsic bank once
    bank_config_dict = read_json(bank_folder / "bank_config.json")
    # Convert fbin and m_arr from lists to numpy arrays
    bank_config_dict["fbin"] = np.array(bank_config_dict["fbin"])
    bank_config_dict["m_arr"] = np.array(bank_config_dict["m_arr"])
    intrinsic_bank_df = pd.read_feather(bank_folder / "intrinsic_sample_bank.feather")

    # Number of samples to compare
    n_comparisons = 1

    # Compare for multiple samples
    for i_idx in range(n_comparisons):
        print(f"\n{'=' * 80}")
        print(f"Comparison for sample {i_idx + 1}/{n_comparisons}")
        print(f"{'=' * 80}")

        # Get the max likelihood point
        max_idx = np.unravel_index(
            intrinsic_selection_results["lnlike_iot_single"][i_idx].argmax(),
            intrinsic_selection_results["lnlike_iot_single"][i_idx].shape,
        )
        o_max, t_max = max_idx
        r_max = intrinsic_selection_results["r_iotp"][i_idx, o_max, t_max]

        # Reconstruct psi and d_luminosity from r
        psi_recon, d_luminosity_recon = intrinsic_selection_results[
            "sdp"
        ].bestfit_response_to_psi_and_d_luminosity(
            r_max, intrinsic_selection_results["det_name"]
        )

        # Reconstruct other parameters
        phi_ref_recon = o_max / n_phi_incoherent * 2 * np.pi
        t_geocenter_recon = (
            intrinsic_selection_results["par_dic_transformed"]["t_geocenter"]
            + intrinsic_selection_results["t_grid_single"][t_max]
            - intrinsic_selection_results["sdp"].likelihood.event_data.tcoarse
            - get_geocenter_delays(
                intrinsic_selection_results["det_name"],
                intrinsic_selection_results["lat"],
                intrinsic_selection_results["lon"],
            )[0]
        )

        # Get intrinsic parameters from bank
        intrinsic_params = intrinsic_bank_df.iloc[
            intrinsic_selection_results["batch_intrinsic_indices"][i_idx]
        ].to_dict()
        intrinsic_params["f_ref"] = bank_config_dict["f_ref"]
        intrinsic_params["l1"] = 0
        intrinsic_params["l2"] = 0

        # Combine all parameters
        reconstructed_par_dic = intrinsic_params | {
            "ra": intrinsic_selection_results["par_dic_transformed"]["ra"],
            "dec": intrinsic_selection_results["par_dic_transformed"]["dec"],
            "t_geocenter": t_geocenter_recon,
            "phi_ref": phi_ref_recon,
            "psi": psi_recon,
            "d_luminosity": d_luminosity_recon,
        }

        print("Reconstructed parameters:")
        print(
            f"  psi: {psi_recon:.6f} (transformed had: {intrinsic_selection_results['par_dic_transformed']['psi']:.6f})"
        )
        print(
            f"  d_luminosity: {d_luminosity_recon:.6f} (transformed had: {intrinsic_selection_results['par_dic_transformed']['d_luminosity']:.6f})"
        )
        print(f"  phi_ref: {phi_ref_recon:.6f}")
        print(f"  t_geocenter: {t_geocenter_recon:.6f}")
        print(f"\nDEBUG t_geocenter reconstruction:")
        print(
            f"  par_dic_transformed['t_geocenter']: {intrinsic_selection_results['par_dic_transformed']['t_geocenter']:.6f}"
        )
        print(
            f"  t_grid_single[{t_max}]: {intrinsic_selection_results['t_grid_single'][t_max]:.6f}"
        )
        print(
            f"  tcoarse: {intrinsic_selection_results['sdp'].likelihood.event_data.tcoarse:.6f}"
        )
        print(
            f"  delay: {get_geocenter_delays(intrinsic_selection_results['det_name'], intrinsic_selection_results['lat'], intrinsic_selection_results['lon'])[0]:.6f}"
        )

        # Now compute likelihood using these reconstructed parameters
        print("\nComputing likelihoods...")
        lnlike_from_params = intrinsic_selection_results[
            "coherent_posterior"
        ].likelihood.lnlike(reconstructed_par_dic)
        lnlike_from_barebones = intrinsic_selection_results["lnlike_iot_single"][
            i_idx, o_max, t_max
        ]

        print(f"\nLikelihood comparison:")
        print(f"  From coherent_posterior.lnlike: {lnlike_from_params:.6f}")
        print(f"  From barebones calculation: {lnlike_from_barebones:.6f}")
        print(f"\nDifferences:")
        print(
            f"  coherent vs barebones: {lnlike_from_params - lnlike_from_barebones:.6f}"
        )

        if abs(lnlike_from_params - lnlike_from_barebones) < 0.01:
            print("\n[PASS] Likelihoods match! Traceability verified.")
        else:
            print("\n[FAIL] Likelihoods don't match. Need to investigate further.")

    print(f"\n{'=' * 80}")
    print(f"Completed {n_comparisons} comparison(s)")
    print(f"{'=' * 80}")
