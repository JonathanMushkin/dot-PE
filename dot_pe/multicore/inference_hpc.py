"""
HPC multi-core parallelized inference pipeline.

Main entry point: run_hpc() - drop-in replacement for inference.run()
with identical signature and outputs, optimized for 20-100+ core systems.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from dot_pe.inference import (
    prepare_run_objects,
    select_intrinsic_samples_across_banks_by_incoherent_likelihood,
    draw_extrinsic_samples,
    run_coherent_inference_per_bank,
    aggregate_and_save_results,
)
from cogwheel.data import EventData
from cogwheel.utils import exp_normalize
from dot_pe.utils import inds_to_blocks, safe_logsumexp
from dot_pe.base_sampler_free_sampling import get_n_effective_total_i_e
from dot_pe.coherent_processing import CoherentLikelihoodProcessor
from dot_pe.likelihood_calculating import LinearFree
from cogwheel.waveform import WaveformGenerator

from .single_detector_hpc import collect_int_samples_from_single_detectors_hpc
from .coherent_processing_hpc import create_likelihood_blocks_hpc
from .config import HPCConfig, load_cached_config
from .utils_hpc import get_machine_info


def select_intrinsic_samples_per_bank_incoherently_hpc(
    *,
    banks: Dict[str, Path],
    event_data: Union[EventData, str, Path],
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
    hpc_config: HPCConfig,
) -> Tuple[
    Dict[str, np.ndarray],
    Optional[Dict[str, np.ndarray]],
    Optional[Dict[str, np.ndarray]],
]:
    """
    HPC parallelized version of select_intrinsic_samples_per_bank_incoherently.

    Uses multiprocessing to parallelize single-detector likelihood evaluation
    across banks and batches.
    """
    # For now, use the original function but with HPC single-detector processing
    # In the future, we could parallelize across banks too
    candidate_inds_by_bank = {}
    lnlikes_by_bank = {}
    lnlikes_di_by_bank = {}

    for bank_id, bank_path in banks.items():
        print(f"\nProcessing bank: {bank_id}")
        bank_rundir = banks_dir / bank_id
        bank_rundir.mkdir(exist_ok=True)

        n_int_k = n_int_dict[bank_id]

        if load_inds and inds_path_dict and bank_id in inds_path_dict:
            print(f"Loading intrinsic samples from {inds_path_dict[bank_id]}")
            loaded = np.load(inds_path_dict[bank_id], allow_pickle=True)
            inds = loaded["inds"]
            lnlikes_di = loaded.get("lnlikes_di")
            incoherent_lnlikes = loaded.get("incoherent_lnlikes")
        else:
            print(f"Collecting intrinsic samples for bank {bank_id}...")
            n_phi_incoherent = n_phi_incoherent if n_phi_incoherent is not None else n_phi

            # Use HPC parallelized version
            inds, lnlikes_di, incoherent_lnlikes = collect_int_samples_from_single_detectors_hpc(
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
                n_procs=hpc_config.n_procs,
                i_batch=hpc_config.i_batch,
                batches_per_task=hpc_config.batches_per_task,
            )

        candidate_inds_by_bank[bank_id] = inds
        lnlikes_by_bank[bank_id] = incoherent_lnlikes
        if lnlikes_di is not None:
            lnlikes_di_by_bank[bank_id] = lnlikes_di
        print(f"Bank {bank_id}: {len(inds)} intrinsic samples evaluated.")

    return candidate_inds_by_bank, lnlikes_by_bank, lnlikes_di_by_bank


def run_coherent_inference_hpc(
    event_data: EventData,
    bank_rundir: Path,
    top_rundir: Path,
    par_dic_0: Dict,
    bank_folder: Union[str, Path],
    n_total_samples: int,
    inds: np.ndarray,
    n_ext: int,
    n_phi: int,
    m_arr: np.ndarray,
    blocksize: int,
    renormalize_log_prior_weights_i: bool = False,
    intrinsic_logw_lookup=None,
    size_limit: int = 10**7,
    max_bestfit_lnlike_diff: float = 20,
    event_data_path: Optional[Union[str, Path]] = None,
    n_procs: Optional[int] = None,
    pairs_per_task: int = 4,
) -> Tuple[float, float, float, float, float, int]:
    """
    HPC parallelized coherent inference: same as run_coherent_inference but
    uses create_likelihood_blocks_hpc for parallel block creation.
    Requires event_data_path (str or Path) for worker processes; if None,
    uses getattr(event_data, 'path', None).
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
    clp.load_extrinsic_samples_data(top_rundir)
    clp.to_json(bank_rundir, overwrite=True)

    path_for_workers = event_data_path or getattr(event_data, "path", None)
    if path_for_workers is None:
        raise ValueError(
            "run_coherent_inference_hpc requires event_data_path or event_data.path for worker processes"
        )

    print(f"Creating {len(i_blocks)} x {len(e_blocks)} likelihood blocks (HPC)...")
    _ = create_likelihood_blocks_hpc(
        clp,
        tempdir=bank_rundir,
        i_blocks=i_blocks,
        e_blocks=e_blocks,
        event_data_path=path_for_workers,
        top_rundir=top_rundir,
        bank_folder=bank_folder,
        par_dic_0=par_dic_0,
        n_phi=n_phi,
        m_arr=m_arr,
        blocksize=blocksize,
        n_ext=n_ext,
        inds=inds,
        size_limit=size_limit,
        max_bestfit_lnlike_diff=max_bestfit_lnlike_diff,
        renormalize_log_prior_weights_i=renormalize_log_prior_weights_i,
        intrinsic_logw_lookup=intrinsic_logw_lookup,
        n_procs=n_procs,
        pairs_per_task=pairs_per_task,
    )

    clp.prob_samples["weights"] = exp_normalize(clp.prob_samples["ln_posterior"].values)
    clp.prob_samples.to_feather(bank_rundir / "prob_samples.feather")

    cache_path = bank_rundir / "intrinsic_sample_processor_cache.json"
    cache_dict = {
        int(k): float(v)
        for k, v in clp.intrinsic_sample_processor.cached_dt_linfree_relative.items()
    }
    with open(cache_path, "w") as f:
        json.dump(cache_dict, f)

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


def run_coherent_inference_per_bank_hpc(
    *,
    banks: Dict[str, Path],
    event_data: EventData,
    event_path: Optional[Union[str, Path]],
    rundir: Path,
    banks_dir: Path,
    par_dic_0: Dict,
    selected_inds_by_bank: Dict[str, np.ndarray],
    n_int_dict: Dict[str, int],
    n_ext: int,
    n_phi: int,
    m_arr: np.ndarray,
    blocksize: int,
    size_limit: int,
    max_bestfit_lnlike_diff: float,
    bank_logw_override_dict: Optional[Dict],
    n_procs: Optional[int] = None,
    pairs_per_task: int = 4,
) -> List[Dict[str, Any]]:
    """Run coherent inference per bank using HPC parallel block creation."""
    if event_path is None:
        return run_coherent_inference_per_bank(
            banks=banks,
            event_data=event_data,
            rundir=rundir,
            banks_dir=banks_dir,
            par_dic_0=par_dic_0,
            selected_inds_by_bank=selected_inds_by_bank,
            n_int_dict=n_int_dict,
            n_ext=n_ext,
            n_phi=n_phi,
            m_arr=m_arr,
            blocksize=blocksize,
            size_limit=size_limit,
            max_bestfit_lnlike_diff=max_bestfit_lnlike_diff,
            bank_logw_override_dict=bank_logw_override_dict,
        )

    print("\n=== Coherent inference per bank (HPC) ===")
    bank_results = []
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
        ) = run_coherent_inference_hpc(
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
            event_data_path=event_path,
            n_procs=n_procs,
            pairs_per_task=pairs_per_task,
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

    return bank_results


def run_hpc(
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
    preselected_indices: Union[np.ndarray, List[int], str, Path, None] = None,
    bank_logw_override: Union[
        Dict[str, Union[np.ndarray, List[float], pd.Series]],
        np.ndarray,
        List[float],
        pd.Series,
        None,
    ] = None,
    coherent_posterior_kwargs: Dict = {},
    hpc_config: Optional[HPCConfig] = None,
    n_procs: Optional[int] = None,
    i_batch: Optional[int] = None,
    batches_per_task: Optional[int] = None,
) -> Path:
    """
    HPC parallelized version of inference.run().

    Identical signature and outputs to inference.run(), but uses multiprocessing
    to parallelize computation across multiple cores. Optimized for 20-100+ core systems.

    Additional Parameters
    ---------------------
    hpc_config : Optional[HPCConfig]
        HPC configuration. If None, will attempt to load from cache or use defaults.
    n_procs : Optional[int]
        Number of worker processes (overrides hpc_config.n_procs if provided)
    i_batch : Optional[int]
        Batch size for intrinsic samples (overrides hpc_config.i_batch if provided)
    batches_per_task : Optional[int]
        Batches per worker task (overrides hpc_config.batches_per_task if provided)

    Returns
    -------
    Path
        Path to rundir (identical to inference.run())
    """
    # Load or create HPC config
    if hpc_config is None:
        cached_config = load_cached_config()
        if cached_config is None:
            # Use defaults based on machine info
            machine_info = get_machine_info()
            cpu_count = machine_info.get("cpu_count", 4)
            hpc_config = HPCConfig(
                n_procs=n_procs or min(cpu_count, 32),
                i_batch=i_batch or single_detector_blocksize,
                batches_per_task=batches_per_task or 1,
            )
        else:
            hpc_config = cached_config

    # Override config with explicit parameters if provided
    if n_procs is not None:
        hpc_config.n_procs = n_procs
    if i_batch is not None:
        hpc_config.i_batch = i_batch
    if batches_per_task is not None:
        hpc_config.batches_per_task = batches_per_task

    print("\n=== HPC Multi-core Inference ===")
    print(f"HPC Config: n_procs={hpc_config.n_procs}, i_batch={hpc_config.i_batch}, "
          f"batches_per_task={hpc_config.batches_per_task}")

    # Step 1: Prepare shared objects (same as original)
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

    # Step 2: Incoherent selection per bank (HPC parallelized)
    # Convert event to path if it's an EventData object (for multiprocessing)
    event_for_hpc = event
    if isinstance(event, EventData):
        # Try to get path from event_data, or use event_dir
        event_path = getattr(ctx["event_data"], "path", None)
        if event_path is None and event_dir:
            event_path = Path(event_dir)
        elif event_path is None:
            raise ValueError(
                "For HPC multiprocessing, pass event as a path (str/Path), not EventData object. "
                "Or ensure event_data has a 'path' attribute or event_dir is provided."
            )
        event_for_hpc = event_path

    candidate_inds_by_bank, lnlikes_by_bank, lnlikes_di_by_bank = (
        select_intrinsic_samples_per_bank_incoherently_hpc(
            banks=ctx["banks"],
            event_data=event_for_hpc,  # Pass path, not EventData object
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
            hpc_config=hpc_config,
        )
    )

    # Step 3: Cross-bank selection (same as original - not parallelized)
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

    # Step 4: Draw extrinsic samples (same as original - not parallelized)
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

    # Step 5: Coherent inference per bank (HPC parallel block creation)
    event_path = (
        event if isinstance(event, (str, Path)) else getattr(ctx["event_data"], "path", None)
    )
    per_bank_results = run_coherent_inference_per_bank_hpc(
        banks=ctx["banks"],
        event_data=ctx["event_data"],
        event_path=event_path,
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
        n_procs=hpc_config.n_procs,
        pairs_per_task=getattr(hpc_config, "pairs_per_task", 4),
    )

    # Step 6: Aggregate and save (same as original)
    return aggregate_and_save_results(
        per_bank_results=per_bank_results,
        banks=ctx["banks"],
        rundir=ctx["rundir"],
        banks_dir=ctx["banks_dir"],
        event_data=ctx["event_data"],
        n_phi=n_phi,
        pr=ctx["pr"],
        n_draws=n_draws,
        draw_subset=draw_subset,
    )


def run_and_profile_hpc(
    *args,
    hpc_config: Optional[HPCConfig] = None,
    **kwargs
) -> Tuple[Path, Any]:
    """
    HPC version of inference.run_and_profile().

    Note: Profiling with multiprocessing can be complex. This is a placeholder
    that calls run_hpc() and returns a basic profile. For detailed profiling,
    use the original inference.run_and_profile() or profile individual components.
    """
    import cProfile
    import pstats
    import io

    profiler = cProfile.Profile()
    profiler.enable()

    try:
        rundir = run_hpc(*args, hpc_config=hpc_config, **kwargs)
    finally:
        profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats("cumulative")
    ps.print_stats()

    return rundir, s.getvalue()
