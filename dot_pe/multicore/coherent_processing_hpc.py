"""
HPC parallelized coherent likelihood block processing.

Workers create (i_block, e_block) likelihood blocks with a fixed threshold,
write block data to disk. Main process loads blocks in order and combines
sequentially via combine_block_dict_into_prob_samples.
"""

import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from cogwheel.waveform import WaveformGenerator
from tqdm import tqdm

from dot_pe.coherent_processing import CoherentLikelihoodProcessor
from dot_pe.likelihood_calculating import LinearFree
from dot_pe.utils import get_event_data, inds_to_blocks

from .utils_hpc import init_worker, partition_block_pairs

# Per-process cache set by init_coherent_worker; read by _worker_create_blocks_and_save.
_coherent_worker_state: Dict[str, Any] = {}


def init_coherent_worker(init_args: Tuple) -> None:
    """
    Build CLP and block index arrays once per worker.
    Called once per process by Pool(initializer=..., initargs=(init_args,)).
    """
    init_worker()
    (
        event_data_path,
        top_rundir,
        bank_folder,
        par_dic_0,
        n_phi,
        m_arr,
        blocksize,
        n_ext,
        inds,
        size_limit,
        max_bestfit_lnlike_diff,
        renormalize_log_prior_weights_i,
        intrinsic_logw_lookup,
        initial_min_bestfit_lnlike_to_keep,
    ) = init_args

    bank_folder = Path(bank_folder)
    top_rundir = Path(top_rundir)
    waveform_dir = bank_folder / "waveforms"
    bank_file_path = bank_folder / "intrinsic_sample_bank.feather"

    with open(bank_folder / "bank_config.json", "r", encoding="utf-8") as f:
        bank_config = json.load(f)
        fbin = np.array(bank_config["fbin"])
        approximant = bank_config["approximant"]

    event_data = get_event_data(event_data_path)
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
    clp.load_extrinsic_samples_data(top_rundir)
    clp.min_bestfit_lnlike_to_keep = initial_min_bestfit_lnlike_to_keep

    i_blocks = inds_to_blocks(inds, blocksize)
    e_blocks = inds_to_blocks(np.arange(n_ext), blocksize)

    _coherent_worker_state["clp"] = clp
    _coherent_worker_state["i_blocks"] = i_blocks
    _coherent_worker_state["e_blocks"] = e_blocks
    _coherent_worker_state["waveform_dir"] = waveform_dir


def _worker_create_blocks_and_save(
    args: Tuple[List[Tuple[int, int]], Path, bool]
) -> Tuple[List[Tuple[int, int, str]], int]:
    """
    Process task_pairs using cached CLP; save each block to tempdir.
    Expects init_coherent_worker to have run first in this process.
    """
    task_pairs, tempdir, compress_blocks = args
    tempdir = Path(tempdir)

    clp = _coherent_worker_state["clp"]
    i_blocks = _coherent_worker_state["i_blocks"]
    e_blocks = _coherent_worker_state["e_blocks"]
    waveform_dir = _coherent_worker_state["waveform_dir"]

    result_files: List[Tuple[int, int, str]] = []
    n_distance_marginalizations_delta = 0

    for i_idx, e_idx in task_pairs:
        i_block = i_blocks[i_idx]
        e_block = e_blocks[e_idx]
        response_subset = clp.full_response_dpe[..., e_block]
        timeshift_subset = clp.full_timeshift_dbe[..., e_block]

        amp_impb, phase_impb = clp.intrinsic_sample_processor.load_amp_and_phase(
            waveform_dir, i_block
        )
        h_impb = amp_impb * np.exp(1j * phase_impb)

        n_before = clp.n_distance_marginalizations
        clp.create_a_likelihood_block(
            h_impb,
            response_subset,
            timeshift_subset,
            i_block,
            e_block,
        )
        n_distance_marginalizations_delta += clp.n_distance_marginalizations - n_before

        block_dict = clp._next_block

        fname = f"block_{i_idx}_{e_idx}_data.npz"
        outpath = tempdir / fname

        if (
            block_dict
            and isinstance(block_dict, dict)
            and "bestfit_lnlike_k" in block_dict
        ):
            save_dict = {k: np.asarray(v) for k, v in block_dict.items()}
            if compress_blocks:
                np.savez_compressed(outpath, **save_dict)
            else:
                np.savez(outpath, **save_dict)
        else:
            if compress_blocks:
                np.savez_compressed(outpath, empty=np.array(True))
            else:
                np.savez(outpath, empty=np.array(True))

        result_files.append((i_idx, e_idx, fname))

    return (result_files, n_distance_marginalizations_delta)


def create_likelihood_blocks_hpc(
    clp: Any,
    tempdir: Path,
    i_blocks: List[np.ndarray],
    e_blocks: List[np.ndarray],
    event_data_path: Optional[Path] = None,
    event_data: Any = None,
    top_rundir: Optional[Path] = None,
    bank_folder: Optional[Path] = None,
    par_dic_0: Optional[Dict] = None,
    n_phi: int = 50,
    m_arr: Optional[np.ndarray] = None,
    blocksize: int = 512,
    n_ext: int = 0,
    inds: Optional[np.ndarray] = None,
    size_limit: int = 10**7,
    max_bestfit_lnlike_diff: float = 20,
    renormalize_log_prior_weights_i: bool = False,
    intrinsic_logw_lookup: Optional[Tuple] = None,
    n_procs: Optional[int] = None,
    pairs_per_task: int = 4,
    compress_blocks: bool = False,
) -> List[str]:
    """
    Create likelihood blocks in parallel; main process combines in order.

    pairs_per_task: on 40–100+ cores, increase so each task is ~100–500 ms.

    Workers use a per-process CLP built once in init_coherent_worker; each task
    receives only (task_pairs, tempdir, compress_blocks). Block dicts are written
    to tempdir/block_{i}_{e}_data.npz (uncompressed by default; set compress_blocks=True
    to save disk). Main process loads blocks in (i_block, e_block) order and
    calls clp.combine_block_dict_into_prob_samples().
    """
    tempdir = Path(tempdir)
    task_groups = partition_block_pairs(i_blocks, e_blocks, pairs_per_task)

    if n_procs is None:
        n_procs = min(len(task_groups), mp.cpu_count() or 4)

    if event_data_path is None and event_data is not None:
        event_data_path = getattr(event_data, "path", None)
    if event_data_path is None or top_rundir is None or bank_folder is None:
        raise ValueError(
            "create_likelihood_blocks_hpc requires event_data_path, top_rundir, bank_folder (and par_dic_0, inds, n_ext, m_arr) for workers"
        )
    if inds is None:
        inds = np.concatenate(i_blocks)
    if m_arr is None:
        m_arr = clp.m_arr

    initial_min = clp.min_bestfit_lnlike_to_keep

    init_args = (
        str(event_data_path),
        str(top_rundir),
        str(bank_folder),
        par_dic_0,
        n_phi,
        m_arr,
        blocksize,
        n_ext,
        inds,
        size_limit,
        max_bestfit_lnlike_diff,
        renormalize_log_prior_weights_i,
        intrinsic_logw_lookup,
        initial_min,
    )

    worker_args = [(task_group, tempdir, compress_blocks) for task_group in task_groups]

    if n_procs > 1 and len(task_groups) > 1:
        with mp.Pool(
            processes=n_procs,
            initializer=init_coherent_worker,
            initargs=(init_args,),
        ) as pool:
            task_results = list(
                pool.imap_unordered(
                    _worker_create_blocks_and_save, worker_args, chunksize=1
                )
            )
    else:
        _coherent_worker_state.clear()
        init_coherent_worker(init_args)
        task_results = [_worker_create_blocks_and_save(a) for a in worker_args]

    block_files: List[Tuple[int, int, str]] = []
    n_distance_total = 0
    for (files, n_dist) in task_results:
        block_files.extend(files)
        n_distance_total += n_dist

    block_files.sort(key=lambda x: (x[0], x[1]))

    for i_idx, e_idx, fname in tqdm(
        block_files, desc="Coherent likelihood blocks (combine)", leave=True
    ):
        data = np.load(tempdir / fname, allow_pickle=True)
        if "empty" in data and data["empty"].item():
            continue
        block_dict = {k: data[k] for k in data.files if k != "empty"}
        clp.combine_block_dict_into_prob_samples(block_dict)

    clp.n_distance_marginalizations = n_distance_total

    return [fname for (_, _, fname) in block_files]
