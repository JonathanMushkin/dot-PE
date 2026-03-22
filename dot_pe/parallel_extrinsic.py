"""Fork-based parallel extrinsic marginalization-info collection.

Both MP and swarm paths use identical fork+copy-on-write (COW) logic
via this module.

Public API
----------
collect_marg_info_parallel
    Parallel replacement for get_marg_info_multibank().
draw_extrinsic_samples_parallel
    Drop-in for inference.draw_extrinsic_samples().

Design
------
Workers are partitioned by waveform *block* rather than by random
sample batches.  Each worker receives samples from a contiguous range
of block files, limiting it to O(n_blocks / n_workers) file reads
instead of all blocks.

Example: for waveform arrays of shape
(blocksize, n_modes, n_pol, n_fbin) = (4096, 4, 2, 378) float64,
one block file is ~99 MB; loading all blocks (amp + phase) totals
~12 GB per worker with random batches.  Block-partitioned workers
each read only ~12 GB / n_workers.

Workers process their samples in sub-batches of SUB_BATCH_SIZE and
share a multiprocessing.Value counter (_accepted_count) so they
collectively stop once n_combine MarginalizationInfo objects have
been accepted across all workers, avoiding unnecessary work.

After all workers finish, the n_combine MarginalizationInfo objects
with the highest n_effective_prior are selected from the pool.
"""

import ctypes
import gc
import json
import pickle
from copy import deepcopy
from multiprocessing import Lock, Pool, Value
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

from cogwheel.waveform import WaveformGenerator
from dot_pe.coherent_processing import CoherentExtrinsicSamplesGenerator
from dot_pe.marginalization import (
    MarginalizationExtrinsicSamplerFreeLikelihood,
)

# Module-level globals: set in main process before Pool creation,
# inherited read-only by fork-based workers via copy-on-write (data)
# or shared memory (Value/Lock).
_ext_setup = None
_accepted_count = None  # multiprocessing.Value('i', 0)
_count_lock = None      # multiprocessing.Lock()
_n_target = None        # stop when _accepted_count.value >= _n_target

# Samples processed per sub-batch inside each worker.
# Smaller value = lower peak resident set size (RSS) per worker
# (~linear scaling with sub-batch size).
# Root cause: each get_marg_info_batch_multibank() call accumulates
# MarginalizationInfo objects via np.concatenate inside
# MarginalizationInfo.update(); freed pages pile up in the heap
# (Python does not return them to the OS), creating a high-water mark.
# Measured with waveform shape (4096, 4, 2, 378) float64 per block:
#   sub-batch size 1024 -> peak +7.6 GB/worker
#   sub-batch size  128 -> peak +1.1 GB/worker  (7x smaller)
SUB_BATCH_SIZE = 128


def _extrinsic_batch_worker(args):
    """Process sub-batches; stop once the global target is reached."""
    batch_sample_idx, batch_bank_idx = args
    all_mi, all_b, all_s = [], [], []

    n = len(batch_sample_idx)
    for start in range(0, n, SUB_BATCH_SIZE):
        # Check global stop before loading waveforms for this sub-batch.
        # Reading Value without the lock is safe: a stale read at
        # worst costs one extra sub-batch of work.
        if _accepted_count.value >= _n_target:
            break

        end = min(start + SUB_BATCH_SIZE, n)
        mi, ub, us = _ext_setup["generator"].get_marg_info_batch_multibank(
            batch_sample_idx[start:end],
            batch_bank_idx[start:end],
            _ext_setup["banks_list"],
            _ext_setup["waveform_dirs_list"],
            _ext_setup["min_lnlike"],
            _ext_setup["min_n_eff"],
        )
        if mi:
            with _count_lock:
                _accepted_count.value += len(mi)
            all_mi.extend(mi)
            all_b.extend(ub)
            all_s.extend(us)

        # Return freed heap pages to the OS so subsequent sub-batches
        # don't accumulate process memory across iterations.
        try:
            ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    return all_mi, all_b, all_s


def _read_block_size(waveform_dir):
    """Read blocksize from bank_config.json next to waveform_dir."""
    bank_config_path = Path(waveform_dir).parent / "bank_config.json"
    with open(bank_config_path) as f:
        return json.load(f)["blocksize"]


def _partition_by_block(
    sample_idx_arr, bank_idx_arr, waveform_dirs_list, n_workers
):
    """
    Partition (sample_idx, bank_idx) pairs into worker batches.

    Each batch contains only samples from a contiguous range of
    waveform block files per bank, so each worker loads only its
    own block files.

    Returns
    -------
    list of (sample_idx_chunk, bank_idx_chunk) tuples, one per worker.
    """
    unique_banks = np.unique(bank_idx_arr)

    # For each bank: group samples by block, split into n_workers groups
    bank_block_groups = {}
    for bank in unique_banks:
        bank_mask = bank_idx_arr == bank
        bank_samples = sample_idx_arr[bank_mask]
        block_size = _read_block_size(waveform_dirs_list[int(bank)])

        block_ids = bank_samples // block_size
        unique_blocks = np.unique(block_ids)

        # Split into n_workers groups of contiguous blocks
        block_groups = np.array_split(unique_blocks, n_workers)
        groups = []
        for block_group in block_groups:
            if len(block_group) == 0:
                continue
            mask = np.isin(block_ids, block_group)
            groups.append((
                bank_samples[mask],
                np.full(int(mask.sum()), bank, dtype=int),
            ))
        bank_block_groups[int(bank)] = groups

    # Merge across banks: worker i gets its group from every bank
    n_groups = max(len(g) for g in bank_block_groups.values())
    worker_batches = []
    for i in range(n_groups):
        s_parts, b_parts = [], []
        for bank, groups in bank_block_groups.items():
            if i < len(groups):
                s, b = groups[i]
                s_parts.append(s)
                b_parts.append(b)
        if s_parts:
            worker_batches.append((
                np.concatenate(s_parts),
                np.concatenate(b_parts),
            ))

    return worker_batches


def collect_marg_info_parallel(
    *,
    ext_generator,
    banks_list,
    waveform_dirs_list,
    sample_idx_arr,
    bank_idx_arr,
    n_combine,
    min_marg_lnlike,
    min_n_eff,
    n_workers,
):
    """
    Parallel replacement for get_marg_info_multibank().

    Partitions samples by waveform block so each worker loads only
    its own block files (O(n_blocks / n_workers), not all blocks).
    Workers process samples in sub-batches of SUB_BATCH_SIZE and
    share a counter via multiprocessing.Value; they stop collectively
    once n_combine MarginalizationInfo objects have been accepted
    globally.  After all workers finish, selects the n_combine
    MarginalizationInfo objects with highest n_effective_prior.

    Parameters
    ----------
    ext_generator : CoherentExtrinsicSamplesGenerator
        Pre-built in the main process; inherited via COW by workers.
    banks_list : list[pd.DataFrame]
    waveform_dirs_list : list[Path]
    sample_idx_arr : np.ndarray
        Intrinsic sample indices (need not be pre-shuffled).
    bank_idx_arr : np.ndarray
        Dense bank indices (0..n_banks-1), same length as
        sample_idx_arr.
    n_combine : int
        Target number of MarginalizationInfo objects to collect.
    min_marg_lnlike : float
    min_n_eff : float
    n_workers : int

    Returns
    -------
    marg_info_list : list  (length n_combine)
    used_bank_idx  : list[int]
    used_sample_idx : list[int]
    """
    global _ext_setup, _accepted_count, _count_lock, _n_target

    # Disable the per-sample arrival-time-shift (dt) cache, mirroring
    # the serial get_marg_info_multibank() which also disables it for
    # multibank runs where sample order is shuffled across banks.
    ext_generator.intrinsic_sample_processor.use_cached_dt = False
    ext_generator.intrinsic_sample_processor.update_cached_dt = False

    _ext_setup = dict(
        generator=ext_generator,
        banks_list=banks_list,
        waveform_dirs_list=waveform_dirs_list,
        min_lnlike=min_marg_lnlike,
        min_n_eff=min_n_eff,
    )

    # Shared counter: workers increment it each time they accept
    # MarginalizationInfo objects and check it before each sub-batch
    # to decide whether to stop.
    _accepted_count = Value('i', 0)
    _count_lock = Lock()
    _n_target = n_combine

    worker_batches = _partition_by_block(
        sample_idx_arr, bank_idx_arr, waveform_dirs_list, n_workers
    )
    n_actual = len(worker_batches)
    total_samples = sum(len(s) for s, _ in worker_batches)
    print(
        f"  [parallel_extrinsic] {n_actual} workers, "
        f"{total_samples} samples "
        f"(block-partitioned, target={n_combine} MI objects)"
    )

    # Collect garbage before forking workers.
    gc.collect()
    try:
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except Exception:
        pass

    # Workers stop collectively once n_combine MarginalizationInfo
    # objects are accepted globally.  May overshoot by at most
    # n_actual extra (one sub-batch per worker); main process trims
    # to n_combine by n_effective_prior below.
    all_mi, all_bank_idx, all_sample_idx = [], [], []
    with Pool(n_actual) as pool:
        for mi_batch, used_b_batch, used_s_batch in tqdm(
            pool.imap_unordered(_extrinsic_batch_worker, worker_batches),
            total=n_actual,
            desc="extrinsic workers",
        ):
            all_mi.extend(mi_batch)
            all_bank_idx.extend(used_b_batch)
            all_sample_idx.extend(used_s_batch)

    if not all_mi:
        raise RuntimeError(
            "No marginalization objects accepted. "
            "Lower thresholds or increase candidate pool."
        )

    print(
        f"  [parallel_extrinsic] {len(all_mi)} MI objects "
        f"accepted across all workers"
    )

    # Select n_combine best by n_effective_prior (descending)
    if len(all_mi) > n_combine:
        order = np.argsort(
            [-mi.n_effective_prior for mi in all_mi]
        )[:n_combine]
        all_mi = [all_mi[i] for i in order]
        all_bank_idx = [all_bank_idx[i] for i in order]
        all_sample_idx = [all_sample_idx[i] for i in order]

    # Release module globals so the caller's gc.collect() +
    # malloc_trim() can free ext_generator
    # (CoherentExtrinsicSamplesGenerator holds ~1 GB at rest for
    # typical event data).  Without this, _ext_setup keeps a live
    # reference and the generator persists into the coherent stage
    # fork, unnecessarily inflating each coherent worker's COW
    # memory footprint.
    _ext_setup = None
    _accepted_count = None
    _count_lock = None

    return all_mi, all_bank_idx, all_sample_idx


def draw_extrinsic_samples_parallel(
    *,
    banks,
    event_data,
    par_dic_0,
    fbin,
    approximant,
    selected_inds_by_bank,
    coherent_score_kwargs,
    seed,
    n_ext,
    rundir,
    n_workers,
    n_combine=16,
    min_marg_lnlike_for_sampling=0.0,
    single_marg_info_min_n_effective_prior=32.0,
):
    """
    Drop-in for inference.draw_extrinsic_samples() with parallel
    MarginalizationInfo collection.

    Builds CoherentExtrinsicSamplesGenerator in the main process,
    then dispatches block-partitioned fork+COW workers for the
    MarginalizationInfo collection loop.  Final extrinsic sampling
    remains serial.

    Does not handle the extrinsic_samples caching path — call
    inference.draw_extrinsic_samples() directly for that.

    Parameters
    ----------
    banks : dict[str, Path]
    event_data : EventData
    par_dic_0 : dict
    fbin : np.ndarray
    approximant : str
    selected_inds_by_bank : dict[str, np.ndarray]
    coherent_score_kwargs : dict
    seed : int or None
    n_ext : int
    rundir : Path
    n_workers : int
    n_combine : int
    min_marg_lnlike_for_sampling : float
    single_marg_info_min_n_effective_prior : float

    Returns
    -------
    extrinsic_samples_df : pd.DataFrame
    response_dpe : np.ndarray
    timeshift_dbe : np.ndarray
    """
    print(
        f"\n=== Extrinsic sample generation "
        f"(parallel, {n_workers} workers) ==="
    )

    rundir = Path(rundir)
    rundir.mkdir(parents=True, exist_ok=True)

    first_bank_path = list(banks.values())[0]
    wfg = WaveformGenerator.from_event_data(event_data, approximant)
    marg_ext_like = MarginalizationExtrinsicSamplerFreeLikelihood(
        event_data, wfg, par_dic_0, fbin,
        coherent_score=coherent_score_kwargs,
    )
    # Note: intrinsic_bank_file and waveform_dir are not used in the
    # multibank get_marg_info_batch_multibank() path; first_bank_path
    # is passed only for convenience.
    ext_generator = CoherentExtrinsicSamplesGenerator(
        likelihood=marg_ext_like,
        intrinsic_bank_file=first_bank_path / "intrinsic_sample_bank.feather",
        waveform_dir=first_bank_path / "waveforms",
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
        bank_df = pd.read_feather(
            bank_path / "intrinsic_sample_bank.feather"
        )
        banks_list.append(bank_df)
        waveform_dirs_list.append(bank_path / "waveforms")
        sample_idx_arrays.append(filtered_inds)
        bank_idx_arrays.append(
            np.full(len(filtered_inds), dense_idx, dtype=int)
        )

    if not sample_idx_arrays:
        raise ValueError(
            "No intrinsic samples available for extrinsic sampling "
            "(all banks empty)."
        )

    sample_idx_arr = np.concatenate(sample_idx_arrays)
    bank_idx_arr = np.concatenate(bank_idx_arrays)

    marg_info_list, used_bank_idx, used_sample_idx = (
        collect_marg_info_parallel(
            ext_generator=ext_generator,
            banks_list=banks_list,
            waveform_dirs_list=waveform_dirs_list,
            sample_idx_arr=sample_idx_arr,
            bank_idx_arr=bank_idx_arr,
            n_combine=n_combine,
            min_marg_lnlike=min_marg_lnlike_for_sampling,
            min_n_eff=single_marg_info_min_n_effective_prior,
            n_workers=n_workers,
        )
    )

    # Merge list into a single MarginalizationInfo object
    # (same logic as serial draw_extrinsic_samples).
    marg_info = deepcopy(marg_info_list[0])
    if len(marg_info_list) > 1:
        marg_info.update_with_list(marg_info_list[1:])

    # Save outputs (mirrors serial draw_extrinsic_samples).
    with open(rundir / "marg_info.pkl", "wb") as f:
        pickle.dump(marg_info, f)
    np.save(rundir / "used_bank_idx.npy", np.array(used_bank_idx, dtype=int))
    np.save(
        rundir / "used_sample_idx.npy",
        np.array(used_sample_idx, dtype=int),
    )

    # Final sampling — fast, serial, reuses the same generator.
    extrinsic_samples_df, response_dpe, timeshift_dbe = (
        ext_generator.draw_extrinsic_samples_from_indices(
            n_ext, marg_info=marg_info,
        )
    )
    extrinsic_samples_df.to_feather(rundir / "extrinsic_samples.feather")
    np.save(arr=response_dpe, file=rundir / "response_dpe.npy")
    np.save(arr=timeshift_dbe, file=rundir / "timeshift_dbe.npy")

    return extrinsic_samples_df, response_dpe, timeshift_dbe
