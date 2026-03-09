"""Fork-based parallel extrinsic marginalization-info collection.

Both MP and swarm paths use identical fork+COW logic via this module.

Public API
----------
collect_marg_info_parallel   -- parallel replacement for get_marg_info_multibank()
draw_extrinsic_samples_parallel -- drop-in for inference.draw_extrinsic_samples()

Design
------
Workers are partitioned by waveform *block* rather than by random sample batches.
Each worker receives only samples from a contiguous range of block files, so it
loads at most (n_blocks_total / n_workers) files.  This avoids the O(8 GB) per-worker
waveform footprint that random batches caused (random 1024-sample batches span all
64 block files = ~12 GB of waveform data per worker).

Workers process their samples in sub-batches of SUB_BATCH_SIZE and share a
multiprocessing.Value counter (_accepted_count) so they collectively stop as
soon as n_combine MI objects have been accepted across all workers.  This avoids
the timeout failure that occurs when workers run all their samples to completion.

After all workers finish, the n_combine MI objects with the highest n_effective_prior
are selected from the pool.
"""

import ctypes
import gc
import json
import pickle
from copy import deepcopy
from multiprocessing import Lock, Pool, Value
from pathlib import Path

import numpy as np
import pandas as pd

# Module-level globals: set in main process before Pool creation,
# inherited read-only by fork-based workers via copy-on-write (data)
# or shared memory (Value/Lock).
_ext_setup = None
_accepted_count = None   # multiprocessing.Value('i', 0) — shared across workers
_count_lock = None       # multiprocessing.Lock()
_n_target = None         # int — stop when _accepted_count.value >= _n_target

SUB_BATCH_SIZE = 128     # samples per sub-batch inside each worker
# NOTE: smaller batch = lower peak RSS per worker (scales ~linearly with batch size).
# Root cause: each get_marg_info_batch_multibank call accumulates MI objects via
# np.concatenate in MarginalizationInfo.update(); freed pages pile up in the heap
# (Python doesn't return them to the OS), creating a high-water mark.
# Measured: B=1024 → +7652 MB/worker; B=128 → ~957 MB/worker (estimated).
# For n=8 workers: B=1024 needs ~62 GB (OOM); B=128 needs ~9 GB (fits in 20 GB).


def _extrinsic_batch_worker(args):
    """Process block-partitioned samples in sub-batches; stop when global target reached."""
    batch_sample_idx, batch_bank_idx = args
    all_mi, all_b, all_s = [], [], []

    n = len(batch_sample_idx)
    for start in range(0, n, SUB_BATCH_SIZE):
        # Check global stop before loading waveforms for this sub-batch.
        # Reading Value without the lock is safe: a stale read at worst costs
        # one extra sub-batch of work, which is acceptable.
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

        # Return freed heap pages to the OS so subsequent sub-batches don't
        # accumulate RSS across iterations (the high-water issue).
        try:
            ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    return all_mi, all_b, all_s


def _read_block_size(waveform_dir):
    """Read blocksize from bank_config.json adjacent to the waveform directory."""
    bank_config_path = Path(waveform_dir).parent / "bank_config.json"
    with open(bank_config_path) as f:
        return json.load(f)["blocksize"]


def _partition_by_block(sample_idx_arr, bank_idx_arr, waveform_dirs_list, n_workers):
    """
    Partition (sample_idx, bank_idx) pairs into worker batches, grouped by block.

    Each batch contains only samples from a contiguous range of waveform block
    files for each bank.  This ensures each worker loads only its own blocks.

    Returns
    -------
    list of (sample_idx_chunk, bank_idx_chunk) tuples, one per worker.
    """
    unique_banks = np.unique(bank_idx_arr)

    # For each bank: group samples by block and split into n_workers groups
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
            groups.append((bank_samples[mask], np.full(int(mask.sum()), bank, dtype=int)))
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
    Parallel replacement for CoherentExtrinsicSamplesGenerator.get_marg_info_multibank().

    Partitions samples by waveform block so each worker loads only its own block
    files (O(block_files_total / n_workers) files, not all files).  Workers process
    samples in sub-batches of SUB_BATCH_SIZE and share a counter via
    multiprocessing.Value; they stop collectively once n_combine MI objects have
    been accepted globally.  After all workers finish, selects the n_combine MI
    objects with highest n_effective_prior.

    Parameters
    ----------
    ext_generator : CoherentExtrinsicSamplesGenerator
        Pre-built in the main process; inherited via COW by workers.
    banks_list : list[pd.DataFrame]
    waveform_dirs_list : list[Path]
    sample_idx_arr : np.ndarray  intrinsic indices (need not be shuffled)
    bank_idx_arr : np.ndarray    corresponding bank indices (dense)
    n_combine : int              target number of MI objects
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

    # Disable dt cache (mirrors get_marg_info_multibank)
    ext_generator.intrinsic_sample_processor.use_cached_dt = False
    ext_generator.intrinsic_sample_processor.update_cached_dt = False

    _ext_setup = dict(
        generator=ext_generator,
        banks_list=banks_list,
        waveform_dirs_list=waveform_dirs_list,
        min_lnlike=min_marg_lnlike,
        min_n_eff=min_n_eff,
    )

    # Shared counter: workers increment this each time they accept MI objects
    # and check it before each sub-batch to decide whether to stop.
    _accepted_count = Value('i', 0)
    _count_lock = Lock()
    _n_target = n_combine

    worker_batches = _partition_by_block(
        sample_idx_arr, bank_idx_arr, waveform_dirs_list, n_workers
    )
    n_actual = len(worker_batches)
    total_samples = sum(len(s) for s, _ in worker_batches)
    print(f"  [parallel_extrinsic] {n_actual} workers, {total_samples} samples "
          f"(block-partitioned, target={n_combine} MI objects)")

    # Free pending GC before forking
    gc.collect()
    try:
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except Exception:
        pass

    # Workers stop collectively once n_combine MI objects are accepted globally.
    # May overshoot by at most n_actual extra (one sub-batch per worker);
    # main process trims to n_combine by n_effective_prior below.
    all_mi, all_bank_idx, all_sample_idx = [], [], []
    with Pool(n_actual) as pool:
        for mi_batch, used_b_batch, used_s_batch in pool.imap_unordered(
            _extrinsic_batch_worker, worker_batches
        ):
            all_mi.extend(mi_batch)
            all_bank_idx.extend(used_b_batch)
            all_sample_idx.extend(used_s_batch)

    if not all_mi:
        raise RuntimeError(
            "No marginalization objects accepted. "
            "Lower thresholds or increase candidate pool."
        )

    print(f"  [parallel_extrinsic] {len(all_mi)} MI objects accepted across all workers")

    # Select n_combine best by n_effective_prior (descending)
    if len(all_mi) > n_combine:
        order = np.argsort([-mi.n_effective_prior for mi in all_mi])[:n_combine]
        all_mi = [all_mi[i] for i in order]
        all_bank_idx = [all_bank_idx[i] for i in order]
        all_sample_idx = [all_sample_idx[i] for i in order]

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
    Drop-in replacement for inference.draw_extrinsic_samples() with parallel MI collection.

    Builds CoherentExtrinsicSamplesGenerator in the main process, then dispatches
    block-partitioned fork+COW workers for the MI collection loop.
    Final extrinsic sampling remains serial.

    Does not handle the extrinsic_samples caching path — call inference.draw_extrinsic_samples()
    directly for that.

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
    from cogwheel.waveform import WaveformGenerator
    from dot_pe.coherent_processing import CoherentExtrinsicSamplesGenerator
    from dot_pe.marginalization import MarginalizationExtrinsicSamplerFreeLikelihood

    print(f"\n=== Extrinsic sample generation (parallel, {n_workers} workers) ===")

    rundir = Path(rundir)
    rundir.mkdir(parents=True, exist_ok=True)

    first_bank_path = list(banks.values())[0]
    wfg = WaveformGenerator.from_event_data(event_data, approximant)
    marg_ext_like = MarginalizationExtrinsicSamplerFreeLikelihood(
        event_data, wfg, par_dic_0, fbin, coherent_score=coherent_score_kwargs
    )
    ext_generator = CoherentExtrinsicSamplesGenerator(
        likelihood=marg_ext_like,
        intrinsic_bank_file=first_bank_path / "intrinsic_sample_bank.feather",
        waveform_dir=first_bank_path / "waveforms",
        seed=seed,
    )

    banks_list, waveform_dirs_list, sample_idx_arrays, bank_idx_arrays = [], [], [], []
    for bank_id, bank_path in banks.items():
        filtered_inds = selected_inds_by_bank[bank_id]
        if len(filtered_inds) == 0:
            continue
        dense_idx = len(banks_list)
        banks_list.append(pd.read_feather(bank_path / "intrinsic_sample_bank.feather"))
        waveform_dirs_list.append(bank_path / "waveforms")
        sample_idx_arrays.append(filtered_inds)
        bank_idx_arrays.append(np.full(len(filtered_inds), dense_idx, dtype=int))

    if not sample_idx_arrays:
        raise ValueError(
            "No intrinsic samples available for extrinsic sampling (all banks empty)."
        )

    sample_idx_arr = np.concatenate(sample_idx_arrays)
    bank_idx_arr = np.concatenate(bank_idx_arrays)

    marg_info_list, used_bank_idx, used_sample_idx = collect_marg_info_parallel(
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

    # Merge (mirrors coherent_processing.py:1285-1287)
    marg_info = deepcopy(marg_info_list[0])
    if len(marg_info_list) > 1:
        marg_info.update_with_list(marg_info_list[1:])

    # Save (mirrors serial behavior in get_marg_info_multibank)
    with open(rundir / "marg_info.pkl", "wb") as f:
        pickle.dump(marg_info, f)
    np.save(rundir / "used_bank_idx.npy", np.array(used_bank_idx, dtype=int))
    np.save(rundir / "used_sample_idx.npy", np.array(used_sample_idx, dtype=int))

    # Final sampling — fast, serial, reuses the same generator
    extrinsic_samples_df, response_dpe, timeshift_dbe = (
        ext_generator.draw_extrinsic_samples_from_indices(n_ext, marg_info=marg_info)
    )
    extrinsic_samples_df.to_feather(rundir / "extrinsic_samples.feather")
    np.save(arr=response_dpe, file=rundir / "response_dpe.npy")
    np.save(arr=timeshift_dbe, file=rundir / "timeshift_dbe.npy")

    return extrinsic_samples_df, response_dpe, timeshift_dbe
