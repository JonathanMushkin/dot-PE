"""Dask-based parallel extrinsic marginalization-info collection.

Public API
----------
collect_marg_info_dask
    Dask replacement for get_marg_info_multibank().
draw_extrinsic_samples_parallel
    Drop-in for inference.draw_extrinsic_samples() with Dask parallelism.

Design
------
Workers are partitioned by waveform *block* rather than by random
sample batches.  Each worker receives samples from a contiguous range
of block files, limiting it to O(n_blocks / n_workers) file reads
instead of all blocks.

Each Dask task rebuilds CoherentExtrinsicSamplesGenerator from scratch
by reloading EventData from the shared filesystem (~0.1–2 s), since
EventData holds LAL/SWIG objects and cannot be pickled.

Workers process their samples in sub-batches of SUB_BATCH_SIZE.
Early stopping: futures are submitted for all partitions; as_completed
is used to collect results, and remaining futures are cancelled once
``n_combine`` MarginalizationInfo objects have been accepted.

After all tasks finish (or are cancelled), the n_combine
MarginalizationInfo objects with the highest n_effective_prior are
selected from the pool.
"""

import gc
import json
import pickle
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

from cogwheel.waveform import WaveformGenerator
from dot_pe.coherent_processing import CoherentExtrinsicSamplesGenerator
from dot_pe.marginalization import (
    MarginalizationExtrinsicSamplerFreeLikelihood,
)

# Samples processed per sub-batch inside each worker.
SUB_BATCH_SIZE = 128


# ─────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────


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

    bank_block_groups = {}
    for bank in unique_banks:
        bank_mask = bank_idx_arr == bank
        bank_samples = sample_idx_arr[bank_mask]
        block_size = _read_block_size(waveform_dirs_list[int(bank)])

        block_ids = bank_samples // block_size
        unique_blocks = np.unique(block_ids)

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


# ─────────────────────────────────────────────────────────────────────
# Dask worker function — pure (no module globals), picklable
# ─────────────────────────────────────────────────────────────────────


def _extrinsic_partition_worker_dask(
    event_path,
    par_dic_0,
    fbin,
    approximant,
    coherent_score_kwargs,
    seed,
    banks_df_list,
    waveform_dirs_strs,
    sample_idx_chunk,
    bank_idx_chunk,
    min_lnlike,
    min_n_eff,
):
    """Pure Dask worker — rebuilds CoherentExtrinsicSamplesGenerator
    from ``event_path`` (shared FS) to avoid pickling LAL/SWIG objects.

    Processes ``sample_idx_chunk`` in sub-batches of SUB_BATCH_SIZE and
    returns all accepted (MarginalizationInfo, bank_idx, sample_idx)
    triples found in this partition.
    """
    import warnings as _w
    _w.filterwarnings("ignore", "Wswiglal-redir-stdio")
    import ctypes
    from dot_pe.utils import get_event_data

    event_data = get_event_data(event_path)
    wfg = WaveformGenerator.from_event_data(event_data, approximant)
    marg_ext_like = MarginalizationExtrinsicSamplerFreeLikelihood(
        event_data, wfg, par_dic_0, fbin,
        coherent_score=coherent_score_kwargs,
    )
    first_bank_dir = Path(waveform_dirs_strs[0]).parent
    ext_gen = CoherentExtrinsicSamplesGenerator(
        likelihood=marg_ext_like,
        intrinsic_bank_file=first_bank_dir / "intrinsic_sample_bank.feather",
        waveform_dir=waveform_dirs_strs[0],
        seed=seed,
    )
    ext_gen.intrinsic_sample_processor.use_cached_dt = False
    ext_gen.intrinsic_sample_processor.update_cached_dt = False

    waveform_dirs_list = [Path(d) for d in waveform_dirs_strs]

    all_mi, all_b, all_s = [], [], []
    n = len(sample_idx_chunk)
    for start in range(0, n, SUB_BATCH_SIZE):
        end = min(start + SUB_BATCH_SIZE, n)
        mi, ub, us = ext_gen.get_marg_info_batch_multibank(
            sample_idx_chunk[start:end],
            bank_idx_chunk[start:end],
            banks_df_list,
            waveform_dirs_list,
            min_lnlike,
            min_n_eff,
        )
        all_mi.extend(mi)
        all_b.extend(ub)
        all_s.extend(us)

        # Return freed heap pages to the OS between sub-batches.
        try:
            ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    return all_mi, all_b, all_s


# ─────────────────────────────────────────────────────────────────────
# Dask orchestrator
# ─────────────────────────────────────────────────────────────────────


def collect_marg_info_dask(
    *,
    client,
    event_path,
    ext_generator,
    banks_list,
    waveform_dirs_list,
    sample_idx_arr,
    bank_idx_arr,
    n_combine,
    min_marg_lnlike,
    min_n_eff,
    par_dic_0,
    fbin,
    approximant,
    coherent_score_kwargs,
    seed,
    n_workers,
):
    """Dask replacement for get_marg_info_multibank().

    Partitions samples by waveform block so each worker loads only its
    own block files.  Submits one Dask future per partition and collects
    via as_completed; cancels remaining futures once ``n_combine``
    MarginalizationInfo objects have been accepted.  Selects the
    ``n_combine`` objects with highest n_effective_prior.

    Parameters
    ----------
    client : dask.distributed.Client
    event_path : str
        Path to event .npz file; passed to workers for reloading EventData.
    ext_generator : CoherentExtrinsicSamplesGenerator
        Built in the main process; used only to set cache flags here.
        Workers rebuild their own generator from ``event_path``.
    banks_list : list[pd.DataFrame]
    waveform_dirs_list : list[Path]
    sample_idx_arr, bank_idx_arr : np.ndarray
    n_combine : int
    min_marg_lnlike, min_n_eff : float
    par_dic_0, fbin, approximant, coherent_score_kwargs, seed
        Passed to workers for generator reconstruction.
    n_workers : int
        Number of partitions (target Dask task count).

    Returns
    -------
    marg_info_list : list  (length <= n_combine)
    used_bank_idx  : list[int]
    used_sample_idx : list[int]
    """
    from dask.distributed import as_completed as _ac

    # Mirror the serial path: disable dt cache for multibank runs.
    ext_generator.intrinsic_sample_processor.use_cached_dt = False
    ext_generator.intrinsic_sample_processor.update_cached_dt = False

    waveform_dirs_strs = [str(d) for d in waveform_dirs_list]

    worker_batches = _partition_by_block(
        sample_idx_arr, bank_idx_arr, waveform_dirs_list, n_workers
    )
    total_samples = sum(len(s) for s, _ in worker_batches)
    print(
        f"  [Dask extrinsic] {len(worker_batches)} tasks, "
        f"{total_samples} samples "
        f"(block-partitioned, target={n_combine} MI objects)"
    )

    gc.collect()

    futures = [
        client.submit(
            _extrinsic_partition_worker_dask,
            event_path, par_dic_0, fbin, approximant,
            coherent_score_kwargs, seed,
            banks_list, waveform_dirs_strs,
            sample_idx_chunk, bank_idx_chunk,
            min_marg_lnlike, min_n_eff,
            pure=False,
        )
        for sample_idx_chunk, bank_idx_chunk in worker_batches
    ]

    all_mi, all_bank_idx, all_sample_idx = [], [], []
    for f in tqdm(_ac(futures), total=len(futures), desc="extrinsic workers"):
        try:
            mi_batch, b_batch, s_batch = f.result()
        except Exception:
            # Future was cancelled or errored — skip it.
            continue
        all_mi.extend(mi_batch)
        all_bank_idx.extend(b_batch)
        all_sample_idx.extend(s_batch)
        if len(all_mi) >= n_combine:
            # Cancel remaining futures (best-effort; may already be running).
            for remaining in futures:
                if remaining.status not in ("finished", "error", "cancelled"):
                    remaining.cancel()
            break  # don't block waiting for cancelled futures

    if not all_mi:
        raise RuntimeError(
            "No marginalization objects accepted. "
            "Lower thresholds or increase candidate pool."
        )

    print(
        f"  [Dask extrinsic] {len(all_mi)} MI objects accepted across all tasks"
    )

    # Select n_combine best by n_effective_prior (descending)
    if len(all_mi) > n_combine:
        order = np.argsort([-mi.n_effective_prior for mi in all_mi])[:n_combine]
        all_mi = [all_mi[i] for i in order]
        all_bank_idx = [all_bank_idx[i] for i in order]
        all_sample_idx = [all_sample_idx[i] for i in order]

    return all_mi, all_bank_idx, all_sample_idx


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


def draw_extrinsic_samples_parallel(
    *,
    client,
    event_path,
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
    Drop-in for inference.draw_extrinsic_samples() with Dask parallelism.

    Builds CoherentExtrinsicSamplesGenerator in the main process for the
    final draw step, while dispatching block-partitioned Dask tasks for
    the MarginalizationInfo collection loop.  Workers rebuild their own
    generator by reloading EventData from ``event_path``.

    Does not handle the extrinsic_samples caching path — call
    inference.draw_extrinsic_samples() directly for that.

    Parameters
    ----------
    client : dask.distributed.Client
    event_path : str
        Path to event .npz file; passed to Dask workers.
    banks : dict[str, Path]
    event_data : EventData
        Used in the main process to build the generator for final sampling.
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
    """
    print(
        f"\n=== Extrinsic sample generation "
        f"(Dask, {n_workers} tasks) ==="
    )

    rundir = Path(rundir)
    rundir.mkdir(parents=True, exist_ok=True)

    first_bank_path = list(banks.values())[0]
    wfg = WaveformGenerator.from_event_data(event_data, approximant)
    marg_ext_like = MarginalizationExtrinsicSamplerFreeLikelihood(
        event_data, wfg, par_dic_0, fbin,
        coherent_score=coherent_score_kwargs,
    )
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

    marg_info_list, used_bank_idx, used_sample_idx = collect_marg_info_dask(
        client=client,
        event_path=event_path,
        ext_generator=ext_generator,
        banks_list=banks_list,
        waveform_dirs_list=waveform_dirs_list,
        sample_idx_arr=sample_idx_arr,
        bank_idx_arr=bank_idx_arr,
        n_combine=n_combine,
        min_marg_lnlike=min_marg_lnlike_for_sampling,
        min_n_eff=single_marg_info_min_n_effective_prior,
        par_dic_0=par_dic_0,
        fbin=fbin,
        approximant=approximant,
        coherent_score_kwargs=coherent_score_kwargs,
        seed=seed,
        n_workers=n_workers,
    )

    # Merge list into a single MarginalizationInfo object
    marg_info = deepcopy(marg_info_list[0])
    if len(marg_info_list) > 1:
        marg_info.update_with_list(marg_info_list[1:])

    # Save outputs (mirrors serial draw_extrinsic_samples)
    with open(rundir / "marg_info.pkl", "wb") as f:
        pickle.dump(marg_info, f)
    np.save(rundir / "used_bank_idx.npy", np.array(used_bank_idx, dtype=int))
    np.save(
        rundir / "used_sample_idx.npy",
        np.array(used_sample_idx, dtype=int),
    )

    # Final sampling — fast, serial, reuses the main-process generator.
    extrinsic_samples_df, response_dpe, timeshift_dbe = (
        ext_generator.draw_extrinsic_samples_from_indices(
            n_ext, marg_info=marg_info,
        )
    )
    extrinsic_samples_df.to_feather(rundir / "extrinsic_samples.feather")
    np.save(arr=response_dpe, file=rundir / "response_dpe.npy")
    np.save(arr=timeshift_dbe, file=rundir / "timeshift_dbe.npy")

    return extrinsic_samples_df, response_dpe, timeshift_dbe
