"""
HPC parallelized coherent likelihood block processing.
"""

import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .utils_hpc import init_worker, partition_block_pairs


def _process_block_pairs(args):
    """
    Worker function to process a group of (i_block, e_block) pairs.

    Parameters
    ----------
    args : tuple
        (block_pairs, processor_init_kwargs, waveform_dir, response_dpe, timeshift_dbe,
         tempdir, i_blocks, e_blocks)

    Returns
    -------
    list
        List of blocknames created
    """
    (
        block_pairs,
        processor_init_kwargs,
        waveform_dir,
        response_dpe,
        timeshift_dbe,
        tempdir,
        i_blocks,
        e_blocks,
    ) = args

    # Initialize worker (disable nested threading)
    init_worker()

    # Import here to avoid issues with multiprocessing
    from dot_pe.coherent_processing import CoherentLikelihoodProcessor

    # Reconstruct processor from kwargs
    processor = CoherentLikelihoodProcessor(**processor_init_kwargs)

    blocknames = []
    for i_block_idx, e_block_idx in block_pairs:
        i_block = i_blocks[i_block_idx]
        e_block = e_blocks[e_block_idx]

        # Load amplitude and phase for the current intrinsic block
        amp_impb, phase_impb = processor.intrinsic_sample_processor.load_amp_and_phase(
            waveform_dir, i_block
        )
        h_impb = amp_impb * np.exp(1j * phase_impb)

        response_subset = response_dpe[..., e_block]
        timeshift_subset = timeshift_dbe[..., e_block]

        # Create the likelihood block
        processor.create_a_likelihood_block(
            h_impb,
            response_subset,
            timeshift_subset,
            i_block,
            e_block,
        )

        # Note: combine_prob_samples_with_next_block() is not thread-safe
        # and should be called sequentially after all blocks are processed.
        # So we don't call it here in the worker.

        blockname = f"block_{i_block_idx}_{e_block_idx}.npz"
        blocknames.append(blockname)

    return blocknames


def create_likelihood_blocks_hpc(
    processor,
    tempdir: Path,
    i_blocks: List[np.ndarray],
    e_blocks: List[np.ndarray],
    response_dpe: np.ndarray,
    timeshift_dbe: np.ndarray,
    waveform_dir: Path,
    n_procs: int = None,
    pairs_per_task: int = 4,
) -> List[str]:
    """
    HPC parallelized version of CoherentLikelihoodProcessor.create_likelihood_blocks.

    Processes (i_block, e_block) pairs in parallel across worker processes.

    Parameters
    ----------
    processor : CoherentLikelihoodProcessor
        Processor instance (will be serialized via init kwargs)
    tempdir : Path
        Temporary directory for block files
    i_blocks : List[np.ndarray]
        List of intrinsic blocks
    e_blocks : List[np.ndarray]
        List of extrinsic blocks
    response_dpe : np.ndarray
        Response matrix
    timeshift_dbe : np.ndarray
        Timeshift matrix
    waveform_dir : Path
        Directory containing waveforms
    n_procs : int, optional
        Number of worker processes (default: cpu_count)
    pairs_per_task : int
        Number of (i_block, e_block) pairs per worker task

    Returns
    -------
    List[str]
        List of blocknames created
    """
    tempdir = Path(tempdir)
    total_pairs = len(i_blocks) * len(e_blocks)

    # Partition block pairs into tasks
    block_pairs = [(i_idx, e_idx) for i_idx in range(len(i_blocks)) for e_idx in range(len(e_blocks))]
    task_groups = partition_block_pairs(i_blocks, e_blocks, pairs_per_task)

    # Determine number of processes
    if n_procs is None:
        n_procs = min(len(task_groups), mp.cpu_count() or 4)

    # Extract processor initialization kwargs for serialization
    # Note: This assumes CoherentLikelihoodProcessor has a way to serialize/deserialize
    # For now, we'll pass the processor directly and rely on pickle
    # In practice, you may need to extract specific kwargs
    processor_init_kwargs = {
        "intrinsic_bank_file": processor.intrinsic_bank_file,
        "waveform_dir": processor.waveform_dir,
        "n_phi": processor.n_phi,
        "m_arr": processor.m_arr,
        "likelihood": processor.likelihood,
        "seed": processor.seed,
        "max_bestfit_lnlike_diff": processor.max_bestfit_lnlike_diff,
        "size_limit": processor.size_limit,
        "int_block_size": processor.int_block_size,
        "ext_block_size": processor.ext_block_size,
        "min_bestfit_lnlike_to_keep": processor.min_bestfit_lnlike_to_keep,
        "full_intrinsic_indices": processor.full_intrinsic_indices,
        "renormalize_log_prior_weights_i": processor.renormalize_log_prior_weights_i,
        "intrinsic_logw_lookup": processor.intrinsic_logw_lookup,
        "n_samples_discarded": processor.n_samples_discarded,
        "logsumexp_discarded_ln_posterior": processor.logsumexp_discarded_ln_posterior,
        "logsumsqrexp_discarded_ln_posterior": processor.logsumsqrexp_discarded_ln_posterior,
        "n_samples_accepted": processor.n_samples_accepted,
        "logsumexp_accepted_ln_posterior": processor.logsumexp_accepted_ln_posterior,
        "logsumsqrexp_accepted_ln_posterior": processor.logsumsqrexp_accepted_ln_posterior,
        "n_distance_marginalizations": processor.n_distance_marginalizations,
    }

    # Prepare arguments for workers
    worker_args = [
        (
            task_group,
            processor_init_kwargs,
            waveform_dir,
            response_dpe,
            timeshift_dbe,
            tempdir,
            i_blocks,
            e_blocks,
        )
        for task_group in task_groups
    ]

    # Process in parallel
    if n_procs > 1 and len(task_groups) > 1:
        with mp.Pool(processes=n_procs, initializer=init_worker) as pool:
            task_results = pool.map(_process_block_pairs, worker_args)
    else:
        # Single process
        task_results = [_process_block_pairs(args) for args in worker_args]

    # Flatten results
    blocknames = []
    for task_result in task_results:
        blocknames.extend(task_result)

    # Note: combine_prob_samples_with_next_block() must be called sequentially
    # after all blocks are created. This is done in the main process.
    # The caller should iterate through blocks in order and call combine_prob_samples_with_next_block()

    return blocknames
