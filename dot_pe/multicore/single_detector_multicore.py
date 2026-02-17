"""
Multicore parallelized single-detector processing for incoherent likelihood evaluation.
"""

import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from dot_pe.inference import run_for_single_detector, _create_single_detector_processor
from dot_pe.utils import get_event_data
from cogwheel.data import EventData
from .utils_multicore import init_worker, partition_indices, group_batches

# Per-process cache set by init_single_detector_worker; read by _process_batch_group.
_worker_state: Dict[str, Any] = {}


def init_single_detector_worker(init_args: Tuple) -> None:
    """
    Build event data and single-detector processors once per worker.
    Called once per process by Pool(initializer=..., initargs=(init_args,)).
    """
    init_worker()
    (
        event_path,
        par_dic_0,
        bank_folder,
        fbin,
        approximant,
        n_phi,
        m_arr,
        single_detector_blocksize,
        n_t,
        detector_names,
    ) = init_args
    event_data = get_event_data(event_path)
    sdp_by_detector = {}
    for det_name in detector_names:
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
    _worker_state["event_data"] = event_data
    _worker_state["sdp_by_detector"] = sdp_by_detector
    _worker_state["detector_names"] = detector_names
    _worker_state["par_dic_0"] = par_dic_0
    _worker_state["bank_folder"] = bank_folder
    _worker_state["fbin"] = fbin
    _worker_state["approximant"] = approximant
    _worker_state["n_phi"] = n_phi
    _worker_state["single_detector_blocksize"] = single_detector_blocksize
    _worker_state["m_arr"] = m_arr
    _worker_state["n_t"] = n_t


def _process_batch_group(batch_group: List[np.ndarray]) -> List[Dict[str, Any]]:
    """
    Process a group of batches using cached event data and SDPs.
    Expects init_single_detector_worker to have run first in this process.
    """
    event_data = _worker_state["event_data"]
    sdp_by_detector = _worker_state["sdp_by_detector"]
    detector_names = _worker_state["detector_names"]

    results = []
    for batch_indices in batch_group:
        batch_start = batch_indices[0] if len(batch_indices) > 0 else 0
        batch_end = batch_indices[-1] + 1 if len(batch_indices) > 0 else 0

        batch_lnlike_di = np.zeros((len(detector_names), len(batch_indices)))
        h_impb = None

        for d, det_name in enumerate(detector_names):
            temp = run_for_single_detector(
                event_data,
                det_name,
                _worker_state["par_dic_0"],
                _worker_state["bank_folder"],
                batch_indices,
                _worker_state["fbin"],
                h_impb,
                _worker_state["approximant"],
                _worker_state["n_phi"],
                _worker_state["single_detector_blocksize"],
                _worker_state["m_arr"],
                _worker_state["n_t"],
                size_limit=10**7,
                sdp=sdp_by_detector[det_name],
            )
            if h_impb is None:
                batch_lnlike_di[d, :] = temp[0]
                h_impb = temp[1]
            else:
                batch_lnlike_di[d, :] = temp

        results.append(
            {
                "batch_start": batch_start,
                "batch_end": batch_end,
                "batch_indices": batch_indices,
                "lnlike_di": batch_lnlike_di,
            }
        )

    return results


def collect_int_samples_from_single_detectors_multicore(
    event_data: Union[EventData, str, Path],
    par_dic_0: Dict,
    single_detector_blocksize: int,
    n_int: int,
    n_phi: int,
    n_t: int,
    bank_folder: Path,
    i_int_start: int = 0,
    max_incoherent_lnlike_drop: float = 20,
    preselected_indices=None,
    apply_threshold: bool = True,
    n_procs: int = None,
    i_batch: int = None,
    batches_per_task: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multicore parallelized version of collect_int_samples_from_single_detectors.

    Uses multiprocessing to parallelize batch processing across workers.
    Each worker processes groups of batches sequentially to minimize overhead.

    Parameters
    ----------
    event_data : EventData
        Event data object
    par_dic_0 : Dict
        Parameter dictionary
    single_detector_blocksize : int
        Block size for single detector processing
    n_int : int
        Number of intrinsic samples
    n_phi : int
        Number of orbital phase samples
    n_t : int
        Number of time samples
    bank_folder : Path
        Path to bank folder
    i_int_start : int
        Starting intrinsic index
    max_incoherent_lnlike_drop : float
        Maximum incoherent likelihood drop for thresholding
    preselected_indices : np.ndarray, optional
        Pre-selected indices to use instead of range
    apply_threshold : bool
        Whether to apply threshold filtering
    n_procs : int, optional
        Number of worker processes (default: cpu_count)
    i_batch : int, optional
        Batch size for partitioning (default: single_detector_blocksize)
    batches_per_task : int
        Number of batches per worker task (default: 1).
        On 40–100+ cores, increase so each task is ~100–500 ms to reduce overhead.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (inds, lnlike_di, incoherent_lnlikes)
    """
    import json

    bank_folder = Path(bank_folder)
    with open(bank_folder / "bank_config.json", "r", encoding="utf-8") as f:
        bank_config = json.load(f)
        fbin = np.array(bank_config["fbin"])
        approximant = bank_config["approximant"]
        m_arr = np.array(bank_config["m_arr"])

    # Get event path and detector names
    if isinstance(event_data, (str, Path)):
        event_path = Path(event_data)
        event_data_obj = get_event_data(event_path)
        detector_names = event_data_obj.detector_names
    else:
        # EventData object - extract path or raise error
        event_path = getattr(event_data, "path", None)
        if event_path is None:
            raise ValueError(
                "For multicore multiprocessing, pass event as a path (str/Path), not EventData object."
            )
        detector_names = event_data.detector_names

    # Determine indices
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

    # Determine batch size
    if i_batch is None:
        i_batch = single_detector_blocksize

    # Partition indices into batches
    batches = partition_indices(intrinsic_indices, i_batch)

    # Group batches into tasks
    task_groups = group_batches(batches, batches_per_task)

    # Determine number of processes
    if n_procs is None:
        n_procs = min(len(task_groups), mp.cpu_count() or 4)

    if isinstance(event_data, (str, Path)):
        event_path = Path(event_data)
    else:
        raise ValueError(
            "For multicore multiprocessing, pass event as a path (str/Path), not EventData object. "
            "EventData objects cannot be pickled for multiprocessing."
        )

    init_args = (
        event_path,
        par_dic_0,
        bank_folder,
        fbin,
        approximant,
        n_phi,
        m_arr,
        single_detector_blocksize,
        n_t,
        detector_names,
    )

    lnlike_di = np.zeros((len(detector_names), len(intrinsic_indices)))
    if n_procs > 1 and len(task_groups) > 1:
        with mp.Pool(
            processes=n_procs,
            initializer=init_single_detector_worker,
            initargs=(init_args,),
        ) as pool:
            task_results = list(
                pool.imap_unordered(_process_batch_group, task_groups, chunksize=1)
            )
    else:
        _worker_state.clear()
        init_single_detector_worker(init_args)
        task_results = [_process_batch_group(tg) for tg in task_groups]

    # Combine results
    for task_result in task_results:
        for batch_result in task_result:
            batch_indices = batch_result["batch_indices"]
            batch_lnlike_di = batch_result["lnlike_di"]
            # Find positions in full intrinsic_indices array
            # Use searchsorted to find where each batch index appears
            batch_positions = np.searchsorted(intrinsic_indices, batch_indices)
            # Verify we found the right positions
            assert np.all(intrinsic_indices[batch_positions] == batch_indices), \
                "Index mismatch in result combination"
            lnlike_di[:, batch_positions] = batch_lnlike_di

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
