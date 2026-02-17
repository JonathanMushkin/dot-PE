"""
Utilities for multicore parallelization:
- Worker initialization (disable nested threading)
- Machine introspection (cores, cache, memory)
- Batch partitioning utilities
"""

import hashlib
import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def init_worker():
    """
    Initialize worker process: disable nested BLAS/OpenMP threading.
    Sets env vars (for libraries that read them at import) and uses
    threadpoolctl at runtime so already-loaded BLAS is also limited.
    """
    thread_vars = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "MKL_DYNAMIC",
    )
    for var in thread_vars:
        if var == "MKL_DYNAMIC":
            os.environ[var] = "FALSE"
        else:
            os.environ[var] = "1"
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(limits=1, user_api="blas")
        threadpoolctl.threadpool_limits(limits=1, user_api="openmp")
    except ImportError:
        pass


def get_machine_info() -> Dict[str, Any]:
    """
    Introspect machine hardware (cores, cache sizes, memory).
    Returns dict with machine characteristics.
    """
    info = {
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }

    # Try to get cache sizes (Linux)
    cache_sizes = {}
    if platform.system() == "Linux":
        for cache_level in range(4):
            cache_path = Path(f"/sys/devices/system/cpu/cpu0/cache/index{cache_level}/size")
            if cache_path.exists():
                try:
                    size_str = cache_path.read_text().strip()
                    # Convert "512K" or "8M" to bytes
                    if size_str.endswith("K"):
                        cache_sizes[f"L{cache_level}"] = int(size_str[:-1]) * 1024
                    elif size_str.endswith("M"):
                        cache_sizes[f"L{cache_level}"] = int(size_str[:-1]) * 1024 * 1024
                    else:
                        cache_sizes[f"L{cache_level}"] = int(size_str)
                except Exception:
                    pass
    info["cache_sizes"] = cache_sizes

    # Try to get memory info (psutil if available, else fallback)
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["total_memory_gb"] = mem.total / (1024**3)
        info["available_memory_gb"] = mem.available / (1024**3)
    except ImportError:
        info["total_memory_gb"] = None
        info["available_memory_gb"] = None

    return info


def get_machine_signature() -> str:
    """
    Create stable machine signature for config caching.
    Based on CPU model, core count, cache sizes, and BLAS backend.
    """
    info = get_machine_info()
    parts = [
        info.get("processor", "unknown"),
        str(info.get("cpu_count", 0)),
        str(sorted(info.get("cache_sizes", {}).items())),
    ]

    # Try to detect BLAS backend
    try:
        import numpy as np

        blas_info = np.__config__.get_info("blas_opt_info", {})
        blas_lib = blas_info.get("libraries", ["unknown"])[0] if blas_info.get("libraries") else "unknown"
        parts.append(blas_lib)
    except Exception:
        parts.append("unknown")

    signature_str = "|".join(parts)
    return hashlib.sha256(signature_str.encode()).hexdigest()[:16]


def partition_indices(indices: np.ndarray, batch_size: int) -> List[np.ndarray]:
    """
    Partition indices into batches of specified size.

    Parameters
    ----------
    indices : np.ndarray
        Array of indices to partition
    batch_size : int
        Size of each batch (last batch may be smaller)

    Returns
    -------
    List[np.ndarray]
        List of index batches
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    batches = []
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        batches.append(indices[start:end])
    return batches


def group_batches(batches: List, batches_per_task: int) -> List[List]:
    """
    Group batches into tasks for worker processing.

    Parameters
    ----------
    batches : List
        List of batches to group
    batches_per_task : int
        Number of batches per worker task

    Returns
    -------
    List[List]
        List of task groups, each containing batches_per_task batches
    """
    if batches_per_task <= 0:
        raise ValueError(f"batches_per_task must be > 0, got {batches_per_task}")
    task_groups = []
    for start in range(0, len(batches), batches_per_task):
        end = min(start + batches_per_task, len(batches))
        task_groups.append(batches[start:end])
    return task_groups


def partition_block_pairs(
    i_blocks: List[np.ndarray], e_blocks: List[np.ndarray], pairs_per_task: int
) -> List[List[Tuple[int, int]]]:
    """
    Partition (i_block, e_block) pairs into tasks for parallel processing.

    Parameters
    ----------
    i_blocks : List[np.ndarray]
        List of intrinsic blocks
    e_blocks : List[np.ndarray]
        List of extrinsic blocks
    pairs_per_task : int
        Number of (i_block, e_block) pairs per worker task

    Returns
    -------
    List[List[Tuple[int, int]]]
        List of task groups, each containing pairs_per_task (i_idx, e_idx) tuples
    """
    all_pairs = [(i_idx, e_idx) for i_idx in range(len(i_blocks)) for e_idx in range(len(e_blocks))]
    task_groups = []
    for start in range(0, len(all_pairs), pairs_per_task):
        end = min(start + pairs_per_task, len(all_pairs))
        task_groups.append(all_pairs[start:end])
    return task_groups
