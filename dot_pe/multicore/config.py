"""
Multicore configuration and autotuning for multi-core parallelization.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .utils_multicore import get_machine_signature, init_worker


@dataclass
class MulticoreConfig:
    """Multicore parallelization configuration."""

    n_procs: int
    i_batch: int
    batches_per_task: int
    i_tile: Optional[int] = None
    e_tile: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_procs": self.n_procs,
            "i_batch": self.i_batch,
            "batches_per_task": self.batches_per_task,
            "i_tile": self.i_tile,
            "e_tile": self.e_tile,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MulticoreConfig":
        """Create from dictionary."""
        return cls(
            n_procs=d["n_procs"],
            i_batch=d["i_batch"],
            batches_per_task=d["batches_per_task"],
            i_tile=d.get("i_tile"),
            e_tile=d.get("e_tile"),
        )


def get_shared_config_cache_dir() -> Path:
    """Return directory for shared (cached) multicore configs."""
    cache_dir = Path.home() / ".cache" / "dotpe"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_shared_config_cache_path() -> Path:
    """Return path to shared cached config for current machine."""
    signature = get_machine_signature()
    return get_shared_config_cache_dir() / f"multicore_config_{signature}.json"


def load_cached_config() -> Optional[MulticoreConfig]:
    """
    Load shared cached multicore config for current machine, if available.

    Returns
    -------
    Optional[MulticoreConfig]
        Cached config or None if not found/invalid
    """
    cache_path = get_shared_config_cache_path()
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r") as f:
            d = json.load(f)
        return MulticoreConfig.from_dict(d)
    except Exception:
        return None


def save_config(config: MulticoreConfig) -> None:
    """Save multicore config to shared cache for current machine."""
    cache_path = get_shared_config_cache_path()
    with open(cache_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)


def autotune_multicore_config(
    benchmark_workload,
    candidate_n_procs: Optional[List[int]] = None,
    candidate_i_batch: Optional[List[int]] = None,
    candidate_batches_per_task: Optional[List[int]] = None,
    max_memory_gb_per_worker: Optional[float] = None,
) -> MulticoreConfig:
    """
    Autotune multicore configuration by benchmarking candidate configs.

    Parameters
    ----------
    benchmark_workload : callable
        Function that takes (n_procs, i_batch, batches_per_task) and returns
        throughput metric (higher is better) and peak_memory_gb_per_worker
    candidate_n_procs : Optional[List[int]]
        Candidate n_procs values (default: [4, 8, 16, 32] capped by cpu_count)
    candidate_i_batch : Optional[List[int]]
        Candidate i_batch values (default: [64, 128, 256, 512, 1024, 2048, 4096])
    candidate_batches_per_task : Optional[List[int]]
        Candidate batches_per_task values (default: [1, 2, 4, 8])
    max_memory_gb_per_worker : Optional[float]
        Maximum memory per worker in GB (safety cap)

    Returns
    -------
    MulticoreConfig
        Best configuration found
    """
    import os

    cpu_count = os.cpu_count() or 4
    if candidate_n_procs is None:
        candidate_n_procs = [min(n, cpu_count) for n in [4, 8, 16, 32] if n <= cpu_count]
    if candidate_i_batch is None:
        candidate_i_batch = [64, 128, 256, 512, 1024, 2048, 4096]
    if candidate_batches_per_task is None:
        candidate_batches_per_task = [1, 2, 4, 8]

    best_config = None
    best_throughput = -1.0

    print("Autotuning multicore configuration...")
    total_candidates = len(candidate_n_procs) * len(candidate_i_batch) * len(candidate_batches_per_task)
    print(f"Testing {total_candidates} candidate configurations...")

    for n_procs in candidate_n_procs:
        for i_batch in candidate_i_batch:
            for batches_per_task in candidate_batches_per_task:
                # Warmup
                try:
                    benchmark_workload(n_procs, i_batch, batches_per_task)
                    benchmark_workload(n_procs, i_batch, batches_per_task)
                except Exception as e:
                    print(f"  Skipping (n_procs={n_procs}, i_batch={i_batch}, batches_per_task={batches_per_task}): {e}")
                    continue

                # Benchmark
                try:
                    throughput, peak_memory_gb = benchmark_workload(n_procs, i_batch, batches_per_task)
                    if max_memory_gb_per_worker and peak_memory_gb > max_memory_gb_per_worker:
                        print(
                            f"  Rejected (n_procs={n_procs}, i_batch={i_batch}, batches_per_task={batches_per_task}): "
                            f"memory {peak_memory_gb:.1f}GB > limit {max_memory_gb_per_worker:.1f}GB"
                        )
                        continue
                    if throughput > best_throughput:
                        best_throughput = throughput
                        best_config = MulticoreConfig(
                            n_procs=n_procs, i_batch=i_batch, batches_per_task=batches_per_task
                        )
                        print(
                            f"  New best: n_procs={n_procs}, i_batch={i_batch}, batches_per_task={batches_per_task}, "
                            f"throughput={throughput:.2f}, memory={peak_memory_gb:.1f}GB"
                        )
                except Exception as e:
                    print(
                        f"  Error (n_procs={n_procs}, i_batch={i_batch}, batches_per_task={batches_per_task}): {e}"
                    )
                    continue

    if best_config is None:
        raise RuntimeError("Autotuning failed: no valid configuration found")

    print(f"\nBest configuration: {best_config}")
    save_config(best_config)
    return best_config
