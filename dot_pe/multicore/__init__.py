"""
High-Performance Computing (HPC) multi-core parallelization for DOT-PE inference.

This module provides process-level parallelized versions of the inference pipeline
optimized for 20-100+ core systems. All implementations follow design principles
from multicore.md: process-level parallelism, disabled nested BLAS threading,
tunable batch sizes, and autotuning.

Main entry point: inference_hpc.run_hpc() - drop-in replacement for inference.run()
"""

from .inference_hpc import run_hpc, run_and_profile_hpc
from .config import HPCConfig, autotune_hpc_config, load_cached_config

__all__ = [
    "run_hpc",
    "run_and_profile_hpc",
    "HPCConfig",
    "autotune_hpc_config",
    "load_cached_config",
]
