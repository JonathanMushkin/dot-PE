"""
Multicore parallelization for DOT-PE inference.

This module provides process-level parallelized versions of the inference pipeline
optimized for 20-100+ core systems. All implementations follow design principles
from multicore.md: process-level parallelism, disabled nested BLAS threading,
tunable batch sizes, and autotuning.

Main entry point: inference_multicore.run_multicore() - drop-in replacement for inference.run()
"""

from .inference_multicore import run_multicore, run_and_profile_multicore
from .config import MulticoreConfig, autotune_multicore_config, load_cached_config

# Backward compatibility: old HPC names point to multicore
run_hpc = run_multicore
run_and_profile_hpc = run_and_profile_multicore
HPCConfig = MulticoreConfig
autotune_hpc_config = autotune_multicore_config

__all__ = [
    "run_multicore",
    "run_and_profile_multicore",
    "MulticoreConfig",
    "autotune_multicore_config",
    "load_cached_config",
    # Backward compat
    "run_hpc",
    "run_and_profile_hpc",
    "HPCConfig",
    "autotune_hpc_config",
]
