# Multicore Parallelization

Multi-core parallelization for DOT-PE inference, optimized for 20-100+ core systems.

## Overview

This module provides process-level parallelized versions of the inference pipeline, designed to scale efficiently across many CPU cores. Parallelism applies to **incoherent selection** (single-detector likelihood batches) and **coherent likelihood block creation** ((i_block, e_block) pairs). All implementations maintain API equivalence with the original `inference.run()` function.

## Design Principles

1. **Process-level parallelism**: Uses `multiprocessing.Pool` for true parallelism (avoids GIL limitations)
2. **Disabled nested threading**: Worker processes disable BLAS/OpenMP threading (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) to prevent thread contention
3. **Tunable batch sizes**: Configurable `i_batch` (intrinsic batch size) and `batches_per_task` (batches per worker) for optimal load balancing
4. **Autotuning**: Machine-specific configuration cached in shared config (see Configuration Caching)
5. **API equivalence**: `run_multicore()` has identical signature and outputs to `inference.run()`

**Task granularity (40–100+ cores):** For good load balance and to amortize IPC/scheduling overhead, tune so each task takes roughly **~100–500 ms** wall time: increase `batches_per_task` (incoherent phase) and `pairs_per_task` (coherent phase) until tasks are in that range. Defaults are conservative; on large core counts, raising these reduces overhead.

## Usage

### Basic Usage

```python
from dot_pe.multicore import run_multicore

# Drop-in replacement for inference.run()
rundir = run_multicore(
    event="path/to/event.npz",
    bank_folder="path/to/bank",
    n_ext=1000,
    n_phi=100,
    n_t=128,
    n_int=10000,
    # ... all other parameters identical to inference.run()
)
```

### With Custom Multicore Configuration

```python
from dot_pe.multicore import run_multicore, MulticoreConfig

config = MulticoreConfig(
    n_procs=32,           # Number of worker processes
    i_batch=512,          # Intrinsic batch size
    batches_per_task=4,   # Batches per worker task
)

rundir = run_multicore(
    event="path/to/event.npz",
    bank_folder="path/to/bank",
    n_ext=1000,
    n_phi=100,
    n_t=128,
    multicore_config=config,
)
```

### Quick Parameters

You can also override config values directly:

```python
rundir = run_multicore(
    event="path/to/event.npz",
    bank_folder="path/to/bank",
    n_ext=1000,
    n_phi=100,
    n_t=128,
    n_procs=64,           # Override n_procs
    i_batch=1024,        # Override i_batch
    batches_per_task=2,  # Override batches_per_task
)
```

### Autotuning

To find optimal configuration for your machine:

```python
from dot_pe.multicore import autotune_multicore_config

# Define a benchmark workload function
def benchmark_workload(n_procs, i_batch, batches_per_task):
    # Run a small subset of your actual workload
    # Return (throughput, peak_memory_gb_per_worker)
    ...
    return throughput, peak_memory_gb

# Autotune
best_config = autotune_multicore_config(
    benchmark_workload,
    candidate_n_procs=[8, 16, 32, 64],
    candidate_i_batch=[256, 512, 1024, 2048],
    candidate_batches_per_task=[1, 2, 4],
    max_memory_gb_per_worker=16.0,
)

# Use the best config
rundir = run_multicore(..., multicore_config=best_config)
```

## Architecture

### Module Structure

- **`inference_multicore.py`**: Main entry point (`run_multicore()`) - orchestrates the multicore pipeline
- **`single_detector_multicore.py`**: Parallelized single-detector likelihood evaluation
- **`coherent_processing_multicore.py`**: Parallelized coherent likelihood block processing
- **`config.py`**: Multicore configuration and autotuning (shared config cache)
- **`utils_multicore.py`**: Worker initialization, machine introspection, batch partitioning

### Parallelization Strategy

1. **Incoherent Selection**: Parallelized across batches of intrinsic samples
   - Each worker processes groups of batches sequentially
   - Results are combined in the main process

2. **Coherent processing**: Parallelize (i_block, e_block) block creation; combine results sequentially in the main process.
   - Workers create likelihood blocks and write block data to disk
   - Main process loads blocks in (i_block, e_block) order and combines into prob_samples

3. **Other Steps**: Use original implementations (cross-bank selection, extrinsic sampling, aggregation)

## Configuration Caching (shared config)

Multicore configurations are automatically cached per machine (shared across runs) based on hardware signature (CPU model, core count, cache sizes, BLAS backend). Cached configs are stored in `~/.cache/dotpe/multicore_config_<signature>.json`.

To clear shared cache and retune:

```python
from dot_pe.multicore.config import get_shared_config_cache_path
import os

cache_path = get_shared_config_cache_path()
if cache_path.exists():
    os.remove(cache_path)
```

## Performance Considerations

### Batch Size Tuning

- **`i_batch`**: Larger batches reduce overhead but increase memory per worker
  - Start with `single_detector_blocksize` (default: 512)
  - Increase if workers are underutilized
  - Decrease if memory is constrained

- **`batches_per_task`**: More batches per task reduce communication overhead
  - Start with 1-2 for fine-grained parallelism
  - Increase to 4-8 for better throughput on high-core-count systems

### Process Count

- **`n_procs`**: Should match available CPU cores (or slightly less to leave headroom)
  - On large systems: use `n_procs=64` or higher
  - On workstations: use `n_procs=min(cpu_count, 32)`

### Memory

Each worker process loads:
- Event data (shared, loaded once per worker)
- Bank waveforms (loaded per batch)
- Intermediate arrays (depends on batch size)

Monitor memory usage and adjust `i_batch` accordingly.

## Limitations

1. **EventData Serialization**: For multiprocessing, pass `event` as a path (str/Path), not an EventData object
2. **Profiling**: `run_and_profile_multicore()` provides basic profiling; detailed multiprocessing profiling requires additional tools

## Migration from `inference.run()`

The API is identical, so migration is straightforward:

```python
# Before
from dot_pe.inference import run
rundir = run(event=..., bank_folder=..., ...)

# After (multicore)
from dot_pe.multicore import run_multicore
rundir = run_multicore(event=..., bank_folder=..., ...)
```

The only difference is that `event` should be a path (not EventData object) for optimal performance.

## Future Work

- [ ] Parallelization across banks
- [ ] GPU acceleration support
- [ ] Distributed memory (MPI) support for multi-node systems
- [ ] Advanced profiling and monitoring tools
