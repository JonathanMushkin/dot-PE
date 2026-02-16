GOAL: Create a High-Performance Multi-Core Inference Path for DOT-PE

Objective

Design and implement a new inference pipeline optimized for large multi-core CPU systems (20–100 cores). This new pipeline must scale efficiently with process-level parallelism while avoiding thread contention, lock overhead, and memory-bandwidth collapse observed in the current implementation.

The new implementation must:
	•	Use coarse-grained multiprocessing (processes, not threads).
	•	Disable nested BLAS/OpenMP threading inside worker processes.
	•	Minimize synchronization and shared mutable state.
	•	Avoid large temporary allocations (especially from repeat or unnecessary reshapes).
	•	Return reduced results from workers (not large intermediate tensors).
	•	Maintain numerical equivalence with the current pipeline.

Scope

Create a dedicated “HPC mode” implementation in a new folder (e.g., hpc/ or dotpe_hpc/) containing:
	•	inference_hpc.py
	•	coherent_processing_hpc.py
	•	likelihood_calculating_hpc.py

The HPC path must parallelize **both**:
	•	Incoherent single-detector likelihood evaluation (already done in dot_pe/multicore).
	•	**Coherent likelihood block creation** (create_likelihood_blocks / (i_block, e_block) pairs). Coherent block creation is typically ~half of runtime and must be parallelized for meaningful multi-core scaling.

The existing code must remain untouched. The HPC path must be opt-in and directly comparable against the original implementation.

Performance Target

The new implementation should:
	•	Show improved wall-clock scaling when increasing from ~4 cores to 20–40+ cores.
	•	Eliminate large cumulative lock/wait overhead.
	•	Improve throughput per batch in coherent inference.
	•	Parallelize coherent likelihood block creation (not only incoherent selection), since that phase is typically ~half of total runtime.

Constraints
	•	Preserve scientific correctness.
	•	Preserve input/output shapes and behavior.
	•	Avoid introducing new heavy dependencies.
	•	Favor clarity and deterministic behavior over excessive abstraction.

Updated HPC Design Principle (for Cursor)

There is no “correct” i_batch

i_batch must be treated as a tunable performance parameter, not a scientific one.

The total number of intrinsic samples (i_total) is dictated by the posterior and may be in the millions.
How you partition those samples into batches (i_batch) is purely an engineering choice.

The optimal value depends on:
	•	Memory bandwidth
	•	Cache sizes (L2/L3)
	•	Whether kernels are compute-bound or memory-bound
	•	Amount of reduction done inside the batch
	•	IPC overhead in multiprocessing
	•	Available RAM per process

⸻

What HPC mode must do
	1.	Expose i_batch as a runtime parameter.
	2.	Allow values as small as ~100.
	3.	Allow values as large as memory allows.
	4.	Optionally include a simple autotuning mode:
	•	run a few trial batch sizes
	•	measure time per sample
	•	choose best throughput

⸻

When small i_batch is good (e.g. 100)
	•	If intermediate tensors scale like (i_batch, n_phi, t, p) and would otherwise be huge.
	•	If memory bandwidth is saturated.
	•	If reduction is aggressive (e.g. you only keep max/lnsumexp).
	•	If per-batch memory footprint dominates.

Small batches:
	•	Improve cache locality.
	•	Reduce peak memory.
	•	Allow better NUMA behavior.
	•	Increase scheduling overhead (so group multiple batches per task).

⸻

When large i_batch is good (e.g. 5000–20000)
	•	If kernels are compute-heavy (waveform gen, spline eval).
	•	If GEMMs benefit from larger dimensions.
	•	If per-batch overhead is non-trivial.
	•	If reduction is cheap relative to compute.

Large batches:
	•	Improve arithmetic intensity.
	•	Reduce per-batch overhead.
	•	Increase risk of memory bandwidth saturation.

⸻

Therefore: HPC code must support
	•	--i-batch
	•	--batches-per-task
	•	--i-tile (optional internal tiling to cap peak memory)
	•	ability to run throughput benchmarks to guide tuning

⸻

Concrete requirement for Cursor

Do NOT hardcode assumptions like:
	•	“i_batch is thousands”
	•	“bigger is better”
	•	“smaller is better”

Instead:
	•	Design kernels that work efficiently for any reasonable i_batch (≥ 10).
	•	Ensure no algorithm depends on large i_batch to function.
	•	Ensure no algorithm explodes memory for large i_batch.
	•	Make batch sizing a first-class runtime parameter.

Add: Autotuning and machine-adaptive execution (HPC mode)

Goal

HPC mode should be able to choose good runtime parameters automatically on a new machine (core count, cache sizes, memory bandwidth effects) by running a very short benchmark and caching the result per host.

What must be auto-tuned (first-class knobs)

Treat these as runtime parameters, not hardcoded assumptions:
	•	n_procs (number of worker processes)
	•	i_batch (intrinsic samples per batch; can be as low as ~100)
	•	batches_per_task (how many batches a worker processes per task to amortize IPC/scheduling)
	•	optional tiling caps: i_tile and/or e_tile if intermediates grow too large

Non-negotiable constraint during tuning and running

In HPC mode, disable nested BLAS/OpenMP threading inside workers:
	•	set in worker initializer (before importing numpy/scipy if possible):
	•	MKL_NUM_THREADS=1
	•	OMP_NUM_THREADS=1
	•	MKL_DYNAMIC=FALSE
	•	optionally NUMEXPR_NUM_THREADS=1

Autotuning must benchmark under the same threading constraints, otherwise it will pick the wrong configuration.

Machine introspection (best-effort)

Implement a small helper that reads:
	•	physical/logical cores (os.cpu_count, psutil if available)
	•	cache sizes (Linux: /sys/devices/system/cpu/cpu0/cache/index*/size)
	•	total memory (psutil.virtual_memory)
	•	optional NUMA topology if available

Use this only to pick candidate configs. Do not rely on cache math alone.

Microbenchmark-based autotuning (required)

Implement autotune_hpc_config(...):
	•	Inputs: a small representative workload slice (one bank, a few batches)
	•	Candidate grids (keep small; seconds total):
	•	n_procs: [min(4, cores), min(8, cores), min(16, cores), min(32, cores)]
	•	i_batch: [64, 128, 256, 512, 1024, 2048, 4096] (extend if memory allows)
	•	batches_per_task: [1, 2, 4, 8]
	•	Benchmark metric: throughput (samples/sec or lnlike/sec), not raw batch time.
	•	Warmup: run 1–2 warmup iterations per candidate to avoid first-call effects.
	•	Choose the best config subject to a safety cap on peak memory per worker (avoid paging/OOM).

Caching tuned results

Cache the chosen config to disk, keyed by a stable machine signature:
	•	CPU model + core count + cache sizes + BLAS backend (MKL/OpenBLAS)
Store as JSON under something like:
	•	~/.cache/dotpe/hpc_config_<signature>.json

CLI behavior

HPC entrypoint should support:
	•	--hpc auto (default): use cached config if present; otherwise run autotune then run inference
	•	--hpc manual: user provides --n-procs --i-batch --batches-per-task ...
	•	--autotune-only: prints recommended config and exits

Additional pitfalls to avoid
	•	Do not run tqdm/progress updates in workers; parent process only.
	•	Do not benchmark with I/O included; preload/memmap before timing.
	•	Do not pick configs based on extremely tiny workloads; benchmark slice must include the real heavy kernels (dh/hh computation + lookup evaluation + reduction), otherwise tuning selects overhead-optimal but throughput-poor settings.