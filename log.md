# dot-pe parallelization — conversation log

## Problem
`inference.run()` takes ~7100 s serial on a 2^20 bank. Goal: proof-of-concept that
finishes in a fraction of that time using an LSF HPC server.

## Profiler breakdown (GW230702_162025/run_3)

| Stage | Time | % |
|---|---|---|
| Incoherent selection (1024 × `run_for_single_detector`) | ~2266 s | 32% |
| Extrinsic sampling (`draw_extrinsic_samples`) | ~549 s | 8% |
| Coherent inference (`create_likelihood_blocks`, 70 blocks) | ~430 s | 6% |
| Lock contention (`_thread.lock.acquire`) | ~2549 s | 36% |
| Setup + postprocess | ~60 s | <1% |

Lock contention dominates because numpy/BLAS was already spawning threads internally,
fighting over cache/memory bus.

## Parallelization targets
- **Incoherent stage**: outer loop over blocks of intrinsic samples — fully independent
- **Coherent stage**: outer loop over i_blocks (one waveform load per i_block, then all e_blocks) — fully independent
- **Extrinsic stage**: stays serial (depends on merged incoherent result)

## Key findings

### 1. BLAS threading doesn't scale beyond ~8 cores
User ran the serial code with more cores available (BLAS-level threading, not
multiprocessing). Negligible improvement beyond 8 cores. Reason: BLAS threads on a
single operation share the same DRAM bus and L3 cache; cache coherence + memory bandwidth
saturates at ~8 threads.

### 2. BLAS threading ≠ Python multiprocessing
The user tested BLAS threading only. Python `multiprocessing.Pool` (one independent
process per block, each with OMP=1) is fundamentally different — no shared cache, no GIL.
Whether it also saturates memory bandwidth on a single node is **untested and unknown**.

### 3. LSF swarm (multi-node) definitively bypasses memory bandwidth
Each LSF job lands on a separate physical node with its own independent DRAM bus.
This is the guaranteed solution regardless of which specific bottleneck (DRAM, cache,
NFS) limits single-node parallelism.

### 4. multiprocessing.Pool might also work (untested)
If the bottleneck is cache contention from BLAS threading (not fundamental DRAM bandwidth),
then independent processes with OMP=1 each might scale beyond 8 on a single node.
Worth testing since it's simpler and avoids per-job startup overhead.

## Architecture constraints (from MISSION.txt)
- Code only in `lsf_swarm/` and `MP/` — do NOT modify `dot_pe/`
- No over-abstraction, no deep wrapper layers, concise
- ~8–20 parallel workers
- Must run on Linux LSF server; code written on macOS
- Queue: `physics-medium` for orchestration, `physics-short`/`short`/`risk` for workers
- Branch: `lsf-swarm`

---

## Implementation status

Both approaches are **implemented and ready to test**.

### Test data
`test_data/setup.py` — idempotent script, run once before benchmarking:
```bash
python test_data/setup.py --n-pool 4
```
Creates:
- `test_data/event/tutorial_event.npz` — gaussian noise + IMRPhenomXPHM injection
- `test_data/bank_small/` — 4 096 samples (smoke-test)
- `test_data/bank_large/` — 262 144 (2^18) samples (benchmark)

### MP approach (`MP/run_mp.py`)
Single file, drop-in for `inference.run()`. Uses `multiprocessing.Pool`.
- Incoherent stage: splits sample range into N chunks, one chunk per worker.
  Each worker reconstructs its own `SingleDetectorProcessor`s and iterates
  over its batches, reusing `h_impb` across detectors (mirrors serial logic).
- Coherent stage: one worker per i_block. Each worker creates a fresh
  `CoherentLikelihoodProcessor`, loads extrinsic data, processes i_block × all e_blocks.
  Main process concatenates results and calls `aggregate_and_save_results()`.
- `OMP_NUM_THREADS=1` set before any imports → inherited by forked workers.
- See README for smoke-test and benchmark commands.

### LSF swarm approach (`lsf_swarm/`)
Three files:
- `worker_incoherent.py` — LSF array worker: reads setup files, scores its block,
  writes `swarm_setup/incoherent/block_{id}.npz`
- `worker_coherent.py` — LSF array worker: reads setup files + selected_inds.npy,
  processes its i_block, writes `swarm_setup/coherent/i_{id}.npz`
- `run_swarm.py` — orchestrator: long-lived physics-medium job that calls
  `prepare_run_objects()`, serializes inputs, submits array jobs, polls with `bjobs`,
  merges results, calls `draw_extrinsic_samples()` and `aggregate_and_save_results()`.
  Fully resumable via `stage_N.done` marker files.
- Single-bank only in current implementation.
- See README for smoke-test and benchmark commands.

### What to report after testing
1. `n_workers` / `--max-concurrent` used
2. **Total wall-clock time** (printed at end of each run)
3. Contents of `<rundir>/run_N/summary_results.json`
4. Any errors or warnings

### Expected outcomes
- If multiprocessing scales → single-node is sufficient, no need for LSF swarm
- If multiprocessing plateaus at ~8 workers → LSF multi-node is required
- If neither scales → NFS I/O is the bottleneck (waveform loading)
