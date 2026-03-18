# Multi-host MP inference via Dask — Progress

## Status: Implementation complete, smoke test passed, tests pending

## Branch
`dev/mp` (off main, which includes profiling changes from dev/mp-profiling)

## Completed

### Files modified
- `dot_pe/mp_inference.py` — Replaced Pool-based code with Dask:
  - Removed globals `_incoherent_setup`, `_thin_setup`
  - Removed Pool-based workers `_incoherent_chunk_worker`, `_thin_coherent_worker`
  - Removed Pool orchestrators `_collect_incoherent_mp`, `_run_coherent_mp`
  - Added pure Dask workers: `_incoherent_chunk_worker_dask`, `_thin_coherent_worker_dask`
  - Added Dask orchestrators: `_collect_incoherent_dask`, `_run_coherent_dask`
  - Added `scheduler_address` param to `run()` (None → LocalCluster, "host:port" → external)
  - Added `--scheduler-address` CLI flag
  - Fixed: suppress `TimeoutError` in `_cluster.close()` (workers running LAL C code don't exit cleanly)

- `dot_pe/parallel_extrinsic.py` — Replaced Pool-based code with Dask:
  - Removed globals `_ext_setup`, `_accepted_count`, `_count_lock`, `_n_target`
  - Removed `_extrinsic_batch_worker` (Pool worker)
  - Removed `collect_marg_info_parallel` (Pool orchestrator)
  - Added pure Dask worker: `_extrinsic_partition_worker_dask`
  - Added `collect_marg_info_dask` (Dask orchestrator with as_completed + cancel)
  - Updated `draw_extrinsic_samples_parallel`: added `client`, `event_path` params; calls `collect_marg_info_dask`
  - Fixed: `break` after cancellation to avoid calling `.result()` on cancelled futures

- `lsf/run_inference_dask.bsub` — NEW: LSF job script template for multi-node Dask runs

### Smoke test result (LocalCluster, n_workers=4)
```
Stage 1 (setup)              19.0 s
Stage 2 (incoherent)         41.8 s
Stage 3 (cross-bank)          0.0 s
Stage 4 (extrinsic)          433.9 s
Stage 5 (coherent)           94.8 s
Stage 6 (postprocess)         5.3 s
Total wall-clock time: 597.9 s
OK
```
All output files present (samples.feather, Posterior.json, prob_samples.feather, etc.)

## Pending

- [x] Run `pytest tests/ -v` — 13/14 PASSED, 1 FAILED (test_multi_bank_inference_3_banks:
  evidence aggregation mismatch in inference.run() — pre-existing, unrelated to Dask)
- [x] Commit 9d3520b pushed to origin/dev/mp
- [ ] Open PR to main (gh not in PATH — use GitHub UI or install gh CLI)
  - PR URL template: https://github.com/JonathanMushkin/dot-PE/pull/new/dev/mp
- [x] Benchmark n_workers=16 with Dask-friendly config (blocksize=512) — COMPLETE

## Benchmark results (tutorial_gpu_event, n_ext=1024, n_phi=50, seed=42)

| Stage | 4 workers (blocksize=2048) | 16 workers (blocksize=512) | Speedup |
|---|---|---|---|
| Stage 1 (setup, serial) | 19.0 s | 43.8 s | — |
| Stage 2 (incoherent) | 41.8 s | 35.3 s | 1.2× |
| Stage 3 (cross-bank, serial) | 0.0 s | 0.0 s | — |
| Stage 4 (extrinsic) | 433.9 s | 323.1 s | **1.3×** |
| Stage 5 (coherent) | 94.8 s | 50.3 s | **1.9×** |
| Stage 6 (postprocess, serial) | 5.3 s | 6.4 s | — |
| **Total** | **597.9 s** | **462.6 s** | **1.3×** |

Notes:
- Stage 2: 4 chunks/workers → 16 chunks/workers (16 parallel workers each with 4096 samples)
- Stage 5: 3 i_blocks → 12 i_blocks (12 parallel workers, ~2× speedup)
- Stage 4: 4 → 16 parallel partitions; early stopping triggers faster when more workers explore sample space
- Stage 1 setup is slower due to Dask LocalCluster startup (43.8 s vs 19.0 s) — one-time overhead
- On a real multi-node cluster with scheduler_address=, Stage 1 overhead stays fixed while Stages 2/4/5 scale with total worker count across all nodes
- [ ] Multi-node LSF test: submit `lsf/run_inference_dask.bsub` on real cluster
