# LSF Swarm Implementation Plan

## Goal

Parallelize the two bottleneck stages of `inference.run()` using LSF job arrays,
reducing wall time from ~7100 s serial to ~900 s parallel (~8x speedup).

## Pipeline timing (serial baseline from profiler)

| Stage | Time | Parallelizable? |
|---|---|---|
| `prepare_run_objects` (setup) | ~60 s | No — serial, done in orchestrator |
| `collect_int_samples_from_single_detectors` (incoherent) | ~2266 s | **Yes** — independent blocks |
| Cross-bank threshold | ~5 s | No — needs all blocks |
| `draw_extrinsic_samples` | ~549 s | No — stays serial (MISSION) |
| `run_coherent_inference` (coherent) | ~430 s | **Yes** — independent i_blocks |
| `postprocess` / `aggregate_and_save_results` | ~10 s | No |
| **Total** | **~7100 s** | |

**Estimated parallel wall time:** ~900 s (incoherent ~170 s + coherent ~60 s + serial ~624 s)

Queue assignments: `physics-medium` (orchestrator), `physics-short` / `short` / `risk` (workers).

---

## File structure

```
lsf_swarm/
├── MISSION.txt              (exists)
├── PLAN.md                  (this file)
├── run_swarm.py             orchestrator — chains all 5 stages, submits LSF arrays, polls
├── worker_incoherent.py     LSF array worker — scores one block of the bank
├── worker_coherent.py       LSF array worker — processes one i_block, all e_blocks
├── lsf_incoherent.bsub      bsub script for incoherent array job
└── lsf_coherent.bsub        bsub script for coherent array job
```

All intermediate data lives under `{rundir}/swarm_setup/`.

---

## Stages

### Stage 1 — Setup (serial, ~60 s)
- Calls `inference.prepare_run_objects(...)` to get `ctx`.
- Serializes to `{rundir}/swarm_setup/`:
  - `event_data.pkl` — pickle of `EventData`
  - `par_dic_0.json`
  - `swarm_config.json` — all run params (n_int, n_ext, n_phi, n_t, blocksize,
    bank_folder, rundir, n_incoherent_jobs, approximant, fbin, m_arr, etc.)
- Creates `swarm_setup/incoherent/` and `swarm_setup/coherent/` subdirs.
- Marker: `swarm_setup/stage_1.done`

### Stage 2 — Incoherent swarm (~170 s wall)
- Computes `n_blocks = ceil(n_int / blocksize_per_job)`.
- Submits LSF array job with `bsub -J "incoh[1-{n_blocks}]%{max_concurrent}"`.
- Polls `bjobs -noheader {JOBID}` every 30 s until all tasks finish.
- Checks all `swarm_setup/incoherent/block_{i}.npz` exist; raises on any missing.
- Marker: `swarm_setup/stage_2.done`

### Stage 2b — Merge incoherent + threshold (~5 s, inline)
- Concatenates all block npz files (fields: `inds`, `lnlike_di`).
- Applies global threshold: `incoherent_lnlikes.max() - max_incoherent_lnlike_drop`.
- Saves `swarm_setup/selected_inds.npy`.
- Marker: `swarm_setup/stage_2b.done`

### Stage 3 — Extrinsic sampling (~549 s, serial)
- Calls `inference.draw_extrinsic_samples(...)` with selected_inds.
- Writes to standard paths: `{rundir}/extrinsic_samples.feather`,
  `{rundir}/response_dpe.npy`, `{rundir}/timeshift_dbe.npy`.
- Marker: `swarm_setup/stage_3.done`

### Stage 4 — Coherent swarm (~60 s wall)
- Splits `selected_inds` into `i_blocks = inds_to_blocks(inds, blocksize)`.
- Submits LSF array job with `bsub -J "coh[1-{len(i_blocks)}]"`.
- Polls until done. Checks all `swarm_setup/coherent/i_{idx}.npz` exist.
- Marker: `swarm_setup/stage_4.done`

### Stage 5 — Merge coherent + postprocess (~10 s)
- Reconstructs a `CoherentLikelihoodProcessor` for accumulation.
- For each `coherent/i_{idx}.npz`: sets `clp._next_block`, calls
  `combine_prob_samples_with_next_block()`.
- Sums discarded sample counters across all workers.
- Saves `prob_samples.feather` and `intrinsic_sample_processor_cache.json`.
- Calls `inference.aggregate_and_save_results(...)` → `samples.feather`,
  `summary_results.json`.
- Marker: `swarm_setup/stage_5.done`

---

## Worker designs

### `worker_incoherent.py`
```
args: --rundir RUNDIR --bank-folder BANK_FOLDER --block-id BLOCK_ID
```
1. Set `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1` at top.
2. Skip if output `incoherent/block_{block_id}.npz` already exists.
3. Load `event_data.pkl`, `par_dic_0.json`, `swarm_config.json`.
4. Derive `inds = arange(block_id * blocksize_per_job, ...)`.
5. Call `inference.run_for_single_detector(...)` per detector, reusing `h_impb`.
6. Save `inds`, `lnlike_di` to npz.

Adapted from `dotpe-nrsur/nrsur/scripts/incoherent_block.py`; no threshold applied.

### `worker_coherent.py`
```
args: --rundir RUNDIR --bank-folder BANK_FOLDER --i-block-idx IDX
```
1. Set thread env vars to 1.
2. Skip if `coherent/i_{i_block_idx}.npz` already exists.
3. Load setup files. Load `selected_inds.npy`, compute `i_blocks`, get this worker's `i_block`.
4. Reconstruct `CoherentLikelihoodProcessor` with `min_bestfit_lnlike_to_keep=-np.inf`
   (no adaptive cutoff — global cutoff applied in Stage 5 merge).
5. Call `clp.load_extrinsic_samples_data(rundir)`.
6. Load waveforms for `i_block` via `clp.intrinsic_sample_processor.load_amp_and_phase(...)`.
7. For each `e_block` in `e_blocks`:
   - Call `clp.create_a_likelihood_block(h_impb, response_sub, timeshift_sub, i_block, e_block)`.
   - Call `clp.combine_prob_samples_with_next_block()` (trims to `size_limit`).
8. Save `clp.prob_samples` (as npz arrays) + discarded counters +
   `cached_dt_linfree_relative` to output npz.

---

## Key design decisions

1. **No modifications to `dot_pe/`** — workers import from it, write nothing to it.

2. **`swarm_config.json` as single source of truth** — written once in Stage 1, read by
   all workers. No CLI parameter duplication.

3. **Incoherent workers: no threshold** — threshold requires global max; applied in
   Stage 2b after all blocks complete.

4. **Coherent workers: `min_bestfit_lnlike_to_keep=-inf`** — adaptive cutoff would
   give each worker a different local max; set to -inf and apply global cutoff in Stage 5.

5. **Coherent workers accumulate e_blocks internally** — call
   `combine_prob_samples_with_next_block()` after each e_block so `prob_samples` is
   bounded by `size_limit`. Only final `prob_samples` is saved, not raw block arrays.

6. **Thread isolation** — `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1` in both bsub
   scripts and workers.

7. **Resume via marker files** — orchestrator checks `stage_N.done` at startup;
   workers skip if output already exists.

8. **LSF concurrency cap** — `bsub -J "name[1-N]%M"` caps concurrent tasks to M
   (default 16 for incoherent, unbounded for coherent since N is small ~10–30).

---

## Data flow

```
Stage 1:  swarm_setup/event_data.pkl
          swarm_setup/par_dic_0.json
          swarm_setup/swarm_config.json

Stage 2:  swarm_setup/incoherent/block_{id}.npz   (inds, lnlike_di)

Stage 2b: swarm_setup/selected_inds.npy

Stage 3:  {rundir}/extrinsic_samples.feather
          {rundir}/response_dpe.npy
          {rundir}/timeshift_dbe.npy

Stage 4:  swarm_setup/coherent/i_{idx}.npz
            (i, e, o columns of prob_samples; dh, hh; discarded counters;
             cached_dt_linfree_keys, cached_dt_linfree_vals)

Stage 5:  {rundir}/banks/{bank_id}/prob_samples.feather
          {rundir}/banks/{bank_id}/intrinsic_sample_processor_cache.json
          {rundir}/prob_samples.feather
          {rundir}/samples.feather
          {rundir}/summary_results.json
```

---

## Implementation sequence

1. `worker_incoherent.py` + `lsf_incoherent.bsub` — adapt from NRSUR reference, test locally.
2. `worker_coherent.py` + `lsf_coherent.bsub` — new, test locally against a real Stage 3 output.
3. `run_swarm.py` — tie everything together, test end-to-end on small bank (n_int=1024).
4. Full test on HPC with 2^20 bank.
