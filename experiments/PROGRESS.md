# Experiment Progress Log

Each row is one submitted experiment.

| timestamp | job_id | mode | bank | n_ext | n_int | n_workers | status | wall_s |
|-----------|--------|------|------|-------|-------|-----------|--------|--------|

---

## Setup

Bank generation command:
```bash
bsub -q physics-short -n 8 -W 120 -o artifacts/banks/setup_%J.out \
     bash -c 'source "$(conda info --base)/etc/profile.d/conda.sh" && \
              conda activate dot-pe && \
              python test_data/setup.py --n-pool 8 --base-dir artifacts/banks'
```

## Phase A — Smoke tests (code validation, ~minutes)

```bash
python experiments/run_experiment.py --mode serial --bank small --n-int 256 --n-ext 128 --queue physics-short
python experiments/run_experiment.py --mode mp     --bank small --n-int 256 --n-ext 128 --queue physics-short --n-workers 4
python experiments/run_experiment.py --mode swarm  --bank small --n-int 256 --n-ext 128 --queue physics-short
```

## Phase B — Small benchmarks (n_ext=512, bank_small = 2^12)

```bash
python experiments/run_experiment.py --mode serial --bank small --n-ext 512
python experiments/run_experiment.py --mode mp     --bank small --n-ext 512 --n-workers 4
python experiments/run_experiment.py --mode mp     --bank small --n-ext 512 --n-workers 8
python experiments/run_experiment.py --mode swarm  --bank small --n-ext 512
```

## Phase C — Medium benchmarks (n_ext=2048, bank_small)

```bash
python experiments/run_experiment.py --mode serial --bank small --n-ext 2048
python experiments/run_experiment.py --mode mp     --bank small --n-ext 2048 --n-workers 8
python experiments/run_experiment.py --mode swarm  --bank small --n-ext 2048
```

## Phase D — Large benchmarks (n_ext=2048, bank_large = 2^18)

```bash
python experiments/run_experiment.py --mode serial --bank large --n-ext 2048
python experiments/run_experiment.py --mode mp     --bank large --n-ext 2048 --n-workers 8
python experiments/run_experiment.py --mode mp     --bank large --n-ext 2048 --n-workers 20
python experiments/run_experiment.py --mode swarm  --bank large --n-ext 2048
```

---

## Extrinsic cache

After first successful run for a (bank, n_ext, seed) group, cache the samples:
```bash
cp artifacts/experiments/YYYYMMDD_HHMMSS_<mode>_<bank>_next<n>/extrinsic_samples_data.pkl \
   artifacts/extrinsic_cache/<bank>_n<n_ext>_seed0/
```
Then pass `--extrinsic-samples artifacts/extrinsic_cache/<bank>_n<n_ext>_seed0/extrinsic_samples_data.pkl`
to subsequent runs.

## Comparison

```bash
python experiments/compare.py
```

---

## Session Notes

### 2026-03-04 15:19 — Bank generation complete

Banks DONE (job 983627, 831s). All artifacts verified.

---

### 2026-03-04 17:10 — Debugging: all jobs exited immediately

All jobs from first auto_run.sh attempt (990962–990975) exited with:
```
ImportError: attempted relative import with no known parent package
```
Root cause: `_build_script_serial` invoked `python {ROOT}/dot_pe/inference.py` directly, but
`inference.py` uses relative imports and must be run as `python -m dot_pe.inference`.

Also found: `_build_script_serial` had wrong CLI flag names (`--bank` vs `--bank_folder`, etc.).
Fixed those too. `wait_job` in `auto_run.sh` also had a race bug (`bjobs -a` sometimes misses
jobs that exit quickly) — switched to `bhist -l`.

### 2026-03-04 17:19 — Phase A passes; Phase B/C/D submitted

**Fixes applied (cumulative):**
1. `run_experiment.py` serial: `python -m dot_pe.inference` (relative import fix)
2. `inference.py:799`: `rundir = Path(rundir)` (str/str TypeError for mp/swarm)
3. `inference.py:822`: `int(i_int_start) if i_int_start is not None else 0`
4. `inference.py:1137`: `bank_rundir.mkdir(parents=True, exist_ok=True)` before np.savez
5. `run_experiment.py`: `rusage[mem=4096]` for serial and swarm bsub
6. `auto_run.sh` / `run_swarm.py` wait_job: `bjobs -a` → `bhist -l` for exit-status
7. `run_experiment.py _make_rundir`: `_wN_` suffix for mp mode to avoid rundir clashes

**Phase A smoke tests: PASSED** (jobs 991752/991753/991754)
| mode   | n_ext | wall_s | ln_evidence |
|--------|-------|--------|-------------|
| serial | 128   | 193    | 5.347       |
| mp/4w  | 128   | 185    | 5.347       |
| swarm  | 128   | 235    | 5.347       |

Phase B/C/D jobs submitted (17:19 IST). Some mp rundir clashes immediately caught and
resubmitted (fix #7 applied mid-session).

---

### 2026-03-04 22:25 — All Phase B/C/D OOM'd; memory formula added

All jobs submitted at 17:19–17:23 (991802–991826) killed with `TERM_MEMLIMIT`.
Root cause: they were submitted with hardcoded `rusage[mem=4096]` before the memory
formula was written. Resubmitted everything with proper formula:

```python
# _mem_per_slot_mb() in run_experiment.py (first version — later revised)
total_mb = max(4096, int(n_ext/128 * 3072 * 1.2))
```

| job  | mode    | bank  | n_ext | mem_MB |
|------|---------|-------|-------|--------|
| 1088 | serial  | small | 512   | 14745  |
| 1089 | mp/4w   | small | 512   | 14745  |
| 1090 | mp/8w   | small | 512   | 1843/slot |
| 1091 | swarm   | small | 512   | 14745  |
| 1092 | serial  | small | 2048  | 58982  |
| 1093 | mp/8w   | small | 2048  | 7372/slot |
| 1094 | swarm   | small | 2048  | 58982  |
| 1095 | serial  | large | 2048  | 58982  |
| 1096 | mp/8w   | large | 2048  | 7372/slot |
| 1097 | mp/20w  | large | 2048  | 2949/slot |
| 1098 | swarm   | large | 2048  | 58982  |

---

### 2026-03-04 23:40 — Results for small bank; large still running

**Successful results:**
| bank  | n_ext | mode   | n_w | wall_s | ln_evidence | n_effective |
|-------|-------|--------|-----|--------|-------------|-------------|
| small | 128   | serial | —   | 193    | 5.347       | 31.1        |
| small | 128   | mp     | 4   | 185    | 5.347       | 31.1        |
| small | 128   | swarm  | —   | 235    | 5.347       | 31.1        |
| small | 512   | serial | —   | 597    | 3.689       | 1376.7      |
| small | 512   | mp     | 8   | 541    | 3.689       | 1377.1      |
| small | 512   | swarm  | —   | 889    | 3.689       | 1377.1      |
| small | 2048  | serial | —   | 628    | 3.552       | 5492.6      |
| small | 2048  | mp     | 8   | 579    | 3.552       | 5493.4      |
| small | 2048  | swarm  | —   | 718    | 3.552       | 5493.4      |
| large | 2048  | serial | —   | 3167   | 3.328       | 177187.5    |

**Memory formula: first version was wrong** — n_ext doesn't affect peak memory.
Formula went through 3 iterations before converging (see MEMORY.md). Currently:
- serial/swarm/small: flat 13–22 GB; large: 15–180 GB (swarm Stage 4 merge)
- mp/small: flat 30 GB (COW-dominated); mp/large: 25 GB/slot

**Running/pending at 23:40:**
| job  | mode  | bank  | n_w | status  |
|------|-------|-------|-----|---------|
| 2029 | swarm | large | —   | RUN     |
| 2047 | mp    | small | 4   | RUN     |
| 2048 | mp    | large | 8   | RUN     |
| 2049 | mp    | large | 20  | PEND    |

---

### 2026-03-07 01:36 — Phase D large OOM analysis and retry

Jobs 2029/2048 OOM'd. Root cause: memory formula was still wrong for large bank.

**OOM postmortem:**
| job      | mode  | mem_requested | failure point          | root cause |
|----------|-------|---------------|------------------------|------------|
| 2029     | swarm | 90000 MB      | Stage 4 merge          | orchestrator holds all bank+marg_info in RAM |
| 2048     | mp    | 79000 MB      | coherent stage         | each worker independently loads bank waveforms |
| (retry)  | mp    | 95000 MB      | coherent stage         | formula `15000+n×10000` still underestimates |
| 2049     | mp    | 175000 MB     | not reached            | killed after earlier run failed |

**Fix (run_experiment.py `_mem_per_slot_mb`):**
- swarm/large: 90000 → **180000 MB** (Stage 4 merge holds all bank+marg_info at once)
- mp/large: `15000 + n×10000` → **25000 MB/slot flat** (each worker fully reconstructs
  a serial environment; COW does not help here — see below)

**Retried:**
| job   | mode  | bank  | n_w | mem_per_slot | total   |
|-------|-------|-------|-----|--------------|---------|
| 54610 | swarm | large | —   | 180000 MB    | 180 GB  |
| 54611 | mp    | large | 4   | 25000 MB     | 100 GB  |
| 54612 | mp    | large | 8   | 25000 MB     | 200 GB  |

**Why MP memory scales with n_workers (investigation):**

The MP coherent stage (`_coherent_iblock_worker`, `run_mp.py:169`) creates a fresh
`CoherentLikelihoodProcessor` *inside* the worker function rather than inheriting one
from the parent via fork. COW therefore gives no benefit for these allocations.

Each worker independently allocates:
1. `LinearFree` likelihood object (WaveformGenerator + event data)
2. `CoherentLikelihoodProcessor.__init__` → `get_summary()` → `dh_weights_dmpb`, `hh_weights_dmppb`
3. `load_extrinsic_samples_data()` → `response_dpe.npy` (shape: n_det × n_phi × n_ext),
   `timeshift_dbe.npy` (shape: n_det × n_b × n_ext), plus `full_extrinsic_samples.feather`
4. Its own waveform block from disk

With serial ≈ 10 GB for large bank, n_workers=8 → ~80+ GB, consistent with observed OOM.

---

## Thin-worker revision (next step)

The current architecture forces each parallel worker to reconstruct a nearly-complete serial
inference environment. A cleaner approach — demonstrated by the `dotpe-nrsur` reference
implementation — is to pre-save all shared data to disk and make workers thin pure-numpy
tasks that load only what they need.

### Reference: `~/dotpe-nrsur/nrsur/` (`~/GW/Collaboration-gw/…` equivalent)

This unsupervised implementation (by Matías Zaldarriaga) shows how to decompose dot-PE into
embarrassingly parallel SLURM array jobs. It targets the NRSur7dq4 waveform model but the
cluster workflow patterns are general. Key files:

| Script | Stage | What it does |
|--------|-------|-------------|
| `launch_incoherent_slurm.sh` | 2b | Submit SLURM array; auto-detects missing blocks; idempotent |
| `incoherent_block.py` | 2b | Thin worker: load setup + one waveform block → score → save `.npz` |
| `merge_incoherent.py` | 2c | Merge all `.npz` scores, select survivors |
| `launch_inference_slurm.sh` | 3  | Submit coherent inference job |
| `slurm_bank_block.sh` | 1  | SLURM wrapper for bank generation (NRSur: 5 threads/task, 0.7s/waveform) |

**Key pattern for thin workers:**
1. **Setup phase** (orchestrator): compute all shared summary data — event weights, extrinsic
   samples, reference waveform — and save to files in `setup/`
2. **Worker phase** (SLURM array): each task loads `setup/` + its waveform block → does pure
   numpy computation → writes one result `.npz`
3. **Merge phase** (orchestrator): concatenate `.npz` files, aggregate posteriors

With this pattern, per-worker peak memory for coherent scoring is just the waveform block
(~4096 templates × n_modes × n_fbin ≈ 1–2 GB) plus the pre-loaded summary arrays.
Total cluster memory = n_workers × ~2 GB instead of n_workers × ~25 GB.

**Optimizations found in nrsur (Phase 4 extrinsic sampling bottleneck):**
- Skip lnlike filter entirely when threshold ≤ 0 (the default) → **3.33× speedup for Phase 4**
- Early-stop lnlike filter after collecting `max(2×n_remaining, 32)` valid candidates → **2.58× speedup**
- Vectorized `_set_d_h_weights` batching was tried and reverted — caused cache thrashing (+35% regression)
- Remaining targets: QMC short-circuit (exit after 16 accepted; ~90% QMC time savings),
  disk pre-loading to `/dev/shm` for NFS-slow clusters

**What needs investigation before designing thin coherent workers:**
- Exact size of `response_dpe` and `timeshift_dbe` for large bank (determine what must be
  pre-saved vs. recomputed per block)
- Whether `get_summary()` or the extrinsic arrays dominate per-worker memory
- What the parent process holds at fork time (to quantify true COW savings if we fix worker init)

---

### 2026-03-07 14:35 — Phase D large: all modes complete (thin-worker refactor)

All three large-bank runs succeeded with `ln_evidence = 3.328` matching serial baseline.

**Successful results:**
| bank  | n_ext | mode   | n_w | wall_s | ln_evidence | n_effective | rundir |
|-------|-------|--------|-----|--------|-------------|-------------|--------|
| large | 2048  | serial | —   | 3167   | 3.328       | 177187.5    | 20260304_222558_serial_large_next2048 |
| large | 2048  | mp     | 4   | 1783   | 3.328       | 177187.5    | 20260307_130338_mp_w4_large_next2048 |
| large | 2048  | mp     | 8   | 949    | 3.328       | 177187.5    | 20260307_134223_mp_w8_large_next2048 |
| large | 2048  | swarm  | —   | ~1390* | 3.328       | 177187.5    | 20260307_120732_swarm_large_next2048 |

*Swarm wall time estimated from stage marker timestamps: stage_1.done→stage_4.done = 1378s,
stage 5 ~10s. Orchestrator OOM'd at 30 GB (stage 4 workers ran on independently),
then resumed via stage5 retry job; full-retry wall clock was 4815s (dominated by
`prepare_run_objects` I/O on slow NFS). Updated memory formula: swarm/large orchestrator
→ 40 GB.

**Memory usage (thin workers):**
- mp/w8/large: Max Memory 48000 MB (earlier thick-worker OOM run); thin worker run succeeded
- swarm/large orchestrator OOM at 30000 MB → now requesting 40000 MB (adequate)
- Coherent workers (thin): ~1–2 GB each (not observed directly; Stage 4 completed in 8 min
  wall clock for 131 blocks with max 20 concurrent)

**mp speedup (large bank, n_ext=2048):**
- serial → mp/4w: 3167 → 1783s = 1.8x
- serial → mp/8w: 3167 → 949s  = 3.3x

---

### 2026-03-08 — Phase E: parallel extrinsic sampling (bench_extrinsic)

**New code:** `dot_pe/parallel_extrinsic.py`, `experiments/bench_extrinsic.py`,
`--n-ext-workers` added to `MP/run_mp.py`, `lsf_swarm/run_swarm.py`,
`experiments/run_experiment.py`.

Design: fork+COW pool dispatches 1024-sample batches to workers; each worker calls
`get_marg_info_batch_multibank()` on the inherited generator; main process collects
16 MI objects then breaks the pool early.

**Smoke tests (2026-03-08 00:15, jobs 77388/77389/77398):**
| job   | mode               | what was tested                        | result |
|-------|--------------------|----------------------------------------|--------|
| 77388 | bench serial (w=1) | `inference.draw_extrinsic_samples()`   | OOM at 8 GB (too small) — code ran correctly before kill |
| 77389 | bench mp (w=4)     | `draw_extrinsic_samples_parallel()`    | OOM at 16 GB (too small) — parallel fork confirmed started |
| 77398 | swarm --n-ext-workers 4 | `run_swarm.py` Stage 3 parallel wiring | **DONE** 298s, 2.7 GB peak ✓ |

Swarm wiring confirmed correct. Serial/MP smoke OOMs were due to undersized memory
allocation (8–16 GB requested vs. ~9–15 GB needed for the large-bank generator).

**Full benchmark (2026-03-08 00:34, jobs 77816–77820), large bank, n_ext=2048:**
| job   | n_workers | mem_alloc | result            | wall_s | peak_mem |
|-------|-----------|-----------|-------------------|--------|----------|
| 77816 | 1 (serial)| 18 GB     | **DONE**          | 797    | 8.6 GB   |
| 77817 | 4         | 24 GB     | TERM_MEMLIMIT     | 815    | 24 GB    |
| 77818 | 8         | 24 GB     | TERM_MEMLIMIT     | 759    | 24 GB    |
| 77819 | 16        | 32 GB     | TERM_MEMLIMIT     | 746    | 32 GB    |
| 77820 | 20        | 40 GB     | TERM_MEMLIMIT     | 746    | 40 GB    |

**Root cause of all parallel OOMs — COW defeated by Python refcounting:**

The COW assumption was wrong for the extrinsic stage. When a forked worker reads
`_d_h_weights` / `_h_h_weights` from the inherited generator, Python's reference
counting modifies the object header on each shared page, triggering a physical page
copy for every accessed page. Unlike the coherent thin workers (which load their own
data from disk and never touch parent arrays), the extrinsic workers read the parent's
large numpy arrays directly, causing full duplicates of the ~8.6 GB generator in each
worker. Observed memory scaling: `(n_workers + 1) × ~8.6 GB`.

**Status:** parallel extrinsic stage is NOT yet viable with the current fork+COW
architecture. Serial baseline measured: **797s** for large bank, n_ext=2048.

---

### 2026-03-08 11:18 — Profile-split result (job 83591)

Ran `bench_extrinsic.py --profile-split` on large bank, n_ext=2048.

**Step 1/2 split:**
| step | function | wall_s | pct |
|------|----------|--------|-----|
| Step 1 | `_get_many_dh_hh` | 0.6 | 0.1% |
| Step 2 | `get_marginalization_info` | 0.2 | 0.0% |
| Other | (filter, I/O, marginalization internals) | 604.7 | 99.9% |

**Top hotspots (by tottime):**
- `numpy.fromfile`: 157s — loading 128 waveform block files from NFS (all 64 blocks = 12.7 GB)
- scipy splines: 72s — `_kde_t_arrival_prob` per-sample
- `take_along_axis`: 55s; `ufunc.reduce`: 46s; `argsort`: 25s — marginalization ops

**Root cause revision:** The COW failure was NOT from `_d_h_weights` object headers
(those arrays are tiny, ~1 MB). The actual problem was random batches of 1024 samples
spanning ALL 64 block files → each worker loads ~12 GB of waveform data per batch.

**Fix: block-partitioned workers** (implemented 2026-03-08)

Instead of random-sample batches, partition the 64 waveform blocks across n_workers.
Each worker loads only its 8 blocks (8×99 MB×2 = 1.6 GB), not all 64. Workers run to
completion on their block range; main process selects best n_combine MI objects by
n_effective_prior.

Key facts that make this work:
- Workers only RETURN MI objects that pass n_effective_prior threshold (~2 per worker)
- Workers access `_d_h_weights` etc. READ-only → data pages COW-shared (tiny header COW)
- Per-worker new allocations: ~600 MB (waveform rows + intermediate)
- Estimated total for 8 workers: parent 8.4 GB + 8×0.6 = ~13 GB

**Submitted job 85753** (2026-03-08 15:00): `bench_block` w=1,4,8, 20 GB, large bank.

---

### 2026-03-08 15:34 — Block-partitioned benchmark results (jobs 87502–87508)

| job   | n_workers | mem_limit | result        | wall_s | peak_mem | notes |
|-------|-----------|-----------|---------------|--------|----------|-------|
| 87502 | 1 (serial)| 20 GB     | **DONE**      | 588    | 8.4 GB   | baseline |
| 87504 | 4         | 10 GB     | TERM_RUNLIMIT | >1800  | 2.9 GB/slot | workers ran all samples, no early stop |
| 87507 | 8         | 20 GB     | TERM_RUNLIMIT | >1800  | 2.9 GB/slot | same |
| 87508 | 16        | 40 GB     | TERM_RUNLIMIT | >1800  | —        | same |

**Root cause of timeouts:** Workers received ALL samples in their block range (~8,346
per worker for w=8) and ran every one with no early stop. Serial finds 16 MI objects
after only 1,024 random samples (1.6% acceptance rate); without early stopping, each
worker does 8× more work than serial, taking ~4,000s per worker vs 488s serial.

**Fix: shared counter (`multiprocessing.Value`) for collective early stop (2026-03-09)**

Added to `parallel_extrinsic.py`:
- `_accepted_count = Value('i', 0)`, `_count_lock = Lock()`, `_n_target = n_combine`
  set before `Pool` creation; inherited via fork into shared POSIX mmap (not COW heap).
- Worker loops in sub-batches of 1,024; checks `_accepted_count.value >= _n_target`
  before each sub-batch (lockless read); increments counter under lock when MI objects
  are accepted.
- Workers from any block stop as soon as the global count reaches 16 — no per-worker
  quota, no variance problem.

**LSF memory accounting confirmed (2026-03-09):**

From `/usr/share/lsf/conf/lsf.conf`:
- `LSF_LINUX_CGROUP_ACCT=Y` + `LSF_REPLACE_PIM_WITH_LINUX_CGROUP=Y`: cgroup accounting
- `LSB_JOB_MEMLIMIT=Y`: limit applies to total job (all processes); limit = rusage[mem] × n_slots
- Cgroup counts shared (COW) pages **once** for the whole job

**Coordinated-stop benchmark (2026-03-09, jobs 112789–112791, 113967–113969):**

All six jobs hit EXACTLY their memory limit — meaning actual usage exceeded ALL of them.

| job    | n_workers | mem_limit (total) | result       |
|--------|-----------|-------------------|--------------|
| 112789 | 4         | 10 GB             | TERM_MEMLIMIT|
| 112790 | 8         | 20 GB             | TERM_MEMLIMIT|
| 112791 | 16        | 40 GB             | TERM_MEMLIMIT|
| 113967 | 4         | 24 GB             | TERM_MEMLIMIT|
| 113968 | 8         | 36 GB             | TERM_MEMLIMIT|
| 113969 | 16        | 56 GB             | TERM_MEMLIMIT|

**Root cause: COW is fully defeated.** Memory scales as `(1 + n_workers) × 8,400 MB`
(parent generator copied in full per worker), consistent with all observed OOMs:

| n_workers | predicted actual | limits tried       | outcome |
|-----------|-----------------|-------------------|---------|
| 4         | 48,400 MB       | 10 GB, 24 GB      | OOM ✗  |
| 8         | 88,400 MB       | 20 GB, 36 GB      | OOM ✗  |
| 16        | 168,400 MB      | 40 GB, 56 GB      | OOM ✗  |

`get_marg_info_batch_multibank` must write into generator arrays internally during
computation (despite `use_cached_dt=False`), COW-copying every data page per worker.
This is the exact Phase D thick-worker problem: identical memory scaling.

**Status: the fork+COW architecture is fundamentally broken for the extrinsic stage.**
The coordinated-stop logic (`Value` counter) is correct and can be kept; the problem
is the memory model. Fix options (same as before, now confirmed necessary):

- **Option B (shared memory):** move large generator arrays into `SharedMemory`/`/dev/shm`
  before forking; workers get views into truly-shared pages that Python refcounting
  cannot COW-copy. Requires identifying which arrays dominate the 8.4 GB.
- **Option C (thin-worker):** precompute and save what workers need to disk, free the
  generator before forking. Same pattern as coherent thin-worker refactor. Most work
  but most robust.

### Phase E next steps: paths to fix parallel extrinsic memory (decided 2026-03-09)

**What we know:**
- Serial extrinsic (w=1): 588s, 8.4 GB peak — this is the target to beat
- Bottleneck: ~157s disk I/O (waveform blocks) + ~350s marginalization ops; dh/hh
  computation is <0.1% of time (confirmed by profile-split job 83591)
- COW is fully defeated: actual memory = `(1 + n_workers) × 8.4 GB`, regardless of
  how much we allocate — `get_marg_info_batch_multibank` writes into generator arrays
  internally, COW-copying every page per worker
- Coordinated early-stop (`Value` counter, implemented in `parallel_extrinsic.py`)
  is correct and can be kept; it is NOT the source of the memory problem
- ~~Option A (split dh/hh from MI)~~: eliminated — dh/hh is <0.1% of runtime,
  parallelizing it gives no meaningful speedup

**Option B — POSIX shared memory for large generator arrays**

Move the large numpy arrays out of the Python heap into
`multiprocessing.shared_memory.SharedMemory` (or `/dev/shm` mmap) before forking.
Workers get `np.ndarray` views into the shared segment. OS never COW-copies those
pages; Python refcounting only touches the small object header (few hundred bytes),
not the data buffer.

Steps:
1. Run `tracemalloc` or `pympler` on a serial run to identify which attributes of
   `ext_generator` dominate the 8.4 GB (candidates: `_d_h_weights`, `_h_h_weights`,
   internal QMC lookup tables, `coherent_score` internals)
2. In `collect_marg_info_parallel`, before forking:
   - Copy each large array into a named `SharedMemory` block
   - Replace generator attribute with `np.ndarray(..., buffer=shm.buf)` view
3. Fork workers via `Pool` — they inherit the generator with shm-backed arrays
4. After pool exits: `shm.unlink()` for each block

Pros: minimal architecture change, keeps the existing worker structure intact
Cons: requires knowing the generator internals; must cover ALL large arrays or COW
still fires; `SharedMemory` cleanup on crash needs care

**Option C — Thin-worker redesign (coherent-stage blueprint)**

Same pattern as `dot_pe/thin_coherent.py` + `lsf_swarm/worker_coherent.py`:
1. **Precompute phase** (main process): run whatever setup `get_marg_info_batch_multibank`
   needs, save outputs to disk under a `setup/` dir (npz/pkl files)
2. **Free phase**: `del ext_generator` and `gc.collect()` before forking — parent
   footprint drops to near zero
3. **Worker phase**: each worker loads only `setup/` + its waveform blocks (~1.6 GB)
   and runs a thin version of the marginalization
4. **Collect phase**: main process reads worker outputs, picks best n_combine

Pros: provably correct memory model (demonstrated for coherent stage); workers have
~1.6 GB footprint regardless of n_workers
Cons: most implementation work; requires understanding which parts of the generator
construction can be precomputed vs. must be per-sample

**Recommended first step whichever path is chosen:**

Run a `tracemalloc` diagnostic on a serial extrinsic run to get a precise breakdown
of the 8.4 GB by object/array. This informs both options:
- For B: tells us exactly which arrays to move to SharedMemory
- For C: tells us what the "thin" setup files need to contain

```python
import tracemalloc
tracemalloc.start()
# ... build ext_generator, run one batch ...
snapshot = tracemalloc.take_snapshot()
for stat in snapshot.statistics('lineno')[:20]:
    print(stat)
```

---

### 2026-03-09 — Phase E RESOLVED: SUB_BATCH_SIZE fix eliminates OOM

**Root cause (final diagnosis, jobs 124483 + 131190):**

COW of the 8.4 GB generator was NOT the culprit.
Generator at rest = only **1225 MB** (Python-level; 365 MB numpy arrays + Python heap).
The OOM came from temporary allocations inside each `get_marg_info_batch_multibank` call:

| source | allocations | memory | mechanism |
|--------|-------------|--------|-----------|
| `marginalization.py:395` `update()` | 1740 concatenations × 503 KiB | 896 MB | `np.concatenate` on d_h/h_h arrays for QMC doubling |
| `cogwheel/utils.py:230` | 368 × 238 KiB | 90 MB | QMC sequence allocation |
| `marginalization.py:356` `__post_init__` | 184 × 460 KiB | 87 MB | `sparse.coo_array.toarray()` → dense 32768-element array |
| **Total (batch-profile, B=1024)** | | **+7652 MB** | freed after call, but malloc heap retains pages (high-water mark) |

**Fix (commit `aebb80e`, 2026-03-09):**
- `SUB_BATCH_SIZE`: 1024 → **128** samples per sub-batch
- Added `malloc_trim(0)` (libc) after each sub-batch to return freed heap pages to OS
- Memory scales ~linearly with batch size: B=128 → **+1090 MB** per sub-batch (measured, job 133787)

**Verification (jobs 133787, 134727, 134728, 136034):**

| job    | test                  | n_workers | result       | wall_s | peak_mem | host |
|--------|-----------------------|-----------|--------------|--------|----------|------|
| 133787 | batch-profile B=128   | 1         | **DONE**     | —      | 2669 MB total | cn386 |
| 134727 | bench parallel-only   | 4         | **DONE**     | 306.7s | **9808 MB** ✓ | cn364 |
| 136034 | bench parallel-only   | 8 (clean) | **DONE**     | 525.1s | **17096 MB** ✓ | cn371 |
| 136149 | bench serial baseline | 1         | **DONE**     | 711.7s | 8566 MB | cn353 |

**Speedup summary (bench_extrinsic, large bank, n_ext=2048):**
| n_workers | wall_s | peak_MB | speedup |
|-----------|--------|---------|---------|
| 1 (serial)| 711.7  | 8566    | 1.0×    |
| 4         | 306.7  | 9808    | **2.32×** |
| 8         | 525.1  | 17096   | 1.35×†  |

†n=8 ran on different host than serial; NFS cache variability dominates. n=4 result is reliable.
Note: bench_extrinsic timing includes generator build (150-200s); actual parallel speedup for
the sampling portion itself is higher. Serial 711.7s vs full-inference serial 605s reflects
NFS cache warming from prior inference stages.

**Phase E: COMPLETE.** Both n=4 and n=8 run successfully within 20 GB.

---

### 2026-03-10 00:29 — Phase F: port dotpe-nrsur lnlike optimizations

Ported two sequential optimizations from `dotpe-nrsur` into `dot_pe/coherent_processing.py`:

1. **Skip lnlike filter when threshold ≤ 0** (the default): with `min_marg_lnlike_for_sampling=0.0`,
   virtually all candidates pass — every `lnlike()` call was wasted. Now takes first `n_to_process`
   candidates directly (batch is already shuffled → unbiased). Applies to both `get_marg_info_batch`
   and `get_marg_info_batch_multibank`.

2. **Early-stop lnlike filter**: for non-zero thresholds, stop after collecting
   `max(2×n_remaining, 32)` valid candidates instead of evaluating the full batch.
   `n_remaining = n_combine - len(marg_info_i)` is passed as `max_filter_valid` from both callers.

Reference speedups measured in dotpe-nrsur (NRSur waveform, N=200 survivors):
- Skip filter alone: **3.33× Phase 4 speedup**
- Early-stop alone: **2.58× Phase 4 speedup**

For our XPHM bank, lnlike is cheap (~0.13s/call vs ~1s for NRSur), so the skip gives a smaller
but still meaningful gain. The `max_filter_valid` reduction in QMC work (loading fewer waveforms
+ running fewer QMC chains) is the larger benefit here.

**Jobs submitted (2026-03-10 00:29–00:33):**

| job    | mode   | bank      | n_ext | n_workers | purpose                     | status  |
|--------|--------|-----------|-------|-----------|------------------------------|---------|
| 168214 | serial | small     | 128   | —         | smoke test (correctness)     | pending |
| 168227 | serial | large     | 2048  | —         | full bench vs baseline 3167s | pending |
| 168730 | mp/w8  | real event| —     | 8 (ext=4) | real event vs baseline ~1250s| pending |
