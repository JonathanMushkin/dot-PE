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
