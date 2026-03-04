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
## SESSION NOTES 2026-03-04 15:19 IST

Banks DONE (job 983627, 831s). All artifacts verified.

---

## SESSION NOTES 2026-03-04 17:10 IST

### Fixed
- `run_experiment.py` `_build_script_serial`: corrected CLI flags:
  - `--bank` → `--bank_folder`
  - `--n-ext` → `--n_ext`
  - `--n-int` → `--n_int`
  - `--extrinsic-samples` → `--extrinsic_samples`
- (run_mp.py and run_swarm.py use hyphenated flags — already correct)

### auto_run.sh ran — all phases EXITed (jobs 990962–990975)
- Phase D large jobs (990972–990975) were RUN at time of check but also EXITed
- `wait_job` bug: jobs exit so fast that `bjobs -noheader -a` finds no EXIT lines
  (LSF clears them before the check) → falsely reports "finished OK"

### Root cause of EXIT: relative import error in inference.py
```
ImportError: attempted relative import with no known parent package
```
`_build_script_serial` calls `python {ROOT}/dot_pe/inference.py`
but inference.py uses relative imports → must be invoked as a module.

### NEXT STEP
1. Fix `_build_script_serial` in run_experiment.py:
   `python {ROOT}/dot_pe/inference.py` → `python -m dot_pe.inference`
   (run from ROOT with `cd {ROOT} && python -m dot_pe.inference ...`)
2. Check if mp/swarm workers have same issue (run_mp.py, run_swarm.py, worker scripts)
3. Kill any still-running Phase D jobs if broken
4. Rerun: `bash experiments/auto_run.sh --skip-banks`

[17:02:14] ==============================
[17:02:14] dot-pe benchmark auto_run.sh
[17:02:14] ==============================
[17:02:14] Log file: /home/projects/barakz/jonatahm/dot-pe-future/experiments/auto_run.log
[17:02:14] 
[17:02:14] === Phase A: smoke tests ===
[17:02:14] Submitting: mode=serial bank=small n_ext=128 n_int=256 n_workers=-

| 2026-03-04 17:02 | 990947 | serial | small | 128 | 256 | - | pending | — |
[17:02:14] Submitting: mode=mp bank=small n_ext=128 n_int=256 n_workers=4

| 2026-03-04 17:02 | 990948 | mp | small | 128 | 256 | 4 | pending | — |
[17:02:14] Submitting: mode=swarm bank=small n_ext=128 n_int=256 n_workers=-

| 2026-03-04 17:02 | 990949 | swarm | small | 128 | 256 | - | pending | — |
[17:02:14]   waiting for serial/small/next128 (job [17:02:14] Submitting: mode=serial bank=small n_ext=128 n_int=256 n_workers=-
Job <990947> is submitted to queue <physics-short>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170214_serial_small_next128
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170214_serial_small_next128/run.log
990947) ...
[17:02:15]   serial/small/next128 (job [17:02:14] Submitting: mode=serial bank=small n_ext=128 n_int=256 n_workers=-
Job <990947> is submitted to queue <physics-short>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170214_serial_small_next128
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170214_serial_small_next128/run.log
990947) finished OK
[17:02:15]   waiting for mp/small/next128 (job [17:02:14] Submitting: mode=mp bank=small n_ext=128 n_int=256 n_workers=4
Job <990948> is submitted to queue <physics-short>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170214_mp_small_next128
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170214_mp_small_next128/run.log
990948) ...
[17:02:15]   mp/small/next128 (job [17:02:14] Submitting: mode=mp bank=small n_ext=128 n_int=256 n_workers=4
Job <990948> is submitted to queue <physics-short>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170214_mp_small_next128
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170214_mp_small_next128/run.log
990948) finished OK
[17:02:15]   waiting for swarm/small/next128 (job [17:02:14] Submitting: mode=swarm bank=small n_ext=128 n_int=256 n_workers=-
Job <990949> is submitted to queue <physics-short>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170214_swarm_small_next128
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170214_swarm_small_next128/run.log
990949) ...
[17:02:15]   swarm/small/next128 (job [17:02:14] Submitting: mode=swarm bank=small n_ext=128 n_int=256 n_workers=-
Job <990949> is submitted to queue <physics-short>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170214_swarm_small_next128
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170214_swarm_small_next128/run.log
990949) finished OK
[17:02:15] Phase A: smoke tests: all jobs OK
[17:02:15] 
[17:02:15] === Phase B: small/n_ext=512 ===
[17:02:15] Submitting: mode=serial bank=small n_ext=512 n_int=- n_workers=1

| 2026-03-04 17:02 | 990950 | serial | small | 512 | full | 1 | pending | — |
[17:02:15] Submitting: mode=mp bank=small n_ext=512 n_int=- n_workers=4

| 2026-03-04 17:02 | 990951 | mp | small | 512 | full | 4 | pending | — |
[17:02:15] Submitting: mode=mp bank=small n_ext=512 n_int=- n_workers=8

| 2026-03-04 17:02 | 990952 | mp | small | 512 | full | 8 | pending | — |
[17:02:15] Submitting: mode=swarm bank=small n_ext=512 n_int=- n_workers=-

| 2026-03-04 17:02 | 990953 | swarm | small | 512 | full | - | pending | — |
[17:02:16]   waiting for serial/small/next512 (job [17:02:15] Submitting: mode=serial bank=small n_ext=512 n_int=- n_workers=1
Job <990950> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170215_serial_small_next512
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170215_serial_small_next512/run.log
990950) ...
[17:02:16]   serial/small/next512 (job [17:02:15] Submitting: mode=serial bank=small n_ext=512 n_int=- n_workers=1
Job <990950> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170215_serial_small_next512
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170215_serial_small_next512/run.log
990950) finished OK
[17:02:16]   waiting for mp/small/next512 (job [17:02:15] Submitting: mode=mp bank=small n_ext=512 n_int=- n_workers=4
Job <990951> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170215_mp_small_next512
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170215_mp_small_next512/run.log
990951) ...
[17:02:16]   mp/small/next512 (job [17:02:15] Submitting: mode=mp bank=small n_ext=512 n_int=- n_workers=4
Job <990951> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170215_mp_small_next512
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170215_mp_small_next512/run.log
990951) finished OK
[17:02:16]   waiting for mp/small/next512 (job [17:02:15] Submitting: mode=mp bank=small n_ext=512 n_int=- n_workers=8
Job <990952> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170215_mp_small_next512
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170215_mp_small_next512/run.log
990952) ...
[17:02:16]   mp/small/next512 (job [17:02:15] Submitting: mode=mp bank=small n_ext=512 n_int=- n_workers=8
Job <990952> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170215_mp_small_next512
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170215_mp_small_next512/run.log
990952) finished OK
[17:02:16]   waiting for swarm/small/next512 (job [17:02:15] Submitting: mode=swarm bank=small n_ext=512 n_int=- n_workers=-
Job <990953> is submitted to queue <physics-short>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170216_swarm_small_next512
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170216_swarm_small_next512/run.log
990953) ...
[17:02:16]   swarm/small/next512 (job [17:02:15] Submitting: mode=swarm bank=small n_ext=512 n_int=- n_workers=-
Job <990953> is submitted to queue <physics-short>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170216_swarm_small_next512
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170216_swarm_small_next512/run.log
990953) finished OK
[17:02:16] Phase B: small/n_ext=512: all jobs OK
[17:02:16] 
[17:02:16] === Phase C: small/n_ext=2048 ===
[17:02:16] Submitting: mode=serial bank=small n_ext=2048 n_int=- n_workers=1

| 2026-03-04 17:02 | 990954 | serial | small | 2048 | full | 1 | pending | — |
[17:02:16] Submitting: mode=mp bank=small n_ext=2048 n_int=- n_workers=8

| 2026-03-04 17:02 | 990955 | mp | small | 2048 | full | 8 | pending | — |
[17:02:16] Submitting: mode=swarm bank=small n_ext=2048 n_int=- n_workers=-

| 2026-03-04 17:02 | 990956 | swarm | small | 2048 | full | - | pending | — |
[17:02:17]   waiting for serial/small/next2048 (job [17:02:16] Submitting: mode=serial bank=small n_ext=2048 n_int=- n_workers=1
Job <990954> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170216_serial_small_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170216_serial_small_next2048/run.log
990954) ...
[17:02:17]   serial/small/next2048 (job [17:02:16] Submitting: mode=serial bank=small n_ext=2048 n_int=- n_workers=1
Job <990954> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170216_serial_small_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170216_serial_small_next2048/run.log
990954) finished OK
[17:02:17]   waiting for mp/small/next2048 (job [17:02:16] Submitting: mode=mp bank=small n_ext=2048 n_int=- n_workers=8
Job <990955> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170216_mp_small_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170216_mp_small_next2048/run.log
990955) ...
[17:02:17]   mp/small/next2048 (job [17:02:16] Submitting: mode=mp bank=small n_ext=2048 n_int=- n_workers=8
Job <990955> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170216_mp_small_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170216_mp_small_next2048/run.log
990955) finished OK
[17:02:17]   waiting for swarm/small/next2048 (job [17:02:16] Submitting: mode=swarm bank=small n_ext=2048 n_int=- n_workers=-
Job <990956> is submitted to queue <physics-short>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170217_swarm_small_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170217_swarm_small_next2048/run.log
990956) ...
[17:02:17]   swarm/small/next2048 (job [17:02:16] Submitting: mode=swarm bank=small n_ext=2048 n_int=- n_workers=-
Job <990956> is submitted to queue <physics-short>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170217_swarm_small_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170217_swarm_small_next2048/run.log
990956) finished OK
[17:02:17] Phase C: small/n_ext=2048: all jobs OK
[17:02:17] 
[17:02:17] === Phase D: large/n_ext=2048 ===
[17:02:17] Submitting: mode=serial bank=large n_ext=2048 n_int=- n_workers=1

| 2026-03-04 17:02 | 990957 | serial | large | 2048 | full | 1 | pending | — |
[17:02:17] Submitting: mode=mp bank=large n_ext=2048 n_int=- n_workers=8

| 2026-03-04 17:02 | 990958 | mp | large | 2048 | full | 8 | pending | — |
[17:02:17] Submitting: mode=mp bank=large n_ext=2048 n_int=- n_workers=20

| 2026-03-04 17:02 | 990959 | mp | large | 2048 | full | 20 | pending | — |
[17:02:18] Submitting: mode=swarm bank=large n_ext=2048 n_int=- n_workers=-

| 2026-03-04 17:02 | 990960 | swarm | large | 2048 | full | - | pending | — |
[17:02:18]   waiting for serial/large/next2048 (job [17:02:17] Submitting: mode=serial bank=large n_ext=2048 n_int=- n_workers=1
Job <990957> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170217_serial_large_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170217_serial_large_next2048/run.log
990957) ...
[17:02:18]   serial/large/next2048 (job [17:02:17] Submitting: mode=serial bank=large n_ext=2048 n_int=- n_workers=1
Job <990957> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170217_serial_large_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170217_serial_large_next2048/run.log
990957) finished OK
[17:02:18]   waiting for mp/large/next2048 (job [17:02:17] Submitting: mode=mp bank=large n_ext=2048 n_int=- n_workers=8
Job <990958> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170217_mp_large_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170217_mp_large_next2048/run.log
990958) ...
[17:02:18]   mp/large/next2048 (job [17:02:17] Submitting: mode=mp bank=large n_ext=2048 n_int=- n_workers=8
Job <990958> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170217_mp_large_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170217_mp_large_next2048/run.log
990958) finished OK
[17:02:18]   waiting for mp/large/next2048 (job [17:02:17] Submitting: mode=mp bank=large n_ext=2048 n_int=- n_workers=20
Job <990959> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170217_mp_large_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170217_mp_large_next2048/run.log
990959) ...
[17:02:18]   mp/large/next2048 (job [17:02:17] Submitting: mode=mp bank=large n_ext=2048 n_int=- n_workers=20
Job <990959> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170217_mp_large_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170217_mp_large_next2048/run.log
990959) finished OK
[17:02:18]   waiting for swarm/large/next2048 (job [17:02:18] Submitting: mode=swarm bank=large n_ext=2048 n_int=- n_workers=-
Job <990960> is submitted to queue <physics-short>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170218_swarm_large_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170218_swarm_large_next2048/run.log
990960) ...
[17:02:18]   swarm/large/next2048 (job [17:02:18] Submitting: mode=swarm bank=large n_ext=2048 n_int=- n_workers=-
Job <990960> is submitted to queue <physics-short>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170218_swarm_large_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170218_swarm_large_next2048/run.log
990960) finished OK
[17:02:18] Phase D: large/n_ext=2048: all jobs OK
[17:02:18] 
[17:02:18] ==============================
[17:02:18] All phases complete!
[17:02:18] ==============================
| bank | n_ext | mode | wall_s | ln_evidence | n_effective | rundir |
|------|-------|------|--------|-------------|-------------|--------|
| large | 2048 | mp | — | — | — | 20260304_170217_mp_large_next2048 |
| large | 2048 | serial | — | — | — | 20260304_170217_serial_large_next2048 |
| large | 2048 | swarm | — | — | — | 20260304_170218_swarm_large_next2048 |
| small | 128 | mp | — | — | — | 20260304_170121_mp_small_next128 |
| small | 128 | mp | — | — | — | 20260304_170214_mp_small_next128 |
| small | 128 | serial | — | — | — | 20260304_170032_serial_small_next128 |
| small | 128 | serial | — | — | — | 20260304_170214_serial_small_next128 |
| small | 128 | swarm | — | — | — | 20260304_170121_swarm_small_next128 |
| small | 128 | swarm | — | — | — | 20260304_170214_swarm_small_next128 |
| small | 512 | mp | — | — | — | 20260304_170215_mp_small_next512 |
| small | 512 | serial | — | — | — | 20260304_170215_serial_small_next512 |
| small | 512 | swarm | — | — | — | 20260304_170216_swarm_small_next512 |
| small | 2048 | mp | — | — | — | 20260304_170216_mp_small_next2048 |
| small | 2048 | serial | — | — | — | 20260304_170216_serial_small_next2048 |
| small | 2048 | swarm | — | — | — | 20260304_170217_swarm_small_next2048 |
[17:02:18] Full log: /home/projects/barakz/jonatahm/dot-pe-future/experiments/auto_run.log
[17:02:55] ==============================
[17:02:55] dot-pe benchmark auto_run.sh
[17:02:55] ==============================
[17:02:55] Log file: /home/projects/barakz/jonatahm/dot-pe-future/experiments/auto_run.log
[17:02:55] 
[17:02:55] === Phase A: smoke tests ===
[17:02:55] Submitting: mode=serial bank=small n_ext=128 n_int=256 n_workers=-

| 2026-03-04 17:02 | 990962 | serial | small | 128 | 256 | - | pending | — |
[17:02:55] Submitting: mode=mp bank=small n_ext=128 n_int=256 n_workers=4

| 2026-03-04 17:02 | 990963 | mp | small | 128 | 256 | 4 | pending | — |
[17:02:55] Submitting: mode=swarm bank=small n_ext=128 n_int=256 n_workers=-

| 2026-03-04 17:02 | 990964 | swarm | small | 128 | 256 | - | pending | — |
[17:02:56]   waiting for serial/small/next128 (job [17:02:55] Submitting: mode=serial bank=small n_ext=128 n_int=256 n_workers=-
Job <990962> is submitted to queue <physics-short>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170255_serial_small_next128
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170255_serial_small_next128/run.log
990962) ...
[17:02:56]   serial/small/next128 (job [17:02:55] Submitting: mode=serial bank=small n_ext=128 n_int=256 n_workers=-
Job <990962> is submitted to queue <physics-short>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170255_serial_small_next128
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170255_serial_small_next128/run.log
990962) finished OK
[17:02:56]   waiting for mp/small/next128 (job [17:02:55] Submitting: mode=mp bank=small n_ext=128 n_int=256 n_workers=4
Job <990963> is submitted to queue <physics-short>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170255_mp_small_next128
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170255_mp_small_next128/run.log
990963) ...
[17:02:56]   mp/small/next128 (job [17:02:55] Submitting: mode=mp bank=small n_ext=128 n_int=256 n_workers=4
Job <990963> is submitted to queue <physics-short>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170255_mp_small_next128
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170255_mp_small_next128/run.log
990963) finished OK
[17:02:56]   waiting for swarm/small/next128 (job [17:02:55] Submitting: mode=swarm bank=small n_ext=128 n_int=256 n_workers=-
Job <990964> is submitted to queue <physics-short>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170255_swarm_small_next128
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170255_swarm_small_next128/run.log
990964) ...
[17:02:56]   swarm/small/next128 (job [17:02:55] Submitting: mode=swarm bank=small n_ext=128 n_int=256 n_workers=-
Job <990964> is submitted to queue <physics-short>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170255_swarm_small_next128
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170255_swarm_small_next128/run.log
990964) finished OK
[17:02:56] Phase A: smoke tests: all jobs OK
[17:02:56] 
[17:02:56] === Phase B: small/n_ext=512 ===
[17:02:56] Submitting: mode=serial bank=small n_ext=512 n_int=- n_workers=1

| 2026-03-04 17:02 | 990965 | serial | small | 512 | full | 1 | pending | — |
[17:02:56] Submitting: mode=mp bank=small n_ext=512 n_int=- n_workers=4

| 2026-03-04 17:02 | 990966 | mp | small | 512 | full | 4 | pending | — |
[17:02:56] Submitting: mode=mp bank=small n_ext=512 n_int=- n_workers=8

| 2026-03-04 17:02 | 990967 | mp | small | 512 | full | 8 | pending | — |
[17:02:56] Submitting: mode=swarm bank=small n_ext=512 n_int=- n_workers=-

| 2026-03-04 17:02 | 990968 | swarm | small | 512 | full | - | pending | — |
[17:02:57]   waiting for serial/small/next512 (job [17:02:56] Submitting: mode=serial bank=small n_ext=512 n_int=- n_workers=1
Job <990965> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170256_serial_small_next512
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170256_serial_small_next512/run.log
990965) ...
[17:02:57]   serial/small/next512 (job [17:02:56] Submitting: mode=serial bank=small n_ext=512 n_int=- n_workers=1
Job <990965> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170256_serial_small_next512
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170256_serial_small_next512/run.log
990965) finished OK
[17:02:57]   waiting for mp/small/next512 (job [17:02:56] Submitting: mode=mp bank=small n_ext=512 n_int=- n_workers=4
Job <990966> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170256_mp_small_next512
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170256_mp_small_next512/run.log
990966) ...
[17:02:57]   mp/small/next512 (job [17:02:56] Submitting: mode=mp bank=small n_ext=512 n_int=- n_workers=4
Job <990966> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170256_mp_small_next512
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170256_mp_small_next512/run.log
990966) finished OK
[17:02:57]   waiting for mp/small/next512 (job [17:02:56] Submitting: mode=mp bank=small n_ext=512 n_int=- n_workers=8
Job <990967> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170256_mp_small_next512
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170256_mp_small_next512/run.log
990967) ...
[17:02:57]   mp/small/next512 (job [17:02:56] Submitting: mode=mp bank=small n_ext=512 n_int=- n_workers=8
Job <990967> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170256_mp_small_next512
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170256_mp_small_next512/run.log
990967) finished OK
[17:02:57]   waiting for swarm/small/next512 (job [17:02:56] Submitting: mode=swarm bank=small n_ext=512 n_int=- n_workers=-
Job <990968> is submitted to queue <physics-short>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170257_swarm_small_next512
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170257_swarm_small_next512/run.log
990968) ...
[17:02:57]   swarm/small/next512 (job [17:02:56] Submitting: mode=swarm bank=small n_ext=512 n_int=- n_workers=-
Job <990968> is submitted to queue <physics-short>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170257_swarm_small_next512
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170257_swarm_small_next512/run.log
990968) finished OK
[17:02:57] Phase B: small/n_ext=512: all jobs OK
[17:02:57] 
[17:02:57] === Phase C: small/n_ext=2048 ===
[17:02:57] Submitting: mode=serial bank=small n_ext=2048 n_int=- n_workers=1

| 2026-03-04 17:02 | 990969 | serial | small | 2048 | full | 1 | pending | — |
[17:02:57] Submitting: mode=mp bank=small n_ext=2048 n_int=- n_workers=8

| 2026-03-04 17:02 | 990970 | mp | small | 2048 | full | 8 | pending | — |
[17:02:58] Submitting: mode=swarm bank=small n_ext=2048 n_int=- n_workers=-

| 2026-03-04 17:02 | 990971 | swarm | small | 2048 | full | - | pending | — |
[17:02:58]   waiting for serial/small/next2048 (job [17:02:57] Submitting: mode=serial bank=small n_ext=2048 n_int=- n_workers=1
Job <990969> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170257_serial_small_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170257_serial_small_next2048/run.log
990969) ...
[17:02:58]   serial/small/next2048 (job [17:02:57] Submitting: mode=serial bank=small n_ext=2048 n_int=- n_workers=1
Job <990969> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170257_serial_small_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170257_serial_small_next2048/run.log
990969) finished OK
[17:02:58]   waiting for mp/small/next2048 (job [17:02:57] Submitting: mode=mp bank=small n_ext=2048 n_int=- n_workers=8
Job <990970> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170257_mp_small_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170257_mp_small_next2048/run.log
990970) ...
[17:02:58]   mp/small/next2048 (job [17:02:57] Submitting: mode=mp bank=small n_ext=2048 n_int=- n_workers=8
Job <990970> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170257_mp_small_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170257_mp_small_next2048/run.log
990970) finished OK
[17:02:58]   waiting for swarm/small/next2048 (job [17:02:58] Submitting: mode=swarm bank=small n_ext=2048 n_int=- n_workers=-
Job <990971> is submitted to queue <physics-short>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170258_swarm_small_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170258_swarm_small_next2048/run.log
990971) ...
[17:02:58]   swarm/small/next2048 (job [17:02:58] Submitting: mode=swarm bank=small n_ext=2048 n_int=- n_workers=-
Job <990971> is submitted to queue <physics-short>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170258_swarm_small_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170258_swarm_small_next2048/run.log
990971) finished OK
[17:02:58] Phase C: small/n_ext=2048: all jobs OK
[17:02:58] 
[17:02:58] === Phase D: large/n_ext=2048 ===
[17:02:58] Submitting: mode=serial bank=large n_ext=2048 n_int=- n_workers=1

| 2026-03-04 17:02 | 990972 | serial | large | 2048 | full | 1 | pending | — |
[17:02:58] Submitting: mode=mp bank=large n_ext=2048 n_int=- n_workers=8

| 2026-03-04 17:02 | 990973 | mp | large | 2048 | full | 8 | pending | — |
[17:02:59] Submitting: mode=mp bank=large n_ext=2048 n_int=- n_workers=20

| 2026-03-04 17:02 | 990974 | mp | large | 2048 | full | 20 | pending | — |
[17:02:59] Submitting: mode=swarm bank=large n_ext=2048 n_int=- n_workers=-

| 2026-03-04 17:02 | 990975 | swarm | large | 2048 | full | - | pending | — |
[17:02:59]   waiting for serial/large/next2048 (job [17:02:58] Submitting: mode=serial bank=large n_ext=2048 n_int=- n_workers=1
Job <990972> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170258_serial_large_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170258_serial_large_next2048/run.log
990972) ...
[17:02:59]   serial/large/next2048 (job [17:02:58] Submitting: mode=serial bank=large n_ext=2048 n_int=- n_workers=1
Job <990972> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170258_serial_large_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170258_serial_large_next2048/run.log
990972) finished OK
[17:02:59]   waiting for mp/large/next2048 (job [17:02:58] Submitting: mode=mp bank=large n_ext=2048 n_int=- n_workers=8
Job <990973> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170258_mp_large_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170258_mp_large_next2048/run.log
990973) ...
[17:02:59]   mp/large/next2048 (job [17:02:58] Submitting: mode=mp bank=large n_ext=2048 n_int=- n_workers=8
Job <990973> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170258_mp_large_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170258_mp_large_next2048/run.log
990973) finished OK
[17:02:59]   waiting for mp/large/next2048 (job [17:02:59] Submitting: mode=mp bank=large n_ext=2048 n_int=- n_workers=20
Job <990974> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170259_mp_large_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170259_mp_large_next2048/run.log
990974) ...
[17:02:59]   mp/large/next2048 (job [17:02:59] Submitting: mode=mp bank=large n_ext=2048 n_int=- n_workers=20
Job <990974> is submitted to queue <physics-medium>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170259_mp_large_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170259_mp_large_next2048/run.log
990974) finished OK
[17:02:59]   waiting for swarm/large/next2048 (job [17:02:59] Submitting: mode=swarm bank=large n_ext=2048 n_int=- n_workers=-
Job <990975> is submitted to queue <physics-short>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170259_swarm_large_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170259_swarm_large_next2048/run.log
990975) ...
[17:02:59]   swarm/large/next2048 (job [17:02:59] Submitting: mode=swarm bank=large n_ext=2048 n_int=- n_workers=-
Job <990975> is submitted to queue <physics-short>.
rundir : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170259_swarm_large_next2048
log    : /home/projects/barakz/jonatahm/dot-pe-future/artifacts/experiments/20260304_170259_swarm_large_next2048/run.log
990975) finished OK
[17:02:59] Phase D: large/n_ext=2048: all jobs OK
[17:02:59] 
[17:02:59] ==============================
[17:02:59] All phases complete!
[17:02:59] ==============================
| bank | n_ext | mode | wall_s | ln_evidence | n_effective | rundir |
|------|-------|------|--------|-------------|-------------|--------|
| large | 2048 | mp | — | — | — | 20260304_170217_mp_large_next2048 |
| large | 2048 | mp | — | — | — | 20260304_170258_mp_large_next2048 |
| large | 2048 | mp | — | — | — | 20260304_170259_mp_large_next2048 |
| large | 2048 | serial | — | — | — | 20260304_170217_serial_large_next2048 |
| large | 2048 | serial | — | — | — | 20260304_170258_serial_large_next2048 |
| large | 2048 | swarm | — | — | — | 20260304_170218_swarm_large_next2048 |
| large | 2048 | swarm | — | — | — | 20260304_170259_swarm_large_next2048 |
| small | 128 | mp | — | — | — | 20260304_170121_mp_small_next128 |
| small | 128 | mp | — | — | — | 20260304_170214_mp_small_next128 |
| small | 128 | mp | — | — | — | 20260304_170255_mp_small_next128 |
| small | 128 | serial | — | — | — | 20260304_170032_serial_small_next128 |
| small | 128 | serial | — | — | — | 20260304_170214_serial_small_next128 |
| small | 128 | serial | — | — | — | 20260304_170255_serial_small_next128 |
| small | 128 | swarm | — | — | — | 20260304_170121_swarm_small_next128 |
| small | 128 | swarm | — | — | — | 20260304_170214_swarm_small_next128 |
| small | 128 | swarm | — | — | — | 20260304_170255_swarm_small_next128 |
| small | 512 | mp | — | — | — | 20260304_170215_mp_small_next512 |
| small | 512 | mp | — | — | — | 20260304_170256_mp_small_next512 |
| small | 512 | serial | — | — | — | 20260304_170215_serial_small_next512 |
| small | 512 | serial | — | — | — | 20260304_170256_serial_small_next512 |
| small | 512 | swarm | — | — | — | 20260304_170216_swarm_small_next512 |
| small | 512 | swarm | — | — | — | 20260304_170257_swarm_small_next512 |
| small | 2048 | mp | — | — | — | 20260304_170216_mp_small_next2048 |
| small | 2048 | mp | — | — | — | 20260304_170257_mp_small_next2048 |
| small | 2048 | serial | — | — | — | 20260304_170216_serial_small_next2048 |
| small | 2048 | serial | — | — | — | 20260304_170257_serial_small_next2048 |
| small | 2048 | swarm | — | — | — | 20260304_170217_swarm_small_next2048 |
| small | 2048 | swarm | — | — | — | 20260304_170258_swarm_small_next2048 |
[17:02:59] Full log: /home/projects/barakz/jonatahm/dot-pe-future/experiments/auto_run.log

| 2026-03-04 17:11 | 991329 | serial | small | 128 | 256 | - | pending | — |

| 2026-03-04 17:11 | 991330 | mp | small | 128 | 256 | 4 | pending | — |

| 2026-03-04 17:11 | 991334 | swarm | small | 128 | 256 | - | pending | — |

| 2026-03-04 17:13 | 991752 | serial | small | 128 | 256 | - | pending | — |

| 2026-03-04 17:13 | 991753 | mp | small | 128 | 256 | 4 | pending | — |

| 2026-03-04 17:13 | 991754 | swarm | small | 128 | 256 | - | pending | — |

| 2026-03-04 17:19 | 991802 | serial | small | 512 | full | - | pending | — |

| 2026-03-04 17:19 | 991803 | mp | small | 512 | full | 4 | pending | — |

| 2026-03-04 17:19 | 991804 | mp | small | 512 | full | 8 | pending | — |

| 2026-03-04 17:19 | 991805 | swarm | small | 512 | full | - | pending | — |

| 2026-03-04 17:19 | 991806 | serial | small | 2048 | full | - | pending | — |

| 2026-03-04 17:19 | 991807 | mp | small | 2048 | full | 8 | pending | — |

| 2026-03-04 17:19 | 991808 | swarm | small | 2048 | full | - | pending | — |

| 2026-03-04 17:19 | 991809 | serial | large | 2048 | full | - | pending | — |

| 2026-03-04 17:19 | 991810 | mp | large | 2048 | full | 8 | pending | — |

| 2026-03-04 17:19 | 991811 | mp | large | 2048 | full | 20 | pending | — |

| 2026-03-04 17:19 | 991812 | swarm | large | 2048 | full | - | pending | — |

---
## SESSION NOTES 2026-03-04 17:19 IST

### Phase A smoke tests: ALL PASSED (jobs 991752/991753/991754)
- serial: Done, 193s wall, 2852MB peak
- mp (4w): Done, 185s wall
- swarm: Done, 235s wall

### Fixes applied this session
1. serial script: `python {ROOT}/dot_pe/inference.py` → `cd {ROOT} && python -m dot_pe.inference`
2. `inference.py:799`: added `rundir = Path(rundir)` (str/str TypeError for mp/swarm)
3. `inference.py:822`: `int(i_int_start)` → handle None → 0
4. `inference.py:1137`: added `bank_rundir.mkdir(parents=True, exist_ok=True)` before np.savez
5. `run_experiment.py`: added `rusage[mem=4096]` to serial and swarm bsub
6. `auto_run.sh` + `run_swarm.py`: `bjobs -a` → `bhist -l` for exit status

### Currently running (submitted 17:19)
| job   | mode   | bank  | n_ext |
|-------|--------|-------|-------|
| 991802 | serial | small | 512 |
| 991803 | mp/4w  | small | 512 |
| 991804 | mp/8w  | small | 512 |
| 991805 | swarm  | small | 512 |
| 991806 | serial | small | 2048 |
| 991807 | mp/8w  | small | 2048 |
| 991808 | swarm  | small | 2048 |
| 991809 | serial | large | 2048 |
| 991810 | mp/8w  | large | 2048 |
| 991811 | mp/20w | large | 2048 |
| 991812 | swarm  | large | 2048 |

| 2026-03-04 17:23 | 991824 | mp | small | 512 | full | 4 | pending | — |

| 2026-03-04 17:23 | 991825 | mp | large | 2048 | full | 8 | pending | — |

| 2026-03-04 17:23 | 991826 | mp | large | 2048 | full | 20 | pending | — |

---
## SESSION NOTES 2026-03-04 17:35 IST (logging off)

### All fixes now applied. Code is stable.

### Fixes applied this session (cumulative)
1. `run_experiment.py` serial: `python -m dot_pe.inference` (relative import fix)
2. `inference.py:799`: `rundir = Path(rundir)` (str/str TypeError for mp/swarm)
3. `inference.py:822`: `int(i_int_start) if i_int_start is not None else 0`
4. `inference.py:1137`: `bank_rundir.mkdir(parents=True, exist_ok=True)` before np.savez
5. `run_experiment.py`: `rusage[mem=4096]` for serial and swarm bsub (serial peaks at ~2852 MB)
6. `auto_run.sh` wait_job: `bjobs -a` → `bhist -l` for exit-status checking
7. `run_swarm.py` _wait_for_job: same `bjobs -a` → `bhist -l` fix
8. `run_experiment.py` _make_rundir: include `_wN_` for mp mode to avoid rundir clashes

### Jobs still running at log-off (17:35 IST)
| job    | mode    | bank  | n_ext | rundir |
|--------|---------|-------|-------|--------|
| 991802 | serial  | small | 512   | 20260304_171921_serial_small_next512 |
| 991804 | mp/8w   | small | 512   | 20260304_171922_mp_small_next512 |
| 991805 | swarm   | small | 512   | 20260304_171922_swarm_small_next512 |
| 991806 | serial  | small | 2048  | 20260304_171927_serial_small_next2048 |
| 991807 | mp/8w   | small | 2048  | 20260304_171927_mp_small_next2048 |
| 991808 | swarm   | small | 2048  | 20260304_171927_swarm_small_next2048 |
| 991809 | serial  | large | 2048  | 20260304_171931_serial_large_next2048 |
| 991812 | swarm   | large | 2048  | 20260304_171932_swarm_large_next2048 |
| 991824 | mp/4w   | small | 512   | 20260304_172313_mp_w4_small_next512 |
| 991825 | mp/8w   | large | 2048  | 20260304_172314_mp_w8_large_next2048 |
| 991826 | mp/20w  | large | 2048  | 20260304_172314_mp_w20_large_next2048 |

### Failed/killed (resubmitted above)
- 991803 (mp/4w small/512): rundir clash with 991804 → Posterior.json FileExistsError → resubmitted as 991824
- 991810 (mp/8w large/2048): rundir clash with 991811 → same error → resubmitted as 991825
- 991811 (mp/20w large/2048): killed (shared broken rundir) → resubmitted as 991826

### On next login
1. Check job outcomes: `bjobs -a` or check run.log for each rundir
2. Run `python experiments/compare.py` for timing table
3. If any jobs failed, check run.log.err and fix/resubmit

---
## SESSION NOTES 2026-03-04 22:25 IST

### All previous Phase B/C/D jobs OOM'd
- Jobs 991802–991826 (submitted 17:19–17:23) all killed with TERM_MEMLIMIT
- Root cause: those jobs had hardcoded `rusage[mem=4096]` embedded at submit time
- The `_mem_per_slot_mb()` formula was added AFTER those jobs were submitted

### Memory formula (now in run_experiment.py)
- `total_mb = max(4096, int(n_ext/128 * 3072 * 1.2))`
- n_ext=512  serial/swarm: 14745 MB; mp/8w: 1843 MB/slot
- n_ext=2048 serial/swarm: 58982 MB; mp/8w: 7372 MB/slot; mp/20w: 2949 MB/slot
- Nodes have 376 GB RAM — formula is well within cluster limits

### Resubmitted all Phase B/C/D (22:25 IST)
| job  | mode    | bank  | n_ext | rundir |
|------|---------|-------|-------|--------|
| 1088 | serial  | small | 512   | 20260304_222551_serial_small_next512 |
| 1089 | mp/4w   | small | 512   | 20260304_222551_mp_w4_small_next512 |
| 1090 | mp/8w   | small | 512   | 20260304_222552_mp_w8_small_next512 |
| 1091 | swarm   | small | 512   | 20260304_222552_swarm_small_next512 |
| 1092 | serial  | small | 2048  | 20260304_222557_serial_small_next2048 |
| 1093 | mp/8w   | small | 2048  | 20260304_222557_mp_w8_small_next2048 |
| 1094 | swarm   | small | 2048  | 20260304_222557_swarm_small_next2048 |
| 1095 | serial  | large | 2048  | 20260304_222558_serial_large_next2048 |
| 1096 | mp/8w   | large | 2048  | 20260304_222558_mp_w8_large_next2048 |
| 1097 | mp/20w  | large | 2048  | 20260304_222558_mp_w20_large_next2048 |
| 1098 | swarm   | large | 2048  | 20260304_222558_swarm_large_next2048 |

Phase B (1088–1091): RUN on cn392 as of submission
Phase C/D (1092–1098): PEND (waiting for 58 GB slots)

Note: Phase B missing mp/4w for small/512 in original plan — included as job 1089.
Note: Phase C missing mp/4w for small/2048 — omitted (not in original plan).

### On next login
1. `bjobs -noheader 1088 1089 1090 1091 1092 1093 1094 1095 1096 1097 1098`
2. If all done: `python experiments/compare.py`
3. If OOM again: re-examine the formula (currently assumes linear scaling from n_ext=128 baseline)

| 2026-03-04 22:25 | 1088 | serial | small | 512 | full | - | pending | — |

| 2026-03-04 22:25 | 1089 | mp | small | 512 | full | 4 | pending | — |

| 2026-03-04 22:25 | 1090 | mp | small | 512 | full | 8 | pending | — |

| 2026-03-04 22:25 | 1091 | swarm | small | 512 | full | - | pending | — |

| 2026-03-04 22:25 | 1092 | serial | small | 2048 | full | - | pending | — |

| 2026-03-04 22:25 | 1093 | mp | small | 2048 | full | 8 | pending | — |

| 2026-03-04 22:25 | 1094 | swarm | small | 2048 | full | - | pending | — |

| 2026-03-04 22:25 | 1095 | serial | large | 2048 | full | - | pending | — |

| 2026-03-04 22:25 | 1096 | mp | large | 2048 | full | 8 | pending | — |

| 2026-03-04 22:25 | 1097 | mp | large | 2048 | full | 20 | pending | — |

| 2026-03-04 22:25 | 1098 | swarm | large | 2048 | full | - | pending | — |

| 2026-03-04 23:22 | 2024 | mp | small | 512 | full | 4 | pending | — |

| 2026-03-04 23:22 | 2025 | mp | small | 512 | full | 8 | pending | — |

| 2026-03-04 23:22 | 2026 | swarm | small | 512 | full | - | pending | — |

| 2026-03-04 23:22 | 2027 | mp | large | 2048 | full | 8 | pending | — |

| 2026-03-04 23:22 | 2028 | mp | large | 2048 | full | 20 | pending | — |

| 2026-03-04 23:22 | 2029 | swarm | large | 2048 | full | - | pending | — |

| 2026-03-04 23:40 | 2047 | mp | small | 512 | full | 4 | pending | — |

| 2026-03-04 23:40 | 2048 | mp | large | 2048 | full | 8 | pending | — |

| 2026-03-04 23:40 | 2049 | mp | large | 2048 | full | 20 | pending | — |

---
## SESSION NOTES 2026-03-04 23:40 IST

### Results so far (successful runs)
| bank  | n_ext | mode   | n_workers | wall_s | ln_evidence | n_effective |
|-------|-------|--------|-----------|--------|-------------|-------------|
| small | 128   | serial | —         | 193    | 5.347       | 31.1        |
| small | 128   | mp     | 4         | 185    | 5.347       | 31.1        |
| small | 128   | swarm  | —         | 235    | 5.347       | 31.1        |
| small | 512   | serial | —         | 597    | 3.689       | 1376.7      |
| small | 512   | mp     | 8         | 541    | 3.689       | 1377.1      |
| small | 512   | swarm  | —         | 889    | 3.689       | 1377.1      |
| small | 2048  | serial | —         | 628    | 3.552       | 5492.6      |
| small | 2048  | mp     | 8         | 579    | 3.552       | 5493.4      |
| small | 2048  | swarm  | —         | 718    | 3.552       | 5493.4      |
| large | 2048  | serial | —         | 3167   | 3.328       | 177187.5    |

### Memory formula iterations
Formula went through 3 wrong versions before converging:
1. `n_ext/128 * 3072 * 1.2` — wrong: n_ext doesn't matter
2. mode+bank flat values with `+ n_workers * 2000 (small) / 8000 (large)` — wrong: small doesn't scale with n_workers; large scales much faster
3. Final: small-mp flat 30 GB; large-mp = 15 + n_workers × 10 GB

### Memory peaks observed
| mode   | bank  | n_workers | peak_MB | formula_MB | result |
|--------|-------|-----------|---------|------------|--------|
| serial | small | 1         | 9206    | 13000      | OK     |
| serial | large | 1         | 10473   | 15000      | OK     |
| swarm  | small | 1         | 16771   | 22000      | OK     |
| swarm  | large | 1         | running | 90000      | ?      |
| mp     | small | 4         | >21000  | 30000      | retry  |
| mp     | small | 8         | 23787   | 30000      | OK     |
| mp     | large | 8         | >79000  | 95000      | retry  |
| mp     | large | 20        | >175000 | 215000     | retry  |

### Currently running/pending (23:40 IST)
| job  | mode  | bank  | n_workers | status  |
|------|-------|-------|-----------|---------|
| 2029 | swarm | large | —         | RUN     |
| 2047 | mp    | small | 4         | RUN     |
| 2048 | mp    | large | 8         | RUN     |
| 2049 | mp    | large | 20        | PEND    |

### compare.py fixes
- Regex now correctly parses `_wN_` suffix → n_workers column added
- Wall time falls back to LSF "Run time" for serial (successful jobs only)

### On next login
1. `bjobs -noheader 2029 2047 2048 2049`
2. `python experiments/compare.py`
