# Design: acceptance of the GPU runner (P5)

## Question

Does `dot_pe_gpu.run` produce `summary_results.json` and `samples.feather` with results similar to
the production runs, in at most half their wall time?

## Terms

- Production run: an o4a or o4b dot-PE rundir under `*/artifacts/pe/dotpe/<EVENT>/run_N/`, made
  with `mp_inference` on 8 cores. Its wall time is `timing.json` `wall_seconds`.
- GPU run: `dot_pe_gpu.run` with the production run's `run_kwargs.json`
  (`research/acceptance/kwargs/<EVENT>.json`), `row_selection="bestfit"` (production's rule), on
  one L40S.
- Production seed: the `seed` in the production `run_kwargs.json`. Stage 4, the extrinsic
  proposal, is cogwheel on the CPU. With the same seed and the same incoherent selection, it draws
  the same extrinsic samples.
- Regime 1: the GPU run's `extrinsic_samples.feather` equals production's. The comparison then
  tests numerical reproduction.
- Regime 2: it differs, because the selection differs at the threshold or a later dot-PE fix
  changed it. The comparison then has seed-scatter character.
- Seed scatter: the spread of the results over GPU runs of one event with different seeds.
- Jensen–Shannon divergence (JS): of the 1-D marginals of chirp mass, ln q, chi_eff, luminosity
  distance and cos θ_JN, in bits, on 50 bins over the 0.5 to 99.5 percentiles of the pair. Its
  estimator floor is the median JS over 20 pairs of independent bootstrap resamples of run A,
  each of A's size.
- Residency cap: the device memory that may keep selected waveforms between stages 2 and 5
  (free memory minus 12 GB). A run above it streams them from disk in stage 5.

## Events

None of these was used in P2 to P4. Each is free of `failure_analysis.json` failure modes where that
file exists; the o4a runs have none.

| # | Event, run | Banks | n_ext | Selected templates | Production wall [s] | Why |
|---|---|---|---|---|---|---|
| 1 | o4b GW240515_005301 run_0 | 6 | 2048 | 409945 | 9947 | nearest the median run |
| 2 | o4b GW240923_204006 run_0 | 3 | 2048 | 275478 | 4942 | the most common configuration |
| 3 | o4a GW231028_153006 run_0 | 2 | 2048 | 35623 | 1824 | high SNR (best fit 244) |
| 4 | o4b GW240920_124024 run_0 | 1 | 2048 | 35016 | 766 | 1 bank, high SNR (best fit 707) |
| 5 | o4a GW231127_061546 run_0 | 1 | 2048 | 2356 | 662 | 1 bank, short; low n_effective_e (30) |
| 6 | o4b GW240916_184352 run_2 | 6 | 8192 | 248509 | 11422 | many banks, large n_ext |
| 7 | o4a GW231002_143916 run_0 | 9 | 3072 | 1512115 | 26598 | 9 banks; selection above the residency cap (disk streaming) |

GW240525_201644 run_2, the development event, is rerun as a regression check only.

## Measures

1. Speed: one timed GPU run per event, in an exclusive L40S job with 8 cores. Its wall time is
   from run start to the written `samples.feather` (`gpu_timing.json` `wall_seconds`). Reported
   with it: the stage times, the read GB/s of stage 2, and whether the event's banks were read by
   an earlier job that day.
2. Regime: whether the GPU run's `extrinsic_samples.feather` equals production's, and how its
   stage-2 selection differs from production's `intrinsic_samples.npz`.
3. Results, from `research/acceptance/compare.py` (production rundir, GPU rundir):
   1. |Δ ln_evidence|;
   2. the ratios of n_effective_i and of n_effective_e;
   3. JS per marginal, with its floor and the sample counts.
4. Seed scatter, for every event in regime 2: two more GPU runs with seeds production seed + 1 and
   + 2 (shared jobs, not timed). The scatter is the largest value of each result measure over the
   three pairs among the three GPU runs.

## Pass criteria (fixed before any result; coded in `compare.py`)

For every event:
1. Speed: GPU wall ≤ 0.5 × production wall. Two exceptions follow the user's decision of
   2026-11-27 (`README.md`):
   1. Events under the short-run cutoff (4 minutes) are excluded. None of the 7 is.
   2. If an event fails only because its CPU-side stages (setup, extrinsic, postprocess) exceed
      the budget, the cutoff may be raised, as that decision describes.

   A failure caused by disk read throughput is a null result. It is not rerun.
2. Results, regime 1:
   1. |Δ ln_evidence| ≤ 1e-3;
   2. n_effective_i and n_effective_e ratios within [0.99, 1.01];
   3. JS ≤ max(0.02 bits, 2 × floor) per marginal.
3. Results, regime 2:
   1. |Δ ln_evidence| ≤ max(0.1, 2 × seed scatter);
   2. n_effective ratios within [0.8, 1.25], or within the seed-scatter ratio range widened by a
      factor 1.25 on each side, whichever is wider;
   3. JS ≤ max(0.02 bits, 2 × seed scatter, 2 × floor) per marginal.
4. Code versions. All production runs predate the incoherent-bound mask, and events 1, 2 and 4
   also predate b0246ea and dce2acd. If an event fails 2 or 3, the current CPU code is rerun on
   it (protocol budget: 2 CPU reruns). If the GPU then meets the criteria against that rerun,
   the event passes, and the report names the reference. If more than 2 events need this, the
   rest are reported as failures, with the code-version difference as the suspected cause.
5. Regression: GW240525 run_2 in regime 1 meets criterion 2. It is not acceptance evidence.

## Confounders

1. Code version: the production runs predate later fixes (criterion 4).
2. Seed regime: regime 1 cannot detect a defect that appears only with other extrinsic draws.
   The seed runs of regime-2 events, and P2's per-block statistical tests, cover that.
3. cProfile: every production run had `profile_enabled`, which inflates production wall time,
   including stage 4, which runs on the CPU in both. The size of the effect is not measured.
4. Disk cache and load: stage-2 read throughput varied from 0.37 to 0.87 GB/s in P3 and P4. Each
   speed result reports its GB/s and whether its banks were warm.
5. Residency size: only event 7 exceeds the cap. A success claim beyond about 1.5M selected
   templates is not made.

## Figure chain

1. For each event: GPU wall time against production wall time, on log axes, with the 0.5× line.
   Does every event lie below it?
2. For each event: the stage times of the GPU run, stacked. Which stage limits the speed-up?
3. For each event and measure: the result difference against its criterion (ratio to the
   threshold). Does any result exceed its threshold?

## Compute

- 8 timed exclusive jobs (7 events and the regression), each with `-W` ≤ 180 and at most
  1.5 GPU-hours.
- At most 14 shared seed runs.
- In total at most 20 GPU-hours, which would bring the project's total to about 21 of 60. Jobs
  run one at a time.
