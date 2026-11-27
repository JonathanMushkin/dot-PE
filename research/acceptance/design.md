# Design: acceptance of the GPU runner (P5)

## Question

Does `dot_pe_gpu.run` produce `summary_results.json` and `samples.feather` with results similar to
the production runs, in at most half their wall time?

## Events

These are production runs from o4b and o4a, read in place. Each event uses its own production
`run_kwargs.json` with `row_selection="bestfit"`, the rule production used, and the same seed.
Stage 4 (cogwheel, CPU) then draws the same extrinsic samples whenever the incoherent selection is
the same.

| # | Event, run | Banks | n_ext | Production wall [s] | Why |
|---|---|---|---|---|---|
| 1 | o4b GW240525_201644 run_2 | 3 | 13312 | 21022 | long; coherent-dominated |
| 2 | o4b GW240515_005301 run_0 | 6 | 2048 | 9947 | nearest the median run |
| 3 | o4b GW240923_204006 run_0 | 3 | 2048 | 4942 | the most common configuration |
| 4 | o4b GW250114_082203 run_0 | 3 | 2048 | 1767 | highest SNR (best fit 3043); incoherent-dominated |
| 5 | o4b GW240920_124024 run_0 | 1 | 2048 | 766 | 1 bank, high SNR |
| 6 | o4a GW231127_061546 run_0 | 1 | 2048 | 662 | o4a; short |

## Measures

1. Speed: the GPU job's wall time from run start to the written `samples.feather`
   (`gpu_timing.json` `wall_seconds`), against the production `timing.json` `wall_seconds`.
   The GPU job is exclusive, on L40S, with at most 8 cores.
2. Results, from `research/acceptance/compare.py` (GPU rundir, production rundir):
   1. |Δ ln_evidence|;
   2. the ratios of n_effective_i and of n_effective_e;
   3. the Jensen–Shannon divergence in bits of the 1-D marginals of chirp mass, ln q, chi_eff,
      luminosity distance and cos θ_JN. The bins are those of P1, and each divergence is
      reported with its estimator floor.
3. Seed scatter. Event 3 is rerun on the GPU with 3 other seeds, and measures 2.1 to 2.3 are
   taken between those runs. This gives the scatter that one seed change produces (P1 had none).

## Pass criteria (fixed before any result)

For every event:
1. Speed: GPU wall ≤ 0.5 × production wall.
2. Results:
   1. |Δ ln_evidence| ≤ max(0.1, 2 × the largest seed-scatter |Δ ln_evidence|);
   2. n_effective_i and n_effective_e ratios within [0.8, 1.25] or within the seed-scatter range,
      whichever is wider;
   3. each marginal's divergence ≤ max(0.02 bits, 2 × the largest seed-scatter divergence of that
      marginal).

The production runs predate later dot-PE fixes (the incoherent-bound mask, the best-fit distance
mask and the top-N fallback). Where one of them triggers on an event, a CPU rerun with current
code (at most 2, protocol budget) replaces production as the reference for that event, and the
report says so.

## Open point for the user

Runs whose production wall time is under about 4 minutes cannot reach 0.5×, because their
CPU-side stages alone (setup, extrinsic, postprocess) take about 2 minutes. Example: o4b
GW240601_061200 run_0, 230 s in total. Event 6 (662 s) is the shortest event in the set. The
report gives the fraction of all 369 production runs for which 0.5× is reachable by
construction, using the measured GPU stage times.

## Compute

6 exclusive GPU jobs, plus 3 shared seed runs, each at most 1 h of GPU time; 9 GPU-hours at
most. Jobs run one at a time.
