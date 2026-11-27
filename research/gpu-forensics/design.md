# Design: GPU forensics (P1, measurement)

## Question

1. Where does the wall time of production dot-PE runs go, stage by stage?
2. Which stages must move to the GPU, and at what speed, for the GPU run to take at most half the
   production wall time?
3. Why did the earlier GPU attempt (`origin/dev/gpu`, March 2026) not beat the multi-core CPU
   code?
4. How much do repeat production runs of the same event, with the same settings, differ? This
   sets the "similar results" thresholds of P5.

## Data (read in place, never written)

1. Production LSF logs: `o4a-pe-paper/artifacts/logs/dotpe_GW*/*.out` (109 with stage timings)
   and `o4b-pe-paper/artifacts/logs/dotpe_GW*/*.out` (260). Each log ends with a "Stage timings"
   block (stages 1–6, in seconds) and "Total wall-clock time".
2. The run directories `*/artifacts/pe/dotpe/<EVENT>/run_N/`: `timing.json`, `run_kwargs.json`,
   `summary_results.json`, `samples.feather`, `profile_output.prof`.
3. `origin/dev/gpu`: `gpu/PROGRESS.md` and `gpu/BENCHMARK.md`; `origin/lsf-swarm` benchmarks (the
   memory note `lsf-swarm-benchmark.md`).

Code state: dot-PE `gpu-v2` at the commit that adds this file. The production runs were made with
the dot-PE commits in their `timing.json`.

## Measures

1. Per run: wall time of stages 1–6 (setup, incoherent, cross-bank, extrinsic, coherent,
   postprocess), with the run's configuration (number of banks, n_int, n_ext, n_phi, blocksize,
   `n_i_inds_used`, `n_distance_marginalizations`). The log is matched to its rundir by event and
   by the total wall time against `timing.json` `wall_seconds` (within 5 s). Logs without a
   matching rundir are kept with config "unknown".
2. Distributions over runs (median, 10th and 90th percentile) of each stage's share of the wall
   time, overall and per configuration.
3. Amdahl table. For the median run and for GW240525_201644 run_2, the GPU time budget per stage
   that gives a total of at most 0.5× and 0.1× the production wall time. The stages left on the
   CPU (setup, extrinsic) are counted at their production time.
4. From the profiles of 3 runs, the CPU-seconds inside the coherent and incoherent stages, split
   into compute, disk reads, the lookup table and DataFrames (already measured for GW240525 run_2;
   repeated for the 3 runs of the most common configuration).
5. Repeat runs: events with 2 or more `status: ok` runs that have the same banks, n_int, n_ext and
   n_phi. Per pair: |Δ ln_evidence|, the ratio of n_effective_i and of n_effective_e, and the
   Jensen–Shannon divergence (bits) of the 1-D marginals of chirp mass, mass ratio, chi_eff and
   luminosity distance in `samples.feather`.

## Gate

Measure 3 decides the GPU scope. A stage stays on the CPU only if its production time, summed with
the other CPU stages, is at most 50% of the 0.5× target for the median run. Otherwise it joins the
GPU scope in P2–P4.

## Compute

One script, `research/gpu-forensics/stage_times.py`, reads text and JSON. It runs on the login node
if it finishes within 1 minute; otherwise it goes through LSF (1 core). Measure 5 reads feather
files: a 1-core LSF job.

## Report

`research/gpu-forensics/YYYY-MM-DD_forensics.md`, with figures in `review/`.

## Revisions

2026-11-27, after the review of the report (findings 2 to 15). These were not in the design and
were added in the report:
1. ln q and cos θ_JN were added to the marginals of measure 5.
2. Per pair, the Jensen–Shannon estimator floor (two random halves of one run) and the sample
   counts were added, and pairs with too few samples are flagged.
3. The DataFrame and argpartition lines of measure 4 were dropped, being under 2% of the
   coherent CPU time.
4. The cross-bank stage was counted among the CPU-side stages.
5. The four questions above are answered one by one in the report. A single-question design was
   not written, because this measurement is the protocol's forensics step, run once.
6. The report file is `2026-11-27_measurement.md` (protocol naming), not `_forensics.md`.
