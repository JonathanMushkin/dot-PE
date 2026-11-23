# E2 design: how far fewer templates and fewer extrinsic samples could shrink the rows (measurement)

Class: measurement. Directions 2 (fewer intrinsic samples) and the user's n_ext question
(2026-11-22) reduce the number of rows by using fewer templates or fewer extrinsic samples; both
are bounded by the minimum effective sample sizes the paper repos demand (o4b-pe-paper
`src/o4a_pe_paper/pe_lib/failure_modes.py`: `MIN_NEFF_I` = 150, `MIN_NEFF_E` = 100).

## Question

For each production run, what is H, the largest factor by which the number of rows could be
reduced by using fewer templates and fewer extrinsic samples while n_effective_i ≥ 150 and
n_effective_e ≥ 100 under both the IAS and LVK priors, if n_effective_i scales in proportion to
the number of templates and n_effective_e in proportion to n_ext?

No prediction: the request carries no hypothesis about H.

## Measure

- p: the prior, IAS (`summary_results.json`) or LVK (`summary_results_lvc.json`).
- h_i = min_p n_effective_i,p / 150 and h_e = min_p n_effective_e,p / 100, the intrinsic and
  extrinsic headroom ratios; a run with h_i < 1 or h_e < 1 already fails a minimum and is marked
  in both figures.
- H = max(1, h_i) × max(1, h_e); unitless.
- In the report, after figure 2, one sentence compares H of GW240525_201644 with the reduction
  it needs, R = N̂(c) / `size_limit`, read from E1's output.

## Variable swept and held fixed

Swept: the production run (all 288 in o4b-pe-paper `dev/row_limit_check/row_limit.csv`).
Held fixed: everything else; the measure reads existing summaries only.

## Glossary

- row, `size_limit`, nominal cut: as in `research/row-limit/design.md`.
- n_effective_i, n_effective_e: the effective number of distinct intrinsic and extrinsic samples
  of the weighted rows (`dot_pe/base_sampler_free_sampling.py`, `get_n_effective_total_i_e`).
- n_ext: the number of extrinsic samples of a run (13 312 for GW240525_201644 run_2).
- h_i, h_e, H, p: defined under "Measure". N̂(c), c: from E1 (`research/rows-above-cut/design.md`).
- headroom: the factor by which an effective sample size exceeds its minimum.
- direction 2: the user's option "use less samples" (fewer templates per bank).
- best-fit log-likelihood (figure 2): `bestfit_lnlike_max` of `summary_results.json`, the maximum
  over the 1e4 posterior draws (`dot_pe/inference.py:1568`), used only as an x-axis to order runs
  by SNR.

## Figure chain

1. Across runs: how much headroom does each run have in n_effective_i and in n_effective_e?
   (scatter of min_p n_effective_i,p / 150 against min_p n_effective_e,p / 100, log axes, lines
   at 1)
2. Across runs: what is H against the best-fit log-likelihood? (scatter, log y)

## Confounders and how each is fixed

- The proportional scaling is an assumption. The GW240525_201644 reruns test it once: the first
  eighth of each bank gave n_effective_i 1535 → 363 under the IAS prior (a factor 4.2 for 8 times
  fewer templates) and n_effective_e 262 → 329, because the floor moved too
  (o4b-pe-paper `dev/row_limit_check/gw240525_runs.md`). The report states H as an estimate
  under the assumption and quotes this one test, without correcting H.
- H multiplies the two headrooms as if independent; the rerun above shows n_effective_e changes
  when templates are cut, so they are not. Stated in the report.
- Summaries written before dot-PE 2403731 had a wrong cross-bank n_effective aggregation; the
  paper repos recomputed them afterwards (`timing.json` note of GW240525_201644 run_2). The
  script reads the summaries as they are now and flags runs whose `timing.json` dot-PE commit
  predates 2403731 (the cross-bank fix) and whose summary file is older than 2026-07-13.
- Runs without `summary_results_lvc.json` use the IAS prior only, and are marked.

## Code and compute

`headroom.py` in this folder, env `o4a-pe`, reads 2 × 288 small JSON files; login node (seconds).
Output `artifacts/research/sample-headroom/headroom.csv` and `review/`.

## Design note, 2026-11-22 12:05 (after the results)

H as defined (max(1, h_i) × max(1, h_e)) is 1 or more for a run that already fails a minimum,
which reads as "no reduction possible" only implicitly. The report states H only for the runs
that pass both minima and marks the others as failing; the definition is unchanged.

## Design note, 2026-11-23 06:52 (after the results; user decision)

The minima 150 and 100 are applied under the IAS prior only. This is the production criterion:
o4b-pe-paper `failure_modes.analyze_rundir` reads `summary_results.json`, the IAS-prior summary.
The earlier version took the minimum over the IAS and LVK priors, which this design chose without
a source; the LVK-reweighted effective sample sizes are lower, and 174 of the 175 runs that
failed a minimum failed it only under the LVK prior. The user (2026-11-23): high n_effective
under the LVK prior is welcome but not a goal of this project. So h_i = n_effective_i,IAS / 150
and h_e = n_effective_e,IAS / 100; the LVK-prior values stay in `headroom.csv` and are quoted in
the report for context only. H's definition is otherwise unchanged.
