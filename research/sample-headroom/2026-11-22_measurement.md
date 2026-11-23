# How far fewer templates and fewer extrinsic samples could shrink dot-PE's rows

This report measures how much room the production runs have to use fewer templates or fewer
extrinsic samples, two ways to store fewer rows against dot-PE's limit on stored rows (rows and
the limit are defined in Setup). Both are bounded by the minimum effective sample sizes the
analysis demands, so the room is the factor by which those effective sample sizes exceed their
minima.

Code and data: `research/sample-headroom/headroom.py`, dot-PE branch `research`, commit 7d87c5d;
it reads the production summaries of the 288 runs in o4b-pe-paper
`dev/row_limit_check/row_limit.csv` (dot-PE commits as recorded per run in each paper repo's
`docs/pe/dot_pe_versions.csv`). It supersedes the version of this report at commit
[b4d7248](https://github.com/JonathanMushkin/dot-PE/blob/b4d7248/research/sample-headroom/2026-11-22_measurement.md).

## Question and answer

The question: for each production run, what is H, the largest factor by which its stored rows
could be reduced by fewer templates and fewer extrinsic samples while n_effective_i, the effective
number of distinct templates, stays at least 150 and n_effective_e, the effective number of
distinct extrinsic samples, at least 100, under the IAS prior (dot-PE's production prior), if
each scales in proportion to its number of samples? H and the other terms are defined in Setup.

The answer: one of the 288 runs already fails a minimum; for the other 287, H has a median of 47
(quartiles 15 and 86) and ranges from 1.7 to 375. Almost all of the room is in templates: the
factor by which n_effective_i exceeds 150 (h_i) has median 19, the factor by which n_effective_e
exceeds 100 (h_e) median 2.4. For GW240525_201644, H = 27, below R = 74, the factor by which its
rows above the nominal cut exceed the row limit; 94 of the 287 runs have H ≥ 74.

## Setup

### Rows and the row limit

A *row* is one (intrinsic, extrinsic, phase) combination: a template, an extrinsic sample (sky
position, orientation, time), and one reference phase. For each row dot-PE computes b
(`bestfit_lnlike`), its log-likelihood maximized over distance. dot-PE stores at most
`size_limit` rows, those of highest b; every production run used `size_limit` = 1e7 (the code
default is 1e6, `dot_pe/coherent_processing.py:35`). The *floor* is the lowest b among the stored rows.

### Effective sample sizes

With p_r the posterior weight of row r under the IAS prior, normalized to sum to 1, P_t the sum
of p_r over the rows of template t and P_x the sum over the rows of extrinsic sample x
(`get_n_effective_total_i_e`, `dot_pe/base_sampler_free_sampling.py`),

    n_effective_i = 1 / Σ_t P_t²,      n_effective_e = 1 / Σ_x P_x².

The values are read from each run's `summary_results.json`, the IAS-prior summary. The minima
150 and 100 are `MIN_NEFF_I` and `MIN_NEFF_E` of o4b-pe-paper
`src/o4a_pe_paper/pe_lib/failure_modes.py`, which applies them to the same file
(`analyze_rundir`).

### Headroom ratios h_i and h_e

    h_i = n_effective_i / 150,      h_e = n_effective_e / 100.

A run with h_i < 1 or h_e < 1 already fails a minimum.

### Row reduction H

For a run with h_i ≥ 1 and h_e ≥ 1,

    H = h_i × h_e.

If n_effective_i scales in proportion to the number of templates and n_effective_e to the number
of extrinsic samples, using h_i times fewer templates and h_e times fewer extrinsic samples
reduces the rows by H and leaves both at their minima. H is not defined for a run that fails a
minimum (`design.md`, design note of 2026-11-22 12:05).

### Reduction needed, R

The *nominal cut* c is the largest b over a run's stored rows minus 20: the rows dot-PE intends
to keep are those with b ≥ c. For GW240525_201644, c = 16.379, and the estimated number of rows
with b ≥ c is N̂(c) = 7.40e8 (measured in the report on rows above the cut,
`research/rows-above-cut/2026-11-22_measurement.md`). Storing all of them within `size_limit`
needs a reduction R = N̂(c) / 1e7 = 74. R is not measured for any other run.

### Best-fit log-likelihood of a run

Figure 2 orders runs by `bestfit_lnlike_max` of `summary_results.json`, the largest b over at most
1e4 posterior draws (`dot_pe/inference.py:1568`, `:1590`). It is below the largest b over the
stored rows (for GW240525_201644, 35.29 against 36.38; `research/progress.md`, 2026-11-22 10:27)
and serves only to order runs by signal-to-noise ratio.

### Choice of prior

The minima are applied under the IAS prior only, as production does (`design.md`, design note of
2026-11-23 06:52, made after the results on the user's decision that high effective sample sizes
under the LVK prior are welcome but not a goal). Under the LVK prior (the LVK spin and distance
prior, by reweighting) the effective sample sizes are lower: 175 runs fail a minimum there,
174 of them only there.

## Results

How much headroom does each run have in n_effective_i and in n_effective_e?

![headroom ratios](review/fig1_headroom_i_e.png)

One run fails: GW241011_233834, with n_effective_i 8.6 and n_effective_e 70 (best-fit
log-likelihood 628). The other 287 have h_i between 1.01 and 147 (median 19) and h_e between
1.00 and 16.9 (median 2.4). Fewer extrinsic samples could therefore reduce the rows by a median
factor of only 2.4; most of the room is in templates. GW240525_201644 has h_i = 10.2 and
h_e = 2.6.

What is H against the best-fit log-likelihood, and how does it compare with the reduction
GW240525_201644 needs?

![H against best fit](review/fig2_H_vs_best.png)

H is largest at low signal-to-noise ratio: for best-fit log-likelihood below 50 its median is 70
(64 of 137 runs have H ≥ 74), for 50 to 100 it is 36 (29 of 112), and above 100 it is 17 (1 of
38). GW240525_201644 has H = 27, below its R = 74: under the proportional assumption, fewer
samples alone cannot bring its stored rows down to the nominal cut.

## Observations

Not part of this measurement's chain: H is largest at low best-fit log-likelihood, and so is the
posterior mass dot-PE's stored rows miss, as measured in another study. For each event the paper repos also have a reference posterior
from cogwheel, a sampler that does not store a fixed number of rows. The median fraction of that
posterior's mass at b below dot-PE's floor is 0.75 for best-fit log-likelihood 30 to 40, 0.10 for
75 to 100 and 0 above 200 (o4b-pe-paper `dev/row_limit_check/HANDOFF_dotpe_row_limit.md`).

## Limitations

- H assumes n_effective_i scales with the number of templates and n_effective_e with the number
  of extrinsic samples. The one test available does not follow it: GW240525_201644 with the
  first eighth of each bank had n_effective_i 363 instead of 1535 (4.2 times lower for 8 times
  fewer templates) and n_effective_e 329 instead of 262 (o4b-pe-paper
  `dev/row_limit_check/gw240525_runs.md`). In that test the floor also moved down, from 24.35 to
  21.19, because the same 1e7 rows were spread over fewer templates; the test therefore mixes
  the effect of fewer templates with that of a lower floor. Taken at face value it suggests H
  underestimates the room, but the direction is not established.
- H multiplies h_i and h_e as if independent; the same test shows they are not.
- R is measured for one event only, so whether any of the 94 runs with H ≥ 74 has enough room
  for its own R is not known.
- The summaries are read as they are on disk. Every `summary_results.json` was written after the
  cross-bank n_effective fix (dot-PE 2403731, 2026-07-13 13:44).

## Reproduction

    python research/sample-headroom/headroom.py     # env o4a-pe, login node, seconds

Output: `artifacts/research/sample-headroom/headroom.csv` (IAS- and LVK-prior values per run) and
`research/sample-headroom/review/`.
