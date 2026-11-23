# Row reduction from marginalizing over phase, GW240525_201644

This report measures how many fewer rows dot-PE would need to keep for GW240525_201644 if it
stored one row per (intrinsic, extrinsic) pair, with the likelihood summed over the reference
phase, instead of one row per phase. That reduction tells how much of the gap between the rows
above the nominal cut and dot-PE's row limit phase marginalization can close.

Code and data: dot-PE branch `research` at commit 38f92b4; the rows are those of the companion
measurement of rows above the cut (E1, `research/rows-above-cut/2026-11-22_measurement.md`),
recomputed from GW240525_201644's production run (o4b-pe-paper `run_2`, dot-PE 2403731). It
supersedes no report.

## Question and answer

The question: in T, the complete set of rows whose distance-maximized log-likelihood b is at
least c = 16.379, the nominal cut, from E1's 120 sampled i-blocks (groups of up to 512
templates), what is ρ(c), the number of rows divided by the number of distinct (intrinsic,
extrinsic) pairs among them (all defined in Setup)?

The answer: ρ(c) = 9.20 rows per pair (68 % bootstrap interval 9.18 to 9.22).

## Setup

A *row* is one (intrinsic, extrinsic, phase) combination; a *pair* is one (bank, i, e): the
template bank, the template index and the extrinsic-sample index. Each pair has n_phi = 100
reference phases on an equally spaced grid (run_2 `run_kwargs.json`). b is a row's
distance-maximized log-likelihood, (d|h)²/(2(h|h)) for (d|h) > 0 (`bestfit_lnlike`,
`dot_pe/thin_coherent.py:258`). The *best fit* is the largest b over run_2's kept rows, 36.379;
the nominal cut is c = 36.379 − 20 = 16.379; the *floor* is the lowest b run_2 kept, 24.35 (E1,
Setup). T holds 9.12e7 rows from 120 of run_2's 1162 i-blocks (E1).

### Rows per pair ρ(x)

For x, a threshold on b,

    ρ(x) = Σ_k (B_k / n_b) R_k(x) / Σ_k (B_k / n_b) P_k(x),

with R_k(x) the rows with b ≥ x and P_k(x) the pairs with at least one row with b ≥ x, summed
over the sampled blocks of bank k, B_k the bank's number of i-blocks and n_b = 40 the sampled
ones (E1's stratum weights). x runs in steps of 0.5 from 16.379, c rounded to three decimals.
The 68 % bootstrap interval is defined as in E1, Setup, "68 % bootstrap interval", with ρ in
place of N̂.

## Results

How many phases does each pair have with b at or above c, the nominal cut?

![phases per pair](review/fig1_phases_per_pair.png)

Pairs have between 1 and 26 of their 100 phases with b ≥ c; the counts are nearly even from 2
to 16 and fall off above 16.

How does ρ(x), the rows per pair with b at or above a threshold x, change between c and the best
fit, the largest b over run_2's kept rows?

![rho(x)](review/fig2_rho.png)

ρ(x) falls from 9.20 at c to 4.55 at x = 24.38, next to the floor, the lowest b run_2 kept. The
request's estimate of about 3 phases per pair, from the top 1e7 rows of production
runs (o4b-pe-paper `dev/row_limit_check/phase_multiplicity.md`), compares with ρ = 4.55 at run_2's floor.

## Limitations

- A phase-marginalized cut would rank pairs by their marginalized likelihood, not by their best
  phase, so the pairs it keeps at a fixed number of rows differ from T's; ρ(c) is the reduction
  for T's pairs.
- ρ(x) is a ratio of two estimated sums, with a small ratio-estimator bias; the bootstrap
  interval carries its spread.
- ρ counts points of the 100-phase grid, so a different grid changes it; this is not measured
  here.
- One event; ρ depends on how sharply the likelihood varies with phase, which depends on the
  signal's higher modes and inclination.
- A phase-marginalized row must also carry what is needed to draw a phase later; its storage cost
  is not measured here.

## Reproduction

    python research/lsf_submit.py --name e4_rho --env o4a-pe -- research/phase-marginalization/rho.py

Output: `artifacts/research/phase-marginalization/` (`result.json`, `rho.csv`) and
`research/phase-marginalization/review/`. LSF job 422570, 22 s, 0.25 GB.

Line numbers in citations of `dot_pe/` refer to commit 38f92b4 (before the phase-marginalization and priority-sampling code was added).
