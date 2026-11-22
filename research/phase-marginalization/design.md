# E4 design: row reduction from marginalizing over phase (measurement)

Class: measurement. Direction 3 of the user's list: store one row per (intrinsic, extrinsic)
pair, with the likelihood marginalized over the reference phase, instead of one row per phase.
This experiment measures how many fewer rows that would leave; it changes no code.

## Question

In T (E1's complete rows of GW240525_201644 run_2 above the nominal cut, from the 120 sampled
i-blocks), what is ρ, the number of rows divided by the number of distinct (intrinsic, extrinsic)
pairs among them?

No prediction: the request carries an estimate ("something like ×3 less samples", from the top
1e7 rows of production runs, o4b-pe-paper `dev/row_limit_check/phase_multiplicity.md`), not a
hypothesis about rows down to the nominal cut; the design records that estimate and does not
predict ρ. The ×3 estimate is compared with ρ(24.35), the value at run_2's floor, in one
sentence of the report.

## Variable swept and held fixed

Swept: the threshold x on b, from c = 16.379 to the best fit over rows, 36.379 (ρ(x) counts
rows and pairs with b ≥ x, a pair counted if any of its phases has b ≥ x). Held fixed: T.

## Measure

ρ(x) = (rows with b ≥ x) / (distinct (i, e) pairs with at least one row with b ≥ x); unitless,
between 1 and n_phi = 100. The answer is ρ(c). The reduction a phase-marginalized row table
would give at the nominal cut is ρ(c), if the marginalized cut keeps the same pairs.

## Display rule (fixed now)

ρ(x) is plotted only where at least 100 pairs remain (summed over T's blocks); the pair count is
drawn on a second axis.

## Glossary

- row, b, nominal cut: as in `research/row-limit/design.md`; T, c: as in
  `research/weight-thinning/design.md` and `research/rows-above-cut/design.md`.
- pair: an (intrinsic, extrinsic) combination, (bank, i, e).
- direction 3: the user's option "marginalize over phi_ref".
- best fit over rows: 36.379, the largest `bestfit_lnlike` in run_2's `prob_samples.feather`
  (E1 design note of 2026-11-22 10:27).
- 68 % bootstrap interval: as defined in `research/rows-above-cut/design.md`.
- ρ(x): defined under "Measure". n_phi: the number of reference phases per pair, 100
  (run_2 `run_kwargs.json` and `mp_coherent_setup_bank_0/setup.json`).

## Figure chain

1. At x = c: how many phases does each pair have above the cut? (histogram over pairs of the
   number of rows per pair, 1 to 100, log count)
2. Across x: how does ρ(x) change between the nominal cut and the best fit? (ρ(x) against x;
   vertical line at run_2's floor 24.35)

## Confounders and how each is fixed

- A phase-marginalized cut would rank pairs by a marginalized likelihood, not by their best
  phase; the set of pairs kept at the same budget then differs. ρ(c) is stated as the reduction
  for the pairs of T, not as the reduction of a specific marginalized cut.
- The phase grid is 100 equally spaced values; ρ counts grid points, so a coarser grid
  (direction "smaller n_phi") would scale rows directly and is not measured here.
- T covers 120 of about 1162 blocks; ρ is estimated as Σ rows / Σ pairs with the B_k / n_b
  stratum weights of E1, a ratio estimator with a small bias; the 68 % bootstrap interval over
  blocks within bank carries its spread.
- Storage: a phase-marginalized row must also carry what is needed to draw a phase later
  (per-mode inner products or per-phase values); its cost per row is out of scope here.

## Code and compute

`rho.py` in this folder, env `dot-pe` or `o4a-pe`, reads E1's block files; one LSF job,
1 core, memory 3 × T's (i, e, b) columns.
