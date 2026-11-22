# E1 design: rows above the nominal cut in GW240525_201644 run_2 (measurement)

Class: measurement. It describes how many rows dot-PE's coherent stage would keep without a row
limit; it claims no flaw (that is E0). E2, E3 and E4 read its outputs.

## Question

How many rows of GW240525_201644's production run (o4b-pe-paper `run_2`) have a
distance-maximized log-likelihood b at or above the nominal cut c = 15.29?

No prediction: the request carries no hypothesis about this number. (Known bounds, not a
prediction: it is at least 1e7, since rows down to b = 24.35 already filled `size_limit`, and at
most the 3.7e10 rows that passed the per-block running threshold, `n_distance_marginalizations`
in run_2's `summary_results.json`.)

## Method

- The coherent stage of run_2 is recomputed block by block with the production worker,
  `dot_pe.thin_coherent.run_thin_iblock`, unchanged, called on the setup that run_2 wrote
  (`load_thin_setup(run_2/mp_coherent_setup_bank_k, run_2)`), with two changes to the loaded
  setup dict: `max_output_per_block` = None (no per-block row cap), and no incoherent bound (as in
  run_2). The worker's running threshold (block maximum − 20) never exceeds c, because no block's
  maximum exceeds run_2's best fit 35.29; so every row with b ≥ c is returned, and the harness
  keeps those.
- i-blocks are formed exactly as `mp_inference` forms them: `inds_to_blocks(inds, 512)` on each
  bank's `run_2/banks/bank_k/intrinsic_samples.npz` `inds`; the extrinsic blocks are
  `inds_to_blocks(arange(13312), 512)`.
- Sample: a stratified random sample of n_b = 40 i-blocks per bank (120 blocks), seed 20261122,
  drawn without replacement (`plan.csv`). A pilot of the first 3 blocks of the sample (one per
  bank) measures time and memory per block before the rest is submitted; it is part of the
  sample and does not change n_b.
- Memory: the worker is called on one extrinsic block at a time, and its output is filtered to
  b ≥ c after each call, so at most one extrinsic block's accepted rows (≤ 512 × 512 × 100) are
  held. Each call starts its own running threshold, which is still ≤ c, so no row with b ≥ c is
  lost.
- c is set at full precision from run_2's `bestfit_lnlike_max` (35.28718742929617 − 20).
- Estimator: N̂(x) = Σ_k (B_k / n_b) Σ_{sampled blocks j in bank k} N_j(x), with B_k the number of
  i-blocks in bank k and N_j(x) the count of rows with b ≥ x in block j; 68 % interval from 2000
  bootstrap resamples of blocks within each bank.
- Harness validation (in Setup, not a result): for each sampled block, every run_2 row of that
  block (`run_2/banks/bank_k/prob_samples.feather`, rows whose `i` is in the block) is found in
  the harness output with the same (i, e, o) and `bestfit_lnlike` equal to 1e-5 relative
  (the harness stores float32). Rows missing because the newer masks remove them (non-positive
  <h|h> or best-fit distance below 1 Mpc) are counted and reported; any other missing or
  mismatched row stops the experiment.

## Variable swept and held fixed

Swept: the threshold x on b, from c = 15.29 to the best fit 35.29. Held fixed: run_2's selected
templates, extrinsic samples, setup and blocking; the dot-PE worker code (branch `research`).

## Measure

N̂(x), the estimated number of rows of run_2 with b ≥ x, a count of (intrinsic, extrinsic,
phase) rows. The answer is N̂(c) with its 68 % interval.

## Glossary

- row, `size_limit`, nominal cut, floor, b: as in `research/row-limit/design.md`.
- c: the nominal cut of run_2, its best fit over kept rows minus 20, 16.379 (see the design note of 2026-11-22 10:35).
- i-block: a group of up to 512 selected templates of one bank, the unit of work of one coherent
  worker (`dot_pe/mp_inference.py:433`).
- N_j(x), N̂(x), B_k, n_b: defined under "Method".
- o: the phase index of a row, one of n_phi = 100 equally spaced reference phases.
- bootstrap within bank: resampling, with replacement, n_b blocks from each bank's sampled blocks
  and recomputing N̂(x); the 16th and 84th percentiles of 2000 resamples give the 68 % interval.

## Figure chain

1. For one block (the pilot block of bank_0): how are its rows distributed in b, from c to its
   maximum? (histogram of b, log count, lines at c and at run_2's floor 24.35)
2. Across the sampled blocks: how much do blocks differ in N_j(c)? (scatter of N_j(c) against
   block maximum b, coloured by bank, log scale)
3. Across the run: how does N̂(x) grow as x decreases from the best fit to c? (N̂(x) with its
   68 % band, log scale; horizontal line at `size_limit` 1e7, vertical lines at the floor and at
   c)

## Confounders and how each is fixed

- Code version: run_2 was made with dot-PE 2403731; the harness runs 4516a13 plus the two merged
  branches, which add masks for non-positive <h|h>, best-fit distance below 1 Mpc and (off here)
  the incoherent bound. The harness validation detects any change in the rows run_2 kept; rows
  masked by the newer code are counted and reported.
- Heavy-tailed block counts: a few blocks may hold most rows. The stratified sample and the
  bootstrap interval carry this; the cap of 120 blocks bounds the cost.
- Per-block cap in run_2: run_2 kept at most 64 724 rows per block (`max_output_per_block`,
  `mp_coherent_setup_bank_0/setup.json`), so run_2's rows are not a uniform top slice; the
  validation therefore only checks inclusion, not equality of sets.

## Outputs and compute

- `harness.py` (this folder) runs one or more blocks per call and writes, per block, the rows
  with b ≥ c (columns `i` int32, `e` int16, `o` int8, `bestfit_lnlike` float32, `lnl_marginalized`
  float32, `ln_posterior` float64, `d_h_1Mpc` float32, `h_h_1Mpc` float32) to
  `artifacts/research/rows-above-cut/blocks/<bank>_<block>.feather`, and the per-block counts to a
  CSV. `analyze.py` computes N̂(x) and the figures.
- LSF: env `dot-pe`, 1 core, 16000 MB, physics-short. Per block about 2 min is expected
  (run_2's coherent stage, 5.22 h × 7 workers / 1162 blocks); the pilot measures it.

## Design note, 2026-11-22 10:35 (before any E1 result was read)

E0's job showed that run_2's `summary_results.json` `bestfit_lnlike_max` (35.287) is the
maximum over the 1e4 posterior draws (`dot_pe/inference.py:1568`, `samples["bestfit_lnlike"]`),
not over the kept rows; the largest `bestfit_lnlike` in run_2's `prob_samples.feather` is
36.379 (bank_0, template 699179). The nominal cut is therefore c = 36.379 − 20 = 16.379, and the
analysis uses this c. The harness, already running for the pilot, keeps rows with
b ≥ 15.287 ≤ c, a superset of the rows with b ≥ c, so it is not changed. A worker's running
threshold can now reach up to c, so N̂(x) is reported only for x ≥ c. The handoff and
o4b-pe-paper `dev/row_limit_check/row_limit.csv` take the nominal cut from the same summary
field; that is recorded in `research/progress.md`, not changed here.
