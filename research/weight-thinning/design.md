# E3 design: thinning rules for the row limit, tested offline on complete rows (feature, correctness)

Class: feature (a capability dot-PE lacks: keeping a reweighted random subset of rows instead of
the top rows). Its one experiment shows (1) whether each rule reproduces a reference posterior and
(2) how it performs against the rule it would replace. No `dot_pe/` code changes; the rules are
applied offline to E1's rows.

## Question

At the row budget production runs face, does each of the thinning rules (a), (b) and (e) below
pass the acceptance test, which asks that it reproduce the posterior of all rows above the
nominal cut without bias and with bounded spread?

## Reference

T: all rows with b ≥ c = 16.379 of E1's 120 sampled i-blocks of GW240525_201644 run_2
(`artifacts/research/rows-above-cut/blocks/`), each with its exact weight. T is a complete,
untruncated row table for those blocks; its posterior is the reference.

## Rules compared

All rules keep at most k rows of T and produce per-row weights ŵ. Let ln w be a row's log weight
on the common scale, `ln_posterior` − ln N_k (`dot_pe/inference.py:1523`; N_k is equal for the
three banks of run_2). Let u ~ U(0, 1] be drawn once per row from a seeded generator, and let the
key be κ = ln w − ln u.

- (d) current rule, as production runs it: per block keep the top k_blk,bank by b, per bank the
  top k by b, then the top k by b over all banks; ŵ = w. A comparator.
- (c) top k by ln w (deterministic), same three stages; ŵ = w. A comparator that cuts on weight
  instead of b.
- (a) priority sampling, one stage: keep the k rows of largest κ; τ = the largest dropped κ;
  ln ŵ = max(ln w, τ).
- (b) priority sampling in the three stages of production: per block keep the k_blk,bank rows of
  largest κ (τ_blk = the block's largest dropped κ), per bank the k of largest κ (τ_bank), then
  the k of largest κ over all banks (τ_glob); with τ = max(τ_blk, τ_bank, τ_glob),
  ln ŵ = max(ln w, τ).
- (e) as (b), with the key built on a defensive weight that protects both priors:
  ln w_def = ln((w / S + w_LVK / S_LVK) / 2), with w_LVK = w × the per-row LVK spin-prior
  ratio and S, S_LVK fixed constants that do not depend on u (here T's sums Σw and Σw_LVK; in
  production any fixed estimate, e.g. from the first block, keeps the rule unbiased and only
  changes its efficiency); with τ = max(τ_blk, τ_bank, τ_glob), ŵ is max(w_def, e^τ) × w / w_def
  for the IAS posterior and × w_LVK / w_def for the LVK posterior.

Why (b) and (e) are unbiased (stated for the report's Setup): under keys κ = ln w − ln u, a row
is kept exactly when κ exceeds T, the largest key dropped from any group that contains it, and
for fixed other keys P(kept) = min(1, w e^(−T)); weighting a kept row by 1/P(kept), that is by
max(w, e^T), makes every sum over rows unbiased (Duffield, Lund & Thorup 2007, J. ACM 54, 32).
Top-k by a fixed key is order-independent, so worker and bank order do not matter.

## Variable swept and held fixed

Swept: the rule. Held fixed: T; k and k_blk; the seeds; the measures.

- k = |T| × 1e7 / N̂(c), the production ratio of kept rows to rows above the cut, with N̂(c)
  from E1, fixed from E1's report before any rule is applied. In production `size_limit`
  applies per bank (`dot_pe/mp_inference.py:467`) and again over all banks
  (`dot_pe/inference.py:1527`); k plays both roles here.
- k_blk,bank: the production block cap, unchanged, because T's blocks are production blocks
  with all their rows: `max(50 000, 2 × 1e7 // B_k)` (`dot_pe/thin_coherent.py:112`), which is
  64 724 for bank_0 (as in run_2's `setup.json`), 50 000 for bank_1 and 50 000 for bank_2.
- 100 seeds per rule (seed s = 0 … 99; numpy `default_rng(s)` over T's rows in a fixed order).

## Measures (all against T's reference values)

Per rule and seed, with ŵ normalised:
1. b_min: the lowest b among kept rows (log-likelihood).
2. Σŵ / Σw over T, the evidence ratio (unitless).
3. The 5 %, 50 % and 95 % quantiles of chi_eff, chirp mass (solar masses) and q, under the IAS
   prior.
4. The posterior weight at chi_eff < 0.4 under the LVK spin prior, P_LVK (unitless). The
   threshold 0.4 is the handoff's measure for GW240525_201644 (o4b-pe-paper
   `dev/highlights/gw240525_tail.txt`), where cogwheel and dot-PE differ most.
5. n_effective_i and n_effective_e of ŵ (counts).

Bias: the mean over seeds minus the reference; spread: the standard deviation over seeds.

## Accept / reject (fixed now)

A rule passes if, on every measure in 2 to 4, |bias| ≤ 2 × spread / √100 + 0.1 × spread (bias
indistinguishable from 0 at 100 seeds, or small against the spread), and its spreads are below
these ceilings: 0.05 for the evidence ratio; 0.1 × T's posterior standard deviation of the
parameter for each quantile; 0.02 for P_LVK; and its mean n_effective_i is not below that of
rule (d). Rules (c) and (d) are deterministic (spread 0) and are comparators only; their
absolute errors against T are reported. Among passing rules, the
one with the smallest spread of P_LVK is chosen for E5; ties go to the simpler rule
((a) < (b) < (e)). If (a) passes and (b) does not, the block-level cap is the problem and E5 must
remove it. If no rule passes, the outcome is reported as a null result and direction 3 (E4)
becomes the next question.

## Glossary

- row, b, `size_limit`, nominal cut: as in `research/row-limit/design.md`; c, N̂(c), i-block:
  as in `research/rows-above-cut/design.md`.
- T, k, k_blk, ln w, u, κ, τ, τ_blk, τ_glob, ŵ, w_LVK, w_def: defined above.
- chi_eff: (m1 s1z + m2 s2z)/(m1 + m2), from the bank's `intrinsic_sample_bank.feather`; chirp
  mass and q = m2/m1 likewise.
- LVK spin-prior ratio: `o4a_pe_paper.pe_lib.lvk_reweight.ln_prior_ratio_fast` on the row's
  template (o4b-pe-paper, read-only). The distance part of the LVK prior is not row-level in
  dot-PE and is left out of P_LVK here.
- seed-to-seed spread, bias: defined under "Measures".

## Figure chain

1. For seed 0: where in b do the kept rows of each rule lie, against T? (histograms of b, T and
   rules (d), (a), (b), (e), log count)
2. For seed 0: what chi_eff posterior under the LVK spin prior does each rule give, against T?
   (weighted histograms)
3. Across seeds: what is P_LVK per seed and rule? (strip plot of the 100 values per rule, line
   at T's value)
4. Across measures: bias and spread of every measure per rule (table).

## Confounders and how each is fixed

- T covers 120 of about 1162 blocks; posteriors on T differ from run_2's full posterior. The test
  is of the rules against T, not of T against cogwheel.
- Rows below c are absent from T; the rules are tested only on the cut that production intends.
- The ratio k/|T| is from N̂(c), itself an estimate; its 68 % interval is reported, and the
  accept rule is applied at the central k only.
- Measures 3 and 4 use normalised ŵ, so they are ratio estimators with a bias of order 1/n_eff,
  and quantiles are nonlinear; only measure 2 is exactly unbiased. The bias criterion's
  0.1 × spread allowance covers this; the report states it.
- n_effective of ŵ is not an estimate of T's n_effective: weights raised to e^τ are more even,
  which raises it. The MIN_NEFF gates of the paper repos would read the thinned value; the
  report gives both.
- Float32 storage of b and of the inner products in T: b is used only for (d) and the figures.

## Code and compute

`thin.py` (rules and measures) and `run.py` in this folder. One LSF job per rule, env `o4a-pe`
(needs `lvk_reweight`; `dot_pe` is not imported), 1 core, memory 3 × the size of T in memory
(set from E1's output), physics-short.
