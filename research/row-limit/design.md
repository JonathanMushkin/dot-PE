# E0 design: posterior mass dot-PE's row limit removes (problem experiment)

Class: bug fix, problem experiment. The solution experiment (E5) runs `test_mass_below_floor.py`
unchanged on the new runs.

## Question

For each test event, what is ΔW, cogwheel's posterior mass at a distance-maximized
log-likelihood below the production run's floor F minus dot-PE's posterior mass there? The two
masses M_cog(F) and M_dot(F) are its components, reported alongside; the prior (IAS, LVK) is a
reported split, not a second sweep.

Event choice: the three test events were chosen after the handoff's results, as its acceptance
events (two of the five highest W and the poster child); the control was chosen from the same
table as the event with best fit > 200, |offset| < 2 and the smallest W_IAS above 0.

## Prediction

From the handoff (o4b-pe-paper `dev/row_limit_check/below_floor.md`): dot-PE's production
posterior has no mass below its floor by construction, while cogwheel's has W_IAS = 0.31 for
GW240525_201644, 0.97 for GW230902_150325 and 0.97 for GW231113_150041, so ΔW ≈ W. The
prediction is disproved if ΔW is within its noise of 0 for any of the three low-SNR events.

## Pass criterion for the solution experiment (fixed now)

A fix passes E0's test when |ΔW| ≤ 2 × its noise and the noise is below 0.1, under both priors, for the control,
GW240525_201644 and GW230902_150325. GW231113_150041 is reported but does not count, because its
best-fit offset (below) can keep ΔW > 0 under a correct fix. In the problem run M_dot(F) = 0 by
construction (F is the lowest kept b); the report states this and presents ΔW, not M_dot, as the
finding.

## Control

GW240621_195059 (o4b, production `run_1`, best fit 401.2, floor 385.0, cogwheel W_IAS 0.004):
a high-SNR event where the floor lies below nearly all of cogwheel's mass, so ΔW should be
within noise of 0. If the control showed ΔW ≫ 0 the measure would be picking up something other
than the row limit.

## Variable swept and held fixed

Swept: the event (control, then GW240525_201644, GW230902_150325, GW231113_150041). Held fixed:
each event's production dot-PE run and production cogwheel run, as named by `production_run` in
the paper repo's specs; F, the production floor, taken once from o4b-pe-paper
`dev/row_limit_check/row_limit.csv` and written into the test script as a constant per event, so
the solution experiment evaluates the same F.

## Measure

- b, the distance-maximized log-likelihood of a posterior sample, (d|h)²/(2(h|h)) for (d|h) > 0,
  else 0. dot-PE stores it as `bestfit_lnlike` (`dot_pe/thin_coherent.py:258`); for cogwheel it
  is computed from the stored `lnl` and `h_h`, (d|h) = lnl + h_h/2 (as in o4b-pe-paper
  `dev/row_limit_check/below_floor.py`, function `bestfit`).
- M_s(F), the posterior mass of sampler s with b < F, from that sampler's posterior samples and
  their weights. Unitless fraction in [0, 1].
- ΔW = M_cog(F) − M_dot(F), under the IAS prior (dot-PE `samples.feather`, cogwheel production
  `samples.feather`) and under the LVK prior (dot-PE `samples_lvc.feather`; cogwheel run under
  the LVK prior where one exists, otherwise the production samples reweighted to the LVK prior,
  as `below_floor.py` does).
- Noise: the binomial standard error of each M, √(M(1−M)/n_eff) with n_eff the Kish effective
  sample size of the weights, added in quadrature.

## Glossary

- row: one (intrinsic, extrinsic, phase) combination kept by dot-PE, a row of
  `prob_samples.feather`.
- `size_limit`: the maximum number of rows dot-PE keeps (1e7 in all production runs).
- nominal cut: the best fit minus `max_bestfit_lnlike_diff` = 20 (`dot_pe/thin_coherent.py:278`).
- floor F: the lowest `bestfit_lnlike` among a run's kept rows.
- b, M_s(F), ΔW: defined under "Measure".
- (d|h), (h|h): the matched-filter inner products of data and template, and of template with
  itself, at the sample's distance (cogwheel) or at 1 Mpc (dot-PE `d_h_1Mpc`, `h_h_1Mpc`);
  cogwheel stores `lnl` = (d|h) − (h|h)/2 and `h_h` = (h|h).
- best fit: the largest b over a run's rows (dot-PE `bestfit_lnlike_max`) or over cogwheel's
  samples.
- offset: dot-PE's best fit minus cogwheel's best fit.
- W_IAS, W_LVK: the handoff's names for M_cog(F) under the IAS and LVK priors.
- n_eff: the Kish effective sample size of weights w, (Σw)²/Σw².
- o: the phase index of a row, one of n_phi = 100 equally spaced reference phases.
- IAS prior: dot-PE's production prior; LVK prior: the LVK spin and distance prior as
  implemented in o4b-pe-paper `src/o4a_pe_paper/pe_lib/lvk_reweight.py`.

## Figure chain

1. Control: what is the distribution of b over cogwheel's posterior and over dot-PE's, relative
   to F and to the nominal cut? (two weighted histograms of b, IAS prior, vertical lines at F and
   at the nominal cut)
2. Each test event (GW240525_201644, GW230902_150325, GW231113_150041): the same figure, one
   panel per event, same conventions.
3. Across the four events: what is ΔW, under each prior, with its noise? (points with error bars
   against dot-PE's best fit)

## Confounders and how each is fixed

- Different likelihood implementations: F and dot-PE's b come from dot-PE's likelihood (its
  relative binning, `asd_drift`), cogwheel's b from cogwheel's. A systematic difference in b
  shifts M_cog(F). The offset of each event is reported next to ΔW.

- Best-fit disagreement: GW231113_150041 has cogwheel's highest b 5.0 below dot-PE's best fit
  (handoff), so part of ΔW there may come from a different mode rather than the row limit. Named
  in the report; the event stays because it is an acceptance event of the handoff.
- LVK reweighting of cogwheel's production samples has low effective sample size for some events;
  the n_eff of each M enters the noise. For GW240525_201644 a cogwheel run under the LVK prior
  exists and is used.
- dot-PE's posterior samples are finite draws (1e4 under the IAS prior); their n_eff enters the
  noise.
- Code version: the production runs were made with dot-PE 2403731 (GW240525_201644 run_2,
  `timing.json`) or other commits listed in o4b-pe-paper `docs/pe/dot_pe_versions.csv`; the
  test reads outputs only, so the dot-PE version of the reading code does not matter.

## Code and compute

`test_mass_below_floor.py` in this folder takes the dot-PE rundir per event as a
command-line argument (`--dotpe EVENT=RUNDIR`, default the production run); F and the cogwheel
runs are fixed in the script. It is run in conda env `o4a-pe` (it imports o4b-pe-paper's
`dev/highlights/compute.py` and `reweighting_check.py` read-only for cogwheel loading and LVK
reweighting). It reads about 10 posterior files; it runs as one LSF job on the `analysis`
profile (1 core, 8000 MB). Output: `artifacts/research/row-limit/mass_below_floor.csv`, and the
figures under `review/`.
