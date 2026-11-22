# E5 design: solution experiment for the row limit

Class: bug fix, solution experiment for `research/row-limit/design.md` (E0). It runs E0's test,
`test_mass_below_floor.py`, unchanged, on dot-PE runs made with the fix, and adds the cost and the
handoff's acceptance numbers. The fix is `row_thinning="pairs"` (dot-PE commit 6454106 on branch
`research`, `dot_pe/row_thinning.py`), rule (f) of E3 (`research/weight-thinning/`).

## Question

With the fix, is ΔW, cogwheel's minus dot-PE's posterior mass below the production floor F
(E0, Setup), within E0's pass criterion for the failing event GW240525_201644 and the working
event GW240621_195059?

## Prediction

From E3: rule (f) reproduced the posterior of every row above the nominal cut of
GW240525_201644 to within its seed-to-seed spread, and those rows reach down to the cut, below F.
Prediction: with the fix, M_dot(F) for GW240525_201644 rises from 0.0004 to within E0's noise of
M_cog(F) = 0.31 (IAS prior) and 0.60 (LVK prior), so that |ΔW| ≤ 2σ_ΔW with σ_ΔW < 0.1 under both
priors; for GW240621_195059, ΔW stays within the same criterion. Disproved if either event fails
the criterion under either prior.

## Runs (2 of the 6-rerun budget)

| event | role | production run | its wall time | expected wall time with the fix |
|---|---|---|---|---|
| GW240525_201644 | failing | o4b `run_2` (dot-PE 2403731) | 5.84 h | about 5.9 h |
| GW240621_195059 | working | o4b `run_1` (dot-PE a94bf7c) | 0.15 h | about 0.15 h |

Each rerun copies the production run's `run_kwargs.json` and changes only: the dot-PE version
(branch `research`), the rundir (`artifacts/research/row-limit/runs/<event>/pairs`),
`row_thinning="pairs"`, and the incoherent stage and extrinsic samples, recycled from the
production run (`load_inds`, `inds_path`, `extrinsic_samples`, as in o4b-pe-paper
`dev/row_limit_check/rerun_gw240525.py`). LSF: the `pe` profile, 8 cores and 128 000 MB,
physics-medium, 7 workers, as production. Expected cost: the coherent stage computes the same
rows as production; the fix adds a per-pair reduction on the accepted rows of each worker call
(sorting by pair) and three extra columns. Expected peak memory: production's (40 GB for
GW240525_201644 run_2) plus at most 30 %; a run above 1.3 × production's peak or 1.5 × its wall
time is reported as a cost failure.

The LVK-prior samples are produced by o4b-pe-paper's `lvk_reweight.reweight_rundir`, with
`ln_posterior` replaced by `ln_posterior + ln_thinning_weight` when it reads
`prob_samples.feather` (that function recomputes weights from `ln_posterior`); o4b-pe-paper is not
modified (`rerun.py`).

## Measures

1. E0's ΔW under both priors, with σ_ΔW, from `test_mass_below_floor.py --dotpe
   EVENT=RUNDIR --tag pairs`, unchanged.
2. The handoff's acceptance numbers for GW240525_201644 (o4b-pe-paper
   `dev/row_limit_check/HANDOFF_dotpe_row_limit.md`): the lowest kept b against the nominal cut
   16.379 (E1); the LVK-prior weight at chi_eff < 0.4 against cogwheel's 0.247; n_effective_i and
   n_effective_e under both priors against run_2's (1535 / 262 IAS, 232 / 98 LVK).
3. Cost: wall time and peak memory (LSF) against production.

## Pass criterion (fixed now)

The fix passes when measure 1 meets E0's criterion for both events under both priors, and its
cost is within the bounds above. Measure 2 is reported against the handoff; n_effective below
run_2's is reported as a finding, not a failure, since thinning trades rows near the floor for
rows down to the cut.

## Figure chain

E0's three figures, unchanged, with tag `pairs` (b histograms of the control, of GW240525_201644
and of GW230902_150325 / GW231113_150041 as in E0, the latter two still from production since
they are not rerun; ΔW per event), then:

4. For GW240525_201644: the chi_eff posterior under the LVK prior, production run_2, the rerun
   and cogwheel. (weighted histograms)

## Confounders and how each is fixed

- E0's test also reads GW230902_150325 and GW231113_150041; they are not rerun (budget and the
  user's request of one failing and one working event), so their ΔW stays the production value
  and is not counted for this experiment.
- The production runs were made with older dot-PE commits; the rerun recycles their incoherent
  stage and extrinsic samples, so the only other code differences are the masks of
  4516a13 and 2130054 (off by default for the incoherent bound), which E1's validation found to
  change no row of GW240525_201644 run_2.
- Thinning is random (seeded by the run seed); one rerun per event gives one draw, whose spread
  E3 bounds (P_LVK spread 0.002 on T).
- The LVK reweighting adds `ln_thinning_weight` through a wrapper; o4b-pe-paper's own reweighting
  of a thinned run would omit it (required follow-up, noted in the report).
