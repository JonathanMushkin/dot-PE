# E5 design: solution experiment for the row limit

Class: bug fix, solution experiment for `research/row-limit/design.md` (E0). It runs E0's test,
`test_mass_below_floor.py`, unchanged, on dot-PE runs made with the fix, and adds the cost and the
handoff's acceptance numbers. The fix is `row_selection="phase_marginalized_priority"` (dot-PE commit 6003d04 and later on branch
`research`, `dot_pe/priority_sampling.py`), rule 4 of the E3 report (`research/weight-thinning/`).

## Question

With the fix, is ΔW, cogwheel's minus dot-PE's posterior mass below the production floor F
(E0, Setup), within E0's pass criterion for the failing event GW240525_201644 and the working
event GW240621_195059?

## Prediction

From E3: rule (f) reproduced the posterior of every row above the nominal cut of
GW240525_201644 to within its seed-to-seed spread, and those rows reach down to the cut, below F.
Prediction: with the fix, M_dot(F) for GW240525_201644 rises from about 0 (0.0004 in E0's table,
F being rounded) to within E0's noise of
M_cog(F) = 0.31 (IAS prior) and 0.60 (LVK prior), so that |ΔW| ≤ 2σ_ΔW with σ_ΔW < 0.1 under both
priors; for GW240621_195059, ΔW stays within the same criterion. Disproved if either event fails
the criterion under either prior.

## Runs (2 of the 6-rerun budget)

| event | role | production run | its wall time | expected wall time with the fix |
|---|---|---|---|---|
| GW240525_201644 | failing | o4b `run_2` (dot-PE 2403731) | 5.84 h | about 5.9 h |
| GW240621_195059 | working | o4b `run_1` (dot-PE a94bf7c) | 0.15 h | about 0.15 h |

Each rerun copies the production run's `run_kwargs.json` and changes only: the dot-PE version
(branch `research`), the rundir (`artifacts/research/row-limit/runs/<event>/phase_marginalized_priority`),
`row_selection="phase_marginalized_priority"`, and the incoherent stage and extrinsic samples, recycled from the
production run (`load_inds`, `inds_path`, `extrinsic_samples`, as in o4b-pe-paper
`dev/row_limit_check/rerun_gw240525.py`). LSF: the `pe` profile, 8 cores and 128 000 MB,
physics-medium, 7 workers, as production. Expected cost: the coherent stage computes the same
rows as production; the fix adds a per-pair reduction on the accepted rows of each worker call
(no sort: the rows arrive grouped by pair) and three extra columns. Expected peak memory: production's (40 GB for
GW240525_201644 run_2) plus at most 30 %; a run above 1.3 × production's peak or 1.5 × its wall
time is reported as a cost failure. These expectations are checked before submission against a
pilot: `smoke_phase_marginalized_block.py` (LSF) times one production i-block of GW240525_201644 run_2 over 4
of its 26 extrinsic blocks with the production cap, under both rules, twice each; its numbers are
recorded in `progress.md` and in the report. Pilot result (LSF 437770, after removing a sort from the
pair reduction): 25.5 s per call under both rules after the first, peak memory 6.6 GB under both;
the expected wall time of each rerun is therefore its production wall time within the run-to-run
scatter, and peak memory production's.

The LVK-prior samples are produced by o4b-pe-paper's `lvk_reweight.reweight_rundir`, with
`ln_posterior` replaced by `ln_posterior + ln_priority_weight` when it reads
`prob_samples.feather` (that function recomputes weights from `ln_posterior`); o4b-pe-paper is not
modified (`rerun.py`).

## Measures

1. E0's ΔW under both priors, with σ_ΔW, from `test_mass_below_floor.py --dotpe
   EVENT=RUNDIR --tag phase_marginalized_priority`, unchanged.
2. The handoff's acceptance numbers for GW240525_201644 (o4b-pe-paper
   `dev/row_limit_check/HANDOFF_dotpe_row_limit.md`): the lowest kept b against the nominal cut
   16.379 (E1); the LVK-prior weight at chi_eff < 0.4 against cogwheel's 0.247; n_effective_i and
   n_effective_e under both priors against run_2's (1535 / 262 IAS, 232 / 98 LVK).
3. Cost: wall time and peak memory (LSF) against production.

## Pass criterion (fixed now)

E0's pass criterion names three counted events (the control, GW240525_201644 and
GW230902_150325) and the condition |ΔW| ≤ 2σ_ΔW with σ_ΔW < 0.1 under both priors
(`research/row-limit/design.md`, "Pass criterion for the solution experiment"). At the user's
request (one failing and one working event, 2026-11-22) this experiment evaluates it on the
control and GW240525_201644 only; GW230902_150325 is not rerun, and the report states that E0's
acceptance is tested only in part. The fix passes when measure 1 meets that condition for both
events under both priors, and its cost is within the bounds above. Measure 2 is reported against the handoff; n_effective below
run_2's is reported as a finding, not a failure, since thinning trades rows near the floor for
rows down to the cut.

## Figure chain

E0's three figures, unchanged, with tag `phase_marginalized_priority` (b histograms of the control, of GW240525_201644
and of GW230902_150325 / GW231113_150041 as in E0, the latter two still from production since
they are not rerun; ΔW per event), then:

4. For GW240525_201644: the chi_eff posterior under the LVK prior, production run_2, the rerun
   and cogwheel. (weighted histograms)
5. Table: measures 2 and 3 for both events, production against the rerun.

## Confounders and how each is fixed

- E0's test also reads GW230902_150325 and GW231113_150041; they are not rerun (budget and the
  user's request of one failing and one working event), so their ΔW stays the production value
  and is not counted for this experiment.
- The production runs were made with older dot-PE commits; the rerun recycles their incoherent
  stage and extrinsic samples, so the only other code differences are the masks of
  4516a13 and 2130054 (off by default for the incoherent bound), which E1's validation found to
  change no row of GW240525_201644 run_2.
- Phase draws and priority sampling are random (seeded by the run seed); one rerun per event gives one draw, whose spread
  E3 bounds (P_LVK spread 0.002 on T).
- E3 tested rule (f) on the rows with b ≥ c; in production a pair's weight sums its phases over
  the worker's accepted rows, which reach down to the worker's running threshold (the block
  maximum minus 20), which can lie below c. The pair weights are then slightly larger than in
  E3's T, and correct for the cut production applies.
- E1's validation of the newer masks covered GW240525_201644 run_2 only, not GW240621_195059
  run_1 (dot-PE a94bf7c); a changed row there would show as a difference from production at
  high b.
- Gates: `tests/test_priority_sampling.py` (6 tests) passed before any rerun (login node, 20 s); the
  full suite runs as LSF job 429572 and must pass before submission.
- The LVK reweighting adds `ln_priority_weight` through a wrapper; o4b-pe-paper's own reweighting
  of a thinned run would omit it (required follow-up, noted in the report).

## Design note, 2026-11-23 07:35 (names only; user request)

The code was renamed so that names describe the two steps, not the goal: module
`dot_pe/row_thinning.py` → `dot_pe/priority_sampling.py`; argument `row_thinning` →
`row_selection`, value `"pairs"` → `"phase_marginalized_priority"`; `pair_reduce` →
`marginalize_phase`; columns `ln_pair_weight` → `ln_phase_marginalized_weight`, `thin_key` →
`priority_key`, `thin_tau` → `priority_tau`, `ln_thinning_weight` → `ln_priority_weight`;
`thinning_seed` → `selection_seed`; test file → `tests/test_priority_sampling.py`. Also
`lc` → `likelihood_calculator` (`thin_coherent.py`) and `likelihood_calc` →
`likelihood_calculator` (`evidence_calibration.py`). No behaviour changes; this design's text
uses the new names, and the commits it cites before the rename carry the old ones.

## Design note, 2026-11-23 08:04 (before the GW240525_201644 result; user decision)

If the fix passes on the control and GW240525_201644, reruns 3 and 4 of 6 follow, at once, on
the other two E0 events: GW230902_150325 (o4a-pe-paper `run_2`) and GW231113_150041
(o4a-pe-paper `run_0`), the production runs E0 used. `rerun.py` now reads O4a runs from
o4a-pe-paper. E0's test is then rerun with all four reruns, and the pass criterion becomes E0's
full one: |ΔW| ≤ 2σ_ΔW with σ_ΔW < 0.1 under both priors for the control, GW240525_201644 and
GW230902_150325 (GW231113_150041 is reported, not counted, as in E0). The report is written
first on the two events, then extended. If the fix fails, a null-result report is written and no
further run is made. The code is not merged into `size-limit` until the user has reviewed the
report.
