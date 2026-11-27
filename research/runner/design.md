# Design: end-to-end GPU runner (P4, engineering phase)

## Question

Does `dot_pe_gpu.run.run` run a production configuration from start to finish on one GPU, write a
rundir that the o4b tools read, and reproduce a production run when given that run's incoherent
selection and extrinsic samples?

## Code

- `dot-pe-gpu/dot_pe_gpu/run.py`: `run()`, with the arguments of `mp_inference.run`, and a
  command line.
- `coherent_stage.py`: stage 5 per bank.
- `stages.py`: stage 2 per bank, and `Residency`.
- `bank.py`: `load_selected`, the disk fallback when the waveforms are not resident (`load_inds`).

Stages 1, 3, 4 and 6 call the CPU functions of `mp_inference.run` unchanged.

## Test runs

1. Like-for-like. GW240525_201644 run_2 settings (3 banks, n_ext = 13312, n_phi = 100,
   `size_limit` 1e7, bestfit), with:
   - `load_inds` = True and `inds_path` = run_2's `banks/bank_k/intrinsic_samples.npz`;
   - `extrinsic_samples` = run_2's `extrinsic_samples.feather`, with `response_dpe.npy` and
     `timeshift_dbe.npy` copied beside it, as `rerun_gw240525.py` in o4b does.

   This exercises stages 1, 3, 4 (recycled), 5 (waveforms from disk) and 6.
2. Fresh. The same event and settings with `row_selection="phase_marginalized_priority"` and
   nothing recycled. This exercises all stages, including stage 2 and the residency.

Both run as shared L40S jobs with `-W` ≤ 30 and `gmem` sized from P3 gate 5. A run that needs
more is split: test 1 first.

## Gates (fixed before any result)

1. Test 1 completes, and its rundir has every file of a CPU rundir except
   `mp_coherent_setup_*`.
2. Test 1 against run_2:
   1. |Δ ln_evidence| ≤ 0.05 in total and per bank;
   2. n_effective_i and n_effective_e within 5%;
   3. `bestfit_lnlike_max` and `lnl_marginalized_max` within 1e-3;
   4. `n_distance_marginalizations` within 0.1%.

   run_2 predates the incoherent-bound mask, so the bound is reported but not expected to move
   these numbers. The P2 test blocks never triggered it at +20.
3. o4b compatibility: o4b's `lvk_reweight.reweight_rundir`, imported read-only in the `o4a-pe`
   env, runs on the test-1 rundir copied to our artifacts.
4. Test 2 completes, and its stage-2 selection equals run_2's `intrinsic_samples.npz` up to
   templates within 1e-3 of the threshold (as P3 gate 3.2). The wall time per stage is reported.
