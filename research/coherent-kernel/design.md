# Design: coherent stage on the GPU (P2, engineering phase)

## Question

Can one GPU compute the coherent stage's per-i-block results (`dot_pe.thin_coherent.run_thin_iblock`)
with the same results as the CPU, at a per-block compute time that, over the 1162 i-blocks of
GW240525_201644 run_2, is at most 10% of that run's coherent stage (18793 s, so at most 1879 s)?
Waveform reads (P3), the per-bank merge and the device setup (P4) are timed in their own phases;
the end-to-end target is tested in P5.

## Code

`dot-pe-gpu/dot_pe_gpu/coherent.py` (`CoherentGPU.run_iblock`) and `lookup_table.py`. The
semantics are listed in the module docstring. The design choices are fixed here:

1. CuPy with NVRTC kernels and cuBLAS (`cupy.matmul`). This is the stack trex_gpu uses on this
   cluster. The torch in our envs is a CPU build.
2. The inner products by mode, <d|h> (4 modes) and <h|h> (10 mode pairs), are formed per
   intrinsic block against all extrinsic samples:
   - the per-template products are formed in complex128 and cast to complex64, because the
     `hh` weights reach 7e48 and the waveform amplitudes are near 1e-20;
   - the matmuls are in complex64.
3. Two passes over the (i, e) pairs. Each loops over the phase grid in registers, so no
   (i, e, o) array exists:
   - pass 1 computes the best-fit maxima for the running threshold;
   - pass 2 computes acceptance, the lookup table and the outputs.
4. In bestfit mode, rows are emitted only if their best fit is at least the k-th largest accepted
   pair maximum, with k = `max_output_per_block`. This bound is exact: the block's k-th best row
   is never below it. Then a top-k selects the block's rows.
5. The random draws (phase choice, priority key) use a counter-based hash. Priority-mode outputs
   therefore match the CPU in distribution, not row by row.

## Test data (read in place)

- GW240525_201644 run_2 (o4b): `mp_coherent_setup_bank_0`, `response_dpe.npy`,
  `timeshift_dbe.npy`, `extrinsic_samples.feather`, and `banks/bank_0/intrinsic_samples.npz`
  (selected `inds` and `incoherent_lnlikes`).
- Bank 0 has 158131 selected templates, so its i-blocks of 512 number 309. Test blocks:
  1. block 0;
  2. the block that holds template 378018, the template with the most rows (6765) in run_2's
     `prob_samples`.
- n_ext = 13312, n_phi = 100 and `max_output_per_block` = 64724, as in production.

## Reference

`research/coherent-kernel/make_reference.py` runs `run_thin_iblock` (dot-PE `gpu-v2`) on the two
blocks in four configurations:
1. bestfit without the bound;
2. bestfit with the bound;
3. priority without the bound;
4. priority with the bound.

It saves the outputs to `artifacts/research/coherent-kernel/ref/`. It is a 1-core CPU job.

## Gates (fixed before any GPU result)

1. Lookup table: |GPU − cogwheel| ≤ 2e-3 nats on 1e6 points that span production's (d|h, h|h)
   range, including out-of-range points.
2. Bestfit mode, per block and per bound setting:
   1. the kept (i, e, o) sets agree on at least 99.9% of rows. Disagreements must be rows whose
      best fit lies within 1e-3 of the block's cut;
   2. over common rows, |Δ bestfit_lnlike| ≤ 1e-3 and |Δ lnl_marginalized| ≤ 2e-3;
   3. relative error in `d_h_1Mpc` and `h_h_1Mpc` ≤ 1e-4;
   4. `n_dist_marg` within 0.01%;
   5. `logsumexp` of the kept rows' `ln_posterior` within 1e-3;
   6. not gated: the discarded-row sums (`logsumexp_disc`, `logsumsqrexp_disc`) are reported.
      The GPU computes them exactly over every discarded row, while the CPU estimates them from
      a random subsample, so their difference measures the CPU estimator's error. They enter only
      `ln_evidence_discarded`, a diagnostic.
3. Priority mode, per block:
   1. over pairs kept by the CPU, |GPU ln W − CPU `ln_phase_marginalized_weight`| ≤ 2e-3;
   2. the block's estimate of logsumexp(ln_posterior + ln_priority_weight) over kept rows, GPU and
      CPU, each within 3 standard errors of the exact block total. The exact total is the GPU
      logsumexp over all pairs, and the standard error is taken from the spread over 20 GPU seeds;
   3. the phase draws: over 300 pairs with at least 2 accepted phases and 20 seeds, the summed
      probability w_o / W of the chosen phase matches its expectation sum_o (w_o / W)^2, as a
      z-score with |z| < 3.29 (two-sided p > 0.001).
4. Speed: the mean GPU time per i-block in the production configuration for new runs
   (`phase_marginalized_priority` with the incoherent bound), from waveform arrays in host memory
   to the output DataFrame, × 1162 ≤ 1879 s on an L40S, measured in an exclusive GPU job.
   The shared correctness job reports the same timing as indicative only.

If gate 2 or 3 fails, the cause is found and fixed in a CPU-side comparison before another GPU
job. If gate 4 fails, the kernel is profiled, and that is logged in `progress.md`.

## Compute

- 1 CPU job (`physics-short`, 1 core, 24 GB, at most 60 min) for the reference.
- 1 GPU job (L40S, shared, `gmem` 16 GB, 4 cores, 32 GB host memory, `-W` 30) for gates 1 to 3.
  A second job is used only if the first fails on a code error.
- Gate 4 runs in one exclusive L40S job (`-W` ≤ 60), shared with the P3 timing gate.

## Revisions (before any GPU result)

2026-11-27, after the design review:
1. The question is restated as per-block compute (review finding 20). The 18793 s baseline also
   includes reads, the merge and cProfile overhead (`profile_enabled` in every production run),
   so the end-to-end comparison belongs to P5.
2. Gate 3.3 is a pooled z-score (finding 17). 20 seeds give too few draws per pair for a
   chi-square.
3. Gate 2.6 is reported, not gated (finding 24). The GPU now sums every discarded row exactly.
4. Gate 4 runs in an exclusive job and in the production configuration (findings 19, 20).
5. The bound configurations lower each template's incoherent log-likelihood by 25 nats
   (`common.BOUND_SHIFT`). The production bound (+20) never triggers on the test blocks
   (finding 25): their slack, the incoherent value + 20 minus the template's best coherent
   fit, has a median of 24 nats, with a 5th to 95th percentile range of 22 to 26.
6. Gate 1 compares the standalone kernel, now compiled with the same `--use_fast_math` as the
   fused kernels, against cogwheel at float64 inputs, as stated (finding 18).
7. Pre-job CPU emulation of the complex64 path (`cpu_precision_check.py`, block 0, first 256
   extrinsic samples, 237674 rows within 20 of the maximum best fit). The maximum |Δ bestfit| is
   2.6e-5 and the maximum |Δ lnl_marginalized| is 2.3e-5, against float64. Gates 2.2 and 2.3 are
   kept (finding 26).
