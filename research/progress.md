# Progress log (GPU project)

Dates are in IST. GPU-minutes are the job's run time on its one GPU. The running total is in the
last column.

## Decisions

| Date | Decision | Reason |
|---|---|---|
| 2026-11-27 | Branch `gpu-v2` from `origin/size-limit` (dcf9ed9). | It contains `priority_sampling` and every bug fix production relies on. |
| 2026-11-27 | Code in `dot-pe-gpu/dot_pe_gpu/`; research in `research/`; env `dot-pe-gpu` (clone of `dot-pe` + `cupy-cuda12x`). | User request; the CuPy stack is the one trex_gpu runs on this cluster; the torch in our envs is a CPU build. |
| 2026-11-27 | trex_gpu read-only guard: global `PreToolUse` hook `~/.claude/hooks/trex_readonly.py` plus deny rules in `~/.claude/settings.json`. | User request. A loosening of the hook for heredoc text was refused by the permission classifier, so the strict version stays; text naming the folder is written with the Write/Edit tools. |
| 2026-11-27 | GPU usage limits and the 60 GPU-hour budget (protocol "GPU usage"). | trex_gpu has no written limits; `short-gpu` fairshare weights GPU time by 10. |
| 2026-11-27 | Engineering phases P2–P4 use one design per phase (protocol "Engineering phases"). | A design per microbenchmark would cost more than the benchmark. |
| 2026-11-27 | P1 gate passed. The CPU-side stages (setup, cross-bank, extrinsic) take 217 s of the 4972 s 0.5× target for the median run (4%), so they stay on the CPU. The incoherent stage (median 23% of the wall time, up to 61%) is in the GPU scope (P3). | `gpu-forensics/2026-11-27_measurement.md` |
| 2026-11-27 | P2 design written (`coherent-kernel/design.md`). Clarification before any GPU result: gate 3.3 (phase draws) is tested as a z-score of the summed probability of the chosen phase against its expectation sum_o p_o^2, pooled over 300 pairs × 20 seeds, \|z\| < 3.29 (two-sided p > 0.001). A per-pair chi-square needs more draws than 20 seeds give. | Same significance level as in the design. |

| 2026-11-27 | Deviations recorded (review finding 1). P1's design and report were committed together, with no separate approval row before its results. The P2 CPU reference (job 868940) ran before P2's design was approved. Neither changes a gate: P1 is a read-only measurement, and the reference is an input to P2, not a GPU result. | Review of 78df887 |
| 2026-11-27 | Design review of P1 and P2 (reviewer agent, 27 findings). Fixed before any GPU job: 1, 16 to 27 (P2 design revisions in `coherent-kernel/design.md` § Revisions; kernel fixes: double-precision Gumbel draw, finite-post guard, NaN `<d|h>` rejected, exact discarded sums, emit margin 1e-3 with an assert, buffer limit, n_phi < 511, uniform draws strictly inside (0, 1); 23 CPU tests in `dot-pe-gpu/tests/test_cpu.py` pass). P1 report findings 2 to 15 are open: the report is fixed before it is shown to the user. | Protocol "Review" |
| 2026-11-27 | P2 design approved (revised version, commit 3d30ec3). The bound configurations' CPU reference is rerun with `BOUND_SHIFT` = 25 (job 869105) before the GPU job. | Gates fixed in `coherent-kernel/design.md` |

| 2026-11-27 | P2 job 869233 result. Gates 1 and 2 pass. Gate 3 fails on 3.1 in 3 of 4 configurations: max \|Δ ln W\| over CPU-kept pairs is 0.0042, 0.032 and 0.25 (limit 2e-3; priority_bound block 0 passes at 2.0e-4). Gates 3.2 and 3.3 pass, with \|z\| ≤ 0.93. Indicative speed (shared job): 0.058 s per i-block, so 68 s projected for run_2's 1162 i-blocks (gate 1879 s). Kept as a null result on 3.1 (`artifacts/research/coherent-kernel/gpu_check_869233.json`). Hypothesis, to be tested before any gate change: a pair's ln W sums only its accepted phases, and the GPU (float32) and CPU (float64) can disagree on whether a row at the running threshold is accepted. A pair whose accepted rows all sit near the threshold then changes W by O(1), and such pairs lie about 20 nats below the block maximum. Test: in the next job, for each pair with \|Δ ln W\| > 2e-3, compute in float64 the smallest \|best fit − threshold\| over its phases, and the pair's ln W relative to the block's largest. | `coherent-kernel/design.md` gate 3.1 |

| 2026-11-27 | P3 design review (13 findings) addressed in `incoherent-gpu/design.md` § Revisions. P3 design approved (commit below). | Protocol "Review" |

## Jobs

| Job | Submitted | Queue / GPU | Purpose | Run time | Peak mem | GPU-min | Total GPU-h | Outcome |
|---|---|---|---|---|---|---|---|---|
| 868939 | 2026-11-27 08:41 | physics-short, 1 core | P2 CPU reference | 0 s | — | 0 | 0 | failed: relative script path (the job runs from /tmp); resubmitted with an absolute path |
| 868940 | 2026-11-27 08:41 | physics-short, 1 core | P2 CPU reference (`make_reference.py`) | 718 s | 5802 MB | 0 | 0 | done; 8 configurations × blocks, 80–85 s per block; bound configurations equal no-bound (bound never triggers), superseded by 869105 |
| 869105 | 2026-11-27 08:56 | physics-short, 1 core | P2 CPU reference, bound configurations with `BOUND_SHIFT` = 25 | 438 s | 5738 MB | 0 | 0 | done; the bound triggers (n_dist_marg 36450283 against 35683307 without it, block 0) |
| 869185 | 2026-11-27 09:04 | short-gpu, L40S shared, gmem 16 GB, 4 cores | P2 gates 1–3, indicative timing (`gpu_check.py --seeds 20`) | 0 s | — | 0 | 0 | failed on node lgn02 before start: "Cannot open your job file: /scratch/…" (LSF node error); resubmitted |
| 869233 | 2026-11-27 09:05 | short-gpu, L40S shared (lgn), gmem 16 GB, 4 cores | P2 gates 1–3, indicative timing | 88 s | 972 MB | 1.5 | 0.03 | gates 1, 2 pass; gate 3.1 fails (see decisions); speed indicative 0.058 s per i-block |
