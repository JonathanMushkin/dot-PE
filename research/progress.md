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
| 2026-11-27 | P2 design approved (revised version, commit below). The bound configurations' CPU reference is rerun with `BOUND_SHIFT` = 25 (job 869105) before the GPU job. | Gates fixed in `coherent-kernel/design.md` |

## Jobs

| Job | Submitted | Queue / GPU | Purpose | Run time | Peak mem | GPU-min | Total GPU-h | Outcome |
|---|---|---|---|---|---|---|---|---|
| 868939 | 2026-11-27 08:41 | physics-short, 1 core | P2 CPU reference | 0 s | — | 0 | 0 | failed: relative script path (the job runs from /tmp); resubmitted with an absolute path |
| 868940 | 2026-11-27 08:41 | physics-short, 1 core | P2 CPU reference (`make_reference.py`) | 718 s | 5802 MB | 0 | 0 | done; 8 configurations × blocks, 80–85 s per block; bound configurations equal no-bound (bound never triggers), superseded by 869105 |
| 869105 | 2026-11-27 08:56 | physics-short, 1 core | P2 CPU reference, bound configurations with `BOUND_SHIFT` = 25 | | | 0 | 0 | running |
