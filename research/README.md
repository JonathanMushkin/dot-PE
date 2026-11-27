# GPU dot-PE: research index

Branch `gpu-v2`, from `size-limit` (dcf9ed9). The code is in `dot-pe-gpu/dot_pe_gpu/`. The rules
are in [protocol.md](protocol.md) and the dated log is in [progress.md](progress.md).

## Goal and success criterion

A GPU run of dot-PE that writes `summary_results.json` and `samples.feather` has two targets:
1. wall time: at most half the production CPU run's `timing.json` `wall_seconds` (8 cores,
   `mp_inference`), on 1 GPU and at most 8 cores, for every acceptance event;
2. results: similar to the o4a/o4b production runs, with the thresholds fixed in
   `acceptance/design.md` before the acceptance runs.

## Phases

| # | Slug | Class | Status |
|---|---|---|---|
| P0 | (setup) | — | done 2026-11-27 |
| P1 | [gpu-forensics](gpu-forensics/) | measurement | open |
| P2 | coherent-kernel | feature (engineering) | not started |
| P3 | incoherent-gpu | feature (engineering) | not started |
| P4 | runner | feature (engineering) | not started |
| P5 | acceptance | feature (acceptance) | not started |

## Open decisions for the user

None.
