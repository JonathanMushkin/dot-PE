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
| P1 | [gpu-forensics](gpu-forensics/) | measurement | report done, reviewed |
| P2 | [coherent-kernel](coherent-kernel/) | feature (engineering) | gates 1–3 pass (3.1 revised after diagnosis); gate 4 (exclusive timing) pending |
| P3 | [incoherent-gpu](incoherent-gpu/) | feature (engineering) | gates 1, 2, 3, 5 pass; gate 4 (exclusive timing) pending |
| P4 | [runner](runner/) | feature (engineering) | code written; CPU dry run of stage 1 passes |
| P5 | acceptance | feature (acceptance) | not started |

## Open decisions for the user

1. "Half the time for every event": runs whose production wall time is under about 4 minutes
   cannot reach 0.5×, because their CPU-side stages (setup, extrinsic proposal, postprocess)
   alone take about 2 minutes. Example: o4b GW240601_061200 run_0, 230 s in total. The acceptance
   set starts at 662 s. The acceptance report will give the fraction of production runs for which
   0.5× is reachable. Is a criterion over typical runs, excluding runs under about 4 minutes,
   acceptable?
