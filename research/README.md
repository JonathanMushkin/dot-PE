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
| P2 | [coherent-kernel](coherent-kernel/) | feature (engineering) | closed: gates 1–4 pass (3.1 revised after diagnosis); 0.058 s per i-block on L40S |
| P3 | [incoherent-gpu](incoherent-gpu/) | feature (engineering) | closed: gates 1–5 pass; 59 s per cold 56 GB bank on L40S |
| P4 | [runner](runner/) | feature (engineering) | closed: gates 1–4 pass; GW240525 run_2 in 722 s fresh (production 21022 s) |
| P5 | acceptance | feature (acceptance) | not started |

## Open decisions for the user

None.

## Decisions by the user

1. 2026-11-27: the speed criterion applies to typical runs, and excludes runs whose production
   wall time is under 4 minutes. If too-short runs fail it again, the cutoff may be raised to
   10 minutes, and raised again in further steps. A cutoff that reaches 1 hour is reported to the
   user before it is used.
