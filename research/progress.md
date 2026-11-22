# Progress log

One row per decision, design approval, gate outcome or LSF job. Times are local (IST).

| date | item | details |
|---|---|---|
| 2026-11-22 | branch | `research` created from `size-limit` 2df17cc; `protocol.md` adapted (hands-off approval, data and compute, gates and budget). |
| 2026-11-22 | budget | User: at most 6 full dot-PE reruns, at most 2 at once; small jobs at most 20 at once. |
| 2026-11-22 | E0 design | `row-limit/design.md` reviewed by a separate agent: REVISE (2 blocking: no pass criterion, rundir not an input; 5 minor), all fixed; re-review APPROVE. Minor suggestion adopted: noise ceiling 0.1 in the pass criterion. |
| 2026-11-22 | E1 design | `rows-above-cut/design.md` reviewed: REVISE (blocking: degenerate sample-size rule, unbounded memory), fixed (n_b = 40 per bank fixed in advance; one extrinsic block per worker call); re-review APPROVE. |
| 2026-11-22 10:20 | LSF 384729 | E0 `test_mass_below_floor.py` (production), o4a-pe, 1 core, 8000 MB, physics-short. |
| 2026-11-22 10:20 | LSF 384732, 384735, 384738 | E1 pilot, sample positions 0, 1, 2 (bank_0 block 74, bank_1 block 98, bank_2 block 245), dot-pe, 1 core, 16000 MB, physics-short. |
| 2026-11-22 10:26 | LSF 384729 done | E0 production test: 337 s, peak 515 MB, exit 0. Output `artifacts/research/row-limit/mass_below_floor_production.csv`. |
| 2026-11-22 10:35 | finding | `summary_results.json` `bestfit_lnlike_max` is the maximum over the posterior draws (`inference.py:1568`), not over the kept rows. GW240525_201644 run_2: 35.287 in the summary, 36.379 over rows, so the nominal cut is 16.379, not 15.29. The handoff and o4b-pe-paper `row_limit.csv` use the summary value for every event (their "floor above the nominal cut" is overstated by the same amount). E1 design note added; E0 is unaffected (F is the floor, measured directly). |
| 2026-11-22 10:36 | E2 scope | User asked whether n_effective_e allows a smaller n_ext; added to E2 as a third direction, measured from the production summaries (n_ext, n_effective_e under both priors, against MIN_NEFF_E = 100). |
| 2026-11-22 10:45 | E2 design | `sample-headroom/design.md` reviewed: REVISE (1 blocking: comparison with E1 placed outside E2's chain; 5 minor), all fixed; approved. E1 design note reviewed: OK; stale 15.29 text in E1 and the best-fit definition in E0's glossary corrected. |
