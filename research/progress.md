# Progress log

One row per decision, design approval, gate outcome or LSF job. Times are local (IST).

| date | item | details |
|---|---|---|
| 2026-11-22 | branch | `research` created from `size-limit` 2df17cc; `protocol.md` adapted (hands-off approval, data and compute, gates and budget). |
| 2026-11-22 | budget | User: at most 6 full dot-PE reruns, at most 2 at once; small jobs at most 20 at once. |
| 2026-11-22 | E0 design | `row-limit/design.md` reviewed by a separate agent: REVISE (3 blocking: no pass criterion, rundir not an input, —; 6 minor), all fixed; re-review APPROVE. Minor suggestion adopted: noise ceiling 0.1 in the pass criterion. |
| 2026-11-22 | E1 design | `rows-above-cut/design.md` reviewed: REVISE (blocking: degenerate sample-size rule, unbounded memory), fixed (n_b = 40 per bank fixed in advance; one extrinsic block per worker call); re-review APPROVE. |
