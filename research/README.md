# Research on dot-PE's row limit

Branch `research` (from `size-limit`, which is dot-PE 4516a13 plus `bugfix-incoherent-bound-mask`
and `seed-pool-inds`). Nothing here is merged into `main`. Method: `protocol.md`; register:
`research-notebook.md`; dated log of decisions and LSF jobs: `progress.md`.

Background: dot-PE keeps at most `size_limit` = 1e7 (intrinsic, extrinsic, phase) rows, chosen by
highest `bestfit_lnlike`; at low SNR the lowest kept value lies well above the nominal cut (best
fit minus 20) and the posterior is truncated there. Source: o4b-pe-paper
https://github.com/JonathanMushkin/o4b-pe-paper/blob/master/dev/row_limit_check/HANDOFF_dotpe_row_limit.md

## Open decisions for the user

None yet.

## Questions

| id | folder | class | question | status |
|---|---|---|---|---|
| E0 | `row-limit/` | bug fix, problem | What fraction of cogwheel's posterior mass lies below dot-PE's floor? | design |
| E1 | `rows-above-cut/` | measurement | How many rows of GW240525_201644 run_2 lie above the nominal cut? | design |
| E2 | `rows-above-cut/` | measurement | Can a larger `size_limit` or a smaller bank alone reach the nominal cut? | planned, needs E1 |
| E3 | `weight-thinning/` | feature, correctness | Which thinning rule keeps the posterior unbiased at 1e7 rows? | planned, needs E1 |
| E4 | `phase-marginalization/` | measurement | How many fewer rows does marginalizing over phase leave? | planned, needs E1 |
| E5 | `row-limit/` | bug fix, solution | E0's test on the chosen fix | planned, needs E3 |
