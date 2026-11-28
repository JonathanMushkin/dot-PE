# P5 corner plots

Design: `research/acceptance/design.md` (revisions 2 and 3); script: `research/acceptance/corner_plots.py`. Parameters and styling: the o4b event gallery.

1. Against the reference (o4b CPU priority-sampling rerun): reference (C0), GPU run on the reference's recycled inputs, seed + 0 (C1), GPU fresh run (C2).
2. Against production (production's method, bestfit): production (C0), GPU run on production's recycled inputs (C1).

| # | event | CPU priority rerun [s] | CPU production [s] | GPU fresh [s] | GPU / faster CPU | GPU |
|---|---|---|---|---|---|---|
| 1 | GW240515_005301 | 10119 | 9947 | 609 | 0.061 | 1x NVIDIA L40S + 8 CPU cores, lgn20 (AMD_EPYC) |
| 2 | GW240923_204006 | 3911 | 4942 | 430 | 0.110 | 1x NVIDIA L40S + 8 CPU cores, lgn20 (AMD_EPYC) |
| 3 | GW231028_153006 | 2934 | 1824 | 125 | 0.068 | 1x NVIDIA L40S + 8 CPU cores, lgn20 (AMD_EPYC) |
| 4 | GW240920_124024 | 542 | 521 | 81 | 0.156 | 1x NVIDIA L40S + 8 CPU cores, lgn20 (AMD_EPYC) |
| 5 | GW241011_233834 | 3331 | 1329 | 423 | 0.318 | 1x NVIDIA L40S + 8 CPU cores, lgn20 (AMD_EPYC) |
| 6 | GW240916_184352 | 10583 | 11422 | 763 | 0.072 | 1x NVIDIA L40S + 8 CPU cores, lgn24 (AMD_EPYC) |
| 7 | GW231002_143916 | 21545 | 26598 | — | — | — |
| R | GW240525_201644 | 23587 | 21022 | 511 | 0.024 | 1x NVIDIA L40S + 8 CPU cores, lgn27 (AMD_EPYC) |

## 1. GW240515_005301

Against the priority-sampling reference:

![GW240515_005301 vs_reference](corners/GW240515_005301_vs_reference.png)

Against production:

![GW240515_005301 vs_production](corners/GW240515_005301_vs_production.png)


## 2. GW240923_204006

Against the priority-sampling reference:

![GW240923_204006 vs_reference](corners/GW240923_204006_vs_reference.png)

Against production:

![GW240923_204006 vs_production](corners/GW240923_204006_vs_production.png)


## 3. GW231028_153006

Against the priority-sampling reference:

![GW231028_153006 vs_reference](corners/GW231028_153006_vs_reference.png)

Against production:

![GW231028_153006 vs_production](corners/GW231028_153006_vs_production.png)


## 4. GW240920_124024

Against the priority-sampling reference:

![GW240920_124024 vs_reference](corners/GW240920_124024_vs_reference.png)

Against production:

![GW240920_124024 vs_production](corners/GW240920_124024_vs_production.png)


## 5. GW241011_233834

Against the priority-sampling reference:

![GW241011_233834 vs_reference](corners/GW241011_233834_vs_reference.png)

Against production:

![GW241011_233834 vs_production](corners/GW241011_233834_vs_production.png)


## 6. GW240916_184352

Against the priority-sampling reference:

![GW240916_184352 vs_reference](corners/GW240916_184352_vs_reference.png)

Against production:

![GW240916_184352 vs_production](corners/GW240916_184352_vs_production.png)


## 7. GW231002_143916

Against the priority-sampling reference:

![GW231002_143916 vs_reference](corners/GW231002_143916_vs_reference.png)

Against production:

![GW231002_143916 vs_production](corners/GW231002_143916_vs_production.png)


## R. GW240525_201644

Against the priority-sampling reference:

![GW240525_201644 vs_reference](corners/GW240525_201644_vs_reference.png)

Against production:

![GW240525_201644 vs_production](corners/GW240525_201644_vs_production.png)

