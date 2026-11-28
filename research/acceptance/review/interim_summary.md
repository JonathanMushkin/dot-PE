# P5 interim summary

Generated 2026-11-28 19:04 by `research/acceptance/interim_summary.py` from `compare_<EVENT>.json` (`evaluate_all.py`). Design: `research/acceptance/design.md`, revisions 2 and 3. Interim: no verdict until the corner plots are reviewed. Corner plots: [corners.md](corners.md).

Terms:
1. Reference run: the o4b CPU priority-sampling rerun of the event (7 workers on 8 cores).
2. Production run: the o4a/o4b CPU production run (8 cores).
3. GPU fresh run: the GPU run with nothing recycled, timed on an exclusive NVIDIA L40S with 8 cores.
4. GPU recycled run: the GPU run on the reference's intrinsic selection (and its extrinsic samples where the reference recycled them), seeds + 0, + 1, + 2.
5. Seed scatter: the largest |Δ ln Z| between pairs of the three GPU recycled runs.
6. JS: Jensen–Shannon divergence of a weighted 1-D marginal [bits]; the largest over mchirp, lnq, chieff, d_luminosity, costheta_jn is listed.

## 1. Speed (criterion: GPU fresh wall time ≤ 0.5 × the faster CPU run)

| # | event | reference [s] | production [s] | GPU fresh [s] | GPU / faster CPU | speed |
|---|---|---|---|---|---|---|
| 1 | GW240515_005301 | 10119 | 9947 | 609 | 0.061 | pass |
| 2 | GW240923_204006 | 3911 | 4942 | 430 | 0.110 | pass |
| 3 | GW231028_153006 | 2934 | 1824 | 125 | 0.068 | pass |
| 4 | GW240920_124024 | 542 | 521 | 81 | 0.156 | pass |
| 5 | GW241011_233834 | 3331 | 1329 | 423 | 0.318 | pass |
| 6 | GW240916_184352 | 10583 | 11422 | 763 | 0.072 | pass |
| R | GW240525_201644 | 23587 | 21022 | 511 | 0.024 | pass |

## 2. Results: GPU recycled run (seed + 0) against the reference (criteria of revision 2)

Criteria: |Δ ln Z| ≤ max(0.1, 2 × seed scatter); n_eff ratios within the seed range widened to at least [0.8, 1.25]; JS ≤ max(0.02 bits, 2 × seed JS, 2 × sampling floor).

| # | event | Δ ln Z | seed scatter | n_eff_i ratio | n_eff_e ratio | largest JS [bits] | JS / threshold, nearest [bits] | results |
|---|---|---|---|---|---|---|---|---|
| 1 | GW240515_005301 | 0.0298 | 2.7e-06 | 1.001 | 1.000 | 0.0020 (lnq) | 0.0020 / 0.0200 (lnq) | pass |
| 2 | GW240923_204006 | 0.0022 | 0.00011 | 1.000 | 0.999 | 0.0033 (costheta_jn) | 0.0033 / 0.0200 (costheta_jn) | pass |
| 3 | GW231028_153006 | 0.0808 | 3.6e-08 | 1.001 | 0.998 | 0.0023 (lnq) | 0.0023 / 0.0200 (lnq) | pass |
| 4 | GW240920_124024 | 0.0002 | 2.4e-07 | 1.000 | 1.000 | 0.0021 (mchirp) | 0.0021 / 0.0200 (mchirp) | pass |
| 5 | GW241011_233834 | 0.0095 | 0 | 0.998 | 0.994 | 0.1503 (d_luminosity) | 0.1420 / 0.2982 (chieff) | pass |
| 6 | GW240916_184352 | 0.0001 | 1.4e-06 | 1.000 | 1.000 | 0.0022 (mchirp) | 0.0022 / 0.0200 (mchirp) | pass |
| R | GW240525_201644 | 0.0118 | 7.4e-06 | 0.985 | 0.997 | 0.0034 (chieff) | 0.0034 / 0.0200 (chieff) | pass |

## 3. GPU fresh run against the reference and against production (reported, no criterion)

Production uses another row-selection rule (bestfit), so the second comparison mixes the GPU difference with the method difference.

| # | event | Δ ln Z vs ref. | n_eff_i ratio vs ref. | n_eff_e ratio vs ref. | largest JS vs ref. | Δ ln Z vs prod. | n_eff_e ratio vs prod. | largest JS vs prod. |
|---|---|---|---|---|---|---|---|---|
| 1 | GW240515_005301 | 0.257 | 1.02 | 0.70 | 0.0033 (costheta_jn) | 0.099 | 0.90 | 0.0068 (mchirp) |
| 2 | GW240923_204006 | 0.055 | 0.96 | 0.94 | 0.0032 (d_luminosity) | 0.459 | 1.34 | 0.0081 (chieff) |
| 3 | GW231028_153006 | 0.467 | 0.98 | 0.92 | 0.0020 (chieff) | 0.459 | 0.93 | 0.0025 (d_luminosity) |
| 4 | GW240920_124024 | 0.072 | 0.92 | 1.13 | 0.0034 (costheta_jn) | 0.071 | 1.13 | 0.0034 (costheta_jn) |
| 5 | GW241011_233834 | 0.555 | 0.93 | 0.81 | 0.2033 (d_luminosity) | 0.555 | 0.81 | 0.2128 (chieff) |
| 6 | GW240916_184352 | 0.605 | 0.86 | 1.35 | 0.0104 (costheta_jn) | 0.689 | 1.58 | 0.0126 (costheta_jn) |
| R | GW240525_201644 | 0.012 | 0.99 | 1.00 | 0.0029 (chieff) | 0.161 | 1.30 | 0.0078 (mchirp) |

## 4. GPU production-recycled run against production (revision 3, regime 1)

Both runs use production's rule (bestfit), intrinsic selection, extrinsic samples, event file and seed, so they should agree up to numerical precision. Criteria: |Δ ln Z| ≤ 1e-3; n_eff ratios in [0.99, 1.01]; JS ≤ max(0.02 bits, 2 × sampling floor). Δ max ln L: GPU minus CPU, the largest best-fit (over phase and distance) and the largest distance-marginalized log-likelihood ratio over all (intrinsic, extrinsic) pairs.

| # | event | Δ ln Z | n_eff_i ratio | n_eff_e ratio | largest JS [bits] | Δ max best-fit ln L | Δ max marginalized ln L | results |
|---|---|---|---|---|---|---|---|---|
| 1 | GW240515_005301 | 0.02947 | 1.0006 | 1.0001 | 0.0020 (mchirp) | -0.1943 | -0.0115 | FAIL: d_ln_evidence |
| 2 | GW240923_204006 | 0.00176 | 1.0002 | 0.9986 | 0.0022 (lnq) | -0.0872 | -0.0703 | FAIL: d_ln_evidence |
| 3 | GW231028_153006 | 0.08079 | 1.0005 | 0.9977 | 0.0021 (d_luminosity) | +0.0645 | +0.0469 | FAIL: d_ln_evidence |
| 4 | GW240920_124024 | 0.00017 | 1.0000 | 1.0000 | 0.0020 (d_luminosity) | -0.0000 | -0.0001 | pass |
| 5 | GW241011_233834 | 0.00948 | 0.9979 | 0.9942 | 0.1488 (chieff) | +0.0095 | +0.0095 | FAIL: d_ln_evidence, js_chieff |
| 6 | GW240916_184352 | 0.00007 | 1.0000 | 0.9998 | 0.0024 (lnq) | -0.3677 | +0.0587 | pass |
| R | GW240525_201644 | 0.00004 | 0.9999 | 1.0000 | 0.0023 (mchirp) | -1.0826 | -0.0001 | pass |

## 5. Points to look at (written 2026-11-28 from the tables above)

1. Recycled Δ ln Z is far above the seed scatter on events 3 (0.081 against 4e-8), 1 (0.030 against 3e-6) and R (0.012 against 7e-6). It passes because of the 0.1 floor, but it is a systematic GPU–CPU difference, not sampling noise. Its cause is not known.
2. Event 5 has a low effective sample size (production n_eff_e 30). Its JS divergences (0.14 to 0.21 bits) are large in every comparison; the threshold there is set by the sampling floor (2 × floor = 0.30 bits on chieff). The corner plot is the better judge.
3. Fresh runs draw their own intrinsic selection and extrinsic samples. Against the reference, |Δ ln Z| reaches 0.61 (event 6) and n_eff_e ratios range from 0.70 to 1.35. There is no CPU repeat-run scatter in this table to set them against.
4. Section 4 fails on 4 of 7 events, and its Δ ln Z equals the recycled Δ ln Z of section 2 (event 3: 0.081 in both). Production and the reference agree with each other; the GPU differs from both by the same amount. On identical inputs the largest marginalized ln L differs by up to 0.070 (event 2) and the largest best-fit ln L by up to 1.08 (event R, whose ln Z nevertheless agrees to 4e-5). This points to the GPU coherent stage, precision or the best-fit definition (not yet examined). P2's coherent-kernel tolerance (relative error of the marginalized ln L below 1e-3, i.e. up to 0.24 at ln L 240) was looser than the 1e-3 absolute ln Z criterion here.

## 6. Not yet available

1. 7 GW231002_143916: finished: GW231002_143916_rec_s0, GW231002_143916_rec_s1, GW231002_143916_rec_s2, GW231002_143916_prod
