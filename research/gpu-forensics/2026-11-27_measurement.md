# Where production dot-PE spends its wall time, and why the earlier GPU attempt did not beat it

This measurement finds which stages of a production dot-PE run take the wall time, and so which
stages a GPU version must accelerate to halve it. It also explains why the earlier GPU attempt
gained no more than the 8-core CPU code, and it sizes the scatter between repeat production runs,
which the acceptance test needs.

The results were made with dot-PE branch `gpu-v2` (scripts only; no dot-PE code was run). The data
are the o4a-pe-paper and o4b-pe-paper production LSF logs (`artifacts/logs/dotpe_GW*/*.out`), the
rundirs `artifacts/pe/dotpe/<EVENT>/run_N/` (made with the dot-PE commits named in each
`timing.json`), and the branch `origin/dev/gpu` at 1847995. Supersedes: none.

## Question and answer

Where does the wall time of production runs go, and why did the earlier GPU attempt not win? The
coherent stage takes a median 74% of the wall time and the incoherent stage 23%, with the CPU-side
stages (setup, extrinsic proposal, postprocess) at about 2% together. The earlier attempt measured
itself against serial CPU at toy settings, where the coherent stage was small, and left disk reads,
the lookup table and the per-row pandas work on the CPU, so one GPU roughly tied 8 CPU cores.

## Prediction

The design (`design.md`) has no prediction. Its gate: a stage stays on the CPU only if the CPU
stages together take at most 50% of the 0.5× target for the median run. They take 217 s of a
4972 s target (4%), so setup, the extrinsic proposal and the cross-bank step stay on the CPU.

## Setup

Stage times are the "Stage timings" block that `mp_inference.run` prints at the end of each LSF
log. The six stages are:
1. setup;
2. incoherent selection per bank;
3. cross-bank threshold;
4. extrinsic proposal;
5. coherent inference per bank;
6. postprocess.

369 logs have the block. 332 of them are matched to a rundir: the matched rundir has the same event
as the log and a `timing.json` `wall_seconds` within 5 s of the log's total. The production runs
used 8 cores with 7 workers.

CPU-seconds by function come from `profile_output.prof`, which aggregates cProfile over the
workers. Cumulative times of nested functions overlap; the element-wise line is the tottime of
`run_thin_iblock` itself.

A repeat pair is two `status: ok` runs of one event with the same bank folders, n_int, n_ext and
n_phi. There are 16 such pairs (12 events). The Jensen–Shannon divergence is computed on 50 bins
spanning the 0.5 to 99.5 percentiles of the pair's union, in bits, with the `weights` column.

## Results

### Which stage takes the wall time?

Share of the wall time per stage, over 369 production runs:

| Stage | Median | 10th percentile | 90th percentile |
|---|---|---|---|
| coherent | 0.74 | 0.31 | 0.92 |
| incoherent | 0.23 | 0.07 | 0.61 |
| extrinsic | 0.01 | 0.00 | 0.04 |
| setup, cross-bank, postprocess | ≤ 0.01 each | | |

The total wall time has a median of 2.76 h, with a 10th to 90th percentile range of 0.78 to
8.88 h. The two common configurations are 3 banks with n_ext = 2048 (95 runs) and 6 banks with
n_ext = 2048 (81 runs). Their medians, in seconds:

| Configuration | Incoherent | Extrinsic | Coherent | Postprocess | Total |
|---|---|---|---|---|---|
| 3 banks, n_ext = 2048 | 1665 | 124 | 4672 | 35 | 6660 (1.85 h) |
| 6 banks, n_ext = 2048 | 3282 | 135 | 6936 | 70 | 10940 (3.04 h) |

The incoherent stage scales with the number of banks (about 550 s per bank of 2^20 templates).
The coherent stage scales with the number of templates kept and with n_ext. The full table per
configuration is printed by `stage_times.py`.

### What must the GPU stages achieve?

The GPU budget is the time left for the incoherent, coherent and postprocess stages once the
CPU-side stages (setup, cross-bank, extrinsic) are counted at their production time:

| Run | Production wall [s] | CPU-side [s] | GPU budget at 0.5× [s] | GPU budget at 0.1× [s] |
|---|---|---|---|---|
| median run (GW250109_010541, 6 banks) | 9944 | 217 | 4755 | 777 |
| GW240525_201644 run_2 (3 banks, n_ext = 13312) | 21022 | 191 | 10320 | 1911 |

The shortest runs set the harder bound. A 1-bank run with n_ext = 2048 takes a median of 720 s,
of which 109 s are CPU-side. The GPU must finish its incoherent stage (one 56 GB bank) and its
coherent stage in 250 s to reach 0.5×.

### What does the CPU spend its time on, inside each stage?

CPU-seconds by function, summed over the 7 workers (cumulative; nested lines overlap):

| | GW240525 run_2 (n_ext 13312) | GW240516 run_0 | GW240924 run_0 | GW241124 run_0 |
|---|---|---|---|---|
| coherent, total | 122,173 | 6,627 | 24,115 | 44,740 |
| ↳ <d\|h>, <h\|h> on the phase grid (matmul, einsum) | 57,392 | 1,025 | 3,321 | 6,233 |
| ↳ distance-marginalization lookup (scipy spline) | 13,231 | 2,476 | 9,773 | 18,126 |
| ↳ element-wise numpy in `run_thin_iblock` | 47,536 | 2,267 | 8,807 | 16,419 |
| incoherent, total | 13,402 | 11,076 | 11,266 | 16,005 |
| ↳ single-detector kernel | 8,950 | 6,606 | 7,676 | 11,917 |
| ↳ waveform reads | 3,135 | 3,689 | 2,767 | 3,225 |
| ↳ linear-free time fit (one `polyfit` per template) | 640 | 1,445 | 1,527 | 1,492 |

The last three columns are 3-bank runs with n_ext = 2048, with wall times of 0.92 h, 1.85 h and
2.89 h. At n_ext = 2048:
- the lookup table takes 37 to 41% of the coherent CPU time;
- the element-wise work (best fit, masks, threshold, gathers) takes 34 to 37%;
- the matmul takes 14 to 15%.

A GPU coherent stage that moves only the matmul, as the earlier attempt did, removes at most 15%
of it.

### How did the earlier GPU attempt compare?

`origin/dev/gpu` (2026-03-13 to 03-17) patched PyTorch kernels into `dot_pe.inference`. The kernels
covered:
- the single-detector lnlike;
- the per-mode <d|h> and <h|h> and the phase grid;
- batched distance sampling;
- vectorized prior transforms;
- some extrinsic matmuls.

Its benchmarks (`gpu/BENCHMARK.md`, one L40S, an injection with signal-to-noise ratio about 11,
n_ext = 512, n_phi = 32), in seconds:

| Bank | serial CPU | GPU | speed-up |
|---|---|---|---|
| 2 × 2^16 templates | 436 | 245 | 1.8× |
| 2^20 templates | 1760 | 644 | 2.7× |

The 8-worker `mp_inference` is 3.3× faster than serial on a 2^18-template bank (949 s against
3167 s, from the `lsf-swarm` benchmarks), so the GPU code was at par with the code it was meant to
replace. Its own profiles give the causes:
1. At n_ext = 512 and n_phi = 32 the coherent stage was 79 to 162 s, so the stage that holds 74%
   of production time was barely exercised.
2. The incoherent kernel took 0.08 s per batch of 2048 templates. Disk reads and per-template CPU
   preprocessing took about 1.5 s per batch, so the GPU was idle about 94% of the time
   (`gpu/PROGRESS.md`, Track G).
3. Every patched call copied numpy arrays to the device and back. The lookup table, the masks,
   the threshold, the DataFrames and the top-k stayed in numpy and pandas.
4. The <h|h> weights reach 3e48, above the float32 maximum of 3.4e38, so those contractions ran in
   complex128.

### How much do repeat production runs differ?

Over 16 repeat pairs (12 events), the median |Δ ln_evidence| is 0.56 and the median
Jensen–Shannon divergence of the 1-D marginals is between 0.011 and 0.018 bits (chirp mass, ln q,
chi_eff, luminosity distance, cos θ_JN). The pairs are not pure repeats. Several differ in dot-PE
version, in seed pool or in a bug fix between the runs:
- GW240630_212937 run_2 against run_7 differ by 9.2e4 in ln_evidence, from the incoherent-bound
  fix;
- GW241201_055758 and GW250109_010541 have n_effective_i ratios above 4000.

Seven pairs have |Δ ln_evidence| ≤ 0.5 and marginal divergences ≤ 0.03 bits; these are the
closest to seed-only repeats. Thresholds for the acceptance test are fixed in its own design from
this table (`artifacts/research/gpu-forensics/repeat_scatter.csv`).

## Observations after the experiment

- Postprocess is at most 1% of the wall time, unlike in the earlier GPU benchmarks. There,
  `draw_subset=False` had made it dominant.
- The extrinsic proposal takes a median of 124 s, about constant over configurations.

## Limitations

- Stage times come from one run per configuration cell for many cells. The cells with 1 to 3 runs
  are indicative only.
- Profile CPU-seconds include contention among 7 workers on one node. They rank the kinds of
  work, but do not give single-core rates.
- The repeat pairs mix code versions, so they bound run-to-run scatter from above, not from below.

## Reproduction

```
python research/gpu-forensics/stage_times.py      # stage table, Amdahl budget; writes stage_times.csv
python research/gpu-forensics/repeat_scatter.py   # repeat pairs; writes repeat_scatter.csv
python research/gpu-forensics/profile_split.py <rundir> [...]
git show origin/dev/gpu:gpu/BENCHMARK.md; git show origin/dev/gpu:gpu/PROGRESS.md
```

All of them run in the `dot-pe` env, in under 15 s each, on the login node.
