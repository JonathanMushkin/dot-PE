# Incoherent kernel: float32 loss on nearly degenerate polarizations

Engineering note (bug fix), found in P5 revision 2. dot-PE branch `gpu-v2`; the fix is in
`dot-pe-gpu/dot_pe_gpu/incoherent.py`.

## Finding

The first fresh GPU run of P5 revision 2, GW240923_204006 (3 banks), selected 100 templates,
against production's 275478. Its global maximum incoherent lnlike was 157.38, from template
692874 of bank 2 (very_low_q_bank_57: m1 = 942, m2 = 53, iota = 1.64). Production's maximum in
that bank was 77.36.

| Template 692874 | H | L | V | Incoherent sum |
|---|---|---|---|---|
| CPU (`run_for_single_detector`, current code) | 2.914 | 14.561 | 1.937 | 19.41 |
| GPU, float32 kernel | 34.934 | 117.699 | 4.745 | 157.38 |
| numpy replay, float32 as on the device | 43.40 | 126.34 | 15.22 | |
| numpy replay, float64 | 2.910 | 14.559 | 1.937 | |

The linear-free time shift is the same on CPU and GPU (0.1225 s). The replay in float64 agrees
with the CPU, and the replay in float32 does not, so the loss is in the arithmetic precision.

## Cause

Per (template, phase, time), the kernel solves a 2x2 system in the two polarizations,
r = <h|h>^-1 <d|h>, lnl = r . <d|h> / 2. When the two polarizations of a template are nearly
proportional, <h|h> is nearly singular, and lnl depends on the difference between the two
polarizations' <d|h> and <h|h> entries. float32 keeps about 7 digits. On a synthetic test with
the cross polarization equal to the plus one times (1 + ε e^{iφ}), at lnlike 37 to 175:

| ε | float32 <d|h>, float32 <h|h> (old kernel) | float32 <d|h>, float64 <h|h> | float64 (new kernel) |
|---|---|---|---|
| 1e-2 | 0.16 | 0.004 | 4e-10 |
| 1e-3 | 55 | 0.057 | 4e-8 |
| 1e-4 | 1.7e6 | 0.41 | 4e-6 |
| 1e-5 | 2.6e7 | 4.0 | 5e-4 |

(maximum absolute lnlike error over 24 templates, against `SingleDetectorProcessor` in float64; float64 keeps the error below 1e-3 down to ε = 1e-5.) P3's gates passed because the templates they
drew had no such case.

## Fix

The single-detector stage computes in float64: the <d|h> matmul in complex128, <h|h>, the
inverse and the quadratic form in double. The coherent stage is not affected: it has no
polarization solve.

## Checks

1. CPU tests (`dot-pe-gpu/tests/test_cpu.py`): 32 pass.
2. `research/acceptance/precision_gpu_check.py` (job 888161, H100 NVL): 2002 templates of
   GW240923 bank 2, including 692873 to 692875, against the CPU. The same templates are finite
   on both, and the largest |Δ lnlike| is 9.4e-7 (H), 1.9e-6 (L) and 4.7e-7 (V). Template
   692874 now scores 2.913994, 14.560585 and 1.936890.
3. Cost: 0.0123 s per 2048 templates on H100 NVL. On L40S (P5 r2 event 2 timed run, job 888166,
   3 banks of 2^20 templates), stage 2 took 297.6 s against 192.6 s with the float32 kernel
   (job 887856): +105 s, or 35 s per bank, including disk reads. The event's total was 429.6 s
   against its target of 1955 s, so the float32 path with a float64 fallback is not needed.
4. End to end (job 888166): 207559, 67229 and 732 templates passed the global threshold (67.61)
   per bank, against production's 207533, 67213 and 732. The global maximum is 87.61.

## Effect on earlier results

1. P3 and P4 matched production's selection to 1 template per bank, so no outlier template
   occurred there.
2. P5 revision 1's event-4 timed run (64 s) used the float32 kernel. Its time stands as a
   revision-1 record only.
3. The recycled P5 runs do not run the incoherent stage and are unaffected.
