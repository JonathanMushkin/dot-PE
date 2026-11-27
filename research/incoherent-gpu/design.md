# Design: incoherent stage and bank residency on the GPU (P3, engineering phase)

## Question

Can one GPU run the incoherent stage of a 3-bank production run (3 × 2^20 templates) with the same
template selection as the CPU, in at most 15% of the 0.5× target wall time of the median run
(at most 746 s)? Can it also keep the selected templates' waveforms on the device, so that the
coherent stage does not read them again?

## Code

- `dot-pe-gpu/dot_pe_gpu/incoherent.py`, `IncoherentGPU`. Its semantics are in the module
  docstring.
- `dot-pe-gpu/dot_pe_gpu/bank.py`, a reader that streams bank blocks from disk into pinned host
  memory with a thread pool, overlapped with GPU work.

## Test data (read in place)

GW240525_201644 run_2: the event `.npz`, and `par_dic_0` and `fbin` from `Posterior.json`. Bank 0
(`bank_56`, 2^20 templates, 256 files of 4096 templates each for amplitude and the same for
phase, 56 GB in total). The reference is `banks/bank_0/intrinsic_samples.npz`, which holds the
production per-detector incoherent log-likelihoods (`lnlikes_di`) of the 158131 selected
templates. It was made with dot-PE 2403731, before the best-fit distance mask of the single-detector
stage (b0246ea), so templates touched by that mask may differ. The current CPU code on 512
templates is a second reference, run inside the GPU job.

## Gates (fixed before any GPU result)

1. Time shift: |dt_GPU − dt_CPU| ≤ 1e-7 s on 512 templates, where dt_CPU is from
   `get_relative_linfree_dt_from_waveform`.
2. Per-detector lnlike, GPU against the current CPU code on 512 templates: |Δ| ≤ 1e-3 nats
   wherever both are finite, and the same set of -inf entries.
3. Against production `lnlikes_di` on the 158131 selected templates of bank 0:
   1. the median |Δ| of the incoherent sum is ≤ 1e-3;
   2. the selection with the production threshold (max − 20) reproduces the production selection
      up to templates within 1e-3 of the threshold, or ones changed by the best-fit distance
      mask. Those templates are counted and listed.
4. Speed: the bank-0 incoherent stage (all 2^20 templates: read, dt, 3 detectors), timed on an
   L40S, × 3 ≤ 746 s. The read throughput from disk (GB/s) is reported separately. If reads
   dominate, the next step is a complex64 bank cache. That step needs its own dated design note,
   because it writes about 28 GB per bank to group storage.

## Compute

- 1 GPU job: L40S, shared, `gmem` 30 GB, 4 cores, 32 GB host memory, `-W` 30.
- Its memory peak is the pinned buffers (2 × 64 MB per file) plus the resident selected
  waveforms (at most 158131 × 3024 × 8 B = 3.8 GB).

## Revisions (before any GPU result)

2026-11-27, after the design review (findings 1 to 13):
1. Gate 4 is tied to the median run, which has 6 banks. The projected 6-bank incoherent time must
   be at most 746 s, so one bank must take at most 124 s (finding 7). It runs in an exclusive job,
   together with P2's gate 4 (finding 6). The shared correctness job reports it as indicative. Read
   time is reported apart (`seconds_waiting_for_reads`) from overall throughput.
2. Gate 3.2 now passes or fails (finding 2). The threshold is the maximum over the three banks'
   production `incoherent_lnlikes` (44.65502) minus `max_incoherent_lnlike_drop` (20), so 24.65502.
   A mismatched template passes if it lies within 1e-3 of the threshold, or if the current CPU
   code reproduces the GPU value. The second case attributes the difference to code changes
   after the production run (the best-fit distance mask).
3. Gates 1 and 2 also run on 512 random templates, 128 from each of 4 random files (finding 8).
   The CPU dry run found no masked (-inf) template among them.
4. New gate 5, residency (finding 4): `Residency.finalize` returns exactly the selected set. On
   256 random selected templates it is equal, element by element, to the disk waveforms
   amp·exp(i·phase) stored the same way, as complex64 of h·2^64. The peak device memory is reported.
5. Memory (finding 5): over all banks of run_2, the selected waveforms are 594510 × 3024 × 8 B =
   14.4 GB on the device as complex64, plus a transient set within the margin of a lower running
   maximum. P4 sizes `gmem` from gate 5's measured peak. `bank.py` uses plain `np.load`, not
   pinned memory, and the files are 99.1 MB each.
6. Code changes (findings 3, 9, 10, 13):
   1. `Residency` also keeps the global top `min_incoherent_survivors` templates, for stage 3's
      fallback.
   2. Waveforms are stored as complex64 of h·2^64, so products stay above float32's subnormal
      range, and are returned as complex128.
   3. The 2×2 determinant and inverse are computed in double.
   4. NaN log-likelihoods are stored as -inf; the CPU would propagate NaN.
   5. Preselected indices are sorted.
7. The text above said that `fbin` comes from `Posterior.json`. The code takes it from the bank's
   `bank_config.json`, as production does (finding 11).
8. CPU tests (finding 1), in `dot-pe-gpu/tests/test_cpu.py`; 30 pass:
   1. the closed-form time shift against `Polynomial.fit`;
   2. a numpy mirror of the kernel against `get_response_over_distance_and_lnlike` on random
      waveforms. It covers near-degenerate polarizations and checks the masks and values in
      float64 and float32;
   3. `Residency` margin, top-N fallback and `finalize`.
