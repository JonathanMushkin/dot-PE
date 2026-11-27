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
