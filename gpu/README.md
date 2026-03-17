# gpu -- GPU-accelerated parameter estimation

Drop-in replacement for `dot_pe.inference` that routes the compute-heavy
stages to GPU (NVIDIA CUDA). All other logic remains in `dot_pe`.

## Usage

```python
from gpu import inference          # instead of: from dot_pe import inference
inference.run(...)                 # identical keyword arguments
inference.run_and_profile(...)     # identical keyword arguments
```

## What runs on GPU

| Stage                          | Speedup (typical)        |
|--------------------------------|--------------------------|
| Incoherent template selection  | 3-6x vs CPU              |
| Distance sampling              | <1 s vs minutes (CPU)    |
| Coherent scoring               | 1-1.4x                   |
| Posterior standardization      | ~1x                      |
| **End-to-end (1 M bank)**      | **~2.7x**                |

Speedup is bank-size dependent: the incoherent stage dominates at large
banks and scales well on GPU. The extrinsic sampling stage (`draw_extrinsic`)
has a fixed setup cost (~75 s for QMC temperature optimisation) that is
CPU-only regardless of bank size.

## Benchmark results (L40S, O3-range event)

Event: chirp mass 20 M_sun, d = 1366 Mpc, SNR ~ 11 (realistic O3 range),
3-detector network (H1, L1, V1).
Waveform shape per template: (4 modes, 2 polarisations, 378 frequency bins).
Run settings: `n_ext=512`, `n_phi=32`, `n_t=64`, `draw_subset=True`,
`blocksize=2048`.

| Bank size          | CPU total | GPU total | Speedup |
|--------------------|-----------|-----------|---------|
| 128 k (2 x 64 k)   | ~436 s    | ~245 s    | 1.8x    |
| 1 M                | ~1760 s   | ~644 s    | 2.7x    |

## Bank preload

For repeated runs on the same bank, preload all waveforms to VRAM once
to eliminate disk I/O per batch:

```python
from gpu import inference
inference._PRELOAD_ENABLED = True
inference.run(...)
```

VRAM requirement: approximately 1.5 GB per 1 M waveforms (complex64).

## Package layout

| File                           | Contents                              |
|--------------------------------|---------------------------------------|
| `inference.py`                 | Entry point: `run()` and `run_and_profile()` |
| `single_detector_gpu.py`       | GPU port of single-detector response  |
| `likelihood_calculating_gpu.py`| GPU port of coherent likelihood       |
| `distance_sampling_gpu.py`     | Batched GPU distance sampling         |
| `extrinsic_gpu.py`             | GPU matmuls for extrinsic sampling    |
| `gpu_constants.py`             | Device and dtype configuration        |
