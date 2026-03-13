# GPU Acceleration Progress

## 2026-03-14

### Step 1 — gpu_info.py + gpu_constants.py
- `gpu/gpu_info.py`: queries PyTorch for GPU properties; auto-generates `gpu_constants.py`
- `gpu/gpu_constants.py`: tuning constants for NVIDIA L40S (44 GB VRAM, FP32 ~91.6 TFLOPS)
- Detected: NVIDIA L40S, 44.39 GB, 142 SMs, CUDA sm_8_9

### Step 2 — profile_run.py
- `gpu/profile_run.py`: CLI profiling harness
  - Generates Gaussian-noise injection (chirp_mass=20, HLV, O3 ASDs) when no `--event-path` given
  - Generates bank when no `--bank-path` given
  - `--gpu` flag switches to GPU-accelerated `gpu.run`
  - Prints top-N cumulative-time functions from `.prof`
- Usage:
  ```bash
  # First run (generates data):
  python gpu/profile_run.py --bank-size 65536

  # Repeated runs (reuse data):
  python gpu/profile_run.py --event-path <path> --bank-path <path>

  # GPU run (same data):
  python gpu/profile_run.py --gpu --event-path <path> --bank-path <path>
  ```

### Step 4 — GPU Acceleration Atoms

#### Track A: single_detector_gpu.py
- `get_response_over_distance_and_lnlike_gpu`: standalone GPU function
  - Inputs/outputs identical to CPU version (numpy → numpy)
  - All 5 heavy ops ported to torch (3 matmuls + 2 einsums) on CUDA
  - Uses `torch.complex64` throughout
- `GPUSingleDetectorProcessor(SingleDetectorProcessor)`: one-line dispatch shim

#### Track B: likelihood_calculating_gpu.py
- `get_dh_by_mode_gpu`: per-mode np.dot loop → single batched `torch.matmul`
- `get_hh_by_mode_gpu`: three `torch.einsum` calls (same subscripts as CPU)
- `get_dh_hh_phi_grid_gpu`: two `torch.einsum` phase-shift contractions
- `GPULikelihoodCalculator(LikelihoodCalculator)`: dispatch shim

#### gpu/run.py
- Monkey-patches `dot_pe.single_detector.SingleDetectorProcessor` and
  `dot_pe.likelihood_calculating.LikelihoodCalculator` before calling `inference.run()`
- `run(**kwargs)` and `run_and_profile(**kwargs)` match the `inference.*` signatures

### Correctness Tests (gpu/test_gpu_correctness.py)
All 4 unit tests passed with synthetic random inputs:
- `test_single_detector` — r_iotp and lnlike_iot vs CPU baseline
- `test_get_dh_by_mode` — dh_iem vs CPU baseline
- `test_get_hh_by_mode` — hh_iem vs CPU baseline
- `test_get_dh_hh_phi_grid` — dh_ieo, hh_ieo vs CPU baseline

Float32 relative precision: <0.1% (expected for complex64 vs float64 CPU baseline).

### Next Steps
- Run `gpu/profile_run.py` CPU baseline and record hotspot timings
- Run `gpu/profile_run.py --gpu` and compare
- Scale up to production bank size (2^20)
