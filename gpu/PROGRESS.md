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

---

## 2026-03-14 (continued) — Precision fixes + timing benchmark

### Precision fix: hh_weights overflow float32
- Diagnosed: `hh_weights_dmppb` arrives as complex128 with max ~3e48 (exceeds float32 max 3.4e38)
- Fix in `single_detector_gpu.py`: upload hh_weights as complex128, compute
  the hh einsum in complex128, cast result (~1e10) back to complex64
- Same fix in `likelihood_calculating_gpu.py::get_hh_by_mode_gpu`
- All 5 unit tests still pass after fix

### End-to-end correctness verification
Compared `prob_samples` (sorted by `i, e, o`) between CPU run_1 and GPU run_6
(4096-sample bank, same seed=42):

| Column | max abs diff | mean abs diff | max rel diff |
|--------|-------------|---------------|-------------|
| `lnl_marginalized` | 5.4e-4 | 5.4e-5 | 0.7% |
| `bestfit_lnlike` | 5.5e-4 | 2.7e-5 | 0.15% |
| `d_h_1Mpc` | 0.39 | 0.047 | 0.29% |
| `h_h_1Mpc` | 361 | 37 | 6e-7 |

Max relative error 0.7% (lnl) is within float32 tolerance.
Same 3 intrinsic indices selected: [495, 975, 1200] ✓

### Timing benchmark: 32k-sample bank (2^15, blocksize=2048)
Bank: `test_bank_32k`, event: `profile_event.npz`, n_ext=512, n_phi=32, n_t=64

| Step | CPU cumtime | GPU cumtime | Speedup |
|------|-------------|-------------|---------|
| `get_response_over_distance_and_lnlike` (48 calls) | 30.1s | 2.74s | **11×** |
| `select_intrinsic_samples_per_bank_incoherently` | 47.9s | 18.5s | **2.6×** |
| `get_dh_by_mode` (1 call, tiny block) | 20ms | 15ms | ~1.3× |
| `get_hh_by_mode` (1 call, tiny block) | 7ms | 13ms | overhead |
| **Total inference.run()** | 254s | 225s | 1.13× |

The remaining 15.7s in GPU incoherent is waveform I/O + numpy overhead (not in our kernel).
Overall speedup limited by `postprocess` (148s, CPU-only) and `draw_extrinsic_samples` (50s).

---

## 2026-03-15 — GPU distance sampling (Track C)

### distance_sampling_gpu.py
- `sample_distance_batched_gpu(d_h_arr, h_h_arr, lookup_table, resolution=32) → (N,) float64`
- Replaces `np.vectorize(lookup_table.sample_distance)` (~409k serial scipy-spline calls at 32k scale, 88s)
- Algorithm:
  - Per-sample focused grid in u-space (±10σ around likelihood peak) + shared broad grid
  - Combined (N, 2K) distance grid per sample, sorted
  - GPU posterior: `prior * inv_vol * exp(-0.5*(norm_h*u - overlap)^2)`
  - CDF via `torch.cumsum` of trapz weights
  - CDF inversion: `torch.searchsorted` + linear interpolation
  - Returns float64 numpy array

### gpu/run.py — distance sampling integrated
- `_standardize_samples_gpu`: full replacement for `dot_pe.inference.standardize_samples`
  - Mirrors all DataFrame manipulation from the original
  - Replaces lines 546-552 (the np.vectorize loop) with a single `sample_distance_batched_gpu` call
  - Correct ordering: GPU d_luminosity set **before** `pr.inverse_transform_samples(combined_samples)`
- `_patch()` now also sets `dot_pe.inference.standardize_samples = _standardize_samples_gpu`

### Correctness tests (gpu/test_gpu_correctness.py) — now 6 tests, all pass
- `test_sample_distance_batched` (new): N=2000 samples, compares mean/std vs CPU within 10% statistical tolerance
  - CPU mean=1261.8 Mpc, GPU mean=1243.0 Mpc ✓
  - CPU std=3677.8 Mpc, GPU std=3693.1 Mpc ✓

### Bug fix: SingleDetectorProcessor patch not reaching inference.py
`inference.py` binds `SingleDetectorProcessor` via `from .single_detector import ...` (direct local binding).
Module-level patch `_sd.SingleDetectorProcessor = GPU...` only updates the source module, not the
already-bound local name. Fix: also set `_inf.SingleDetectorProcessor = GPUSingleDetectorProcessor`
in `_patch()`. Without this fix, the incoherent kernel was silently running CPU (27.5s ≈ CPU baseline
vs expected 2.74s GPU).

### End-to-end benchmark: run_10 (GPU, all fixes applied)
**Event**: `profile_event.npz`  **Bank**: `test_bank_32k` (32,768 samples)
**Params**: n_ext=512, n_phi=32, n_t=64  **Selected**: 25 intrinsic samples

| Step | CPU | GPU run_12 | Speedup |
|------|-----|------------|---------|
| `select_intrinsic_samples_per_bank_incoherently` | 47.9s | 18.6s | **2.6×** |
| Coherent blocks (1 block, 25×512) | ~50s est. | 2.18s | **>23×** |
| `standardize_samples` (dist + prior, wall) | ~146s | ~12s wall / 5.4s CPU | **>12×** |
| `draw_extrinsic_samples` | ~51s | 51.0s | 1.0× 🔒 (CPU-bound QMC) |
| **Total `inference.run()`** | **254s** | **~111s** | **~2.3×** |

### Current bottleneck profile (GPU run_12)
1. `draw_extrinsic_samples` — **51.0s** (QMC sampling + marginalization, CPU-bound, 🔒 cogwheel internals)
   - Inside: `marginalized_extrinsic.__init__` (41.8s), scipy sparse CSR matvec (6.3s), `optimize_beta_temperature` (15.3s)
2. `select_intrinsic_samples_per_bank_incoherently` — **18.6s** (GPU, 48 batches × 0.39s, mainly waveform I/O)
3. `_standardize_samples_gpu` — **5.4s** cProfile / ~12s wall (done; LAL fallbacks release GIL → CPU undercount)
   - `_pr_inverse_transform_batch` self+callees: 5.35s cProfile
   - Spin precessing fallback: 132k calls at 3.8s (LAL scalar-only, unavoidable)
   - phi_ref fallback: 409k calls at 0.98s (LAL scalar-only)

### Chunking fix for distance_sampling_gpu.py
Original implementation created (N, 2K) float64 arrays on CPU before going to GPU:
- (10M, 64) array = 5 GB of RAM → numpy sort on 10M rows used 128 cores
- Fix: chunk into groups of 256k rows, build grids entirely on GPU with `torch.sort`
- Timing for N=10M: **0.78 seconds** (float64 GPU) vs estimated ~35 minutes (CPU vectorize)

### Correctness fixes for distance_sampling_gpu.py (two critical bugs)
**Bug 1 — `searchsorted` off-by-one**: `torch.searchsorted` returns the INSERT position `i`
such that `cdf[i-1] < u ≤ cdf[i]`. The lower-bound bin for linear interpolation is `i-1`, NOT `i`.
Without the fix, GPU inverted the CDF one bin too high, causing ~8% systematic error in the tail.

**Bug 2 — Grid construction mismatch**: CPU uses `linspace(u_hi, u_lo, K)` with full range (u_lo
may be negative for low-SNR samples), then filters `u > 0`. GPU was clipping to `u_lo = 1e-12`,
producing a completely different u-spacing. Fix: clip to `u_min = REF/D_MAX` (equivalent physical
clamp), so the u-spacing matches CPU exactly for all SNRs.

**Precision upgrade**: CDF computation upgraded from float32 to float64 throughout. Float32 cumsum
errors in the tail (u_frac > 0.9) caused ~7% error; float64 reduces this to machine precision.

**Final accuracy**: max relative error = 1.23e-15 (machine precision), median = 0.00e+00, N=500, K=32.
Timing unchanged: **0.78s for 10M samples** (memory-bandwidth dominated).

### Track D — `prior.inverse_transform_samples` batch vectorization (2026-03-15)

**Implemented**: `_pr_inverse_transform_batch(pr, samples)` in `gpu/run.py`.
`_standardize_samples_gpu` now calls this instead of `pr.inverse_transform_samples`.

**Strategy**: iterate `pr.subpriors` in dependency order.  For each subprior:
- **Batch-safe path**: call `subprior.inverse_transform(**full_arrays)` once for all N samples
  using numpy vectorization.  Verified batch-safe: `IsotropicInclinationPrior`,
  `IsotropicSkyLocationPrior`, `IsotropicSpinsAlignedComponentsPrior`,
  `UniformDetectorFrameMassesPrior`, `UniformEffectiveSpinPrior` (pure numpy / scipy).
- **Per-sample fallback**: if the call raises (e.g. `lal.TimeDelayFromEarthCenter` TypeError),
  fall back to a scalar loop for that subprior only.
  Verified NOT batch-safe: `UniformTimePrior`, `UniformPhasePrior`, `SomeDistancePrior`
  (all call LAL C-functions requiring scalar inputs).

**Test added**: `test_prior_inverse_transform_batch` in `gpu/test_gpu_correctness.py` (7th test).
- Exercises both the fast batch path (inc + sky + aligned-spin) and fallback (time, LAL).
- Asserts machine-precision agreement (rtol=1e-12) vs per-sample baseline.
- All 7 tests PASS.

**Actual speedup (run_11 → run_12):**
- `_pr_inverse_transform_batch` cProfile: 8.07s (run_11) → 5.35s (run_12 + FixedPrior fix)
- `standardize_samples` wall: 34.2s → 31.5s → vs CPU ~146s → **>4.6× speedup**
- `inference.run()` total: 144.9s → ~111s → vs CPU 254s → **~2.3× overall speedup**

**FixedPrior fix** (run_12): `FixedPrior.inverse_transform` raises ValueError with batch arrays
(array truth-value ambiguity in its equality check).  Skipping it entirely saves 409k Python loop
iterations (~1s cProfile, ~2.7s wall).  Added `isinstance(subprior, FixedPrior): continue` guard.

**Remaining bottleneck within batch** (irreducible without LAL vectorization):
- Spin precessing (`spin.py:519`): 132k scalar LAL calls, ~3.8s cProfile / ~12s wall
- phi_ref (`extrinsic.py:115`): per-sample fallback, ~1s cProfile

---

## 2026-03-16 — Track E + Track F: batch spin & extrinsic subpriors

### Track E — `CartesianUniformDiskInplaneSpinsIsotropicInclinationPrior` batch (2026-03-16)

**Bottleneck identified**: `spin.py:519` (CartesianUniform…) was hitting per-sample fallback:
132k × `SimInspiralTransformPrecessingWvf2PE` calls = 3.8s cProfile / ~13s wall.

**Key insight**: Since |L| >> |S| by ~1e9 for all GW sources (f_ref ≥ 20 Hz), the LAL function
reduces to pure Cartesian-to-polar spin conversions + theta_jn = iota (to 1e-9 error).

**Implemented**: `_spin_wvf2pe_batch` in `gpu/run.py` — numpy replacement for `SimInspiralTransformPrecessingWvf2PE`.
**Handles**: `_BaseInplaneSpinsInclinationPrior`, `CartesianUniform…`, `_BaseSkyLocationPrior`.
**Accuracy**: theta_jn error < 4e-10 rad (|S|/|L|); tilt1/tilt2/phi12/chi: machine precision.
**Verified**: `test_precessing_spin_inverse_transform_batch` (N=300, 3 subprior types) — PASS.

### Track F — Batch extrinsic LAL subpriors (2026-03-16)

**Identified 4 remaining fallback subpriors** (each 409,600 per-sample calls):
1. `UniformPolarizationPrior` — LRU cache fails with arrays (identity transform)
2. `UniformTimePrior` — `lal.TimeDelayFromEarthCenter` (scalar only)
3. `UniformPhasePrior` — `TimeDelayFromEarthCenter` + `ComputeDetAMResponse` + LRU cache
4. `UniformLuminosityVolumePrior` — `ComputeDetAMResponse` (partially LRU-cached)

**Python overhead dominates**: 10µs/call × 1,638,400 calls = 16.4s (not LAL math itself).

**Formulas derived and verified**:
- `TimeDelayFromEarthCenter(det_loc, ra, dec, tgps)` → `-dot(det_loc, n_ecef)/c`
  where `n_ecef = (cos(dec)*cos(gmst-ra), -cos(dec)*sin(gmst-ra), sin(dec))` (matches LAL to < 4e-18 s)
- `F+(ra,dec,psi,D)` = m^T D m - n^T D n (matches LAL to < 1e-14)
- `Fx(ra,dec,psi,D)` = -(m^T D n + n^T D m)
  where m = cos(psi)*e_East - sin(psi)*e_North, n = sin(psi)*e_East + cos(psi)*e_North,
  e_East = (sin(gha), cos(gha), 0), e_North = (-sin(dec)*cos(gha), sin(dec)*sin(gha), cos(dec))

**Implemented**: `_fplus_fcross_batch`, `_time_delay_batch`, `_geometric_factor_batch`,
`_batch_extrinsic_subprior` in `gpu/run.py`.
**Verified**: `test_extrinsic_lal_subprior_batch` (N=500, 4 types, machine precision) — PASS.

### Combined run_14 benchmark (32k bank, n_ext=512, n_phi=32, n_t=64)

| Step | CPU | GPU run_12 | GPU run_14 | Speedup (vs CPU) |
|------|-----|------------|------------|-----------------|
| `select_intrinsic_samples` | 47.9s | 18.6s | 18.7s | **2.6×** |
| Coherent blocks (25×512) | ~50s est. | 2.18s | 2.17s | **>23×** |
| `standardize_samples` (wall) | ~146s | ~31.5s | **1.06s** | **>137×** |
| `draw_extrinsic_samples` | ~51s | 51.0s | 50.7s | 1.0× 🔒 |
| **Total `inference.run()`** | **254s** | **~111s** | **~81s** | **~3.1×** |

**Standardize speedup**: 146s → 1.06s = **>137×** (all subpriors now vectorized).
**Remaining bottlenecks**:
1. `draw_extrinsic_samples`: 50.7s (🔒 cogwheel QMC, locked)
2. `select_intrinsic_samples`: 18.7s (GPU, 2.6×, mainly waveform I/O)
3. Other (waveform loading, sparse matvec): ~11s

**Test suite**: 9 tests, all PASS.

---

## 2026-03-16 — Real event check (nb03_event, mchirp≈75) + check_real_event.py

### gpu/check_real_event.py
- Loads CPU baseline from an existing run, runs GPU inference on the same
  event/bank/params, compares `prob_samples` sorted by `(i,e,o)`, asserts
  max relative error < 1% on `lnl_marginalized` and `bestfit_lnlike`.
- Default paths: nb03_event (m1=123, m2=62, mchirp≈75, HLV, O3 ASDs,
  IMRPhenomXODE injection), bank=test_bank_32k, n_ext=512, n_phi=32, n_t=64.

### nb03_event GPU run (run_3, 32k bank)
- **5798 intrinsic samples** selected (vs 25 for the profile_event).
  High selection count because the bank covers a wide mchirp range and the
  event has a high SNR (~53 lnl max).
- Coherent: 3 blocks (5798 samples / blocksize=2048)
- draw_extrinsic: ~514s (QMC on 5798 intrinsic samples, 🔒 locked)
- standardize: 23.6s wall
- **Total wall time: 705.6s** (dominated by draw_extrinsic on 5798 samples)

### Correctness result
| Column | max_rel | mean_rel | Status |
|--------|---------|----------|--------|
| `lnl_marginalized` | 0.000e+00 | 0.000e+00 | **PASS** |
| `bestfit_lnlike`   | 0.000e+00 | 0.000e+00 | **PASS** |

GPU run_3 agrees **bit-for-bit** with the existing GPU run_1 (both use the
same GPU code path, confirming full determinism).

**CPU vs GPU comparison note**: A CPU run of nb03_event is not practical
(coherent step alone would take ~3h for 5798 intrinsic samples).  The CPU vs
GPU comparison at 0.7% max relative error was already verified on the
profile_event (25 intrinsic samples) — see 2026-03-14 section.

### Final status
- **9 unit tests: all PASS**
- **GPU determinism on high-SNR event (mchirp≈75): CONFIRMED**
- **CPU vs GPU agreement (profile_event, 32k bank): CONFIRMED at <0.7%**
- **check_real_event.py**: written at `gpu/check_real_event.py`

---

## 2026-03-16 — 1M-bank scaling benchmark (2^20 templates)

### Setup
- **Bank**: `test_bank_1048576` (1,048,576 templates), q∈[0.25,1], mchirp∈[10,40]
- **Event**: `profile_event.npz` re-generated at **d=1366 Mpc** (chirp_mass=20, HLV O3 ASDs,
  seed=20250314) → lnlike≈63.8, SNR≈11.3 (in the target 50–70 range)
- **Params**: single_detector_blocksize=2048 → **512 incoherent batches** (vs 16 at 32k)

### Incoherent step scaling benchmark (clean isolated runs)

| Mode | Batches | Time | Rate | Speedup |
|------|---------|------|------|---------|
| GPU | 512 | **405.4s** | 0.79s/batch | — |
| CPU | 512 | **1323.7s** | 2.58s/batch | — |
| **GPU vs CPU** | — | — | — | **3.27×** |

GPU speedup improves at 1M bank (3.27×) vs 32k bank (2.6×) because:
- More batches → GPU warmup overhead amortized
- GPU fully utilized across all 512 batches
- CPU: 2.58s/batch stable (numpy-bound)

### Incoherent selection result
- **39,318 templates** passed the 20-lnl threshold (max incoherent lnlike = 76.74)
- High survivor count at SNR~11 with 1M-template dense bank
  (compared to 25 survivors at 32k bank with the same event)

### draw_extrinsic bottleneck note
With 39,318 incoherent survivors, `draw_extrinsic_samples` would need to process
~2,457 QMC groups (n_combine=16) — estimated ~17h. This step is locked (cogwheel
CPU QMC), so the full end-to-end run at 1M bank is dominated by it at SNR~11.
At higher SNR (fewer survivors near threshold), or with a targeted bank subset,
the full pipeline completes in comparable time to 32k.

### Summary: Incoherent speedup scales with bank size

| Bank size | GPU (select_intrinsic) | CPU | Speedup |
|-----------|----------------------|-----|---------|
| 32k (2^15) | 18.7s (16 batches) | 47.9s | **2.6×** |
| 1M (2^20) | 405.4s (512 batches) | 1323.7s | **3.3×** |

GPU advantage grows with bank size — consistent with better GPU utilization at scale.

---

## 2026-03-16 — Track G: GPU bank preload + transfer cost analysis

### Transfer cost analysis (1M bank, answered question)
CPU↔GPU memory transfers are **not** the bottleneck (~1s out of 405s, <0.3%).
Real bottleneck is **disk I/O + CPU waveform processing** per batch:
- `load_amp_and_phase`: 424.9s cumtime (514 calls × 0.83s)
- `get_relative_linfree_dt_from_waveform`: 327.7s cumtime (514 calls × 0.64s)
- GPU kernel itself: 40.9s total across all 1536 calls = **0.08s/batch**
GPU sits idle ~94% of the time waiting for CPU to load from disk.

### Track G: GPU bank preload
**Idea**: load and process all waveform files once at startup, store entire bank as GPU tensor.
During incoherent loop, slice from GPU memory — zero disk I/O per batch.

**Memory**: 1M × 4 modes × 2 pol × 378 freq bins × complex64 ≈ **24 GB** (fits in 44 GB L40S VRAM).

**Implemented in `gpu/run.py`**:
- `_preload_bank_to_gpu(bank_folder, sdp)`: loads all blocks via `ThreadPool(8)`, uploads to GPU
- `_make_preloaded_run_for_single_detector(original_fn)`: wraps `run_for_single_detector`; on first
  call triggers lazy preload, then slices `h_gpu[inds]` each batch (zero disk I/O)
- `_PRELOAD_ENABLED` flag (default False); set via `gpu.run._PRELOAD_ENABLED = True`
- `_patch()` now also registers the preload wrapper on `_inf.run_for_single_detector`

**`gpu/single_detector_gpu.py`**: `_to_c64` updated to accept torch tensors (skips re-upload).

**`gpu/profile_run.py`**: `--preload-bank` flag added.

**Expected speedup (1M bank)**:
| Phase | Before | After Track G |
|-------|--------|---------------|
| Preload (one-time) | 0s | ~60s |
| Incoherent | 405s | ~41s |
| Everything else | ~558s | ~558s |
| **Total** | **~963s** | **~660s** |

**Test added**: `test_preloaded_bank_lnlikes` — verifies GPU tensor slice gives bit-identical lnlikes vs numpy path.
**Test suite**: 10 tests, all PASS.

### Key insight: draw_extrinsic does not scale with survivor count
`draw_extrinsic_samples` has **fixed cost with large internal variance** — it does not depend on
how many intrinsic samples survived the incoherent threshold. Cost is purely from the QMC
convergence loop (variable number of chunks until n_effective threshold is met).

---

## 2026-03-16 — Phase F filter skip (merged from origin/main)

### Problem
`get_marg_info_batch_multibank` was calling `self.likelihood.lnlike()` on all 1024 batch
candidates as a pre-filter. With `min_marg_lnlike_for_sampling=0.0` (default), virtually
all candidates pass, spending ~330–562s on lnlike pre-evaluation with no net benefit.

### Fix (in dot_pe/coherent_processing.py, from main branch)
When `min_marg_lnlike_for_sampling <= 0`, skip the filter entirely and take the first
`n_to_process` candidates directly from the batch. Merged into `dev/gpu` via
`git merge origin/main`.

### Impact
- draw_extrinsic: **692s → 114s** (6× improvement) — dominant fix for this stage
- H1/H2/H3 GPU patches contributed negligible additional gain relative to this

---

## 2026-03-16 — Track H: GPU acceleration of draw_extrinsic_samples

### H1: np.bincount replacing scipy.sparse scatter-add
`MarginalizationInfoSamplerFree.__post_init__` constructed two `sparse.coo_array` objects
just to do scatter-add (hidden in matrix construction overhead). Replaced with `np.bincount`
— zero dependencies, single C loop.

Patch: `_fast_post_init` in `gpu/extrinsic_gpu.py`, registered as:
```python
MarginalizationInfoSamplerFree.__post_init__ = _fast_post_init
```

### H2: GPU phasor matmuls in `_get_dh_hh_qo`
`CoherentScoreSamplerFree._get_dh_hh_qo` computes `real_matmul(dh_qm, phasor)` and
`real_matmul(hh_qm, phasor)` for each QMC chunk (~2100 calls). Patched to run on GPU:
- `_dh_phasor` / `_hh_phasor` uploaded to GPU once at init
- Per-chunk: upload `dh_qm`, `hh_qm` → torch matmuls → download result

Patch: `_get_dh_hh_qo_gpu` in `gpu/extrinsic_gpu.py`.

### H3: GPU batched matmuls in `_get_many_dh_hh`
`MarginalizationExtrinsicSamplerFreeLikelihood._get_many_dh_hh` had an 8-iteration loop of
complex matmuls plus a 5-index einsum. Replaced with `torch.einsum`.

**Critical bug**: `h_h_weights` values ~3e48 exceed float32 max (3.4e38). Fix: compute
einsum in complex128 (preserving full precision), then cast result back to complex64.
Same pattern as the Track A `hh_weights_dmppb` overflow fix.

Patch: `_get_many_dh_hh_gpu` in `gpu/extrinsic_gpu.py`.

### Profile internals (draw_extrinsic, GPU, nb03/nb04 consistent after Phase F)

| Function | Time | Calls |
|---|---|---|
| `optimize_beta_temperature` | 75–82s | 1 |
| `_get_marginalization_info_chunk` | ~40s | ~2100 |
| `_fast_post_init` (H1) | ~24s | ~3600 |
| `_kde_t_arrival_prob` | ~24s | ~1480 |
| `_get_lnnumerators_important_flippsi` | ~20s | ~2100 |
| `_fitpack2` spline | ~18s | ~2100 |
| `argsort` | ~14s | ~6200 |
| `_get_dh_hh_qo_gpu` (H2) | ~12s | ~2100 |

`optimize_beta_temperature` (75–82s, 1 call, pure CPU QMC setup) is now the dominant
bottleneck — not GPU-patchable in current form.

### Tests
- `test_fast_post_init` and `test_get_dh_hh_qo_gpu` added; **12 tests total, all PASS**

---

## 2026-03-16 — Production benchmarks (nb03/nb04/nb05)

All runs use: `draw_subset=True, n_ext=512, n_phi=32, n_t=64, blocksize=2048`
Event: `nb04_event` (chirp_mass=20, d=1366 Mpc, lnlike_max≈76, SNR≈11 — realistic O3)

### nb04 — 2 × 64k banks (128k total)

| Stage | CPU | GPU | Speedup |
|---|---|---|---|
| Incoherent bank_0 (65k) | 150 | 27 | 5.6× |
| Incoherent bank_1 (65k) | 96 | 27 | 3.6× |
| draw_extrinsic | 111 | 119 | ~1× |
| Coherent (both banks) | 79 | 72 | ~1× |
| Standardizing | 0.4 | 0.1 | — |
| **Total** | **~436** | **~245** | **1.8×** |

draw_extrinsic is GPU-neutral — `optimize_beta_temperature` (75s, CPU-only) dominates both paths.

### nb05 — single 1M bank

| Stage | CPU | GPU | Speedup |
|---|---|---|---|
| Incoherent (1M) | 1490 | 417 | 3.6× |
| draw_extrinsic | 106 | 112 | ~1× |
| Coherent (20 blocks) | 162 | 115 | 1.4× |
| Standardizing | 2 | 0.3 | — |
| **Total** | **~1760** | **~644** | **2.7×** |

At 1M scale, incoherent dominates (~85% of CPU time) → end-to-end speedup jumps to 2.7×
vs 1.8× at 128k. See `gpu/BENCHMARK.md` for full details.

### draw_subset=True fix
`gpu/profile_run.py` had `draw_subset=False` hardcoded — causing 7640s standardizing in
CPU baselines (20M prob_samples rows all vectorized by `np.vectorize`). Fixed to
`draw_subset=True` to match production notebook default (downsamples to ~n_effective/2
rows before standardizing → 0.1–2s).

### Centralized report
`gpu/BENCHMARK.md` created — auto-updated after every benchmark run.
