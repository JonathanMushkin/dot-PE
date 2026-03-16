# GPU Acceleration Benchmark Report

**Event:** nb04_event (chirp_mass=20, d=1366 Mpc, lnlike_max≈76, SNR≈11 — realistic O3 range)
**Machine:** L40S GPU node
**Settings:** n_ext=512, n_phi=32, n_t=64, draw_subset=True, blocksize=2048

All times in seconds. CPU = dot_pe.inference (no GPU patches). GPU = gpu.run (Tracks A–H patches active).

---

## nb03 — single 32k bank

| Stage | GPU |
|---|---|
| Incoherent (32k) | 14 |
| draw_extrinsic | 119 |
| Coherent | 27 |
| Standardizing | 0.1 |
| **Total** | **~160** |

*CPU baseline for nb03 not yet collected.*

---

## nb04 — 2 × 64k banks (128k total)

| Stage | CPU | GPU | Speedup |
|---|---|---|---|
| Incoherent bank_0 (65k) | 150 | 27 | 5.6× |
| Incoherent bank_1 (65k) | 96 | 27 | 3.6× |
| draw_extrinsic | 111 | 119 | ~1× |
| Coherent (both banks) | 79 | 72 | ~1× |
| Standardizing | 0.4 | 0.1 | — |
| **Total** | **~436** | **~245** | **1.8×** |

**Note:** draw_extrinsic is GPU-neutral at this bank size — `optimize_beta_temperature`
(75s, CPU-only QMC setup, called once) dominates both paths. GPU advantage comes
entirely from incoherent selection.

---

## nb05 — single 1M bank

| Stage | CPU | GPU | Speedup |
|---|---|---|---|
| Incoherent (1M) | 1490 | 417 | 3.6× |
| draw_extrinsic | 106 | 112 | ~1× |
| Coherent (20 blocks) | 162 | 115 | 1.4× |
| Standardizing | 2 | 0.3 | — |
| **Total** | **~1760** | **~644** | **2.7×** |

39318 survivors (threshold 56.74). draw_extrinsic **identical to 32k/128k** — confirms
fixed cost dominated by `optimize_beta_temperature`. `batches=1/39`: accepted 16 on first
batch of 1024 from 39k survivors.

CPU incoherent rate: ~2.9s/batch (512 batches × 2048 samples). GPU: ~0.81s/batch — 3.6×
faster. At 1M scale incoherent dominates (~85% of CPU time), giving 2.7× end-to-end
speedup vs 1.8× at 128k. Coherent also faster on GPU (1.4×) at this scale.

---

## draw_extrinsic internals (GPU, nb03/nb04 consistent)

| Function | Time | Calls | Note |
|---|---|---|---|
| `optimize_beta_temperature` | 75–82s | 1 | **dominant — CPU QMC setup, not GPU-patchable yet** |
| `_get_marginalization_info_chunk` | ~40s | ~2100 | QMC loop |
| `_fast_post_init` (H1) | ~24s | ~3600 | bincount; scipy sparse still 8.7s inside |
| `_kde_t_arrival_prob` | ~24s | ~1480 | KDE, CPU |
| `_get_lnnumerators_important_flippsi` | ~20s | ~2100 | |
| `_fitpack2` spline (in H2) | ~18s | ~2100 | CPU spline inside GPU patch |
| `argsort` | ~14s | ~6200 | |
| `take_along_axis` | ~12s | ~7400 | |
| `_get_dh_hh_qo_gpu` (H2) | ~12s | ~2100 | |
| `get_marg_info_batch_multibank` | ~9s | 1 | was 562s before Phase F filter skip |

**Key optimization history for draw_extrinsic:**
- Before Phase F filter skip: 692s (GPU) / ~677s (CPU)
- After Phase F filter skip (merged from main): 114s (GPU) / 111s (CPU) — **6× improvement**
- H1/H2/H3 GPU patches: negligible additional gain after filter skip dominates

---

## Track history summary

| Track | Change | Speedup (incoherent, 32k) |
|---|---|---|
| A | GPU single-detector lnlike | baseline → GPU |
| B–F | Various GPU + correctness | incremental |
| G | Bank preload to VRAM | eliminates per-batch disk I/O |
| H | GPU extrinsic (H1 bincount, H2/H3 matmuls) | draw_extrinsic: 692s → 114s (with Phase F) |
| Phase F (merged from main) | Skip lnlike pre-filter when threshold=0 | draw_extrinsic: 692s → 114s |
