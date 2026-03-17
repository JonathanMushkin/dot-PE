"""
Batched GPU replacement for cogwheel's LookupTable.sample_distance.

The CPU bottleneck at 2^18+ scale:
  ~10M calls x np.vectorize(lookup_table.sample_distance(..., resolution=32))
  Each call: build ~64-point distance grid, fit scipy spline, sample CDF.

GPU replacement:
  Chunked kernel over batches of CHUNK rows x K grid points:
  - Focused grid built on GPU (avoids large CPU allocations)
  - Posterior, CDF, and searchsorted all on GPU
  - torch.sort replaces np.sort (faster, stays on GPU)
  -> sub-second per 1M rows (GPU), linear scaling.

Entry point: sample_distance_batched_gpu(
    d_h_arr, h_h_arr, lookup_table, resolution=32)
"""

import numpy as np
import torch

from gpu.gpu_constants import DEVICE

# Process at most this many rows at once (controls GPU memory usage)
# (256k, 64) float32 ~ 64 MB per tensor; ~10 tensors -> ~640 MB per chunk
_CHUNK = 256_000


def _sample_chunk(
    d_h: np.ndarray,    # (C,) float64
    h_h: np.ndarray,    # (C,) float64
    REF: float,
    D_MAX: float,
    inv_vol: float,
    prior_name: str,
    lookup_table,
    K: int,
    # optional (C,) fractional quantiles in (0,1) for testing
    _u_fracs: np.ndarray = None,
) -> np.ndarray:
    """Process one chunk of C samples entirely on GPU."""
    C = len(d_h)

    # --- scalar quantities on CPU (lightweight) ---
    norm_h  = np.sqrt(h_h)        # (C,)
    overlap = d_h / norm_h         # (C,)
    u_peak  = d_h / (REF * h_h)  # (C,)  peak in u-space
    delta_u = 10.0 / norm_h      # (C,)  10sigma half-width

    # Mirror CPU grid construction exactly:
    #   CPU: linspace(u_peak+delta_u, u_peak-delta_u, K),
    #        then filter to u > 0 & d < D_MAX
    # GPU: same linspace spacing, but clip u to [REF/D_MAX, inf)
    #      instead of filtering.
    # Points with u < REF/D_MAX map to d >= D_MAX where the
    # posterior is ~0, so they add negligible weight to the CDF
    # while keeping K fixed for batching.
    u_hi = u_peak + delta_u  # (C,)  always positive
    u_lo = u_peak - delta_u  # (C,)  may be negative
    u_min = REF / D_MAX      # scalar

    # All computation in float64 for accurate CDF accumulation
    # (float32 cumsum errors blow up in the tail at u_frac > 0.9)
    F64 = torch.float64

    # Move scalars to GPU in float64
    norm_h_g  = torch.from_numpy(norm_h).to(DEVICE, dtype=F64)   # (C,)
    overlap_g = torch.from_numpy(overlap).to(DEVICE, dtype=F64)  # (C,)
    u_lo_g    = torch.from_numpy(u_lo).to(DEVICE, dtype=F64)     # (C,)
    u_hi_g    = torch.from_numpy(u_hi).to(DEVICE, dtype=F64)     # (C,)

    # Focused grid on GPU: descending in u (matches CPU linspace
    # direction), clipped at u_min so d stays within [d_MIN, D_MAX].
    t_g = torch.linspace(
        0.0, 1.0, K, device=DEVICE, dtype=F64
    )  # (K,)
    u_foc_g = (
        u_hi_g.unsqueeze(1)
        - (u_hi_g - u_lo_g).unsqueeze(1) * t_g.unsqueeze(0)
    )  # (C,K)
    u_foc_g = torch.clamp(u_foc_g, min=float(u_min))
    d_foc_g = torch.clamp(REF / u_foc_g, 0.0, D_MAX)  # (C,K)

    # Broad grid on GPU: (C, K) uniform in d-space
    d_broad_g = torch.linspace(
        D_MAX / K, D_MAX, K, device=DEVICE, dtype=F64
    )  # (K,)
    d_broad_g = d_broad_g.unsqueeze(0).expand(C, -1)  # (C,K)

    # Combined grid (C, 2K), sorted per row
    d_g, _ = torch.sort(
        torch.cat([d_broad_g, d_foc_g], dim=1), dim=1
    )  # (C, 2K)

    # Prior
    if prior_name == 'euclidean':
        prior_g = 4.0 * np.pi * d_g ** 2
    else:
        d_np   = d_g.cpu().numpy()
        prior_g = torch.from_numpy(
            lookup_table.d_luminosity_prior(d_np)
        ).to(DEVICE, dtype=F64)

    # Gaussian likelihood: exp(-0.5*(norm_h * REF/d - overlap)^2)
    d_safe      = torch.clamp(d_g, min=1e-9)
    u_g         = REF / d_safe                             # (C, 2K)
    exponent    = -0.5 * (
        norm_h_g.unsqueeze(1) * u_g - overlap_g.unsqueeze(1)
    ) ** 2
    posterior_g = prior_g * inv_vol * torch.exp(
        torch.clamp(exponent, min=-1000.0)
    )
    posterior_g = posterior_g * (d_g > 0).double()

    # CDF via trapezoidal rule
    # (float64 cumsum avoids precision loss in the tail)
    delta_d = d_g[:, 1:] - d_g[:, :-1]              # (C, 2K-1)
    trap    = (
        0.5 * (posterior_g[:, :-1] + posterior_g[:, 1:]) * delta_d
    )                                                  # (C, 2K-1)
    cdf_g   = torch.zeros(C, 2 * K, device=DEVICE, dtype=F64)
    cdf_g[:, 1:] = torch.cumsum(trap, dim=1)

    # Sample uniformly from CDF
    cdf_max = cdf_g[:, -1]                             # (C,)
    if _u_fracs is not None:
        u_samp = (
            torch.from_numpy(_u_fracs).to(DEVICE, dtype=F64) * cdf_max
        )
    else:
        u_samp = (
            torch.rand(C, device=DEVICE, dtype=F64) * cdf_max  # (C,)
        )

    # CDF inversion via searchsorted + linear interpolation.
    # searchsorted returns the INSERT position i s.t.
    # cdf[i-1] < u_samp <= cdf[i], so the lower bound for
    # interpolation is idx = insert_pos - 1.
    idx = (
        torch.searchsorted(
            cdf_g.contiguous(), u_samp.unsqueeze(1)
        ).squeeze(1) - 1
    )
    idx   = torch.clamp(idx, 0, 2 * K - 2).long()
    rows  = torch.arange(C, device=DEVICE)

    d_lo   = d_g[rows, idx]
    d_hi   = d_g[rows, idx + 1]
    cdf_lo = cdf_g[rows, idx]
    cdf_hi = cdf_g[rows, idx + 1]

    denom  = torch.clamp(cdf_hi - cdf_lo, min=1e-300)
    frac   = torch.clamp((u_samp - cdf_lo) / denom, 0.0, 1.0)
    d_out  = d_lo + frac * (d_hi - d_lo)

    return d_out.cpu().numpy()


def sample_distance_batched_gpu(
    d_h_arr: np.ndarray,
    h_h_arr: np.ndarray,
    lookup_table,
    resolution: int = 32,
    # (N,) fractional quantiles in (0,1) -- for testing only
    _u_fracs: np.ndarray = None,
) -> np.ndarray:
    """
    Batched GPU replacement for
    ``lookup_table.sample_distance`` called over arrays.

    Processes rows in chunks of ``_CHUNK`` to bound GPU memory usage.

    Parameters
    ----------
    d_h_arr : (N,) float64
        Inner products <d|h> at 1 Mpc.
    h_h_arr : (N,) float64
        Inner products <h|h> at 1 Mpc.
    lookup_table : cogwheel LookupTable
        Supplies prior, D_MAX, REFERENCE_DISTANCE, _inverse_volume.
    resolution : int
        Grid resolution per sample (default 32).
    _u_fracs : (N,) float64, optional
        Fractional quantile values in (0, 1) to use instead of
        random draws. For deterministic testing only.

    Returns
    -------
    (N,) float64 array of luminosity distance samples.
    """
    N = len(d_h_arr)
    REF      = float(lookup_table.REFERENCE_DISTANCE)
    D_MAX    = float(lookup_table.d_luminosity_max)
    inv_vol  = float(lookup_table._inverse_volume)
    prior_nm = lookup_table.d_luminosity_prior_name
    K        = resolution

    d_h = np.asarray(d_h_arr, dtype=np.float64)
    h_h = np.asarray(h_h_arr, dtype=np.float64)
    out = np.empty(N, dtype=np.float64)

    for start in range(0, N, _CHUNK):
        end = min(start + _CHUNK, N)
        chunk_fracs = _u_fracs[start:end] if _u_fracs is not None else None
        out[start:end] = _sample_chunk(
            d_h[start:end], h_h[start:end],
            REF, D_MAX, inv_vol, prior_nm, lookup_table, K,
            _u_fracs=chunk_fracs,
        )

    torch.cuda.synchronize()
    return out
