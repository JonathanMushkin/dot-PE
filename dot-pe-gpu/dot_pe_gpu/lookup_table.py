"""Distance-marginalization lookup table on the GPU.

Exact port of cogwheel's ``LookupTable.lnlike_marginalized``: the same interpolation coordinates
(``_get_x_y``) and the same bicubic interpolating B-spline (scipy ``RectBivariateSpline``, knots
``tx``, ``ty`` and coefficients ``c`` taken from its ``tck``). Out-of-range arguments are clamped
to the knot span, as FITPACK's ``fpbisp`` does.

``LOOKUP_DEVICE_SRC`` is a CUDA device function, ``lnlike_marg(dh, hh, ...)``, for inclusion in
fused kernels. ``bispev_numpy`` is the same algorithm in numpy, for CPU tests.
"""

from __future__ import annotations

import numpy as np

K = 3  # spline degree in both coordinates (RectBivariateSpline default)


def table_arrays(lookup_table=None):
    """(tx, ty, c, d_luminosity_max, sigmas, z0) from a cogwheel LookupTable (default: dot-PE's)."""
    if lookup_table is None:
        from cogwheel.likelihood import LookupTable
        lookup_table = LookupTable()
    spl = lookup_table._interpolated_table
    if tuple(spl.degrees) != (K, K):
        raise ValueError(f"lookup table spline degrees {spl.degrees}, expected {(K, K)}")
    if lookup_table.REFERENCE_DISTANCE != 1.0:
        raise ValueError("REFERENCE_DISTANCE != 1 is not supported")
    tx, ty, c = spl.tck
    return (np.asarray(tx, float), np.asarray(ty, float), np.asarray(c, float),
            float(lookup_table.d_luminosity_max), float(lookup_table._SIGMAS),
            float(lookup_table._Z0))


def _basis(t, x):
    """Interval index l (t[l] <= x < t[l+1], clamped) and the K+1 non-zero B-spline values."""
    n = len(t)
    x = np.clip(x, t[K], t[n - K - 1])
    l = np.clip(np.searchsorted(t, x, side="right") - 1, K, n - K - 2)
    h = np.zeros((K + 1,) + np.shape(x))
    h[0] = 1.0
    for j in range(1, K + 1):
        hh = h[:j].copy()
        h[0] = 0.0
        for i in range(1, j + 1):
            tli = t[l + i]
            tlj = t[l + i - j]
            f = hh[i - 1] / (tli - tlj)
            h[i - 1] = h[i - 1] + f * (tli - x)
            h[i] = f * (x - tlj)
    return l, h


def bispev_numpy(tx, ty, c, x, y):
    """Evaluate the bicubic spline at points (x, y) (any matching shapes)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ny_c = len(ty) - K - 1
    lx, hx = _basis(tx, x)
    ly, hy = _basis(ty, y)
    out = np.zeros(np.broadcast(x, y).shape)
    for i in range(K + 1):
        for j in range(K + 1):
            out += hx[i] * hy[j] * c[(lx - K + i) * ny_c + (ly - K + j)]
    return out


def lnlike_marginalized_numpy(d_h, h_h, tables):
    tx, ty, c, dmax, sigmas, z0 = tables
    norm_h = np.sqrt(h_h)
    overlap = d_h / norm_h
    x = np.log(norm_h / (dmax * (sigmas + np.abs(overlap))))
    v = overlap / z0
    y = v / (1 + np.abs(v))
    return bispev_numpy(tx, ty, c, x, y) + d_h ** 2 / h_h / 2


# CUDA device code. Knots and coefficients are float32 in global memory (read through the
# read-only cache); the arithmetic is float32 except the interval search.
LOOKUP_DEVICE_SRC = r"""
__device__ __forceinline__ int lt_interval(const float* __restrict__ t, int n, float x) {
    // largest l in [3, n-5] with t[l] <= x (x already clamped to [t[3], t[n-4]])
    int lo = 3, hi = n - 5;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (__ldg(t + mid) <= x) lo = mid; else hi = mid - 1;
    }
    return lo;
}

__device__ __forceinline__ void lt_basis(const float* __restrict__ t, int l, float x, float h[4]) {
    float hh[4];
    h[0] = 1.f; h[1] = 0.f; h[2] = 0.f; h[3] = 0.f;
    #pragma unroll
    for (int j = 1; j <= 3; ++j) {
        #pragma unroll
        for (int i = 0; i < j; ++i) hh[i] = h[i];
        h[0] = 0.f;
        #pragma unroll
        for (int i = 1; i <= j; ++i) {
            float tli = __ldg(t + l + i), tlj = __ldg(t + l + i - j);
            float f = hh[i - 1] / (tli - tlj);
            h[i - 1] += f * (tli - x);
            h[i] = f * (x - tlj);
        }
    }
}

// cogwheel LookupTable.lnlike_marginalized(d_h, h_h); NaN if h_h <= 0.
__device__ __forceinline__ float lnlike_marg(
        float dh, float hh,
        const float* __restrict__ tx, int ntx,
        const float* __restrict__ ty, int nty,
        const float* __restrict__ cf,
        float dmax, float sigmas, float z0) {
    if (!(hh > 0.f)) return __int_as_float(0x7fc00000);
    float norm_h = sqrtf(hh);
    float ov = dh / norm_h;
    float x = logf(norm_h / (dmax * (sigmas + fabsf(ov))));
    float v = ov / z0;
    float y = v / (1.f + fabsf(v));
    x = fminf(fmaxf(x, __ldg(tx + 3)), __ldg(tx + ntx - 4));
    y = fminf(fmaxf(y, __ldg(ty + 3)), __ldg(ty + nty - 4));
    int lx = lt_interval(tx, ntx, x), ly = lt_interval(ty, nty, y);
    float hx[4], hy[4];
    lt_basis(tx, lx, x, hx);
    lt_basis(ty, ly, y, hy);
    int nyc = nty - 4;
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float* row = cf + (lx - 3 + i) * nyc + (ly - 3);
        float r = hy[0] * __ldg(row) + hy[1] * __ldg(row + 1)
                + hy[2] * __ldg(row + 2) + hy[3] * __ldg(row + 3);
        s += hx[i] * r;
    }
    return s + 0.5f * dh * dh / hh;
}
"""

_ELEMENTWISE_SRC = LOOKUP_DEVICE_SRC + r"""
extern "C" __global__ void lnlike_marg_kernel(
        const float* __restrict__ dh, const float* __restrict__ hh, float* __restrict__ out,
        long long n,
        const float* __restrict__ tx, int ntx, const float* __restrict__ ty, int nty,
        const float* __restrict__ cf, float dmax, float sigmas, float z0) {
    long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) out[k] = lnlike_marg(dh[k], hh[k], tx, ntx, ty, nty, cf, dmax, sigmas, z0);
}
"""


class GPULookupTable:
    """Device copy of the lookup table; ``lnlike_marginalized`` on CuPy float32 arrays."""

    def __init__(self, lookup_table=None):
        import cupy as cp
        self.cp = cp
        tx, ty, c, dmax, sigmas, z0 = table_arrays(lookup_table)
        self.tables = (tx, ty, c, dmax, sigmas, z0)
        self.tx = cp.asarray(tx, dtype=cp.float32)
        self.ty = cp.asarray(ty, dtype=cp.float32)
        self.c = cp.asarray(c, dtype=cp.float32)
        self.ntx, self.nty = len(tx), len(ty)
        self.scalars = (np.float32(dmax), np.float32(sigmas), np.float32(z0))
        self._kernel = cp.RawKernel(_ELEMENTWISE_SRC, "lnlike_marg_kernel")

    def kernel_args(self):
        """Arguments to append to a fused kernel call that includes ``LOOKUP_DEVICE_SRC``."""
        return (self.tx, np.int32(self.ntx), self.ty, np.int32(self.nty), self.c, *self.scalars)

    def lnlike_marginalized(self, d_h, h_h):
        cp = self.cp
        dh = cp.ascontiguousarray(d_h, dtype=cp.float32).ravel()
        hh = cp.ascontiguousarray(h_h, dtype=cp.float32).ravel()
        out = cp.empty_like(dh)
        n = dh.size
        threads = 256
        self._kernel(((n + threads - 1) // threads,), (threads,),
                     (dh, hh, out, np.int64(n), *self.kernel_args()))
        return out.reshape(cp.shape(d_h))
