"""Coherent stage of dot-PE on the GPU: one intrinsic block against all extrinsic samples.

``CoherentGPU.run_iblock`` is the device counterpart of ``dot_pe.thin_coherent.run_thin_iblock``
and returns the same tuple. Semantics kept from the CPU code:

1. Per row (i, e, o): <d|h> and <h|h> on the phase grid, the best-fit log-likelihood
   0.5 <d|h>^2 / <h|h> (0 where <d|h> <= 0), ``invalid_bestfit_mask`` and, if given,
   ``incoherent_bound_mask`` (both set the best fit to -inf).
2. The running threshold over extrinsic blocks of ``blocksize_e`` (in order): a row of extrinsic
   block j is accepted if its best fit exceeds max over earlier blocks j' < j of
   (block maximum - ``max_bestfit_lnlike_diff``); -inf for j = 0.
3. Accepted rows get ``lnl_marginalized`` from the lookup table and
   ``ln_posterior`` = lnl + logw_i + logw_e.
4. ``row_selection="bestfit"``: the block keeps its ``max_output_per_block`` accepted rows of
   highest best fit (the same set as the CPU's incremental trims).
   ``row_selection="phase_marginalized_priority"``: one row per (i, e) pair with accepted rows,
   phase drawn with probability w_o / W, ``ln_phase_marginalized_weight`` = ln W - ln N_k,
   ``priority_key`` = that - ln u; the block keeps its ``max_output_per_block`` pairs of largest
   key and ``priority_tau`` = the largest key it dropped.
5. Discarded rows (not accepted): count, log-sum of exp(ln_posterior) and of exp(2 ln_posterior)
   over every discarded row with a finite value, with lnl = 0 where <d|h> <= 0. The CPU estimates
   the same sums from a random subsample, so the two agree only within its sampling error.

Nothing of size (i, e, o) is stored: two passes over the (i, e) pairs each loop over the phase
grid in registers. Only the kept rows leave the device. Random numbers come from a counter-based
hash of (seed, bank, template, extrinsic sample, phase), so a GPU run is reproducible, but its
draws differ from numpy's.
"""

from __future__ import annotations

import itertools

import numpy as np

from .lookup_table import LOOKUP_DEVICE_SRC, GPULookupTable

MIN_D_LUMINOSITY = 1.0  # dot_pe.likelihood_calculating.MIN_D_LUMINOSITY
INCOHERENT_BOUND_TOL = 20.0  # dot_pe.likelihood_calculating.INCOHERENT_BOUND_TOL
DISC_STRIDE = 1  # 1: discarded-row statistics over every discarded row (exact)
EMIT_MARGIN = 1e-3  # emit rows down to emit_cut - EMIT_MARGIN: pass 1 and 2 need not agree to the bit
MAX_EMIT_ROWS = 200_000_000  # device buffer limit for emitted rows (bestfit mode)

_KERNEL_SRC = LOOKUP_DEVICE_SRC + r"""
#define NM 4
#define NMM 10
#define MAXPHI 256
#define INF_F __int_as_float(0x7f800000)

__device__ __forceinline__ unsigned long long splitmix64(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// uniform in (0, 1), 52 bits, float64: extremes 2^-53 and 1 - 2^-53, both exact
__device__ __forceinline__ double u01d(unsigned long long h) {
    return ((double)(h >> 12) + 0.5) * 2.220446049250313e-16;
}

struct Pair {
    float dr[NM], di[NM], hr[NMM], hi[NMM];
};

__device__ __forceinline__ void load_pair(const float2* __restrict__ dh, const float2* __restrict__ hh,
                                          long long p, long long npairs, Pair& q) {
    #pragma unroll
    for (int m = 0; m < NM; ++m) { float2 v = dh[m * npairs + p]; q.dr[m] = v.x; q.di[m] = v.y; }
    #pragma unroll
    for (int m = 0; m < NMM; ++m) { float2 v = hh[m * npairs + p]; q.hr[m] = v.x; q.hi[m] = v.y; }
}

// row values at phase o; phasors in shared memory: sp[(m * nphi + o)]
__device__ __forceinline__ void row_at(const Pair& q, const float2* sdp, const float2* shp, int o, int nphi,
                                       float& dh, float& hh) {
    float a = 0.f, b = 0.f;
    #pragma unroll
    for (int m = 0; m < NM; ++m) { float2 f = sdp[m * nphi + o]; a += q.dr[m] * f.x - q.di[m] * f.y; }
    #pragma unroll
    for (int m = 0; m < NMM; ++m) { float2 f = shp[m * nphi + o]; b += q.hr[m] * f.x - q.hi[m] * f.y; }
    dh = a; hh = b;
}

__device__ __forceinline__ float bestfit_of(float dh, float hh, float bound) {
    // invalid_bestfit_mask: hh <= 0, or dh > 0 and hh / dh < MIN_D_LUMINOSITY
    if (!isfinite(dh) || !(hh > 0.f) || (dh > 0.f && hh < MIN_D * dh)) return (-INF_F);
    float b = dh > 0.f ? 0.5f * dh * dh / hh : 0.f;
    return b > bound ? (-INF_F) : b;  // incoherent_bound_mask (bound = +inf if none)
}

__device__ __forceinline__ void load_phasors(float2* sdp, float2* shp, const float2* __restrict__ dp,
                                             const float2* __restrict__ hp, int nphi) {
    for (int k = threadIdx.x; k < NM * nphi; k += blockDim.x) sdp[k] = dp[k];
    for (int k = threadIdx.x; k < NMM * nphi; k += blockDim.x) shp[k] = hp[k];
    __syncthreads();
}

// Pass 1: maximum best fit over the phase grid, per pair.
extern "C" __global__ void k_pairmax(
        const float2* __restrict__ dh, const float2* __restrict__ hh, int n_i, int n_e, int nphi,
        const float2* __restrict__ dp, const float2* __restrict__ hp, const float* __restrict__ bound_i,
        float* __restrict__ pmax) {
    extern __shared__ float2 sm[];
    float2* sdp = sm; float2* shp = sm + NM * nphi;
    load_phasors(sdp, shp, dp, hp, nphi);
    long long npairs = (long long)n_i * n_e;
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npairs) return;
    int i = (int)(p / n_e);
    float bound = bound_i[i];
    Pair q; load_pair(dh, hh, p, npairs, q);
    float mx = (-INF_F);
    for (int o = 0; o < nphi; ++o) {
        float d, h; row_at(q, sdp, shp, o, nphi, d, h);
        mx = fmaxf(mx, bestfit_of(d, h, bound));
    }
    pmax[p] = mx;
}

// Pass 2: acceptance, counts, discarded statistics, and (mode 0) row emission above `emit_cut`
// or (mode 1) phase marginalization and priority key per pair.
extern "C" __global__ void k_eval(
        const float2* __restrict__ dh, const float2* __restrict__ hh, int n_i, int n_e, int nphi,
        const float2* __restrict__ dp, const float2* __restrict__ hp, const float* __restrict__ bound_i,
        const float* __restrict__ thr_eb, int eb_size,
        const float* __restrict__ logw_i, const float* __restrict__ logw_e,
        const long long* __restrict__ ibank, unsigned long long seed, int bank_index, double ln_n_k,
        long long disc_stride,
        int mode, float emit_cut, int* __restrict__ emit_count, int emit_cap,
        int* __restrict__ out_i, int* __restrict__ out_e, int* __restrict__ out_o,
        float* __restrict__ out_lnl, float* __restrict__ out_post, float* __restrict__ out_bf,
        float* __restrict__ out_dh, float* __restrict__ out_hh,
        double* __restrict__ pair_lnw, double* __restrict__ pair_key,
        int* __restrict__ pair_nacc, float* __restrict__ pair_disc,
        const float* __restrict__ tx, int ntx, const float* __restrict__ ty, int nty,
        const float* __restrict__ cf, float dmax, float sigmas, float z0) {
    extern __shared__ float2 sm[];
    float2* sdp = sm; float2* shp = sm + NM * nphi;
    load_phasors(sdp, shp, dp, hp, nphi);
    long long npairs = (long long)n_i * n_e;
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npairs) return;
    int i = (int)(p / n_e), e = (int)(p % n_e);
    float bound = bound_i[i];
    float thr = thr_eb[e / eb_size];
    float lw = logw_i[i] + logw_e[e];
    unsigned long long hbase = splitmix64(seed ^ splitmix64(((unsigned long long)bank_index << 40)
                                                            ^ (unsigned long long)ibank[i]));
    Pair q; load_pair(dh, hh, p, npairs, q);

    int nacc = 0;
    // discarded-row statistics (subsample): count, max-shifted sums of exp(x) and exp(2x)
    float dn = 0.f, dm = (-INF_F), ds1 = 0.f, ds2 = 0.f;
    // phase marginalization state (mode 1)
    float wm = (-INF_F), ws = 0.f; double gbest = (-INF_F);
    int obest = -1; float b_lnl = 0.f, b_post = 0.f, b_bf = 0.f, b_dh = 0.f, b_hh = 0.f;

    for (int o = 0; o < nphi; ++o) {
        float d, h; row_at(q, sdp, shp, o, nphi, d, h);
        float bf = bestfit_of(d, h, bound);
        if (bf > thr) {
            ++nacc;
            float lnl = lnlike_marg(d, h, tx, ntx, ty, nty, cf, dmax, sigmas, z0);
            float post = lnl + lw;
            if (mode == 0) {
                if (bf >= emit_cut) {
                    int k = atomicAdd(emit_count, 1);
                    if (k < emit_cap) {
                        out_i[k] = i; out_e[k] = e; out_o[k] = o; out_lnl[k] = lnl; out_post[k] = post;
                        out_bf[k] = bf; out_dh[k] = d; out_hh[k] = h;
                    }
                }
            } else if (isfinite(post)) {
                // online log-sum-exp of the accepted rows' ln_posterior
                if (post > wm) { ws = ws * __expf(wm - post) + 1.f; wm = post; }
                else ws += __expf(post - wm);
                // Gumbel-max phase draw, probability w_o / W; double precision (not fast-math)
                double u = u01d(splitmix64(hbase ^ ((unsigned long long)e << 9) ^ (unsigned long long)o));
                double g = (double)post - log(-log(u));
                if (obest < 0 || g > gbest) { gbest = g; obest = o; b_lnl = lnl; b_post = post; b_bf = bf; b_dh = d; b_hh = h; }
            }
        } else {
            if (disc_stride == 1 || ((p * nphi + o) % disc_stride) == 0) {
                float lnl = d > 0.f ? lnlike_marg(d, h, tx, ntx, ty, nty, cf, dmax, sigmas, z0) : 0.f;
                float x = lnl + lw;
                if (isfinite(x)) {
                    dn += 1.f;
                    if (x > dm) { ds1 = ds1 * __expf(dm - x) + 1.f; ds2 = ds2 * __expf(2.f * (dm - x)) + 1.f; dm = x; }
                    else { ds1 += __expf(x - dm); ds2 += __expf(2.f * (x - dm)); }
                }
            }
        }
    }
    pair_nacc[p] = nacc;
    pair_disc[3 * p] = dn;
    pair_disc[3 * p + 1] = dn > 0.f ? dm + __logf(ds1) : (-INF_F);
    pair_disc[3 * p + 2] = dn > 0.f ? 2.f * dm + __logf(ds2) : (-INF_F);
    if (mode == 1) {
        if (obest >= 0) {
            double lnw = (double)wm + log((double)ws) - ln_n_k;
            double u = u01d(splitmix64(hbase ^ ((unsigned long long)e << 9) ^ 511ULL));
            pair_lnw[p] = lnw;
            pair_key[p] = lnw - log(u);
            out_o[p] = obest; out_lnl[p] = b_lnl; out_post[p] = b_post; out_bf[p] = b_bf;
            out_dh[p] = b_dh; out_hh[p] = b_hh;
        } else {
            pair_lnw[p] = (-INF_F); pair_key[p] = (-INF_F);
        }
    }
}
""".replace("MIN_D", repr(float(MIN_D_LUMINOSITY)) + "f")


def running_thresholds(eb_max, max_diff):
    """Threshold per extrinsic block of thin_coherent.run_thin_iblock's growing cut.

    eb_max[j]: maximum valid best fit in extrinsic block j (over the i-block). The CPU raises
    its threshold to (maximum of the accepted rows - max_diff) after each block; when block j has
    accepted rows, their maximum is eb_max[j]. Returns thr[j], the threshold block j is cut at.
    """
    eb_max = np.asarray(eb_max, float)
    thr = np.full(len(eb_max), -np.inf)
    run = -np.inf
    for j, m in enumerate(eb_max):
        thr[j] = run
        if m > run:
            run = max(run, m - max_diff)
    return thr


def _logsumexp_cp(cp, x):
    x = x[cp.isfinite(x)]
    if x.size == 0:
        return -np.inf
    m = x.max()
    return float(m + cp.log(cp.exp(x - m).sum()))


class CoherentGPU:
    """Device state of one bank's coherent stage; ``run_iblock`` per intrinsic block.

    Parameters
    ----------
    setup : dict from ``dot_pe.thin_coherent.load_thin_setup`` (plus the keys mp_inference adds:
        ``row_selection``, ``ln_N_k``, ``selection_seed``, ``bank_index``, optional
        ``incoherent_lnlike_i``).
    blocksize_e : size of the extrinsic blocks of the running threshold (production: blocksize).
    """

    THREADS = 256

    def __init__(self, setup, blocksize_e=512, lookup=None):
        import cupy as cp
        self.cp = cp
        self.setup = setup
        self.row_selection = setup.get("row_selection", "bestfit")
        if self.row_selection not in ("bestfit", "phase_marginalized_priority"):
            raise ValueError(self.row_selection)
        self.use_priority = self.row_selection == "phase_marginalized_priority"
        if self.use_priority and setup.get("ln_N_k") is None:
            raise ValueError('row_selection="phase_marginalized_priority" needs setup["ln_N_k"]')
        self.ln_n_k = float(setup.get("ln_N_k") or 0.0)
        self.seed = int(setup.get("selection_seed") or 0) & 0xFFFFFFFFFFFFFFFF
        self.bank_index = int(setup.get("bank_index", 0))
        self.n_phi = int(setup["n_phi"])
        self.m_arr = np.asarray(setup["m_arr"])
        if len(self.m_arr) != 4:
            raise NotImplementedError("kernels are compiled for 4 modes")
        if self.n_phi >= 511:
            raise ValueError("n_phi must be < 511 (phase index uses 9 bits of the random-number counter)")
        self.max_diff = float(setup["max_bestfit_lnlike_diff"])
        self.max_per_block = setup.get("max_output_per_block")
        self.blocksize_e = int(blocksize_e)
        self.lookup = lookup or GPULookupTable()

        self.m_inds, self.mprime_inds = map(
            list, zip(*itertools.combinations_with_replacement(range(len(self.m_arr)), 2)))

        response = np.asarray(setup["response_dpe"], float)            # (d, p, e)
        timeshift = np.asarray(setup["timeshift_dbe"])                   # (d, b, e)
        asd = np.asarray(setup["asd_drift"], float)
        n_d, n_p, n_e = response.shape
        n_b = timeshift.shape[1]
        self.n_e = n_e
        # ext[x = (d, p, b), e] = conj(timeshift[d, b, e]) * response[d, p, e] / asd[d]^2
        ext = (np.conj(timeshift)[:, None, :, :]
               * (response * asd[:, None, None] ** -2)[:, :, None, :]).reshape(n_d * n_p * n_b, n_e)
        self.ext = cp.asarray(ext.astype(np.complex64))
        # ff[(d, p, P), e] = response[d, p, e] response[d, P, e] / asd[d]^2
        ff = np.einsum("dpe, dPe, d -> dpPe", response, response, asd ** -2).reshape(-1, n_e)
        self.ff = cp.asarray(ff.astype(np.complex64))
        self.dh_w = cp.asarray(np.asarray(setup["dh_weights_dmpb"], np.complex128))  # (d, m, p, b)
        self.hh_w = cp.asarray(np.asarray(setup["hh_weights_dmppb"], np.complex128))  # (d, mm, p, P, b)
        self.n_d, self.n_p, self.n_b = n_d, n_p, n_b
        self.fbin = cp.asarray(np.asarray(setup["fbin"], float))
        self.dt_linfree = np.asarray(setup["dt_linfree"], float)
        self.logw_i = np.asarray(setup["logw_i"], float)
        self.logw_e = cp.asarray(np.asarray(setup["log_prior_weights_e"], np.float32))
        inc = setup.get("incoherent_lnlike_i")
        self.incoherent_i = None if inc is None else np.asarray(inc, float)

        phi = np.linspace(0, 2 * np.pi, self.n_phi, endpoint=False)
        dp = np.exp(-1j * np.outer(self.m_arr, phi))
        hp = np.exp(1j * np.outer(self.m_arr[self.m_inds] - self.m_arr[self.mprime_inds], phi))
        self.dp = cp.asarray(dp.astype(np.complex64))
        self.hp = cp.asarray(hp.astype(np.complex64))

        mod = cp.RawModule(code=_KERNEL_SRC, options=("--use_fast_math",))
        self.k_pairmax = mod.get_function("k_pairmax")
        self.k_eval = mod.get_function("k_eval")
        self.shared_bytes = (4 + 10) * self.n_phi * 8
        self.profile = False   # True: synchronize and record stage times in self.timings
        self.debug = False     # True: keep per-pair device arrays in self.last
        self.timings = {}
        self.last = {}

    def _tick(self, name, t0):
        if not self.profile:
            return t0
        import time
        self.cp.cuda.Device().synchronize()
        t = time.perf_counter()
        self.timings[name] = self.timings.get(name, 0.0) + t - t0
        return t

    # ── per-block inner products ────────────────────────────────────────────
    def inner_products(self, i_block, amp=None, phase=None, h=None):
        """dh (4, n_i, n_e) and hh (10, n_i, n_e) complex64 on the device.

        Waveforms either as host amp, phase (the time shift ``dt_linfree`` is applied here) or as
        a device complex128 array h (i, m, p, b) with the time shift already applied.
        """
        cp = self.cp
        if h is None:
            dt = cp.asarray(self.dt_linfree[i_block])
            ph = cp.asarray(phase) + 2 * np.pi * dt[:, None, None, None] * self.fbin[None, None, None, :]
            h = cp.asarray(amp) * cp.exp(1j * ph)                          # (i, m, p, b) complex128
        n_i = h.shape[0]
        # A[m, i, (d, p, b)] = dh_w[d, m, p, b] * conj(h[i, m, p, b])  (float64, then complex64)
        a = (cp.transpose(self.dh_w, (1, 0, 2, 3))[:, None]                 # (m, 1, d, p, b)
             * cp.conj(cp.transpose(h, (1, 0, 2, 3)))[:, :, None])           # (m, i, 1, p, b)
        a = a.reshape(4, n_i, -1).astype(cp.complex64)
        dh = cp.matmul(a, self.ext)                                         # (m, i, e)
        hh_idmpP = cp.einsum("dmpPb, impb, imPb -> midpP", self.hh_w,
                             h[:, self.m_inds], cp.conj(h[:, self.mprime_inds]))
        hh = cp.matmul(hh_idmpP.reshape(10, n_i, -1).astype(cp.complex64), self.ff)  # (mm, i, e)
        return cp.ascontiguousarray(dh), cp.ascontiguousarray(hh)

    # ── one intrinsic block ─────────────────────────────────────────────────
    def run_iblock(self, i_block, amp=None, phase=None, return_device=False, h=None):
        import time
        cp = self.cp
        t0 = time.perf_counter()
        i_block = np.asarray(i_block, dtype=np.int64)
        n_i, n_e, n_phi = len(i_block), self.n_e, self.n_phi
        npairs = n_i * n_e
        dh, hh = self.inner_products(i_block, amp, phase, h)
        t0 = self._tick("inner_products", t0)
        bound = (np.full(n_i, np.inf) if self.incoherent_i is None
                 else self.incoherent_i[i_block] + INCOHERENT_BOUND_TOL)
        bound_d = cp.asarray(bound.astype(np.float32))
        grid = ((npairs + self.THREADS - 1) // self.THREADS,)
        blk = (self.THREADS,)

        # Pass 1: pair maxima; running threshold per extrinsic block.
        pmax = cp.empty(npairs, cp.float32)
        self.k_pairmax(grid, blk, (dh, hh, np.int32(n_i), np.int32(n_e), np.int32(n_phi),
                                   self.dp, self.hp, bound_d, pmax), shared_mem=self.shared_bytes)
        n_eb = (n_e + self.blocksize_e - 1) // self.blocksize_e
        pm = pmax.reshape(n_i, n_e)
        pad = n_eb * self.blocksize_e - n_e
        if pad:
            pm = cp.concatenate([pm, cp.full((n_i, pad), -cp.inf, cp.float32)], axis=1)
        eb_max = pm.reshape(n_i, n_eb, self.blocksize_e).max(axis=(0, 2)).get().astype(float)
        thr = running_thresholds(eb_max, self.max_diff)
        thr_d = cp.asarray(thr.astype(np.float32))
        t0 = self._tick("pass1_threshold", t0)
        thr_pair = cp.repeat(thr_d, self.blocksize_e)[:n_e][None, :]
        accepted_pair = pm[:, :n_e] > thr_pair if pad else pm > thr_pair
        n_acc_pairs = int(accepted_pair.sum())

        k = self.max_per_block
        if self.use_priority or k is None:
            emit_cut = -np.inf
        else:
            # The block's k-th best accepted row is >= the k-th largest accepted pair maximum.
            acc_max = cp.where(accepted_pair, pm[:, :n_e], -cp.inf).ravel()
            emit_cut = (-np.inf if n_acc_pairs <= k
                        else float(cp.partition(acc_max, npairs - k)[npairs - k]))
        if np.isfinite(emit_cut):
            emit_cut -= EMIT_MARGIN
            emit_cap = int((acc_max >= emit_cut - EMIT_MARGIN).sum()) * n_phi
        else:
            emit_cap = n_acc_pairs * n_phi
        if not self.use_priority and emit_cap > MAX_EMIT_ROWS:
            raise RuntimeError(f"bestfit mode would emit up to {emit_cap} rows (> {MAX_EMIT_ROWS}); "
                               "set max_output_per_block")
        cap = npairs if self.use_priority else max(1, emit_cap)

        ln_ = lambda dtype: cp.empty(cap, dtype)
        out_i, out_e, out_o = ln_(cp.int32), ln_(cp.int32), ln_(cp.int32)
        out_lnl, out_post, out_bf, out_dh, out_hh = (ln_(cp.float32) for _ in range(5))
        pair_lnw = cp.empty(npairs if self.use_priority else 1, cp.float64)
        pair_key = cp.empty(npairs if self.use_priority else 1, cp.float64)
        pair_nacc = cp.empty(npairs, cp.int32)
        pair_disc = cp.empty(3 * npairs, cp.float32)
        emit_count = cp.zeros(1, cp.int32)
        n_rows = npairs * n_phi
        disc_stride = DISC_STRIDE
        logw_i = cp.asarray(self.logw_i[i_block].astype(np.float32))
        ibank = cp.asarray(i_block)
        self.k_eval(grid, blk, (
            dh, hh, np.int32(n_i), np.int32(n_e), np.int32(n_phi), self.dp, self.hp, bound_d,
            thr_d, np.int32(self.blocksize_e), logw_i, self.logw_e, ibank,
            np.uint64(self.seed), np.int32(self.bank_index), np.float64(self.ln_n_k),
            np.int64(disc_stride), np.int32(1 if self.use_priority else 0), np.float32(emit_cut),
            emit_count, np.int32(cap), out_i, out_e, out_o, out_lnl, out_post, out_bf, out_dh,
            out_hh, pair_lnw, pair_key, pair_nacc, pair_disc, *self.lookup.kernel_args(),
        ), shared_mem=self.shared_bytes)

        t0 = self._tick("pass2_eval", t0)
        if self.debug:
            self.last = {"thr": thr, "pair_lnw": pair_lnw, "pair_key": pair_key,
                         "pair_nacc": pair_nacc, "pmax": pmax, "emit_cut": emit_cut,
                         "out_o": out_o}
        n_dist_marg = int(pair_nacc.sum())
        n_disc = n_rows - n_dist_marg
        disc = pair_disc.reshape(npairs, 3)
        n_sampled = float(disc[:, 0].sum())
        if n_disc and n_sampled > 0:
            sf = np.log(n_disc / n_sampled)
            lse_disc = _logsumexp_cp(cp, disc[:, 1].astype(cp.float64)) + sf
            lsse_disc = _logsumexp_cp(cp, disc[:, 2].astype(cp.float64)) + sf
        else:
            lse_disc = lsse_disc = -np.inf

        tau_blk = -np.inf
        if self.use_priority:
            sel = cp.flatnonzero(cp.isfinite(pair_key))
            if k is not None and len(sel) > k:
                keys = pair_key[sel]
                order = cp.argsort(-keys)
                tau_blk = float(keys[order[k]])
                sel = sel[order[:k]]
            cols = {
                "i": ibank[sel // n_e], "e": (sel % n_e).astype(cp.int64), "o": out_o[sel],
                "lnl_marginalized": out_lnl[sel], "ln_posterior": out_post[sel],
                "bestfit_lnlike": out_bf[sel], "d_h_1Mpc": out_dh[sel], "h_h_1Mpc": out_hh[sel],
                "ln_phase_marginalized_weight": pair_lnw[sel], "priority_key": pair_key[sel],
            }
            cols["priority_tau"] = cp.full(len(sel), tau_blk)
        else:
            n_emit = int(emit_count.get()[0])
            if n_emit > cap:
                raise RuntimeError(f"emit buffer overflow: {n_emit} > {cap}")
            if np.isfinite(emit_cut) and n_emit < k:
                raise RuntimeError(f"emitted {n_emit} rows < k = {k} above emit_cut {emit_cut}")
            sel = cp.arange(n_emit)
            if k is not None and n_emit > k:
                sel = cp.argsort(-out_bf[:n_emit])[:k]
            cols = {
                "i": ibank[out_i[sel]], "e": out_e[sel].astype(cp.int64), "o": out_o[sel],
                "lnl_marginalized": out_lnl[sel], "ln_posterior": out_post[sel],
                "bestfit_lnlike": out_bf[sel], "d_h_1Mpc": out_dh[sel], "h_h_1Mpc": out_hh[sel],
            }
        t0 = self._tick("select", t0)
        if return_device:
            return cols, n_disc, lse_disc, lsse_disc, n_dist_marg
        import pandas as pd
        df = pd.DataFrame({c: (v.get().astype(np.int64) if c in ("i", "e", "o")
                               else v.get().astype(np.float64)) for c, v in cols.items()})
        cached_dt_slice = {int(idx): float(self.dt_linfree[idx]) for idx in i_block}
        self._tick("to_host", t0)
        return df, n_disc, lse_disc, lsse_disc, cached_dt_slice, n_dist_marg
