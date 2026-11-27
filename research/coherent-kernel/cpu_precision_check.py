"""Pre-job check (review finding 26): complex64 inner products, emulated in numpy, against the
float64 CPU path, on block 0 and the first `n_e` extrinsic samples.

Emulates CoherentGPU.inner_products (products in complex128, cast to complex64, complex64
matmuls) and the float32 phase combination; reports |d bestfit| and |d lnl| over rows within
20 of the maximum best fit. Usage: python research/coherent-kernel/cpu_precision_check.py [n_e]
"""

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from dot_pe.likelihood_calculating import LikelihoodCalculator, invalid_bestfit_mask  # noqa: E402


def main(n_e=256):
    s = common.setup_for("bestfit", False)
    b = np.load(common.OUT / "ref/block0_inds.npy")
    amp = np.load(common.OUT / "ref/block0_amp.npy")
    phase = np.load(common.OUT / "ref/block0_phase.npy")
    dt = s["dt_linfree"][b]
    h = amp * np.exp(1j * (phase + 2 * np.pi * dt[:, None, None, None] * s["fbin"]))
    R, T, asd = s["response_dpe"][..., :n_e], s["timeshift_dbe"][..., :n_e], s["asd_drift"]
    lc = LikelihoodCalculator(s["n_phi"], s["m_arr"])
    dh64, hh64 = lc.get_dh_hh_ieo(s["dh_weights_dmpb"], h, R, T, s["hh_weights_dmppb"], asd)

    # complex64 emulation
    m_inds, mp_inds = map(list, zip(*itertools.combinations_with_replacement(range(4), 2)))
    ext = (np.conj(T)[:, None] * (R * asd[:, None, None] ** -2)[:, :, None]).reshape(-1, n_e).astype(np.complex64)
    ff = np.einsum("dpe, dPe, d -> dpPe", R, R, asd ** -2).reshape(-1, n_e).astype(np.complex64)
    a = (np.transpose(s["dh_weights_dmpb"], (1, 0, 2, 3))[:, None]
         * np.conj(np.transpose(h, (1, 0, 2, 3)))[:, :, None]).reshape(4, len(b), -1).astype(np.complex64)
    dh_m = np.matmul(a, ext)                                                   # (m, i, e)
    hh_x = np.einsum("dmpPb, impb, imPb -> midpP", s["hh_weights_dmppb"], h[:, m_inds], np.conj(h[:, mp_inds]))
    hh_m = np.matmul(hh_x.reshape(10, len(b), -1).astype(np.complex64), ff)   # (mm, i, e)
    phi = np.linspace(0, 2 * np.pi, s["n_phi"], endpoint=False)
    m = np.asarray(s["m_arr"])
    dp = np.exp(-1j * np.outer(m, phi)).astype(np.complex64)
    hp = np.exp(1j * np.outer(m[m_inds] - m[mp_inds], phi)).astype(np.complex64)
    dh32 = np.einsum("mie, mo -> ieo", dh_m, dp).real.astype(np.float32)
    hh32 = np.einsum("mie, mo -> ieo", hh_m, hp).real.astype(np.float32)

    bf = lambda d, hh: np.where(invalid_bestfit_mask(d, hh), -np.inf, 0.5 * d ** 2 / hh * (d > 0))
    b64, b32 = bf(dh64, hh64), bf(dh32.astype(float), hh32.astype(float))
    top = b64 > np.nanmax(b64) - 20
    lnl64 = lc.lookup_table.lnlike_marginalized(dh64[top], hh64[top])
    lnl32 = lc.lookup_table.lnlike_marginalized(dh32[top].astype(float), hh32[top].astype(float))
    print(f"rows within 20 of max: {top.sum()}; max bestfit {np.nanmax(b64):.3f}")
    print(f"max |d bestfit| {np.abs(b64[top] - b32[top]).max():.2e}; "
          f"p99 {np.percentile(np.abs(b64[top] - b32[top]), 99):.2e}")
    print(f"max |d lnl| {np.abs(lnl64 - lnl32).max():.2e}")
    print(f"max rel d dh {np.abs((dh64[top] - dh32[top]) / dh64[top]).max():.2e}, "
          f"rel d hh {np.abs((hh64[top] - hh32[top]) / hh64[top]).max():.2e}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 256)
