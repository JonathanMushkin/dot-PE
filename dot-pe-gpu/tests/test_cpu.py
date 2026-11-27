"""CPU tests of dot_pe_gpu (no GPU needed): run with pytest from any directory."""

import re
from pathlib import Path

import numpy as np
import pytest

from dot_pe_gpu.coherent import _KERNEL_SRC, running_thresholds
from dot_pe_gpu.lookup_table import lnlike_marginalized_numpy, table_arrays, bispev_numpy


def test_lookup_numpy_port_matches_cogwheel():
    from cogwheel.likelihood import LookupTable
    lt = LookupTable()
    tb = table_arrays(lt)
    rng = np.random.default_rng(1)
    norm_h = 10 ** rng.uniform(0.5, 7.5, 100_000)
    ov = rng.uniform(-40, 120, 100_000)
    dh, hh = ov * norm_h, norm_h ** 2
    np.testing.assert_allclose(lnlike_marginalized_numpy(dh, hh, tb), lt.lnlike_marginalized(dh, hh),
                               rtol=1e-12, atol=1e-9)
    x, y = rng.uniform(-14, 3, 1000), rng.uniform(-0.7, 1.2, 1000)  # out of range: clamped
    np.testing.assert_allclose(bispev_numpy(*tb[:3], x, y), lt._interpolated_table(x, y, grid=False),
                               atol=1e-10)


def _cpu_accepted(bf_ieo, e_size, max_diff):
    """The acceptance loop of thin_coherent.run_thin_iblock, on a best-fit array."""
    min_bestfit = -np.inf
    acc = np.zeros(bf_ieo.shape, bool)
    for j0 in range(0, bf_ieo.shape[1], e_size):
        blk = bf_ieo[:, j0:j0 + e_size]
        a = blk > min_bestfit
        acc[:, j0:j0 + e_size] = a
        if np.any(a):
            cand = float(blk[a].max()) - max_diff
            if cand > min_bestfit:
                min_bestfit = cand
    return acc


@pytest.mark.parametrize("seed", range(20))
def test_running_thresholds_match_cpu_loop(seed):
    rng = np.random.default_rng(seed)
    n_i, n_e, n_o, e_size = 7, 53, 5, 8
    bf = rng.normal(0, 15, (n_i, n_e, n_o)) + rng.normal(0, 20, n_e)[None, :, None]
    bf[rng.random(bf.shape) < 0.2] = -np.inf          # masked rows
    bf[:, rng.integers(0, n_e, 3)] = -np.inf          # an empty extrinsic sample
    if seed % 4 == 0:
        bf[:, :e_size] = -np.inf                      # an empty first extrinsic block
    ref = _cpu_accepted(bf, e_size, 20.0)
    n_eb = -(-n_e // e_size)
    pad = np.full((n_i, n_eb * e_size - n_e, n_o), -np.inf)
    eb_max = np.concatenate([bf, pad], axis=1).reshape(n_i, n_eb, e_size, n_o).max(axis=(0, 2, 3))
    thr = running_thresholds(eb_max, 20.0)
    got = bf > np.repeat(thr, e_size)[:n_e][None, :, None]
    assert np.array_equal(ref, got)


def test_uniform_draw_ranges():
    m = re.search(r"\(\(double\)\(h >> (\d+)\) \+ 0\.5\) \* ([0-9.e-]+)", _KERNEL_SRC)
    shift, scale = int(m.group(1)), float(m.group(2))
    assert scale == 2.0 ** -(64 - shift)
    lo, hi = (0 + 0.5) * scale, ((2 ** 64 - 1 >> shift) + 0.5) * scale
    assert 0 < lo and hi < 1  # strictly inside (0, 1): -log(-log u) and -log u are finite


def test_package_imports_without_gpu():
    import dot_pe_gpu.bank, dot_pe_gpu.incoherent, dot_pe_gpu.stages  # noqa: F401
    assert Path(dot_pe_gpu.bank.__file__).exists()


# ── P3: incoherent stage ────────────────────────────────────────────────────


def _random_waveforms(rng, n_i=16, n_b=40, amp=1e-20):
    fbin = np.linspace(20, 500, n_b)
    t0 = rng.normal(0, 0.01, n_i)
    phase = (2 * np.pi * fbin[None, None, None, :] * t0[:, None, None, None]
             + rng.normal(0, 0.3, (n_i, 4, 2, n_b)).cumsum(-1) * 0.05)
    amps = amp * rng.uniform(0.2, 1.0, (n_i, 4, 2, 1)) * (fbin / 100.0) ** (-7 / 6)
    return fbin, amps * np.exp(1j * phase)


def test_linfree_dt_matches_polynomial_fit():
    from dot_pe_gpu.incoherent import linfree_dt
    rng = np.random.default_rng(3)
    fbin, h = _random_waveforms(rng)
    h2plus0 = h[0, 0, 0] * np.exp(1j * rng.normal(0, 0.01, len(fbin)))
    w = rng.uniform(0.1, 1.0, len(fbin))
    got = linfree_dt(np, h[:, 0, 0], h2plus0, w, fbin)
    ratio = h[:, 0, 0] / h2plus0
    ref = [-np.polynomial.Polynomial.fit(fbin, np.unwrap(np.angle(r)), 1, w=w * np.sqrt(np.abs(r))).deriv()(0)
           / (2 * np.pi) for r in ratio]
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_single_detector_mirror_matches_dot_pe(seed):
    from dot_pe.single_detector import SingleDetectorProcessor
    from dot_pe_gpu.incoherent import single_det_lnlike_numpy
    rng = np.random.default_rng(seed)
    fbin, h = _random_waveforms(rng, n_i=24)
    n_b, n_t, n_phi = len(fbin), 16, 12
    psd_w = 1e40 * rng.uniform(0.5, 1.5, n_b)
    dh_w = (psd_w * np.exp(1j * rng.uniform(0, 2 * np.pi, (4, 2, n_b))) * 1e-1)
    hh_w = np.broadcast_to(psd_w, (10, 2, 2, n_b)).astype(complex) * 1e18
    if seed == 4:
        h[:, :, 1] = h[:, :, 0] * (1 + 1e-6)  # near-degenerate polarizations
    t = np.linspace(-0.01, 0.01, n_t)
    ts = np.exp(-2j * np.pi * fbin[:, None] * t[None])
    m_arr = np.array([2, 1, 3, 4])
    _, lnl_iot = SingleDetectorProcessor.get_response_over_distance_and_lnlike(
        dh_w[None], hh_w[None], h, ts[None], np.array([1.0]), n_phi, m_arr)
    ref = lnl_iot.max(axis=(1, 2))
    got64 = single_det_lnlike_numpy(dh_w, hh_w, h, np.conj(ts), n_phi, m_arr, dtype=np.float64)
    np.testing.assert_array_equal(np.isfinite(ref), np.isfinite(got64))
    fin = np.isfinite(ref)
    np.testing.assert_allclose(got64[fin], ref[fin], rtol=1e-9)
    got32 = single_det_lnlike_numpy(dh_w, hh_w, h, np.conj(ts), n_phi, m_arr, dtype=np.float32)
    np.testing.assert_array_equal(fin, np.isfinite(got32))  # same masked templates in float32
    if fin.any():
        assert np.abs(got32[fin] - ref[fin]).max() <= 1e-3 * max(1.0, np.abs(ref[fin]).max())


def test_residency_margin_top_n_and_finalize():
    from dot_pe_gpu.stages import Residency
    rng = np.random.default_rng(7)
    res = Residency(margin=20.0, top_n=5, xp=np)
    store = {}
    for bank, lo in [("b0", 0.0), ("b1", 30.0)]:
        for s in range(3):
            inds = np.arange(s * 50, (s + 1) * 50)
            lnl = rng.uniform(lo, lo + 15, 50)
            h = (rng.normal(size=(50, 4, 2, 8)) + 1j * rng.normal(size=(50, 4, 2, 8))) * 1e-20
            res.offer(bank, inds, lnl, h)
            store.update({(bank, i): (l, hh) for i, l, hh in zip(inds, lnl, h)})
    gmax = max(l for l, _ in store.values())
    # relative threshold: all b1 templates within 20 of the max; none of b0 (max 15 < 45 - 20)
    sel1 = np.array(sorted(i for (b, i), (l, _) in store.items() if b == "b1" and l >= gmax - 20))
    out = res.finalize("b1", sel1)
    ref = np.array([store[("b1", i)][1] for i in sel1])
    np.testing.assert_allclose(out, ref.astype(np.complex64).astype(np.complex128), rtol=1e-6, atol=0)
    # the fallback set (top 5 overall) is in b1; b0 must raise for anything pruned
    with pytest.raises(RuntimeError):
        res.finalize("b0", np.array([0]))
    # top-N fallback when the maximum is anomalous: a lone spike in a new bank
    res2 = Residency(margin=20.0, top_n=3, xp=np)
    res2.offer("a", np.arange(10), np.arange(10.0), np.ones((10, 4, 2, 8), complex) * 1e-20)
    res2.offer("b", np.array([0]), np.array([1e5]), np.ones((1, 4, 2, 8), complex) * 1e-20)
    got = res2.finalize("a", np.array([8, 9]))  # the stage-3 fallback keeps the top of bank a
    assert got.shape == (2, 4, 2, 8) and np.allclose(np.abs(got), 1e-20, rtol=1e-6)


def test_residency_cap_overflow():
    from dot_pe_gpu.stages import Residency
    res = Residency(margin=20.0, top_n=3, xp=np, max_bytes=50 * 4 * 2 * 8 * 8 * 1.5)  # room for ~75 rows
    h = np.ones((50, 4, 2, 8), complex) * 1e-20
    res.offer("a", np.arange(50), np.zeros(50), h)
    assert not res.overflowed
    res.offer("a", np.arange(50, 100), np.zeros(50), h)
    assert res.overflowed and res.nbytes() == 0
    assert res.finalize("a", np.arange(10)) is None
    res.offer("a", np.arange(100, 150), np.zeros(50), h)  # ignored after overflow
    assert res.nbytes() == 0


def test_iter_selected_chunks(tmp_path):
    import json
    from dot_pe_gpu.bank import iter_selected
    rng = np.random.default_rng(2)
    wdir = tmp_path / "waveforms"
    wdir.mkdir()
    blocksize, n_files = 16, 5
    json.dump({"blocksize": blocksize}, open(tmp_path / "bank_config.json", "w"))
    amps, phases = [], []
    for k in range(n_files):
        a, p = rng.uniform(0.5, 1, (blocksize, 4, 2, 6)) * 1e-20, rng.uniform(-3, 3, (blocksize, 4, 2, 6))
        np.save(wdir / f"amplitudes_block_{k}.npy", a)
        np.save(wdir / f"phase_block_{k}.npy", p)
        amps.append(a)
        phases.append(p)
    amp, phase = np.concatenate(amps), np.concatenate(phases)
    inds = np.sort(rng.choice(blocksize * n_files, 37, replace=False))
    got = list(iter_selected(tmp_path, inds, chunk=8, xp=np, threads=2))
    assert [s for s, _ in got] == list(range(0, 37, 8))
    h = np.concatenate([c for _, c in got]).astype(np.complex128) / 2.0 ** 64
    np.testing.assert_allclose(h, amp[inds] * np.exp(1j * phase[inds]), rtol=1e-6)
