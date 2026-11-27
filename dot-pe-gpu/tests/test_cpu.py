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
