#!/usr/bin/env python
"""Smoke check of row_thinning="pairs" in the production worker on real data (not a report).

One i-block of GW240525_201644 run_2, a few extrinsic blocks, no row caps: the pairs mode must
return one row per (i, e) pair and the sum of its pair weights must equal the sum of the row
weights of the bestfit mode over the same accepted rows.
"""

from pathlib import Path

import numpy as np

from dot_pe import thin_coherent
from dot_pe.utils import inds_to_blocks, safe_logsumexp

RUN = Path("/home/projects/barakz/Collaboration-gw/mushkin/dot-pe/o4b-pe-paper/"
           "artifacts/pe/dotpe/GW240525_201644/run_2")
BANK = "/home/projects/barakz/Collaboration-gw/mushkin/dot-pe/o4b-pe-paper/artifacts/banks/tiled/bank_56"


def main():
    setup = thin_coherent.load_thin_setup(RUN / "mp_coherent_setup_bank_0", RUN)
    setup["max_output_per_block"] = None
    inds = np.load(RUN / "banks/bank_0/intrinsic_samples.npz")["inds"]
    i_block = inds_to_blocks(inds, 512)[74]
    e_blocks = inds_to_blocks(np.arange(13312), 512)[:4]
    ln_n = np.log(100 * 13312 * 1048576)
    out = {}
    for mode in ("bestfit", "pairs"):
        s = dict(setup, row_thinning=mode, ln_N_k=ln_n, thinning_seed=1)
        df, *_ = thin_coherent.run_thin_iblock(i_block, e_blocks, s, Path(BANK) / "waveforms")
        out[mode] = df
        print(mode, len(df), "rows")
    b, p = out["bestfit"], out["pairs"]
    n_pairs = len(b[["i", "e"]].drop_duplicates())
    lz_b = safe_logsumexp(b["ln_posterior"].to_numpy())
    lz_p = safe_logsumexp(p["ln_pair_weight"].to_numpy() + ln_n)
    print(f"pairs in bestfit output {n_pairs}, rows in pairs output {len(p)}")
    print(f"ln sum w: bestfit {lz_b:.10f}, pairs {lz_p:.10f}, diff {lz_b - lz_p:.2e}")
    assert len(p) == n_pairs and abs(lz_b - lz_p) < 1e-9
    assert np.all(np.isneginf(p["thin_tau"]))
    print("smoke ok")


if __name__ == "__main__":
    main()
