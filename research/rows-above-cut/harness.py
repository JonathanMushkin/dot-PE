#!/usr/bin/env python
"""E1 harness: rows of GW240525_201644 run_2 with bestfit_lnlike >= the nominal cut, per i-block.

Recomputes run_2's coherent stage for selected i-blocks with the production worker
`dot_pe.thin_coherent.run_thin_iblock`, on the setup run_2 wrote, with no per-block row cap.
Design: research/rows-above-cut/design.md.

Usage:
    python harness.py --plan                 # write the stratified block sample (no compute)
    python harness.py --blocks 0 1 2         # compute sample positions 0, 1, 2
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dot_pe import thin_coherent
from dot_pe.utils import inds_to_blocks

RUN = Path(
    "/home/projects/barakz/Collaboration-gw/mushkin/dot-pe/o4b-pe-paper/"
    "artifacts/pe/dotpe/GW240525_201644/run_2"
)
OUT = Path(__file__).resolve().parents[2] / "artifacts/research/rows-above-cut"
BEST_FIT = 35.28718742929617  # run_2 summary_results.json bestfit_lnlike_max
CUT = BEST_FIT - 20.0
FLOOR = 24.35
BLOCKSIZE = 512
N_EXT = 13312
SEED = 20261122
MAX_PER_BANK = 40
BANKS = ("bank_0", "bank_1", "bank_2")


def bank_blocks(bank: str) -> list[np.ndarray]:
    inds = np.load(RUN / f"banks/{bank}/intrinsic_samples.npz")["inds"]
    return inds_to_blocks(inds, BLOCKSIZE)


def plan() -> pd.DataFrame:
    """Stratified random order of i-blocks: position p holds bank p % 3, interleaved."""
    rng = np.random.default_rng(SEED)
    per_bank = {}
    for b in BANKS:
        n = len(bank_blocks(b))
        per_bank[b] = (n, rng.permutation(n)[:MAX_PER_BANK])
    rows = []
    for r in range(MAX_PER_BANK):
        for b in BANKS:
            n, order = per_bank[b]
            rows.append({"pos": len(rows), "bank": b, "block": int(order[r]), "B_k": n})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "plan.csv", index=False)
    return df


def run_block(bank: str, block: int, kw: dict) -> dict:
    t0 = time.time()
    k = BANKS.index(bank)
    setup = thin_coherent.load_thin_setup(RUN / f"mp_coherent_setup_{bank}", RUN)
    setup["max_output_per_block"] = None
    i_block = bank_blocks(bank)[block]
    e_blocks = inds_to_blocks(np.arange(N_EXT), BLOCKSIZE)
    waveform_dir = Path(kw["bank_folder"][k]) / "waveforms"
    # One extrinsic block per call bounds memory; each call's running threshold is <= CUT.
    parts, n_returned, n_disc = [], 0, 0
    for e_block in e_blocks:
        df, nd, *_ = thin_coherent.run_thin_iblock(i_block, [e_block], setup, waveform_dir)
        n_returned += len(df)
        n_disc += nd
        parts.append(df[df["bestfit_lnlike"].to_numpy() >= CUT])
    keep = pd.concat(parts, ignore_index=True)
    del parts
    out = pd.DataFrame({
        "i": keep["i"].to_numpy(np.int32),
        "e": keep["e"].to_numpy(np.int16),
        "o": keep["o"].to_numpy(np.int8),
        "bestfit_lnlike": keep["bestfit_lnlike"].to_numpy(np.float32),
        "lnl_marginalized": keep["lnl_marginalized"].to_numpy(np.float32),
        "ln_posterior": keep["ln_posterior"].to_numpy(np.float64),
        "d_h_1Mpc": keep["d_h_1Mpc"].to_numpy(np.float32),
        "h_h_1Mpc": keep["h_h_1Mpc"].to_numpy(np.float32),
    })
    (OUT / "blocks").mkdir(parents=True, exist_ok=True)
    out.to_feather(OUT / f"blocks/{bank}_{block}.feather")

    # Harness validation: every run_2 row of this block is in the harness output.
    r2 = pd.read_feather(
        RUN / f"banks/{bank}/prob_samples.feather",
        columns=["i", "e", "o", "bestfit_lnlike"],
    )
    r2 = r2[np.isin(r2["i"].to_numpy(), i_block)]
    m = r2.merge(keep[["i", "e", "o", "bestfit_lnlike"]], on=["i", "e", "o"],
                 how="left", suffixes=("_run2", "_h"))
    found = m["bestfit_lnlike_h"].notna()
    rel = np.abs(m["bestfit_lnlike_h"] - m["bestfit_lnlike_run2"]) / m["bestfit_lnlike_run2"]
    # Missing run_2 rows: explained if the newer masks (invalid_bestfit_mask) remove them.
    n_missing_masked = 0
    if (~found).any():
        from dot_pe.likelihood_calculating import invalid_bestfit_mask

        miss = r2[~found.to_numpy()].merge(
            pd.read_feather(RUN / f"banks/{bank}/prob_samples.feather",
                            columns=["i", "e", "o", "d_h_1Mpc", "h_h_1Mpc"]),
            on=["i", "e", "o"])
        n_missing_masked = int(invalid_bestfit_mask(
            miss["d_h_1Mpc"].to_numpy(), miss["h_h_1Mpc"].to_numpy()).sum())
    b = out["bestfit_lnlike"].to_numpy()
    return {
        "bank": bank, "block": block, "n_templates": len(i_block),
        "n_rows_total": len(i_block) * N_EXT * setup["n_phi"],
        "n_returned": n_returned, "n_disc": int(n_disc),
        "N_cut": int((b >= CUT).sum()), "N_floor": int((b >= FLOOR).sum()),
        "b_max": float(b.max()) if len(b) else np.nan,
        "run2_rows": len(r2), "run2_found": int(found.sum()),
        "run2_missing_masked": n_missing_masked,
        "validation_ok": bool(len(r2) - int(found.sum()) == n_missing_masked
                              and (not found.any() or float(rel[found].max()) <= 1e-5)),
        "run2_max_rel_diff": float(rel[found].max()) if found.any() else np.nan,
        "wall_s": time.time() - t0,
        "maxrss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--blocks", type=int, nargs="*", default=[])
    a = ap.parse_args()
    if a.plan:
        print(plan().head(9).to_string())
        return
    pl = pd.read_csv(OUT / "plan.csv").set_index("pos")
    kw = json.loads((RUN / "run_kwargs.json").read_text())
    (OUT / "counts").mkdir(parents=True, exist_ok=True)
    for p in a.blocks:
        r = run_block(pl.loc[p, "bank"], int(pl.loc[p, "block"]), kw)
        r.update({"pos": p, "B_k": int(pl.loc[p, "B_k"])})
        pd.DataFrame([r]).to_csv(OUT / f"counts/pos_{p:03d}.csv", index=False)
        print(json.dumps(r))


if __name__ == "__main__":
    main()
