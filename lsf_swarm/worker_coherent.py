#!/usr/bin/env python3
"""
Thin LSF array worker: coherent inference for one i_block × all e_blocks.

Unlike the previous thick worker, this process does NOT instantiate
CoherentLikelihoodProcessor or LinearFree.  All expensive pre-computation
(summary weights, dt_linfree cache) is done once by run_swarm.py Stage 2.5
and stored under {rundir}/swarm_setup/coherent_setup/.

Per-worker memory: ~1–2 GB (weights + extrinsic arrays + one waveform block).

Output
------
{rundir}/swarm_setup/coherent/i_{i_block_idx}.npz
    prob_samples columns: i, e, o, lnl_marginalized, ln_posterior,
                          bestfit_lnlike, d_h_1Mpc, h_h_1Mpc
    discarded stats: n_samples_discarded, logsumexp_discarded,
                     logsumsqrexp_discarded, n_distance_marginalizations
    cached timeshifts: cached_dt_keys, cached_dt_vals
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dot_pe import thin_coherent
from dot_pe.utils import inds_to_blocks


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--rundir", required=True)
    p.add_argument(
        "--i-block-idx",
        type=int,
        default=None,
        help="1-indexed i_block index (default: $LSB_JOBINDEX)",
    )
    args = p.parse_args()

    rundir = Path(args.rundir)
    i_block_idx = (
        args.i_block_idx
        if args.i_block_idx is not None
        else int(os.environ["LSB_JOBINDEX"])
    )

    out_path = rundir / "swarm_setup" / "coherent" / f"i_{i_block_idx}.npz"
    if out_path.exists():
        print(f"[skip] {out_path} already exists")
        return

    swarm_dir = rundir / "swarm_setup"
    setup_dir = swarm_dir / "coherent_setup"

    with open(swarm_dir / "swarm_config.json") as f:
        cfg = json.load(f)

    blocksize = cfg["blocksize"]
    n_ext = cfg["n_ext"]
    waveform_dir = Path(cfg["bank_folder"]) / "waveforms"

    selected_inds = np.load(swarm_dir / "selected_inds.npy")
    i_blocks = inds_to_blocks(selected_inds, blocksize)
    e_blocks = inds_to_blocks(np.arange(n_ext), blocksize)

    if i_block_idx < 1 or i_block_idx > len(i_blocks):
        raise ValueError(
            f"i_block_idx {i_block_idx} out of range [1, {len(i_blocks)}]"
        )

    i_block = i_blocks[i_block_idx - 1]

    setup = thin_coherent.load_thin_setup(setup_dir, rundir)

    ps, n_disc, logsumexp_disc, logsumsqrexp_disc, cached_dt, n_dist_marg = (
        thin_coherent.run_thin_iblock(i_block, e_blocks, setup, waveform_dir)
    )

    def _col(df, col, dtype):
        return df[col].values.astype(dtype) if len(df) else np.array([], dtype=dtype)

    np.savez(
        out_path,
        i                           = _col(ps, "i", np.int64),
        e                           = _col(ps, "e", np.int64),
        o                           = _col(ps, "o", np.int64),
        lnl_marginalized            = _col(ps, "lnl_marginalized", float),
        ln_posterior                = _col(ps, "ln_posterior", float),
        bestfit_lnlike              = _col(ps, "bestfit_lnlike", float),
        d_h_1Mpc                    = _col(ps, "d_h_1Mpc", float),
        h_h_1Mpc                    = _col(ps, "h_h_1Mpc", float),
        n_samples_discarded         = np.array(n_disc),
        logsumexp_discarded         = np.array(logsumexp_disc),
        logsumsqrexp_discarded      = np.array(logsumsqrexp_disc),
        n_distance_marginalizations = np.array(n_dist_marg),
        cached_dt_keys              = np.array(list(cached_dt.keys()), dtype=np.int64),
        cached_dt_vals              = np.array(list(cached_dt.values()), dtype=float),
    )
    print(f"[done] i_block {i_block_idx}: {len(ps)} samples → {out_path}")


if __name__ == "__main__":
    main()
