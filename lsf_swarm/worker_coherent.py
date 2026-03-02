#!/usr/bin/env python3
"""
LSF array worker: processes one i_block × all e_blocks for coherent inference.

Called by run_swarm.py via bsub.  The i_block index comes from $LSB_JOBINDEX
(1-indexed) or --i-block-idx on the CLI.

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
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cogwheel.waveform import WaveformGenerator
from dot_pe.coherent_processing import CoherentLikelihoodProcessor
from dot_pe.likelihood_calculating import LinearFree
from dot_pe.utils import inds_to_blocks


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rundir",      required=True)
    p.add_argument("--i-block-idx", type=int, default=None,
                   help="1-indexed i_block index (default: $LSB_JOBINDEX)")
    args = p.parse_args()

    rundir      = Path(args.rundir)
    i_block_idx = args.i_block_idx if args.i_block_idx is not None else int(os.environ["LSB_JOBINDEX"])

    out_path = rundir / "swarm_setup" / "coherent" / f"i_{i_block_idx}.npz"
    if out_path.exists():
        print(f"[skip] {out_path} already exists")
        return

    setup_dir = rundir / "swarm_setup"

    with open(setup_dir / "event_data.pkl", "rb") as f:
        event_data = pickle.load(f)
    with open(setup_dir / "par_dic_0.json") as f:
        par_dic_0 = json.load(f)
    with open(setup_dir / "swarm_config.json") as f:
        cfg = json.load(f)

    bank_folder             = Path(cfg["bank_folder"])
    fbin                    = np.array(cfg["fbin"])
    approximant             = cfg["approximant"]
    m_arr                   = np.array(cfg["m_arr"])
    n_phi                   = cfg["n_phi"]
    n_ext                   = cfg["n_ext"]
    blocksize               = cfg["blocksize"]
    size_limit              = cfg.get("size_limit", 10**7)
    max_bestfit_lnlike_diff = cfg.get("max_bestfit_lnlike_diff", 20.0)

    selected_inds = np.load(setup_dir / "selected_inds.npy")
    i_blocks      = inds_to_blocks(selected_inds, blocksize)

    if i_block_idx < 1 or i_block_idx > len(i_blocks):
        raise ValueError(f"i_block_idx {i_block_idx} out of range [1, {len(i_blocks)}]")

    i_block  = i_blocks[i_block_idx - 1]
    e_blocks = inds_to_blocks(np.arange(n_ext), blocksize)

    bank_file_path = bank_folder / "intrinsic_sample_bank.feather"
    waveform_dir   = bank_folder / "waveforms"

    wfg               = WaveformGenerator.from_event_data(event_data, approximant)
    likelihood_linfree = LinearFree(event_data, wfg, par_dic_0, fbin)

    # min_bestfit_lnlike_to_keep=None → -inf (see CLP init); each worker
    # adapts its own threshold within its i_block, which is correct.
    clp = CoherentLikelihoodProcessor(
        bank_file_path,
        waveform_dir,
        n_phi,
        m_arr,
        likelihood_linfree,
        size_limit=size_limit,
        max_bestfit_lnlike_diff=max_bestfit_lnlike_diff,
    )
    clp.load_extrinsic_samples_data(rundir)

    amp, phase = clp.intrinsic_sample_processor.load_amp_and_phase(waveform_dir, i_block)
    h_impb = amp * np.exp(1j * phase)

    for e_block in e_blocks:
        clp.create_a_likelihood_block(
            h_impb,
            clp.full_response_dpe[..., e_block],
            clp.full_timeshift_dbe[..., e_block],
            i_block,
            e_block,
        )
        clp.combine_prob_samples_with_next_block()

    ps         = clp.prob_samples
    cached_dt  = clp.intrinsic_sample_processor.cached_dt_linfree_relative

    np.savez(
        out_path,
        # prob_samples columns
        i                        = ps["i"].values.astype(np.int64),
        e                        = ps["e"].values.astype(np.int64),
        o                        = ps["o"].values.astype(np.int64),
        lnl_marginalized         = ps["lnl_marginalized"].values.astype(float),
        ln_posterior             = ps["ln_posterior"].values.astype(float),
        bestfit_lnlike           = ps["bestfit_lnlike"].values.astype(float),
        d_h_1Mpc                 = ps["d_h_1Mpc"].values.astype(float),
        h_h_1Mpc                 = ps["h_h_1Mpc"].values.astype(float),
        # discarded stats (scalars stored as 0-d arrays)
        n_samples_discarded       = np.array(clp.n_samples_discarded),
        logsumexp_discarded       = np.array(clp.logsumexp_discarded_ln_posterior),
        logsumsqrexp_discarded    = np.array(clp.logsumsqrexp_discarded_ln_posterior),
        n_distance_marginalizations = np.array(clp.n_distance_marginalizations),
        # cached per-sample timeshifts
        cached_dt_keys            = np.array(list(cached_dt.keys()), dtype=np.int64),
        cached_dt_vals            = np.array(list(cached_dt.values()), dtype=float),
    )
    print(f"[done] i_block {i_block_idx}: {len(ps)} samples → {out_path}")


if __name__ == "__main__":
    main()
