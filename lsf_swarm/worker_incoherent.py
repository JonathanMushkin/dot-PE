#!/usr/bin/env python3
"""
LSF array worker: scores one block of intrinsic samples (all detectors).

Called by run_swarm.py via bsub.  The block ID comes from $LSB_JOBINDEX
(1-indexed) or --block-id on the CLI.

Output
------
{rundir}/swarm_setup/incoherent/block_{block_id}.npz
    inds      : int array, absolute bank indices this block processed
    lnlike_di : float array, shape (n_det, n_samples)
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

from dot_pe.inference import _create_single_detector_processor, run_for_single_detector


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rundir",    required=True)
    p.add_argument("--block-id",  type=int, default=None,
                   help="1-indexed block ID (default: $LSB_JOBINDEX)")
    args = p.parse_args()

    rundir   = Path(args.rundir)
    block_id = args.block_id if args.block_id is not None else int(os.environ["LSB_JOBINDEX"])

    out_path = rundir / "swarm_setup" / "incoherent" / f"block_{block_id}.npz"
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

    bank_folder        = Path(cfg["bank_folder"])
    fbin               = np.array(cfg["fbin"])
    approximant        = cfg["approximant"]
    m_arr              = np.array(cfg["m_arr"])
    n_phi              = cfg["n_phi"]
    n_t                = cfg["n_t"]
    blocksize_per_job  = cfg["blocksize_per_job"]
    incoherent_bsize   = cfg["incoherent_blocksize"]
    n_int              = cfg["n_int"]
    size_limit         = cfg.get("size_limit", 10**7)

    # This worker's sample range
    sample_start = (block_id - 1) * blocksize_per_job
    sample_end   = min(block_id * blocksize_per_job, n_int)

    if sample_start >= n_int:
        print(f"[skip] block {block_id}: start {sample_start} >= n_int {n_int}")
        np.savez(out_path,
                 inds=np.array([], dtype=int),
                 lnlike_di=np.zeros((len(event_data.detector_names), 0)))
        return

    chunk_inds = np.arange(sample_start, sample_end)
    n_chunk    = len(chunk_inds)
    n_det      = len(event_data.detector_names)
    lnlike_di  = np.zeros((n_det, n_chunk))

    sdp_by_det = {
        det: _create_single_detector_processor(
            event_data, det, par_dic_0, bank_folder,
            fbin, approximant, n_phi, m_arr,
            incoherent_bsize, size_limit,
        )
        for det in event_data.detector_names
    }

    for b_start in range(0, n_chunk, incoherent_bsize):
        b_end      = min(b_start + incoherent_bsize, n_chunk)
        batch_inds = chunk_inds[b_start:b_end]
        h_impb     = None
        for d, det_name in enumerate(event_data.detector_names):
            result = run_for_single_detector(
                event_data, det_name, par_dic_0, bank_folder,
                batch_inds, fbin, h_impb, approximant, n_phi,
                incoherent_bsize, m_arr, n_t, size_limit,
                sdp=sdp_by_det[det_name],
            )
            if h_impb is None:
                lnlike_di[d, b_start:b_end] = result[0]
                h_impb = result[1]
            else:
                lnlike_di[d, b_start:b_end] = result

    np.savez(out_path, inds=chunk_inds, lnlike_di=lnlike_di)
    print(f"[done] block {block_id}: samples [{sample_start}, {sample_end}) → {out_path}")


if __name__ == "__main__":
    main()
