"""Shared test data for the P2 checks: GW240525_201644 run_2, bank 0 (read in place)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dot_pe import thin_coherent
from dot_pe.likelihood_calculating import incoherent_lookup
from dot_pe.utils import inds_to_blocks

RUNDIR = Path("/home/projects/barakz/Collaboration-gw/mushkin/dot-pe/o4b-pe-paper/artifacts/pe/"
              "dotpe/GW240525_201644/run_2")
BANK_ID = "bank_0"
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts/research/coherent-kernel"
BLOCKSIZE = 512
TOP_TEMPLATE = 378018
CONFIGS = [("bestfit", False), ("bestfit", True),
           ("phase_marginalized_priority", False), ("phase_marginalized_priority", True)]


def bank_path():
    return Path(json.load(open(RUNDIR / "run_kwargs.json"))["bank_folder"][0])


def selected():
    z = np.load(RUNDIR / "banks" / BANK_ID / "intrinsic_samples.npz")
    return z["inds"], z["incoherent_lnlikes"]


def test_blocks():
    inds, _ = selected()
    blocks = inds_to_blocks(inds, BLOCKSIZE)
    k_top = next(k for k, b in enumerate(blocks) if TOP_TEMPLATE in set(b.tolist()))
    return {0: blocks[0], k_top: blocks[k_top]}


def setup_for(row_selection, bound, seed=240525201644):
    s = thin_coherent.load_thin_setup(RUNDIR / f"mp_coherent_setup_{BANK_ID}", RUNDIR)
    n_ext = len(s["log_prior_weights_e"])
    s["row_selection"] = row_selection
    s["ln_N_k"] = float(np.log(s["n_phi"] * n_ext * len(s["logw_i"])))
    s["selection_seed"] = seed
    s["bank_index"] = 0
    if bound:
        inds, lnl = selected()
        s["incoherent_lnlike_i"] = incoherent_lookup(len(s["logw_i"]), inds, lnl)
    return s


def e_blocks(setup):
    return inds_to_blocks(np.arange(len(setup["log_prior_weights_e"])), BLOCKSIZE)


def tag(row_selection, bound):
    return f"{'bestfit' if row_selection == 'bestfit' else 'priority'}_{'bound' if bound else 'nobound'}"
