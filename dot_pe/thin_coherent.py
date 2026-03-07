"""Thin-worker implementation for the coherent inference stage.

Eliminates per-worker CoherentLikelihoodProcessor / LinearFree by:
  1. Pre-computing summary weights and dt-correction cache once in the
     orchestrator (precompute_coherent_setup).
  2. Exposing a lightweight per-i_block function that only needs ~1–2 GB
     (run_thin_iblock).

Public API
----------
precompute_coherent_setup(setup_dir, bank_path, event_data, par_dic_0,
                          fbin, approximant, m_arr, n_phi, size_limit,
                          max_bestfit_lnlike_diff, selected_inds, blocksize)
    Idempotent — skips if setup_dir/setup.done already exists.

load_thin_setup(setup_dir, rundir) -> dict
    Load all pre-computed arrays from disk.  Call once per worker process;
    reuse for every i_block assigned to that process.

run_thin_iblock(i_block, e_blocks, setup, waveform_dir)
    -> (prob_samples_df, n_disc, logsumexp_disc, logsumsqrexp_disc,
        cached_dt_slice, n_dist_marg)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .coherent_processing import CoherentLikelihoodProcessor
from .likelihood_calculating import LikelihoodCalculator
from .sample_processing import IntrinsicSampleProcessor
from .utils import safe_logsumexp, inds_to_blocks

_SETUP_MARKER = "setup.done"


# ─────────────────────────────────────────────────────────────────────────────
# Pre-computation (orchestrator, runs once)
# ─────────────────────────────────────────────────────────────────────────────


def precompute_coherent_setup(
    setup_dir,
    bank_path,
    event_data,
    par_dic_0,
    fbin,
    approximant,
    m_arr,
    n_phi,
    size_limit,
    max_bestfit_lnlike_diff,
    selected_inds,
    blocksize=512,
):
    """
    Instantiate CoherentLikelihoodProcessor once, then save to setup_dir/:
      - summary weights (dh_weights_dmpb, hh_weights_dmppb)
      - asd_drift, fbin
      - log_prior_weights_i array (indexed by bank position)
      - dt_linfree array (indexed by bank position)
      - setup.json config

    Idempotent: skips if setup_dir/setup.done already exists.
    """
    setup_dir = Path(setup_dir)
    setup_dir.mkdir(parents=True, exist_ok=True)

    if (setup_dir / _SETUP_MARKER).exists():
        print(f"  [thin] coherent setup already done: {setup_dir}")
        return

    bank_path = Path(bank_path)
    bank_file_path = bank_path / "intrinsic_sample_bank.feather"
    waveform_dir = bank_path / "waveforms"

    from cogwheel.waveform import WaveformGenerator
    from .likelihood_calculating import LinearFree

    wfg = WaveformGenerator.from_event_data(event_data, approximant)
    likelihood_linfree = LinearFree(event_data, wfg, par_dic_0, fbin)

    clp = CoherentLikelihoodProcessor(
        bank_file_path,
        waveform_dir,
        n_phi,
        m_arr,
        likelihood_linfree,
        size_limit=size_limit,
        max_bestfit_lnlike_diff=max_bestfit_lnlike_diff,
    )

    # Summary weights — the outputs of get_summary(), called in CLP.__init__
    np.save(setup_dir / "dh_weights_dmpb.npy", clp.dh_weights_dmpb)
    np.save(setup_dir / "hh_weights_dmppb.npy", clp.hh_weights_dmppb)
    np.save(setup_dir / "asd_drift.npy", likelihood_linfree.asd_drift)
    np.save(setup_dir / "fbin.npy", fbin)

    # Log-prior weights as a full array indexed by bank position
    np.save(setup_dir / "logw_i.npy", clp.full_log_prior_weights_i)

    # Config
    with open(setup_dir / "setup.json", "w") as f:
        json.dump(
            {
                "n_phi": int(n_phi),
                "m_arr": np.asarray(m_arr).tolist(),
                "max_bestfit_lnlike_diff": float(max_bestfit_lnlike_diff),
                "size_limit": int(size_limit),
            },
            f,
        )

    # Pre-compute dt_linfree correction for every selected intrinsic index.
    # This is the only part that requires LinearFree (via ISP.load_amp_and_phase).
    isp = clp.intrinsic_sample_processor
    n_selected = len(selected_inds)
    print(f"  [thin] pre-computing dt_linfree for {n_selected} selected indices...")
    for i_block in inds_to_blocks(selected_inds, blocksize):
        isp.load_amp_and_phase(waveform_dir, i_block)

    # Save as full array indexed by bank position (zeros for unused positions)
    n_bank = len(clp.full_log_prior_weights_i)
    dt_full = np.zeros(n_bank, dtype=float)
    for idx, dt in isp.cached_dt_linfree_relative.items():
        dt_full[int(idx)] = float(dt)
    np.save(setup_dir / "dt_linfree.npy", dt_full)

    (setup_dir / _SETUP_MARKER).touch()
    print(f"  [thin] coherent setup saved to {setup_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Setup loader (call once per worker process)
# ─────────────────────────────────────────────────────────────────────────────


def load_thin_setup(setup_dir, rundir):
    """
    Load all pre-computed setup data into a dict.

    Parameters
    ----------
    setup_dir : path to coherent_setup/ directory (written by precompute_coherent_setup)
    rundir    : path to the top-level run directory (contains response_dpe.npy etc.)

    Returns a dict suitable for passing directly to run_thin_iblock().
    """
    setup_dir = Path(setup_dir)
    rundir = Path(rundir)

    with open(setup_dir / "setup.json") as f:
        cfg = json.load(f)

    extrinsic_df = pd.read_feather(rundir / "extrinsic_samples.feather")

    return {
        "dh_weights_dmpb":     np.load(setup_dir / "dh_weights_dmpb.npy"),
        "hh_weights_dmppb":    np.load(setup_dir / "hh_weights_dmppb.npy"),
        "asd_drift":           np.load(setup_dir / "asd_drift.npy"),
        "fbin":                np.load(setup_dir / "fbin.npy"),
        "dt_linfree":          np.load(setup_dir / "dt_linfree.npy"),
        "logw_i":              np.load(setup_dir / "logw_i.npy"),
        "response_dpe":        np.load(rundir / "response_dpe.npy"),
        "timeshift_dbe":       np.load(rundir / "timeshift_dbe.npy"),
        "log_prior_weights_e": extrinsic_df["log_prior_weights"].values,
        "n_phi":               cfg["n_phi"],
        "m_arr":               np.asarray(cfg["m_arr"]),
        "max_bestfit_lnlike_diff": cfg["max_bestfit_lnlike_diff"],
        "size_limit":          cfg["size_limit"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Thin i_block worker (pure numpy, no event data or LinearFree required)
# ─────────────────────────────────────────────────────────────────────────────


def run_thin_iblock(i_block, e_blocks, setup, waveform_dir):
    """
    Process one i_block × all e_blocks for coherent inference.

    Uses a growing bestfit-lnlike threshold (same logic as
    CoherentLikelihoodProcessor.create_a_likelihood_block) but without
    the size_limit top-k merge — all samples passing the threshold are
    returned.  The caller (orchestrator) is responsible for any global
    top-k merge across i_blocks.

    Parameters
    ----------
    i_block      : np.ndarray  absolute bank indices for this intrinsic block
    e_blocks     : list[np.ndarray]  extrinsic index blocks
    setup        : dict returned by load_thin_setup()
    waveform_dir : str or Path  directory containing waveform .npy files

    Returns
    -------
    prob_samples_df    : pd.DataFrame (may be empty)
    n_disc             : int
    logsumexp_disc     : float
    logsumsqrexp_disc  : float
    cached_dt_slice    : dict {int → float}  dt values for this i_block
    n_dist_marg        : int
    """
    waveform_dir = Path(waveform_dir)

    dh_w          = setup["dh_weights_dmpb"]
    hh_w          = setup["hh_weights_dmppb"]
    asd_drift     = setup["asd_drift"]
    fbin          = setup["fbin"]
    dt_linfree    = setup["dt_linfree"]
    logw_i        = setup["logw_i"]
    response_dpe  = setup["response_dpe"]
    timeshift_dbe = setup["timeshift_dbe"]
    logw_e        = setup["log_prior_weights_e"]
    n_phi         = setup["n_phi"]
    m_arr         = setup["m_arr"]
    max_diff      = setup["max_bestfit_lnlike_diff"]

    # Load raw waveforms and apply pre-cached dt correction (pure numpy)
    amp, phase = IntrinsicSampleProcessor._load_amp_and_phase(waveform_dir, i_block)
    dt_i = dt_linfree[i_block]  # shape (n_i,)
    phase = phase + 2 * np.pi * dt_i[:, None, None, None] * fbin[None, None, None, :]
    h_impb = amp * np.exp(1j * phase)

    lc = LikelihoodCalculator(n_phi, m_arr)
    rng = np.random.default_rng(int(i_block[0]) if len(i_block) else 0)

    # Running state
    min_bestfit = -np.inf
    all_frames = []
    n_disc = 0
    logsumexp_disc = -np.inf
    logsumsqrexp_disc = -np.inf
    n_dist_marg = 0

    for e_block in e_blocks:
        dh_ieo, hh_ieo = lc.get_dh_hh_ieo(
            dh_w, h_impb,
            response_dpe[..., e_block],
            timeshift_dbe[..., e_block],
            hh_w, asd_drift,
        )

        bestfit_ieo = 0.5 * (dh_ieo ** 2) / hh_ieo * (dh_ieo > 0)
        accepted = bestfit_ieo > min_bestfit

        # ── Accepted samples ──────────────────────────────────────────────
        if np.any(accepted):
            blk_max = float(bestfit_ieo[accepted].max())
            candidate = blk_max - max_diff
            if candidate > min_bestfit:
                min_bestfit = candidate

            n_dist_marg += int(np.sum(accepted))
            i_k, e_k, o_k = np.where(accepted)
            dh_k = dh_ieo[i_k, e_k, o_k]
            hh_k = hh_ieo[i_k, e_k, o_k]
            bank_i_k = i_block[i_k]
            bank_e_k = e_block[e_k]

            lnl_k = lc.lookup_table.lnlike_marginalized(dh_k, hh_k)
            ln_post_k = lnl_k + logw_i[bank_i_k] + logw_e[bank_e_k]

            all_frames.append(pd.DataFrame({
                "i":               bank_i_k.astype(np.int64),
                "e":               bank_e_k.astype(np.int64),
                "o":               o_k.astype(np.int64),
                "lnl_marginalized": lnl_k,
                "ln_posterior":    ln_post_k,
                "bestfit_lnlike":  bestfit_ieo[i_k, e_k, o_k],
                "d_h_1Mpc":        dh_k,
                "h_h_1Mpc":        hh_k,
            }))

        # ── Discarded samples stats ───────────────────────────────────────
        n_disc_blk = int(np.sum(~accepted))
        if n_disc_blk:
            n_disc += n_disc_blk
            if n_disc_blk < 1000:
                subset_size = n_disc_blk
            elif n_disc_blk < 10 ** 8:
                subset_size = int(np.sqrt(n_disc_blk))
            else:
                subset_size = 10 ** 4

            i_d, e_d, o_d = np.where(~accepted)
            if subset_size < len(i_d):
                sub = rng.choice(len(i_d), subset_size, replace=False)
                i_d, e_d, o_d = i_d[sub], e_d[sub], o_d[sub]

            dh_d = dh_ieo[i_d, e_d, o_d]
            hh_d = hh_ieo[i_d, e_d, o_d]
            lnl_d = np.zeros(len(i_d))
            pos = dh_d > 0
            if np.any(pos):
                lnl_d[pos] = lc.lookup_table.lnlike_marginalized(dh_d[pos], hh_d[pos])

            ln_post_d = lnl_d + logw_i[i_block[i_d]] + logw_e[e_block[e_d]]
            sf = np.log(n_disc_blk / len(i_d))
            logsumexp_disc = safe_logsumexp(
                [logsumexp_disc, safe_logsumexp(ln_post_d) + sf]
            )
            logsumsqrexp_disc = safe_logsumexp(
                [logsumsqrexp_disc, safe_logsumexp(2 * ln_post_d) + sf]
            )

    if all_frames:
        prob_samples_df = pd.concat(all_frames, ignore_index=True)
    else:
        prob_samples_df = pd.DataFrame(columns=[
            "i", "e", "o", "lnl_marginalized", "ln_posterior",
            "bestfit_lnlike", "d_h_1Mpc", "h_h_1Mpc",
        ])

    cached_dt_slice = {int(idx): float(dt_linfree[idx]) for idx in i_block}

    return (
        prob_samples_df,
        n_disc,
        logsumexp_disc,
        logsumsqrexp_disc,
        cached_dt_slice,
        n_dist_marg,
    )
