"""Stage 5 on the GPU: one bank's coherent stage, the device counterpart of the per-bank body of
``dot_pe.mp_inference._run_coherent_mp``.

Differences from the CPU stage:
1. no ``mp_coherent_setup_<bank>`` directory is written; the summary weights come from one
   ``CoherentLikelihoodProcessor`` built in memory;
2. the linear-free time shifts of the selected templates are fitted on the device from resident
   waveforms (same fit, same likelihood as ``precompute_coherent_setup``), so no waveform file is
   read again;
3. the per-bank merge (concatenate, keep ``size_limit`` rows by ``bestfit_lnlike`` or
   ``priority_key``, track the largest dropped key) runs on the device.
The outputs (``banks/<bank>/prob_samples.feather``, ``intrinsic_sample_processor_cache.json`` and
the per-bank result dict) are those of the CPU stage.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .coherent import CoherentGPU
from .incoherent import linfree_dt

INT_COLS = ("i", "e", "o")


def coherent_setup_in_memory(*, bank_path, event_data, par_dic_0, fbin, approximant, m_arr, n_phi,
                             size_limit, max_bestfit_lnlike_diff, n_selected, blocksize, rundir):
    """The dict ``load_thin_setup`` returns, without writing or reading a setup directory.

    Returns (setup, likelihood_linfree); ``dt_linfree`` is left empty for the caller to fill.
    """
    from dot_pe.coherent_processing import CoherentLikelihoodProcessor
    from dot_pe.likelihood_calculating import LinearFree
    from dot_pe.utils import waveform_generator_from_config

    bank_path = Path(bank_path)
    bank_config = json.load(open(bank_path / "bank_config.json"))
    wfg = waveform_generator_from_config(event_data, bank_config)
    lk = LinearFree(event_data, wfg, par_dic_0, fbin)
    clp = CoherentLikelihoodProcessor(bank_path / "intrinsic_sample_bank.feather", bank_path / "waveforms",
                                      n_phi, m_arr, lk, size_limit=size_limit,
                                      max_bestfit_lnlike_diff=max_bestfit_lnlike_diff)
    n_blocks = -(-n_selected // blocksize)
    extrinsic_df = pd.read_feather(Path(rundir) / "extrinsic_samples.feather")
    setup = {
        "dh_weights_dmpb": clp.dh_weights_dmpb, "hh_weights_dmppb": clp.hh_weights_dmppb,
        "asd_drift": lk.asd_drift, "fbin": np.asarray(fbin), "logw_i": clp.full_log_prior_weights_i,
        "response_dpe": np.load(Path(rundir) / "response_dpe.npy"),
        "timeshift_dbe": np.load(Path(rundir) / "timeshift_dbe.npy"),
        "log_prior_weights_e": extrinsic_df["log_prior_weights"].values,
        "n_phi": int(n_phi), "m_arr": np.asarray(m_arr),
        "max_bestfit_lnlike_diff": float(max_bestfit_lnlike_diff), "size_limit": int(size_limit),
        "max_output_per_block": int(max(50_000, 2 * size_limit // max(1, n_blocks))),
    }
    return setup, lk


def _top_k(cp, cols, rank, k):
    """Keep the k rows of largest `rank`; return (cols, largest dropped rank or -inf)."""
    n = len(rank)
    if n <= k:
        return cols, -np.inf
    order = cp.argsort(-rank)
    dropped = float(rank[order[k]])
    keep = order[:k]
    return {c: v[keep] for c, v in cols.items()}, dropped


def coherent_bank_gpu(*, bank_id, bank_index, bank_path, event_data, par_dic_0, fbin, approximant, m_arr,
                      n_phi, n_ext, n_int_k, inds, h_raw_scaled, h_scale, blocksize, size_limit,
                      max_bestfit_lnlike_diff, rundir, bank_rundir, row_selection, selection_seed,
                      selected_lnlikes=None, lookup=None, timings=None):
    """Coherent stage for one bank; returns the per-bank result dict of ``_run_coherent_mp``.

    h_raw_scaled: device complex64 (n_sel, m, p, b) of raw waveforms × h_scale, rows aligned with
    `inds` (sorted). None: read them from the bank files.
    selected_lnlikes: the selected templates' incoherent log-likelihoods (aligned with `inds`), for
    the incoherent bound (``incoherent_bound_mask``); None disables it, as in the CPU stage.
    """
    import cupy as cp
    from dot_pe import priority_sampling
    from dot_pe.base_sampler_free_sampling import get_n_effective_total_i_e
    from cogwheel.utils import exp_normalize
    from dot_pe.utils import inds_to_blocks

    t0 = time.perf_counter()
    bank_rundir = Path(bank_rundir)
    bank_rundir.mkdir(parents=True, exist_ok=True)
    inds = np.asarray(inds, np.int64)
    n_total = n_phi * n_ext * n_int_k
    if len(inds) == 0:
        from dot_pe.coherent_processing import CoherentLikelihoodProcessor as CLP
        empty = pd.DataFrame(columns=CLP.PROB_SAMPLES_COLS).astype(
            dict(zip(CLP.PROB_SAMPLES_COLS, CLP.PROB_SAMPLES_COLS_DTYPES)))
        empty.to_feather(bank_rundir / "prob_samples.feather")
        json.dump({}, open(bank_rundir / "intrinsic_sample_processor_cache.json", "w"))
        return {"bank_id": bank_id, "lnZ_k": -np.inf, "lnZ_discarded_k": -np.inf, "N_k": n_total,
                "n_effective_k": 0.0, "n_effective_i_k": 0.0, "n_effective_e_k": 0.0,
                "n_distance_marginalizations_k": 0, "n_inds_used": 0}

    setup, lk = coherent_setup_in_memory(
        bank_path=bank_path, event_data=event_data, par_dic_0=par_dic_0, fbin=fbin,
        approximant=approximant, m_arr=m_arr, n_phi=n_phi, size_limit=size_limit,
        max_bestfit_lnlike_diff=max_bestfit_lnlike_diff, n_selected=len(inds), blocksize=blocksize,
        rundir=rundir)
    t_setup = time.perf_counter() - t0

    if h_raw_scaled is None:
        from .bank import load_selected
        h_raw_scaled, h_scale = load_selected(bank_path, inds)
    fbin_d = cp.asarray(np.asarray(fbin, float))
    m2 = list(np.asarray(m_arr)).index(2)
    h0 = cp.asarray(np.asarray(lk._h2plus0_fbin, np.complex128))
    pw = cp.asarray(np.asarray(lk._polyfit_weights, float))
    dt = cp.empty(len(inds))
    for s in range(0, len(inds), 8192):  # time shifts of the selected templates
        dt[s:s + 8192] = linfree_dt(cp, h_raw_scaled[s:s + 8192, m2, 0].astype(cp.complex128) / h_scale,
                                    h0, pw, fbin_d)
    dt_full = np.zeros(len(setup["logw_i"]))
    dt_h = dt.get()
    dt_full[inds] = dt_h
    setup.update({"dt_linfree": dt_full, "row_selection": row_selection, "ln_N_k": float(np.log(n_total)),
                  "selection_seed": selection_seed, "bank_index": bank_index})
    if selected_lnlikes is not None:
        from dot_pe.likelihood_calculating import incoherent_lookup
        setup["incoherent_lnlike_i"] = incoherent_lookup(len(setup["logw_i"]), inds, selected_lnlikes)

    g = CoherentGPU(setup, blocksize_e=blocksize, lookup=lookup)
    use_priority = g.use_priority
    rank_col = "priority_key" if use_priority else "bestfit_lnlike"
    acc, tau_bank = None, -np.inf
    n_disc_total, n_dist_marg = 0, 0
    lse_terms, lsse_terms = [], []
    t1 = time.perf_counter()
    pos = 0
    for i_block in inds_to_blocks(inds, blocksize):
        n = len(i_block)
        h = (h_raw_scaled[pos:pos + n].astype(cp.complex128) / h_scale
             * cp.exp(2j * np.pi * dt[pos:pos + n, None, None, None] * fbin_d[None, None, None, :]))
        pos += n
        cols, n_disc, lse, lsse, n_dm = g.run_iblock(i_block, h=h, return_device=True)
        n_disc_total += n_disc
        n_dist_marg += n_dm
        lse_terms.append(lse)
        lsse_terms.append(lsse)
        if acc is None:
            acc = cols
        else:
            acc = {c: cp.concatenate([acc[c], cols[c]]) for c in acc}
        if len(acc[rank_col]) > 2 * size_limit:
            acc, dropped = _top_k(cp, acc, acc[rank_col], size_limit)
            tau_bank = max(tau_bank, dropped)
    acc, dropped = _top_k(cp, acc, acc[rank_col], size_limit)
    tau_bank = max(tau_bank, dropped)
    t_blocks = time.perf_counter() - t1

    combined = pd.DataFrame({c: (v.get().astype(np.int64) if c in INT_COLS else v.get().astype(np.float64))
                             for c, v in acc.items()})
    if use_priority:
        combined = priority_sampling.raise_tau(combined, tau_bank)
        combined["ln_priority_weight"] = priority_sampling.ln_priority_weight(combined, np.log(n_total))
    else:
        combined["ln_priority_weight"] = 0.0
    ln_w = combined["ln_posterior"].values + combined["ln_priority_weight"].values
    combined["weights"] = exp_normalize(ln_w)
    combined.to_feather(bank_rundir / "prob_samples.feather")
    json.dump({int(k): float(v) for k, v in zip(inds, dt_h)},
              open(bank_rundir / "intrinsic_sample_processor_cache.json", "w"))

    from dot_pe.utils import safe_logsumexp
    ln_evidence = safe_logsumexp(ln_w) - np.log(n_total)
    ln_evidence_discarded = safe_logsumexp(lse_terms) - np.log(n_total)
    n_eff, n_eff_i, n_eff_e = get_n_effective_total_i_e(combined, assume_normalized=False)
    if timings is not None:
        timings[f"5_coherent_{bank_id}"] = {"cpu_setup": t_setup, "blocks": t_blocks,
                                            "write": time.perf_counter() - t1 - t_blocks,
                                            "n_iblocks": -(-len(inds) // blocksize)}
    del g
    cp.get_default_memory_pool().free_all_blocks()
    return {"bank_id": bank_id, "lnZ_k": ln_evidence, "lnZ_discarded_k": ln_evidence_discarded,
            "N_k": n_total, "n_effective_k": n_eff, "n_effective_i_k": n_eff_i, "n_effective_e_k": n_eff_e,
            "n_distance_marginalizations_k": n_dist_marg, "n_inds_used": len(inds)}
