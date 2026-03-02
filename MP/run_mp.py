#!/usr/bin/env python3
"""
Multiprocessing drop-in for inference.run().
Parallelizes the two heavy loops; everything else delegates to dot_pe.inference.

    Stage 1 — Setup:        inference.prepare_run_objects()                  (serial)
    Stage 2 — Incoherent:   Pool over chunks of intrinsic samples            (NEW)
    Stage 3 — Cross-bank:   inference.select_intrinsic_..._incoherent_...()  (serial)
    Stage 4 — Extrinsic:    inference.draw_extrinsic_samples()               (serial)
    Stage 5 — Coherent:     Pool, one worker per i_block                     (NEW)
    Stage 6 — Postprocess:  inference.aggregate_and_save_results()           (serial)

Each worker inherits OMP_NUM_THREADS=1 set before the Pool is created.

──────────────────────────────────────────────────────────────────────────────
QUICK-START (copy-paste this into your terminal / LSF script)
──────────────────────────────────────────────────────────────────────────────

1. Create test data once (from the project root):

    python test_data/setup.py --n-pool 4

2. Smoke-test with the small bank (should finish in a few minutes):

    python MP/run_mp.py \\
        --event  test_data/event/tutorial_event.npz \\
        --bank   test_data/bank_small \\
        --rundir /tmp/mp_smoke \\
        --n-ext 512 --n-phi 50 --n-workers 4

3. Timing benchmark with the large bank:

    python MP/run_mp.py \\
        --event  test_data/event/tutorial_event.npz \\
        --bank   test_data/bank_large \\
        --rundir /tmp/mp_bench \\
        --n-ext 4096 --n-phi 100 --n-workers 16

4. LSF single-node job (16 cores):

    bsub -q physics-medium -n 16 -R "span[hosts=1]" \\
         -o mp_%J.out -e mp_%J.err \\
         python MP/run_mp.py \\
             --event  test_data/event/tutorial_event.npz \\
             --bank   test_data/bank_large \\
             --n-ext 4096 --n-phi 100 --n-workers 16

──────────────────────────────────────────────────────────────────────────────
WHAT TO REPORT AFTER A TEST RUN
──────────────────────────────────────────────────────────────────────────────
After each run, please report:
  1. n_workers used
  2. Wall-clock time (printed as "Total wall-clock time: X s" at the end,
     or check the LSF .out file)
  3. The contents of <rundir>/run_N/summary_results.json
  4. Any errors or unexpected warnings

Compare against the serial baseline (inference.run_and_profile with the
same bank/event/n-ext/n-phi) to measure speedup.
"""

# ── OMP override: must happen BEFORE numpy/scipy are loaded ──────────────────
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ── standard imports ─────────────────────────────────────────────────────────
import argparse
import json
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cogwheel.utils import exp_normalize
from cogwheel.waveform import WaveformGenerator
from dot_pe import inference
from dot_pe.base_sampler_free_sampling import get_n_effective_total_i_e
from dot_pe.coherent_processing import CoherentLikelihoodProcessor
from dot_pe.inference import _create_single_detector_processor, run_for_single_detector
from dot_pe.likelihood_calculating import LinearFree
from dot_pe.utils import inds_to_blocks, safe_logsumexp


# ─────────────────────────────────────────────────────────────────────────────
# Worker functions — must be module-level to be picklable on spawn-start systems
# ─────────────────────────────────────────────────────────────────────────────


def _incoherent_chunk_worker(args):
    """
    Score a contiguous range of intrinsic samples for all detectors.

    Parameters (packed in args tuple)
    ----------------------------------
    sample_start, sample_end : absolute bank indices [start, end)
    batch_size               : sub-batch size (mirrors single_detector_blocksize)
    event_data, par_dic_0    : passed directly
    bank_folder              : str path
    fbin, approximant, n_phi, m_arr, n_t, size_limit : bank/inference config

    Returns
    -------
    chunk_inds  : np.ndarray  shape (N,)   absolute indices
    lnlike_di   : np.ndarray  shape (n_det, N)  per-detector log-likelihoods
    """
    (
        sample_start,
        sample_end,
        batch_size,
        event_data,
        par_dic_0,
        bank_folder,
        fbin,
        approximant,
        n_phi,
        m_arr,
        n_t,
        size_limit,
    ) = args

    bank_folder = Path(bank_folder)
    chunk_inds = np.arange(sample_start, sample_end)
    n_chunk = len(chunk_inds)
    n_det = len(event_data.detector_names)
    lnlike_di = np.zeros((n_det, n_chunk))

    # Build per-detector processors once per worker process
    sdp_by_det = {
        det: _create_single_detector_processor(
            event_data, det, par_dic_0, bank_folder,
            fbin, approximant, n_phi, m_arr, batch_size, size_limit,
        )
        for det in event_data.detector_names
    }

    for b_start in range(0, n_chunk, batch_size):
        b_end = min(b_start + batch_size, n_chunk)
        batch_inds = chunk_inds[b_start:b_end]
        h_impb = None
        for d, det_name in enumerate(event_data.detector_names):
            result = run_for_single_detector(
                event_data, det_name, par_dic_0, bank_folder,
                batch_inds, fbin, h_impb, approximant, n_phi,
                batch_size, m_arr, n_t, size_limit,
                sdp=sdp_by_det[det_name],
            )
            if h_impb is None:          # first detector: also returns waveforms
                lnlike_di[d, b_start:b_end] = result[0]
                h_impb = result[1]
            else:
                lnlike_di[d, b_start:b_end] = result

    return chunk_inds, lnlike_di


def _coherent_iblock_worker(args):
    """
    Process one i_block × all e_blocks for coherent inference.

    Parameters (packed in args tuple)
    ----------------------------------
    i_block                  : np.ndarray of intrinsic indices for this block
    e_blocks                 : list of np.ndarray extrinsic index blocks
    bank_folder              : str path
    n_phi, m_arr             : inference config
    par_dic_0                : reference parameter dict
    event_data               : EventData
    top_rundir               : str path  (where extrinsic_samples.feather lives)
    size_limit               : int
    max_bestfit_lnlike_diff  : float
    intrinsic_logw_lookup    : None  or  (inds, logw) tuple

    Returns
    -------
    prob_samples             : pd.DataFrame
    n_samples_discarded      : int
    logsumexp_discarded      : float
    logsumsqrexp_discarded   : float
    cached_dt                : dict {int → float}
    n_distance_marginalizations : int
    """
    (
        i_block,
        e_blocks,
        bank_folder,
        n_phi,
        m_arr,
        par_dic_0,
        event_data,
        top_rundir,
        size_limit,
        max_bestfit_lnlike_diff,
        intrinsic_logw_lookup,
    ) = args

    bank_folder = Path(bank_folder)
    top_rundir = Path(top_rundir)

    with open(bank_folder / "bank_config.json") as f:
        bank_config = json.load(f)
    fbin = np.array(bank_config["fbin"])
    approximant = bank_config["approximant"]

    wfg = WaveformGenerator.from_event_data(event_data, approximant)
    likelihood_linfree = LinearFree(event_data, wfg, par_dic_0, fbin)

    bank_file_path = bank_folder / "intrinsic_sample_bank.feather"
    waveform_dir = bank_folder / "waveforms"

    clp = CoherentLikelihoodProcessor(
        bank_file_path,
        waveform_dir,
        n_phi,
        m_arr,
        likelihood_linfree,
        size_limit=size_limit,
        max_bestfit_lnlike_diff=max_bestfit_lnlike_diff,
        intrinsic_logw_lookup=intrinsic_logw_lookup,
    )
    clp.load_extrinsic_samples_data(top_rundir)

    # Load waveforms once for this i_block, then iterate over all e_blocks
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

    return (
        clp.prob_samples,
        clp.n_samples_discarded,
        clp.logsumexp_discarded_ln_posterior,
        clp.logsumsqrexp_discarded_ln_posterior,
        dict(clp.intrinsic_sample_processor.cached_dt_linfree_relative),
        clp.n_distance_marginalizations,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Parallel incoherent selection
# ─────────────────────────────────────────────────────────────────────────────


def _collect_incoherent_mp(
    event_data, par_dic_0, bank_folder,
    fbin, approximant, m_arr,
    n_int, n_phi, n_t, batch_size, size_limit, n_workers,
):
    """
    Parallel replacement for collect_int_samples_from_single_detectors().
    Returns (inds, lnlike_di, incoherent_lnlikes) — threshold NOT applied.

    The sample range [0, n_int) is split into n_workers chunks.
    Each worker reconstructs its own per-detector SDPs and processes
    its chunk of batches, reusing h_impb across detectors per batch.
    """
    # Round chunk size up to a multiple of batch_size so batch boundaries align
    raw_chunk = max(batch_size, (n_int + n_workers - 1) // n_workers)
    chunk_size = ((raw_chunk + batch_size - 1) // batch_size) * batch_size

    sample_ranges = []
    s = 0
    while s < n_int:
        sample_ranges.append((s, min(s + chunk_size, n_int)))
        s += chunk_size

    worker_args = [
        (s, e, batch_size, event_data, par_dic_0, str(bank_folder),
         fbin, approximant, n_phi, m_arr, n_t, size_limit)
        for s, e in sample_ranges
    ]

    n_actual = min(n_workers, len(worker_args))
    print(f"  [MP] incoherent: {len(worker_args)} chunks → {n_actual} workers")

    with Pool(n_actual) as pool:
        results = pool.map(_incoherent_chunk_worker, worker_args)

    all_inds = np.concatenate([r[0] for r in results])
    all_lnlike_di = np.concatenate([r[1] for r in results], axis=1)
    incoherent_lnlikes = all_lnlike_di.sum(axis=0)

    return all_inds, all_lnlike_di, incoherent_lnlikes


# ─────────────────────────────────────────────────────────────────────────────
# Parallel coherent inference
# ─────────────────────────────────────────────────────────────────────────────


def _run_coherent_mp(
    *,
    banks, event_data, rundir, banks_dir, par_dic_0,
    selected_inds_by_bank, n_int_dict, n_ext, n_phi, m_arr,
    blocksize, size_limit, max_bestfit_lnlike_diff,
    bank_logw_override_dict, n_workers,
):
    """
    Parallel replacement for run_coherent_inference_per_bank().
    One worker per i_block; the main process merges the results.
    """
    print("\n=== Coherent inference per bank (MP) ===")
    per_bank_results = []

    for bank_id, bank_path in banks.items():
        print(f"\nProcessing bank: {bank_id}")
        bank_rundir = banks_dir / bank_id
        bank_rundir.mkdir(parents=True, exist_ok=True)

        inds = selected_inds_by_bank[bank_id]
        n_int_k = n_int_dict[bank_id]
        n_total_samples = n_phi * n_ext * n_int_k

        if len(inds) == 0:
            empty = pd.DataFrame(
                columns=CoherentLikelihoodProcessor.PROB_SAMPLES_COLS
            ).astype({
                k: v for k, v in zip(
                    CoherentLikelihoodProcessor.PROB_SAMPLES_COLS,
                    CoherentLikelihoodProcessor.PROB_SAMPLES_COLS_DTYPES,
                )
            })
            empty.to_feather(bank_rundir / "prob_samples.feather")
            with open(bank_rundir / "intrinsic_sample_processor_cache.json", "w") as f:
                json.dump({}, f)
            per_bank_results.append({
                "bank_id": bank_id,
                "lnZ_k": -np.inf, "lnZ_discarded_k": -np.inf,
                "N_k": n_total_samples, "n_effective_k": 0.0,
                "n_effective_i_k": 0.0, "n_effective_e_k": 0.0,
                "n_distance_marginalizations_k": 0, "n_inds_used": 0,
            })
            continue

        intrinsic_logw_lookup = None
        if bank_logw_override_dict and bank_id in bank_logw_override_dict:
            override_logw = np.asarray(bank_logw_override_dict[bank_id])[inds]
            intrinsic_logw_lookup = (inds, override_logw)

        i_blocks = inds_to_blocks(inds, blocksize)
        e_blocks = inds_to_blocks(np.arange(n_ext), blocksize)

        worker_args = [
            (
                i_block, e_blocks,
                str(bank_path), n_phi, m_arr,
                par_dic_0, event_data, str(rundir),
                size_limit, max_bestfit_lnlike_diff,
                intrinsic_logw_lookup,
            )
            for i_block in i_blocks
        ]

        n_actual = min(n_workers, len(i_blocks))
        print(f"  [MP] coherent: {len(i_blocks)} i_blocks → {n_actual} workers")

        with Pool(n_actual) as pool:
            results = pool.map(_coherent_iblock_worker, worker_args)

        # Merge results from all i_block workers
        partial_dfs = [r[0] for r in results if len(r[0]) > 0]
        n_disc_total = sum(r[1] for r in results)
        logsumexp_disc = safe_logsumexp([r[2] for r in results])
        logsumsqrexp_disc = safe_logsumexp([r[3] for r in results])
        cached_dt = {}
        for r in results:
            cached_dt.update(r[4])
        n_dist_marg = sum(r[5] for r in results)

        if partial_dfs:
            combined = pd.concat(partial_dfs, ignore_index=True)
        else:
            combined = pd.DataFrame(columns=CoherentLikelihoodProcessor.PROB_SAMPLES_COLS)

        combined["weights"] = exp_normalize(combined["ln_posterior"].values)
        combined.to_feather(bank_rundir / "prob_samples.feather")

        cache_path = bank_rundir / "intrinsic_sample_processor_cache.json"
        with open(cache_path, "w") as f:
            json.dump({int(k): float(v) for k, v in cached_dt.items()}, f)

        ln_evidence = safe_logsumexp(combined["ln_posterior"].values) - np.log(n_total_samples)
        ln_evidence_discarded = logsumexp_disc - np.log(n_total_samples)
        n_eff, n_eff_i, n_eff_e = get_n_effective_total_i_e(combined, assume_normalized=False)

        per_bank_results.append({
            "bank_id": bank_id,
            "lnZ_k": ln_evidence,
            "lnZ_discarded_k": ln_evidence_discarded,
            "N_k": n_total_samples,
            "n_effective_k": n_eff,
            "n_effective_i_k": n_eff_i,
            "n_effective_e_k": n_eff_e,
            "n_distance_marginalizations_k": n_dist_marg,
            "n_inds_used": len(inds),
        })

    return per_bank_results


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def run(
    event,
    bank_folder,
    n_ext,
    n_phi,
    n_t,
    n_int=None,
    blocksize=512,
    single_detector_blocksize=512,
    n_workers=None,
    seed=None,
    size_limit=10**7,
    draw_subset=True,
    n_draws=None,
    event_dir=None,
    rundir=None,
    max_incoherent_lnlike_drop=20,
    max_bestfit_lnlike_diff=20,
    mchirp_guess=None,
    extrinsic_samples=None,
    bank_logw_override=None,
):
    """
    Run inference with multiprocessing.  Drop-in for inference.run().

    Extra argument vs inference.run()
    -----------------------------------
    n_workers : int or None
        Number of parallel workers.  None → os.cpu_count().
    """
    t0 = time.time()
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    print(f"[MP] n_workers = {n_workers}")

    # ── Stage 1: setup (serial) ───────────────────────────────────────────────
    ctx = inference.prepare_run_objects(
        event=event,
        bank_folder=bank_folder,
        n_int=n_int,
        n_ext=n_ext,
        n_phi=n_phi,
        n_t=n_t,
        blocksize=blocksize,
        single_detector_blocksize=single_detector_blocksize,
        i_int_start=0,
        seed=seed,
        load_inds=False,
        inds_path=None,
        size_limit=size_limit,
        draw_subset=draw_subset,
        n_draws=n_draws,
        event_dir=event_dir,
        rundir=rundir,
        coherent_score_min_n_effective_prior=100,
        max_incoherent_lnlike_drop=max_incoherent_lnlike_drop,
        max_bestfit_lnlike_diff=max_bestfit_lnlike_diff,
        mchirp_guess=mchirp_guess,
        extrinsic_samples=extrinsic_samples,
        n_phi_incoherent=None,
        preselected_indices=None,
        bank_logw_override=bank_logw_override,
        coherent_posterior_kwargs={},
    )

    # ── Stage 2: incoherent selection (parallelized) ──────────────────────────
    print("\n=== Incoherent selection per bank (MP) ===")
    candidate_inds_by_bank = {}
    lnlikes_by_bank = {}
    lnlikes_di_by_bank = {}

    for bank_id, bank_path in ctx["banks"].items():
        print(f"\nProcessing bank: {bank_id}")
        inds, lnlike_di, incoherent_lnlikes = _collect_incoherent_mp(
            event_data=ctx["event_data"],
            par_dic_0=ctx["par_dic_0"],
            bank_folder=Path(bank_path),
            fbin=ctx["fbin"],
            approximant=ctx["approximant"],
            m_arr=ctx["m_arr"],
            n_int=ctx["n_int_dict"][bank_id],
            n_phi=n_phi,
            n_t=n_t,
            batch_size=single_detector_blocksize,
            size_limit=size_limit,
            n_workers=n_workers,
        )
        candidate_inds_by_bank[bank_id] = inds
        lnlikes_by_bank[bank_id] = incoherent_lnlikes
        lnlikes_di_by_bank[bank_id] = lnlike_di
        print(f"Bank {bank_id}: {len(inds)} intrinsic samples evaluated.")

    # ── Stage 3: cross-bank threshold (serial) ────────────────────────────────
    selected_inds_by_bank, _, _ = (
        inference.select_intrinsic_samples_across_banks_by_incoherent_likelihood(
            banks=ctx["banks"],
            candidate_inds_by_bank=candidate_inds_by_bank,
            incoherent_lnlikes_by_bank=lnlikes_by_bank,
            lnlikes_di_by_bank=lnlikes_di_by_bank,
            max_incoherent_lnlike_drop=max_incoherent_lnlike_drop,
            banks_dir=ctx["banks_dir"],
            event_data=ctx["event_data"],
        )
    )

    # ── Stage 4: extrinsic sampling (serial) ──────────────────────────────────
    inference.draw_extrinsic_samples(
        banks=ctx["banks"],
        event_data=ctx["event_data"],
        par_dic_0=ctx["par_dic_0"],
        fbin=ctx["fbin"],
        approximant=ctx["approximant"],
        selected_inds_by_bank=selected_inds_by_bank,
        coherent_score_kwargs=ctx["coherent_score_kwargs"],
        seed=seed,
        n_ext=n_ext,
        rundir=ctx["rundir"],
        extrinsic_samples=extrinsic_samples,
    )

    # ── Stage 5: coherent inference (parallelized) ────────────────────────────
    per_bank_results = _run_coherent_mp(
        banks=ctx["banks"],
        event_data=ctx["event_data"],
        rundir=ctx["rundir"],
        banks_dir=ctx["banks_dir"],
        par_dic_0=ctx["par_dic_0"],
        selected_inds_by_bank=selected_inds_by_bank,
        n_int_dict=ctx["n_int_dict"],
        n_ext=n_ext,
        n_phi=n_phi,
        m_arr=ctx["m_arr"],
        blocksize=blocksize,
        size_limit=size_limit,
        max_bestfit_lnlike_diff=max_bestfit_lnlike_diff,
        bank_logw_override_dict=ctx.get("bank_logw_override_dict"),
        n_workers=n_workers,
    )

    # ── Stage 6: postprocess (serial) ─────────────────────────────────────────
    result = inference.aggregate_and_save_results(
        per_bank_results=per_bank_results,
        banks=ctx["banks"],
        event_data=ctx["event_data"],
        rundir=ctx["rundir"],
        banks_dir=ctx["banks_dir"],
        n_phi=n_phi,
        pr=ctx["pr"],
        n_draws=n_draws,
        draw_subset=draw_subset,
    )

    print(f"\nTotal wall-clock time: {time.time() - t0:.1f} s")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description="dot-pe multiprocessing inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--event",    required=True, help="path to event .npz file")
    p.add_argument("--bank",     required=True, help="path to bank folder")
    p.add_argument("--rundir",   default=None,  help="output directory (auto-named if omitted)")
    p.add_argument("--n-workers", type=int, default=None, help="parallel workers (default: cpu_count)")
    p.add_argument("--n-ext",    type=int, default=4096)
    p.add_argument("--n-phi",    type=int, default=100)
    p.add_argument("--n-t",      type=int, default=128)
    p.add_argument("--n-int",    type=int, default=None, help="number of intrinsic samples (default: full bank)")
    p.add_argument("--blocksize", type=int, default=512)
    p.add_argument("--single-detector-blocksize", type=int, default=512)
    p.add_argument("--seed",     type=int, default=None)
    p.add_argument("--mchirp-guess", type=float, default=None)
    p.add_argument("--max-incoherent-lnlike-drop", type=float, default=20.0)
    p.add_argument("--max-bestfit-lnlike-diff",    type=float, default=20.0)
    p.add_argument("--no-draw-subset", action="store_false", dest="draw_subset")
    args = p.parse_args()

    run(
        event=args.event,
        bank_folder=args.bank,
        n_ext=args.n_ext,
        n_phi=args.n_phi,
        n_t=args.n_t,
        n_int=args.n_int,
        blocksize=args.blocksize,
        single_detector_blocksize=args.single_detector_blocksize,
        n_workers=args.n_workers,
        seed=args.seed,
        rundir=args.rundir,
        mchirp_guess=args.mchirp_guess,
        max_incoherent_lnlike_drop=args.max_incoherent_lnlike_drop,
        max_bestfit_lnlike_diff=args.max_bestfit_lnlike_diff,
        draw_subset=args.draw_subset,
    )


if __name__ == "__main__":
    main()
