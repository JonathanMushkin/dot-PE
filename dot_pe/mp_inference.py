"""Multiprocessing drop-in for inference.run().

Parallelises the two heavy loops; everything else delegates to
dot_pe.inference.

Stage 1 — Setup:      inference.prepare_run_objects()        (serial)
Stage 2 — Incoherent: Pool over chunks of intrinsic samples  (parallel)
Stage 3 — Cross-bank: inference.select_intrinsic_...         (serial)
Stage 4 — Extrinsic:  draw_extrinsic_samples[_parallel]()    (serial
                                                              /parallel)
Stage 5 — Coherent:   Pool, one thin worker per block        (parallel)
Stage 6 — Postprocess:inference.aggregate_and_save_results() (serial)

Public API
----------
run(..., n_workers, n_ext_workers) -> Path
    Drop-in for inference.run(); see run() docstring for extra
    parameters.

NOTE: OMP_NUM_THREADS and related BLAS thread variables
    When used as a CLI (``python -m dot_pe.mp_inference``) the env
    overrides at module top fire before numpy — correct order.  In a
    notebook, numpy is already loaded by import time; set these in
    Cell 1 before any imports:

        import os
        os.environ["OMP_NUM_THREADS"]      = "1"
        os.environ["MKL_NUM_THREADS"]      = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"]  = "1"
"""

# ── OpenMP thread override — must happen before numpy is loaded ──────
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ── standard imports ──────────────────────────────────────────────────
import argparse
import ctypes
import gc
import json
import resource
import time
import warnings
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

from cogwheel.utils import exp_normalize
from dot_pe import inference, thin_coherent
from dot_pe.base_sampler_free_sampling import get_n_effective_total_i_e
from dot_pe.coherent_processing import CoherentLikelihoodProcessor
from dot_pe.inference import (
    _create_single_detector_processor,
    run_for_single_detector,
)
from dot_pe.parallel_extrinsic import draw_extrinsic_samples_parallel
from dot_pe.utils import inds_to_blocks, safe_logsumexp


def _log_rss(label: str) -> None:
    """Print current and peak resident set size (RSS) in MB."""
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        cur_mb = 0.0
        with open("/proc/self/status") as _f:
            for _line in _f:
                if _line.startswith("VmRSS:"):
                    cur_mb = int(_line.split()[1]) / 1024
                    break
    except Exception:
        cur_mb = peak_kb / 1024
    print(f"  [RSS] {label}: cur={cur_mb:.0f} MB  peak={peak_kb / 1024:.0f} MB")


# ─────────────────────────────────────────────────────────────────────
# Worker functions — must be at module level to be picklable
# ─────────────────────────────────────────────────────────────────────

# Module-level setup dict for incoherent workers — populated in the
# main process before Pool creation, then inherited by workers via
# copy-on-write (COW) fork (no per-task pickling of large objects).
_incoherent_setup = None


def _incoherent_chunk_worker(args):
    """Score a contiguous range of intrinsic samples across all detectors."""
    sample_start, sample_end = args

    s = _incoherent_setup
    event_data = s["event_data"]
    par_dic_0 = s["par_dic_0"]
    bank_folder = Path(s["bank_folder"])
    fbin = s["fbin"]
    approximant = s["approximant"]
    n_phi = s["n_phi"]
    m_arr = s["m_arr"]
    n_t = s["n_t"]
    size_limit = s["size_limit"]
    batch_size = s["batch_size"]

    chunk_inds = np.arange(sample_start, sample_end)
    n_chunk = len(chunk_inds)
    n_det = len(event_data.detector_names)
    lnlike_di = np.zeros((n_det, n_chunk))

    # Build per-detector processors once per worker process
    sdp_by_det = {
        det: _create_single_detector_processor(
            event_data,
            det,
            par_dic_0,
            bank_folder,
            fbin,
            approximant,
            n_phi,
            m_arr,
            batch_size,
            size_limit,
        )
        for det in event_data.detector_names
    }

    for b_start in range(0, n_chunk, batch_size):
        b_end = min(b_start + batch_size, n_chunk)
        batch_inds = chunk_inds[b_start:b_end]
        h_impb = None
        for d, det_name in enumerate(event_data.detector_names):
            result = run_for_single_detector(
                event_data,
                det_name,
                par_dic_0,
                bank_folder,
                batch_inds,
                fbin,
                h_impb,
                approximant,
                n_phi,
                batch_size,
                m_arr,
                n_t,
                size_limit,
                sdp=sdp_by_det[det_name],
            )
            if h_impb is None:  # first detector: also returns waveforms
                lnlike_di[d, b_start:b_end] = result[0]
                h_impb = result[1]
            else:
                lnlike_di[d, b_start:b_end] = result

    return chunk_inds, lnlike_di


# Module-level setup dict — populated in the main process before Pool
# creation, then inherited by fork-based workers via copy-on-write
# (no per-task pickling).
_thin_setup = None


def _thin_coherent_worker(args):
    """Thin coherent worker: process one intrinsic block x all extrinsic."""
    i_block, e_blocks, waveform_dir = args
    return thin_coherent.run_thin_iblock(i_block, e_blocks, _thin_setup, waveform_dir)


# ─────────────────────────────────────────────────────────────────────
# Parallel incoherent selection
# ─────────────────────────────────────────────────────────────────────


def _collect_incoherent_mp(
    event_data,
    par_dic_0,
    bank_folder,
    fbin,
    approximant,
    m_arr,
    n_int,
    n_phi,
    n_t,
    batch_size,
    size_limit,
    n_workers,
):
    """
    Parallel replacement for collect_int_samples_from_single_detectors().

    Splits [0, n_int) into n_workers chunks.  Workers inherit large
    shared objects via copy-on-write (COW) fork from _incoherent_setup.
    Returns (inds, lnlike_di, incoherent_lnlikes) with no threshold.
    """
    global _incoherent_setup

    # Round chunk size up so it aligns with batch_size boundaries.
    raw_chunk = max(batch_size, (n_int + n_workers - 1) // n_workers)
    chunk_size = ((raw_chunk + batch_size - 1) // batch_size) * batch_size

    sample_ranges = []
    s = 0
    while s < n_int:
        sample_ranges.append((s, min(s + chunk_size, n_int)))
        s += chunk_size

    worker_args = [(s, e) for s, e in sample_ranges]

    n_actual = min(n_workers, len(worker_args))
    print(f"  [MP] incoherent: {len(worker_args)} chunks -> {n_actual} workers")

    _incoherent_setup = {
        "event_data": event_data,
        "par_dic_0": par_dic_0,
        "bank_folder": str(bank_folder),
        "fbin": fbin,
        "approximant": approximant,
        "n_phi": n_phi,
        "m_arr": m_arr,
        "n_t": n_t,
        "size_limit": size_limit,
        "batch_size": batch_size,
    }
    try:
        with Pool(n_actual) as pool:
            results = list(tqdm(
                pool.imap(_incoherent_chunk_worker, worker_args),
                total=len(worker_args),
                desc="incoherent chunks",
            ))
    finally:
        _incoherent_setup = None  # release references

    all_inds = np.concatenate([r[0] for r in results])
    all_lnlike_di = np.concatenate([r[1] for r in results], axis=1)
    incoherent_lnlikes = all_lnlike_di.sum(axis=0)

    return all_inds, all_lnlike_di, incoherent_lnlikes


# ─────────────────────────────────────────────────────────────────────
# Parallel coherent inference
# ─────────────────────────────────────────────────────────────────────


def _run_coherent_mp(
    *,
    banks,
    event_data,
    rundir,
    banks_dir,
    par_dic_0,
    selected_inds_by_bank,
    n_int_dict,
    n_ext,
    n_phi,
    m_arr,
    blocksize,
    size_limit,
    max_bestfit_lnlike_diff,
    bank_logw_override_dict,
    n_workers,
    fbin,
    approximant,
):
    """
    Parallel replacement for run_coherent_inference_per_bank().

    Pre-computes summary weights and per-sample arrival-time-shift (dt)
    cache once per bank in the main process, then dispatches thin
    workers (one per intrinsic-sample block).  Each worker materialises
    only its own complex overlap array of shape
    (blocksize, n_ext, n_phi) complex128; for (512, 2048, 100) this is
    ~1.7 GB per worker.
    """
    global _thin_setup

    print("\n=== Coherent inference per bank (MP, thin workers) ===")
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
            ).astype(
                {
                    k: v
                    for k, v in zip(
                        CoherentLikelihoodProcessor.PROB_SAMPLES_COLS,
                        CoherentLikelihoodProcessor.PROB_SAMPLES_COLS_DTYPES,
                    )
                }
            )
            empty.to_feather(bank_rundir / "prob_samples.feather")
            cache_file = bank_rundir / "intrinsic_sample_processor_cache.json"
            with open(cache_file, "w") as f:
                json.dump({}, f)
            per_bank_results.append(
                {
                    "bank_id": bank_id,
                    "lnZ_k": -np.inf,
                    "lnZ_discarded_k": -np.inf,
                    "N_k": n_total_samples,
                    "n_effective_k": 0.0,
                    "n_effective_i_k": 0.0,
                    "n_effective_e_k": 0.0,
                    "n_distance_marginalizations_k": 0,
                    "n_inds_used": 0,
                }
            )
            continue

        # ── Pre-computation (main process, once per bank) ─────────────
        setup_dir = rundir / f"mp_coherent_setup_{bank_id}"
        print(f"  [MP] pre-computing coherent setup for bank {bank_id}...")
        thin_coherent.precompute_coherent_setup(
            setup_dir=setup_dir,
            bank_path=Path(bank_path),
            event_data=event_data,
            par_dic_0=par_dic_0,
            fbin=fbin,
            approximant=approximant,
            m_arr=m_arr,
            n_phi=n_phi,
            size_limit=size_limit,
            max_bestfit_lnlike_diff=max_bestfit_lnlike_diff,
            selected_inds=inds,
            blocksize=blocksize,
        )

        # Free CoherentLikelihoodProcessor (CLP) memory back to the
        # OS before forking workers.
        gc.collect()
        try:
            ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        _log_rss("after precompute + gc/trim (before coherent pool)")

        # Load setup into main process; workers inherit via COW fork
        _thin_setup = thin_coherent.load_thin_setup(setup_dir, rundir)
        _log_rss("after load_thin_setup (coherent Pool fork point)")

        waveform_dir = str(Path(bank_path) / "waveforms")
        i_blocks = inds_to_blocks(inds, blocksize)
        e_blocks = inds_to_blocks(np.arange(n_ext), blocksize)

        worker_args = [(i_block, e_blocks, waveform_dir) for i_block in i_blocks]

        n_actual = min(n_workers, len(i_blocks))
        print(f"  [MP] coherent: {len(i_blocks)} i_blocks -> {n_actual} workers")

        # Incremental merge: process results as they arrive to cap
        # memory rather than collecting all i_block results first.
        accumulated = None
        n_disc_total = 0
        logsumexp_disc = -np.inf
        logsumsqrexp_disc = -np.inf
        cached_dt = {}
        n_dist_marg = 0

        with Pool(n_actual) as pool:
            for r in tqdm(
                pool.imap_unordered(_thin_coherent_worker, worker_args),
                total=len(worker_args),
                desc="coherent i_blocks",
            ):
                df, n_disc, lse_disc, lsse_disc, dt_slice, n_dm = r
                n_disc_total += n_disc
                logsumexp_disc = safe_logsumexp([logsumexp_disc, lse_disc])
                logsumsqrexp_disc = safe_logsumexp([logsumsqrexp_disc, lsse_disc])
                cached_dt.update(dt_slice)
                n_dist_marg += n_dm
                if len(df) > 0:
                    if accumulated is None:
                        accumulated = df
                    else:
                        accumulated = pd.concat([accumulated, df], ignore_index=True)
                        if len(accumulated) > size_limit:
                            top_idx = np.argpartition(
                                accumulated["bestfit_lnlike"].values,
                                -size_limit,
                            )[-size_limit:]
                            accumulated = accumulated.iloc[top_idx].reset_index(
                                drop=True
                            )

        if accumulated is not None:
            combined = accumulated
        else:
            combined = pd.DataFrame(
                columns=CoherentLikelihoodProcessor.PROB_SAMPLES_COLS
            )

        combined["weights"] = exp_normalize(combined["ln_posterior"].values)
        combined.to_feather(bank_rundir / "prob_samples.feather")

        cache_path = bank_rundir / "intrinsic_sample_processor_cache.json"
        with open(cache_path, "w") as f:
            json.dump({int(k): float(v) for k, v in cached_dt.items()}, f)

        ln_evidence = safe_logsumexp(combined["ln_posterior"].values) - np.log(
            n_total_samples
        )
        ln_evidence_discarded = logsumexp_disc - np.log(n_total_samples)
        n_eff, n_eff_i, n_eff_e = get_n_effective_total_i_e(
            combined, assume_normalized=False
        )

        per_bank_results.append(
            {
                "bank_id": bank_id,
                "lnZ_k": ln_evidence,
                "lnZ_discarded_k": ln_evidence_discarded,
                "N_k": n_total_samples,
                "n_effective_k": n_eff,
                "n_effective_i_k": n_eff_i,
                "n_effective_e_k": n_eff_e,
                "n_distance_marginalizations_k": n_dist_marg,
                "n_inds_used": len(inds),
            }
        )

    return per_bank_results


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


def run(
    event: Union[str, Path],
    bank_folder: Union[
        str,
        Path,
        List[Union[str, Path]],
        Tuple[Union[str, Path], ...],
    ],
    n_ext: int,
    n_phi: int,
    n_t: int,
    n_int: Union[int, List[int], Dict[str, int], None] = None,
    blocksize: int = 512,
    single_detector_blocksize: int = 512,
    n_workers: Optional[int] = None,
    n_ext_workers: int = 1,
    seed: Optional[int] = None,
    size_limit: int = 10**7,
    draw_subset: bool = True,
    n_draws: Optional[int] = None,
    event_dir: Union[str, Path, None] = None,
    rundir: Union[str, Path, None] = None,
    max_incoherent_lnlike_drop: float = 20,
    max_bestfit_lnlike_diff: float = 20,
    mchirp_guess: Optional[float] = None,
    extrinsic_samples: Union[str, Path, None] = None,
    bank_logw_override: Union[
        Dict[str, Union[List[float], pd.Series]],
        List[float],
        pd.Series,
        None,
    ] = None,
) -> Path:
    """Run inference with multiprocessing.  Drop-in for inference.run().

    Parameters
    ----------
    n_workers : int or None
        Workers for incoherent + coherent stages.  None -> os.cpu_count().
    n_ext_workers : int
        Workers for extrinsic MarginalizationInfo collection
        (Stage 4).  1 = serial.
    """
    t0 = time.time()
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    print(f"[MP] n_workers = {n_workers}")

    # ── Stage 1: setup (serial) ───────────────────────────────────────
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
    # coherent_posterior is not used beyond Stage 1 (only ctx["pr"]
    # is needed for Stage 6).  Drop it now so its likelihood can be
    # freed before Stage 2, shrinking the resident set size (RSS)
    # that forked workers inherit via copy-on-write.
    del ctx["coherent_posterior"]
    gc.collect()
    try:
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    _log_rss("after Stage 1 + del coherent_posterior")

    # ── Stage 2: incoherent selection (parallelized) ──────────────────
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

    # Free incoherent pool overhead (fragmented heap pages, inter-
    # process communication buffers) before Stage 3/4/5 so all
    # Stage 5 paths start with equally clean parent process memory,
    # regardless of whether extrinsic samples are pre-loaded or
    # freshly drawn.
    gc.collect()
    try:
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    _log_rss("after Stage 2 incoherent + gc/trim")

    # ── Stage 3: cross-bank threshold (serial) ────────────────────────
    (selected_inds_by_bank, _, _) = (
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

    # ── Stage 4: extrinsic sampling ───────────────────────────────────
    if n_ext_workers > 1 and extrinsic_samples is None:
        draw_extrinsic_samples_parallel(
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
            n_workers=n_ext_workers,
        )
    else:
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

    # Free MarginalizationInfo objects before the coherent stage.
    gc.collect()
    try:
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    _log_rss("after Stage 4 extrinsic + gc/trim")

    # ── Stage 5: coherent inference (parallelized, thin workers) ──────
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
        fbin=ctx["fbin"],
        approximant=ctx["approximant"],
    )

    # ── Stage 6: postprocess (serial) ─────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def main():
    """CLI entry point; see module docstring for usage."""
    p = argparse.ArgumentParser(
        description="dot-pe multiprocessing inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--event", required=True, help="path to event .npz file")
    p.add_argument("--bank", required=True, help="path to bank folder")
    p.add_argument(
        "--rundir",
        default=None,
        help="output directory (auto-named if omitted)",
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="parallel workers (default: cpu_count)",
    )
    p.add_argument(
        "--n-ext-workers",
        type=int,
        default=1,
        help="workers for parallel extrinsic sampling (1 = serial)",
    )
    p.add_argument("--n-ext", type=int, default=4096)
    p.add_argument("--n-phi", type=int, default=100)
    p.add_argument("--n-t", type=int, default=128)
    p.add_argument(
        "--n-int",
        type=int,
        default=None,
        help="number of intrinsic samples (default: full bank)",
    )
    p.add_argument("--blocksize", type=int, default=512)
    p.add_argument("--single-detector-blocksize", type=int, default=512)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--mchirp-guess", type=float, default=None)
    p.add_argument("--max-incoherent-lnlike-drop", type=float, default=20.0)
    p.add_argument("--max-bestfit-lnlike-diff", type=float, default=20.0)
    p.add_argument("--no-draw-subset", action="store_false", dest="draw_subset")
    p.add_argument(
        "--extrinsic-samples",
        default=None,
        help="path to cached extrinsic_samples.feather to skip re-drawing",
    )
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
        n_ext_workers=args.n_ext_workers,
        seed=args.seed,
        rundir=args.rundir,
        mchirp_guess=args.mchirp_guess,
        max_incoherent_lnlike_drop=args.max_incoherent_lnlike_drop,
        max_bestfit_lnlike_diff=args.max_bestfit_lnlike_diff,
        draw_subset=args.draw_subset,
        extrinsic_samples=args.extrinsic_samples,
    )


if __name__ == "__main__":
    main()
