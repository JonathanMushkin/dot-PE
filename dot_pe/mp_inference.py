"""Dask-based drop-in for inference.run().

Parallelises the three heavy loops via dask.distributed; everything else
delegates to dot_pe.inference.

Stage 1 — Setup:      inference.prepare_run_objects()          (serial)
Stage 2 — Incoherent: Dask futures, one per chunk              (parallel)
Stage 3 — Cross-bank: inference.select_intrinsic_...           (serial)
Stage 4 — Extrinsic:  draw_extrinsic_samples_parallel()        (parallel)
Stage 5 — Coherent:   Dask futures, one per intrinsic block    (parallel)
Stage 6 — Postprocess:inference.aggregate_and_save_results()   (serial)

Public API
----------
run(..., n_workers, scheduler_address) -> Path
    Drop-in for inference.run(); see run() docstring for extra parameters.

Backend
-------
scheduler_address=None (default)
    A dask.distributed.LocalCluster is started internally with
    ``n_workers`` worker processes (same machine, single-host).

scheduler_address="host:8786" (multi-node)
    Connects to an externally managed Dask scheduler.  The caller is
    responsible for starting scheduler + workers (e.g., via the LSF
    job script in lsf/run_inference_dask.bsub).  ``n_workers`` is
    ignored; the cluster size is whatever was launched externally.

Worker design
-------------
Stage-2 workers reload EventData from the shared filesystem (~0.1–2 s)
to avoid pickling LAL/SWIG objects.  All other setup data (par_dic_0,
fbin, m_arr, …) is passed as pickle-able scalars/arrays.

Stage-5 workers load the ~500 MB thin_setup from disk (setup_dir/*.npy
on the shared FS) to avoid sending it over the network per worker.

Profiling
---------
When ``profile=True``, workers write .prof files to ``profile_dir`` on
the shared FS.  Works unchanged across nodes.  ``run_and_profile()``
wraps ``run(profile=True)``.

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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cProfile
import pstats

import numpy as np
import pandas as pd
from tqdm import tqdm

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
# Dask worker functions — pure (no module globals), picklable
# ─────────────────────────────────────────────────────────────────────


def _incoherent_chunk_worker_inner(sample_start, sample_end, s):
    """Core incoherent computation — pure, no globals.

    ``s`` is a dict containing event_data, par_dic_0, and all other
    setup fields.  Called by ``_incoherent_chunk_worker_dask`` after
    it has loaded event_data from disk.
    """
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


def _incoherent_chunk_worker_dask(event_path, setup_dict, sample_start, sample_end):
    """Pure Dask worker — reloads EventData from the shared filesystem.

    ``setup_dict`` contains only pickle-able scalars/arrays (no EventData).
    EventData is reloaded from ``event_path`` (~0.1–2 s, negligible vs
    37–86 s compute).
    """
    import warnings as _w
    _w.filterwarnings("ignore", "Wswiglal-redir-stdio")
    from dot_pe.utils import get_event_data

    profile_dir = setup_dict.get("_profile_dir")
    if profile_dir:
        _prof = cProfile.Profile()
        _prof.enable()
    try:
        event_data = get_event_data(event_path)
        return _incoherent_chunk_worker_inner(
            sample_start,
            sample_end,
            {**setup_dict, "event_data": event_data},
        )
    finally:
        if profile_dir:
            _prof.disable()
            _prof.dump_stats(
                f"{profile_dir}/incoherent_{os.getpid()}_{sample_start}.prof"
            )


def _thin_coherent_worker_dask(
    i_block, e_blocks, waveform_dir, setup_dir, rundir, profile_dir=None
):
    """Pure Dask worker — loads thin_setup from the shared FS.

    Loads ``setup_dir/*.npy`` (~0.5–2 s) instead of receiving the
    ~500 MB dict over the network.
    """
    import warnings as _w
    _w.filterwarnings("ignore", "Wswiglal-redir-stdio")
    from dot_pe import thin_coherent as _tc

    if profile_dir:
        _prof = cProfile.Profile()
        _prof.enable()
    try:
        thin_setup = _tc.load_thin_setup(setup_dir, rundir)
        return _tc.run_thin_iblock(i_block, e_blocks, thin_setup, waveform_dir)
    finally:
        if profile_dir:
            _prof.disable()
            _prof.dump_stats(
                f"{profile_dir}/coherent_{os.getpid()}_{i_block[0]}.prof"
            )


# ─────────────────────────────────────────────────────────────────────
# Dask orchestrators
# ─────────────────────────────────────────────────────────────────────


def _collect_incoherent_dask(
    client,
    event_path,
    setup_dict,
    n_int,
    batch_size,
    n_workers,
    profile_dir=None,
):
    """Dask replacement for collect_int_samples_from_single_detectors().

    Splits [0, n_int) into chunks aligned to batch_size boundaries and
    submits one Dask future per chunk.  Returns (inds, lnlike_di,
    incoherent_lnlikes) with no threshold applied.
    """
    from dask.distributed import as_completed as _ac

    raw_chunk = max(batch_size, (n_int + n_workers - 1) // n_workers)
    chunk_size = ((raw_chunk + batch_size - 1) // batch_size) * batch_size

    sample_ranges = []
    s = 0
    while s < n_int:
        sample_ranges.append((s, min(s + chunk_size, n_int)))
        s += chunk_size

    print(f"  [Dask] incoherent: {len(sample_ranges)} chunks submitted")
    sd = {**setup_dict, "_profile_dir": profile_dir}
    futures = [
        client.submit(
            _incoherent_chunk_worker_dask, event_path, sd, s, e, pure=False
        )
        for s, e in sample_ranges
    ]

    results = []
    for f in tqdm(_ac(futures), total=len(futures), desc="incoherent chunks"):
        results.append(f.result())

    # Sort by sample_start so concatenation preserves index order
    results.sort(key=lambda r: r[0][0])

    all_inds = np.concatenate([r[0] for r in results])
    all_lnlike_di = np.concatenate([r[1] for r in results], axis=1)
    incoherent_lnlikes = all_lnlike_di.sum(axis=0)

    return all_inds, all_lnlike_di, incoherent_lnlikes


def _run_coherent_dask(
    *,
    client,
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
    fbin,
    approximant,
    profile_dir=None,
):
    """Dask replacement for run_coherent_inference_per_bank().

    Pre-computes summary weights once per bank in the main process
    (writing to setup_dir on the shared FS), then dispatches one Dask
    future per intrinsic-sample block.  Workers load setup_dir/*.npy
    (~0.5–2 s) instead of receiving the ~500 MB dict over the network.
    Results are merged incrementally as futures complete.
    """
    from dask.distributed import as_completed as _ac

    print("\n=== Coherent inference per bank (Dask, thin workers) ===")
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
        setup_dir = rundir / f"dask_coherent_setup_{bank_id}"
        print(f"  [Dask] pre-computing coherent setup for bank {bank_id}...")
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

        gc.collect()
        try:
            ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        _log_rss("after precompute + gc/trim (before Dask coherent submit)")

        waveform_dir = str(Path(bank_path) / "waveforms")
        i_blocks = inds_to_blocks(inds, blocksize)
        e_blocks = inds_to_blocks(np.arange(n_ext), blocksize)

        print(f"  [Dask] coherent: {len(i_blocks)} i_blocks submitted")
        futures = [
            client.submit(
                _thin_coherent_worker_dask,
                i_block, e_blocks, waveform_dir,
                str(setup_dir), str(rundir), profile_dir,
                pure=False,
            )
            for i_block in i_blocks
        ]

        accumulated = None
        n_disc_total = 0
        logsumexp_disc = -np.inf
        logsumsqrexp_disc = -np.inf
        cached_dt = {}
        n_dist_marg = 0

        for f in tqdm(_ac(futures), total=len(futures), desc="coherent i_blocks"):
            df, n_disc, lse_disc, lsse_disc, dt_slice, n_dm = f.result()
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
    scheduler_address: Optional[str] = None,
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
    profile: bool = False,
) -> Path:
    """Run inference with Dask.  Drop-in for inference.run().

    Parameters
    ----------
    n_workers : int or None
        Workers for incoherent + coherent stages.
        - ``scheduler_address=None``: sets the LocalCluster size.
          None → os.cpu_count().
        - ``scheduler_address=...``: ignored; cluster size is whatever
          was started externally.
    scheduler_address : str or None
        - None (default): start a ``LocalCluster`` with ``n_workers``
          processes on this machine (single-host, backward-compatible).
        - "host:port": connect to an externally managed Dask scheduler
          (multi-node LSF/Slurm jobs).
    """
    from dask.distributed import Client, LocalCluster

    t0 = time.time()
    t_stages = {}

    # Path string for Dask workers to reload EventData from shared FS
    event_path = str(event)

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    # ── Start Dask client ─────────────────────────────────────────────
    _cluster = None
    if scheduler_address is not None:
        client = Client(scheduler_address)
        _dask_n_workers = max(1, len(client.nthreads()))
        print(
            f"[Dask] connected to {scheduler_address}, "
            f"{_dask_n_workers} workers"
        )
    else:
        _cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
        client = Client(_cluster)
        _dask_n_workers = n_workers
        print(f"[Dask] LocalCluster with {n_workers} workers")

    try:
        # ── Stage 1: setup (serial) ───────────────────────────────────
        _t = time.perf_counter()
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
        del ctx["coherent_posterior"]
        gc.collect()
        try:
            ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        _log_rss("after Stage 1 + del coherent_posterior")
        t_stages["1_setup"] = time.perf_counter() - _t

        if profile:
            profiles_dir = Path(ctx["rundir"]) / "worker_profiles"
            profiles_dir.mkdir(exist_ok=True)
            profile_dir_str = str(profiles_dir)
        else:
            profiles_dir = None
            profile_dir_str = None

        # ── Stage 2: incoherent selection (Dask) ─────────────────────
        _t = time.perf_counter()
        print("\n=== Incoherent selection per bank (Dask) ===")
        candidate_inds_by_bank = {}
        lnlikes_by_bank = {}
        lnlikes_di_by_bank = {}

        for bank_id, bank_path in ctx["banks"].items():
            print(f"\nProcessing bank: {bank_id}")
            setup_dict = {
                "par_dic_0": ctx["par_dic_0"],
                "bank_folder": str(bank_path),
                "fbin": ctx["fbin"],
                "approximant": ctx["approximant"],
                "n_phi": n_phi,
                "m_arr": ctx["m_arr"],
                "n_t": n_t,
                "size_limit": size_limit,
                "batch_size": single_detector_blocksize,
            }
            inds, lnlike_di, incoherent_lnlikes = _collect_incoherent_dask(
                client=client,
                event_path=event_path,
                setup_dict=setup_dict,
                n_int=ctx["n_int_dict"][bank_id],
                batch_size=single_detector_blocksize,
                n_workers=_dask_n_workers,
                profile_dir=profile_dir_str,
            )
            candidate_inds_by_bank[bank_id] = inds
            lnlikes_by_bank[bank_id] = incoherent_lnlikes
            lnlikes_di_by_bank[bank_id] = lnlike_di
            print(f"Bank {bank_id}: {len(inds)} intrinsic samples evaluated.")

        gc.collect()
        try:
            ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        _log_rss("after Stage 2 incoherent + gc/trim")
        t_stages["2_incoherent"] = time.perf_counter() - _t

        # ── Stage 3: cross-bank threshold (serial) ────────────────────
        _t = time.perf_counter()
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
        t_stages["3_crossbank"] = time.perf_counter() - _t

        # ── Stage 4: extrinsic sampling (Dask) ───────────────────────
        _t = time.perf_counter()
        if extrinsic_samples is None:
            draw_extrinsic_samples_parallel(
                client=client,
                event_path=event_path,
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
                n_workers=_dask_n_workers,
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

        gc.collect()
        try:
            ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        _log_rss("after Stage 4 extrinsic + gc/trim")
        t_stages["4_extrinsic"] = time.perf_counter() - _t

        # ── Stage 5: coherent inference (Dask, thin workers) ──────────
        _t = time.perf_counter()
        per_bank_results = _run_coherent_dask(
            client=client,
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
            fbin=ctx["fbin"],
            approximant=ctx["approximant"],
            profile_dir=profile_dir_str,
        )
        t_stages["5_coherent"] = time.perf_counter() - _t

        # ── Stage 6: postprocess (serial) ─────────────────────────────
        _t = time.perf_counter()
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
        t_stages["6_postprocess"] = time.perf_counter() - _t

    finally:
        client.close()
        if _cluster is not None:
            try:
                _cluster.close()
            except Exception:
                pass  # workers running C extensions may not exit cleanly

    if profile:
        prof_files = sorted(profiles_dir.glob("*.prof"))
        if prof_files:
            combined = pstats.Stats(str(prof_files[0]))
            for p in prof_files[1:]:
                combined.add(str(p))
            combined.dump_stats(Path(ctx["rundir"]) / "profile_output.prof")
            with open(Path(ctx["rundir"]) / "profile_output.txt", "w") as f:
                ps = pstats.Stats(str(prof_files[0]), stream=f)
                for p in prof_files[1:]:
                    ps.add(str(p))
                ps.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats()
            print(
                f"[profile] {len(prof_files)} worker profiles merged -> "
                f"{ctx['rundir']}/profile_output.prof"
            )

    print("\nStage timings:")
    labels = {
        "1_setup":       "Stage 1 (setup)",
        "2_incoherent":  "Stage 2 (incoherent)",
        "3_crossbank":   "Stage 3 (cross-bank)",
        "4_extrinsic":   "Stage 4 (extrinsic)",
        "5_coherent":    "Stage 5 (coherent)",
        "6_postprocess": "Stage 6 (postprocess)",
    }
    for key, label in labels.items():
        print(f"  {label:<28} {t_stages.get(key, 0):.1f} s")

    print(f"\nTotal wall-clock time: {time.time() - t0:.1f} s")
    return result


def run_and_profile(**kwargs) -> Path:
    """
    Drop-in for run() that profiles all worker processes.

    Writes to rundir:
      worker_profiles/incoherent_<pid>_<start>.prof  — one per incoherent worker
      worker_profiles/coherent_<pid>_<iblock>.prof   — one per coherent worker
      profile_output.prof   — merged binary (view with snakeviz)
      profile_output.txt    — merged human-readable, sorted by cumtime
    Also prints a stage-level wall-clock table.
    """
    return run(**kwargs, profile=True)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def main():
    """CLI entry point; see module docstring for usage."""
    p = argparse.ArgumentParser(
        description="dot-pe Dask inference",
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
        help="LocalCluster workers (default: cpu_count; ignored with --scheduler-address)",
    )
    p.add_argument(
        "--scheduler-address",
        default=None,
        help="Dask scheduler address (host:port) for multi-node runs. "
             "If omitted, a LocalCluster is started on this machine.",
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
        scheduler_address=args.scheduler_address,
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
