"""End-to-end dot-PE run on one GPU: drop-in for ``dot_pe.mp_inference.run``.

Stages 1 (setup), 3 (cross-bank threshold), 4 (extrinsic proposal) and 6 (postprocess) call the
same CPU functions as ``mp_inference.run``. Stage 2 (incoherent selection) and stage 5 (coherent
stage) run on the GPU (``stages.incoherent_bank_gpu``, ``coherent_stage.coherent_bank_gpu``); the
waveforms read in stage 2 stay on the device for stage 5 (``stages.Residency``). The rundir gets
the same files as a CPU run except ``mp_coherent_setup_<bank>/``, plus ``gpu_timing.json``.

Command line: ``python -m dot_pe_gpu.run --event ... --bank-folder ... --n-ext ... --n-phi ...
--n-t ... [--row-selection ...] [--rundir ...]`` (see ``--help``).
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np


def _stage_print(t_stages, t0):
    labels = {"1_setup": "Stage 1 (setup)", "2_incoherent": "Stage 2 (incoherent)",
              "3_crossbank": "Stage 3 (cross-bank)", "4_extrinsic": "Stage 4 (extrinsic)",
              "5_coherent": "Stage 5 (coherent)", "6_postprocess": "Stage 6 (postprocess)"}
    print("\nStage timings:")
    for key, label in labels.items():
        print(f"  {label:<28} {t_stages.get(key, 0):.1f} s")
    print(f"\nTotal wall-clock time: {time.time() - t0:.1f} s")


def run(event, bank_folder, n_ext, n_phi, n_t, n_int=None, blocksize=512, single_detector_blocksize=512,
        seed=None, size_limit=10 ** 7, draw_subset=True, n_draws=None, event_dir=None, rundir=None,
        max_incoherent_lnlike_drop=20, min_incoherent_survivors=100, max_bestfit_lnlike_diff=20,
        mchirp_guess=None, extrinsic_samples=None, bank_logw_override=None, load_inds=False,
        inds_path=None, seed_pool_lnlike_drop=None, seed_pool_inds=None, row_selection="bestfit",
        preselected_indices=None, n_ext_workers=1, n_workers=None, profile=False, reader_threads=32,
        stop_after_setup=False, residency_max_gb=None, residency=True):
    """Same arguments and outputs as ``dot_pe.mp_inference.run``; ``n_workers`` and ``profile`` are
    accepted and ignored. ``stop_after_setup``: run stage 1 only (a CPU dry run).
    ``residency_max_gb``: cap on the device memory that keeps selected waveforms between stages 2
    and 5 (default: free device memory minus 12 GB); past it, stage 5 streams them from disk.
    ``residency=False`` always streams. Returns the rundir."""
    import cupy as cp
    from dot_pe import inference, priority_sampling
    from dot_pe.mp_inference import draw_extrinsic_samples_parallel, explicit_seed_pool, restrict_seed_pool
    from .coherent_stage import coherent_bank_gpu
    from .lookup_table import GPULookupTable
    from .stages import H_SCALE, Residency, incoherent_bank_gpu

    if seed_pool_inds is not None and seed_pool_lnlike_drop is not None:
        raise ValueError("seed_pool_inds and seed_pool_lnlike_drop are exclusive")
    if bank_logw_override is not None:
        raise NotImplementedError("bank_logw_override is not supported on the GPU path yet")
    t0 = time.time()
    t_stages, detail = {}, {}

    # ── Stage 1: setup (CPU) ────────────────────────────────────────────
    _t = time.perf_counter()
    ctx = inference.prepare_run_objects(
        event=event, bank_folder=bank_folder, n_int=n_int, n_ext=n_ext, n_phi=n_phi, n_t=n_t,
        blocksize=blocksize, single_detector_blocksize=single_detector_blocksize, i_int_start=0,
        seed=seed, load_inds=load_inds, inds_path=inds_path, size_limit=size_limit,
        draw_subset=draw_subset, n_draws=n_draws, event_dir=event_dir, rundir=rundir,
        coherent_score_min_n_effective_prior=100, max_incoherent_lnlike_drop=max_incoherent_lnlike_drop,
        min_incoherent_survivors=min_incoherent_survivors, max_bestfit_lnlike_diff=max_bestfit_lnlike_diff,
        mchirp_guess=mchirp_guess, extrinsic_samples=extrinsic_samples, n_phi_incoherent=None,
        preselected_indices=preselected_indices, bank_logw_override=None, coherent_posterior_kwargs={},
        extra_run_kwargs={
            "seed_pool_lnlike_drop": None if seed_pool_lnlike_drop is None else float(seed_pool_lnlike_drop),
            "seed_pool_inds": (None if seed_pool_inds is None
                               else {b: [int(i) for i in v] for b, v in seed_pool_inds.items()}),
            "row_selection": priority_sampling.check_mode(row_selection),
            "gpu": True,
        },
    )
    del ctx["coherent_posterior"]
    gc.collect()
    t_stages["1_setup"] = time.perf_counter() - _t
    if stop_after_setup:
        print(f"Stage 1 done in {t_stages['1_setup']:.1f} s; stopping (stop_after_setup).")
        return ctx["rundir"]
    print("GPU:", cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode())

    # ── Stage 2: incoherent selection (GPU) ─────────────────────────────
    _t = time.perf_counter()
    cand, lnl_by_bank, lnl_di_by_bank = {}, {}, {}
    resid = None
    if load_inds:
        if inds_path is None:
            raise ValueError("load_inds=True requires inds_path to be set")
        paths = ({k: Path(v) for k, v in inds_path.items()} if isinstance(inds_path, dict)
                 else {next(iter(ctx["banks"])): Path(inds_path)})
        for bank_id in ctx["banks"]:
            z = np.load(paths[bank_id], allow_pickle=False)
            cand[bank_id], lnl_by_bank[bank_id], lnl_di_by_bank[bank_id] = (
                z["inds"], z["incoherent_lnlikes"], z["lnlikes_di"])
    else:
        if residency_max_gb is None:
            free, _ = cp.cuda.runtime.memGetInfo()
            residency_max_gb = max(0.0, free / 1e9 - 12.0)
        resid = (Residency(margin=max_incoherent_lnlike_drop, top_n=min_incoherent_survivors,
                           max_bytes=residency_max_gb * 1e9) if residency else None)
        detail["residency_max_gb"] = residency_max_gb
        print("\n=== Incoherent selection per bank (GPU) ===")
        for bank_id, bank_path in ctx["banks"].items():
            pre = None
            if ctx.get("preselected_indices_dict") is not None:
                pre = ctx["preselected_indices_dict"].get(bank_id)
                if isinstance(pre, (str, Path)):
                    pre = np.load(pre)
            inds, lnl_di, lnl = incoherent_bank_gpu(
                event_data=ctx["event_data"], par_dic_0=ctx["par_dic_0"], bank_folder=Path(bank_path),
                fbin=ctx["fbin"], approximant=ctx["approximant"], m_arr=ctx["m_arr"], n_phi=n_phi, n_t=n_t,
                n_int=ctx["n_int_dict"][bank_id], preselected_indices=pre, size_limit=size_limit,
                residency=resid, bank_id=bank_id, threads=reader_threads, timings=detail)
            if resid is not None and resid.overflowed and "residency_overflow_bank" not in detail:
                detail["residency_overflow_bank"] = bank_id
            cand[bank_id], lnl_by_bank[bank_id], lnl_di_by_bank[bank_id] = inds, lnl, lnl_di
            print(f"Bank {bank_id}: {len(inds)} intrinsic samples evaluated (device residency "
                  + ("off" if resid is None else "over its cap" if resid.overflowed
                     else f"{resid.nbytes() / 1e9:.2f} GB") + ").")
    t_stages["2_incoherent"] = time.perf_counter() - _t

    # ── Stage 3: cross-bank threshold (CPU) ─────────────────────────────
    _t = time.perf_counter()
    sel_inds, sel_lnl, _ = inference.select_intrinsic_samples_across_banks_by_incoherent_likelihood(
        banks=ctx["banks"], candidate_inds_by_bank=cand, incoherent_lnlikes_by_bank=lnl_by_bank,
        lnlikes_di_by_bank=lnl_di_by_bank, max_incoherent_lnlike_drop=max_incoherent_lnlike_drop,
        banks_dir=ctx["banks_dir"], event_data=ctx["event_data"],
        min_incoherent_survivors=min_incoherent_survivors)
    t_stages["3_crossbank"] = time.perf_counter() - _t
    seed_inds = sel_inds
    if seed_pool_lnlike_drop is not None:
        seed_inds = restrict_seed_pool(sel_inds, sel_lnl, seed_pool_lnlike_drop)
    elif seed_pool_inds is not None:
        seed_inds = explicit_seed_pool(sel_inds, seed_pool_inds)

    # ── Stage 4: extrinsic proposal (CPU) ───────────────────────────────
    _t = time.perf_counter()
    common = dict(banks=ctx["banks"], event_data=ctx["event_data"], par_dic_0=ctx["par_dic_0"],
                  fbin=ctx["fbin"], approximant=ctx["approximant"], selected_inds_by_bank=seed_inds,
                  coherent_score_kwargs=ctx["coherent_score_kwargs"], seed=seed, n_ext=n_ext,
                  rundir=ctx["rundir"])
    if n_ext_workers > 1 and extrinsic_samples is None:
        draw_extrinsic_samples_parallel(**common, n_workers=n_ext_workers)
    else:
        inference.draw_extrinsic_samples(**common, extrinsic_samples=extrinsic_samples)
    gc.collect()
    t_stages["4_extrinsic"] = time.perf_counter() - _t

    # ── Stage 5: coherent stage (GPU) ───────────────────────────────────
    _t = time.perf_counter()
    print("\n=== Coherent inference per bank (GPU) ===")
    lookup = GPULookupTable()
    per_bank = []
    for k, (bank_id, bank_path) in enumerate(ctx["banks"].items()):
        inds = np.asarray(sel_inds[bank_id], np.int64)
        order = np.argsort(inds)
        inds = inds[order]
        lnl = np.asarray(sel_lnl[bank_id])[order]
        h = None
        if resid is not None and len(inds):
            h = resid.finalize(bank_id, inds, scaled=True)
        print(f"Bank {bank_id}: {len(inds)} selected templates "
              f"({'device-resident' if h is not None else 'streamed from disk'}).")
        per_bank.append(coherent_bank_gpu(
            bank_id=bank_id, bank_index=k, bank_path=Path(bank_path), event_data=ctx["event_data"],
            par_dic_0=ctx["par_dic_0"], fbin=ctx["fbin"], approximant=ctx["approximant"], m_arr=ctx["m_arr"],
            n_phi=n_phi, n_ext=n_ext, n_int_k=ctx["n_int_dict"][bank_id], inds=inds, h_raw_scaled=h,
            h_scale=H_SCALE, blocksize=blocksize, size_limit=size_limit,
            max_bestfit_lnlike_diff=max_bestfit_lnlike_diff, rundir=ctx["rundir"],
            bank_rundir=ctx["banks_dir"] / bank_id, row_selection=row_selection, selection_seed=seed,
            selected_lnlikes=lnl, lookup=lookup, timings=detail, reader_threads=reader_threads))
        del h
        cp.get_default_memory_pool().free_all_blocks()
    t_stages["5_coherent"] = time.perf_counter() - _t

    # ── Stage 6: postprocess (CPU) ──────────────────────────────────────
    _t = time.perf_counter()
    result = inference.aggregate_and_save_results(
        per_bank_results=per_bank, banks=ctx["banks"], event_data=ctx["event_data"], rundir=ctx["rundir"],
        banks_dir=ctx["banks_dir"], n_phi=n_phi, pr=ctx["pr"], n_draws=n_draws, draw_subset=draw_subset,
        size_limit=size_limit, row_selection=row_selection)
    t_stages["6_postprocess"] = time.perf_counter() - _t

    json.dump({"stages_seconds": t_stages, "detail": detail, "wall_seconds": time.time() - t0,
               "gpu": cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode()},
              open(Path(ctx["rundir"]) / "gpu_timing.json", "w"), indent=1, default=float)
    _stage_print(t_stages, t0)
    return result


def main():
    ap = argparse.ArgumentParser(description="dot-PE on one GPU (drop-in for mp_inference)")
    ap.add_argument("--event", required=True)
    ap.add_argument("--bank-folder", nargs="+", required=True)
    ap.add_argument("--n-ext", type=int, required=True)
    ap.add_argument("--n-phi", type=int, required=True)
    ap.add_argument("--n-t", type=int, required=True)
    ap.add_argument("--n-int", type=int, default=None)
    ap.add_argument("--blocksize", type=int, default=512)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--size-limit", type=int, default=10 ** 7)
    ap.add_argument("--rundir", default=None)
    ap.add_argument("--event-dir", default=None)
    ap.add_argument("--mchirp-guess", type=float, default=None)
    ap.add_argument("--extrinsic-samples", default=None)
    ap.add_argument("--row-selection", default="bestfit")
    ap.add_argument("--max-incoherent-lnlike-drop", type=float, default=20)
    ap.add_argument("--max-bestfit-lnlike-diff", type=float, default=20)
    ap.add_argument("--seed-pool-lnlike-drop", type=float, default=None)
    ap.add_argument("--kwargs-json", default=None,
                    help="JSON file of run() keyword arguments (overrides the flags above)")
    a = ap.parse_args()
    kw = dict(event=a.event, bank_folder=a.bank_folder if len(a.bank_folder) > 1 else a.bank_folder[0],
              n_ext=a.n_ext, n_phi=a.n_phi, n_t=a.n_t, n_int=a.n_int, blocksize=a.blocksize, seed=a.seed,
              size_limit=a.size_limit, rundir=a.rundir, event_dir=a.event_dir, mchirp_guess=a.mchirp_guess,
              extrinsic_samples=a.extrinsic_samples, row_selection=a.row_selection,
              max_incoherent_lnlike_drop=a.max_incoherent_lnlike_drop,
              max_bestfit_lnlike_diff=a.max_bestfit_lnlike_diff, seed_pool_lnlike_drop=a.seed_pool_lnlike_drop)
    if a.kwargs_json:
        kw.update(json.load(open(a.kwargs_json)))
    run(**kw)


if __name__ == "__main__":
    main()
