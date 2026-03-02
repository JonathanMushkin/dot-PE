#!/usr/bin/env python3
"""
LSF swarm orchestrator for dot-pe inference.

Chains 5 stages, submitting LSF array jobs for the two heavy loops:
  Stage 1  setup + serialize inputs           (this process, ~60 s)
  Stage 2  incoherent scoring swarm           (LSF array, physics-short)
  Stage 2b merge + threshold                  (this process, ~5 s)
  Stage 3  extrinsic sampling                 (this process, ~549 s)
  Stage 4  coherent inference swarm           (LSF array, physics-short)
  Stage 5  merge + postprocess + save         (this process, ~10 s)

Marker files (stage_N.done) allow resuming a failed run without re-doing
completed stages — just re-run the same command.

──────────────────────────────────────────────────────────────────────────────
QUICK-START (copy-paste into your terminal / physics-medium LSF job)
──────────────────────────────────────────────────────────────────────────────

1. Create test data once (from the project root):

    python test_data/setup.py --n-pool 4

2. Smoke-test with the small bank (submit as a physics-medium job):

    bsub -q physics-medium -n 1 -W 60 \\
         -o swarm_%J.out -e swarm_%J.err \\
         python lsf_swarm/run_swarm.py \\
             --event  test_data/event/tutorial_event.npz \\
             --bank   test_data/bank_small \\
             --rundir /tmp/swarm_smoke \\
             --n-ext 512 --n-phi 50

3. Timing benchmark with the large bank:

    bsub -q physics-medium -n 1 -W 120 \\
         -o swarm_%J.out -e swarm_%J.err \\
         python lsf_swarm/run_swarm.py \\
             --event  test_data/event/tutorial_event.npz \\
             --bank   test_data/bank_large \\
             --rundir /tmp/swarm_bench \\
             --n-ext 4096 --n-phi 100

──────────────────────────────────────────────────────────────────────────────
WHAT TO REPORT AFTER A TEST RUN
──────────────────────────────────────────────────────────────────────────────
After each run, please report:
  1. Total wall-clock time (printed as "Total wall-clock time: X s" at the end
     of the orchestrator's output / LSF .out file)
  2. Time per stage (printed at the start of each stage)
  3. The contents of <rundir>/run_N/summary_results.json
  4. Any errors or unexpected warnings

Compare against the serial baseline (inference.run_and_profile with the
same bank/event/n-ext/n-phi) to measure speedup.
"""

# ── thread override ───────────────────────────────────────────────────────────
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import math
import pickle
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cogwheel.utils import exp_normalize
from dot_pe import inference
from dot_pe.base_sampler_free_sampling import get_n_effective_total_i_e
from dot_pe.coherent_processing import CoherentLikelihoodProcessor
from dot_pe.utils import inds_to_blocks, safe_logsumexp


# ─────────────────────────────────────────────────────────────────────────────
# LSF helpers
# ─────────────────────────────────────────────────────────────────────────────

_WORKER_DIR = Path(__file__).resolve().parent


def _bsub(job_name, n_jobs, worker_cmd, rundir, queue="physics-short",
          max_concurrent=20, memory_mb=6000):
    """
    Submit an LSF array job and return the integer job ID.

    worker_cmd should contain the literal string '$LSB_JOBINDEX' where
    the 1-indexed task number should be substituted by LSF.
    """
    logs_dir = rundir / "logs"
    logs_dir.mkdir(exist_ok=True)

    script = f"""\
#!/bin/bash
#BSUB -J "{job_name}[1-{n_jobs}]%{max_concurrent}"
#BSUB -q {queue}
#BSUB -n 1
#BSUB -R "rusage[mem={memory_mb}]"
#BSUB -o {logs_dir}/{job_name}_%J_%I.out
#BSUB -e {logs_dir}/{job_name}_%J_%I.err

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

{worker_cmd}
"""
    result = subprocess.run(
        ["bsub"],
        input=script,
        capture_output=True,
        text=True,
        check=True,
    )
    m = re.search(r"Job <(\d+)>", result.stdout)
    if not m:
        raise RuntimeError(f"Could not parse job ID from bsub output:\n{result.stdout}")
    job_id = int(m.group(1))
    print(f"  [LSF] submitted {job_name}[1-{n_jobs}], job_id={job_id}")
    return job_id


def _wait_for_job(job_id, poll_interval=30):
    """Block until all tasks of job_id are DONE or EXIT; raise on any EXIT."""
    print(f"  [LSF] waiting for job {job_id}...", flush=True)
    while True:
        result = subprocess.run(
            ["bjobs", "-noheader", str(job_id)],
            capture_output=True, text=True,
        )
        lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
        if not lines:
            break  # job no longer tracked (all DONE)
        active = [l for l in lines if any(s in l for s in ("RUN", "PEND", "SSUSP", "USUSP"))]
        if not active:
            break
        print(f"  [LSF] job {job_id}: {len(active)} tasks still active, "
              f"sleeping {poll_interval} s...", flush=True)
        time.sleep(poll_interval)

    # Check for failures
    result = subprocess.run(
        ["bjobs", "-noheader", "-a", str(job_id)],
        capture_output=True, text=True,
    )
    n_failed = sum(1 for l in result.stdout.splitlines() if "EXIT" in l)
    if n_failed:
        raise RuntimeError(f"Job {job_id}: {n_failed} tasks exited with error — check logs")


# ─────────────────────────────────────────────────────────────────────────────
# Stage helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stage_done(swarm_setup_dir, n):
    return (swarm_setup_dir / f"stage_{n}.done").exists()


def _mark_done(swarm_setup_dir, n):
    (swarm_setup_dir / f"stage_{n}.done").touch()


# ─────────────────────────────────────────────────────────────────────────────
# Main run
# ─────────────────────────────────────────────────────────────────────────────

def run(
    event,
    bank_folder,
    n_ext,
    n_phi,
    n_t=128,
    n_int=None,
    blocksize=512,
    incoherent_blocksize=512,
    blocksize_per_job=4096,
    seed=None,
    size_limit=10**7,
    draw_subset=True,
    n_draws=None,
    event_dir=None,
    rundir=None,
    max_incoherent_lnlike_drop=20,
    max_bestfit_lnlike_diff=20,
    mchirp_guess=None,
    worker_queue="physics-short",
    max_concurrent=20,
):
    """
    Run LSF swarm inference.  Single-bank only in this version.

    All intermediate files go under {rundir}/swarm_setup/.
    Resumable: completed stages are skipped via marker files.
    """
    t0 = time.time()

    # ── Stage 1: setup ────────────────────────────────────────────────────────
    ctx = inference.prepare_run_objects(
        event=event,
        bank_folder=bank_folder,
        n_int=n_int,
        n_ext=n_ext,
        n_phi=n_phi,
        n_t=n_t,
        blocksize=blocksize,
        single_detector_blocksize=incoherent_blocksize,
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
        extrinsic_samples=None,
        n_phi_incoherent=None,
        preselected_indices=None,
        bank_logw_override=None,
        coherent_posterior_kwargs={},
    )

    rundir       = ctx["rundir"]
    swarm_dir    = rundir / "swarm_setup"
    swarm_dir.mkdir(exist_ok=True)
    (swarm_dir / "incoherent").mkdir(exist_ok=True)
    (swarm_dir / "coherent").mkdir(exist_ok=True)

    # Single-bank: take the first (and only) bank
    if len(ctx["banks"]) > 1:
        raise NotImplementedError(
            "run_swarm.py currently supports single-bank runs only. "
            "For multi-bank, run each bank separately."
        )
    bank_id   = list(ctx["banks"].keys())[0]
    bank_path = Path(ctx["banks"][bank_id])
    n_int_k   = ctx["n_int_dict"][bank_id]

    if not _stage_done(swarm_dir, 1):
        print("\n=== Stage 1: serializing setup for workers ===")
        with open(swarm_dir / "event_data.pkl", "wb") as f:
            pickle.dump(ctx["event_data"], f)
        with open(swarm_dir / "par_dic_0.json", "w") as f:
            json.dump(ctx["par_dic_0"], f)
        swarm_cfg = {
            "bank_folder":         str(bank_path.resolve()),
            "fbin":                ctx["fbin"].tolist(),
            "approximant":         ctx["approximant"],
            "m_arr":               ctx["m_arr"].tolist(),
            "n_phi":               n_phi,
            "n_t":                 n_t,
            "n_ext":               n_ext,
            "n_int":               n_int_k,
            "blocksize":           blocksize,
            "blocksize_per_job":   blocksize_per_job,
            "incoherent_blocksize": incoherent_blocksize,
            "size_limit":          size_limit,
            "max_incoherent_lnlike_drop": max_incoherent_lnlike_drop,
            "max_bestfit_lnlike_diff":    max_bestfit_lnlike_diff,
        }
        with open(swarm_dir / "swarm_config.json", "w") as f:
            json.dump(swarm_cfg, f, indent=2)
        _mark_done(swarm_dir, 1)
        print(f"  Stage 1 done ({time.time()-t0:.0f} s elapsed)")
    else:
        print("=== Stage 1: [skip] already done ===")
        with open(swarm_dir / "swarm_config.json") as f:
            swarm_cfg = json.load(f)

    # ── Stage 2: incoherent swarm ─────────────────────────────────────────────
    n_incoh_jobs = math.ceil(n_int_k / blocksize_per_job)

    if not _stage_done(swarm_dir, 2):
        print(f"\n=== Stage 2: incoherent swarm ({n_incoh_jobs} jobs) ===")
        worker_cmd = (
            f"python {_WORKER_DIR}/worker_incoherent.py "
            f"--rundir {rundir} --block-id $LSB_JOBINDEX"
        )
        job_id = _bsub(
            job_name="incoh",
            n_jobs=n_incoh_jobs,
            worker_cmd=worker_cmd,
            rundir=rundir,
            queue=worker_queue,
            max_concurrent=max_concurrent,
        )
        _wait_for_job(job_id)

        # Verify all output files exist
        missing = [
            i for i in range(1, n_incoh_jobs + 1)
            if not (swarm_dir / "incoherent" / f"block_{i}.npz").exists()
        ]
        if missing:
            raise RuntimeError(f"Missing incoherent output files for blocks: {missing}")

        _mark_done(swarm_dir, 2)
        print(f"  Stage 2 done ({time.time()-t0:.0f} s elapsed)")
    else:
        print("=== Stage 2: [skip] already done ===")

    # ── Stage 2b: merge incoherent blocks + threshold ─────────────────────────
    if not _stage_done(swarm_dir, "2b"):
        print("\n=== Stage 2b: merging incoherent blocks + threshold ===")
        all_inds, all_lnlike_di = [], []
        for i in range(1, n_incoh_jobs + 1):
            d = np.load(swarm_dir / "incoherent" / f"block_{i}.npz")
            all_inds.append(d["inds"])
            all_lnlike_di.append(d["lnlike_di"])
        all_inds     = np.concatenate(all_inds)
        all_lnlike_di = np.concatenate(all_lnlike_di, axis=1)
        incoherent_lnlikes = all_lnlike_di.sum(axis=0)

        # Apply global threshold
        global_max       = incoherent_lnlikes.max()
        threshold        = global_max - max_incoherent_lnlike_drop
        selected_mask    = incoherent_lnlikes >= threshold
        selected_inds    = all_inds[selected_mask]

        print(f"  Global max lnlike: {global_max:.2f}, threshold: {threshold:.2f}")
        print(f"  Selected {len(selected_inds)} / {len(all_inds)} intrinsic samples")

        np.save(swarm_dir / "selected_inds.npy", selected_inds)

        # Also write to banks_dir (for aggregate_and_save_results compatibility)
        bank_rundir = ctx["banks_dir"] / bank_id
        bank_rundir.mkdir(exist_ok=True)
        selected_lnlikes_di = all_lnlike_di[:, selected_mask]
        np.savez(
            bank_rundir / "intrinsic_samples.npz",
            inds=selected_inds,
            lnlikes_di=selected_lnlikes_di,
            incoherent_lnlikes=incoherent_lnlikes[selected_mask],
        )

        _mark_done(swarm_dir, "2b")
        print(f"  Stage 2b done ({time.time()-t0:.0f} s elapsed)")
    else:
        print("=== Stage 2b: [skip] already done ===")
        selected_inds = np.load(swarm_dir / "selected_inds.npy")
        print(f"  {len(selected_inds)} selected intrinsic samples")

    # ── Stage 3: extrinsic sampling (serial) ──────────────────────────────────
    if not _stage_done(swarm_dir, 3):
        print(f"\n=== Stage 3: extrinsic sampling ({n_ext} samples) ===")
        selected_inds_by_bank = {bank_id: selected_inds}
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
            rundir=rundir,
            extrinsic_samples=None,
        )
        _mark_done(swarm_dir, 3)
        print(f"  Stage 3 done ({time.time()-t0:.0f} s elapsed)")
    else:
        print("=== Stage 3: [skip] already done ===")

    # ── Stage 4: coherent swarm ───────────────────────────────────────────────
    i_blocks     = inds_to_blocks(selected_inds, blocksize)
    n_coh_jobs   = len(i_blocks)

    if not _stage_done(swarm_dir, 4):
        print(f"\n=== Stage 4: coherent swarm ({n_coh_jobs} i_blocks) ===")
        worker_cmd = (
            f"python {_WORKER_DIR}/worker_coherent.py "
            f"--rundir {rundir} --i-block-idx $LSB_JOBINDEX"
        )
        job_id = _bsub(
            job_name="coh",
            n_jobs=n_coh_jobs,
            worker_cmd=worker_cmd,
            rundir=rundir,
            queue=worker_queue,
            max_concurrent=max_concurrent,
        )
        _wait_for_job(job_id)

        missing = [
            i for i in range(1, n_coh_jobs + 1)
            if not (swarm_dir / "coherent" / f"i_{i}.npz").exists()
        ]
        if missing:
            raise RuntimeError(f"Missing coherent output files for i_blocks: {missing}")

        _mark_done(swarm_dir, 4)
        print(f"  Stage 4 done ({time.time()-t0:.0f} s elapsed)")
    else:
        print("=== Stage 4: [skip] already done ===")

    # ── Stage 5: merge coherent + postprocess ────────────────────────────────
    if not _stage_done(swarm_dir, 5):
        print("\n=== Stage 5: merging coherent results + postprocess ===")

        partial_dfs, cached_dt = [], {}
        n_disc_total = 0
        logsumexp_disc = -np.inf
        logsumsqrexp_disc = -np.inf
        n_dist_marg = 0

        for i in range(1, n_coh_jobs + 1):
            d = np.load(swarm_dir / "coherent" / f"i_{i}.npz")
            n = len(d["i"])
            if n > 0:
                partial_dfs.append(pd.DataFrame({
                    "i":               d["i"],
                    "e":               d["e"],
                    "o":               d["o"],
                    "lnl_marginalized": d["lnl_marginalized"],
                    "ln_posterior":    d["ln_posterior"],
                    "bestfit_lnlike":  d["bestfit_lnlike"],
                    "d_h_1Mpc":        d["d_h_1Mpc"],
                    "h_h_1Mpc":        d["h_h_1Mpc"],
                }))
            n_disc_total  += int(d["n_samples_discarded"])
            logsumexp_disc = safe_logsumexp([float(d["logsumexp_discarded"]), logsumexp_disc])
            logsumsqrexp_disc = safe_logsumexp([float(d["logsumsqrexp_discarded"]), logsumsqrexp_disc])
            n_dist_marg   += int(d["n_distance_marginalizations"])
            for k, v in zip(d["cached_dt_keys"], d["cached_dt_vals"]):
                cached_dt[int(k)] = float(v)

        combined = (
            pd.concat(partial_dfs, ignore_index=True)
            if partial_dfs
            else pd.DataFrame(columns=CoherentLikelihoodProcessor.PROB_SAMPLES_COLS)
        )
        combined["weights"] = exp_normalize(combined["ln_posterior"].values)

        # Save per-bank prob_samples + cache for aggregate_and_save_results
        bank_rundir = ctx["banks_dir"] / bank_id
        bank_rundir.mkdir(parents=True, exist_ok=True)
        combined.to_feather(bank_rundir / "prob_samples.feather")
        with open(bank_rundir / "intrinsic_sample_processor_cache.json", "w") as f:
            json.dump({int(k): float(v) for k, v in cached_dt.items()}, f)

        n_total_samples = n_phi * n_ext * n_int_k
        ln_evidence     = safe_logsumexp(combined["ln_posterior"].values) - np.log(n_total_samples)
        ln_evidence_disc = logsumexp_disc - np.log(n_total_samples)
        n_eff, n_eff_i, n_eff_e = get_n_effective_total_i_e(combined, assume_normalized=False)

        per_bank_results = [{
            "bank_id":                         bank_id,
            "lnZ_k":                           ln_evidence,
            "lnZ_discarded_k":                 ln_evidence_disc,
            "N_k":                             n_total_samples,
            "n_effective_k":                   n_eff,
            "n_effective_i_k":                 n_eff_i,
            "n_effective_e_k":                 n_eff_e,
            "n_distance_marginalizations_k":   n_dist_marg,
            "n_inds_used":                     len(selected_inds),
        }]

        inference.aggregate_and_save_results(
            per_bank_results=per_bank_results,
            banks=ctx["banks"],
            event_data=ctx["event_data"],
            rundir=rundir,
            banks_dir=ctx["banks_dir"],
            n_phi=n_phi,
            pr=ctx["pr"],
            n_draws=n_draws,
            draw_subset=draw_subset,
        )

        _mark_done(swarm_dir, 5)
        print(f"  Stage 5 done ({time.time()-t0:.0f} s elapsed)")
    else:
        print("=== Stage 5: [skip] already done ===")

    print(f"\nTotal wall-clock time: {time.time()-t0:.1f} s")
    return rundir


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="dot-pe LSF swarm inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--event",    required=True, help="path to event .npz")
    p.add_argument("--bank",     required=True, help="path to bank folder")
    p.add_argument("--rundir",   default=None,  help="output dir (auto-named if omitted)")
    p.add_argument("--n-ext",    type=int, default=4096)
    p.add_argument("--n-phi",    type=int, default=100)
    p.add_argument("--n-t",      type=int, default=128)
    p.add_argument("--n-int",    type=int, default=None, help="intrinsic samples (default: full bank)")
    p.add_argument("--blocksize", type=int, default=512)
    p.add_argument("--incoherent-blocksize", type=int, default=512,
                   help="batch size inside each incoherent worker")
    p.add_argument("--blocksize-per-job", type=int, default=4096,
                   help="intrinsic samples per incoherent worker job")
    p.add_argument("--seed",     type=int, default=None)
    p.add_argument("--mchirp-guess", type=float, default=None)
    p.add_argument("--max-incoherent-lnlike-drop", type=float, default=20.0)
    p.add_argument("--max-bestfit-lnlike-diff",    type=float, default=20.0)
    p.add_argument("--no-draw-subset", action="store_false", dest="draw_subset")
    p.add_argument("--worker-queue", default="physics-short",
                   help="LSF queue for array workers")
    p.add_argument("--max-concurrent", type=int, default=20,
                   help="max simultaneously running worker tasks")
    args = p.parse_args()

    run(
        event=args.event,
        bank_folder=args.bank,
        n_ext=args.n_ext,
        n_phi=args.n_phi,
        n_t=args.n_t,
        n_int=args.n_int,
        blocksize=args.blocksize,
        incoherent_blocksize=args.incoherent_blocksize,
        blocksize_per_job=args.blocksize_per_job,
        seed=args.seed,
        rundir=args.rundir,
        mchirp_guess=args.mchirp_guess,
        max_incoherent_lnlike_drop=args.max_incoherent_lnlike_drop,
        max_bestfit_lnlike_diff=args.max_bestfit_lnlike_diff,
        draw_subset=args.draw_subset,
        worker_queue=args.worker_queue,
        max_concurrent=args.max_concurrent,
    )


if __name__ == "__main__":
    main()
