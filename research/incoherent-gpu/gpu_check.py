"""P3 GPU check: gates 1-4 of research/incoherent-gpu/design.md.

Writes artifacts/research/incoherent-gpu/gpu_check.json.
Usage (GPU job): python research/incoherent-gpu/gpu_check.py [--dry-run] [--n-files N]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

RUNDIR = Path("/home/projects/barakz/Collaboration-gw/mushkin/dot-pe/o4b-pe-paper/artifacts/pe/"
              "dotpe/GW240525_201644/run_2")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts/research/incoherent-gpu"
GATE_SECONDS = 746.0  # 15% of the median run's 0.5x target (4972 s); that run has 6 banks


def build(n_phi=100, n_t=128):
    from dot_pe import inference
    from dot_pe.inference import get_event_data, _create_single_detector_processor
    kw = json.load(open(RUNDIR / "run_kwargs.json"))
    post = json.load(open(RUNDIR / "Posterior.json"))["init_kwargs"]
    par_dic_0 = post["likelihood"]["init_kwargs"]["par_dic_0"]
    bank = Path(kw["bank_folder"][0])
    cfg = json.load(open(bank / "bank_config.json"))
    fbin = np.array(cfg["fbin"])
    m_arr = np.asarray(cfg["m_arr"])
    ev = get_event_data(kw["event"])
    sdps = [_create_single_detector_processor(ev, det, par_dic_0, bank, fbin, cfg["approximant"], n_phi,
                                              m_arr, 512, 10 ** 7) for det in ev.detector_names]
    return dict(approximant=cfg["approximant"], kw=kw, par_dic_0=par_dic_0, bank=bank, fbin=fbin, m_arr=m_arr, ev=ev, sdps=sdps,
                n_phi=n_phi, n_t=n_t, inference=inference)


def cpu_reference(b, inds):
    """Current CPU code: lnlike_di and dt on the templates `inds`."""
    lnl = np.zeros((len(b["sdps"]), len(inds)))
    h = None
    for d, det in enumerate(b["ev"].detector_names):
        r = b["inference"].run_for_single_detector(
            b["ev"], det, b["par_dic_0"], b["bank"], inds, b["fbin"], h, b["approximant"],
            b["n_phi"], 512, b["m_arr"], b["n_t"], 10 ** 7, sdp=b["sdps"][d])
        if h is None:
            lnl[d], h = r
        else:
            lnl[d] = r
    isp = b["sdps"][0].intrinsic_sample_processor
    dt = np.array([isp.cached_dt_linfree_relative[int(i)] for i in inds])
    return lnl, dt


def gate12(g, b, inds, ref_lnl, ref_dt, IntrinsicSampleProcessor):
    amp, phase = IntrinsicSampleProcessor._load_amp_and_phase(b["bank"] / "waveforms", inds)
    lnl, dt, _ = g.run_batch(amp, phase)
    dt = dt.get()
    fin = np.isfinite(ref_lnl) & np.isfinite(lnl)
    g1 = {"max_abs_dt_s": float(np.abs(dt - ref_dt).max())}
    g1["pass"] = g1["max_abs_dt_s"] <= 1e-7
    g2 = {"max_abs_d_lnl": float(np.abs(lnl - ref_lnl)[fin].max()) if fin.any() else 0.0,
          "same_inf_set": bool((np.isfinite(ref_lnl) == np.isfinite(lnl)).all()),
          "n_inf_cpu": int((~np.isfinite(ref_lnl)).sum()), "n_inf_gpu": int((~np.isfinite(lnl)).sum())}
    g2["pass"] = g2["max_abs_d_lnl"] <= 1e-3 and g2["same_inf_set"]
    return g1, g2


def main(dry_run=False, n_files=None):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    b = build()
    t_build = time.perf_counter() - t0
    n_det = len(b["sdps"])
    print(f"CPU setup ({n_det} SingleDetectorProcessors): {t_build:.1f} s", flush=True)
    z = np.load(RUNDIR / "banks/bank_0/intrinsic_samples.npz")
    sel, prod_di = z["inds"], z["lnlikes_di"]
    # production threshold: max over all banks' incoherent sums - max_incoherent_lnlike_drop
    gmax_prod = max(float(np.load(f)["incoherent_lnlikes"].max())
                    for f in sorted((RUNDIR / "banks").glob("bank_*/intrinsic_samples.npz")))
    thr = gmax_prod - float(b["kw"]["max_incoherent_lnlike_drop"])
    from dot_pe_gpu.bank import bank_layout
    _, blocksize, nf = bank_layout(b["bank"])
    rng = np.random.default_rng(0)
    # random templates from 4 random files (each file holds `blocksize` templates; reads stay small)
    rfiles = rng.choice(nf, 4, replace=False)
    rand = np.concatenate([f * blocksize + rng.choice(blocksize, 128, replace=False) for f in rfiles])
    tests = {"selected": sel[:512], "random": np.sort(rand)}
    refs = {}
    t0 = time.perf_counter()
    for name, inds in tests.items():
        refs[name] = cpu_reference(b, inds)
    t_cpu = time.perf_counter() - t0
    print(f"CPU reference on 2 x 512 templates: {t_cpu:.1f} s; threshold {thr:.6f}", flush=True)
    if dry_run:
        import dot_pe_gpu.incoherent, dot_pe_gpu.stages  # noqa: F401  (import check without GPU)
        print("prod vs current CPU (512 selected): max |d sum|",
              float(np.nanmax(np.abs(refs["selected"][0].sum(0) - prod_di[:, :512].sum(0)))))
        print("random templates: n -inf per detector", (~np.isfinite(refs["random"][0])).sum(1).tolist())
        return
    import cupy as cp
    from dot_pe_gpu.bank import iter_blocks
    from dot_pe_gpu.incoherent import IncoherentGPU
    from dot_pe_gpu.stages import Residency
    from dot_pe.sample_processing import IntrinsicSampleProcessor
    print("device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), flush=True)
    g = IncoherentGPU(b["sdps"], b["par_dic_0"], b["n_t"], b["n_phi"], b["m_arr"])
    res = {"cpu_setup_seconds": t_build, "cpu_ref_seconds": t_cpu, "threshold": thr}
    g.run_batch(*IntrinsicSampleProcessor._load_amp_and_phase(b["bank"] / "waveforms", sel[:8]))  # warm-up
    for name, inds in tests.items():
        g1, g2 = gate12(g, b, inds, *refs[name], IntrinsicSampleProcessor)
        res[f"gate1_dt_{name}"], res[f"gate2_cpu_{name}"] = g1, g2
        print("gate1", name, g1, "gate2", name, g2, flush=True)

    # The whole bank, streamed, with the device residency (data for gates 3, 4 and 5)
    files = range(nf if n_files is None else n_files)
    all_lnl = np.full((n_det, nf * blocksize), -np.inf)
    resid = Residency(margin=float(b["kw"]["max_incoherent_lnlike_drop"]), top_n=100)
    t_wait, nbytes = 0.0, 0
    mem0 = cp.get_default_memory_pool().used_bytes()
    peak = 0
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    t_last = t0
    for k, inds, amp, phase in iter_blocks(b["bank"], files):
        t_wait += time.perf_counter() - t_last
        nbytes += amp.nbytes + phase.nbytes
        for s0 in range(0, len(inds), 2048):
            out, _, h_raw = g.run_batch(amp[s0:s0 + 2048], phase[s0:s0 + 2048], return_raw=True)
            all_lnl[:, inds[s0:s0 + 2048]] = out
            resid.offer("bank_0", inds[s0:s0 + 2048], out.sum(axis=0), h_raw)
        peak = max(peak, cp.get_default_memory_pool().used_bytes())
        t_last = time.perf_counter()
    cp.cuda.Device().synchronize()
    t_bank = time.perf_counter() - t0
    n_done = len(files)
    scale = nf / n_done
    res["gate4_speed_indicative"] = {
        "files": n_done, "seconds": t_bank, "seconds_waiting_for_reads": t_wait, "GB_read": nbytes / 1e9,
        "overall_GBps": nbytes / 1e9 / t_bank, "projected_bank_seconds": t_bank * scale,
        "projected_6_banks_seconds": 6 * t_bank * scale, "gate_seconds": GATE_SECONDS,
        "exclusive": "--exclusive" in sys.argv}
    res["gate4_speed_indicative"]["pass"] = res["gate4_speed_indicative"]["projected_6_banks_seconds"] <= GATE_SECONDS
    print("gate4", res["gate4_speed_indicative"], flush=True)

    # Gate 3: against production
    cov = sel[sel < n_done * blocksize]
    pos = np.searchsorted(sel, cov)
    s_gpu, s_prod = all_lnl[:, cov].sum(0), prod_di[:, pos].sum(0)
    d = np.abs(s_gpu - s_prod)
    fin = np.isfinite(d)
    g3 = {"n_covered": int(len(cov)), "median_abs_d_sum": float(np.median(d[fin])) if fin.any() else np.nan,
          "p99_abs_d_sum": float(np.percentile(d[fin], 99)) if fin.any() else np.nan,
          "max_abs_d_sum": float(d[fin].max()) if fin.any() else np.nan,
          "n_nonfinite_gpu": int((~np.isfinite(s_gpu)).sum())}
    g3["pass_3_1"] = bool(g3["median_abs_d_sum"] <= 1e-3)
    if n_done == nf:
        tot = all_lnl.sum(0)
        gpu_sel = np.flatnonzero(tot >= thr)
        mism = np.union1d(np.setdiff1d(sel, gpu_sel), np.setdiff1d(gpu_sel, sel))
        near = np.abs(tot[mism] - thr) <= 1e-3
        # mask attribution: the current CPU code on the mismatches (production predates the mask)
        attributed = np.zeros(len(mism), bool)
        if len(mism) and len(mism) <= 4096:
            cur = cpu_reference(b, mism)[0].sum(0)
            attributed = (np.isfinite(cur) == np.isfinite(tot[mism])) & (
                ~np.isfinite(cur) | (np.abs(np.where(np.isfinite(cur), cur, 0) - np.where(np.isfinite(tot[mism]), tot[mism], 0)) <= 1e-3))
            attributed &= ~near
        g3.update({"n_selected_prod": int(len(sel)), "n_selected_gpu": int(len(gpu_sel)),
                   "n_mismatch": int(len(mism)), "n_near_threshold": int(near.sum()),
                   "n_attributed_to_current_cpu_code": int(attributed.sum()),
                   "unexplained_examples": mism[~(near | attributed)][:20].tolist()})
        g3["pass_3_2"] = bool(len(mism) <= 4096 and (near | attributed).all())

        # Gate 5: residency returns exactly the selected set, equal to the waveforms from disk
        t5 = time.perf_counter()
        h_sel = resid.finalize("bank_0", gpu_sel)
        chk = np.sort(rng.choice(gpu_sel, min(256, len(gpu_sel)), replace=False))
        amp, phase = IntrinsicSampleProcessor._load_amp_and_phase(b["bank"] / "waveforms", chk)
        want = ((amp * np.exp(1j * phase)) * 2.0 ** 64).astype(np.complex64).astype(np.complex128) / 2.0 ** 64
        got = h_sel[cp.asarray(np.searchsorted(gpu_sel, chk))].get()
        res["gate5_residency"] = {"n_selected": int(len(gpu_sel)), "shape": list(h_sel.shape),
                                  "equal_to_disk_complex64": bool(np.array_equal(got, want)),
                                  "max_rel_err": float(np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-300))),
                                  "resident_bytes_finalized": int(h_sel.nbytes // 2),
                                  "peak_pool_bytes": int(peak - mem0), "seconds": time.perf_counter() - t5}
        res["gate5_residency"]["pass"] = res["gate5_residency"]["equal_to_disk_complex64"]
        print("gate5", res["gate5_residency"], flush=True)
    res["gate3_prod"] = g3
    print("gate3", g3, flush=True)
    json.dump(res, open(OUT / "gpu_check.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    a = sys.argv
    main(dry_run="--dry-run" in a, n_files=int(a[a.index("--n-files") + 1]) if "--n-files" in a else None)
