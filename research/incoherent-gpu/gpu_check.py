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
GATE_SECONDS = 746.0


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


def main(dry_run=False, n_files=None):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    b = build()
    t_build = time.perf_counter() - t0
    print(f"CPU setup (3 SingleDetectorProcessors): {t_build:.1f} s", flush=True)
    z = np.load(RUNDIR / "banks/bank_0/intrinsic_samples.npz")
    sel, prod_di = z["inds"], z["lnlikes_di"]
    test = sel[:512]
    t0 = time.perf_counter()
    ref_lnl, ref_dt = cpu_reference(b, test)
    t_cpu = time.perf_counter() - t0
    print(f"CPU reference on 512 templates: {t_cpu:.1f} s", flush=True)
    if dry_run:
        print("prod vs current CPU (512): max |d sum|",
              float(np.nanmax(np.abs(ref_lnl.sum(0) - prod_di[:, :512].sum(0)))))
        return
    import cupy as cp
    from dot_pe_gpu.bank import iter_blocks, bank_layout
    from dot_pe_gpu.incoherent import IncoherentGPU
    from dot_pe.sample_processing import IntrinsicSampleProcessor
    print("device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), flush=True)
    g = IncoherentGPU(b["sdps"], b["par_dic_0"], b["n_t"], b["n_phi"], b["m_arr"])
    res = {"cpu_setup_seconds": t_build, "cpu_ref_seconds_512": t_cpu}

    # Gates 1-2 on the 512 test templates
    amp, phase = IntrinsicSampleProcessor._load_amp_and_phase(b["bank"] / "waveforms", test)
    g.run_batch(amp, phase)  # warm-up
    lnl, dt, _ = g.run_batch(amp, phase)
    dt = dt.get()
    fin = np.isfinite(ref_lnl) & np.isfinite(lnl)
    res["gate1_dt"] = {"max_abs_dt_s": float(np.abs(dt - ref_dt).max())}
    res["gate1_dt"]["pass"] = res["gate1_dt"]["max_abs_dt_s"] <= 1e-7
    res["gate2_cpu"] = {"max_abs_d_lnl": float(np.abs(lnl - ref_lnl)[fin].max()),
                        "same_inf_set": bool((np.isfinite(ref_lnl) == np.isfinite(lnl)).all()),
                        "n_inf_cpu": int((~np.isfinite(ref_lnl)).sum()), "n_inf_gpu": int((~np.isfinite(lnl)).sum())}
    res["gate2_cpu"]["pass"] = res["gate2_cpu"]["max_abs_d_lnl"] <= 1e-3 and res["gate2_cpu"]["same_inf_set"]
    print("gate1", res["gate1_dt"], "gate2", res["gate2_cpu"], flush=True)

    # Gate 4 (and the data for gate 3): the whole bank, streamed
    wdir, blocksize, nf = bank_layout(b["bank"])
    files = range(nf if n_files is None else n_files)
    all_lnl = np.full((3, nf * blocksize), np.nan)
    t_read = 0.0
    nbytes = 0
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    t_last = t0
    for k, inds, amp, phase in iter_blocks(b["bank"], files):
        t_got = time.perf_counter()
        t_read += t_got - t_last  # time waiting for the reader
        nbytes += amp.nbytes + phase.nbytes
        for s in range(0, len(inds), 2048):
            out, _, _ = g.run_batch(amp[s:s + 2048], phase[s:s + 2048])
            all_lnl[:, inds[s:s + 2048]] = out
        t_last = time.perf_counter()
    cp.cuda.Device().synchronize()
    t_bank = time.perf_counter() - t0
    n_done = len(files)
    scale = nf / n_done
    res["gate4_speed"] = {"files": n_done, "seconds": t_bank, "seconds_waiting_for_reads": t_read,
                          "GB_read": nbytes / 1e9, "read_GBps": nbytes / 1e9 / t_bank,
                          "projected_3_banks_seconds": 3 * t_bank * scale, "gate_seconds": GATE_SECONDS}
    res["gate4_speed"]["pass"] = res["gate4_speed"]["projected_3_banks_seconds"] <= GATE_SECONDS
    print("gate4", res["gate4_speed"], flush=True)

    # Gate 3: against production on the selected templates covered
    cov = sel[sel < n_done * blocksize]
    pos = np.searchsorted(sel, cov)
    s_gpu = all_lnl[:, cov].sum(0)
    s_prod = prod_di[:, pos].sum(0)
    d = np.abs(s_gpu - s_prod)
    fin = np.isfinite(d)
    thr = np.nanmax(np.r_[s_prod]) - 20  # local to the covered templates; global check below
    gmax = np.nanmax(np.where(np.isfinite(all_lnl.sum(0)), all_lnl.sum(0), -np.inf))
    res["gate3_prod"] = {"n_covered": int(len(cov)), "median_abs_d_sum": float(np.median(d[fin])),
                         "p99_abs_d_sum": float(np.percentile(d[fin], 99)), "max_abs_d_sum": float(d[fin].max()),
                         "n_nonfinite_gpu": int((~np.isfinite(s_gpu)).sum()),
                         "gpu_max_incoherent_lnlike_covered": float(gmax)}
    if n_done == nf:
        # production threshold: global max over all banks - 20 (log: 44.66 - 20)
        thr = 44.66 - 20
        tot = all_lnl.sum(0)
        gpu_sel = np.flatnonzero(np.where(np.isfinite(tot), tot, -np.inf) >= thr)
        miss = np.setdiff1d(sel, gpu_sel)
        extra = np.setdiff1d(gpu_sel, sel)
        near = lambda x: np.abs(tot[x] - thr) <= 1e-3 if len(x) else np.array([], bool)
        res["gate3_prod"].update({"n_selected_prod": int(len(sel)), "n_selected_gpu": int(len(gpu_sel)),
                                  "n_missing": int(len(miss)), "n_extra": int(len(extra)),
                                  "n_missing_near_thr": int(near(miss).sum()), "n_extra_near_thr": int(near(extra).sum()),
                                  "missing_examples": miss[:20].tolist(), "extra_examples": extra[:20].tolist()})
    res["gate3_prod"]["pass"] = res["gate3_prod"]["median_abs_d_sum"] <= 1e-3
    print("gate3", res["gate3_prod"], flush=True)
    json.dump(res, open(OUT / "gpu_check.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    a = sys.argv
    main(dry_run="--dry-run" in a, n_files=int(a[a.index("--n-files") + 1]) if "--n-files" in a else None)
