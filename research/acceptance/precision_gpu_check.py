"""GPU check of the float64 incoherent kernel (2026-11-27_incoherent_precision.md).

On GW240923_204006 bank 2 (very_low_q_bank_57), with the setup of the fresh GPU run's rundir:
1. the templates 692874, 692873, 692875 and N random templates: GPU lnlike_di against the CPU
   code (`inference.run_for_single_detector`); gate: same finite set, max |delta| <= 1e-3 on
   finite entries, per detector;
2. throughput: run_batch time per 2048 templates (median of 5), projected to 2^20 templates.
Also reports the float64 numpy mirror against the CPU (expected to agree to 1e-9 if the inputs
are the same).

Usage (GPU job): python research/acceptance/precision_gpu_check.py GPU_RUNDIR [N]
Writes artifacts/research/acceptance/precision_gpu_check.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def main(rundir, n_rand=2000):
    import cupy as cp
    from dot_pe import inference
    from dot_pe.inference import get_event_data, _create_single_detector_processor
    from dot_pe.sample_processing import IntrinsicSampleProcessor
    from dot_pe_gpu.incoherent import IncoherentGPU, single_detector_timeshifts, single_det_lnlike_numpy
    rd = Path(rundir)
    kw = json.load(open(rd / "run_kwargs.json"))
    par_dic_0 = json.load(open(rd / "Posterior.json"))["init_kwargs"]["likelihood"]["init_kwargs"]["par_dic_0"]
    bank = Path(kw["bank_folder"][2])
    cfg = json.load(open(bank / "bank_config.json"))
    fbin, m_arr = np.array(cfg["fbin"]), np.asarray(cfg["m_arr"])
    ev = get_event_data(kw["event"])
    rng = np.random.default_rng(0)
    # random templates from the two waveform files (4096 templates each) around the problem ones
    n_bank = 2 ** 20
    inds = np.unique(np.r_[[692873, 692874, 692875], 692224 + rng.choice(8192, n_rand, replace=False)])
    sdps = [_create_single_detector_processor(ev, det, par_dic_0, bank, fbin, cfg["approximant"], kw["n_phi"],
                                              m_arr, 512, 10 ** 7) for det in ev.detector_names]
    t0 = time.time()
    cpu = np.zeros((len(sdps), len(inds)))
    hc = None
    for d, det in enumerate(ev.detector_names):
        r = inference.run_for_single_detector(ev, det, par_dic_0, bank, inds, fbin, hc, cfg["approximant"],
                                              kw["n_phi"], 512, m_arr, kw["n_t"], 10 ** 7, sdp=sdps[d])
        if hc is None:
            cpu[d], hc = r
        else:
            cpu[d] = r
    t_cpu = time.time() - t0
    amp, phase = IntrinsicSampleProcessor._load_amp_and_phase(bank / "waveforms", inds)
    g = IncoherentGPU(sdps, par_dic_0, kw["n_t"], kw["n_phi"], m_arr)
    gpu, dt, _ = g.run_batch(amp, phase)
    dt = dt.get()
    # float64 mirror with the same inputs as the device
    h = amp * np.exp(1j * phase) * np.exp(2j * np.pi * dt[:, None, None, None] * fbin[None, None, None, :])
    mir = np.array([single_det_lnlike_numpy(np.asarray(s.dh_weights_dmpb[0]) * float(s.likelihood.asd_drift[0]) ** -2,
                                            np.asarray(s.hh_weights_dmppb[0]) * float(s.likelihood.asd_drift[0]) ** -2,
                                            h, np.conj(single_detector_timeshifts(s, par_dic_0, kw["n_t"])[0]),
                                            kw["n_phi"], m_arr) for s in sdps])
    out = {"n_templates": int(len(inds)), "cpu_seconds": t_cpu, "per_detector": {}}
    ok = True
    for d, det in enumerate(ev.detector_names):
        fin = np.isfinite(cpu[d]) & np.isfinite(gpu[d])
        dd = np.abs(gpu[d] - cpu[d])[fin]
        dm = np.abs(mir[d] - cpu[d])[np.isfinite(cpu[d]) & np.isfinite(mir[d])]
        k = int(np.argmax(np.where(fin, np.abs(gpu[d] - cpu[d]), -1)))
        rec = {"same_finite": bool((np.isfinite(cpu[d]) == np.isfinite(gpu[d])).all()),
               "max_abs_d_gpu_cpu": float(dd.max()), "p99_abs_d_gpu_cpu": float(np.percentile(dd, 99)),
               "argmax_template": int(inds[k]), "argmax_cpu": float(cpu[d, k]), "argmax_gpu": float(gpu[d, k]),
               "max_abs_d_mirror_cpu": float(dm.max()), "max_abs_d_gpu_mirror": float(np.nanmax(np.abs(gpu[d] - mir[d])[fin]))}
        rec["pass"] = rec["same_finite"] and rec["max_abs_d_gpu_cpu"] <= 1e-3
        ok &= rec["pass"]
        out["per_detector"][det] = rec
    sel = np.isin(inds, [692873, 692874, 692875])
    out["problem_templates"] = {"inds": inds[sel].tolist(), "cpu": cpu[:, sel].tolist(), "gpu": gpu[:, sel].tolist()}
    # throughput
    a2, p2 = amp[:2048], phase[:2048]
    if len(a2) < 2048:
        reps = -(-2048 // len(a2))
        a2, p2 = np.concatenate([amp] * reps)[:2048], np.concatenate([phase] * reps)[:2048]
    ts = []
    for _ in range(6):
        cp.cuda.Device().synchronize(); t = time.time()
        g.run_batch(a2, p2)
        cp.cuda.Device().synchronize(); ts.append(time.time() - t)
    out["batch_2048_seconds"] = float(np.median(ts[1:]))
    out["projected_bank_seconds"] = out["batch_2048_seconds"] * n_bank / 2048
    out["pass"] = bool(ok)
    print(json.dumps(out, indent=1))
    (REPO / "artifacts/research/acceptance").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(REPO / "artifacts/research/acceptance/precision_gpu_check.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
