"""Locate the step where the GPU incoherent stage departs from the CPU on given templates.

Replays IncoherentGPU on numpy: the linear-free dt (dot_pe_gpu.incoherent.linfree_dt with xp=numpy)
against the CPU's cached dt, then the single-detector lnlike mirror in float64 and float32, with the
GPU dt and with the CPU dt, against the CPU lnlike.

Usage: python research/acceptance/outlier_diag.py GPU_RUNDIR BANK_INDEX IND [IND ...]
"""

import json
import sys
from pathlib import Path

import numpy as np


def main(rundir, bank_index, inds):
    from dot_pe import inference
    from dot_pe.inference import get_event_data, _create_single_detector_processor
    from dot_pe.sample_processing import IntrinsicSampleProcessor
    from dot_pe_gpu.incoherent import linfree_dt, single_detector_timeshifts, single_det_lnlike_numpy
    rd = Path(rundir)
    kw = json.load(open(rd / "run_kwargs.json"))
    par_dic_0 = json.load(open(rd / "Posterior.json"))["init_kwargs"]["likelihood"]["init_kwargs"]["par_dic_0"]
    bank = Path(kw["bank_folder"][bank_index])
    cfg = json.load(open(bank / "bank_config.json"))
    fbin, m_arr = np.array(cfg["fbin"]), np.asarray(cfg["m_arr"])
    ev = get_event_data(kw["event"])
    inds = np.array(inds)
    sdps = [_create_single_detector_processor(ev, det, par_dic_0, bank, fbin, cfg["approximant"], kw["n_phi"],
                                              m_arr, 512, 10 ** 7) for det in ev.detector_names]
    cpu = np.zeros((len(sdps), len(inds)))
    h_cache = None
    for d, det in enumerate(ev.detector_names):
        r = inference.run_for_single_detector(ev, det, par_dic_0, bank, inds, fbin, h_cache, cfg["approximant"],
                                              kw["n_phi"], 512, m_arr, kw["n_t"], 10 ** 7, sdp=sdps[d])
        if h_cache is None:
            cpu[d], h_cache = r
        else:
            cpu[d] = r
    isp = sdps[0].intrinsic_sample_processor
    dt_cpu = np.array([isp.cached_dt_linfree_relative[int(i)] for i in inds])

    amp, phase = IntrinsicSampleProcessor._load_amp_and_phase(bank / "waveforms", inds)
    h = amp * np.exp(1j * phase)
    lk = sdps[0].likelihood
    m2 = list(m_arr).index(2)
    dt_gpu = linfree_dt(np, h[:, m2, 0], np.asarray(lk._h2plus0_fbin, np.complex128),
                        np.asarray(lk._polyfit_weights, float), np.asarray(lk.fbin, float))
    out = {"inds": inds.tolist(), "dt_cpu": dt_cpu.tolist(), "dt_gpu_formula": dt_gpu.tolist(), "cpu": cpu.tolist()}
    for name, dt in (("gpu_dt", dt_gpu), ("cpu_dt", dt_cpu)):
        hs = h * np.exp(2j * np.pi * dt[:, None, None, None] * fbin[None, None, None, :])
        for dtype in (np.float64, np.float32):
            res = []
            for sdp in sdps:
                drift_inv2 = float(sdp.likelihood.asd_drift[0]) ** -2
                ts = single_detector_timeshifts(sdp, par_dic_0, kw["n_t"])[0]
                res.append(single_det_lnlike_numpy(np.asarray(sdp.dh_weights_dmpb[0]) * drift_inv2,
                                                   np.asarray(sdp.hh_weights_dmppb[0]) * drift_inv2,
                                                   hs, np.conj(ts), kw["n_phi"], m_arr, dtype=dtype).tolist())
            out[f"mirror_{name}_{np.dtype(dtype).name}"] = res
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3:]])
