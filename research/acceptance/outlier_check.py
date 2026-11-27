"""CPU single-detector lnlike of given templates, with the current dot-PE code, against the GPU
run's values. Diagnoses a fresh GPU run whose incoherent maximum exceeds production's.

Usage: python research/acceptance/outlier_check.py GPU_RUNDIR BANK_INDEX IND [IND ...]
"""

import json
import sys
from pathlib import Path

import numpy as np


def main(rundir, bank_index, inds):
    from dot_pe import inference
    from dot_pe.inference import get_event_data, _create_single_detector_processor
    rd = Path(rundir)
    kw = json.load(open(rd / "run_kwargs.json"))
    par_dic_0 = json.load(open(rd / "Posterior.json"))["init_kwargs"]["likelihood"]["init_kwargs"]["par_dic_0"]
    bank = Path(kw["bank_folder"][bank_index])
    cfg = json.load(open(bank / "bank_config.json"))
    fbin, m_arr = np.array(cfg["fbin"]), np.asarray(cfg["m_arr"])
    ev = get_event_data(kw["event"])
    inds = np.array(inds)
    lnl = np.zeros((len(ev.detector_names), len(inds)))
    h = None
    for d, det in enumerate(ev.detector_names):
        sdp = _create_single_detector_processor(ev, det, par_dic_0, bank, fbin, cfg["approximant"], kw["n_phi"],
                                                m_arr, 512, 10 ** 7)
        r = inference.run_for_single_detector(ev, det, par_dic_0, bank, inds, fbin, h, cfg["approximant"],
                                              kw["n_phi"], 512, m_arr, kw["n_t"], 10 ** 7, sdp=sdp)
        if h is None:
            lnl[d], h = r
        else:
            lnl[d] = r
    out = {"detectors": list(ev.detector_names), "inds": inds.tolist(), "cpu_lnlike_di": lnl.tolist(),
           "cpu_incoherent": lnl.sum(0).tolist()}
    g = np.load(rd / "banks" / f"bank_{bank_index}" / "intrinsic_samples.npz")
    pos = {int(i): k for k, i in enumerate(g["inds"])}
    out["gpu_lnlike_di"] = [g["lnlikes_di"][:, pos[int(i)]].tolist() if int(i) in pos else None for i in inds]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3:]])
