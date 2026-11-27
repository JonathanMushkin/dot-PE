"""GPU stages of a dot-PE run: incoherent selection (stage 2) with waveform residency.

``Residency`` keeps raw waveforms h = amp exp(i phase) (no time shift; complex64) on the device
for templates whose incoherent log-likelihood is within ``margin`` of the running maximum over
all banks seen so far. Templates that the final cross-bank threshold (global maximum −
``max_incoherent_lnlike_drop``) keeps are always resident, if
``margin >= max_incoherent_lnlike_drop``. The coherent stage then reads no waveform files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .bank import bank_layout, iter_blocks
from .incoherent import IncoherentGPU


class Residency:
    def __init__(self, margin):
        import cupy as cp
        self.cp = cp
        self.margin = float(margin)
        self.running_max = -np.inf
        self.banks = {}  # bank_id -> {"inds": list of np arrays, "lnl": list, "h": list of device arrays}

    def offer(self, bank_id, inds, lnl, h):
        """Keep the templates of a batch within margin of the running maximum."""
        cp = self.cp
        fin = np.isfinite(lnl)
        if fin.any():
            self.running_max = max(self.running_max, float(lnl[fin].max()))
        keep = np.flatnonzero(fin & (lnl >= self.running_max - self.margin))
        b = self.banks.setdefault(bank_id, {"inds": [], "lnl": [], "h": []})
        if len(keep):
            b["inds"].append(inds[keep])
            b["lnl"].append(lnl[keep])
            b["h"].append(h[cp.asarray(keep)])
        self._prune()

    def _prune(self):
        cp = self.cp
        cut = self.running_max - self.margin
        for b in self.banks.values():
            for j in range(len(b["lnl"])):
                k = b["lnl"][j] >= cut
                if not k.all():
                    b["inds"][j], b["lnl"][j] = b["inds"][j][k], b["lnl"][j][k]
                    b["h"][j] = b["h"][j][cp.asarray(np.flatnonzero(k))]

    def finalize(self, bank_id, selected_inds):
        """Device array (n_sel, m, p, b) of raw h for `selected_inds` (sorted), in that order."""
        cp = self.cp
        b = self.banks[bank_id]
        inds = np.concatenate(b["inds"]) if b["inds"] else np.zeros(0, int)
        h = cp.concatenate(b["h"]) if b["h"] else None
        order = np.argsort(inds)
        inds = inds[order]
        pos = np.searchsorted(inds, selected_inds)
        if len(selected_inds) and (pos.max() >= len(inds) or not np.array_equal(inds[pos], selected_inds)):
            raise RuntimeError(f"{bank_id}: selected templates missing from the device residency")
        out = h[cp.asarray(order[pos])] if len(selected_inds) else None
        self.banks[bank_id] = {"inds": [], "lnl": [], "h": []}  # free the rest
        return out

    def nbytes(self):
        return sum(x.nbytes for b in self.banks.values() for x in b["h"])


def incoherent_bank_gpu(*, event_data, par_dic_0, bank_folder, fbin, approximant, m_arr, n_phi, n_t,
                        n_int, preselected_indices, size_limit, residency, bank_id, batch=2048,
                        threads=8, timings=None):
    """Stage 2 for one bank: (inds, lnlike_di, incoherent_lnlikes), as ``_collect_incoherent_mp``."""
    import time
    from dot_pe.inference import _create_single_detector_processor
    t0 = time.perf_counter()
    sdps = [_create_single_detector_processor(event_data, det, par_dic_0, bank_folder, fbin, approximant,
                                              n_phi, m_arr, 512, size_limit)
            for det in event_data.detector_names]
    g = IncoherentGPU(sdps, par_dic_0, n_t, n_phi, m_arr)
    t1 = time.perf_counter()
    wdir, blocksize, n_files = bank_layout(bank_folder)
    if preselected_indices is not None:
        want = np.unique(np.asarray(preselected_indices, int))
    else:
        want = np.arange(int(n_int))
    files = np.unique(want // blocksize)
    lnl_di = np.full((len(sdps), len(want)), -np.inf)
    pos_of = {}
    t_wait = 0.0
    tw = time.perf_counter()
    for k, inds, amp, phase in iter_blocks(bank_folder, files, threads=threads):
        t_wait += time.perf_counter() - tw
        m = np.isin(inds, want, assume_unique=True)
        if not m.all():
            inds, amp, phase = inds[m], amp[m], phase[m]
        pos = np.searchsorted(want, inds)
        for s in range(0, len(inds), batch):
            out, _, h_raw = g.run_batch(amp[s:s + batch], phase[s:s + batch],
                                        return_raw=residency is not None)
            lnl_di[:, pos[s:s + batch]] = out
            if residency is not None:
                residency.offer(bank_id, inds[s:s + batch], out.sum(axis=0), h_raw)
        tw = time.perf_counter()
    if timings is not None:
        timings[f"2_incoherent_{bank_id}"] = {"cpu_setup": t1 - t0, "gpu_and_read": time.perf_counter() - t1,
                                              "waiting_for_reads": t_wait}
    lnl_di = np.where(np.isfinite(lnl_di), lnl_di, -np.inf)
    return want, lnl_di, lnl_di.sum(axis=0)
