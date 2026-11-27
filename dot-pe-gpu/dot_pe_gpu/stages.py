"""GPU stages of a dot-PE run: incoherent selection (stage 2) with waveform residency.

``Residency`` keeps raw waveforms h = amp exp(i phase) (no time shift) on the device, stored as
complex64 of h * H_SCALE (amplitudes near 1e-20 would otherwise reach float32's subnormal range
in products); ``finalize`` returns complex128 h. It keeps two sets:

1. templates whose incoherent log-likelihood is within ``margin`` of the running maximum over all
   banks seen so far. With ``margin >= max_incoherent_lnlike_drop`` this holds every template the
   relative cross-bank threshold keeps, because the running maximum never exceeds the final one;
2. the ``top_n`` templates of largest incoherent log-likelihood over all banks, whatever the
   margin: stage 3 falls back to these when fewer than ``min_incoherent_survivors`` pass.

The coherent stage then reads no waveform files. Non-finite per-detector log-likelihoods are
stored as -inf (the CPU would carry NaN into the sum).
"""

from __future__ import annotations

import numpy as np

from .bank import bank_layout, iter_blocks
from .incoherent import IncoherentGPU

H_SCALE = 2.0 ** 64  # exact power of two: scaling is lossless


class Residency:
    def __init__(self, margin, top_n=100, xp=None):
        if xp is None:
            import cupy as xp
        self.xp = xp
        self.margin = float(margin)
        self.top_n = int(top_n)
        self.running_max = -np.inf
        self.banks = {}  # bank_id -> {"inds": [np arrays], "lnl": [np arrays], "h": [device arrays]}
        self.top = {"bank": np.zeros(0, object), "inds": np.zeros(0, np.int64),
                    "lnl": np.zeros(0), "h": None}

    def offer(self, bank_id, inds, lnl, h_raw):
        """Offer a batch: inds (n,), lnl (n,) host arrays, h_raw (n, m, p, b) device (unscaled)."""
        xp = self.xp
        inds = np.asarray(inds, np.int64)
        lnl = np.where(np.isfinite(lnl), lnl, -np.inf)
        if np.isfinite(lnl).any():
            self.running_max = max(self.running_max, float(lnl.max()))
        hs = (h_raw * H_SCALE).astype(xp.complex64)
        keep = np.flatnonzero(lnl >= self.running_max - self.margin)
        b = self.banks.setdefault(bank_id, {"inds": [], "lnl": [], "h": []})
        if len(keep):
            b["inds"].append(inds[keep])
            b["lnl"].append(lnl[keep])
            b["h"].append(hs[xp.asarray(keep)])
        self._update_top(bank_id, inds, lnl, hs)
        self._prune()

    def _update_top(self, bank_id, inds, lnl, hs):
        xp = self.xp
        if self.top_n <= 0:
            return
        n_new = min(self.top_n, len(lnl))
        cand = np.argpartition(-lnl, n_new - 1)[:n_new] if n_new < len(lnl) else np.arange(len(lnl))
        t = self.top
        bank = np.concatenate([t["bank"], np.full(len(cand), bank_id, object)])
        tinds = np.concatenate([t["inds"], inds[cand]])
        tlnl = np.concatenate([t["lnl"], lnl[cand]])
        th = hs[xp.asarray(cand)] if t["h"] is None else xp.concatenate([t["h"], hs[xp.asarray(cand)]])
        if len(tlnl) > self.top_n:
            k = np.argpartition(-tlnl, self.top_n - 1)[:self.top_n]
            bank, tinds, tlnl, th = bank[k], tinds[k], tlnl[k], th[xp.asarray(k)]
        self.top = {"bank": bank, "inds": tinds, "lnl": tlnl, "h": th}

    def _prune(self):
        xp = self.xp
        cut = self.running_max - self.margin
        for b in self.banks.values():
            for j in range(len(b["lnl"])):
                k = b["lnl"][j] >= cut
                if not k.all():
                    b["inds"][j], b["lnl"][j] = b["inds"][j][k], b["lnl"][j][k]
                    b["h"][j] = b["h"][j][xp.asarray(np.flatnonzero(k))]

    def finalize(self, bank_id, selected_inds):
        """complex128 device array (n_sel, m, p, b) of raw h for `selected_inds`, in that order."""
        xp = self.xp
        selected_inds = np.asarray(selected_inds, np.int64)
        b = self.banks.get(bank_id, {"inds": [], "lnl": [], "h": []})
        in_top = self.top["bank"] == bank_id
        parts_i = b["inds"] + [self.top["inds"][in_top]]
        parts_h = b["h"] + ([self.top["h"][xp.asarray(np.flatnonzero(in_top))]] if in_top.any() else [])
        inds = np.concatenate(parts_i) if parts_i else np.zeros(0, np.int64)
        inds, first = np.unique(inds, return_index=True)  # top set may repeat margin-set entries
        pos = np.searchsorted(inds, selected_inds)
        if len(selected_inds) and (pos.max() >= len(inds) or not np.array_equal(inds[pos], selected_inds)):
            missing = np.setdiff1d(selected_inds, inds)
            raise RuntimeError(f"{bank_id}: {len(missing)} selected templates are not resident "
                               f"(e.g. {missing[:5].tolist()})")
        if not len(selected_inds):
            out = None
        else:
            h = xp.concatenate(parts_h)
            out = h[xp.asarray(first[pos])].astype(xp.complex128) / H_SCALE
        self.banks[bank_id] = {"inds": [], "lnl": [], "h": []}  # free the rest of this bank
        return out

    def nbytes(self):
        n = sum(x.nbytes for b in self.banks.values() for x in b["h"])
        return n + (0 if self.top["h"] is None else self.top["h"].nbytes)


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
        # sorted, without repeats (the CPU keeps the given order; per-bank results are re-sorted)
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
