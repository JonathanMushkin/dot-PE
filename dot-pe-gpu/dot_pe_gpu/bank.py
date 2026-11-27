"""Streaming reads of a dot-PE bank's waveform files, overlapped with GPU work.

A bank folder holds ``waveforms/amplitudes_block_{k}.npy`` and ``phase_block_{k}.npy``, each
(blocksize, n_modes, n_pol, n_fbin) float64 for the templates k*blocksize ... (k+1)*blocksize-1.
``iter_blocks`` yields (k, template indices, amp, phase) in file order. A thread pool reads up
to ``prefetch`` files ahead; numpy file reads release the GIL, so reads overlap with the
caller's GPU work.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def bank_layout(bank_folder):
    bank_folder = Path(bank_folder)
    cfg = json.load(open(bank_folder / "bank_config.json"))
    wdir = bank_folder / "waveforms"
    n_files = len(list(wdir.glob("amplitudes_block_*.npy")))
    return wdir, int(cfg["blocksize"]), n_files


def _read(wdir, k):
    amp = np.load(wdir / f"amplitudes_block_{k}.npy")
    phase = np.load(wdir / f"phase_block_{k}.npy")
    return amp, phase


def iter_blocks(bank_folder, files=None, threads=8, prefetch=8):
    """Yield (k, inds, amp, phase) for the given file numbers (default: all), in order."""
    wdir, blocksize, n_files = bank_layout(bank_folder)
    files = list(range(n_files)) if files is None else list(files)
    with ThreadPoolExecutor(threads) as pool:
        futs = {}
        nxt = 0
        for j, k in enumerate(files):
            while nxt < len(files) and nxt < j + prefetch:
                futs[nxt] = pool.submit(_read, wdir, files[nxt])
                nxt += 1
            amp, phase = futs.pop(j).result()
            inds = np.arange(k * blocksize, k * blocksize + len(amp))
            yield k, inds, amp, phase
