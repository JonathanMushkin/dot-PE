"""CPU reference for P2: run_thin_iblock on the test blocks in the four configurations.

Writes artifacts/research/coherent-kernel/ref/{tag}_block{k}.feather and .json (scalars), and
the waveform arrays of each block (amp/phase .npy) so the GPU check uses identical inputs.
Usage (LSF, 1 core): python research/coherent-kernel/make_reference.py [--dry-run]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

from dot_pe import thin_coherent  # noqa: E402
from dot_pe.sample_processing import IntrinsicSampleProcessor  # noqa: E402


def main(dry_run=False):
    out = common.OUT / "ref"
    out.mkdir(parents=True, exist_ok=True)
    blocks = common.test_blocks()
    wdir = common.bank_path() / "waveforms"
    print("test blocks:", {k: (len(b), int(b[0]), int(b[-1])) for k, b in blocks.items()}, flush=True)
    for k, b in blocks.items():
        np.save(out / f"block{k}_inds.npy", b)
        if not (out / f"block{k}_amp.npy").exists():
            amp, phase = IntrinsicSampleProcessor._load_amp_and_phase(wdir, b)
            np.save(out / f"block{k}_amp.npy", amp)
            np.save(out / f"block{k}_phase.npy", phase)
    if dry_run:
        s = common.setup_for("bestfit", True)
        print("setup ok:", {k: getattr(v, "shape", v) for k, v in s.items() if k != "_profile_dir"})
        return
    for rs, bound in common.CONFIGS:
        s = common.setup_for(rs, bound)
        eb = common.e_blocks(s)
        for k, b in blocks.items():
            t0 = time.time()
            df, n_disc, lse, lsse, _, n_dm = thin_coherent.run_thin_iblock(b, eb, s, wdir)
            dt = time.time() - t0
            name = f"{common.tag(rs, bound)}_block{k}"
            df.reset_index(drop=True).to_feather(out / f"{name}.feather")
            json.dump({"n_disc": int(n_disc), "logsumexp_disc": float(lse),
                       "logsumsqrexp_disc": float(lsse), "n_dist_marg": int(n_dm),
                       "n_rows": int(len(df)), "seconds": dt}, open(out / f"{name}.json", "w"))
            print(f"{name}: {len(df)} rows, n_dist_marg {n_dm}, {dt:.1f} s", flush=True)


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
