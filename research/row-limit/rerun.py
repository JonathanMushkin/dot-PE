#!/usr/bin/env python
"""E5: rerun a production dot-PE run with row_thinning="pairs", then the LVK-prior reweighting.

Design: research/row-limit/design_solution.md. Everything as in the production run's
`run_kwargs.json` except: the dot-PE version (branch `research`), the rundir, the incoherent stage
and extrinsic samples recycled from the production run (`load_inds`, `inds_path`,
`extrinsic_samples`), and `row_thinning`. The LVK reweighting is o4b-pe-paper's
`lvk_reweight.reweight_rundir` with `ln_posterior` replaced by `ln_posterior + ln_thinning_weight`
(that function recomputes weights from `ln_posterior` and would otherwise drop the correction).

Env: dot-pe (this repo) with o4b-pe-paper's `src` on the path for the reweighting.
Usage: python rerun.py --event GW240525_201644 --run run_2 [--row-thinning pairs] [--tag pairs]
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

O4B = Path("/home/projects/barakz/Collaboration-gw/mushkin/dot-pe/o4b-pe-paper")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts/research/row-limit/runs"


def run_dotpe(event: str, run: str, row_thinning: str, tag: str, n_workers: int) -> Path:
    import inspect

    from dot_pe import mp_inference

    src = O4B / "artifacts/pe/dotpe" / event / run
    kw = json.loads((src / "run_kwargs.json").read_text())
    banks = {f"bank_{k}": b for k, b in enumerate(kw["bank_folder"])}
    event_dir = OUT / event
    rundir = event_dir / tag
    allowed = set(inspect.signature(mp_inference.run).parameters)
    args = {k: v for k, v in kw.items() if k in allowed}
    args.update(
        n_int=None if kw["n_int"] in (None, "None") else kw["n_int"],
        event_dir=str(event_dir),
        rundir=str(rundir),
        load_inds=True,
        inds_path={b: str(src / f"banks/{b}/intrinsic_samples.npz") for b in banks},
        extrinsic_samples=str(src / "extrinsic_samples.feather"),
        row_thinning=row_thinning,
        n_workers=n_workers,
    )
    args.pop("i_int_start", None)
    print(json.dumps({k: str(v) for k, v in args.items()}, indent=1), flush=True)
    t0 = time.time()
    mp_inference.run(**args)
    wall = time.time() - t0
    (rundir / "rerun_timing.json").write_text(json.dumps({
        "event": event, "source_run": str(src), "row_thinning": row_thinning,
        "wall_seconds": wall, "n_workers": n_workers,
        "maxrss_children_gb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6,
        "maxrss_self_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
    }, indent=2))
    return rundir


def reweight_lvk(rundir: Path) -> dict:
    """o4b-pe-paper's reweight_rundir, with the thinning correction folded into ln_posterior."""
    import pandas as pd

    sys.path.insert(0, str(O4B / "src"))
    from o4a_pe_paper.pe_lib import lvk_reweight as L

    orig = pd.read_feather
    path = (rundir / "prob_samples.feather").resolve()
    n_corrected = []

    def read_with_correction(p, *a, **k):
        df = orig(p, *a, **k)
        if Path(p).resolve() == path and "ln_thinning_weight" in df.columns:
            df["ln_posterior"] = df["ln_posterior"] + df["ln_thinning_weight"]
            n_corrected.append(1)
        return df

    L.pd.read_feather = read_with_correction
    try:
        out = L.reweight_rundir(rundir)
    finally:
        L.pd.read_feather = orig
    assert n_corrected, "prob_samples.feather was not read through the correction"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--event", required=True)
    ap.add_argument("--run", required=True, help="production run name, e.g. run_2")
    ap.add_argument("--row-thinning", default="pairs")
    ap.add_argument("--tag", default="pairs")
    ap.add_argument("--n-workers", type=int, default=7)
    ap.add_argument("--lvk-only", action="store_true")
    a = ap.parse_args()
    rundir = OUT / a.event / a.tag
    if not a.lvk_only:
        rundir = run_dotpe(a.event, a.run, a.row_thinning, a.tag, a.n_workers)
    print(json.dumps(reweight_lvk(rundir), indent=1, default=str))


if __name__ == "__main__":
    main()
