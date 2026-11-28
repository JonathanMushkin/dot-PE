"""Write the P5 (revision 2) GPU run arguments from the o4b priority-sampling reruns.

Each reference is a CPU rerun in o4b `artifacts/dev/priority_sampling/<phase>/<EVENT>/`, made with
dot-PE 8d55dae (whose `dot_pe/` equals gpu-v2's) and `row_selection="phase_marginalized_priority"`.
Its `rerun_info.json` holds its arguments (as strings) and its wall time.

Per event this writes, into research/acceptance/kwargs/ps/:
1. <EVENT>_fresh.json: the reference's settings with nothing recycled (the timed run);
2. <EVENT>_rec_s{0,1,2}.json: the reference's arguments exactly (its intrinsic selection and,
   where it recycled them, its extrinsic samples), with seed + 0, + 1, + 2.
3. <EVENT>_prod.json (revision 3): the reference's arguments with production's rule (bestfit),
   production's extrinsic samples and production's copy of the event file;
and index.json: event -> reference rundir, production (source) rundir.

Usage: python research/acceptance/make_ps_kwargs.py
"""

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
O = Path("/home/projects/barakz/Collaboration-gw/mushkin/dot-pe/o4b-pe-paper/artifacts/dev/priority_sampling")
OUT_RUNS = REPO / "artifacts/research/acceptance/ps"
OUT_KW = REPO / "research/acceptance/kwargs/ps"

# Events of design.md revision 2 (number: event, o4b phase folder of its reference).
EVENTS = {
    1: ("GW240515_005301", "p4_remaining"),
    2: ("GW240923_204006", "p4_remaining"),
    3: ("GW231028_153006", "p4_remaining"),
    4: ("GW240920_124024", "p4_remaining"),
    5: ("GW241011_233834", "p4_remaining"),
    6: ("GW240916_184352", "p4_remaining"),
    7: ("GW231002_143916", "p5_seed_pool"),
    "R": ("GW240525_201644", "p3_compromised"),
}
DROP = {"n_workers", "event_dir", "rundir"}


def parse(v):
    if not isinstance(v, str):
        return v
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def main():
    OUT_KW.mkdir(parents=True, exist_ok=True)
    index = {}
    for num, (ev, phase) in EVENTS.items():
        ref = O / phase / ev
        info = json.load(open(ref / "rerun_info.json"))
        args = {k: parse(v) for k, v in info["args"].items() if k not in DROP}
        assert args["row_selection"] == "phase_marginalized_priority", ev
        if isinstance(args["bank_folder"], str):
            args["bank_folder"] = [args["bank_folder"]]
        index[ev] = {"number": num, "reference_rundir": str(ref), "production_rundir": info["source_run"],
                     "reference_wall_seconds": info["wall_seconds"], "dot_pe_commit": info["dot_pe_commit"],
                     "recycled_extrinsic": args["extrinsic_samples"] is not None}
        fresh = dict(args, load_inds=False, inds_path=None, extrinsic_samples=None,
                     event_dir=str(OUT_RUNS), rundir=str(OUT_RUNS / f"{ev}_fresh"))
        json.dump(fresh, open(OUT_KW / f"{ev}_fresh.json", "w"), indent=1)
        for k in range(3):
            rec = dict(args, seed=args["seed"] + k, event_dir=str(OUT_RUNS),
                       rundir=str(OUT_RUNS / f"{ev}_rec_s{k}"))
            json.dump(rec, open(OUT_KW / f"{ev}_rec_s{k}.json", "w"), indent=1)
        # revision 3: production-recycled run (production's rule, selection, extrinsic samples, event file)
        src = Path(info["source_run"])
        n_ext_prod = json.load(open(src / "run_kwargs.json"))["n_ext"]  # event 7's rerun drew 2048, production 3072
        prod = dict(args, row_selection="bestfit", n_ext=n_ext_prod, seed_pool_lnlike_drop=None,
                    extrinsic_samples=str(src / "extrinsic_samples.feather"), event=str(src / f"{ev}.npz"),
                    event_dir=str(OUT_RUNS), rundir=str(OUT_RUNS / f"{ev}_prod"))
        assert Path(prod["extrinsic_samples"]).exists() and Path(prod["event"]).exists(), ev
        json.dump(prod, open(OUT_KW / f"{ev}_prod.json", "w"), indent=1)
    json.dump(index, open(OUT_KW / "index.json", "w"), indent=1)
    for ev, v in index.items():
        print(v["number"], ev, round(v["reference_wall_seconds"]), "recycled_extrinsic", v["recycled_extrinsic"])


if __name__ == "__main__":
    main()
