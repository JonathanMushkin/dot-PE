"""P4 gate 3: o4b's LVK reweighting reads a GPU rundir (o4a-pe env; o4b code imported read-only).

Writes samples_lvc.feather and summary_results_lvc.json into the given rundir (ours, under
artifacts/), not into the o4b repo. Usage: python research/runner/o4b_compat.py RUNDIR
"""
import json
import sys

from o4a_pe_paper.pe_lib import lvk_reweight

out = lvk_reweight.reweight_rundir(sys.argv[1])
print(json.dumps(out, default=float))
