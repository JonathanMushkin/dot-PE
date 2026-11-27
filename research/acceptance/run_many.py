"""Run dot_pe_gpu.run on several kwargs JSON files in sequence (one GPU job).
Usage: python research/acceptance/run_many.py [--stop-after-setup] K1.json [K2.json ...]
--stop-after-setup: stage 1 only, the CPU dry run (rundirs get a _dryrun suffix)."""
import json
import sys
import traceback

from dot_pe_gpu.run import run

dry = "--stop-after-setup" in sys.argv
for f in [x for x in sys.argv[1:] if x != "--stop-after-setup"]:
    print(f"\n===== {f}", flush=True)
    try:
        kw = json.load(open(f))
        if dry:
            kw.update(stop_after_setup=True, rundir=kw["rundir"] + "_dryrun")
        run(**kw)
    except Exception:
        traceback.print_exc()
