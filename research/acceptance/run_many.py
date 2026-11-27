"""Run dot_pe_gpu.run on several kwargs JSON files in sequence (one GPU job).
Usage: python research/acceptance/run_many.py K1.json [K2.json ...]"""
import json
import sys
import traceback

from dot_pe_gpu.run import run

for f in sys.argv[1:]:
    print(f"\n===== {f}", flush=True)
    try:
        run(**json.load(open(f)))
    except Exception:
        traceback.print_exc()
