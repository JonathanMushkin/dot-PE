"""Run several research scripts in one job, in order: python research/run_scripts.py 'a.py args' 'b.py args'.
Each runs as __main__ with its own argv; a failure is printed and the next one still runs."""
import runpy
import shlex
import sys
import traceback

for spec in sys.argv[1:]:
    argv = shlex.split(spec)
    print(f"\n===== {spec}", flush=True)
    sys.argv = argv
    try:
        runpy.run_path(argv[0], run_name="__main__")
    except SystemExit:
        pass
    except Exception:
        traceback.print_exc()
