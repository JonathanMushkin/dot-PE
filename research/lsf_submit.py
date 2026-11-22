#!/usr/bin/env python
"""Submit one command to LSF with the research jobs' standard header; print the job id.

Usage: python research/lsf_submit.py --name NAME --env dot-pe|o4a-pe [--cores 1] [--mem-mb 16000]
       [--queue physics-short] [--walltime-min 600] -- <command ...>
Logs: artifacts/research/logs/<NAME>/%J.{out,err}. The bsub file is kept next to the logs.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--env", required=True, choices=["dot-pe", "o4a-pe"])
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("--mem-mb", type=int, default=16000, help="memory of the whole job")
    ap.add_argument("--queue", default="physics-short")
    ap.add_argument("--walltime-min", type=int, default=600)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    logs = REPO / "artifacts/research/logs" / a.name
    logs.mkdir(parents=True, exist_ok=True)
    py = f"/home/projects/barakz/jonatahm/miniconda3/envs/{a.env}/bin/python"
    pythonpath = f"export PYTHONPATH={REPO}:$PYTHONPATH" if a.env == "dot-pe" else ""
    script = f"""#!/bin/bash
#BSUB -J {a.name}
#BSUB -o {logs}/%J.out
#BSUB -e {logs}/%J.err
#BSUB -n {a.cores}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem={a.mem_mb // a.cores}]"
#BSUB -q {a.queue}
#BSUB -W {a.walltime_min}
set -e
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
{pythonpath}
cd /tmp
echo "[$(date '+%F %T')] starting on $(hostname); dot-PE $(git -C {REPO} rev-parse --short HEAD)"
{py} {' '.join(shlex.quote(c) for c in cmd)}
echo "[$(date '+%F %T')] finished"
"""
    f = logs / f"{a.name}.bsub"
    f.write_text(script)
    out = subprocess.run(["bsub"], stdin=f.open(), capture_output=True, text=True)
    m = re.search(r"Job <(\d+)>", out.stdout)
    print(m.group(1) if m else out.stdout + out.stderr)


if __name__ == "__main__":
    main()
