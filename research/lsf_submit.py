#!/usr/bin/env python
"""Submit one command to LSF with the research jobs' standard header; print the job id.

Usage: python research/lsf_submit.py --name NAME --env dot-pe|dot-pe-gpu|o4a-pe [--cores 1]
       [--mem-mb 16000] [--queue physics-short] [--walltime-min 600]
       [--gpu MODEL --gmem GB [--exclusive]] -- <command ...>
GPU jobs (--gpu, e.g. NVIDIAL40S, NVIDIAA100_SXM4_80GB, NVIDIAH100NVL): queue short-gpu, 1 GPU,
`module load CUDA/12.6.0`, nvidia-smi header in the log. Limits in research/protocol.md "GPU usage"
are enforced here: walltime <= 30 min shared / 180 min exclusive, cores <= 4 shared / 8 exclusive,
gmem <= 40 GB shared.
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
    ap.add_argument("--env", required=True, choices=["dot-pe", "dot-pe-gpu", "o4a-pe"])
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("--mem-mb", type=int, default=16000, help="memory of the whole job")
    ap.add_argument("--queue", default="physics-short")
    ap.add_argument("--walltime-min", type=int, default=600)
    ap.add_argument("--gpu", default=None, help="LSF gmodel name; implies queue short-gpu")
    ap.add_argument("--gmem", type=int, default=None, help="GPU memory to reserve [GB] (shared)")
    ap.add_argument("--exclusive", action="store_true", help="exclusive GPU (timing benchmarks)")
    ap.add_argument("--dry-run", action="store_true", help="print the bsub file, do not submit")
    ap.add_argument("--end-to-end", action="store_true",
                    help="end-to-end GPU run (P4/P5): up to 8 cores and 96 GB host memory")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    gpu_lines = ""
    nthr = 1
    gpu_setup = ""
    if a.gpu:
        lim_w, lim_n = (180, 8) if a.exclusive else (30, 8 if a.end_to_end else 4)
        lim_mem = 96000 if a.end_to_end else (64000 if a.exclusive else 32000)
        if a.mem_mb > lim_mem:
            raise SystemExit(f"GPU limits: host memory <= {lim_mem} MB")
        if a.walltime_min > lim_w or a.cores > lim_n:
            raise SystemExit(f"GPU limits: walltime <= {lim_w} min, cores <= {lim_n}")
        if not a.exclusive and (a.gmem is None or a.gmem > 40):
            raise SystemExit("shared GPU job needs --gmem <= 40")
        a.queue = "short-gpu"
        if a.end_to_end or a.exclusive:
            nthr = a.cores  # as production: BLAS/OpenMP threads match the cores
        model = "" if a.gpu == "any" else f"gmodel={a.gpu}:"  # "any": untimed jobs on any model
        g = f"num=1:{model}" + ("j_exclusive=yes" if a.exclusive
                                         else f"gmem={a.gmem}G:j_exclusive=no")
        gpu_lines = f'#BSUB -gpu "{g}"\n'
        gpu_setup = ("source /etc/profile >/dev/null 2>&1\n"
                     "module load CUDA/12.6.0 >/dev/null 2>&1\n"
                     "export CUDA_PATH=$EBROOTCUDA LD_LIBRARY_PATH=$EBROOTCUDA/lib64:$LD_LIBRARY_PATH\n"
                     "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu "
                     "--format=csv\n")
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    logs = REPO / "artifacts/research/logs" / a.name
    logs.mkdir(parents=True, exist_ok=True)
    py = f"/home/projects/barakz/jonatahm/miniconda3/envs/{a.env}/bin/python"
    pythonpath = (f"export PYTHONPATH={REPO}:{REPO}/dot-pe-gpu:$PYTHONPATH"
                  if a.env in ("dot-pe", "dot-pe-gpu") else "")
    script = f"""#!/bin/bash
#BSUB -J {a.name}
#BSUB -o {logs}/%J.out
#BSUB -e {logs}/%J.err
#BSUB -n {a.cores}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem={a.mem_mb // a.cores}]"
#BSUB -q {a.queue}
#BSUB -W {a.walltime_min}
{gpu_lines}{gpu_setup}set -e
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS={nthr} MKL_NUM_THREADS={nthr} OPENBLAS_NUM_THREADS={nthr} NUMEXPR_NUM_THREADS={nthr}
{pythonpath}
cd /tmp
echo "[$(date '+%F %T')] starting on $(hostname); dot-PE $(git -C {REPO} rev-parse --short HEAD)"
{py} {' '.join(shlex.quote(c) for c in cmd)}
echo "[$(date '+%F %T')] finished"
"""
    if a.dry_run:
        print(script)
        return
    f = logs / f"{a.name}.bsub"
    f.write_text(script)
    out = subprocess.run(["bsub"], stdin=f.open(), capture_output=True, text=True)
    m = re.search(r"Job <(\d+)>", out.stdout)
    print(m.group(1) if m else out.stdout + out.stderr)


if __name__ == "__main__":
    main()
