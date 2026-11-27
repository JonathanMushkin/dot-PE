"""Submit the P5 revision-2 GPU jobs in order, keeping at most 2 of our short-gpu jobs running or
pending (protocol "GPU usage", item 2). Runs on the login node: it only polls `bjobs` and calls
`lsf_submit.py`. Each submission is appended to artifacts/research/acceptance/ps/submitted.tsv.

Usage: python research/acceptance/drive.py [--from N]   (N: first job number of JOBS to submit)
"""

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
K = REPO / "research/acceptance/kwargs/ps"
LOG = REPO / "artifacts/research/acceptance/ps/submitted.tsv"
PY = "/home/projects/barakz/jonatahm/miniconda3/envs/dot-pe/bin/python"
MAX_LIVE = 2

E = {1: "GW240515_005301", 2: "GW240923_204006", 3: "GW231028_153006", 4: "GW240920_124024",
     5: "GW241011_233834", 6: "GW240916_184352", 7: "GW231002_143916", "R": "GW240525_201644"}


def timed(n, w):
    return (f"p5r2_t{n}", ["--gpu", "NVIDIAL40S", "--exclusive", "--cores", "8", "--mem-mb", "64000",
                           "--walltime-min", str(w), "--end-to-end"], [f"{E[n]}_fresh.json"])


def rec(n, seeds, w=60, gmem=24):
    return (f"p5r2_r{n}_" + "".join(map(str, seeds)),
            ["--gpu", "any", "--gmem", str(gmem), "--cores", "8", "--mem-mb", "64000",
             "--walltime-min", str(w), "--end-to-end"], [f"{E[n]}_rec_s{s}.json" for s in seeds])


# Order: small events first, so that results arrive early; event 7 (disk streaming) last.
JOBS = [timed(2, 60), rec(2, [0, 1, 2]), timed(4, 30), rec(4, [0, 1, 2]),
        timed(3, 30), rec(3, [0, 1, 2]), timed(5, 30), rec(5, [0, 1, 2]),
        timed(1, 90), rec(1, [0, 1]), rec(1, [2]),
        timed(6, 120), rec(6, [0, 1]), rec(6, [2]),
        timed("R", 90), rec("R", [0, 1]), rec("R", [2]),
        timed(7, 180), rec(7, [0]), rec(7, [1]), rec(7, [2])]


def live():
    out = subprocess.run(["bjobs", "-q", "short-gpu", "-noheader", "-o", "jobid stat"],
                         capture_output=True, text=True).stdout
    return sum(1 for line in out.splitlines() if line.split()[-1:] and line.split()[-1] in ("RUN", "PEND", "PSUSP", "USUSP", "SSUSP"))


def main():
    start = int(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 0
    LOG.parent.mkdir(parents=True, exist_ok=True)
    for n, (name, opts, kws) in enumerate(JOBS):
        if n < start:
            continue
        while live() >= MAX_LIVE:
            time.sleep(60)
        cmd = [PY, str(REPO / "research/lsf_submit.py"), "--name", name, "--env", "dot-pe-gpu", *opts, "--",
               str(REPO / "research/acceptance/run_many.py"), *[str(K / k) for k in kws]]
        r = subprocess.run(cmd, capture_output=True, text=True)
        jid = r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else "FAILED"
        line = f"{n}\t{name}\t{jid}\t{time.strftime('%Y-%m-%d %H:%M')}\t{' '.join(opts)}\t{' '.join(kws)}"
        print(line, flush=True)
        with open(LOG, "a") as f:
            f.write(line + "\n")
        if jid == "FAILED":
            print(r.stdout, r.stderr, flush=True)
            sys.exit(1)
        time.sleep(20)  # let bjobs see the new job
    print("all submitted", flush=True)


if __name__ == "__main__":
    main()
