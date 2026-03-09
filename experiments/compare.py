#!/usr/bin/env python3
"""
Scan artifacts/experiments/*/run.log, extract timing and evidence, print Markdown table.

Usage:
    python experiments/compare.py
    python experiments/compare.py --artifacts-dir /path/to/artifacts
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPERIMENTS = ROOT / "artifacts" / "experiments"

_BANK_SIZE = {"small": "2^12", "large": "2^18"}

_WALL_RE     = re.compile(r"Total wall-clock time:\s*([\d.]+)\s*s", re.IGNORECASE)
_LSF_TIME_RE = re.compile(r"Run time\s*:\s*([\d.]+)\s*sec", re.IGNORECASE)
_MODE_RE     = re.compile(
    r"\d{8}_\d{6}_(serial|mp|swarm)(?:_[cw](\d+))?_(small|large)_next(\d+)"
)


def _parse_rundir(rundir):
    """Return (mode, n_workers, bank, n_ext) from the directory name."""
    m = _MODE_RE.search(rundir.name)
    if not m:
        return None, None, None, None
    mode = m.group(1)
    n_workers = int(m.group(2)) if m.group(2) else None
    bank = m.group(3)
    n_ext = int(m.group(4))
    return mode, n_workers, bank, n_ext


def _parse_log(log_path):
    """Return wall_time (float or None) from run.log.

    Prefers 'Total wall-clock time' (emitted by mp/swarm runners).
    Falls back to LSF 'Run time' only for successfully completed jobs
    (serial doesn't emit 'Total wall-clock time').
    """
    try:
        text = log_path.read_text()
    except OSError:
        return None
    m = _WALL_RE.search(text)
    if m:
        return float(m.group(1))
    if "Successfully completed" not in text:
        return None
    m = _LSF_TIME_RE.search(text)
    return float(m.group(1)) if m else None


def _parse_summary(rundir):
    """Return (ln_evidence, n_effective) from summary_results.json if present."""
    for candidate in rundir.rglob("summary_results.json"):
        try:
            d = json.loads(candidate.read_text())
            return d.get("ln_evidence"), d.get("n_effective")
        except (OSError, json.JSONDecodeError):
            pass
    return None, None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts-dir", type=Path, default=DEFAULT_EXPERIMENTS,
                   help="Path to experiments/ directory")
    args = p.parse_args()

    exp_dir = args.artifacts_dir
    if not exp_dir.exists():
        print(f"No experiments directory found at {exp_dir}")
        return

    rows = []
    for rundir in sorted(exp_dir.iterdir()):
        if not rundir.is_dir():
            continue
        mode, n_workers, bank, n_ext = _parse_rundir(rundir)
        if mode is None:
            continue
        log_path = rundir / "run.log"
        wall = _parse_log(log_path)
        ln_ev, n_eff = _parse_summary(rundir)
        rows.append((bank, n_ext, mode, n_workers, wall, ln_ev, n_eff, rundir.name))

    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3] or 0))

    HEADERS = ["bank", "n_ext", "mode", "n_workers", "wall_s", "ln_evidence", "n_effective", "rundir"]

    formatted = []
    for bank, n_ext, mode, n_workers, wall, ln_ev, n_eff, name in rows:
        if mode == "serial":
            nw_str = str(n_workers) if n_workers is not None else "1"
        elif n_workers is not None:
            nw_str = str(n_workers)
        else:
            nw_str = "—"
        wall_str = f"{wall:.1f}" if wall is not None else "—"
        lne_str  = f"{ln_ev:.3f}" if ln_ev is not None else "—"
        neff_str = f"{n_eff:.1f}" if n_eff is not None else "—"
        bank_str = _BANK_SIZE.get(bank, bank)
        formatted.append([bank_str, str(n_ext), mode, nw_str, wall_str, lne_str, neff_str, name])

    col_widths = [len(h) for h in HEADERS]
    for row in formatted:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells):
        return "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"

    print(fmt_row(HEADERS))
    print("|-" + "-|-".join("-" * w for w in col_widths) + "-|")
    for row in formatted:
        print(fmt_row(row))


if __name__ == "__main__":
    main()
