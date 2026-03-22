"""Collect benchmark results and write a plain-text report.

Run after all benchmark configs have completed:
    python collect_benchmark_results.py

Reads gpu/artifacts/benchmark_scaling/*/benchmark_timings.json
Writes gpu/artifacts/benchmark_scaling/report.txt
"""
import json
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT / "gpu/artifacts/benchmark_scaling"

STAGE_KEYS = [
    "1_setup", "2_incoherent", "3_crossbank",
    "4_extrinsic", "5_coherent", "6_postprocess",
]
STAGE_NAMES = {
    "1_setup":       "Stage 1 (setup)",
    "2_incoherent":  "Stage 2 (incoherent)",
    "3_crossbank":   "Stage 3 (cross-bank)",
    "4_extrinsic":   "Stage 4 (extrinsic)",
    "5_coherent":    "Stage 5 (coherent)",
    "6_postprocess": "Stage 6 (postprocess)",
}

# Expected config order
CONFIG_ORDER = [
    "benchmark_serial",
    "benchmark_mp_8w", "benchmark_mp_16w", "benchmark_mp_32w",
    "benchmark_dask_8w", "benchmark_dask_16w", "benchmark_dask_32w",
]

CONFIG_LABELS = {
    "benchmark_serial":   "Serial (8 BLAS threads)",
    "benchmark_mp_8w":    "MP  8w  (1 thread/worker)",
    "benchmark_mp_16w":   "MP 16w  (1 thread/worker)",
    "benchmark_mp_32w":   "MP 32w  (1 thread/worker)",
    "benchmark_dask_8w":  "Dask  8w (1 thread/worker)",
    "benchmark_dask_16w": "Dask 16w (1 thread/worker)",
    "benchmark_dask_32w": "Dask 32w (1 thread/worker)",
}

# ── load results ───────────────────────────────────────────────────────
results = {}
for rundir in sorted(OUTPUT_ROOT.glob("benchmark_*")):
    timing_file = rundir / "benchmark_timings.json"
    if timing_file.exists():
        with open(timing_file) as f:
            results[rundir.name] = json.load(f)

if not results:
    print(f"No timing files found in {OUTPUT_ROOT}")
    raise SystemExit(1)

# Respect expected order, append any extras at the end
ordered_keys = [k for k in CONFIG_ORDER if k in results]
ordered_keys += [k for k in results if k not in CONFIG_ORDER]

# ── formatting helpers ─────────────────────────────────────────────────
def fmt(val, width=10):
    if val is None or (isinstance(val, float) and val != val):
        return f"{'—':>{width}}"
    return f"{val:>{width}.1f}"

COL_W = 10
LABEL_W = 28

# ── build report ──────────────────────────────────────────────────────
lines = []
def p(*args): lines.append(" ".join(str(a) for a in args))

p("Scaling Benchmark Results")
p("=" * 80)
p()
p(f"Event : gpu/artifacts/nb04/nb04_event.npz")
p(f"Bank  : gpu/artifacts/profile_run/test_bank_1048576/  (1M samples, IMRPhenomXODE)")
p(f"Kwargs: n_ext=512  n_phi=32  n_t=64  blocksize=2048  seed=42")
p(f"        max_incoherent_lnlike_drop=20.0")
p()

# ── per-stage wall-clock table ─────────────────────────────────────────
p("Wall-clock time per stage (seconds)")
p("-" * 80)
header = f"{'Config':<{LABEL_W}}" + "".join(
    f"{STAGE_NAMES[k]:>{COL_W}}" for k in STAGE_KEYS
) + f"{'Total':>{COL_W}}"
p(header)
p("-" * 80)

for key in ordered_keys:
    r = results[key]
    label = CONFIG_LABELS.get(key, key)
    row = f"{label:<{LABEL_W}}"
    for sk in STAGE_KEYS:
        row += fmt(r.get(sk), COL_W)
    row += fmt(r.get("total"), COL_W)
    p(row)

p()

# ── speedup table (relative to serial) ────────────────────────────────
serial_key = next((k for k in ordered_keys if "serial" in k), None)

if serial_key:
    serial = results[serial_key]
    p(f"Speedup vs {CONFIG_LABELS.get(serial_key, serial_key)}")
    p("-" * 80)
    header2 = f"{'Config':<{LABEL_W}}" + "".join(
        f"{STAGE_NAMES[k]:>{COL_W}}" for k in STAGE_KEYS
    ) + f"{'Total':>{COL_W}}"
    p(header2)
    p("-" * 80)

    for key in ordered_keys:
        r = results[key]
        label = CONFIG_LABELS.get(key, key)
        row = f"{label:<{LABEL_W}}"
        for sk in STAGE_KEYS:
            s = serial.get(sk)
            c = r.get(sk)
            if s and c and c > 0:
                row += f"{s/c:>{COL_W}.2f}x"
            else:
                row += f"{'—':>{COL_W}}"
        s = serial.get("total")
        c = r.get("total")
        if s and c and c > 0:
            row += f"{s/c:>{COL_W}.2f}x"
        else:
            row += f"{'—':>{COL_W}}"
        p(row)
    p()

# ── per-config detail ──────────────────────────────────────────────────
p("Per-config detail")
p("-" * 80)
for key in ordered_keys:
    r = results[key]
    label = CONFIG_LABELS.get(key, key)
    p(f"  {label}")
    p(f"    module    : {r.get('module', '?')}")
    p(f"    n_workers : {r.get('n_workers', 0)}")
    p(f"    omp_threads: {r.get('omp_threads', '?')}")
    for sk in STAGE_KEYS:
        v = r.get(sk)
        if v is not None:
            p(f"    {STAGE_NAMES[sk]:<26}: {v:.1f} s")
    p(f"    {'Total':<26}: {r.get('total', 0):.1f} s")
    p()

# ── write report ───────────────────────────────────────────────────────
report_path = OUTPUT_ROOT / "report.txt"
report_text = "\n".join(lines)
print(report_text)
with open(report_path, "w") as f:
    f.write(report_text + "\n")
print(f"\nReport saved: {report_path}")
