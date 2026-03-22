#!/usr/bin/env bash
# Run the scaling benchmark in 3 parallel tmux sessions.
#
# Sessions:
#   benchmark_serial  — inference          (OMP_NUM_THREADS=8)
#   benchmark_mp      — mp_inference  8/16/32w  (OMP_NUM_THREADS=1)
#   benchmark_dask    — dask_inference 8/16/32w (OMP_NUM_THREADS=1)
#
# After all sessions finish, collect results:
#   python collect_benchmark_results.py

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/jonatahm/miniconda3/envs/dot-pe/bin/python"
RUNNER="$REPO_ROOT/run_one_benchmark.py"

# ── kill any existing benchmark sessions ──────────────────────────────
for s in benchmark_serial benchmark_mp benchmark_dask; do
    tmux kill-session -t "$s" 2>/dev/null && echo "Killed existing session: $s" || true
done

# ── helper: start a session and send a command ────────────────────────
start_session() {
    local session="$1"
    local cmd="$2"
    tmux new-session -d -s "$session" -x 220 -y 50
    tmux send-keys -t "$session" "cd $REPO_ROOT && $cmd" Enter
    echo "Started tmux session: $session"
}

# ── Serial (OMP=8, no n_workers) ─────────────────────────────────────
start_session benchmark_serial \
    "$PYTHON $RUNNER inference 0 benchmark_serial 8"

# ── MP: 8w -> 16w -> 32w sequentially (OMP=1 per worker) ─────────────
start_session benchmark_mp \
    "$PYTHON $RUNNER mp_inference  8 benchmark_mp_8w  1 && \
     $PYTHON $RUNNER mp_inference 16 benchmark_mp_16w 1 && \
     $PYTHON $RUNNER mp_inference 32 benchmark_mp_32w 1"

# ── Dask: 8w -> 16w -> 32w sequentially (OMP=1 per worker) ──────────
start_session benchmark_dask \
    "$PYTHON $RUNNER dask_inference  8 benchmark_dask_8w  1 && \
     $PYTHON $RUNNER dask_inference 16 benchmark_dask_16w 1 && \
     $PYTHON $RUNNER dask_inference 32 benchmark_dask_32w 1"

# ── instructions ──────────────────────────────────────────────────────
echo ""
echo "All 3 sessions running.  Attach with:"
echo "  tmux attach -t benchmark_serial"
echo "  tmux attach -t benchmark_mp"
echo "  tmux attach -t benchmark_dask"
echo ""
echo "When all sessions finish, collect results:"
echo "  python collect_benchmark_results.py"
echo ""
echo "Output dir: $REPO_ROOT/gpu/artifacts/benchmark_scaling/"
