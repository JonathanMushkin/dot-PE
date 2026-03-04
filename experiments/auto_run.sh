#!/bin/bash
# experiments/auto_run.sh
#
# Chains all benchmark phases unattended.  Run inside tmux:
#
#   tmux new -s benchmark
#   bash experiments/auto_run.sh [--skip-banks]
#   Ctrl+B D   (detach)
#
# Options:
#   --skip-banks   skip bank generation (use if banks already exist)
#
# All output (including errors) is tee'd to experiments/auto_run.log.
# Job errors are also captured from LSF log files.
# After all phases finish, run in a new session:
#   python experiments/compare.py

set -uo pipefail   # -u: unbound vars are errors; NO -e so we can handle errors ourselves

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACTS="$ROOT/artifacts"
BANKS="$ARTIFACTS/banks"
EXPERIMENTS="$ARTIFACTS/experiments"
PROGRESS="$SCRIPT_DIR/PROGRESS.md"
RUNLOG="$SCRIPT_DIR/auto_run.log"

CONDA_BASE="$(conda info --base)"
CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"

SKIP_BANKS=0
for arg in "$@"; do
  [[ "$arg" == "--skip-banks" ]] && SKIP_BANKS=1
done

# Redirect all stdout+stderr through tee into RUNLOG for the whole script
exec > >(tee -a "$RUNLOG") 2>&1

# ── helpers ──────────────────────────────────────────────────────────────────

log() {
  local msg="[$(date '+%H:%M:%S')] $*"
  echo "$msg"
  echo "$msg" >> "$PROGRESS"
}

log_error() {
  local msg="[$(date '+%H:%M:%S')] ERROR: $*"
  echo "$msg" >&2
  echo "$msg" >> "$PROGRESS"
}

# Wait for an LSF job; returns 0 on success, 1 on any EXIT tasks.
# On failure, dumps last 30 lines of all .err log files for that job.
wait_job() {
  local job_id=$1 label=${2:-job}
  log "  waiting for $label (job $job_id) ..."
  while bjobs -noheader "$job_id" 2>/dev/null | grep -qE '\bRUN\b|\bPEND\b|\bSUSP\b'; do
    sleep 60
  done

  # bhist is more reliable than bjobs -a for recently-completed jobs
  local hist n_exit
  hist=$(bhist -l "$job_id" 2>/dev/null || true)
  n_exit=$(echo "$hist" | grep -c 'Exited with exit code' || true)
  if [[ "$n_exit" -gt 0 ]]; then
    log_error "$label (job $job_id): $n_exit tasks exited with error"
    # Dump the LSF output log (contains stderr from bsub -e redirected to same file)
    while IFS= read -r -d '' errfile; do
      log_error "--- $errfile (last 30 lines) ---"
      tail -30 "$errfile" >&2 || true
    done < <(find "$EXPERIMENTS" -name "run.log.err" -newer "$PROGRESS" -print0 2>/dev/null)
    find "$BANKS" -name "*_${job_id}.err" -exec tail -30 {} \; 2>/dev/null || true
    return 1
  fi
  log "  $label (job $job_id) finished OK"
  return 0
}

# Submit an experiment via run_experiment.py; echoes job_id on stdout.
# On bsub failure, logs error and echoes "FAILED".
run_exp() {
  local mode=$1 bank=$2 n_ext=$3 n_int=$4 n_workers=$5 queue=$6
  shift 6
  local extra=("$@")

  local args=(--mode "$mode" --bank "$bank" --n-ext "$n_ext" --seed 0 --queue "$queue")
  [[ "$n_int" != "-" ]] && args+=(--n-int "$n_int")
  [[ "$n_workers" != "-" ]] && args+=(--n-workers "$n_workers")
  args+=("${extra[@]}")

  log "Submitting: mode=$mode bank=$bank n_ext=$n_ext n_int=$n_int n_workers=$n_workers"

  local output rc
  output=$(python "$SCRIPT_DIR/run_experiment.py" "${args[@]}" 2>&1) || rc=$?
  echo "$output"   # visible in log via tee

  if [[ "${rc:-0}" -ne 0 ]]; then
    log_error "run_experiment.py failed for mode=$mode bank=$bank n_ext=$n_ext"
    log_error "$output"
    echo "FAILED"
    return 1
  fi

  local job_id
  job_id=$(echo "$output" | grep -oP 'Job <\K\d+' | head -1)
  if [[ -z "$job_id" ]]; then
    log_error "Could not parse job ID from output: $output"
    echo "FAILED"
    return 1
  fi

  echo "$job_id"
}

verify_banks() {
  local ok=1
  [[ -f "$BANKS/event/tutorial_event.npz" ]]              || { log_error "MISSING: event npz";           ok=0; }
  [[ -f "$BANKS/bank_small/intrinsic_sample_bank.feather" ]] || { log_error "MISSING: bank_small feather"; ok=0; }
  [[ -d "$BANKS/bank_small/waveforms" ]]                  || { log_error "MISSING: bank_small waveforms"; ok=0; }
  [[ -f "$BANKS/bank_large/intrinsic_sample_bank.feather" ]] || { log_error "MISSING: bank_large feather"; ok=0; }
  [[ -d "$BANKS/bank_large/waveforms" ]]                  || { log_error "MISSING: bank_large waveforms"; ok=0; }
  return $((1 - ok))
}

cache_extrinsic() {
  local bank=$1 n_ext=$2
  local cache_dir="$ARTIFACTS/extrinsic_cache/${bank}_n${n_ext}_seed0"
  if [[ -f "$cache_dir/extrinsic_samples_data.pkl" ]]; then
    echo "$cache_dir/extrinsic_samples_data.pkl"
    return
  fi
  local pkl
  pkl=$(find "$EXPERIMENTS" -maxdepth 2 \
        -path "*_${bank}_next${n_ext}/extrinsic_samples_data.pkl" \
        -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | awk '{print $2}')
  if [[ -n "$pkl" ]]; then
    mkdir -p "$cache_dir"
    cp "$pkl" "$cache_dir/extrinsic_samples_data.pkl"
    log "  cached extrinsic samples → $cache_dir"
    echo "$cache_dir/extrinsic_samples_data.pkl"
  fi
}

# Run a phase: takes a label and a list of "mode:bank:n_ext:n_int:n_workers:queue" specs.
# Submits all jobs, waits for all, logs errors but does NOT abort the script.
run_phase() {
  local phase_label=$1; shift
  log ""
  log "=== $phase_label ==="
  bqueues 2>/dev/null | grep -E "physics-(short|medium)" || true

  local job_ids=() labels=() all_ok=1
  local spec mode bank n_ext n_int n_workers queue
  for spec in "$@"; do
    IFS=: read -r mode bank n_ext n_int n_workers queue extra_args <<< "$spec"
    local extra=()
    [[ -n "$extra_args" ]] && IFS=',' read -ra extra <<< "$extra_args"

    local job_id
    job_id=$(run_exp "$mode" "$bank" "$n_ext" "$n_int" "$n_workers" "$queue" "${extra[@]}") || {
      log_error "$phase_label: failed to submit $mode/$bank/next$n_ext"
      all_ok=0
      continue
    }
    if [[ "$job_id" == "FAILED" ]]; then
      all_ok=0
      continue
    fi
    job_ids+=("$job_id")
    labels+=("$mode/$bank/next$n_ext")
  done

  local i
  for i in "${!job_ids[@]}"; do
    wait_job "${job_ids[$i]}" "${labels[$i]}" || all_ok=0
  done

  if [[ "$all_ok" -eq 1 ]]; then
    log "$phase_label: all jobs OK"
  else
    log_error "$phase_label: one or more jobs failed — continuing to next phase"
  fi
}

# ── main ─────────────────────────────────────────────────────────────────────

log "=============================="
log "dot-pe benchmark auto_run.sh"
log "=============================="
log "Log file: $RUNLOG"

# ── Bank generation ───────────────────────────────────────────────────────────

if [[ "$SKIP_BANKS" -eq 0 ]]; then
  if verify_banks 2>/dev/null; then
    log "Banks already exist, skipping generation"
  else
    log "=== Submitting bank generation ==="
    mkdir -p "$BANKS"
    BANK_OUT=$(bsub -q physics-short -n 8 -W 120 \
      -o "$BANKS/setup_%J.out" -e "$BANKS/setup_%J.err" \
      bash -c "set -e; source '$CONDA_SH' && conda activate dot-pe && \
               export PYTHONPATH='$ROOT':\$PYTHONPATH && \
               python '$ROOT/test_data/setup.py' --n-pool 8 --base-dir '$BANKS' \
               2>&1" 2>&1) || {
      log_error "bsub for bank generation failed: $BANK_OUT"
      exit 1
    }
    echo "$BANK_OUT"
    BANK_JOB=$(echo "$BANK_OUT" | grep -oP 'Job <\K\d+' | head -1)
    log "  bank job $BANK_JOB submitted"

    if ! wait_job "$BANK_JOB" "bank-setup"; then
      log_error "Bank generation failed. Dumping setup log:"
      find "$BANKS" -name "setup_${BANK_JOB}.out" -exec cat {} \; 2>/dev/null || true
      find "$BANKS" -name "setup_${BANK_JOB}.err" -exec cat {} \; 2>/dev/null || true
      exit 1
    fi
  fi

  if ! verify_banks; then
    log_error "Banks missing after generation. Aborting."
    exit 1
  fi
  log "Banks verified OK"
fi

# ── Phase A: smoke tests ──────────────────────────────────────────────────────

run_phase "Phase A: smoke tests" \
  "serial:small:128:256:-:physics-short" \
  "mp:small:128:256:4:physics-short" \
  "swarm:small:128:256:-:physics-short"

# ── Phase B: small bank, n_ext=512 ───────────────────────────────────────────

run_phase "Phase B: small/n_ext=512" \
  "serial:small:512:-:1:physics-medium" \
  "mp:small:512:-:4:physics-medium" \
  "mp:small:512:-:8:physics-medium" \
  "swarm:small:512:-:-:physics-short"

EXT_SMALL_512=$(cache_extrinsic small 512) || true

# ── Phase C: small bank, n_ext=2048 ──────────────────────────────────────────

EXT_C_ARGS=""
EXT_C=$(cache_extrinsic small 2048) || true
[[ -n "${EXT_C:-}" ]] && EXT_C_ARGS="--extrinsic-samples,$EXT_C"

run_phase "Phase C: small/n_ext=2048" \
  "serial:small:2048:-:1:physics-medium:${EXT_C_ARGS}" \
  "mp:small:2048:-:8:physics-medium:${EXT_C_ARGS}" \
  "swarm:small:2048:-:-:physics-short:${EXT_C_ARGS}"

# ── Phase D: large bank, n_ext=2048 ──────────────────────────────────────────

EXT_D_ARGS=""
EXT_D=$(cache_extrinsic large 2048) || true
[[ -n "${EXT_D:-}" ]] && EXT_D_ARGS="--extrinsic-samples,$EXT_D"

run_phase "Phase D: large/n_ext=2048" \
  "serial:large:2048:-:1:physics-medium:${EXT_D_ARGS}" \
  "mp:large:2048:-:8:physics-medium:${EXT_D_ARGS}" \
  "mp:large:2048:-:20:physics-medium:${EXT_D_ARGS}" \
  "swarm:large:2048:-:-:physics-short:${EXT_D_ARGS}"

# ── Summary ───────────────────────────────────────────────────────────────────

log ""
log "=============================="
log "All phases complete!"
log "=============================="
python "$SCRIPT_DIR/compare.py" 2>&1 | tee -a "$PROGRESS"
log "Full log: $RUNLOG"
