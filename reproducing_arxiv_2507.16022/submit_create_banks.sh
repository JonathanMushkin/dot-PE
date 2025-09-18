#!/bin/bash
"""
Submit jobs to create banks for arXiv:2507.16022 experiments.

This script submits jobs to create all required banks:
- Regular banks: bank_mchirp_{3,5,10,20,50,100} with 2^16 samples
- Dense banks: bank_mchirp_dense_{3,5,10,20,50,100} with 2^18 samples

Each bank covers different chirp mass ranges to test convergence
behavior across the gravitational wave parameter space.

Usage: ./submit_create_banks.sh [n_jobs]
  n_jobs: Number of parallel jobs to use (default: 1)
"""

# Parse command line arguments
if [ -z "$1" ]; then
    N_JOBS=1  # Default to 1 job if not specified
else
    N_JOBS=$1
fi

# Configuration
REPRODUCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BANKS_HOMEDIR="$REPRODUCE_DIR/banks"  # Directory where banks will be created
CREATE_BANKS_SCRIPT="$REPRODUCE_DIR/create_banks.py"
N_CORES=8          # Cores for bank generation
MEMORY="16G"       # Memory per job
WALLTIME=2880      # 48 hours wall time

# Bank configurations
BANK_TYPES=("regular" "dense")
MASS_RANGES=(3 5 10 20 50 100)
TOTAL_BANKS=$(( ${#BANK_TYPES[@]} * ${#MASS_RANGES[@]} ))

# Calculate banks per job
BANKS_PER_JOB=$(( (TOTAL_BANKS + N_JOBS - 1) / N_JOBS ))

# Create logs and banks directories
mkdir -p logs
mkdir -p "$BANKS_HOMEDIR"

echo "Submitting bank creation jobs..."
echo "Script: $CREATE_BANKS_SCRIPT"
echo "Banks will be created in: $BANKS_HOMEDIR"
echo "Total banks: $TOTAL_BANKS"
echo "Number of jobs: $N_JOBS"
echo "Banks per job: $BANKS_PER_JOB"
echo "Cores per job: $N_CORES"
echo "Memory per job: $MEMORY"
echo "Wall time: $WALLTIME minutes"

# Check if create_banks.py exists
if [ ! -f "$CREATE_BANKS_SCRIPT" ]; then
    echo "Error: create_banks.py not found: $CREATE_BANKS_SCRIPT"
    exit 1
fi

# Create list of all banks to process
ALL_BANKS=()
for bank_type in "${BANK_TYPES[@]}"; do
    for mass_range in "${MASS_RANGES[@]}"; do
        if [ "$bank_type" = "regular" ]; then
            ALL_BANKS+=("$mass_range")
        else
            ALL_BANKS+=("dense_$mass_range")
        fi
    done
done

# Submit jobs
for ((job_id=1; job_id<=N_JOBS; job_id++)); do
    start_idx=$(( (job_id - 1) * BANKS_PER_JOB ))
    end_idx=$(( start_idx + BANKS_PER_JOB - 1 ))
    
    # Don't exceed total number of banks
    if [ $end_idx -ge $TOTAL_BANKS ]; then
        end_idx=$((TOTAL_BANKS - 1))
    fi
    
    # Skip if no banks for this job
    if [ $start_idx -ge $TOTAL_BANKS ]; then
        break
    fi
    
    # Build bank list for this job
    BANKS_FOR_JOB=""
    for ((i=start_idx; i<=end_idx; i++)); do
        BANKS_FOR_JOB="${BANKS_FOR_JOB} ${ALL_BANKS[i]}"
    done
    
    echo "Job $job_id will process banks:$BANKS_FOR_JOB"
    
    # Submit LSF job
    bsub -J "create_banks_${job_id}" \
         -q physics-medium \
         -R "affinity[core(${N_CORES})]" \
         -R "rusage[mem=${MEMORY}]" \
         -W $WALLTIME \
         -o "logs/create_banks_${job_id}_output.log" \
         -e "logs/create_banks_${job_id}_error.log" \
         /bin/bash -c "
         source ~/miniconda3/etc/profile.d/conda.sh && conda activate dot-pe && \
         export OMP_NUM_THREADS=$N_CORES && \
         cd \"$REPRODUCE_DIR\" && \
         python \"$CREATE_BANKS_SCRIPT\" --banks_dir=\"$BANKS_HOMEDIR\" --mass_ranges$BANKS_FOR_JOB \
         || echo \"Bank creation failed for job $job_id\" >> logs/failed_bank_creation.log
         "
done

echo ""
echo "Bank creation jobs submitted!"
echo "Submitted $N_JOBS job(s) to create $TOTAL_BANKS banks"
echo ""
echo "Expected output in $BANKS_HOMEDIR:"
echo "- Regular banks: bank_mchirp_{3,5,10,20,50,100}/"
echo "- Dense banks: bank_mchirp_dense_{3,5,10,20,50,100}/"
echo ""
echo "Monitor progress with: bjobs -w"
echo "Check logs in: logs/create_banks_*"
echo "Check for failures in: logs/failed_bank_creation.log"
