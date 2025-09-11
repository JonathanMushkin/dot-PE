#!/bin/bash
"""
Submit convergence runs for arXiv:2507.16022 convergence experiments.

This script submits jobs for convergence runs using regular banks.
Each injection is processed multiple times (20 repeats) with varying 
parameters to test convergence behavior.

Convergence runs test:
1. Intrinsic convergence: vary n_int from 2^6 to 2^16, fix n_ext=2^10
2. Extrinsic convergence: fix n_int=2^16, vary n_ext from 2^1 to 2^10

Regular banks used: bank_mchirp_* (not dense)
"""

# Configuration
BANK_NAMES=("bank_mchirp_3" "bank_mchirp_5" "bank_mchirp_10" "bank_mchirp_20" "bank_mchirp_50" "bank_mchirp_100")
REPRODUCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N_INJECTIONS=1024  # Number of injections per mass range
BATCH_SIZE=1       # Process 1 injection per job (convergence runs are computationally intensive)
N_CORES=32         # Cores per job (more cores needed for convergence runs)
REPEATS=20         # Number of repeats per parameter combination
CONVERGENCE_SCRIPT="$REPRODUCE_DIR/convergence.py"

# Convergence test types
CONVERGENCE_TYPES=("intrinsic" "extrinsic")

# Create logs directory
mkdir -p logs

echo "Starting convergence runs submission..."
echo "Banks: ${BANK_NAMES[@]}"
echo "Convergence types: ${CONVERGENCE_TYPES[@]}"
echo "Injections per bank: $N_INJECTIONS"
echo "Batch size: $BATCH_SIZE"
echo "Repeats per parameter: $REPEATS"

for BANK_NAME in "${BANK_NAMES[@]}"; do
    for CONV_TYPE in "${CONVERGENCE_TYPES[@]}"; do
        echo "Submitting jobs for $BANK_NAME - $CONV_TYPE convergence"
        
        # Calculate number of batches needed
        NUM_BATCHES=$(( (N_INJECTIONS + BATCH_SIZE - 1) / BATCH_SIZE ))
        
        # Bank and events directories
        BANK_FOLDER="$REPRODUCE_DIR/$BANK_NAME"
        EVENTS_HOMEDIR="$REPRODUCE_DIR/${BANK_NAME}_events"
        
        # Check if bank exists
        if [ ! -d "$BANK_FOLDER" ]; then
            echo "Warning: Bank folder not found: $BANK_FOLDER"
            echo "Skipping $BANK_NAME"
            continue
        fi
        
        # Check if injections exist
        INJECTION_FILE="$EVENTS_HOMEDIR/injections.feather"
        if [ ! -f "$INJECTION_FILE" ]; then
            echo "Warning: Injection file not found: $INJECTION_FILE"
            echo "Skipping $BANK_NAME"
            continue
        fi
        
        # Submit LSF job array
        bsub -J "convergence_${BANK_NAME}_${CONV_TYPE}[1-$NUM_BATCHES]" \
             -q physics-long \
             -R "affinity[core(${N_CORES})]" \
             -R "rusage[mem=64G]" \
             -W 7200 \
             -o "logs/convergence_${BANK_NAME}_${CONV_TYPE}_%I_output.log" \
             -e "logs/convergence_${BANK_NAME}_${CONV_TYPE}_%I_error.log" \
             /bin/bash -c "
             source ~/miniconda3/etc/profile.d/conda.sh && conda activate dot-pe && \
             export OMP_NUM_THREADS=$N_CORES && \
             i_start=\$(( (\$LSB_JOBINDEX - 1) * $BATCH_SIZE )) && \
             i_end=\$(( i_start + $BATCH_SIZE - 1 < $N_INJECTIONS ? i_start + $BATCH_SIZE - 1 : $N_INJECTIONS - 1 )) && \
             for i in \$(seq \$i_start \$i_end); do \
                 echo \"Processing injection \$i for $BANK_NAME ($CONV_TYPE convergence)\" && \
                 python \"$CONVERGENCE_SCRIPT\" \
                     \"$BANK_FOLDER\" \
                     \"$EVENTS_HOMEDIR\" \
                     \"\$i\" \
                     --run_type convergence \
                     --convergence_type \"$CONV_TYPE\" \
                     --repeats \"$REPEATS\" \
                     --base_seed 42 \
                     || echo \"Failed for $BANK_NAME injection \$i ($CONV_TYPE)\" >> logs/failed_convergence_jobs.log; \
             done
             "
        
        echo "Submitted $NUM_BATCHES jobs for $BANK_NAME ($CONV_TYPE)"
    done
done

echo ""
echo "Convergence runs submission completed!"
echo ""
echo "Summary:"
echo "- Banks: ${#BANK_NAMES[@]}"
echo "- Convergence types per bank: ${#CONVERGENCE_TYPES[@]}"
echo "- Total job arrays: $(( ${#BANK_NAMES[@]} * ${#CONVERGENCE_TYPES[@]} ))"
echo "- Jobs per array: $NUM_BATCHES"
echo "- Total jobs: $(( ${#BANK_NAMES[@]} * ${#CONVERGENCE_TYPES[@]} * NUM_BATCHES ))"
echo ""
echo "Expected runs per injection:"
echo "- Intrinsic convergence: 11 parameter combinations × $REPEATS repeats = $(( 11 * REPEATS )) runs"
echo "- Extrinsic convergence: 10 parameter combinations × $REPEATS repeats = $(( 10 * REPEATS )) runs"
echo "- Total per injection: $(( (11 + 10) * REPEATS )) runs"
echo ""
echo "Monitor progress with: bjobs -w"
echo "Check logs in: logs/convergence_*"
