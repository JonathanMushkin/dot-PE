#!/bin/bash
"""
Submit reference runs for arXiv:2507.16022 convergence experiments.

This script submits jobs for reference runs using dense banks.
Each injection is processed once with optimal parameters to establish
the reference/ground truth for convergence testing.

Reference runs use:
- Dense banks (bank_mchirp_dense_*)
- Single run per injection 
- Optimal parameters: n_int=2^16, n_ext=2^10
"""

# Configuration
BANK_NAMES=("bank_mchirp_dense_3" "bank_mchirp_dense_5" "bank_mchirp_dense_10" "bank_mchirp_dense_20" "bank_mchirp_dense_50" "bank_mchirp_dense_100")
REPRODUCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N_INJECTIONS=1024  # Number of injections per mass range
BATCH_SIZE=4       # Process 4 injections per job
N_CORES=16         # Cores per job
CONVERGENCE_SCRIPT="$REPRODUCE_DIR/convergence.py"

# Create logs directory
mkdir -p logs

echo "Starting reference runs submission..."
echo "Banks: ${BANK_NAMES[@]}"
echo "Injections per bank: $N_INJECTIONS"
echo "Batch size: $BATCH_SIZE"

for BANK_NAME in "${BANK_NAMES[@]}"; do
    echo "Submitting jobs for $BANK_NAME"
    
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
    bsub -J "reference_${BANK_NAME}[1-$NUM_BATCHES]" \
         -q physics-medium \
         -R "affinity[core(${N_CORES})]" \
         -R "rusage[mem=32G]" \
         -W 4320 \
         -o "logs/reference_${BANK_NAME}_%I_output.log" \
         -e "logs/reference_${BANK_NAME}_%I_error.log" \
         /bin/bash -c "
         source ~/miniconda3/etc/profile.d/conda.sh && conda activate dot-pe && \
         export OMP_NUM_THREADS=$N_CORES && \
         i_start=\$(( (\$LSB_JOBINDEX - 1) * $BATCH_SIZE )) && \
         i_end=\$(( i_start + $BATCH_SIZE - 1 < $N_INJECTIONS ? i_start + $BATCH_SIZE - 1 : $N_INJECTIONS - 1 )) && \
         for i in \$(seq \$i_start \$i_end); do \
             echo \"Processing injection \$i for $BANK_NAME\" && \
             python \"$CONVERGENCE_SCRIPT\" \
                 \"$BANK_FOLDER\" \
                 \"$EVENTS_HOMEDIR\" \
                 \"\$i\" \
                 --run_type reference \
                 --base_seed 42 \
                 || echo \"Failed for $BANK_NAME injection \$i\" >> logs/failed_reference_jobs.log; \
         done
         "
    
    echo "Submitted $NUM_BATCHES jobs for $BANK_NAME"
done

echo "Reference runs submission completed!"
echo "Monitor progress with: bjobs -w"
echo "Check logs in: logs/reference_*"
