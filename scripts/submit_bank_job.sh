#!/bin/bash
# LSF job submission script for creating GW231123 waveform bank
# 
# Usage:
#   bsub < submit_bank_job.sh
# 
# Or submit with custom job name:
#   bsub -J "my_bank_job" < submit_bank_job.sh
#
# This script will automatically detect and regenerate missing waveform files
# Perfect for scattered missing files.
#
#BSUB -J create_bank           # Job name
#BSUB -o bank_creation_%J.out  # Output file (%J = job ID)
#BSUB -e bank_creation_%J.err  # Error file
#BSUB -n 4                     # Number of cores (reduced for memory efficiency)
#BSUB -R "span[hosts=1]"       # All cores on same host
#BSUB -R "rusage[mem=10000]"   # Memory reservation (10GB per core = 40GB total)
#BSUB -q physics-long          # Physics long queue (10080 min = 7 days limit)
#BSUB -W 3000                  # Wall time limit (50 hours = ~2.1 days)

# Load any required modules (adjust as needed)
# module load python/3.13
# source activate dot-pe

# Set working directory
cd "/home/projects/barakz/jonatahm/GW/Collaboration-gw/mushkin/dot-pe/heavy_events"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dot-pe

# Run the bank creation script with auto-detect resume
echo "Running bank creation with auto-detect resume functionality"
python create_GW231123_bank.py GW231123_bank_5 \
    --bank-size $((2**25)) \
    --n-pool 4 \
    --blocksize $((2**14)) \
    --limited-fbin \
    --resume

# Print resource usage summary
echo "Job resource usage:"
bjobs -l $LSB_JOBID

echo "Job completed at $(date)"
