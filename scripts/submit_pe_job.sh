#!/bin/bash
# LSF job submission script for GW231123 parameter estimation
# 
# Usage:
#   bsub < submit_pe_job.sh
# 
# Or submit with custom job name:
#   bsub -J "my_pe_job" < submit_pe_job.sh
#
# This script runs parameter estimation on GW231123 using the created waveform bank
#
#BSUB -J pe_GW231123            # Job name
#BSUB -o pe_analysis_%J.out     # Output file (%J = job ID)
#BSUB -e pe_analysis_%J.err     # Error file
#BSUB -n 8                      # Number of cores for PE analysis
#BSUB -R "span[hosts=1]"        # All cores on same host
#BSUB -R "rusage[mem=8000]"     # Memory reservation (8GB per core = 64GB total)
#BSUB -q physics-long           # Physics long queue (10080 min = 7 days limit)
#BSUB -W 10080                  # Wall time limit (168 hours = 7 days)

# Load any required modules (adjust as needed)
# module load python/3.13
# source activate dot-pe

# Set working directory
cd "/home/projects/barakz/jonatahm/GW/Collaboration-gw/mushkin/dot-pe/heavy_events"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dot-pe

# Run the PE analysis script
echo "Running parameter estimation for GW231123"
echo "Start time: $(date)"
python run_GW231123_pe.py

# Print resource usage summary
echo "Job resource usage:"
bjobs -l $LSB_JOBID

echo "Job completed at $(date)"
