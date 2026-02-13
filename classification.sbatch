#!/bin/bash
################################ Slurm options #################################
#SBATCH --job-name=classification_gliph
#SBATCH --partition=fast
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --account=airr_2025_ml_blosum
#SBATCH --output=/shared/projects/airr_2025_ml_blosum/pipeline_complete/out/classification_gliph-%j.out
#SBATCH --error=/shared/projects/airr_2025_ml_blosum/pipeline_complete/out/classification_gliph-%j.err
################################################################################

# Print job info
echo "########################################"
echo "Job started: $(date --iso-8601=seconds)"
echo "Running on node: $(hostname)"
echo "CPUs available: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: ${SLURM_MEM_PER_NODE}MB"
echo "########################################"
echo ""


# ---- Load conda environment ----
CONDA_ROOT=/shared/software/miniconda/
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate tcrkit


#  ---- Verify environment ---- 
echo "Python version: $(python --version)"
echo "Python path: $(which python)"


# ---- Run Python script ----
echo "Starting TCR classification pipeline..."
snakemake --cores 4

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "########################################"
    echo "✓ Job completed successfully!"
    echo "Job finished: $(date --iso-8601=seconds)"
    echo "########################################"
else
    echo ""
    echo "########################################"
    echo "✗ Job failed with exit code $?"
    echo "Job finished: $(date --iso-8601=seconds)"
    echo "########################################"
    exit 1
fi