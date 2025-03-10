#!/bin/bash
#
#SBATCH --account=caps
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=2pt-pipeline
#SBATCH --partition=caps
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=aaronjo2@illinois.edu
#
# End of embedded SBATCH options
#

# for env variables and conda
source ~/.bashrc

conda activate shearXlensing

set -x

# unbuffered python output
export PYTHONUNBUFFERED=T
export OMP_NUM_THREADS=$SLURM_NTASKS_PER_NODE
hostname
echo threads: $OMP_NUM_THREADS

date
time run_nx2pt $@
date
