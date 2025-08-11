#!/bin/bash
#
#SBATCH --account=caps
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --hint=compute_bound
#SBATCH --job-name=shear-2pt
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

export OMP_NUM_THREADS=$SLURM_NTASKS_PER_NODE
hostname
echo threads: $OMP_NUM_THREADS

date
time python scripts/calc_shear_realspace.py $@
date
