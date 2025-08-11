#!/bin/bash

#SBATCH -A caps
##SBATCH -A aaronjo2-ic
#SBATCH -p caps
##SBATCH -p IllinoisComputes
#SBATCH --time=04:00:00
#SBATCH -J sim_alms
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

source ~/.bashrc
conda activate shearXlensing

date
export OMP_NUM_THREADS=$SLURM_NTASKS
echo $OMP_NUM_THREADS threads
python scripts/simulated_alms.py $@
date
