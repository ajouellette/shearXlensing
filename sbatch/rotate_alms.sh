#!/bin/bash

#SBATCH -A caps
#SBATCH -p caps
#SBATCH --time=03:30:00
#SBATCH -J rotate_alm
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

source ~/.bashrc
conda activate shearXlensing
export OMP_NUM_THREADS=16
time python scripts/get_rotated_alm.py $@
