#!/bin/bash

#SBATCH -A caps
#SBATCH -p caps
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=09:00:00
#SBATCH --mem=300G
#SBATCH -J quaia-split
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

date
source ~/.bashrc
conda activate shearXlensing
export OMP_NUM_THREADS=40
time python scripts/split_quaia.py $@
