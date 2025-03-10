#!/bin/bash

#SBATCH -A caps
#SBATCH -p caps
#SBATCH --time=04:00:00
#SBATCH -J mock_cats
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=250G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

source ~/.bashrc
conda activate shearXlensing

date
export PYTHONUNBUFFERED=T
export OMP_NUM_THREADS=$SLURM_NTASKS
echo $OMP_NUM_THREADS threads
python scripts/make_mock_cats.py $@
date
