#!/bin/bash

#SBATCH -A caps
#SBATCH -p caps
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=05:00:00
#SBATCH --mem=380G
#SBATCH -J quaia-selection
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

date
source ~/.bashrc
conda activate shearXlensing
export OMP_NUM_THREADS=16
time python scripts/quaia_selection_function.py $@
