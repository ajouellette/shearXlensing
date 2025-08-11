#!/bin/bash
#
#SBATCH --account=caps
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --job-name=cosmosis-maxlike
#SBATCH --partition=caps
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# End of embedded SBATCH options
#

# for env variables and conda
source ~/.bashrc

conda activate shearXlensing

set -x

date
export OMP_NUM_THREADS=$SLURM_NTASKS
echo $OMP_NUM_THREADS

hostname
time cosmosis-campaign $@
date
