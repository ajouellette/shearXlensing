#!/bin/bash
#
#SBATCH --account=caps
#SBATCH --time=2:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=2pt-pipeline
#SBATCH --partition=caps
##SBATCH --partition=caps-main
##SBATCH --partition=caps-himem
#SBATCH --output=pipeline.out
#SBATCH --error=pipeline.err
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=aaronjo2@illinois.edu
#
# End of embedded SBATCH options
#

# for env variables and conda
source ~/.bashrc

conda activate shearXlensing

# unbuffered python output
export PYTHONUNBUFFERED=T
export OMP_NUM_THREADS=$SLURM_NTASKS_PER_NODE
hostname
echo threads: $OMP_NUM_THREADS

time run_nx2pt $@
