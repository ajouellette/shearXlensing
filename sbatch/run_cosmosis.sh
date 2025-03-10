#!/bin/bash
#
#SBATCH --account=caps
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --job-name=cosmosis
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
#source cosmosis-configure --no-omp

#time mpirun -n $SLURM_NTASKS_PER_NODE cosmosis --mpi $@
#time srun cosmosis-campaign --mpi $@

set -x

date
export OMP_NUM_THREADS=1

mpirun hostname | uniq -c
time mpirun cosmosis-campaign --mpi $@
date
