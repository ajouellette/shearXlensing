#!/bin/bash
#
#SBATCH --account=caps
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=cosmosis
#SBATCH --partition=caps
#SBATCH --output=job.out
#SBATCH --error=job.err
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=aaronjo2@illinois.edu
#
# End of embedded SBATCH options
#

# for env variables and conda
source ~/.bashrc

conda activate shearXlensing
#source cosmosis-configure --no-omp

time mpirun -n $SLURM_NTASKS_PER_NODE cosmosis --mpi $@
