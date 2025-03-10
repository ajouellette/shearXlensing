#!/bin/bash
#
#SBATCH --account=caps
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
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

#time mpirun -n $SLURM_NTASKS_PER_NODE cosmosis-campaign --mpi $@

export OMP_NUM_THREADS=$SLURM_NTASKS_PER_NODE
echo $OMP_NUM_THREADS

cosmosis-campaign $1 -l
for run in $(cosmosis-campaign $1 -l); do
    if [[ $run == *"fisher"* ]]; then
        echo running $run
        if [[ $run == *"spt"* ]]; then
            echo skipping $run
            continue
        fi
        time cosmosis-campaign $1 -r $run
    fi
done
