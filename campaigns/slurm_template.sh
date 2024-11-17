#!/bin/bash
#
#SBATCH --account=caps
#SBATCH --time={time}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={tasks_per_node}
#SBATCH --mem-per-cpu=4000
#SBATCH --job-name={job_name}
#SBATCH --partition=caps
#SBATCH --output={log}/%j.out
#SBATCH --error={log}/%j.err
#
# End of embedded SBATCH options
#

# for env variables and conda
source ~/.bashrc

conda activate shearXlensing

set -x

date
export OMP_NUM_THREADS=1

mpirun hostname | uniq -c
mpirun {command}
date
