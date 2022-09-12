#!/bin/bash

#SBATCH -J test_job
#SBATCH -o %N.%j.o         # Name of stdout output file
#SBATCH -e %N.%j.e         # Name of stderr error file
#SBATCH -p shared                      # Queue (partition) name
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --export=ALL
#SBATCH -t 00:10:00                    # Run time (hh:mm:ss)
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=all                # Send email at begin, end, or fail of job
#SBATCH --account=TG-MCB190004

# Other commands must follow all #SBATCH directives...

# file to be executed
PROGNAME="flow_past_sphere.py"

# print some details
date
echo Job name: $SLURM_JOB_NAME
echo Execution dir: $SLURM_SUBMIT_DIR
echo Number of processes: $SLURM_NTASKS

# load anaconda and activate environment
module load anaconda3
source activate pyaxisymflow
conda env list

# set smp num threads the same as ---cpus-per-task
SMP_NUM_THREADS=4
export OMP_NUM_THREADS=$SMP_NUM_THREADS

# execute the program
~/.conda/envs/pyaxisymflow/bin/python -u ${PROGNAME} --num_threads=$SMP_NUM_THREADS
