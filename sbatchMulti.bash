#!/bin/bash
#SBATCH -A m4189
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH -n 1
#SBATCH -c 16

export OMP_NUM_THREADS=8 
export SLURM_CPU_BIND="cores"
srun ./ColliderMultiCore.x /global/u2/l/lpkolanz/SpaceLab/jobs/singleCoreComparison1/ 50 2>sim_err.log 1>sim_out.log