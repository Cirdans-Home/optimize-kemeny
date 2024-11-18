#!/bin/bash
#SBATCH --job-name=KEMENY
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --time=1-00:00:00
#SBATCH -o dense_experiment.log

module purge
module load gcc/13.2.0 matlab
module list

matlab -batch "dense_experiment"

