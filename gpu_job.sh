#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

module load anaconda
srun --gres=gpu:1 ./run_all.sh "run_files"