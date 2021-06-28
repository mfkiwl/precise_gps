#!/bin/bash
#SBATCH --gres=gpu:teslap100:1
#SBATCH --time=24:00:00

module load anaconda
module load cuda
source activate precise_gps_env

chmod +x ./run_all.sh 
srun --gres=gpu:1 ./run_all.sh "run_files"