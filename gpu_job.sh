#!/bin/bash
#SBATCH --gres=gpu:teslak80:1
#SBATCH --time=12:00:00

module load anaconda
module load cuda
source activate precise_gps

cd documents/precise_gps

chmod +x ./run_all.sh 
srun --gres=gpu:1 ./run_all.sh "run_files"