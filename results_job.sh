#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00

module load anaconda

export TF_XLA_FLAGS=--tf_xla_enable_xla_devices
srun --gres=gpu:1 python results.py -n "Boston" -d "boston" -l 4 -s 5
srun --gres=gpu:1 python results.py -n "Redwine" -d "redwine" -l 4 -s 5
srun --gres=gpu:1 python results.py -n "Concrete" -d "concrete" -l 4 -s 5
srun --gres=gpu:1 python results.py -n "Energy" -d "energy" -l 4 -s 5
srun --gres=gpu:1 python results.py -n "Whitewine" -d "whitewine" -l 4 -s 5
srun --gres=gpu:1 python results.py -n "Protein" -d "protein" -l 4 -s 5
srun --gres=gpu:1 python results.py -n "Yacht" -d "yacht" -l 4 -s 5