#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00

module load anaconda
source activate precise_gps_env

export TF_XLA_FLAGS=--tf_xla_enable_xla_devices
results=$(ls results/raw)
for result in $results
do
    srun --gres=gpu:1 python results.py -n "$result" -d "$result" -l 1 -s 5
done