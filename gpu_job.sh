#!/bin/bash
file_name=$1
> ${file_name}.sh 
echo "#!/bin/bash" >> ${file_name}.sh
echo "#SBATCH --gres=gpu:1" >> ${file_name}.sh
echo "#SBATCH --time=12:00:00" >> ${file_name}.sh
echo "#SBATCH --mem=24000M" >> ${file_name}.sh
echo "#SBATCH --output=${file_name}.out" >> ${file_name}.sh

echo "module load anaconda" >> ${file_name}.sh
echo "source activate precise_gps_env" >> ${file_name}.sh
echo "export TF_XLA_FLAGS=--tf_xla_enable_xla_devices" >> ${file_name}.sh
echo "srun --gres=gpu:1 python train_models.py -f \"run_files/${file_name}.json\"" >> ${file_name}.sh

sbatch ${file_name}.sh