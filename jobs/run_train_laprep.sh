#!/bin/bash
#SBATCH --time=3:30:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err

module --quiet purge

module load python/3.10 libffi OpenSSL
source ~/scratch/EvaluationLaplacians/lap_env/bin/activate
cd ~/scratch/EvaluationLaplacians/wt_online_dev_sami/


EXP_NAME=$1
CONFIG_NAME=$2
BATCH_SIZE=$3
SEED=$4
ENV_NAME=$5

#full_python_command
#python wt_online_dev_sami/train_laprepr.py --use_wandb= --deactivate_training= --wandb_offline= --save_model= --obs_mode= --config_file= --save_dir= --n_samples= --batch_size= --discount= --total_train_steps= --max_episode_steps= --seed= --env_name= --lr= --hidden_dims= --barrier_initial_val= --lr_barrier_coefs=


python_command="python train_laprepr.py $EXP_NAME --use_wandb   --config_file=$CONFIG_NAME --save_dir=~/scratch/lap_logs/  --batch_size=$BATCH_SIZE   --seed=$SEED --env_name=$ENV_NAME"

echo "The folllowing python code is being run: "
echo "$python_command"

eval "$python_command"
