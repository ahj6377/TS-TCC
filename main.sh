#!/bin/bash

#SBATCH --job-name --gcd4da-cop-pahse0-tsm_hmdb2ucf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=3G
#SBATCH --partition batch_ce_ugrad
#SBATCH 
#SBATCH -x sw1
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out


python main.py --seed 23 --training_mode self_supervised --selected_dataset sleepEDF
python main.py --seed 23 --training_mode fine_tune --selected_dataset sleepEDF

python main.py --seed 24 --training_mode self_supervised --selected_dataset sleepEDF
python main.py --seed 24 --training_mode fine_tune --selected_dataset sleepEDF
exit 0