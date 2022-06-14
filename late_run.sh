#!/bin/bash

#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
# training
python train.py --config_file configs/latefusion.yaml

# evaluation
python evaluate.py --experiment_path experiments/best_late

python eval_prediction.py --prediction experiments/best_late/prediction.csv --label /dssg/home/acct-stu/stu464/data/audio_visual_scenes/evaluation_setup/fold1_evaluate.csv


