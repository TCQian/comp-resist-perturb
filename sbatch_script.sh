#!/bin/bash

#SBATCH --job-name=BNI                   # Job name
#SBATCH --time=48:00:00                  # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100-40:1
#SBATCH --mail-type=ALL                  # Get email for all status updates
#SBATCH --mail-user=e0407638@u.nus.edu   # Email for notifications
#SBATCH --mem=16G                        # Request 16GB of memory

source ~/.bashrc
conda activate bni

DATA_DIR=$(pwd)/data
MODEL_DIR=$(pwd)/data
OUTPUT_DIR=$(pwd)/output/sd_train_output
VISUAL_OUTPUT_DIR=$(pwd)/visual_output/sd_train_output
srun python sd_train.py --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --model_dir $MODEL_DIR --visual_output_dir $VISUAL_OUTPUT_DIR

DATA_DIR=$(pwd)/data
MODEL_DIR=$(pwd)/data
OUTPUT_DIR=$(pwd)/output/sd_train_perturb_output
VISUAL_OUTPUT_DIR=$(pwd)/visual_output/sd_train_perturb_output
srun python sd_train_perturb.py --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --model_dir $MODEL_DIR --visual_output_dir $VISUAL_OUTPUT_DIR

DATA_DIR=$(pwd)/data
MODEL_DIR=$(pwd)/data
OUTPUT_DIR=$(pwd)/output/sd_train_adaptive_perturb_output
VISUAL_OUTPUT_DIR=$(pwd)/visual_output/sd_train_adaptive_perturb_output
srun python sd_train_adaptive_perturb.py --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --model_dir $MODEL_DIR --visual_output_dir $VISUAL_OUTPUT_DIR