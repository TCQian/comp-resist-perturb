#!/bin/bash

#SBATCH --job-name=BNI                   # Job name
#SBATCH --time=48:00:00                  # Time limit hrs:min:sec
#SBATCH --gres=gpu:nv:1
#SBATCH --mail-type=ALL                  # Get email for all status updates
#SBATCH --mail-user=e0407638@u.nus.edu   # Email for notifications
#SBATCH --mem=16G                        # Request 16GB of memory

source ~/.bashrc
conda activate bni

srun python sd_train.py