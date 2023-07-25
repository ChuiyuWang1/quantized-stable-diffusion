#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --output=ldm-%j

module purge
module load pytorch/1.12.1
source activate ldm
cd latent-diffusion
# CUDA_VISIBLE_DEVICES=0 python main.py -b models/ldm/cin256/config.yaml -t --gpus 0,
# CUDA_VISIBLE_DEVICES=0 python scripts/sample_diffusion_classcond.py -r models/ldm/cin256-v2/model.ckpt -n 5000 -l samples/cin256-v2_orig -c 250 --batch_size 10
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation.py
# CUDA_VISIBLE_DEVICES=0 python scripts/sample_diffusion.py -r models/ldm/celeba256/model.ckpt -l samples/celeba256_orig -c 100 -n 2030