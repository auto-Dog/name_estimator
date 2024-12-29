#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1

export CUDA_VISIBLE_DEVICES=4
python train.py --prefix vit_cn4a --size 512 --patch 16 --batchsize 32