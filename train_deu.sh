#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1

python train.py --prefix vit_cn1 --size 384 --patch 16 --batchsize 64