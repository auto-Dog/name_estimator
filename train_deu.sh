#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:2

python train.py --prefix vit_cn3 --size 512 --patch 16 --batchsize 32