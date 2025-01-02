#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:2
#SBATCH -w inspur1

python train.py --prefix vit_cn3c --size 512 --patch 16 --batchsize 32