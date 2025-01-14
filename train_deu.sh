#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -w inspur1

python train.py --prefix vit_cn5 --size 512 --patch 16 --batchsize 32