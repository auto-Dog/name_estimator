#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -w inspur0

# python train.py --prefix vit_cn6 --size 256 --patch 8 --batchsize 64
# python train.py --prefix vit_cn5b_nt --size 512 --patch 16 --batchsize 32
python train_enhancement.py --prefix vit_cn6 --size 256 --patch 8 --batchsize 32