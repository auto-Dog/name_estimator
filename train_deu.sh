#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -w inspur1

# python train.py --prefix vit_cn6a --size 240 --patch 10 --batchsize 64
# python train.py --prefix vit_cn6a --from_check_point "model_vit_cn6a.pth" --size 240 --patch 10 --batchsize 64
python train_enhancement.py --prefix vit_cn6a --size 240 --patch 10 --batchsize 48