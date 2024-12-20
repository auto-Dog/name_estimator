import numpy as np
import sys
import torch
import torch.nn as nn
import torch.distributions as dist

class conditionP(nn.Module):
    '''假设条件分布是预期值附近的高斯分布，给出某一批样本的概率'''
    def __init__(self) -> None:
        super().__init__()

    def forward(self,x,x_target):
        mu = x
        # 创建一个高斯分布，均值为mu，方差为1
        gaussian = dist.Normal(mu, torch.ones_like(mu))
        
        # 计算输入x_target在该高斯分布下的概率密度
        log_probs:torch.Tensor = gaussian.log_prob(x_target)
        batch_dim = log_probs.shape[0]
        log_probs = log_probs.reshape(batch_dim,-1)
        # 返回概率密度的负对数，作为损失
        # return -log_probs.mean(dim=1)
        return -log_probs.mean()
    
if __name__ == '__main__':
    image = torch.randn((2,3,4,4))
    patch = torch.randn((2,3,4,4))
    cp_test = conditionP()
    out = cp_test(image,patch)
    print(out)