# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F

class criticNet(nn.Module):
    def __init__(self,input_size=256):
        super().__init__()
        self.downsample = nn.AdaptiveMaxPool2d(32)
        self.img_cnn = nn.Sequential(
            DoubleConv(3,256),
            DoubleConv(256,1024),
            # nn.AdaptiveAvgPool2d((32,32)),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.pooling = nn.MaxPool2d(2)
        self.ci_mlp = nn.Sequential(
            nn.Linear(3,256),
            nn.ReLU(),
            nn.Linear(256,512),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024+768+512,1024),
            nn.ReLU(),
            nn.Linear(1024,1)
        )

    def forward(self,input_y):
        # img, id, embedding at id
        img,ids,embedding = input_y[0],input_y[1],input_y[2]
        img = self.downsample(img)
        # extract original value
        ori_shape = ids.shape
        batch_index = torch.arange(ori_shape[0],dtype=torch.long)   # 配合第二维度索引使用
        ci = img[batch_index,:,ids//32,ids%32]
        ci = self.ci_mlp(ci.squeeze(-1).squeeze(-1))

        img = self.img_cnn(img)   # B,C,8,8
        # x[batch_index,:,ids//32,ids%32] *= 100   # B,3,1,1
        img = self.avgpool(img).squeeze(-1).squeeze(-1)
        w = embedding.reshape(-1,768)
        allf = torch.cat([img,w,ci],dim=1)
        out = self.fc(allf)
        return out

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2, do not change (H,W)"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels, affine=True),   # for WGAN-GP
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, stride=2),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
    
if __name__ == '__main__':
    y = torch.rand(32,768).cuda()
    label = torch.randint(0,1024,(32,)).cuda()
    x = torch.rand(32,3,256,256).cuda()
    model = criticNet().cuda()
    out = model([x,label,y])
    print(out.shape)