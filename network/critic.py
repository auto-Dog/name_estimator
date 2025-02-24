# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F

class criticNet(nn.Module):
    def __init__(self,input_size=256):
        super().__init__()
        self.conditionneck = nn.Sequential(
            DoubleConv(3,64),
            nn.MaxPool2d(2),
            DoubleConv(64,512),
            nn.AdaptiveAvgPool2d((32,32)),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.pooling = nn.MaxPool2d(2)
        self.fc = nn.Sequential(
            nn.Linear(512+768,1024),
            nn.ReLU(),
            nn.Linear(1024,1)
        )

    def forward(self,input_y):
        img,ids,embedding = input_y[0],input_y[1],input_y[2]
        # img, id, embedding at id
        x = self.conditionneck(img)   # B,C,32,32
        # extract original value
        ori_shape = ids.shape
        batch_index = torch.arange(ori_shape[0],dtype=torch.long)   # 配合第二维度索引使用
        x[batch_index,:,ids//32,ids%32] *= 100   # B,3,1,1
        x = self.avgpool(x).squeeze(-1).squeeze(-1)
        y = embedding.reshape(-1,768)
        yx = torch.cat([x,y],dim=1)
        out = self.fc(yx)
        return out

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2, do not change (H,W)"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
if __name__ == '__main__':
    y = torch.rand(2,768)
    label = torch.randint(0,1024,(2,))
    x = torch.rand(2,3,256,256)
    model = criticNet()
    out = model([x,label,y])
    print(out.shape)