# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F

class criticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = DoubleConv(768+64,512)
        self.layer2 = DoubleConv(512,256)
        self.layer3 = DoubleConv(256,256)
        self.conditionneck = nn.Sequential(
            nn.AdaptiveAvgPool2d((64,64)),
            nn.Conv2d(3,64,(3,3),2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.pooling = nn.MaxPool2d(2)
        self.fc = nn.Linear(256,1)

    def forward(self,y,x):
        x = self.conditionneck(x)
        yx = torch.cat([y,x],dim=1)
        yx = self.layer1(yx)
        yx = self.pooling(yx)
        yx = self.layer2(yx)
        yx = self.pooling(yx)
        yx = self.layer3(yx)
        yx = self.avgpool(yx)
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