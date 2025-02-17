#!/usr/bin/env
# -*- coding: utf-8 -*-
from __future__ import absolute_import
import torch
import os, sys
sys.path.append('..')
import time
import numpy as np
from functools import reduce

def count_params(model):
    if issubclass(model.__class__, torch.nn.Module):
        return sum(reduce(lambda x,y: x*y, p.size()) for p in model.parameters() if p.requires_grad)
    else:
        return reduce(lambda x,y: x*y, model.size())


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)


class LossTracker:
    def __init__(self, loss_names):
        if "total_loss" not in loss_names:
            loss_names.append("total_loss")
        self.losses = {key: 0 for key in loss_names}
        self.loss_weights = {key: 1 for key in loss_names}

    def weight_the_losses(self, exclude_loss=("total_loss")):
        for k, _ in self.losses.items():
            if k not in exclude_loss:
                self.losses[k] *= self.loss_weights[k]

    def get_total_loss(self, exclude_loss=("total_loss")):
        self.losses["total_loss"] = 0
        for k, v in self.losses.items():
            if k not in exclude_loss:
                self.losses["total_loss"] += v

    def set_loss_weights(self, loss_weight_dict):
        for k, _ in self.losses.items():
            if k in loss_weight_dict:
                w = loss_weight_dict[k]
            else:
                w = 1.0
            self.loss_weights[k] = w

    def update(self, loss_weight_dict):
        self.set_loss_weights(loss_weight_dict)
        self.weight_the_losses()
        self.get_total_loss()

def patch_split(bchw_image:torch.Tensor,patch_size=(4,4)):
    '''Given BCHW Tensor Image(B=1), split it into patches and stack them in the batch dimension'''
    assert (len(bchw_image.shape)==4 and bchw_image.shape[0]==1),'When trying to split into patches, found that batch size is not 1'
    ori_shape = bchw_image.shape
    patches = torch.nn.functional.unfold(bchw_image,kernel_size=patch_size,stride=patch_size)   # out (B,CpHpW,nP) tensor
    # print(patches.shape)    # debug
    patches = patches.squeeze(0)
    patches = patches.permute(1,0).contiguous() # transpose dims, out (nP,CpHpW)
    nP = patches.shape[0]
    patches = patches.view(nP,ori_shape[1],patch_size[0],patch_size[1]).contiguous()
    return patches

def patch_compose(bchw_patches:torch.Tensor,output_size=(32,32)):
    '''Given BCHW Tensor Image Batches, compose them to a 1CHW image'''
    # patchlize
    ori_shape = bchw_patches.shape
    patches = bchw_patches.view(ori_shape[0],-1).permute(1,0).contiguous()   
    patches = patches.unsqueeze(0)  # out (B,CpHpW,nP) tensor
    # print(patches.shape)    # debug
    reconstructed_image = torch.nn.functional.fold(patches,output_size=output_size,kernel_size=(ori_shape[2],ori_shape[3]),stride=(ori_shape[2],ori_shape[3]))

    return reconstructed_image
