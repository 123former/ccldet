# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule

from mmdet.models.builder import NECKS
from mmdet.models.utils import CSPLayer
import pdb
import cv2
import numpy as np
import os


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            # nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.convT = nn.Conv2d(in_planes // ratio,
                               in_planes // 2,
                               kernel_size=1,
                               bias=False)
        self.convV = nn.Conv2d(in_planes // ratio,
                               in_planes // 2,
                               kernel_size=1,
                               bias=False)
        # self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=3)

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))

        avgoutT = self.convT(avgout)
        maxoutT = self.convT(maxout)

        avgoutV = self.convV(avgout)
        maxoutV = self.convV(maxout)

        W = self.softmax(torch.cat((avgoutT + maxoutT, avgoutV + maxoutV), 3))
        wT = W[:, :, :, 0:1]
        wV = W[:, :, :, 1:]

        return wT, wV


# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.convT = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.convV = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.softmax = nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        xT = self.convT(x)
        xV = self.convV(x)
        W = self.softmax(torch.cat((xT, xV), 1))
        wT = W[:, 0:1, :, :]
        wV = W[:, 1:2, :, :]
        return wT, wV


class chx(nn.Module):
    def __init__(self, planes, flag1=True, flag2=False):
        super(chx, self).__init__()
        self.ca = ChannelAttention(planes)  # planes是feature map的通道个数
        self.sa = SpatialAttention()
        self.flag1 = flag1
        self.flag2 = flag2

    def forward(self, featT, featV):
        x = torch.cat((featT, featV), 1)
        if self.flag1:
            w_C_T, w_C_V = self.ca(x)
            featT = w_C_T * featT  # 广播机制
            featV = w_C_V * featV
        if self.flag2:
            w_S_T, w_S_V = self.sa(x)
            featT = w_S_T * featT  # 广播机制
            featV = w_S_V * featV
        # x = torch.cat((featT, featV), 1)
        x = torch.add(featT, featV)
        return x

class Attention_neck_cnn(nn.Module):
    def __init__(self,
                 in_channels=[512 * 2, 1024 * 2, 2048 * 2]
                 ):
        super().__init__()

        self.Model = nn.ModuleList()
        for i in range(len(in_channels)):
            self.Model.append(
                chx(in_channels[i])
            )


    def forward(self, x1, x2):
        # x1 rgb, x2 fre
        feats_list = []
        feats_list.append(x1[0])
        for i in range(1, len(x1)):
            feat = self.Model[i-1](x1[i], x2[i-1])
            feats_list.append(feat)
        feats = tuple(feats_list)
        return feats