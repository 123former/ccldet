import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
import pdb
import e2cnn.nn as enn
from ..utils import (build_enn_divide_feature, build_enn_norm_layer,
                     build_enn_trivial_feature, ennAvgPool, ennConv,
                     ennMaxPool, ennReLU, ennTrivialConv, ennAdaptiveMaxPool,
                     ennAdaptiveAvgPool, ennSoftmax, ennConcat)

class ChannelAttention(enn.EquivariantModule):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = ennAdaptiveAvgPool(in_planes, size=1)
        self.max_pool = ennAdaptiveMaxPool(in_planes, size=1)

        self.sharedMLP = enn.SequentialModule(
            ennConv(in_planes, in_planes // ratio, 1, bias=False),
            ennReLU(in_planes // ratio),
            # nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
    )
        self.convT = ennConv(in_planes // ratio,
                               in_planes // 2,
                               kernel_size=1,
                               bias=False)
        self.convV = ennConv(in_planes // ratio,
                               in_planes // 2,
                               kernel_size=1,
                               bias=False)
        self.softmax = ennSoftmax(in_planes // 2, dim=3)

    def forward(self, x):
        #pdb.set_trace()
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))

        avgoutT = self.convT(avgout)
        maxoutT = self.convT(maxout)

        avgoutV = self.convV(avgout)
        maxoutV = self.convV(maxout)

        W = self.softmax(ennConcat([avgoutT + maxoutT, avgoutV + maxoutV], dim=3))
        wT = enn.GeometricTensor(W.tensor[:, :, :, 0:1], W.type)
        wV = enn.GeometricTensor(W.tensor[:, :, :, 1:], W.type)

        return wT, wV

    def evaluate_output_shape(self, input_shape):
        """Evaluate output shape."""
        assert input_shape == self.in_type.size
        if self.downsample is not None:
            return self.downsample.evaluate_output_shape(input_shape)
        else:
            return input_shape


# 空间注意力模块
class SpatialAttention(enn.EquivariantModule):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.convT = ennConv(2, 1, kernel_size, padding=padding, bias=False)
        self.convV = ennConv(2, 1, kernel_size, padding=padding, bias=False)

        self.softmax = ennSoftmax(2, dim=1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # TODO 带方向的tensor的求均值、求最大值
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = ennConcat([avgout, maxout], dim=1)
        xT = self.convT(x)
        xV = self.convV(x)
        W = self.softmax(ennConcat([xT, xV], dim=1))

        #TODO 带方向的tensor的赋值
        wT = W[:, 0:1, :, :]
        wV = W[:, 1:2, :, :]
        return wT, wV

    def evaluate_output_shape(self, input_shape):
        """Evaluate output shape."""
        assert input_shape == self.in_type.size
        if self.downsample is not None:
            return self.downsample.evaluate_output_shape(input_shape)
        else:
            return input_shape


class chx(enn.EquivariantModule):
    def __init__(self, planes, flag1=True, flag2=False):
        super(chx, self).__init__()
        self.ca = ChannelAttention(planes)  # planes是feature map的通道个数
        # self.sa = SpatialAttention()
        self.flag1 = flag1
        self.flag2 = flag2

    def forward(self, featT, featV):
        x = ennConcat([featT, featV], dim=1)
        if self.flag1:
            w_C_T, w_C_V = self.ca(x)
            featT = enn.GeometricTensor(w_C_T.tensor * featT.tensor, w_C_T.type)  # 广播机制
            featV = enn.GeometricTensor(w_C_V.tensor * featV.tensor, w_C_V.type)
        if self.flag2:
            w_S_T, w_S_V = self.sa(x)
            featT = w_S_T * featT  # 广播机制
            featV = w_S_V * featV
        # x = ennConcat([featT, featV], dim=1)
        x = featT+featV
        return x

    def evaluate_output_shape(self, input_shape):
        """Evaluate output shape."""
        assert input_shape == self.in_type.size
        if self.downsample is not None:
            return self.downsample.evaluate_output_shape(input_shape)
        else:
            return input_shape


class Attention_neck(nn.Module):
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
