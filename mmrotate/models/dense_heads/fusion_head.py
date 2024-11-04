import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
import pdb
import e2cnn.nn as enn
from ..utils import (build_enn_divide_feature, build_enn_norm_layer,
                     build_enn_trivial_feature, ennAvgPool, ennConv,
                     ennMaxPool, ennReLU, ennTrivialConv)


class BasicFusionBlock(enn.EquivariantModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 style='pytorch',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')
                 ):
        super(BasicFusionBlock, self).__init__()
        assert style in ['pytorch', 'caffe']
        self.in_type = build_enn_divide_feature(in_channels)
        self.out_type = build_enn_divide_feature(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.norm1_name, norm1 = build_enn_norm_layer(
            self.out_channels, postfix=1)

        self.norm2_name, norm2 = build_enn_norm_layer(
            self.out_channels, postfix=2)

        self.norm2_name, norm2 = build_enn_norm_layer(
            self.out_channels, postfix=3)

        self.conv1 = ennConv(
            in_channels,
            self.out_channels,
            kernel_size=1,
            stride=self.stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.relu1 = ennReLU(self.out_channels)

    @property
    def norm1(self):
        """Get normalizion layer's name."""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """Get normalizion layer's name."""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """Get normalizion layer's name."""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)

        return out

    def evaluate_output_shape(self, input_shape):
        """Evaluate output shape."""
        assert input_shape == self.in_type.size
        if self.downsample is not None:
            return self.downsample.evaluate_output_shape(input_shape)
        else:
            return input_shape

class Fusion_head(nn.Module):
    def __init__(self):
        super().__init__()
