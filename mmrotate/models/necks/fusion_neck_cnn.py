import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
import pdb
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer


class BasicFusionBlock(BaseModule):
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.norm1_name, norm1 = build_norm_layer(norm_cfg,
            self.out_channels, postfix=1)

        self.conv1 = ConvModule(
            in_channels,
            self.out_channels,
            kernel_size=1,
            stride=self.stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.relu1 = nn.ReLU(inplace=True)

    @property
    def norm1(self):
        """Get normalizion layer's name."""
        return getattr(self, self.norm1_name)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)

        return out


class FusionModel_cnn(nn.Module):
    def __init__(self,
                 in_channels=[256 * 2, 512 * 2, 1024 * 2, 2048 * 2],
                 out_channels=[256, 512, 1024, 2048],
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='Swish')
                 ):
        super().__init__()

        self.Model = nn.ModuleList()
        for i in range(len(in_channels)):
            self.Model.append(
                BasicFusionBlock(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    stride=stride,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg
                )
            )


    def forward(self, x1, x2):
        feats_list = []
        for i in range(len(x1)):
            feat_list = []
            # pdb.set_trace()
            feat_list.append(x1[i])
            feat_list.append(x2[i])
            # pdb.set_trace()
            feat = torch.cat(feat_list, 1)
            # pdb.set_trace()
            feat = self.Model[i](feat)
            feats_list.append(feat)
        feats = tuple(feats_list)
        return feats
