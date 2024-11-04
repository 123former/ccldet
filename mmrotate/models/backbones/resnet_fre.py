# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.builder import BACKBONES
from mmdet.models.utils import ResLayer
import pdb
from mmdet.core import multi_apply
import cv2
import numpy as np
import math
from numba import cuda
import time


class BasicBlock(BaseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(init_cfg)
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


@BACKBONES.register_module()
class ResNet_Fre(BaseModule):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stem_channels (int | None): Number of stem channels. If not specified,
            it will be the same as `base_channels`. Default: None.
        base_channels (int): Number of base channels of res layer. Default: 64.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 1, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None):
        super(ResNet_Fre, self).__init__(init_cfg)
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm2'))
                    elif block is Bottleneck:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        self.fre_mode = True

        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2 ** i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                init_cfg=block_init_cfg)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2 ** (
                len(self.stage_blocks) - 1)
        self.expansion = self.block.expansion
        ######################################################################
        # 64->256
        _in_channels = 64
        _out_channels = base_channels * self.expansion
        # pdb.set_trace()
        self.conv2 = build_conv_layer(
            conv_cfg,
            _in_channels,
            _out_channels,
            kernel_size=1,
            stride=1,
            bias=False)

        # 512->256
        _in_channels = base_channels * self.expansion * 2
        _out_channels = base_channels * self.expansion
        self.conv3 = build_conv_layer(
            conv_cfg,
            _in_channels,
            _out_channels,
            kernel_size=1,
            stride=1,
            bias=False)
        _in_channels = _out_channels
        self.conv4 = build_conv_layer(
            conv_cfg,
            _in_channels,
            _out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, _out_channels, postfix=2)
        self.add_module(self.norm2_name, norm2)
        self.relu2 = nn.ReLU(inplace=True)

        self.init()

    def make_stage_plugins(self, plugins, stage_idx):
        """Make plugins for ResNet ``stage_idx`` th stage.

        Currently we support to insert ``context_block``,
        ``empirical_attention_block``, ``nonlocal_block`` into the backbone
        like ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be:

        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type='xxx', arg1='xxx'),
            ...          stages=(False, True, True, True),
            ...          position='after_conv2'),
            ...     dict(cfg=dict(type='yyy'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='1'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='2'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3')
            ... ]
            >>> self = ResNet(depth=18)
            >>> stage_plugins = self.make_stage_plugins(plugins, 0)
            >>> assert len(stage_plugins) == 3

        Suppose ``stage_idx=0``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm2_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        if not  self.fre_mode:
            if self.deep_stem:
                x = self.stem(x)
            else:
                x = self.conv1(x)
                x = self.norm1(x)
                x = self.relu(x)
            x = self.maxpool(x)
            outs = []
            for i, layer_name in enumerate(self.res_layers):
                res_layer = getattr(self, layer_name)
                x = res_layer(x)
                if i in self.out_indices:
                    outs.append(x)
        else:
            outs = []

            x = torch.tensor(x, dtype=torch.float32)
            # pdb.set_trace()
            x = self.conv2(x)  # 64->256
            for i, layer_name in enumerate(self.res_layers):
                if i == 0:
                    continue
                res_layer = getattr(self, layer_name)
                x = res_layer(x)
                if i in self.out_indices:
                    # if i == 1:
                    #     # 512->256
                    #     x_ = self.conv3(x)
                    #     x_ = self.conv4(x_)
                    #     x_ = self.norm2(x_)
                    #     x_ = self.relu2(x_)
                    #     # pdb.set_trace()
                    #     x_ = self.up_sample(x_)
                    #     outs.append(x_)
                    outs.append(x)
            # pdb.set_trace()
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNet_Fre, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def get_block_list_mutli(self, imgs):
        bach, channel, width, height = imgs.shape
        block_tensor_list = []
        # 分解bach
        for b in range(bach):
            # dct变换并重新组织
            block_tensor_list.append(self.get_block_list_numba(imgs[b, :, :, :]))
        blocks_tensor = torch.stack(block_tensor_list)
        return blocks_tensor

    def get_block_list(self, img_f32):
        # pdb.set_trace()
        img_f32 = img_f32.detach().clone().cpu().numpy()
        # pdb.set_trace()
        channel, height, width = img_f32.shape
        img_f32 = img_f32.transpose(1, 2, 0)
        # pdb.set_trace()
        block_y = height // 8
        block_x = width // 8
        height_ = block_y * 8
        width_ = block_x * 8
        block_tensor_list = []
        count = 0
        for c in range(channel):
            img_f32_cut = img_f32[:height_, :width_, c]
            block_list = []

            new_block_list = []
            for h in range(block_y):
                for w in range(block_x):
                    # 对图像块进行dct变换
                    img_block = img_f32_cut[8 * h: 8 * (h + 1), 8 * w: 8 * (w + 1)]
                    img_dct = cv2.dct(img_block)
                    img_dct_log2 = np.log(abs(img_dct) + 1e-6)
                    block_list.append(img_dct_log2)
                    count = count + 1
            # pdb.set_trace()
            image_new = np.zeros((block_y, block_x), dtype=np.float32)
            for i in range(8):
                for j in range(8):
                    for h in range(block_y):
                        for w in range(block_x):
                            image_new[h, w] = block_list[h * block_x + w][i, j]
                    # pdb.set_trace()
                    new_block_list.append(torch.from_numpy(image_new))
                    image_new = np.zeros((block_y, block_x), dtype=np.float32)
                    # pdb.set_trace()
            block_tensor_list.append(torch.stack(new_block_list))
        # pdb.set_trace()
        # TODO 更多融合频率方法
        block_tensor = sum(block_tensor_list)
        # pdb.set_trace()
        return block_tensor.cuda()

    def get_small_imgs_list_mutli(self, imgs):
        bach, channel, width, height = imgs.shape
        block_tensor_list = []
        # 分解bach
        for b in range(bach):
            # dct变换并重新组织
            block_tensor_list.append(self.get_block_list_numba(imgs[b, :, :, :]))
        blocks_tensor = torch.stack(block_tensor_list)
        return blocks_tensor

    def get_small_img_list(self, img_f32):
        img_f32 = img_f32.detach().clone().cpu().numpy()
        # pdb.set_trace()
        channel, height, width = img_f32.shape
        img_f32 = img_f32.transpose(1, 2, 0)
        block_y = height // 8
        block_x = width // 8
        small_img = cv2.resize(img_f32, (block_x, block_y))
        small_img = small_img.transpose(2, 0, 1)
        result = torch.from_numpy(small_img)
        return result.cuda()

    # ————————————————————————numba——————————————————————————————#
    def get_block_list_numba(self, img_f32):
        img_f32 = img_f32.detach().clone().cpu().numpy()
        channel, height, width = img_f32.shape
        img_f32 = img_f32.transpose(1, 2, 0)
        block_y = height // 8
        block_x = width // 8
        img_device = cuda.to_device(img_f32[:, :, 0])
        result = cuda.device_array((64, block_y, block_x))

        alpha_device = cuda.to_device(self.alpha)
        beta_device = cuda.to_device(self.beta)
        mcos_device = cuda.to_device(self.mcos)
        threads_per_block = (8, 8)
        blocks_per_grid = (block_y, block_x)
        dct_gpu[blocks_per_grid, threads_per_block](img_device, alpha_device, beta_device, mcos_device, result)
        cuda.synchronize()
        block_list = []

        for c in range(channel):
            img_f32_ = img_f32[:, :, c]
            img_device = cuda.to_device(img_f32_)
            dct_gpu[blocks_per_grid, threads_per_block](img_device, alpha_device, beta_device, mcos_device, result)
            cuda.synchronize()
            out = result.copy_to_host()
            block_list.append(out)
        # TODO 更多融合频率方法
        block_tensor = torch.from_numpy(sum(block_list)).cuda()
        # pdb.set_trace()
        return block_tensor

    def init(self):
        dct_size = 8
        mcos = np.zeros((dct_size, dct_size), dtype=np.float32)
        alpha = np.zeros(dct_size, dtype=np.float32)
        beta = np.zeros(dct_size, dtype=np.float32)
        alpha[1:] = beta[1:] = math.sqrt(2. / dct_size)
        alpha[0] = beta[0] = math.sqrt(1. / dct_size)
        for i in range(dct_size):
            for j in range(dct_size):
                mcos[i][j] = math.cos((2 * i + 1) * math.pi * j / (2 * dct_size))
        self.alpha = alpha
        self.beta = beta
        self.mcos = mcos

    # ————————————————————————numba pytorch——————————————————————————————#
    def get_block_list_numba_(self, in_img_f32):
        in_img_f32 = in_img_f32.permute(1, 2, 0)
        img_f32 = cuda.as_cuda_array(in_img_f32)
        height, width, channel = img_f32.shape
        block_y = height // 8
        block_x = width // 8
        img_device = cuda.to_device(img_f32[:, :, 0])
        result = cuda.device_array((64, block_y, block_x))

        alpha_device = cuda.to_device(self.alpha)
        beta_device = cuda.to_device(self.beta)
        mcos_device = cuda.to_device(self.mcos)
        threads_per_block = (8, 8)
        blocks_per_grid = (block_y, block_x)
        dct_gpu[blocks_per_grid, threads_per_block](img_device, alpha_device, beta_device, mcos_device, result)
        cuda.synchronize()
        block_list = []

        for c in range(channel):
            img_f32_ = img_f32[:, :, c]
            img_device = cuda.to_device(img_f32_)
            dct_gpu[blocks_per_grid, threads_per_block](img_device, alpha_device, beta_device, mcos_device, result)
            cuda.synchronize()
            out = result.copy_to_host()
            block_list.append(out)
        # TODO 更多融合频率方法
        block_tensor = torch.from_numpy(sum(block_list)).cuda()
        # pdb.set_trace()
        return block_tensor

    def init_(self):
        dct_size = 8
        mcos = torch.zeros((dct_size, dct_size), dtype=torch.float32)
        alpha = torch.zeros(dct_size, dtype=torch.float32)
        beta = torch.zeros(dct_size, dtype=torch.float32)
        alpha[1:] = beta[1:] = math.sqrt(2. / dct_size)
        alpha[0] = beta[0] = math.sqrt(1. / dct_size)
        for i in range(dct_size):
            for j in range(dct_size):
                mcos[i][j] = math.cos((2 * i + 1) * math.pi * j / (2 * dct_size))
        self.alpha = alpha
        self.beta = beta
        self.mcos = mcos


@cuda.jit
def dct_gpu(block, alpha, beta, mcos, result):
    dimx = cuda.blockDim.x
    dimy = cuda.blockDim.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tmp = 0.

    for i in range(8):
        for j in range(8):
            x = i + bx * dimx
            y = j + by * dimy
            tmp += block[x][y] * mcos[i][tx] * mcos[j][ty]

    pos_c = tx * 8 + ty
    pos_x = bx
    pos_y = by
    result[pos_c][pos_x][pos_y] = math.log(abs(alpha[tx] * beta[ty] * tmp) + 1e-6)
