# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import mmcv
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_linear_layer
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import ConvFCBBoxHead
from ...builder import ROTATED_HEADS
from .rotated_bbox_head import RotatedBBoxHead
from mmrotate.core.bbox.iou_calculators.rotate_iou2d_calculator import rbbox_overlaps
import pdb
from pycocotools import mask as maskUtils
import numpy as np
import cv2
from ...builder import ROTATED_HEADS, build_loss
from torchvision.transforms.functional import normalize
import numba
from numba import cuda
import time
import warnings

warnings.filterwarnings("ignore")

@ROTATED_HEADS.register_module()
class RotatedConvFCBBoxHead(RotatedBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg

    Args:
        num_shared_convs (int, optional): number of ``shared_convs``.
        num_shared_fcs (int, optional): number of ``shared_fcs``.
        num_cls_convs (int, optional): number of ``cls_convs``.
        num_cls_fcs (int, optional): number of ``cls_fcs``.
        num_reg_convs (int, optional): number of ``reg_convs``.
        num_reg_fcs (int, optional): number of ``reg_fcs``.
        conv_out_channels (int, optional): output channels of convolution.
        fc_out_channels (int, optional): output channels of fc.
        conv_cfg (dict, optional): Config of convolution.
        norm_cfg (dict, optional): Config of normalization.
        init_cfg (dict, optional): Config of initialization.
    """

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 reg_channel=5,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(RotatedConvFCBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = (reg_channel if self.reg_class_agnostic else reg_channel *
                                                                       self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        if init_cfg is None:
            self.init_cfg += [
                dict(
                    type='Xavier',
                    layer='Linear',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        """Forward function."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred


@ROTATED_HEADS.register_module()
class RotatedShared2FCBBoxHead(RotatedConvFCBBoxHead):
    """Shared2FC RBBox head."""

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(RotatedShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@ROTATED_HEADS.register_module()
class RotatedKFIoUShared2FCBBoxHead(RotatedConvFCBBoxHead):
    """KFIoU RoI head."""

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(RotatedKFIoUShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """Loss function."""
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                bbox_pred_decode = self.bbox_coder.decode(
                    rois[:, 1:], bbox_pred)
                bbox_targets_decode = self.bbox_coder.decode(
                    rois[:, 1:], bbox_targets)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                    pos_bbox_pred_decode = bbox_pred_decode.view(
                        bbox_pred_decode.size(0), 5)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                    pos_bbox_pred_decode = bbox_pred_decode.view(
                        bbox_pred_decode.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    pred_decode=pos_bbox_pred_decode,
                    targets_decode=bbox_targets_decode[pos_inds.type(
                        torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses


@ROTATED_HEADS.register_module()
class RotatedSharedVFIOU2FCBBoxHead(RotatedConvFCBBoxHead):
    """Shared2FC RBBox head."""

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(RotatedSharedVFIOU2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """Loss function."""
        losses = dict()

        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # pdb.set_trace()
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)

            bbox_pred_decode = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
            bbox_targets_decode = self.bbox_coder.decode(
                rois[:, 1:], bbox_targets)

            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred_decode.view(
                    bbox_pred_decode.size(0), 5)[pos_inds.type(torch.bool)]
            else:
                pos_bbox_pred = bbox_pred_decode.view(
                    bbox_pred_decode.size(0), -1,
                    5)[pos_inds.type(torch.bool),
                       labels[pos_inds.type(torch.bool)]]

            iou_targets_rf = rbbox_overlaps(pos_bbox_pred,
                                            bbox_targets_decode[pos_inds.type(torch.bool)].detach(),
                                            is_aligned=True).clamp(min=1e-6)

            pos_ious = iou_targets_rf.clone().detach()
            cls_iou_targets = torch.zeros_like(cls_score)
            cls_iou_targets[pos_inds.type(torch.bool), labels[pos_inds.type(torch.bool)]] = pos_ious
            for i in range(labels.shape[0]):
                if labels[i] == self.num_classes:
                    cls_iou_targets[i, self.num_classes] = 1

            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score[pos_inds.type(torch.bool), :],
                    cls_iou_targets[pos_inds.type(torch.bool), :],
                    avg_factor=avg_factor)  # sum(pos_inds)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        return losses


@ROTATED_HEADS.register_module()
class RotatedSharedVis2FCBBoxHead(RotatedConvFCBBoxHead):
    """Shared2FC RBBox head."""

    def __init__(self, thre_list=[50, 100, 150, 200], fc_out_channels=1024, multi=False, *args, **kwargs):
        super(RotatedSharedVis2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        loss_vis = dict(
            type='SmoothL1Loss', beta=0.25, loss_weight=1.0)
        self.image = None
        self.loss_vis = build_loss(loss_vis)
        self.vis_pred = True
        self.multi = multi
        self.thre_list = thre_list

    def rgb_to_hsv(self, img):
        eps = 1e-8
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

        hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + eps))[
            img[:, 2] == img.max(1)[0]]
        hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + eps))[
            img[:, 1] == img.max(1)[0]]
        hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + eps))[
            img[:, 0] == img.max(1)[0]]) % 6

        hue[img.min(1)[0] == img.max(1)[0]] = 0.0
        hue = hue / 6

        saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + eps)
        saturation[img.max(1)[0] == 0] = 0

        value = img.max(1)[0]

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        hsv = torch.cat([hue, saturation, value], dim=1)
        return hsv

    def get_info(self, img, img_metas):
        self.img = img.clone().detach()
        # pdb.set_trace()
        self.mean = img_metas[0]['img_norm_cfg']['mean']
        self.std = img_metas[0]['img_norm_cfg']['std']

    def obb2ploy(self, rboxes):

        x = rboxes[:, 0]
        y = rboxes[:, 1]
        w = rboxes[:, 2]
        h = rboxes[:, 3]
        a = rboxes[:, 4]
        cosa = torch.cos(a)
        sina = torch.sin(a)
        wx, wy = w / 2 * cosa, w / 2 * sina
        hx, hy = -h / 2 * sina, h / 2 * cosa
        p1x, p1y = x - wx - hx, y - wy - hy
        p2x, p2y = x + wx - hx, y + wy - hy
        p3x, p3y = x + wx + hx, y + wy + hy
        p4x, p4y = x - wx + hx, y - wy + hy
        return torch.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y], dim=-1)

    def init_vis(self, pos_inds, vis_perd, vis_gt, bach, thre=35, weight=1.0):
        MEAN = [-mean / std for mean, std in zip(self.mean, self.std)]
        STD = [1 / std for std in self.std]
        image = normalize(self.img, MEAN, STD)
        img_hsv = self.rgb_to_hsv(image)
        v_mask = img_hsv[:, 2, :, :] > thre
        v_mask_new = torch.tensor(v_mask, dtype=torch.float32)
        pos_inds_ = torch.tensor(pos_inds, dtype=torch.float32)
        vis_perd_ = vis_perd.clone().detach()
        vis_perd_ploy = self.obb2ploy(vis_perd_)
        vis_gt_ploy = self.obb2ploy(vis_gt)
        # 将data数据拷贝到GPU
        img_device = cuda.as_cuda_array(v_mask_new)

        rbbox_device = cuda.as_cuda_array(vis_perd_ploy)
        gt_rbbox_device = cuda.as_cuda_array(vis_gt_ploy)
        ind_device = cuda.as_cuda_array(pos_inds_)

        out_device = cuda.device_array([bach, vis_perd.shape[0]])
        cuda_cal[vis_perd.shape[0], bach](out_device, img_device, rbbox_device, ind_device)
        up_part_vi = out_device.copy_to_host()

        gt_out_device = cuda.device_array([bach, vis_perd.shape[0]])
        cuda_cal[vis_perd.shape[0], bach](gt_out_device, img_device, gt_rbbox_device, ind_device)
        gt_up_part_vi = gt_out_device.copy_to_host()

        up_part_vi = torch.from_numpy(up_part_vi[0, :]).cuda()
        gt_up_part_vi = torch.from_numpy(gt_up_part_vi[0, :]).cuda()

        bottom_part = torch.ceil(vis_perd[:, 2]) * torch.ceil(vis_perd[:, 3])
        rates_vi = torch.div(up_part_vi, bottom_part)

        gt_bottom_part = torch.ceil(vis_gt[:, 2]) * torch.ceil(vis_gt[:, 3])
        gt_rates_vi = torch.div(gt_up_part_vi, gt_bottom_part)

        rates_vi = rates_vi.reshape((rates_vi.shape[0], 1))
        gt_rates_vi = gt_rates_vi.reshape((gt_rates_vi.shape[0], 1))

        vis_weights = torch.ones_like(gt_rates_vi) * weight
        return rates_vi, gt_rates_vi, vis_weights

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """Loss function.

        Args:
            cls_score (torch.Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 5).
            rois (torch.Tensor): Boxes to be transformed. Has shape
                (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            labels (torch.Tensor): Shape (n*bs, ).
            label_weights(torch.Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
            bbox_targets(torch.Tensor):Regression target for all
                  proposals, has shape (num_proposals, 5), the
                  last dimension 5 represents [cx, cy, w, h, a].
            bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 5) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 5).
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.
        """
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            # pdb.set_trace()
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # pdb.set_trace()
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        if self.vis_pred:
            pred_bboxes = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
            gt_bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_targets)
            bach = self.img.shape[0]
            vis_perd = pred_bboxes[pos_inds.type(torch.bool)]
            vis_gt = gt_bboxes[pos_inds.type(torch.bool)]
            if self.multi:
                thre_list = self.thre_list
                weight_list = [1.0, 1.0, 1.0, 1.0]
                rates_vi_list = []
                gt_rates_vi_list = []
                vis_weights_list = []
                for i in range(len(thre_list)):
                    thre = thre_list[i]
                    weight = weight_list[i]
                    rates_vi, gt_rates_vi, vis_weights = self.init_vis(pos_inds, vis_perd, vis_gt, bach, thre, weight)
                    rates_vi_list.append(rates_vi)
                    gt_rates_vi_list.append(gt_rates_vi)
                    vis_weights_list.append(vis_weights)
                rates_vi = torch.cat(rates_vi_list)
                gt_rates_vi = torch.cat(gt_rates_vi_list)
                vis_weights = torch.cat(vis_weights_list)
            else:
                rates_vi, gt_rates_vi, vis_weights = self.init_vis(pos_inds, vis_perd, vis_gt, bach)
            losses['loss_vis'] = self.loss_vis(
                rates_vi,
                gt_rates_vi,
                vis_weights,
                avg_factor=gt_rates_vi.size(0),
                reduction_override=reduction_override)
            # pdb.set_trace()

        return losses


@ROTATED_HEADS.register_module()
class RotatedSharedVis2FCHBBoxHead(ConvFCBBoxHead):
    """Shared2FC RBBox head."""

    def __init__(self, fc_out_channels=1024, thre_list=[50, 100, 150, 200], multi=False, *args, **kwargs):
        super(RotatedSharedVis2FCHBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        loss_vis = dict(
            type='SmoothL1Loss', beta=0.25, loss_weight=1.0)
        self.image = None
        self.loss_vis = build_loss(loss_vis)
        self.vis_pred = False
        self.multi = multi
        self.thre_list = thre_list

    # TODO需要写一个img、mean、std输入的接口
    # def get_image_v(self, im):
    #     img = im.cpu().numpy()
    #     img = np.array(img).transpose(1, 2, 0)
    #     # pdb.set_trace()
    #     image = mmcv.imdenormalize(img, self.mean, self.std, to_bgr=True)
    #     img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #     return (img_hsv[:, :, 2] > 35)

    def get_info(self, img, img_metas):
        self.img = img.clone().detach()
        self.mean = img_metas[0]['img_norm_cfg']['mean']
        self.std = img_metas[0]['img_norm_cfg']['std']

    def rgb_to_hsv(self, img):
        eps = 1e-8
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

        hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + eps))[
            img[:, 2] == img.max(1)[0]]
        hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + eps))[
            img[:, 1] == img.max(1)[0]]
        hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + eps))[
            img[:, 0] == img.max(1)[0]]) % 6

        hue[img.min(1)[0] == img.max(1)[0]] = 0.0
        hue = hue / 6

        saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + eps)
        saturation[img.max(1)[0] == 0] = 0

        value = img.max(1)[0]

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        hsv = torch.cat([hue, saturation, value], dim=1)
        return hsv

    def get_image_v(self, im, thre=35):
        MEAN = [-mean / std for mean, std in zip(self.mean, self.std)]
        STD = [1 / std for std in self.std]
        image = normalize(im, MEAN, STD).unsqueeze(0)
        img_hsv = self.rgb_to_hsv(image)
        return (img_hsv[0, 2, :, :] > thre)

    def polygons_to_mask(self, img_shape, polygons):
        # pdb.set_trace()
        mask = np.zeros(img_shape, dtype=np.uint8)
        polygons = np.asarray(polygons, np.int32)  # 这里必须是int32，其他类型使用fillPoly会报错
        shape = polygons.shape
        polygons = polygons.reshape(shape[0], -1, 2)
        # pdb.set_trace()
        cv2.fillPoly(mask, polygons, color=1)  # 非int32 会报错
        return mask

    def init_rgb_weight(self, box_mask, v_mask):
        box_mask = torch.tensor(box_mask).cuda()
        mask_vi = torch.stack((box_mask.bool(), v_mask.bool()), -1)
        vi_sum = torch.sum(torch.all(mask_vi, dim=-1))
        weight_vi = vi_sum
        return weight_vi

    # TODO需要写一个box转rate的函数
    # 包括两个功能：
    # 1. box2mask
    # 2. 计算box面积
    def get_vis_rate(self, img, rboxes, thre=35):
        """
        :param box:
        :return:
        """
        poly = self.hbb2ploy(rboxes).clone().detach().cpu().numpy()
        img_v = self.get_image_v(img, thre)
        # pdb.set_trace()
        up_part = []
        for i in range(poly.shape[0]):
            # pdb.set_trace()
            mask = self.polygons_to_mask(img_v.shape, [poly[i, :]])
            up_part.append(self.init_rgb_weight(mask, img_v))
        # pdb.set_trace()
        up_part = torch.tensor(up_part, dtype=torch.int32).cuda()
        bottom_part = torch.ceil(rboxes[:, 2]) * torch.ceil(rboxes[:, 3])
        rate_vi = torch.div(up_part, bottom_part)
        # pdb.set_trace()
        return rate_vi

    def get_info(self, img, img_metas):
        self.img = img.clone().detach()
        # pdb.set_trace()
        self.mean = img_metas[0]['img_norm_cfg']['mean']
        self.std = img_metas[0]['img_norm_cfg']['std']

    def hbb2ploy(self, boxes):

        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        x1 = x
        y1 = y
        x2 = x1 + w
        y2 = y1 + h
        return torch.stack([x1, y1, x1, y2, x2, y2, x2, y1], dim=-1)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """Loss function.

        Args:
            cls_score (torch.Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 5).
            rois (torch.Tensor): Boxes to be transformed. Has shape
                (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            labels (torch.Tensor): Shape (n*bs, ).
            label_weights(torch.Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
            bbox_targets(torch.Tensor):Regression target for all
                  proposals, has shape (num_proposals, 5), the
                  last dimension 5 represents [cx, cy, w, h, a].
            bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 5) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 5).
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.
        """
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            # pdb.set_trace()
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # pdb.set_trace()
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        if self.vis_pred:
            pred_bboxes = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
            gt_bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_targets)

            bach = self.img.shape[0]
            vis_perd = pred_bboxes[pos_inds.type(torch.bool)]
            vis_gt = gt_bboxes[pos_inds.type(torch.bool)]
            location = 0

            if self.multi:
                thre_list = self.thre_list
                weight_list = [1.0, 1.0, 1.0, 1.0]

                rate_vi_list = []
                gt_rate_vi_list = []
                for i in range(len(thre_list)):
                    thre = thre_list[i]
                    for i in range(bach):
                        pre_location = location
                        location = sum(pos_inds[i * 512:i * 512 + 512]) + pre_location

                        rate_vi = self.get_vis_rate(self.img[i, :, :, :], vis_perd[pre_location:location, :], thre)
                        gt_rate_vi = self.get_vis_rate(self.img[i, :, :, :], vis_gt[pre_location:location, :], thre)
                        rate_vi_list.append(rate_vi)
                        gt_rate_vi_list.append(gt_rate_vi)
                    rates_vi = torch.cat(rate_vi_list)
                    gt_rates_vi = torch.cat(gt_rate_vi_list)
                    rates_vi = rates_vi.reshape((rates_vi.shape[0], 1))
                    gt_rates_vi = gt_rates_vi.reshape((gt_rates_vi.shape[0], 1))
                    vis_weights = torch.ones_like(gt_rates_vi)

            else:
                rate_vi_list = []
                gt_rate_vi_list = []
                for i in range(bach):
                    pre_location = location
                    location = sum(pos_inds[i * 512:i * 512 + 512]) + pre_location

                    rate_vi = self.get_vis_rate(self.img[i, :, :, :], vis_perd[pre_location:location, :])
                    gt_rate_vi = self.get_vis_rate(self.img[i, :, :, :], vis_gt[pre_location:location, :])
                    rate_vi_list.append(rate_vi)
                    gt_rate_vi_list.append(gt_rate_vi)
                rates_vi = torch.cat(rate_vi_list)
                gt_rates_vi = torch.cat(gt_rate_vi_list)
                rates_vi = rates_vi.reshape((rates_vi.shape[0], 1))
                gt_rates_vi = gt_rates_vi.reshape((gt_rates_vi.shape[0], 1))
                vis_weights = torch.ones_like(gt_rates_vi)
            # pdb.set_trace()

            losses['loss_vis'] = self.loss_vis(
                rates_vi,
                gt_rates_vi,
                vis_weights,
                avg_factor=gt_rates_vi.size(0),
                reduction_override=reduction_override)
            # pdb.set_trace()

        return losses


@cuda.jit
def cuda_cal(out_device, img_device, rbbox_device, ind_device):
    # j定义为这个thread在所在的block块中的位置（0 <= j <= 761）
    j = cuda.threadIdx.x
    # i定义为这个block块在gird中的位置（0 <= i <= 1199）
    i = cuda.blockIdx.x

    pre_location = 0

    ind_sum = 0
    cur_index = 0
    for k in range(ind_device.shape[0]):
        ind_sum = ind_sum + ind_device[k]
        if (ind_sum == i):
            cur_index = k
            break
    bach_num = int(cur_index / 512)

    img = img_device[bach_num]
    box = rbbox_device[i, :]
    px1, px2 = min(rbbox_device[i, 0], rbbox_device[i, 2], rbbox_device[i, 4], rbbox_device[i, 6]), max(
        rbbox_device[i, 0], rbbox_device[i, 2], rbbox_device[i, 4], rbbox_device[i, 6])
    py1, py2 = min(rbbox_device[i, 1], rbbox_device[i, 3], rbbox_device[i, 5], rbbox_device[i, 7]), max(
        rbbox_device[i, 1], rbbox_device[i, 3], rbbox_device[i, 5], rbbox_device[i, 7])

    sum_img = 0
    for n in range(px1, px2 + 1):
        for m in range(py1, py2 + 1):
            sum_img = sum_img + img[n][m]
    out_device[j, i] = sum_img / 2
