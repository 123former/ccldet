# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector
import cv2
from mmrotate.models.necks.fusion_neck import FusionModel
from mmrotate.models.necks.fusion_neck_cnn import FusionModel_cnn
from mmrotate.models.necks.attention_neck import Attention_neck
from mmrotate.models.necks.attention_neck_cnn import Attention_neck_cnn
from mmdet.core import multi_apply
from mmrotate.core.bbox.iou_calculators.rotate_iou2d_calculator import RBboxOverlaps2D
import pdb
import numpy
from mmrotate.core import multiclass_nms_rotated, rbbox2result, multiclass_nms_rotated_new
import cv2
import math
import time

@ROTATED_DETECTORS.register_module()
class RotatedTwoStageDetector_FusionT(RotatedBaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RotatedTwoStageDetector_FusionT, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.fre_mode = False
        if isinstance(backbone, list):
            self.fre_mode = True
            self.backbone_fre = build_backbone(backbone[0])
            self.backbone = build_backbone(backbone[1])
            if hasattr(self.backbone, 'in_type'):
                self.fre_fusion_model = Attention_neck(in_channels=[512 * 2, 1024 * 2, 2048 * 2])
            else:
                self.fre_fusion_model = Attention_neck_cnn(in_channels=[512 * 2, 1024 * 2, 2048 * 2])
        else:
            self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)
            if isinstance(roi_head.bbox_head, list):
                self.bbox_head_is_list = True
            else:
                self.bbox_head_is_list = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if hasattr(self.backbone, 'in_type'):
            self.fusion_model = FusionModel()
        else:
            self.fusion_model = FusionModel_cnn()
        if isinstance(roi_head.bbox_head, list):
            self.num_classes = roi_head.bbox_head[0].num_classes
        else:
            self.num_classes = roi_head.bbox_head.num_classes

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_bbox))

    def assign_weight(self, img_metas, imgs, gt_bboxes, gt_labels, gt_multi_num):
        gt_bboxes_infrared = []
        gt_bboxes_rgb = []
        gt_labels_infrared = []
        gt_labels_rgb = []

        for bach in range(gt_multi_num.shape[0]):
            bbox_num = gt_multi_num[bach]
            # pdb.set_trace()
            gt_bboxes_infrared.append(gt_bboxes[bach][:int(bbox_num[0]), :])
            gt_bboxes_rgb.append(gt_bboxes[bach][int(bbox_num[0]):, :])
            gt_labels_infrared.append(gt_labels[bach][:int(bbox_num[0])])
            gt_labels_rgb.append(gt_labels[bach][int(bbox_num[0]):])

        gt_bboxes_infrared, gt_bboxes_rgb, gt_labels_infrared, gt_labels_rgb, img_meta_infrared, img_meta_rgb = multi_apply(
            self.single_weight, img_metas, imgs, gt_bboxes_infrared, gt_bboxes_rgb, gt_labels_infrared, gt_labels_rgb)
        return (gt_bboxes_infrared, gt_bboxes_rgb, gt_labels_infrared, gt_labels_rgb, img_meta_infrared, img_meta_rgb)

    def single_weight(self, img_meta, img, gt_bboxes_infrared, gt_bboxes_rgb, gt_labels_infrared, gt_labels_rgb):
        Iou_cal = RBboxOverlaps2D()
        thre_align = 0.5
        thre_miss = 0.1
        # pdb.set_trace()
        # 计算光照强度权重
        if 'rgb_hist' in img_meta:
            weight_iv = self.get_weight_iv(img_meta['rgb_hist'])
        # pdb.set_trace()
        infrared_list = [i for i in range(gt_bboxes_infrared.shape[0])]
        rgb_list = [i for i in range(gt_bboxes_rgb.shape[0])]
        # 计算同一场景不同模态目标之间的IOU
        ious = Iou_cal(gt_bboxes_infrared, gt_bboxes_rgb)
        rows_align, cols_align = torch.where(ious > thre_align)
        rows_miss, cols_miss = torch.where(ious > thre_miss)
        # 记录没有对齐的框
        res_infrared_list = [i for i in infrared_list if i not in rows_align and i in rows_miss]
        res_rgb_list = [i for i in rgb_list if i not in cols_align and i in cols_miss]

        # 计算可将光中没有对齐的框的权重，以红外中的目标为参照
        # gt_bboxes_infrared_weights = torch.ones_like(gt_labels_infrared)
        # gt_bboxes_rgb_weights = torch.ones_like(gt_labels_rgb) * weight_iv
        gt_bboxes_infrared_weights = numpy.ones(gt_labels_infrared.shape)
        gt_bboxes_rgb_weights = numpy.ones(gt_labels_rgb.shape) * weight_iv
        for row, col in zip(res_infrared_list, res_rgb_list):
            gt_bboxes_rgb_weights[col] = ious[row, col] * weight_iv
        # pdb.set_trace()
        # 记录各个模态丢失的框
        infrared_list = [i for i in range(gt_bboxes_infrared.shape[0])]
        rgb_list = [i for i in range(gt_bboxes_rgb.shape[0])]

        res_infrared_list = [i for i in infrared_list if i not in rows_miss]
        res_rgb_list = [i for i in rgb_list if i not in cols_miss]

        # 计算各个模态中丢失的框的权重
        for col in res_rgb_list:
            gt_bboxes_infrared = torch.cat((gt_bboxes_infrared, gt_bboxes_rgb[col:col + 1, :]))
            gt_labels_infrared = torch.cat((gt_labels_infrared, gt_labels_rgb[col:col + 1]))
            gt_bboxes_infrared_weights = numpy.append(gt_bboxes_infrared_weights, 0.5)
        for row in res_infrared_list:
            gt_bboxes_rgb = torch.cat((gt_bboxes_rgb, gt_bboxes_infrared[row:row + 1, :]))
            gt_labels_rgb = torch.cat((gt_labels_rgb, gt_labels_infrared[row:row + 1]))
            gt_bboxes_rgb_weights = numpy.append(gt_bboxes_rgb_weights, 0.5)
        # pdb.set_trace()
        assert gt_bboxes_infrared.shape[0] == gt_bboxes_infrared_weights.shape[0], "infrared_weights is wrong"
        assert gt_bboxes_rgb.shape[0] == gt_bboxes_rgb_weights.shape[0], "rgb_weights is wrong"
        img_meta_infrared = img_meta.copy()
        img_meta_rgb = img_meta.copy()
        img_meta_infrared['weight_infrared'] = torch.tensor(gt_bboxes_infrared_weights)
        img_meta_rgb['weight_rgb'] = torch.tensor(gt_bboxes_rgb_weights)
        return gt_bboxes_infrared, gt_bboxes_rgb, gt_labels_infrared, gt_labels_rgb, img_meta_infrared, img_meta_rgb

    def get_weight_iv(self, hist):
        pix_thre = 80
        weight_iv = numpy.sum(hist[pix_thre:, :]) / numpy.sum(hist)
        return weight_iv

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        if img.shape[1] == 6:
            img_infrared = img[:, 0:3, :, :]
            img_rgb = img[:, 3:, :, :]

        x_infrared = self.backbone(img_infrared)
        x_rgb = self.backbone(img_rgb)
        x_fusion = self.fusion_model(x_rgb, x_infrared)
        # pdb.set_trace()
        if self.neck:
            x_fusion = self.neck(x_fusion)
            x_infrared = self.neck(x_infrared)
            x_rgb = self.neck(x_rgb)

        return x_rgb, x_infrared, x_fusion

    def extract_feat_(self, img):
        """Directly extract features from the backbone+neck."""
        if img.shape[1] == 6:
            img_infrared = img[:, 0:3, :, :]
            img_rgb = img[:, 3:, :, :]
        # pdb.set_trace()
        x_infrared = self.backbone(img_infrared, RGB=False)
        x_rgb = self.backbone(img_rgb, RGB=True)
        x_fusion = self.fusion_model(x_rgb, x_infrared)
        # pdb.set_trace()
        if self.neck:
            x_fusion = self.neck(x_fusion)
            x_infrared = self.neck(x_infrared)
            x_rgb = self.neck(x_rgb)

        return x_rgb, x_infrared, x_fusion

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_multi_num,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        result = self.assign_weight(img_metas, img, gt_bboxes, gt_labels, gt_multi_num)

        (gt_bboxes_infrared, gt_bboxes_rgb, gt_labels_infrared, gt_labels_rgb, img_meta_infrared,
         img_meta_rgb) = result[:6]

        x_rgb, x_infrared, x_fusion = self.extract_feat(img)

        losses = dict()

        # train infrared image
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x_infrared,
                img_meta_infrared,
                gt_bboxes_infrared,
                gt_labels_infrared=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            for name, value in rpn_losses.items():
                losses['infrared_{}'.format(name)] = (value)
            # losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x_infrared, img_meta_infrared, proposal_list,
                                                 gt_bboxes_infrared, gt_labels_infrared,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)

        for name, value in roi_losses.items():
            losses['infrared_{}'.format(name)] = (value)

        # train rgb image
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)

            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x_rgb,
                img_meta_rgb,
                gt_bboxes_rgb,
                gt_labels_rgb=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            for name, value in rpn_losses.items():
                losses['rgb_{}'.format(name)] = (value)
            # losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x_rgb, img_meta_rgb, proposal_list,
                                                 gt_bboxes_rgb, gt_labels_rgb,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        for name, value in roi_losses.items():
            losses['rgb_{}'.format(name)] = (value)

        # train fusion image
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x_fusion,
                img_meta_infrared,
                gt_bboxes_infrared,
                gt_labels_infrared=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            for name, value in rpn_losses.items():
                losses['fusion_{}'.format(name)] = (value)
            # losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x_fusion, img_meta_infrared, proposal_list,
                                                 gt_bboxes_infrared, gt_labels_infrared,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        for name, value in roi_losses.items():
            losses['fusion_{}'.format(name)] = (value)
            # pdb.set_trace()
            # if value > 100:
            #     pdb.set_trace()
        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'

        cfg = self.test_cfg.rcnn
        x_rgb, x_infrared, x_fusion = self.extract_feat(img)

        # test rgb image
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x_rgb, img_metas)
        else:
            proposal_list = proposals

        results_rgb = self.roi_head.simple_test(
            x_rgb, proposal_list, img_metas, rescale=rescale)
        # pdb.set_trace()

        if 'rgb_hist' in img_metas[0]:
            weight_iv = [ele['rgb_hist'] for ele in img_metas]
        # pdb.set_trace()
        #
        # results_rgb = self.split_result(results_rgb, self.num_classes, weight=weight_iv)
        # pdb.set_trace()
        # test infrared image
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x_infrared, img_metas)
        else:
            proposal_list = proposals

        results_infrared = self.roi_head.simple_test(
            x_infrared, proposal_list, img_metas, rescale=rescale)

        # results_infrared = self.split_result(results_infrared, self.num_classes)

        # test fusion image
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x_fusion, img_metas)
        else:
            proposal_list = proposals

        results_fusion = self.roi_head.simple_test(
            x_fusion, proposal_list, img_metas, rescale=rescale)

        # results_fusion = self.split_result(results_fusion, self.num_classes)
        #
        # bboxes, scores = self.merge_results(results_rgb, results_infrared, results_fusion)
        #
        # bbox_results = self.get_results(bboxes, scores, self.num_classes, cfg)
        # pdb.set_trace()
        return results_rgb

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def get_results(self, bboxes, scores, num_classes, cfg):
        # nms = dict(iou_thr=0.5)
        det_bboxes_list = []
        det_labels_list = []
        for i in range(len(bboxes)):
            det_bboxes, det_labels = multiclass_nms_rotated_new(
                bboxes[i], scores[i], cfg.score_thr, cfg.nms, cfg.max_per_img)
            # pdb.set_trace()
            det_bboxes_list.append(det_bboxes)
            det_labels_list.append(det_labels)

        bbox_results = [rbbox2result(det_bboxes_list[i], det_labels_list[i], num_classes) for i in
                        range(len(det_bboxes_list))]
        return bbox_results

    def split_result(self, results, num_classes, weight=1):
        bbox_results = results
        num_imgs = len(results)
        num_classes = num_classes  # num classes
        results_list = []

        for i in range(num_imgs):
            result_dict = dict()
            bbox_list = []
            score_list = []
            for j in range(num_classes):
                bboxes = torch.zeros(bbox_results[i][j].shape[0], 5)
                scores = torch.zeros(bbox_results[i][j].shape[0], num_classes)

                if bbox_results[i][j].shape[0] != 0:
                    bboxes = torch.tensor(bbox_results[i][j][:, :5])
                    scores[:, j] = torch.tensor(bbox_results[i][j][:, 5])

                bbox_list.append(bboxes)
                score_list.append(scores)
            # pdb.set_trace()
            if isinstance(weight, list):
                result_dict['bboxes'] = torch.cat(bbox_list)
                result_dict['scores'] = torch.cat(score_list) * weight[i]
            else:
                result_dict['bboxes'] = torch.cat(bbox_list)
                result_dict['scores'] = torch.cat(score_list)
            # pdb.set_trace()
            results_list.append(result_dict)
        return results_list

    def merge_results(self, results_list1, results_list2, results_list3):
        bboxes = []
        scores = []
        for result1, result2, result3 in zip(results_list1, results_list2, results_list3):
            bbox = torch.cat((result1['bboxes'], result2['bboxes'], result3['bboxes']))
            score = torch.cat((result1['scores'], result2['scores'], result3['scores']))
            bboxes.append(bbox)
            scores.append(score)
        return bboxes, scores


@ROTATED_DETECTORS.register_module()
class RotatedTwoStageDetector_FusionT_2BN(RotatedTwoStageDetector_FusionT):

    def assign_weight(self, img_metas, imgs, gt_bboxes, gt_labels, gt_multi_num, gt_rgb_weight):
        gt_bboxes_infrared = []
        gt_bboxes_rgb = []
        gt_labels_infrared = []
        gt_labels_rgb = []

        for bach in range(gt_multi_num.shape[0]):
            bbox_num = gt_multi_num[bach]
            gt_bboxes_infrared.append(gt_bboxes[bach][:int(bbox_num[0]), :])
            gt_bboxes_rgb.append(gt_bboxes[bach][int(bbox_num[0]):, :])
            gt_labels_infrared.append(gt_labels[bach][:int(bbox_num[0])])
            gt_labels_rgb.append(gt_labels[bach][int(bbox_num[0]):])

        gt_bboxes_infrared, gt_bboxes_rgb, gt_labels_infrared, gt_labels_rgb, img_meta_infrared, img_meta_rgb = multi_apply(
            self.single_weight, img_metas, imgs, gt_bboxes_infrared, gt_bboxes_rgb, gt_labels_infrared, gt_labels_rgb,
            gt_rgb_weight)
        return (gt_bboxes_infrared, gt_bboxes_rgb, gt_labels_infrared, gt_labels_rgb, img_meta_infrared, img_meta_rgb)

    def single_weight(self, img_meta, img, gt_bboxes_infrared, gt_bboxes_rgb, gt_labels_infrared, gt_labels_rgb,
                      gt_rgb_weight):
        Iou_cal = RBboxOverlaps2D()
        thre_align = 0.5
        thre_miss = 0.1

        infrared_list = [i for i in range(gt_bboxes_infrared.shape[0])]
        rgb_list = [i for i in range(gt_bboxes_rgb.shape[0])]
        ious = Iou_cal(gt_bboxes_infrared, gt_bboxes_rgb)
        rows_align, cols_align = torch.where(ious > thre_align)
        rows_miss, cols_miss = torch.where(ious > thre_miss)
        res_infrared_list = [i for i in infrared_list if i not in rows_align and i in rows_miss]
        res_rgb_list = [i for i in rgb_list if i not in cols_align and i in cols_miss]

        gt_bboxes_infrared_weights = numpy.ones(gt_labels_infrared.shape)
        gt_bboxes_rgb_weights = gt_rgb_weight.cpu().numpy()

        for row, col in zip(res_infrared_list, res_rgb_list):
            gt_bboxes_rgb_weights[col] = ious[row, col] * gt_bboxes_rgb_weights[col]
        # pdb.set_trace()
        # 记录各个模态丢失的框
        infrared_list = [i for i in range(gt_bboxes_infrared.shape[0])]
        rgb_list = [i for i in range(gt_bboxes_rgb.shape[0])]

        res_infrared_list = [i for i in infrared_list if i not in rows_miss]
        res_rgb_list = [i for i in rgb_list if i not in cols_miss]

        # 计算各个模态中丢失的框的权重
        for col in res_rgb_list:
            gt_bboxes_infrared = torch.cat((gt_bboxes_infrared, gt_bboxes_rgb[col:col + 1, :]))
            gt_labels_infrared = torch.cat((gt_labels_infrared, gt_labels_rgb[col:col + 1]))
            gt_bboxes_infrared_weights = numpy.append(gt_bboxes_infrared_weights, 0.5)
        for row in res_infrared_list:
            gt_bboxes_rgb = torch.cat((gt_bboxes_rgb, gt_bboxes_infrared[row:row + 1, :]))
            gt_labels_rgb = torch.cat((gt_labels_rgb, gt_labels_infrared[row:row + 1]))
            gt_bboxes_rgb_weights = numpy.append(gt_bboxes_rgb_weights, 0.5)

        assert gt_bboxes_infrared.shape[0] == gt_bboxes_infrared_weights.shape[0], "infrared_weights is wrong"
        assert gt_bboxes_rgb.shape[0] == gt_bboxes_rgb_weights.shape[0], pdb.set_trace()
        img_meta_infrared = img_meta.copy()
        img_meta_rgb = img_meta.copy()
        img_meta_infrared['weight_infrared'] = torch.from_numpy(gt_bboxes_infrared_weights)
        img_meta_rgb['weight_rgb'] = torch.from_numpy(gt_bboxes_rgb_weights)
        return gt_bboxes_infrared, gt_bboxes_rgb, gt_labels_infrared, gt_labels_rgb, img_meta_infrared, img_meta_rgb

    def get_weight_iv(self, hist):
        pix_thre = 80
        weight_iv = numpy.sum(hist[pix_thre:, :]) / numpy.sum(hist)
        return weight_iv

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        if img.shape[1] == 6:
            img_infrared = img[:, 0:3, :, :]
            img_rgb = img[:, 3:, :, :]

        x_infrared = self.backbone(img_infrared)
        x_rgb = self.backbone(img_rgb)
        if self.fre_mode:
            with torch.no_grad():
                # x_fre_inf = self.backbone_fre.get_block_list_mutli(img_infrared)
                x_fre_rgb = self.backbone_fre.get_block_list_mutli(img_rgb)
            x_fre_rgb = self.backbone_fre(x_fre_rgb)
            x_rgb = self.fre_fusion_model(x_rgb, x_fre_rgb)
            # x_infrared = self.fre_fusion_model(x_infrared, x_fre_rgb)

        x_fusion = self.fusion_model(x_rgb, x_infrared)

        # pdb.set_trace()
        if self.neck:
            x_fusion = self.neck(x_fusion)
            x_infrared = self.neck(x_infrared)
            x_rgb = self.neck(x_rgb)

        return x_rgb, x_infrared, x_fusion

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_multi_num,
                      gt_rgb_weight,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # pdb.set_trace()
        # print(img_metas[0]['filename'])
        # print(img_metas[1]['filename'])
        with torch.no_grad():
            result = self.assign_weight(img_metas, img, gt_bboxes, gt_labels, gt_multi_num, gt_rgb_weight)
            (gt_bboxes_infrared, gt_bboxes_rgb, gt_labels_infrared, gt_labels_rgb, img_meta_infrared,
             img_meta_rgb) = result[:6]
        x_rgb, x_infrared, x_fusion = self.extract_feat(img)
        losses = dict()

        # train infrared image
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x_infrared,
                img_meta_infrared,
                gt_bboxes_infrared,
                gt_labels_infrared=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            for name, value in rpn_losses.items():
                losses['infrared_{}'.format(name)] = (value)
            # losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x_infrared, img_meta_infrared, proposal_list,
                                                 gt_bboxes_infrared, gt_labels_infrared,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)

        for name, value in roi_losses.items():
            losses['infrared_{}'.format(name)] = (value)

        # train rgb image
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)

            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x_rgb,
                img_meta_rgb,
                gt_bboxes_rgb,
                gt_labels_rgb=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            for name, value in rpn_losses.items():
                losses['rgb_{}'.format(name)] = (value)
            # losses.update(rpn_losses)
        else:
            proposal_list = proposals

        self.vis_loss = False
        if self.bbox_head_is_list:
            if hasattr(self.roi_head.bbox_head[0], 'get_info'):
                self.vis_loss = True
                self.roi_head.bbox_head[0].vis_pred = True
                self.roi_head.bbox_head[1].vis_pred = True
            if self.vis_loss:
                self.roi_head.bbox_head[0].get_info(img[:, 3:, :, :], img_meta_rgb)
                self.roi_head.bbox_head[1].get_info(img[:, 3:, :, :], img_meta_rgb)
        else:
            if hasattr(self.roi_head.bbox_head, 'get_info'):
                self.vis_loss = True
                self.roi_head.bbox_head.vis_pred = True
            if self.vis_loss:
                self.roi_head.bbox_head.get_info(img[:, 3:, :, :], img_meta_rgb)

        roi_losses = self.roi_head.forward_train(x_rgb, img_meta_rgb, proposal_list,
                                                 gt_bboxes_rgb, gt_labels_rgb,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)

        if self.bbox_head_is_list:
            self.roi_head.bbox_head[0].vis_pred = False
            self.roi_head.bbox_head[1].vis_pred = False
        else:
            self.roi_head.bbox_head.vis_pred = False

        for name, value in roi_losses.items():
            losses['rgb_{}'.format(name)] = (value)

        # train fusion image
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x_fusion,
                img_meta_infrared,
                gt_bboxes_infrared,
                gt_labels_infrared=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            for name, value in rpn_losses.items():
                losses['fusion_{}'.format(name)] = (value)
            # losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x_fusion, img_meta_infrared, proposal_list,
                                                 gt_bboxes_infrared, gt_labels_infrared,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        for name, value in roi_losses.items():
            losses['fusion_{}'.format(name)] = (value)
            if math.isnan(value):
                pdb.set_trace()

        return losses
