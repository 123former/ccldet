# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ROTATED_DETECTORS
from .two_stage_fusion import RotatedTwoStageDetector_Fusion, RotatedTwoStageDetector_FusionT, RotatedTwoStageDetector_FusionT_2BN
from mmrotate.core.bbox.iou_calculators.rotate_iou2d_calculator import RBboxOverlaps2D
import torch
import pdb
import cv2
from mmdet.core import multi_apply

@ROTATED_DETECTORS.register_module()
class ReDet_Fusion(RotatedTwoStageDetector_FusionT_2BN):
    """Implementation of `ReDet: A Rotation-equivariant Detector for Aerial
    Object Detection.`__

    __ https://openaccess.thecvf.com/content/CVPR2021/papers/Han_ReDet_A_Rotation-Equivariant_Detector_for_Aerial_Object_Detection_CVPR_2021_paper.pdf  # noqa: E501, E261.
    """

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(ReDet_Fusion, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        