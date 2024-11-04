# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from mmdet.datasets.pipelines import LoadImageFromFile, LoadAnnotations
import os.path as osp
import torch
from ..builder import ROTATED_PIPELINES
import cv2
import pdb

@ROTATED_PIPELINES.register_module()
class LoadPatchFromImage(LoadImageFromFile):
    """Load an patch from the huge image.

    Similar with :obj:`LoadImageFromFile`, but only reserve a patch of
    ``results['img']`` according to ``results['win']``.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with image in ``results['img']``.

        Returns:
            dict: The dict contains the loaded patch and meta information.
        """

        img = results['img']
        x_start, y_start, x_stop, y_stop = results['win']
        width = x_stop - x_start
        height = y_stop - y_start

        patch = img[y_start:y_stop, x_start:x_stop]
        if height > patch.shape[0] or width > patch.shape[1]:
            patch = mmcv.impad(patch, shape=(height, width))

        if self.to_float32:
            patch = patch.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = patch
        results['img_shape'] = patch.shape
        results['ori_shape'] = patch.shape
        results['img_fields'] = ['img']
        return results


@ROTATED_PIPELINES.register_module()
class LoadImageFromFile_Fusion:
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='bgr',
                 file_client_args=dict(backend='disk'),
                 rgb_hist=True):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.rgb_hist = rgb_hist

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        imgname = results['img_info']['filename']
        # pdb.set_trace()
        img_list_ = []
        if results['img_prefix'] is not None:
            for img_path in results['img_prefix']:
                filename = osp.join(img_path, imgname)
                # print(results['img_prefix'])
                # print(filename)
                img_bytes = self.file_client.get(filename)
                img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
                if self.to_float32:
                    img = img.astype(np.float32)
                img_list_.append(img)
        else:
            for img_path in results['img_info']['filename']:
                img_bytes = self.file_client.get(img_path)
                img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
                if self.to_float32:
                    img = img.astype(np.float32)
                img_list_.append(img)

        img_list = img_list_
        img = cv2.merge(img_list)
        if self.rgb_hist:
            # hist = self.get_hist(img[:, :, 3:])
            hist = self.img_to_GRAY(img[:, :, 3:])
        # print('img:', img.shape)
        if results['img_prefix'] is not None:
            results['filename'] = [osp.join(img_dir, results['img_info']['filename']) for img_dir in results['img_prefix']]
        else:
            results['filename'] = results['img_info']['filename']
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        if self.rgb_hist:
            results['rgb_hist'] = hist

        return results

    def get_hist(self, img):

        pix_thre = 125
        # image = img.transpose(1, 2, 0)
        image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        # weight_iv = np.sum(hist[pix_thre:, :]) / np.sum(hist)
        # pdb.set_trace()
        return hist

    def img_to_GRAY(self, img):
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        r,c = gray_img.shape[:2]   # row and col of a gray image
        piexs_sum=r*c              #sum of pixels in a gray image
        dark_points = (gray_img < 60)
        target_array = gray_img[dark_points]
        dark_sum = target_array.size
        dark_prop=dark_sum/(piexs_sum)
        if dark_prop >=0.45:
            w_dark = 1-dark_prop
        else:
            w_dark = 1
        return w_dark

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"channel_order='{self.channel_order}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@ROTATED_PIPELINES.register_module()
class LoadAnnotations_Fusion(LoadAnnotations):

    def __init__(self, with_rgb_weight=False, **kwargs):
        self.with_rgb_weight = with_rgb_weight
        super(LoadAnnotations_Fusion, self).__init__(**kwargs)

    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        if self.denorm_bbox:
            bbox_num = results['gt_bboxes'].shape[0]
            if bbox_num != 0:
                h, w = results['img_shape'][:2]
                results['gt_bboxes'][:, 0::2] *= w
                results['gt_bboxes'][:, 1::2] *= h

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')

        gt_is_group_ofs = ann_info.get('gt_is_group_ofs', None)
        if gt_is_group_ofs is not None:
            results['gt_is_group_ofs'] = gt_is_group_ofs.copy()

        results['gt_multi_num'] = ann_info['count'].copy()
        # self.with_rgb_weight = True
        return results

    def _load_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results

    def _load_rgb_weights(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_rgb_weight'] = results['ann_info']['rgb_weight'].copy()
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        if self.with_rgb_weight:
            results = self._load_rgb_weights(results)
        return results
