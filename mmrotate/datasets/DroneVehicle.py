import glob
import os
import os.path as osp
import re
import tempfile
import time
import zipfile
from collections import defaultdict
from functools import partial

import mmcv
import numpy as np
import torch
from mmcv.ops import nms_rotated
from mmdet.datasets.custom import CustomDataset

from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
from .builder import ROTATED_DATASETS
from mmdet.datasets.pipelines import Compose
import pdb
import cv2
from pycocotools import mask as maskUtils
from mmrotate.core import imshow_det_rbboxes


@ROTATED_DATASETS.register_module()
class DroneVehicleDataset(CustomDataset):
    # CLASSES = ('car', 'bus', 'truck', 'van', 'feright_car')
    CLASSES = ('car', 'bus', 'truck', 'van', 'feright_car')

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 file_client_args=dict(backend='disk'),
                 version='oc',
                 difficulty=100,
                 region_vi=False):
        # pdb.set_trace()
        self.ann_file = ann_file.split('+')
        self.data_root = data_root
        self.img_prefix = img_prefix.split('+')
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.file_client = mmcv.FileClient(**file_client_args)
        self.CLASSES = self.get_classes(classes)
        self.version = version
        self.difficulty = difficulty
        self.region_vi = region_vi
        # print(self.img_prefix)
        # 输入标签路径和图片路径为list里面有成对的红外和可见光图像的路径
        if self.data_root is not None:
            if not osp.isabs(self.ann_file[0]):
                self.ann_file = [osp.join(self.data_root, ann) for ann in self.ann_file]
            if not (self.img_prefix is None or osp.isabs(self.img_prefix[0])):
                self.img_prefix = [osp.join(self.data_root, img) for img in self.img_prefix]
        # 默认不同模态同一场景的图片名是相同的
        self.img_names = os.listdir(self.img_prefix[0])
        # pdb.set_trace()
        self.data_infos = self.load_annotations(self.ann_file)
        # pdb.set_trace()

        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)

        # super(DroneVehicleDataset, self).__init__(ann_file, pipeline, **kwargs)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations_txt(self, ann_folders):
        """
            Args:
                ann_folders: folder that contains DOTA v1 annotations txt files
        """
        cls_map = {c: i
                   for i, c in enumerate(self.CLASSES)
                   }  # in mmdet v2.0 label is 0-based

        ann_files = [glob.glob(ann_folder + '/*.txt') for ann_folder in ann_folders]
        data_infos = []
        # 由于同一场景的不同模态图片名称一样，因此只需要记录一个文件名即可
        if not ann_files[0]:
            ann_files = glob.glob(ann_folders[0] + '/*.jpg')
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.jpg'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            # 将同一场景不同模态的图片归纳到一起
            ann_same_con_list = self.get_same_condition(ann_files)
            for same_con in ann_same_con_list:
                data_info = {}
                gt_bboxes = []
                gt_labels = []
                gt_polygons = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                gt_rgb_weights = []
                gt_polygons_ignore = []
                gt_multi_num = []
                count = 0
                # for ann_file in same_con:
                for i in range(len(same_con)):
                    ann_file = same_con[i]
                    # if i!=0:
                    #     continue
                    img_id = osp.split(ann_file)[1][:-4]

                    img_name = img_id + '.jpg'
                    data_info['filename'] = img_name
                    data_info['ann'] = {}

                    if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
                        gt_multi_num.append(count)
                        continue

                    with open(ann_file) as f:
                        s = f.readlines()
                        for si in s:
                            if ' ' in si:
                                bbox_info = si.rstrip('\n').split(' ')
                            elif ',' in si:
                                bbox_info = si.rstrip('\n').split(',')

                            poly = np.array(bbox_info[:8], dtype=np.float32)
                            try:
                                x, y, w, h, a = poly2obb_np(poly, self.version)
                            except:  # noqa: E722
                                continue

                            cls_name = bbox_info[8]
                            difficulty = 1  # int(bbox_info[9])
                            if cls_name not in self.CLASSES:
                                continue
                            label = cls_map[cls_name]

                            if i == 1 and self.region_vi:
                                gt_rgb_weights.append(float(bbox_info[-1]))

                            if difficulty > self.difficulty:
                                pass
                            else:
                                count = count + 1
                                gt_bboxes.append([x, y, w, h, a])
                                gt_labels.append(label)
                                gt_polygons.append(poly)

                    gt_multi_num.append(count)

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)
                    data_info['ann']['polygons'] = np.array(
                        gt_polygons, dtype=np.float32)
                    data_info['ann']['count'] = np.array(
                        gt_multi_num, dtype=np.float32)
                    if self.region_vi:
                        # pdb.set_trace()
                        data_info['ann']['rgb_weight'] = np.array(
                            gt_rgb_weights, dtype=np.float32)
                        # pdb.set_trace()
                        assert data_info['ann']['rgb_weight'].shape[0] == (
                                gt_multi_num[1] - gt_multi_num[0]), pdb.set_trace()
                else:
                    data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                          dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons'] = np.zeros((0, 8),
                                                            dtype=np.float32)
                    data_info['ann']['count'] = np.zeros((0, 2), dtype=np.float32)
                    # pdb.set_trace()
                    if self.region_vi:
                        # pdb.set_trace()
                        data_info['ann']['rgb_weight'] = np.array([],
                                                                  dtype=np.float32)

                if gt_polygons_ignore:
                    data_info['ann']['bboxes_ignore'] = np.array(
                        gt_bboxes_ignore, dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        gt_labels_ignore, dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.array(
                        gt_polygons_ignore, dtype=np.float32)
                else:
                    data_info['ann']['bboxes_ignore'] = np.zeros(
                        (0, 5), dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        [], dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.zeros(
                        (0, 8), dtype=np.float32)

                data_infos.append(data_info)

        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        return data_infos

    def load_annotations(self, ann_folder):
        self.cat_ids = {c: i
                        for i, c in enumerate(self.CLASSES)
                        }  # in mmdet v2.0 label is 0-based
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        ann_files = glob.glob(ann_folder[0] + '/*.txt')
        i = 1
        self.img_ids = []
        data_infos = []
        for ann_file in ann_files:
            info = {}
            img_id = ann_file.split('/')[-1].replace('.txt', '')
            img_name = img_id + '.jpg'
            info['id'] = i
            info['filename'] = img_name
            info['basename'] = img_id
            info['width'] = 896
            info['height'] = 712
            self.img_ids.append(i)
            i = i + 1
            data_infos.append(info)

        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        base_name = self.data_infos[idx]['basename']
        return self._parse_ann_info(base_name)

    def _parse_ann_info(self, base_name):
        gt_bboxes = []
        gt_labels = []
        gt_polygons = []
        gt_bboxes_ignore = []
        gt_labels_ignore = []
        gt_rgb_weights = []
        gt_polygons_ignore = []
        gt_multi_num = []
        count = 0
        ann_files = [os.path.join(ele, base_name + '.txt') for ele in self.ann_file]
        ann = {}
        for i in range(len(ann_files)):
            ann_file = ann_files[i]
            if os.path.getsize(ann_file) != 0:
                with open(ann_file) as f:
                    s = f.readlines()
                    for si in s:
                        if ' ' in si:
                            bbox_info = si.rstrip('\n').split(' ')
                        elif ',' in si:
                            bbox_info = si.rstrip('\n').split(',')
                        if len(bbox_info) == 0:
                            continue
                        poly = np.array(bbox_info[:8], dtype=np.float32)
                        try:
                            x, y, w, h, a = poly2obb_np(poly, self.version)
                        except:
                            continue
                        # pdb.set_trace()
                        cls_name = bbox_info[8]
                        difficulty = 1  # int(bbox_info[9])
                        if cls_name not in self.CLASSES:
                            continue
                        label = self.cat_ids[cls_name]

                        if i == 1 and self.region_vi:
                            gt_rgb_weights.append(float(bbox_info[-1]))

                        if difficulty > self.difficulty:
                            pass
                        else:
                            count = count + 1
                            gt_bboxes.append([x, y, w, h, a])
                            gt_labels.append(label)
                            gt_polygons.append(poly)
                    gt_multi_num.append(count)
            else:
                gt_multi_num.append(count)

        if gt_bboxes:
            ann['bboxes'] = np.array(gt_bboxes, dtype=np.float32)
            ann['labels'] = np.array(gt_labels, dtype=np.int64)
            ann['polygons'] = np.array(gt_polygons, dtype=np.float32)
            ann['count'] = np.array(gt_multi_num, dtype=np.float32)

            if self.region_vi:
                # pdb.set_trace()
                ann['rgb_weight'] = np.array(gt_rgb_weights, dtype=np.float32)
                # pdb.set_trace()

                assert ann['rgb_weight'].shape[0] == (
                        gt_multi_num[1] - gt_multi_num[0]), pdb.set_trace()
        else:
            gt_multi_num = [0, 0]
            ann['bboxes'] = np.zeros((0, 5), dtype=np.float32)
            ann['labels'] = np.array([], dtype=np.int64)
            ann['polygons'] = np.zeros((0, 8), dtype=np.float32)
            ann['count'] = np.array(gt_multi_num, dtype = np.float32)
            if self.region_vi:
                # pdb.set_trace()
                ann['rgb_weight'] = np.array([], dtype=np.float32)

        if gt_polygons_ignore:
            ann['bboxes_ignore'] = np.array(gt_bboxes_ignore, dtype=np.float32)
            ann['labels_ignore'] = np.array(gt_labels_ignore, dtype=np.int64)
            ann['polygons_ignore'] = np.array(gt_polygons_ignore, dtype=np.float32)
        else:
            ann['bboxes_ignore'] = np.zeros((0, 5), dtype=np.float32)
            ann['labels_ignore'] = np.array([], dtype=np.int64)
            ann['polygons_ignore'] = np.zeros((0, 8), dtype=np.float32)
        # print(ann)
        # pdb.set_trace()
        return ann

    # 将同一场景不同模态的图片归纳到一起
    def get_same_condition(self, ann_files):
        ann_same_con_list = []
        for ele in self.img_names:
            name = ele.replace('.jpg', '.txt')
            if '.ipynb_checkpoints' in ele:
                continue
            same_con = []
            for annfile in ann_files:
                for ann in annfile:
                    if name in ann:
                        same_con.append(ann)
                        break
            assert len(same_con) == len(ann_files), pdb.set_trace()
            ann_same_con_list.append(same_con)
        return ann_same_con_list

    # 根据设置的阈值生成可见度mask
    def get_image_v(self, image_name):
        image = cv2.imread(os.path.join(self.img_prefix[1], image_name))
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        return (img_hsv[:, :, 2] > 80)

    # 根据box_mask计算box对应的权重
    def init_rgb_weight(self, box_mask, v_mask):
        box_pix_sum = np.sum(box_mask)
        mask_vi = box_mask & v_mask
        vi_sum = np.sum(mask_vi)
        weight_vi = vi_sum / box_pix_sum
        return weight_vi

    # def _filter_imgs(self):
    #     """Filter images without ground truths."""
    #     valid_inds = []
    #     for i, data_info in enumerate(self.data_infos):
    #         image_path_1 = os.path.join(self.img_prefix[0], data_info['filename'])
    #         image_path_2 = os.path.join(self.img_prefix[1], data_info['filename'])
    #         img_1 = mmcv.imread(image_path_1)
    #         img_2 = mmcv.imread(image_path_2)
    #         if (not self.filter_empty_gt
    #                 or data_info['ann']['labels'].size > 0):
    #             if img_1.shape[0] > 100 and img_2.shape[0] > 100:
    #                 valid_inds.append(i)
    #             else:
    #                 print(data_info['filename'])
    #     return valid_inds

    def annToRLE(self, segm, h, w):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        else:
            # rle
            rle = segm
        return rle

    def annToMask(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        segm, h, w = ann
        rle = self.annToRLE(segm, h, w)
        m = maskUtils.decode(rle)
        return m

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        All set to 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        nproc = min(nproc, os.cpu_count())
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_rbbox_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=self.CLASSES,
                logger=logger,
                nproc=nproc)
            eval_results['mAP'] = mean_ap
        else:
            raise NotImplementedError

        return eval_results

    def merge_det(self, results, nproc=4):
        """Merging patch bboxes into full image.

        Args:
            results (list): Testing results of the dataset.
            nproc (int): number of process. Default: 4.
        """
        collector = defaultdict(list)
        for idx in range(len(self)):
            result = results[idx]
            img_id = self.img_ids[idx]
            splitname = img_id.split('__')
            oriname = splitname[0]
            pattern1 = re.compile(r'__\d+___\d+')
            x_y = re.findall(pattern1, img_id)
            x_y_2 = re.findall(r'\d+', x_y[0])
            x, y = int(x_y_2[0]), int(x_y_2[1])
            new_result = []
            for i, dets in enumerate(result):
                bboxes, scores = dets[:, :-1], dets[:, [-1]]
                ori_bboxes = bboxes.copy()
                ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                    [x, y], dtype=np.float32)
                labels = np.zeros((bboxes.shape[0], 1)) + i
                new_result.append(
                    np.concatenate([labels, ori_bboxes, scores], axis=1))

            new_result = np.concatenate(new_result, axis=0)
            collector[oriname].append(new_result)

        merge_func = partial(_merge_func, CLASSES=self.CLASSES, iou_thr=0.1)
        if nproc <= 1:
            print('Single processing')
            merged_results = mmcv.track_iter_progress(
                (map(merge_func, collector.items()), len(collector)))
        else:
            print('Multiple processing')
            merged_results = mmcv.track_parallel_progress(
                merge_func, list(collector.items()), nproc)

        return zip(*merged_results)

    def _results2submission(self, id_list, dets_list, out_folder=None):
        """Generate the submission of full images.

        Args:
            id_list (list): Id of images.
            dets_list (list): Detection results of per class.
            out_folder (str, optional): Folder of submission.
        """
        if osp.exists(out_folder):
            raise ValueError(f'The out_folder should be a non-exist path, '
                             f'but {out_folder} is existing')
        os.makedirs(out_folder)

        files = [
            osp.join(out_folder, 'Task1_' + cls + '.txt')
            for cls in self.CLASSES
        ]
        file_objs = [open(f, 'w') for f in files]
        for img_id, dets_per_cls in zip(id_list, dets_list):
            for f, dets in zip(file_objs, dets_per_cls):
                if dets.size == 0:
                    continue
                bboxes = obb2poly_np(dets, self.version)
                for bbox in bboxes:
                    txt_element = [img_id, str(bbox[-1])
                                   ] + [f'{p:.2f}' for p in bbox[:-1]]
                    f.writelines(' '.join(txt_element) + '\n')

        for f in file_objs:
            f.close()

        target_name = osp.split(out_folder)[-1]
        with zipfile.ZipFile(
                osp.join(out_folder, target_name + '.zip'), 'w',
                zipfile.ZIP_DEFLATED) as t:
            for f in files:
                t.write(f, osp.split(f)[-1])

        return files

    def format_results(self, results, submission_dir=None, nproc=4, **kwargs):
        """Format the results to submission text (standard format for DOTA
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            submission_dir (str, optional): The folder that contains submission
                files. If not specified, a temp folder will be created.
                Default: None.
            nproc (int, optional): number of process.

        Returns:
            tuple:

                - result_files (dict): a dict containing the json filepaths
                - tmp_dir (str): the temporal directory created for saving \
                    json files when submission_dir is not specified.
        """
        nproc = min(nproc, os.cpu_count())
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            f'The length of results is not equal to '
            f'the dataset len: {len(results)} != {len(self)}')
        if submission_dir is None:
            submission_dir = tempfile.TemporaryDirectory()
        else:
            tmp_dir = None

        print('\nMerging patch bboxes into full image!!!')
        start_time = time.time()
        id_list, dets_list = self.merge_det(results, nproc)
        stop_time = time.time()
        print(f'Used time: {(stop_time - start_time):.1f} s')

        result_files = self._results2submission(id_list, dets_list,
                                                submission_dir)

        return result_files, tmp_dir

    def draw_gt_box(self, ann_folders):
        """
            Args:
                ann_folders: folder that contains DOTA v1 annotations txt files
        """
        cls_map = {c: i
                   for i, c in enumerate(self.CLASSES)
                   }  # in mmdet v2.0 label is 0-based
        ann_files = glob.glob(ann_folder + '/*.txt')
        data_infos = []
        if not ann_files:  # test phase
            ann_files = glob.glob(ann_folder + '/*.jpg')
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.png'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.jpg'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                gt_bboxes = []
                gt_labels = []

                if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
                    continue

                with open(ann_file) as f:
                    s = f.readlines()
                    for si in s:
                        bbox_info = si.rstrip('\n').split(',')

                        poly = np.array(bbox_info[:8], dtype=np.float32)
                        try:
                            x, y, w, h, a = poly2obb_np(poly, self.version)
                        except:  # noqa: E722
                            continue
                        # pdb.set_trace()
                        cls_name = bbox_info[8]
                        difficulty = 1  # int(bbox_info[9])
                        if cls_name not in self.CLASSES:
                            continue
                        label = cls_map[cls_name]

                        if difficulty > self.difficulty:
                            pass
                        else:
                            gt_bboxes.append([x, y, w, h, a])
                            gt_labels.append(label)

                if gt_bboxes:
                    bboxes = np.array(
                        gt_bboxes, dtype=np.float32)
                    labels = np.array(
                        gt_labels, dtype=np.int64)
                segms = None
                img = os.path.join(self.img_prefix[1], image_name)
                img = imshow_det_rbboxes(
                    img,
                    bboxes,
                    labels,
                    segms,
                    class_names=self.CLASSES,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None)

        return data_infos


def _merge_func(info, CLASSES, iou_thr):
    """Merging patch bboxes into full image.

    Args:
        CLASSES (list): Label category.
        iou_thr (float): Threshold of IoU.
    """
    img_id, label_dets = info
    label_dets = np.concatenate(label_dets, axis=0)

    labels, dets = label_dets[:, 0], label_dets[:, 1:]

    big_img_results = []
    for i in range(len(CLASSES)):
        if len(dets[labels == i]) == 0:
            big_img_results.append(dets[labels == i])
        else:
            try:
                cls_dets = torch.from_numpy(dets[labels == i]).cuda()
            except:  # noqa: E722
                cls_dets = torch.from_numpy(dets[labels == i])
            nms_dets, keep_inds = nms_rotated(cls_dets[:, :5], cls_dets[:, -1],
                                              iou_thr)
            big_img_results.append(nms_dets.cpu().numpy())
    return img_id, big_img_results
