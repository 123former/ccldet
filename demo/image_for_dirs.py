# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import mmrotate  # noqa: F401
import mmcv
import pdb
import cv2
import os
from tqdm import tqdm
import os


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img_dirs', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def getFileList(dir, Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist


# classnames = ['0', '1', '2', '3', '4', '07']
classnames = ['0', '1', '2', '3', '4', '05', '08', '09']
# classnames = ['0', '1', '2', '3', '4']
# classnames = ['0', '2', '07']
# classnames = ['1', '3', '4']
dota_colormap = [
    (54, 67, 244),
    (120, 188, 50),
    (176, 39, 156),
    (183, 58, 103),
    (181, 81, 63),
    (243, 150, 33),
    (120, 188, 0),
    (136, 150, 0),
    (80, 175, 76),
    (74, 195, 139),
    (57, 220, 205),
    (59, 235, 255),
    (0, 152, 255),
    (34, 87, 255),
    (72, 85, 121)]


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # pdb.set_trace()
    # test a single image
    img_list = getFileList(args.img_dirs, [], 'jpg')
    json_list = getFileList(args.img_dirs, [], 'json')
    root = '/home/f523/guazai/sdb/gzjc/dataset_trains_1.5/'
    for img in tqdm(img_list):
        if 'aug' in img:
            continue
        if 'concat' in img:
            continue
        if '.jpg' not in img:
            continue

        basename = os.path.basename(img)
        out_dir = img.replace(root, args.out_file).replace(basename, '')

        temp = img.replace('.jpg', '.json')

        if temp in json_list:
            out_path = os.path.join(out_dir, '0.jpg')
        else:
            out_path = os.path.join(out_dir, basename)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        img_path = img
        result = inference_detector(model, img_path)

        img = draw_poly_detections(img_path,
                                   result,
                                   classnames,
                                   scale=1,
                                   threshold=0.5,
                                   out_path=out_path,
                                   putText=True,
                                   colormap=dota_colormap)


# def main(args):
#     # build the model from a config file and a checkpoint file
#     model = init_detector(args.config, args.checkpoint, device=args.device)
#     # pdb.set_trace()
#     # test a single image
#     img_dir_list = os.listdir(args.img_dirs)
#     for dir_name in img_dir_list:
#         # pdb.set_trace()
#         img_dir = os.path.join(args.img_dirs, dir_name)
#         img_list = os.listdir(img_dir)
#
#         for img in tqdm(img_list):
#             if '.jpg' not in img:
#                 continue
#             img_path = os.path.join(img_dir, img)
#
#             out_dir = os.path.join(args.out_file, dir_name)
#             temp = img.replace('.jpg', '.json')
#             if temp in img_list:
#                 out_path = os.path.join(out_dir, '0.jpg')
#             else:
#                 out_path = os.path.join(out_dir, img)
#
#             if not os.path.exists(out_dir):
#                 os.makedirs(out_dir)
#             result = inference_detector(model, img_path)
#
#             img = draw_poly_detections(img_path,
#                                        result,
#                                        classnames,
#                                        scale=1,
#                                        threshold=0.5,
#                                        out_path=out_path,
#                                        putText=True,
#                                        colormap=dota_colormap)


def draw_poly_detections(img, detections, class_names, scale, threshold=0.2, out_path=None, putText=False,
                         showStart=False,
                         colormap=None):
    """

    :param img:
    :param detections:
    :param class_names:
    :param scale:
    :param cfg:
    :param threshold:
    :return:
    """
    import pdb

    import random
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    h, w, c = img.shape
    color_white = (255, 255, 255)

    for j, name in enumerate(class_names):
        if name == '4':
            continue
        if name == '3':
            continue
        if name == '05':
            continue
        if name == '08':
            continue
        if name == '09':
            continue

        if colormap is None:
            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        else:
            color = colormap[j]
        try:
            dets = detections[j]
        except:
            pdb.set_trace()

        for det in dets:
            # pdb.set_trace()
            bbox = [det[0], det[1], det[0], det[3], det[2], det[3], det[2], det[1]]  # * scale
            score = det[-1]
            if score < threshold:
                continue
            bbox = list(map(int, bbox))
            if showStart:
                cv2.circle(img, (bbox[0], bbox[1]), 3, (0, 0, 255), -1)
            for i in range(3):
                cv2.line(img, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i + 1) * 2], bbox[(i + 1) * 2 + 1]), color=color,
                         thickness=8, lineType=cv2.LINE_AA)
            cv2.line(img, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=color, thickness=2, lineType=cv2.LINE_AA)
            if putText:
                cv2.putText(img, '%s %.3f' % (class_names[j], score), (bbox[0] - 20, bbox[1] + 100),
                            color=color, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=3)
            # cv2.resize(img, (int(h/10), int(w/10), c), temp_img)
            cv2.imwrite(out_path, cv2.resize(img, (409, 700)))
    return img


def save_box_part(img, detections, class_names, scale, threshold=0.2, out_path=None):
    """

    :param img:
    :param detections:
    :param class_names:
    :param scale:
    :param cfg:
    :param threshold:
    :return:
    """
    import pdb

    import random
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    color_white = (255, 255, 255)
    count = 0
    for j, name in enumerate(class_names):
        try:
            dets = detections[j]
        except:
            pdb.set_trace()
        if (j == 0):
            continue
        for det in dets:
            # pdb.set_trace()
            bbox = [det[0], det[1], det[0], det[3], det[2], det[3], det[2], det[1]]  # * scale
            score = det[-1]
            if score < threshold:
                continue
            # box_part = img[int(det[0]):int(det[2]), int(det[1]):int(det[3]), :]
            box_part = img[int(det[1]) - 10:int(det[3]) + 10, int(det[0]) - 10:int(det[2]) + 10, :]
            if box_part.shape[0] == 0:
                continue
            if box_part.shape[1] == 0:
                continue
            try:
                cv2.imwrite(out_path.replace(".jpg", "_" + str(count) + ".jpg"), box_part)
                count = count + 1
            except:
                continue

    return img


if __name__ == '__main__':
    args = parse_args()
    main(args)
