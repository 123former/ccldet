# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot, inference_detector_fusion

import mmrotate  # noqa: F401

import pdb

import os


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img_path', help='Image file')
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


def write_txt(results, img_path, classes, out_path):
    txt_lines = []
    with open(out_path, 'w') as f:
        for i in range(len(results)):
            for j in range(results[i].shape[0]):
                line_list = results[i][j].tolist()
                line_list = [str(ele) for ele in line_list]
                line_list.append(classes[i])
                line_list.append('\n')
                line_list = ' '.join(line_list)
                f.writelines(line_list)

    # pdb.set_trace()


classes = ['person', ]


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # pdb.set_trace()
    # test a single image
    rgb_path = os.path.join(args.img_path, 'test/testimg')
    inf_path = os.path.join(args.img_path, 'test/testimgr')
    rgb_list = os.listdir(rgb_path)
    inf_list = os.listdir(inf_path)

    out_file = args.out_file
    rgb_out_file = os.path.join(out_file, 'rgb')
    inf_out_file = os.path.join(out_file, 'inf')

    if not os.path.exists(rgb_out_file):
        os.makedirs(rgb_out_file)

    if not os.path.exists(inf_out_file):
        os.makedirs(inf_out_file)

    for rgb_img, inf_img in zip(rgb_list, inf_list):
        rgb_img_path = os.path.join(rgb_path, rgb_img)
        inf_img_path = os.path.join(inf_path, inf_img)
        img_path = [inf_img_path, rgb_img_path]
        result_rgb, result_inf = inference_detector_fusion(model, img_path)
        # pdb.set_trace()
        # write the txt
        # txt_out_path = os.path.join(inf_out_file, inf_img).replace('.jpg', '.txt')
        # write_txt(result_inf[0], img_path, classes, txt_out_path)
        #
        # txt_out_path = os.path.join(rgb_out_file, rgb_img).replace('.jpg', '.txt')
        # write_txt(result_rgb[0], img_path, classes, txt_out_path)
        # show the results
        show_result_pyplot(
            model,
            rgb_img_path,
            result_rgb[0],
            palette=args.palette,
            score_thr=args.score_thr,
            out_file=os.path.join(rgb_out_file, rgb_img))

        show_result_pyplot(
            model,
            inf_img_path,
            result_inf[0],
            palette=args.palette,
            score_thr=args.score_thr,
            out_file=os.path.join(inf_out_file, inf_img))


if __name__ == '__main__':
    args = parse_args()
    main(args)
