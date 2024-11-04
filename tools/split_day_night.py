import mmcv
import os
import shutil
import numpy as np
import pdb


def move_file(ref_txt, source_dir, ext='.jpg'):
    file_list = ['test/test_0/', 'test/test_1/', 'test/test_2/']
    with open(ref_txt, 'r') as f:
        while True:
            line = f.readline()
            if line:
                line_split = line.split(' ')
                name = line_split[0].replace('.jpg', ext)
                flag = int(line_split[1])
                source_file = os.path.join(source_dir, name)
                dst_path = source_dir.replace('test/', file_list[flag])

                if not os.path.exists(dst_path):
                    os.makedirs(dst_path)
                dst_file = os.path.join(dst_path, name)
                shutil.copy(source_file, dst_file)
            else:
                break


# 0 弱光夜晚 1 强光夜晚 2 白天
def write_day_night(img_dir, result_txt):
    imgs_list = os.listdir(img_dir)
    w_n = 0
    s_n = 0
    d = 0

    with open(result_txt, 'w') as f:
        for img in imgs_list:
            img_path = os.path.join(img_dir, img)
            flag = get_day_night(img_path)
            if flag == 0:
                w_n = w_n + 1
            elif flag == 1:
                s_n = s_n + 1
            elif flag == 2:
                d = d + 1
            line = img + ' ' + str(flag) + '\n'
            # print(line)
            # pdb.set_trace()
            f.writelines(line)
    print('w_n:', w_n)
    print('s_n:', s_n)
    print('day:', d)


def get_day_night(img_path):
    img = mmcv.imread(img_path)
    img_bgr = mmcv.rgb2bgr(img)
    img_hsv = mmcv.bgr2hsv(img_bgr)
    crop_region = np.array([100, 100, 739, 611])
    img_d_hsv = mmcv.imcrop(img_hsv, crop_region)
    flag = 0
    if np.average(img_d_hsv[:, :, 2]) < 50:
        flag = 0
    elif np.average(img_d_hsv[:, :, 2]) < 120:
        flag = 1
    else:
        flag = 2
    return flag


def main_1():
    img_dir = '/home/f523/guazai/disk3/shangxiping/redet/rgb_infrared_Vehicle/test/testimg'
    result_txt = '/home/f523/guazai/disk3/shangxiping/redet/rgb_infrared_Vehicle/test/result.txt'
    write_day_night(img_dir, result_txt)


def main_2():
    source_dir = '/home/f523/guazai/disk3/shangxiping/redet/rgb_infrared_Vehicle/test/testlabelr_txt'
    ref_txt = '/home/f523/guazai/disk3/shangxiping/redet/rgb_infrared_Vehicle/test/result.txt'
    move_file(ref_txt, source_dir, ext='.txt')


main_2()
