import os
import cv2
from tqdm import tqdm
import pdb
import json
import random
import numpy as np

count = 0
count_aug = 0

class_name_to_id = {'putongqunban_up': 1,
                    'putongqunban_medium': 1,
                    'putongqunban_down': 1,
                    'zhuanxiangjiaqunban_up': 1,
                    'zhuanxiangjiaqunban_medium': 1,
                    'zhuanxiangjiaqunban_down': 1,
                    'diban_lost': 0,
                    'geshan_lost': 2,
                    'gaiban_open': 3,
                    '0': 0,
                    '1': 1,
                    '2': 2,
                    '3': 3,
                    '4': 4,
                    '05': 5,
                    '08': 6,
                    '09': 7
                    }


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


def make_mask(pos_size):
    mask = np.zeros(pos_size)
    x_mask = int(random.randint(4, 9) * pos_size[0] / 10)
    y_mask = int(random.randint(2, 5) * pos_size[1] / 10)
    y_step = y_mask / (x_mask)
    k = y_mask
    for i in range(x_mask):
        k = k - y_step
        for j in range(int(k)):
            try:
                mask[i][j] = 1
            except:
                pdb.set_trace()
    mask_bool = (mask == 1)
    return mask_bool, (x_mask, y_mask)


def cut_pic(json_path, img_path, out_dir, basename):
    global count
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    with open(json_path, 'r') as fp:
        json_data = json.load(fp)

        objects = json_data['shapes']
        for obj in objects:
            label = obj['label']
            box = obj['points']

            try:
                cls = str(class_name_to_id[label])
            except:
                continue

            if cls in ['0', '2']:
                out_path = os.path.join(out_dir, cls + '_1_m_' + basename.replace('.json', '_') + str(count) + '.jpg')
            else:
                continue

            min_x = min(box[0][0], box[1][0])
            min_y = min(box[0][1], box[1][1])
            max_x = max(box[0][0], box[1][0])
            max_y = max(box[0][1], box[1][1])

            min_x = max(min_x, 0)
            min_y = max(min_y, 0)
            max_x = min(max_x, 2047)
            max_y = min(max_y, 5599)

            img_slice = img[int(min_y): int(max_y), int(min_x):int(max_x), :]

            if (max_y - min_y) < h * 0.15:
                continue

            try:
                cv2.imwrite(out_path, img_slice)
                count = count + 1
            except:
                pdb.set_trace()


def get_info(cls):
    split_info = cls.split('_')
    position = split_info[0]
    camera = split_info[1]
    direction = split_info[2]

    return position, camera, direction


def get_suitable_slice(position, camera, direction, img_dir, img_slice):
    qunban_dir = os.path.join(img_dir, )
    qunban_list = os.listdir(qunban_dir)
    img_h, img_w, _ = img_slice.shape
    slice_list = []
    temp_list1 = []
    for slice in qunban_list:
        p, c, d = get_info(slice)
        if [p, c, d] == [position, camera, direction]:
            temp_list1.append(slice)
    try:
        random_index = random.randint(0, len(temp_list1) - 1)
    except:
        return slice_list

    ele = temp_list1[random_index]
    ele_path = os.path.join(img_dir, ele)
    ele_img = cv2.imread(ele_path)
    ele_h, ele_w, _ = ele_img.shape

    slice_list.append(ele)

    if ele_h < img_h:
        gap_h = img_h - ele_h
        temp_list2 = []
        for slice in qunban_list:
            p, c, d = get_info(slice)
            if [p, c, d] == ['2', camera, 'm']:
                slice_path = os.path.join(img_dir, slice)
                temp_img = cv2.imread(slice_path)
                temp_h, temp_w, _ = temp_img.shape
                if temp_h >= gap_h:
                    temp_list2.append(slice)
        try:
            random_index = random.randint(0, len(temp_list2) - 1)
        except:
            return []
        ele = temp_list2[random_index]
        slice_list.append(ele)
    return slice_list


def pad_img(slice_dir, img_slice, cls, flag):
    p, c, d = get_info(cls)
    slice_list = get_suitable_slice(p, c, d, slice_dir, img_slice)
    slice_length = len(slice_list)

    # img = np.ones_like(img_slice) * 114
    img = img_slice
    ele_path = os.path.join(slice_dir, slice_list[0])

    if 'to' in ele_path:
        s_flag = 1
    else:
        s_flag = 0

    if flag == s_flag:
        rate = 1
    elif flag == 1 and s_flag == 0:
        rate = 5
    elif flag == 0 and s_flag == 1:
        rate = 0.2

    ele_img_temp = cv2.imread(ele_path)
    k = max(img.shape[1], img.shape[0]) / (ele_img_temp.shape[1] * rate)
    ele_img = cv2.resize(ele_img_temp, (int(ele_img_temp.shape[1] * rate * k), int(ele_img_temp.shape[0] * rate * k)))
    h = min(img.shape[0], ele_img.shape[0])
    w = min(img.shape[1], ele_img.shape[1])

    if 'd' == d:
        img[-h:-1, -w:-1, :] = ele_img[-h:-1, -w:-1, :] * 0.99 + img[-h:-1, -w:-1, :] * 0.01
    elif 'u' == d:
        img[0:h, -w:-1, :] = ele_img[0:h, -w:-1, :] * 0.99 + img[0:h, -w:-1, :] * 0.01
    elif 'm' == d:
        img[0:h, -w:-1, :] = ele_img[0:h, -w:-1, :] * 0.99 + img[0:h, -w:-1, :] * 0.01

    return img, (h, w)


def pad_img_0(slice_dir, img_slice, cls, flag):
    p, c, d = get_info(cls)
    slice_list = get_suitable_slice(p, c, d, slice_dir, img_slice)
    # slice_length = len(slice_list)

    img = img_slice
    ele_path = os.path.join(slice_dir, slice_list[0])

    ele_img_temp = cv2.imread(ele_path)
    h = img.shape[0]
    w = img.shape[1]
    ele_img = cv2.resize(ele_img_temp, (img.shape[1], img.shape[0]))

    mode = random.randint(0, 20)
    if mode < 10:
        alpha = random.randint(10, 15)
        ele_img = ele_img * (alpha / 10)
        img[0:h, 0:w, :] = ele_img[0:h, 0:w, :] * 0.8
    elif mode < 18:
        alpha = random.randint(2, 5)
        ele_img = ele_img * (alpha / 10) + np.average(img) * 0.8
        img[0:h, 0:w, :] = ele_img[0:h, 0:w, :] * 0.8
    # elif mode >= 18:
    #     mask, (h, w) = make_mask(ele_img.shape)
    #     alpha = random.randint(10, 15)
    #     img[mask] = ele_img[mask] * (alpha / 10)
    return img.astype(np.uint8), (h, w)


def paste_slice2pic(slice_dir, img_path, json_path, new_json_path=None, new_img_path=None):
    global count_aug
    img = cv2.imread(img_path)

    in_fp = open(json_path, 'r')
    json_data = json.load(in_fp)

    objects = json_data['shapes']

    if 'to' in img_path:
        flag = 1
    else:
        flag = 0
    for obj in objects:
        label = obj['label']
        box = obj['points']

        if str(label) not in ['0', '08']:
            # cv2.imwrite(new_img_path, img)
            # with open(new_json_path, 'w')as fp:
            #     json.dump(json_data, fp)
            continue
        if label == '0':
            cls = '0_1_m'
        elif label == '08':
            cls = '2_1_m'

        min_x = min(box[0][0], box[1][0])
        min_y = min(box[0][1], box[1][1])
        max_x = max(box[0][0], box[1][0])
        max_y = max(box[0][1], box[1][1])

        min_x = max(min_x, 0)
        min_y = max(min_y, 0)
        max_x = min(max_x, 2047)
        max_y = min(max_y, 5599)

        img_slice = img[int(min_y):int(max_y), int(min_x): int(max_x), :]

        if cls == '0_1_m':
            img_slice, (h, w) = pad_img_0(slice_dir, img_slice, cls, flag)
            new_box = [[min_x, min_y], [min_x + w, min_y + h]]
            obj['label'] = '0'
            # obj['points'] = new_box
        elif cls == '2_1_m':
            img_slice, (h, w) = pad_img(slice_dir, img_slice, cls, flag)
            obj['label'] = '2'
            new_box = [[min_x, min_y], [min_x + w, min_y + h]]
            obj['points'] = new_box

            # cv2.namedWindow('img_slice', cv2.WINDOW_NORMAL)
            # cv2.imshow('img_slice', img_slice)
            # cv2.waitKey()
        if img_slice is not None:
            new_img_path = new_img_path.replace('.jpg', '_' + str(count_aug) + '.jpg')
            img[int(min_y):int(max_y), int(min_x): int(max_x), :] = img_slice[:, :, :]
            cv2.imwrite(new_img_path, img)
            if len(json_data['shapes']) > 0:
                new_json_path = new_json_path.replace('.json', '_' + str(count_aug) + '.json')
                with open(new_json_path, 'w')as fp:
                    json.dump(json_data, fp)
            count_aug = count_aug + 1
            break
        else:
            break


def pickl_data(img_list, json_list, txt_file, root):
    fp = open(txt_file, 'a')
    img_dict = dict()
    car_cls = ['1A', '300AF', '400AF', '400BF', '400AF-A', '400AF-Z', '400BF-Z', '2A', '2E', '380AL', '6F', '1A-A',
               '3C']
    cls_sample_num = 24000
    img_count = 0
    label_count = 0
    for img in img_list:
        if '10mm' in img:
            continue

        if img.replace('.jpg', '.json') in json_list:
            if 'aug' in img:
                continue
            file_path = img.replace(root, '')
            fp.write(file_path + '\n')
            label_count = label_count + 1
        else:
            for cls in car_cls:
                if cls in img and (cls + '-') not in img:
                    if cls not in img_dict.keys():
                        img_dict[cls] = []
                    img_dict[cls].append(img)
                    img_count = img_count + 1

    print('label', label_count)
    # print('img_num', img_count)
    # sample_gap = int(img_count / cls_sample_num) + 1
    # for cls in car_cls:
    #     cls_len = len(img_dict[cls])
    #     cls_num = 0
    #     for img in tqdm(img_dict[cls]):
    #         try:
    #             img_ = cv2.imread(img)
    #         except:
    #             continue
    #         mode = random.randint(0, sample_gap)
    #         if mode == 1:
    #             file_path = img.replace(root, '')
    #             fp.write(file_path + '\n')
    #             cls_num = cls_num + 1
    #     print(cls, cls_num)


if __name__ == '__main__':
    org_img_folder = '/home/f523/guazai/sdb/gzjc/dataset_trains_1.5'
    slice_dir = '/home/f523/guazai/sdb/gzjc/slice'
    aug_out_dir = '/home/f523/guazai/sdb/gzjc/dataset_trains_1.5/aug_all'
    root = '/home/f523/guazai/sdb/gzjc/dataset_trains_1.5/'
    txt_file = '/home/f523/guazai/disk3/shangxiping/data/动车故障/new_sample/defect_txt.txt'

    if not os.path.exists(slice_dir):
        os.makedirs(slice_dir)

    if not os.path.exists(aug_out_dir):
        os.makedirs(aug_out_dir)

    json_list = getFileList(org_img_folder, [], 'json')
    img_list = getFileList(org_img_folder, [], 'jpg')

    # for label in tqdm(json_list):
    #     if 'aug' in label:
    #         continue
    #     json_path = label
    #     img_path = label.replace('json', 'jpg')
    #     basename = os.path.basename(json_path)
    #     cut_pic(json_path, img_path, slice_dir, basename)

    # for label in tqdm(json_list):
    #     if 'aug' in label:
    #         continue
    #     json_path = label
    #     img_path = label.replace('json', 'jpg')
    #     basename = os.path.basename(json_path)
    #
    #     new_img_path = os.path.join(aug_out_dir, basename.replace('.json', '.jpg'))
    #     new_json_path = os.path.join(aug_out_dir, basename)
    #
    #     paste_slice2pic(slice_dir, img_path, json_path, new_json_path=new_json_path, new_img_path=new_img_path)

    pickl_data(img_list, json_list, txt_file, root)
