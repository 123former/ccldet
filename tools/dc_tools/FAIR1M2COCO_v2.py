import dota_utils as util
import os
import cv2
import json
from PIL import Image
import xml.etree.ElementTree as ET
import random
from pathlib import Path
import pdb
import mmcv
import shutil

wordname_15 = ['Boeing737', 'Boeing777', 'Boeing747', 'Boeing787', 'A321', 'A220', 'A330', 'A350',
               'C919', 'ARJ21', 'Passenger Ship', 'Motorboat', 'Fishing Boat', 'Tugboat', 'Engineering Ship',
               'Liquid Cargo Ship', 'Dry Cargo Ship', 'Warship', 'Small Car', 'Bus', 'Cargo Truck',
               'Dump Truck', 'Van', 'Trailer', 'Truck Tractor', 'Excavator', 'Tractor',
               'Baseball Field', 'Basketball Court', 'Football Field', 'Tennis Court',
               'Bridge', 'Intersection', 'Roundabout', 'other-airplane', 'other-ship',
               'other-vehicle']

wordname_16 = ['0', '1']
# wordname_ = ['person', 'person?', 'cyclist', 'people']
wordname = ['0', '1', '2', '3', '4', '05', '08', '09']

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


# wordname_text = ['text']


def split_train_val(srcpath, val_destfile, train_destfile, num):
    imageparent = os.path.join(srcpath, 'images')
    labelpatent = os.path.join(srcpath, 'labelTxt')
    trainfile = os.path.join(train_destfile, 'train_list.txt')
    valfile = os.path.join(val_destfile, 'val_list.txt')
    val_imageparent = os.path.join(val_destfile, 'images')
    val_labelpatent = os.path.join(val_destfile, 'labelTxt')
    train_imageparent = os.path.join(train_destfile, 'images')
    train_labelpatent = os.path.join(train_destfile, 'labelTxt')

    if not os.path.exists(val_imageparent):
        os.makedirs(val_imageparent)

    if not os.path.exists(val_labelpatent):
        os.makedirs(val_labelpatent)

    if not os.path.exists(train_imageparent):
        os.makedirs(train_imageparent)

    if not os.path.exists(train_labelpatent):
        os.makedirs(train_labelpatent)

    trainlist = []
    vallist = []

    list_ = os.listdir(imageparent)
    numlist = random.sample(range(0, len(list_)), int(len(list_) * num))

    for n in range(0, len(list_)):
        if n in numlist:
            vallist.append(list_[n])
        else:
            trainlist.append(list_[n])
    with open(trainfile, 'w') as f_train:
        for line in trainlist:
            f_train.write(line + ' ' + '\n')
            image_path = os.path.join(imageparent, line)
            dest_imagepath = os.path.join(train_imageparent, line)
            label_path = os.path.join(labelpatent, line.replace('tif', 'txt'))
            dest_labelpath = os.path.join(train_labelpatent, line.replace('tif', 'txt'))
            shutil.copyfile(image_path, dest_imagepath)
            shutil.copyfile(label_path, dest_labelpath)
    with open(valfile, 'w') as f_val:
        for line in vallist:
            f_val.write(line + ' ' + '\n')
            image_path = os.path.join(imageparent, line)
            dest_imagepath = os.path.join(val_imageparent, line)
            label_path = os.path.join(labelpatent, line.replace('tif', 'txt'))
            dest_labelpath = os.path.join(val_labelpatent, line.replace('tif', 'txt'))
            shutil.copyfile(image_path, dest_imagepath)
            shutil.copyfile(label_path, dest_labelpath)


qunban_ = ['1_1_u_', '1_2_u_', '1_1_d_', '1_2_d_', '2_1_m_', '2_2_m_', '2_1_d_', '2_2_d_', '2_1_u_', '2_2_u_']


def FAIR1M2COCOTrain_Val(srcpath, destfile, cls_names, dir=None, ext='.jpg'):
    # set difficult to filter '2', '1', or do not filter, set '-1'

    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'labels')

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1

    with open(dir, 'r') as f_in:
        lines = f_in.readlines()
        filelist = [x.strip().split(' ') for x in lines]
    # print('filelist', filelist)

    # filelist = os.listdir(imageparent)

    with open(destfile, 'w') as f_out:
        filenames = filelist
        # pdb.set_trace()
        for file in filenames:
            # if ('_1.jpg' in file) or ('_2.jpg' in file) or ('_8.jpg' in file) or ('_9.jpg' in file):
            #     continue
            # if ('_3.jpg' in file) or ('_4.jpg' in file) or ('_5.jpg' in file) or ('_6.jpg' in file) or ('_7.jpg' in file):
            #     continue

            if 'expand' in file:
                continue
            # basename = util.custombasename(file)
            # print('basename', basename)
            # image_id = int(basename[1:])
            # pdb.set_trace()
            file = file[0]
            imagepath = os.path.join(srcpath, file)
            img = cv2.imread(imagepath)
            try:
                height, width, c = img.shape
            except:
                print(imagepath)
                continue

            # if 'to' in file:
            #     height, width = 2048, 3500
            # else:
            #     height, width = 409, 700
            single_image = {}
            single_image['file_name'] = file
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)
            ## python ./DOTA_devkit/FAIR1M2COCO.py
            # annotations
            annpath = os.path.join(srcpath,
                                   file.replace('.jpg', '.json'))  # .replace('fused_rfnnest_700_wir_6.0_wvi_3.0_', '')
            # objects = util.parse_dota_poly2(annpath)   #ValueError: need at least one array to concatenate
            # print(objects)
            objects = util.parse_dongche_poly2(annpath)  # TypeError: 'NoneType' object is not iterable
            # if len(objects) > 0:
            #     draw_poly_detections(imagepath, objects, "/home/f523/guazai/disk3/shangxiping/redet/luoshuan/" + basename + '.jpg')
            if objects is None:
                image_id = image_id + 1
                continue
            for obj in objects:
                # if obj['difficult'] == difficult:
                #     print('difficult: ', difficult)
                #     continue
                single_obj = {}
                single_obj['area'] = obj['area']
                # if obj['name'] not in wordname:
                #     continue
                try:
                    single_obj['category_id'] = class_name_to_id[obj['name']] + 1
                except:
                    if obj['name'] == '0_1_m_':
                        single_obj['category_id'] = cls_names.index('0') + 1
                    elif obj['name'] in qunban_:
                        single_obj['category_id'] = cls_names.index('1') + 1
                    # elif obj['name'] == '07':
                    #     single_obj['category_id'] = cls_names.index('0') + 1
                    else:
                        continue

                single_obj['segmentation'] = []
                single_obj['segmentation'].append(obj['poly'])
                single_obj['iscrowd'] = 0
                single_obj['difficult'] = obj['difficult']
                xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                                         max(obj['poly'][0::2]), max(obj['poly'][1::2])

                width, height = xmax - xmin, ymax - ymin
                single_obj['bbox'] = xmin, ymin, width, height
                # modified
                single_obj['area'] = width * height
                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1
        json.dump(data_dict, f_out)


def FAIR1M2COCOTrain_Val_single(srcpath, destfile, dir, cls_names, ext='.jpg', key='.style1'):
    # set difficult to filter '2', '1', or do not filter, set '-1'

    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'labelTxt')

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1

    # with open(dir, 'r') as f_in:
    #     lines = f_in.readlines()
    #     filelist = [x.strip().split(' ') for x in lines]
    # print('filelist', filelist)

    filelist = os.listdir(imageparent)
    with open(destfile, 'w') as f_out:
        filenames = filelist
        # pdb.set_trace()
        for file in filenames:
            if key not in file:
                continue
            basename = util.custombasename(file)
            # print('basename', basename)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + ext)
            img = cv2.imread(imagepath)
            try:
                height, width, c = img.shape
            except:
                pdb.set_trace()

            single_image = {}
            single_image['file_name'] = basename + ext
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)
            ## python ./DOTA_devkit/FAIR1M2COCO.py
            # annotations
            annpath = os.path.join(labelparent, basename + '.txt')  # .replace('fused_rfnnest_700_wir_6.0_wvi_3.0_', '')
            # objects = util.parse_dota_poly2(annpath)   #ValueError: need at least one array to concatenate
            # print(objects)
            objects = util.parse_dota_poly2(annpath)  # TypeError: 'NoneType' object is not iterable
            # pdb.set_trace()
            for obj in objects:
                # if obj['difficult'] == difficult:
                #     print('difficult: ', difficult)
                #     continue
                single_obj = {}
                single_obj['area'] = obj['area']
                # if obj['name'] not in wordname:
                #     continue
                single_obj['category_id'] = cls_names.index(obj['name']) + 1
                single_obj['segmentation'] = []
                single_obj['segmentation'].append(obj['poly'])
                single_obj['iscrowd'] = 0
                single_obj['difficult'] = obj['difficult']
                xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                                         max(obj['poly'][0::2]), max(obj['poly'][1::2])

                width, height = xmax - xmin, ymax - ymin
                single_obj['bbox'] = xmin, ymin, width, height
                # modified
                single_obj['area'] = width * height
                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1
        json.dump(data_dict, f_out)


def draw_ploy_image(label_path, image_path):
    label_list = os.listdir(label_path)
    image_list = os.listdir(image_path)
    img_save_path = '/home/f523/guazai/disk3/shangxiping/redet/ReDet-master/kaist/draw/'
    value_txt = '/home/f523/guazai/disk3/shangxiping/redet/ReDet-master/kaist/train/value_txt.txt'
    for label in label_list:
        file = os.path.join(label_path, label)
        # pdb.set_trace()
        with open(file) as f:
            ploys = []
            lines_list = f.readlines()
            for line in lines_list:
                split_line = line.split(' ')
                # pdb.set_trace()
                if len(split_line) == 12:
                    ploy = [int(split_line[1]), int(split_line[2]),
                            int(split_line[1]), int(split_line[2]) + int(split_line[4]),
                            int(split_line[1]) + int(split_line[3]),
                            int(split_line[2]) + int(split_line[4]),
                            int(split_line[1]) + int(split_line[3]), int(split_line[2])
                            ]
                    ploys.append(ploy)

            if len(ploys) > 0:
                draw_poly_detections(os.path.join(image_path, label.replace('.txt', '.jpg')), ploys,
                                     img_save_path + label.replace('.txt', '.jpg'))
                with open(value_txt, 'a') as ft:
                    ft.write(label.replace('.txt', '.jpg') + '\n')


def draw_poly_detections(img, detections, img_save_path):
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
    import cv2
    import random
    img = mmcv.imread(img)
    color_white = (255, 255, 255)
    color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
    for det in detections:
        bbox = det['poly']
        # pdb.set_trace()
        # bbox = list(map(int, bbox))
        # if showStart:
        #     cv2.circle(img, (bbox[0], bbox[1]), 3, (0, 0, 255), -1)
        for i in range(3):
            cv2.line(img, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i + 1) * 2], bbox[(i + 1) * 2 + 1]), color=color,
                     thickness=2, lineType=cv2.LINE_AA)
        cv2.line(img, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=color, thickness=2, lineType=cv2.LINE_AA)
    cv2.imwrite(img_save_path, img)
    pdb.set_trace()


def draw_poly_detections_(img, detections, img_save_path):
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
    import cv2
    import random
    img = mmcv.imread(img)
    color_white = (255, 255, 255)
    color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
    for det in detections:
        bbox = det[:8]
        # pdb.set_trace()
        # bbox = list(map(int, bbox))
        # if showStart:
        #     cv2.circle(img, (bbox[0], bbox[1]), 3, (0, 0, 255), -1)
        for i in range(3):
            cv2.line(img, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i + 1) * 2], bbox[(i + 1) * 2 + 1]), color=color,
                     thickness=2, lineType=cv2.LINE_AA)
        cv2.line(img, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=color, thickness=2, lineType=cv2.LINE_AA)
    cv2.imwrite(img_save_path, img)
    # pdb.set_trace()
    # return img


if __name__ == '__main__':
    # split_train_val(r'/home/f523/guazai/disk3/chx/data/plane',
    #                 r'/home/f523/guazai/disk3/chx/data/sar_plane_val+train/val',
    #                 r'/home/f523/guazai/disk3/chx/data/sar_plane_val+train/train', 0.2)
    # dir = '/home/f523/guazai/disk3/shangxiping/redet/ReDet-master/kaist/test'
    # # #
    # FAIR1M2COCOTrain_Val('/home/f523/guazai/disk3/shangxiping/redet/hrsc',
    #                      '/home/f523/guazai/disk3/shangxiping/redet/hrsc/hrsc_test_fire.json',
    #                      wordname_16
    #                      )  #

    FAIR1M2COCOTrain_Val('/home/f523/guazai/sdb/gzjc/dataset_trains_1.5',
                         '/home/f523/guazai/disk3/shangxiping/data/动车故障/new_sample/defect.json',
                         wordname,
                         dir=r'/home/f523/guazai/disk3/shangxiping/data/动车故障/new_sample/defect_txt.txt'
                         )  #
    # FAIR1M2COCOTrain_Val('/home/f523/guazai/sdb/故障检测/dataset_trains',
    #                      '/home/f523/guazai/disk3/shangxiping/data/动车故障/test_1.json',
    #                      wordname,
    #                      dir=r'/home/f523/guazai/disk3/shangxiping/data/动车故障/test_1.txt'
    #                      )
    # FAIR1M2COCOTrain_Val('/home/f523/guazai/sdb/故障检测/dataset_trains',
    #                      '/home/f523/guazai/disk3/shangxiping/data/动车故障/test_2.json',
    #                      wordname,
    #                      dir=r'/home/f523/guazai/disk3/shangxiping/data/动车故障/test_2.txt'
    #                      )
    # FAIR1M2COCOTrain_Val('/home/f523/guazai/sdb/故障检测/dataset_trains',
    #                      '/home/f523/guazai/disk3/shangxiping/data/动车故障/test_3.json',
    #                      wordname,
    #                      dir=r'/home/f523/guazai/disk3/shangxiping/data/动车故障/test_3.txt'
    #                      )
    # FAIR1M2COCOTrain_Val('/home/f523/guazai/sdb/data/truck/yolo/train',
    #                      '/home/f523/guazai/sdb/data/truck/yolo/train/train.json',
    #                      wordname_16
    #                      )
    # FAIR1M2COCOTrain_Val('/home/f523/guazai/disk3/chx/data/sar_plane/val',
    #                      '/home/f523/guazai/disk3/chx/data/sar_plane/val/sar_val.json',
    #                      wordname)
