import xml.etree.ElementTree as ET
import os
import argparse
import shutil
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, help='input path')
parser.add_argument('--output_path', type=str, help='output path')

opt = parser.parse_args()

# CLASSES = ('car', 'bus', 'truck', 'van', 'feright_car')
CLASSES = ('car', 'bicycle', 'person')

def parse_poly_FILR(filename):
    """
        parse ground truth in the format:
        [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    lines = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bnd_box = obj.find('polygon')
        if bnd_box:
            x1 = bnd_box.find('x1').text
            y1 = bnd_box.find('y1').text
            x2 = bnd_box.find('x2').text
            y2 = bnd_box.find('y2').text
            x3 = bnd_box.find('x3').text
            y3 = bnd_box.find('y3').text
            x4 = bnd_box.find('x4').text
            y4 = bnd_box.find('y4').text
            lines.append([x1, y1, x2, y2, x3, y3, x4, y4, name])
        else:
            bnd_box = obj.find('bndbox')
            if bnd_box:
                x1 = bnd_box.find('xmin').text
                y1 = bnd_box.find('ymin').text
                x2 = bnd_box.find('xmax').text
                y2 = bnd_box.find('ymax').text
                lines.append([x1, y1, x1, y2, x2, y2, x2, y1, name])
        if obj.find('point'):
            continue

        # print(points)

    return lines

def parse_poly(filename):
    """
        parse ground truth in the format:
        [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    lines = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == 'feright car':
            name = 'feright_car'
        if name == 'feright':
            name = 'feright_car'
        bnd_box = obj.find('polygon')
        if bnd_box:
            x1 = bnd_box.find('x1').text
            y1 = bnd_box.find('y1').text
            x2 = bnd_box.find('x2').text
            y2 = bnd_box.find('y2').text
            x3 = bnd_box.find('x3').text
            y3 = bnd_box.find('y3').text
            x4 = bnd_box.find('x4').text
            y4 = bnd_box.find('y4').text
            lines.append([x1, y1, x2, y2, x3, y3, x4, y4, name])
        else:
            bnd_box = obj.find('bndbox')
            if bnd_box:
                x1 = bnd_box.find('xmin').text
                y1 = bnd_box.find('ymin').text
                x2 = bnd_box.find('xmax').text
                y2 = bnd_box.find('ymax').text
                lines.append([x1, y1, x1, y2, x2, y2, x2, y1, name])
        if obj.find('point'):
            continue

        # print(points)

    return lines

def parse_poly_hrsc(filename):
    """
        parse ground truth in the format:
        [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    tree = ET.parse(filename)
    root = tree.getroot().find('HRSC_Objects')
    lines = []
    # pdb.set_trace()
    for obj in root.findall('HRSC_Object'):
        name = 'ship'

        # bnd_box = obj.find('HRSC_Object')

        if obj:
            x1 = obj.find('box_xmin').text
            y1 = obj.find('box_ymin').text
            x2 = obj.find('box_xmax').text
            y2 = obj.find('box_ymax').text
            lines.append([x1, y1, x1, y2, x2, y2, x2, y1, name])
        # pdb.set_trace()
    return lines

if __name__ == "__main__":
    input_path = opt.input_path
    output_path = opt.output_path

    xml_path_list = os.listdir(input_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for xml_file in xml_path_list:
        xml_path = os.path.join(input_path, xml_file)
        objects = parse_poly_FILR(xml_path)
        txt_path = os.path.join(output_path, xml_file.replace('.xml', '.txt'))
        txt_file = open(txt_path, 'w')
        for ele in objects:
            txt_file.write(','.join(ele) + '\n')
        # pdb.set_trace()
        txt_file.close()
        # pdb.set_trace()
