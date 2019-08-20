import os
import numpy as np
import xml.etree.ElementTree as ET
from collections import namedtuple
from random import shuffle


LabeledObject = namedtuple('LabeledObject', ['label', 'xc', 'yc', 'w', 'h'])
LabeledImage = namedtuple('LabeledImage', ['filepath', 'size', 'objects'])


class Rect:
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0 = x0, y0
        self.x1, self.y1 = x1, y1

    def as_array(self):
        return np.array([self.x0, self.y0, self.x1, self.y1])


NormRect = namedtuple('NormRect', ['xc', 'yc', 'w', 'h'])

DefaultBox = namedtuple('DefaultBox', ['rect', 'fm_x', 'fm_y', 'scale', 'fm'])


def norm_rect_to_rect(imgsize: tuple, rect: NormRect):
    xc = rect.xc * imgsize[0]
    yc = rect.yc * imgsize[1]
    w_half = rect.w * imgsize[0] / 2
    h_half = rect.h * imgsize[1] / 2
    return Rect(int(xc - w_half), int(yc - h_half), int(xc + w_half), int(yc + h_half))


def rect_to_norm_rect(imgsize: tuple, rect: Rect):
    xc = (rect.x0 + rect.x1) / 2.0 / imgsize[0]
    yc = (rect.y0 + rect.y1) / 2.0 / imgsize[1]
    w = float(rect.x1 - rect.x0) / imgsize[0]
    h = float(rect.y1 - rect.y0) / imgsize[1]
    return NormRect(xc, yc, w, h)


def default_boxes_to_array(default_boxes, img_size):
    arr = np.zeros((len(default_boxes), 4))
    for i, box in enumerate(default_boxes):
        rect = norm_rect_to_rect(img_size, box.rect)
        arr[i] = rect.as_array()
    return arr


def calc_jaccard_overlap(box, prior_boxes):
    area_prior = (prior_boxes[:, 2] - prior_boxes[:, 0] + 1) * (prior_boxes[:, 3] - prior_boxes[:, 1] + 1)
    area_box = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

    xmin = np.maximum(box[0], prior_boxes[:, 0])
    ymin = np.maximum(box[1], prior_boxes[:, 1])
    xmax = np.minimum(box[2], prior_boxes[:, 2])
    ymax = np.minimum(box[3], prior_boxes[:, 3])

    w = np.maximum(0, xmax - xmin + 1)
    h = np.maximum(0, ymax - ymin + 1)
    intersection = w * h
    return intersection / (area_box + area_prior - intersection)


def calc_overlap(box, prior_boxes, threshold=0.5):
    pass


class Dataset(object):
    def __init__(self):
        self.data = []
        self.label_map = None

    def split(self, fractions=[0.99, 0.01]):
        ret_datasets = []
        shuffle(self.data)
        n = len(self.data)
        counter = 0
        for frac in fractions:
            portion = int(n * frac)
            ds = Dataset()
            ret_datasets.append(ds)
            ds.data = self.data[counter:counter+portion]
            ds.label_map = self.label_map.copy()
            counter += portion
        return ret_datasets

    def extend(self, other):
        if other.label_map != self.label_map:
            raise RuntimeError("Datasets must have identical label map")
        self.data += other.data
        return self

    def shuffle(self):
        shuffle(self.data)

    def batch(self, batchsize):
        n = len(self.data)
        for i in range(0, n, batchsize):
            last = min(i + batchsize, n)
            portion = self.data[i:last]
            yield portion




class VocDataset(Dataset):
    def __init__(self, root_directory=None):
        Dataset.__init__(self)
        self.label_map = {'background': 0,
                          'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5,
                          'bus': 6, 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10,
                          'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15,
                          'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20}

        if root_directory is not None:
            self.init(root_directory)

    def init(self, root_directory):
        annotations_root = os.path.join(root_directory, 'Annotations')
        images_root = os.path.join(root_directory, 'Images')
        annotations_files = os.listdir(annotations_root)

        for filename in annotations_files:
            xmlpath = os.path.join(annotations_root, filename)
            try:
                self.__parse_xml(images_root, xmlpath)
            except Exception as e:
                print(str(e))

    def __parse_xml(self, images_root, filepath):
        tree = ET.parse(filepath)
        root = tree.getroot()

        filename = root.find('filename').text
        filepath = os.path.join(images_root, filename)
        size = root.find('size')
        img_w = int(size.find('width').text)
        img_h = int(size.find('height').text)

        labeled_image = LabeledImage(filepath, (img_h, img_w), [])
        for o in root.iter('object'):
            label = self.label_map[o.find('name').text]
            bbox = o.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            xc = (xmin + xmax) / 2.0 / img_w
            yc = (ymin + ymax) / 2.0 / img_h
            w = float(xmax - xmin) / img_w
            h = float(ymax - ymin) / img_h

            labeled_image.objects.append(LabeledObject(label=label, xc=xc, yc=yc, w=w, h=h))

        self.data.append(labeled_image)


class DataGenerator:
    def __init__(self, dataset, batch_size):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        pass


def get_train_val_data_generator(dataset):
    pass


if __name__ == '__main__':
    ds1 = VocDataset()
    #ds1 = ds1.extend(VocDataset('/home/arthur/Workspace/projects/github/ssd.tf/VOC2007'))
    #ds1 = ds1.extend(VocDataset('/home/arthur/Workspace/projects/github/ssd.tf/VOC2008'))

    ds1 = ds1.extend(VocDataset('/data/Workspace/data/VOCtest_06-Nov-2007/VOC2007'))
    ds1 = ds1.extend(VocDataset('/data/Workspace/data/VOCdevkit/VOC2012'))

    ds2, ds3 = ds1.split()

    for portion in ds2.batch(4):
        print(len(portion))
    pass