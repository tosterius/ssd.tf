import os
import cv2
import numpy as np
import pickle
import xml.etree.ElementTree as ET
from collections import namedtuple
from random import shuffle
from profiles import SSD_300

from utils import NormRect, norm_rect_to_rect, calc_jaccard_overlap, encode_location

DefaultBox = namedtuple('DefaultBox', ['rect', 'fm_x', 'fm_y', 'scale', 'fm'])

LabeledObject = namedtuple('LabeledObject', ['rect', 'label'])


def lo_to_abs_rects(img_size, list_of_lo):
    ret = []
    for lo in list_of_lo:
        rect = norm_rect_to_rect(img_size, lo.rect)
        ret.append([rect, lo.label])
    return ret


class LabeledImageFile:
    def __init__(self, filepath, size, objects):
        self.filepath = filepath
        self.size = size
        self.objects = objects


LabeledImage = namedtuple('LabeledImage', ['data', 'objects'])


def get_prior_boxes(profile):
    """
    Get sizes of default bounding boxes for all scales.
    See https://arxiv.org/pdf/1512.02325.pdf page 6
    :param profile:
    :return:
    """
    n_maps = len(profile.maps)
    box_sizes = []
    for i in range(n_maps):
        scale = profile.maps[i].scale
        aspect_ratios = profile.maps[i].ratios
        sizes_for_aspects = []
        for acpect_ratio in aspect_ratios:
            sqrt_ar = np.sqrt(acpect_ratio)
            w = scale * sqrt_ar
            h = scale / sqrt_ar
            sizes_for_aspects.append((w, h))

        # additional default box for the aspect ratio of 1:
        add_scale_one = profile.maps[i + 1].scale if i < n_maps - 1 else profile.max_scale
        s_prime = np.sqrt(scale * add_scale_one)
        sizes_for_aspects.append((s_prime, s_prime))
        box_sizes.append(sizes_for_aspects)

    default_boxes = []
    for k in range(n_maps):
        # f_k is the size of the k-th feature map
        f_k = profile.maps[k].size[0]
        s = profile.maps[k].scale
        for (w, h) in box_sizes[k]:
            for j in range(f_k):
                yc = (j + 0.5) / f_k
                for i in range(f_k):
                    xc = (i + 0.5) / f_k
                    default_boxes.append(DefaultBox(NormRect(xc, yc, w, h), i, j, s, k))
    return default_boxes


def default_boxes_to_array(default_boxes, img_size):
    arr = np.zeros((len(default_boxes), 4))
    for i, box in enumerate(default_boxes):
        # the rect absolute coordinates might be out of img_size
        # it does not matter because we need to compute overlap with gt boxes
        rect = norm_rect_to_rect(img_size, box.rect)
        # [x0 y0 x1 y1]
        arr[i] = rect.as_array()
    return arr


def calc_overlap(box_as_array, prior_boxes, threshold=0.5):
    overlaps = calc_jaccard_overlap(box_as_array, prior_boxes)
    flags = overlaps > threshold
    nonzero_idxs = np.nonzero(flags)[0]
    return [(i, overlaps[i]) for i in nonzero_idxs]


def batch_iterator(data_list, batch_size):
    """
    Returns iterator through data_list in batch of batch_size
    :param data_list:
    :param batch_size:
    :return:
    """
    n = len(data_list)
    for i in range(0, n, batch_size):
        last = min(i + batch_size, n)
        portion = data_list[i:last]
        yield portion


def split(data_list, fractions=[0.99, 0.01]):
    """
    Splits up data_list into batches
    :param data_list:
    :param fractions:
    :return:
    """
    ret_data_lists = []
    n = len(data_list)
    counter = 0
    for frac in fractions:
        portion_size = int(n * frac)
        portion = data_list[counter:counter + portion_size]
        ret_data_lists.append(portion)
        counter += portion_size
    return ret_data_lists


class Dataset(object):
    def __init__(self):
        self.data_list = []  # list of objects of type LabeledImage
        self.label_names = {}  # label list [idx] -> name
        self.label_map = {}  # label map  [name] -> idx


class VocDataset(Dataset):
    def __init__(self, root_directory, pickled_path=None):
        Dataset.__init__(self)

        self.label_names = ['background',
                            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                            'bus', 'car', 'cat', 'chair', 'cow',
                            'diningtable', 'dog', 'horse', 'motorbike', 'person',
                            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        self.label_map = {key: value for (value, key) in enumerate(self.label_names)}

        if pickled_path is not None and os.path.isfile(pickled_path):
            self.load(pickled_path)
            return

        if root_directory is not None:
            self.init(root_directory)

        if pickled_path is not None:
            self.dump(pickled_path)

    def dump(self, filepath):
        with open(filepath, "wb") as pickle_out:
            pickle.dump(self.data_list, pickle_out)

    def load(self, filepath):
        with open(filepath, "rb") as pickle_in:
            self.data_list = pickle.load(pickle_in)

    def init(self, root_directory):
        annotations_root = os.path.join(root_directory, 'Annotations')
        images_root = os.path.join(root_directory, 'JPEGImages')
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
        if not os.path.exists(filepath):
            raise FileNotFoundError('File "' + filepath + '" does not exist')
        size = root.find('size')
        img_w = int(size.find('width').text)
        img_h = int(size.find('height').text)

        labeled_file = LabeledImageFile(filepath, (img_h, img_w), [])
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

            labeled_file.objects.append(LabeledObject(label=label, rect=NormRect(xc, yc, w, h)))

        self.data_list.append(labeled_file)


class ImageLoader:
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, labeled_file):
        img_raw = cv2.imread(labeled_file.filepath, cv2.IMREAD_COLOR)
        data = cv2.resize(img_raw, self.img_size).astype(np.float)
        objects = labeled_file.objects
        return LabeledImage(data, objects)


class ImageAugmentator:
    def __init__(self):
        pass

    def __call__(self, labeled_file):
        # preprocessing TODO:
        return labeled_file


class LabelGenerator:
    def __init__(self, profile):
        self.overlap_thresh = 0.5
        self.img_size = profile.imgsize
        self.default_boxes_rel = get_prior_boxes(profile)
        self.default_boxes_abs = default_boxes_to_array(self.default_boxes_rel, self.img_size)
        self.n_prior_boxes = len(self.default_boxes_rel)

    def get(self, dataset, batch_size, preprocessor):
        def generator(ds, batch_size, preprocessor):
            shuffle(ds.data_list)
            n_classes = len(ds.label_names)
            data, labels, gt = [], [], []
            n = len(ds.data_list)
            for i, labeled_file in enumerate(ds.data_list):
                labeled_image = preprocessor(labeled_file)
                label = self.__process_labeled_file(labeled_image, n_classes)
                n_no_object = np.count_nonzero(label[:, n_classes - 1])
                if n_no_object < label.shape[0]:
                    data.append(labeled_image.data)
                    labels.append(label)
                    gt.append(labeled_image.objects)
                if len(data) >= batch_size or i == n - 1 and len(data) > 0:
                    data_arr = np.array(data, dtype=np.float32)
                    labels_arr = np.array(labels, dtype=np.float32)
                    gt_ret = gt[:]
                    data, labels, gt = [], [], []
                    yield data_arr, labels_arr, gt_ret

        return generator(dataset, batch_size, preprocessor)

    def __process_labeled_file(self, labeled_file, n_classes):
        label_dim = n_classes + 4
        label = np.zeros((self.n_prior_boxes, label_dim), dtype=np.float32)
        label[:, n_classes - 1] = 1

        tmp_map = {}
        for labeled_object in labeled_file.objects:
            rect = norm_rect_to_rect(self.img_size, labeled_object.rect)  # debug
            overlaps = calc_overlap(rect.as_array(), self.default_boxes_abs, self.overlap_thresh)

            for id, score in overlaps:
                if id in tmp_map and tmp_map[id] >= score:
                    continue
                tmp_map[id] = score
                label[id, :n_classes] = 0.0
                label[id, labeled_object.label] = 1.0
                label[id, n_classes:] = encode_location(labeled_object.rect, self.default_boxes_rel[id].rect)

        return label


if __name__ == '__main__':

    ds = VocDataset('/data/Workspace/data/VOCDebug')
    lg = LabelGenerator(SSD_300)
    loader = ImageLoader(SSD_300.imgsize)
    generator = lg.get(ds, 8, loader)
    for item in generator:
        print(len(item))
    pass
