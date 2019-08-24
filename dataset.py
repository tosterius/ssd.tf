import os
import numpy as np
import xml.etree.ElementTree as ET
from collections import namedtuple
from random import shuffle
from profiles import voc_ssd_300


NormRect = namedtuple('NormRect', ['xc', 'yc', 'w', 'h'])


class Rect:
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0 = x0, y0
        self.x1, self.y1 = x1, y1

    def as_array(self):
        return np.array([self.x0, self.y0, self.x1, self.y1])


LabeledObject = namedtuple('LabeledObject', ['label', 'rect'])
LabeledImage = namedtuple('LabeledImage', ['filepath', 'size', 'objects'])

DefaultBox = namedtuple('DefaultBox', ['rect', 'fm_x', 'fm_y', 'scale', 'fm'])


def get_prior_boxes(profile):
    """
    Get sizes of default bounding boxes for every scale.
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
        # the rect absolute coordinates might be out of img_size
        # it does not matter because we need to compute overlap with gt boxes
        rect = norm_rect_to_rect(img_size, box.rect)
        # [x0 y0 x1 y1]
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
    overlaps = calc_jaccard_overlap(box, prior_boxes)
    flags = overlaps > threshold
    nonzero_idxs = np.nonzero(flags)[0]
    return [(i, overlaps[i]) for i in nonzero_idxs]


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

            labeled_image.objects.append(LabeledObject(label=label, rect=NormRect(xc, yc, w, h)))

        self.data.append(labeled_image)


def encode_location(gt_rect: NormRect, default_box_rect: NormRect):
    return np.array([
        gt_rect.xc - default_box_rect.xc,
        gt_rect.yc - default_box_rect.yc,
        gt_rect.w / default_box_rect.w,
        gt_rect.h / default_box_rect.h,
    ])


def decode_location(det_rect: np.ndarray, default_box: NormRect):
    return NormRect(
        default_box.xc + det_rect[0],
        default_box.yc + det_rect[1],
        default_box.w * det_rect[2],
        default_box.h + det_rect[3],
    )

class ImageLoader:
    def __call__(self, labeled_image):
        pass

class LabelGenerator:
    def __init__(self, profile, infinity=True):
        self.infinity = infinity
        self.overlap_thresh = 0.5
        self.n_classes = profile.n_classes
        self.label_dim = self.n_classes + 4
        self.imgsize = profile.imgsize
        self.default_boxes_rel = get_prior_boxes(profile)
        self.default_boxes_abs = default_boxes_to_array(self.default_boxes_rel, self.imgsize)
        self.n_prior_boxes = len(self.default_boxes_rel)

    def get(self, dataset, batchsize):
        while True:
            dataset.shuffle()
            raw_batches = dataset.batch(batchsize)
            for raw_batch in raw_batches:
                ret = []
                for labeled_image in raw_batch:
                    ret.append(self.process_labeled_image(labeled_image))
                yield ret
            if self.infinity is not True:
                break

    def process_labeled_image(self, labeled_image):
        label = np.zeros((self.n_prior_boxes, self.label_dim), dtype=np.float32)

        map = {}
        for labeled_object in labeled_image.objects:
            rect = norm_rect_to_rect(self.imgsize, labeled_object.rect)  # debug
            overlaps = calc_overlap(rect.as_array(), self.default_boxes_abs, self.overlap_thresh)

            for id, score in overlaps:
                if id in map and map[id] >= score:
                    continue
                map[id] = score
                label[id, :self.n_classes + 1] = 0.0
                label[id, labeled_object.label] = 1.0
                label[id, self.n_classes + 1:] = encode_location(labeled_object.rect, self.default_boxes_rel[id].rect)

            #best_overlap = max(overlaps, key=lambda x: x[1])
        return label


def get_train_val_data_generator(dataset):
    pass


if __name__ == '__main__':

    ds1 = VocDataset()
    #ds1 = ds1.extend(VocDataset('/home/arthur/Workspace/projects/github/ssd.tf/VOC2007'))
    #ds1 = ds1.extend(VocDataset('/home/arthur/Workspace/projects/github/ssd.tf/VOC2008'))

    ds = VocDataset('/data/Workspace/data/VOCDebug')
    lg = LabelGenerator(voc_ssd_300, True)
    generator = lg.get(ds, 8)
    for item in generator:
        print(len(item))
    pass