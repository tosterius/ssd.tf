import os
import cv2
import numpy as np
import pickle
import xml.etree.ElementTree as ET
from collections import namedtuple
import random
from profiles import SSD_300
import utils

# representation of an object on the image
LabeledObject = namedtuple('LabeledObject', ['rect', 'label', 'filepath'])

# representation of an image file in the dataset, size - (h, w), objects - a list of LabeledObjects
LabeledImageFile = namedtuple('LabeledImageFile', ['filepath', 'size', 'objects'])

# similar to LabeledImageFile but 'data' here is a binary data of an image
LabeledImage = namedtuple('LabeledImage', ['data', 'objects'])


def lo_to_abs_rects(img_size, list_of_lo):
    """
    Converts list of LabeledObjects to Rects with scaling
    :param img_size:
    :param list_of_lo:
    :return:
    """
    ret = []
    for lo in list_of_lo:
        rect = utils.norm_rect_to_rect(img_size, lo.rect)
        ret.append([rect, lo.label])
    return ret


def default_boxes_to_array(default_boxes, img_size):
    """
    Scales the default boxes coordinates to their absolute values
    :param default_boxes:
    :param img_size: (h, w)
    :return:
    """
    arr = np.zeros((len(default_boxes), 4))
    for i, box in enumerate(default_boxes):
        # the rect absolute coordinates might be out of img_size
        # it does not matter because we need to compute overlap with gt boxes
        rect = utils.norm_rect_to_rect(img_size, box.rect)
        # [x0 y0 x1 y1]
        arr[i] = rect.as_array()
    return arr


def calc_overlap(box_as_array, prior_boxes, threshold=0.5):
    overlaps = utils.calc_jaccard_overlap(box_as_array, prior_boxes)
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

    def create(self, root_directory, pickled_path=None):

        if pickled_path is not None and os.path.isfile(pickled_path):
            self._load_pkl(pickled_path)
            return

        if root_directory is not None:
            self.init_data_list(root_directory)

        if pickled_path is not None:
            self._dump_pkl(pickled_path)

    def extract(self, k=0.05):
        ds = Dataset()
        ds.label_names = self.label_names
        ds.label_map = self.label_map
        n = int(k * len(self.data_list))
        ds.data_list = self.data_list[:n]
        self.data_list = self.data_list[n:]
        return ds

    def _dump_pkl(self, filepath):
        with open(filepath, "wb") as pickle_out:
            pickle.dump(self.data_list, pickle_out)

    def _load_pkl(self, filepath):
        with open(filepath, "rb") as pickle_in:
            self.data_list = pickle.load(pickle_in)

    def decode_dict(self, d):
        return {self.label_names[k]: v for k, v in d.items()}

    def init_data_list(self, root_directory):
        raise NotImplementedError("This is data initialization point")


class VocDataset(Dataset):
    def __init__(self, root_directory=None, pickled_path=None):
        Dataset.__init__(self)

        self.label_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                            'bus', 'car', 'cat', 'chair', 'cow',
                            'diningtable', 'dog', 'horse', 'motorbike', 'person',
                            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
                            'background']

        self.label_map = {key: value for (value, key) in enumerate(self.label_names)}

        if root_directory is not None:
            self.create(root_directory, pickled_path)

    def init_data_list(self, root_directory):
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

            labeled_file.objects.append(LabeledObject(label=label, rect=utils.NormRect(xc, yc, w, h), filepath=filepath))

        self.data_list.append(labeled_file)


class GaussianNoiser:
    def __call__(self, labeled_file):
        data = labeled_file.data
        k = random.choice([3, 5])
        blured = cv2.GaussianBlur(data, (k, k), 0)
        return LabeledImage(blured, labeled_file.objects)


class SpeckleNoiser:
    def __call__(self, labeled_file):
        data = labeled_file.data
        noise = np.random.normal(0, 0.05, data.shape)
        out = data + data * noise
        out = np.clip(out, 0, 255)
        return LabeledImage(out, labeled_file.objects)


class ImageLoader:
    def __init__(self, img_size):
        self.img_size = img_size
        self.processors = []

    def __call__(self, labeled_file):
        img_raw = cv2.imread(labeled_file.filepath, cv2.IMREAD_COLOR)
        data = cv2.resize(img_raw, self.img_size).astype(np.float)
        objects = labeled_file.objects
        ret = LabeledImage(data, objects)
        for processor in self.processors:
            if random.choice([True, False]):
                ret = processor(ret)
        return ret


class LabelGenerator:
    """
    A LabelGenerator object processes the records from dataset
    and generates input data for the SSD net
    """
    def __init__(self, profile):
        self.overlap_thresh = 0.5
        self.img_size = profile.imgsize
        self.default_boxes_rel = utils.get_default_boxes(profile)
        self.default_boxes_abs = default_boxes_to_array(self.default_boxes_rel, self.img_size)
        self.n_prior_boxes = len(self.default_boxes_rel)

    def get(self, dataset, batch_size, preprocessor):
        def generator(ds, batch_size, preprocessor):
            random.shuffle(ds.data_list)
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
            rect = utils.norm_rect_to_rect(self.img_size, labeled_object.rect)  # debug
            overlaps = calc_overlap(rect.as_array(), self.default_boxes_abs, self.overlap_thresh)

            for id, score in overlaps:
                if id in tmp_map and tmp_map[id] >= score:
                    continue
                tmp_map[id] = score
                label[id, :n_classes] = 0.0
                label[id, labeled_object.label] = 1.0
                label[id, n_classes:] = utils.encode_location(labeled_object.rect, self.default_boxes_rel[id].rect)

        return label


if __name__ == '__main__':

    ds = VocDataset('/data/Workspace/data/VOCDebug')
    lg = LabelGenerator(SSD_300)
    loader = ImageLoader(SSD_300.imgsize)
    generator = lg.get(ds, 8, loader)
    for item in generator:
        print(len(item))
    pass
