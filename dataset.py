import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from collections import namedtuple
from random import shuffle
from profiles import SSD_300


NormRect = namedtuple('NormRect', ['xc', 'yc', 'w', 'h'])


class Rect:
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0 = x0, y0
        self.x1, self.y1 = x1, y1

    def as_array(self):
        return np.array([self.x0, self.y0, self.x1, self.y1])


DefaultBox = namedtuple('DefaultBox', ['rect', 'fm_x', 'fm_y', 'scale', 'fm'])

LabeledObject = namedtuple('LabeledObject', ['label', 'rect'])


class LabeledImage:
    def __init__(self, filepath, size, objects, data=None):
        self.filepath = filepath
        self.size = size
        self.objects = objects
        self.data = data


def nms(detections, threshold):
    rects = np.empty(shape=(len(detections), 4))
    scores = np.empty(shape=(len(detections), ))
    for i, det in enumerate(detections):
        rects[i] = det[2].as_array()
        scores[i] = det[1]

    xmin, ymin, xmax, ymax = rects[:, 0], rects[:, 1], rects[:, 2], rects[:, 3]

    area = (xmax - xmin + 1) * (ymax - ymin + 1)
    idxs = np.argsort(scores)
    pick = []

    while len(idxs) > 0:
        last = idxs.shape[0] - 1
        i = idxs[last]
        idxs = np.delete(idxs, last)
        pick.append(i)

        xxmin = np.maximum(xmin[i], xmin[idxs])
        xxmax = np.minimum(xmax[i], xmax[idxs])
        yymin = np.maximum(ymin[i], ymin[idxs])
        yymax = np.minimum(ymax[i], ymax[idxs])

        w = np.maximum(0, xxmax - xxmin + 1)
        h = np.maximum(0, yymax - yymin + 1)
        intersection = w * h

        union = area[i] + area[idxs] - intersection
        iou = intersection / union
        overlap = iou > threshold
        suppress = np.nonzero(overlap)[0]
        idxs = np.delete(idxs, suppress)

    return pick


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


def norm_rect_to_rect(img_size: tuple, rect: NormRect):
    xc = rect.xc * img_size[0]
    yc = rect.yc * img_size[1]
    w_half = rect.w * img_size[0] / 2
    h_half = rect.h * img_size[1] / 2
    return Rect(int(xc - w_half), int(yc - h_half), int(xc + w_half), int(yc + h_half))


def rect_to_norm_rect(img_size: tuple, rect: Rect):
    xc = (rect.x0 + rect.x1) / 2.0 / img_size[0]
    yc = (rect.y0 + rect.y1) / 2.0 / img_size[1]
    w = float(rect.x1 - rect.x0) / img_size[0]
    h = float(rect.y1 - rect.y0) / img_size[1]
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


def calc_jaccard_overlap(box_as_array, prior_boxes):
    area_prior = (prior_boxes[:, 2] - prior_boxes[:, 0] + 1) * (prior_boxes[:, 3] - prior_boxes[:, 1] + 1)
    area_box = (box_as_array[2] - box_as_array[0] + 1) * (box_as_array[3] - box_as_array[1] + 1)

    xmin = np.maximum(box_as_array[0], prior_boxes[:, 0])
    ymin = np.maximum(box_as_array[1], prior_boxes[:, 1])
    xmax = np.minimum(box_as_array[2], prior_boxes[:, 2])
    ymax = np.minimum(box_as_array[3], prior_boxes[:, 3])

    w = np.maximum(0, xmax - xmin + 1)
    h = np.maximum(0, ymax - ymin + 1)
    intersection = w * h
    return intersection / (area_box + area_prior - intersection)


def calc_overlap(box_as_array, prior_boxes, threshold=0.5):
    overlaps = calc_jaccard_overlap(box_as_array, prior_boxes)
    flags = overlaps > threshold
    nonzero_idxs = np.nonzero(flags)[0]
    return [(i, overlaps[i]) for i in nonzero_idxs]


def encode_location(gt_rect: NormRect, default_box_rect: NormRect):
    # according to  eq.2 on page 5 in the main article https://arxiv.org/pdf/1512.02325.pdf
    return np.array([
        (gt_rect.xc - default_box_rect.xc) / default_box_rect.w,
        (gt_rect.yc - default_box_rect.yc) / default_box_rect.h,
        np.log(gt_rect.w / default_box_rect.w),
        np.log(gt_rect.h / default_box_rect.h),
    ])


def decode_location(det_rect: np.ndarray, default_box_rect: NormRect):
    # inverse transform for encode_location
    return NormRect(
        default_box_rect.xc + det_rect[0] * default_box_rect.w,
        default_box_rect.yc + det_rect[1] * default_box_rect.h,
        default_box_rect.w * np.exp(det_rect[2]),
        default_box_rect.h + np.exp(det_rect[3]),
    )


def predictions_to_bboxes(predictions, default_boxes, confidence_thresh):
    decoded_detections = []
    n_classes = predictions.shape[1] - 4
    bbox_labels = np.argmax(predictions[:, :n_classes - 1], axis=1)
    bbox_confidences = predictions[np.arange(len(bbox_labels)), bbox_labels]
    sorted_detections = np.argsort(bbox_confidences)

    for i in reversed(sorted_detections):
        if bbox_confidences[i] < confidence_thresh:
            break
        norm_rect = decode_location(predictions[i, n_classes:], default_boxes[i].rect)
        decoded_detections.append([bbox_labels[i], bbox_confidences[i], norm_rect])
    return decoded_detections


def net_results_to_bboxes(predictions, default_boxes, img_size,
                          confidence_thresh=0.1, overlap_thresh=0.5, number_thresh=100):
    result = []
    decoded_detections = predictions_to_bboxes(predictions, default_boxes, confidence_thresh)[:number_thresh]
    grouped_by_label_detections = {}
    for det in decoded_detections:
        det[2] = norm_rect_to_rect(img_size, det[2])
        if det[0] in grouped_by_label_detections:
            grouped_by_label_detections[det[0]].append(det)
        else:
            grouped_by_label_detections[det[0]] = [det,]

    for label, detections in grouped_by_label_detections.items():
        pick = nms(detections, overlap_thresh)
        for i in pick:
            result.append(detections[i])
    return result


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
        self.data_list = []     # list of objects of type LabeledImage
        self.label_names = {}   # label list [idx] -> name
        self.label_map = {}     # label map  [name] -> idx


class VocDataset(Dataset):
    def __init__(self, root_directory):
        Dataset.__init__(self)

        self.label_names = ['background',
                            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                            'bus', 'car', 'cat', 'chair', 'cow',
                            'diningtable', 'dog', 'horse', 'motorbike', 'person',
                            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        self.label_map = {key: value for (value, key) in enumerate(self.label_names)}

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

        labeled_file = LabeledImage(filepath, (img_h, img_w), [])
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
        labeled_file.data = cv2.resize(img_raw, self.img_size)
        return labeled_file


class ImageAugmentator:
    def __init__(self):
        pass

    def __call__(self, labeled_file):
        # preprocessing TODO:
        return labeled_file


class LabelGenerator:
    def __init__(self, profile, infinity=True):
        self.infinity = infinity
        self.overlap_thresh = 0.5
        self.img_size = profile.imgsize
        self.default_boxes_rel = get_prior_boxes(profile)
        self.default_boxes_abs = default_boxes_to_array(self.default_boxes_rel, self.img_size)
        self.n_prior_boxes = len(self.default_boxes_rel)

    def get(self, dataset, batch_size, preprocessor):
        n_classes = len(dataset.label_names)
        while True:
            shuffle(dataset.data_list)
            data, labels, gt = [], [], []
            while len(data) < batch_size:
                for labeled_file in dataset.data_list:
                    labeled_image = preprocessor(labeled_file)
                    label = self.__process_labeled_file(labeled_image, n_classes)

                    n_no_object = np.count_nonzero(label[:, n_classes - 1])
                    if n_no_object < label.shape[0]:
                        data.append(labeled_image.data)
                        labels.append(label)
                        gt.append(labeled_image.objects)

            data = np.array(data, dtype=np.float32)
            labels = np.array(labels, dtype=np.float32)
            yield data, labels, gt

            if self.infinity is not True:
                break

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
    lg = LabelGenerator(SSD_300, True)
    loader = ImageLoader(SSD_300.imgsize)
    generator = lg.get(ds, 8, loader)
    for item in generator:
        print(len(item))
    pass
