import os
import cv2
import numpy as np
from collections import namedtuple
from collections import defaultdict, Counter


NormRect = namedtuple('NormRect', ['xc', 'yc', 'w', 'h'])


class Rect:
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0 = x0, y0
        self.x1, self.y1 = x1, y1

    def __repr__(self):
        return "Rect({}, {}, {}, {})".format(self.x0, self.y0, self.x1, self.y1)

    def as_array(self, dtype=None):
        return np.array([self.x0, self.y0, self.x1, self.y1], dtype=dtype)


DefaultBox = namedtuple('DefaultBox', ['rect', 'fm_x', 'fm_y', 'scale', 'fm'])

DetectedObject = namedtuple('DetectedObject', ['rect', 'label', 'score'])


def norm_rect_to_rect(img_size: tuple, rect: NormRect):
    """
    Safely converts normalized rect into rect with absolute coordinates
    :param img_size:
    :param rect:
    :return:
    """
    xc = rect.xc * img_size[1]
    yc = rect.yc * img_size[0]
    w_half = rect.w * img_size[1] / 2.0
    h_half = rect.h * img_size[0] / 2.0
    x0 = min(max(0, xc - w_half), img_size[1] - 1)
    x1 = max(x0, min(img_size[1] - 1, xc + w_half))
    y0 = min(max(0, yc - h_half), img_size[0] - 1)
    y1 = max(y0, min(img_size[0] - 1, yc + h_half))
    return Rect(int(x0), int(y0), int(x1), int(y1))


def rect_to_norm_rect(img_size: tuple, rect: Rect):
    """
    Normalizes a rect
    :param img_size:
    :param rect:
    :return:
    """
    xc = (rect.x0 + rect.x1) / 2.0 / img_size[1]
    yc = (rect.y0 + rect.y1) / 2.0 / img_size[0]
    w = float(rect.x1 - rect.x0) / img_size[1]
    h = float(rect.y1 - rect.y0) / img_size[0]
    return NormRect(xc, yc, w, h)


def get_default_boxes(profile):
    """
    Computes sizes of default bounding boxes for all scales.
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


def nms(detections, threshold):
    """
    :param detections: [[norm_rects, label, score, rects], ..]
    :param threshold: float < 1.0
    :return:
    """
    rects = np.empty(shape=(len(detections), 4))
    scores = np.empty(shape=(len(detections),))
    for i, det in enumerate(detections):
        rects[i] = det[-1].as_array()
        scores[i] = det[-2]

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


def calc_jaccard_overlap(box_as_array, default_boxes):
    # For each default box calculates IOU
    area_prior = (default_boxes[:, 2] - default_boxes[:, 0] + 1) * (default_boxes[:, 3] - default_boxes[:, 1] + 1)
    area_box = (box_as_array[2] - box_as_array[0] + 1) * (box_as_array[3] - box_as_array[1] + 1)

    xmin = np.maximum(box_as_array[0], default_boxes[:, 0])
    ymin = np.maximum(box_as_array[1], default_boxes[:, 1])
    xmax = np.minimum(box_as_array[2], default_boxes[:, 2])
    ymax = np.minimum(box_as_array[3], default_boxes[:, 3])

    w = np.maximum(0, xmax - xmin + 1)
    h = np.maximum(0, ymax - ymin + 1)
    intersection = w * h
    return intersection / (area_box + area_prior - intersection)


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
        default_box_rect.h * np.exp(det_rect[3]),
    )


def net_predictions_to_bboxes(predictions, default_boxes, confidence_thresh, number_thresh):
    """
    Converts network prediction to human understandable format
    :param predictions: network output of sizes (num_of_default_boxes, num_of_classes + 1(bg) + 4(rect))
    :param default_boxes:
    :param confidence_thresh:
    :param number_thresh:
    :return:  [NormRect, label, score]
    """
    decoded_detections = []
    n_classes = predictions.shape[1] - 4
    bbox_labels = np.argmax(predictions[:, :n_classes - 1], axis=1)
    bbox_confidences = predictions[np.arange(len(bbox_labels)), bbox_labels]
    sorted_detections = np.argsort(bbox_confidences)[-number_thresh:]

    for i in reversed(sorted_detections):
        if bbox_confidences[i] < confidence_thresh:
            break
        norm_rect = decode_location(predictions[i, n_classes:], default_boxes[i].rect)
        decoded_detections.append([norm_rect, bbox_labels[i], bbox_confidences[i]])
    return decoded_detections


def get_filtered_result_bboxes(results, default_boxes, img_size,
                               confidence_thresh=0.02, overlap_thresh=0.45, number_thresh=100):
    """
    Converts network output to human understandable format and filters it on the basis of thresholds
    :param results: network output of sizes (batch_size, num_of_default_boxes, num_of_classes + 1(bg) + 4(rect))
    :param default_boxes: default boxes produced by get_default_boxes
    :param img_size: network input image size
    :param confidence_thresh:
    :param overlap_thresh: nms threshold
    :param number_thresh: is used to filter top K=number_thresh boxes with the highest confidence score
    :return:
    """
    result = []

    decoded_detections = net_predictions_to_bboxes(results, default_boxes, confidence_thresh, number_thresh)
    grouped_by_label_detections = {}
    for det in decoded_detections:
        # det here is like [NormRect, label, score]
        det.append(norm_rect_to_rect(img_size, det[0]))
        if det[1] in grouped_by_label_detections:
            grouped_by_label_detections[det[1]].append(det)
        else:
            grouped_by_label_detections[det[1]] = [det, ]

    for label, detections in grouped_by_label_detections.items():
        pick = nms(detections, overlap_thresh)
        for i in pick:
            result.append(detections[i])
    return result


# -------------------------------------- precision calculator ----------------------------------------------------------

class PrecisionMetric:
    def __init__(self):
        self.overlap_thresh = 0.5
        self.gt_samples = []
        self.det_rects = defaultdict(list)
        self.det_confs = defaultdict(list)
        self.det_sample_ids = defaultdict(list)

    def add(self, gt_sample, detections):
        self.gt_samples.append(gt_sample)
        for _, label, score, rect in detections:
            self.det_rects[label].append(rect.as_array())
            self.det_confs[label].append(score)
            self.det_sample_ids[label].append(len(self.gt_samples) - 1)

    def reset(self):
        self.det_rects.clear()
        self.det_confs.clear()
        self.det_sample_ids.clear()
        self.gt_samples.clear()

    def calc_and_reset(self):
        precisions, mean = self.calc()
        self.reset()
        return precisions, mean

    def calc(self):
        mean = 0.0
        precisions = {}
        label_counter = Counter()
        gt_map = defaultdict(dict)

        for sample_id, boxes in enumerate(self.gt_samples):
            gt_rects_by_class = defaultdict(list)
            for gt_rect, gt_label in boxes:
                label_counter[gt_label] += 1
                gt_rects_by_class[gt_label].append(gt_rect.as_array(np.float32))

            for gt_label, gt_rects in gt_rects_by_class.items():
                arr = np.array(gt_rects)
                match = np.zeros((len(gt_rects)), dtype=np.bool)
                gt_map[gt_label][sample_id] = (arr, match)

        for gt_label, matches_by_sample_id in gt_map.items():
            rects = np.array(self.det_rects[gt_label], dtype=np.float32)
            n_rects = rects.shape[0]
            if n_rects == 0:
                precisions[gt_label] = 0.0
                continue

            confs = np.array(self.det_confs[gt_label], dtype=np.float32)
            sample_ids = np.array(self.det_sample_ids[gt_label], dtype=np.int)
            idxs_max = np.argsort(-confs)
            rects = rects[idxs_max]
            sample_ids = sample_ids[idxs_max]

            tps, fps = np.zeros(n_rects), np.zeros(n_rects)

            for i in range(n_rects):
                sample_id = sample_ids[i]
                if sample_id not in matches_by_sample_id:
                    fps[i] = 1
                    continue

                gt_box, matched = matches_by_sample_id[sample_id]
                det_box = rects[i]
                iou = calc_jaccard_overlap(det_box, gt_box)
                max_idx = np.argmax(iou)

                if matched[max_idx]:
                    fps[i] = 1
                    continue

                if iou[max_idx] < self.overlap_thresh:
                    fps[i] = 1
                    continue

                tps[i] = 1
                matched[max_idx] = True

            fps = np.cumsum(fps)
            tps = np.cumsum(tps)
            recall = tps / label_counter[gt_label]
            prec = tps / (tps + fps)
            ap = 0
            for r_tilde in np.arange(0, 1.1, 0.1):
                prec_rec = prec[recall >= r_tilde]
                if prec_rec.size > 0:
                    ap += np.amax(prec_rec)

            ap /= 11.
            precisions[gt_label] = ap
            mean += ap
        n_classes = len(precisions)
        mean = 0 if n_classes == 0 else mean / n_classes
        return precisions, mean


# --------------------------------------------drawing ------------------------------------------------------------------

def draw_rect(img, rect, label, score, rect_color, font_color):
    text = str(label) if score is None else '{}: {:.2}'.format(str(label), score)
    text_width = 10 * len(text)
    rect_text = Rect(rect.x0, rect.y0 - 20, rect.x0 + text_width, rect.y0)
    cv2.rectangle(img, (int(rect.x0), int(rect.y0)), (int(rect.x1), int(rect.y1)), rect_color, 1)
    cv2.rectangle(img, (int(rect_text.x0), int(rect_text.y0)), (int(rect_text.x1), int(rect_text.y1)), rect_color,
                  -1)
    pos_text = (rect.x0 + 2, rect.y0 - 5)
    cv2.putText(img, text, pos_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color)
    return img


def draw_detections(destpath, filepath, detections, label_names=None, n=5):
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    sorted(detections, key=lambda d: d[-2])  # sort by score
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (100, 0, 255), (200, 150, 200)]
    text_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 200, 0)]
    n = min(n, len(detections))
    for i in range(n):
        det = detections[i]
        rect = norm_rect_to_rect((h, w), det[0])
        label = det[1] if label_names is None else label_names[det[1]]
        draw_rect(img, rect, label, det[-2], colors[i], text_colors[i])
    cv2.imwrite(destpath, img)


# ----------------------------------------------- fs utils -------------------------------------------------------------

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def list_files(directory_path, exts):
    from fnmatch import fnmatch

    def check_extension(file_path, exts):
        for ext in exts:
            curr_ext = os.path.splitext(file_path)[1]
            if fnmatch(curr_ext, ext):
                return True
        return False

    file_list = list()
    for root, subdirs, files in os.walk(directory_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            if check_extension(file_path, exts):
                file_list.append(file_path)
    return file_list
