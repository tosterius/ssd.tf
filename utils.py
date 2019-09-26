import numpy as np
from collections import namedtuple

NormRect = namedtuple('NormRect', ['xc', 'yc', 'w', 'h'])


class Rect:
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0 = x0, y0
        self.x1, self.y1 = x1, y1

    def as_array(self):
        return np.array([self.x0, self.y0, self.x1, self.y1])


DetectedObject = namedtuple('DetectedObject', ['rect', 'label', 'score'])


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


def nms(detections, threshold):
    """

    :param detections: [[label, score, norm_rects], ..]
    :param threshold: float < 1.0
    :return:
    """
    rects = np.empty(shape=(len(detections), 4))
    scores = np.empty(shape=(len(detections),))
    for i, det in enumerate(detections):
        rects[i] = det[0].as_array()
        scores[i] = det[2]

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


def nn_predictions_to_bboxes(predictions, default_boxes, confidence_thresh):
    decoded_detections = []
    n_classes = predictions.shape[1] - 4
    bbox_labels = np.argmax(predictions[:, :n_classes - 1], axis=1)
    bbox_confidences = predictions[np.arange(len(bbox_labels)), bbox_labels]
    sorted_detections = np.argsort(bbox_confidences)

    for i in reversed(sorted_detections):
        if bbox_confidences[i] < confidence_thresh:
            break
        norm_rect = decode_location(predictions[i, n_classes:], default_boxes[i].rect)
        decoded_detections.append([norm_rect, bbox_labels[i], bbox_confidences[i]])
    return decoded_detections


def net_results_to_bboxes(predictions, default_boxes, img_size,
                          confidence_thresh=0.1, overlap_thresh=0.5, number_thresh=100):
    result = []
    decoded_detections = nn_predictions_to_bboxes(predictions, default_boxes, confidence_thresh)[:number_thresh]
    grouped_by_label_detections = {}
    for det in decoded_detections:
        det[0] = norm_rect_to_rect(img_size, det[0])
        if det[0] in grouped_by_label_detections:
            grouped_by_label_detections[det[1]].append(det)
        else:
            grouped_by_label_detections[det[1]] = [det, ]

    for label, detections in grouped_by_label_detections.items():
        pick = nms(detections, overlap_thresh)
        for i in pick:
            result.append(detections[i])
    return result
