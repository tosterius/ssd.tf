import os
import argparse
import tensorflow as tf
import numpy as np

import ssd
import dataset
import utils

from collections import defaultdict, Counter
from profiles import SSD_300


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


class PrecisionMetric:
    def __init__(self):
        self.overlap_thresh = 0.5
        self.gt_samples = []
        self.det_rects = defaultdict(list)
        self.det_confs = defaultdict(list)
        self.det_sample_ids = defaultdict(list)

    def add(self, gt_sample, detections):
        self.gt_samples.append(gt_sample)
        for rect, label, score in detections:
            self.det_rects[label].append(rect.as_array())
            self.det_confs[label].append(score)
            self.det_sample_ids[label].append(len(self.gt_samples) - 1)

    def reset(self):
        self.det_rects.clear()
        self.det_confs.clear()
        self.det_sample_ids.clear()
        self.gt_samples.clear()

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
                precisions[gt_label] = 0
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
                iou = utils.calc_jaccard_overlap(det_box, gt_box)
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


def train_from_scratch(n_epochs, lr, batch_size, data_set, vgg_checkpoint_path, checkpoints_dir, log_dir='./', profile=SSD_300):
    tf.reset_default_graph()
    # graph = tf.get_default_graph()

    precision_metric = PrecisionMetric()

    with tf.Session() as session:

        net = ssd.SSD(session, profile, len(data_set.label_names))

        # net.load_metagraph('./checkpoints/ssd/checkpoint-epoch-000.ckpt.meta',
        #                    './checkpoints/ssd')

        # for op in graph.get_operations():
        #     print(op.name)

        net.build_with_vgg(vgg_checkpoint_path)

        global_step = tf.Variable(0, trainable=False)
        net.init_loss_and_optimizer(lr, global_step=global_step)

        summary_writer = tf.summary.FileWriter(log_dir)
        saver = tf.train.Saver()

        session.run(tf.global_variables_initializer())

        lg = dataset.LabelGenerator(profile, True)
        loader = dataset.ImageLoader(profile.imgsize)
        generator = lg.get(data_set, batch_size, loader)
        for epoch in range(n_epochs):
            for x, y, gt in generator:
                feed = {net.input: x, net.gt: y}
                result, loss_batch, _ = session.run([net.result, net.loss, net.optimizer], feed_dict=feed)
                for i in range(result.shape[0]):
                    detections = utils.get_filtered_result_bboxes(result[i], lg.default_boxes_rel, profile.imgsize)
                    gt_objects = dataset.lo_to_abs_rects(profile.imgsize, gt[i])
                    precision_metric.add(gt_objects, detections)

                    # todo
                print(loss_batch)
                precisions, mean = precision_metric.calc()
                print(precisions)

            precision_metric.calc()

            checkpoint_path = os.path.join(checkpoints_dir, 'checkpoint-epoch-%03d.ckpt' % epoch)
            print('Checkpoint "%s" was created' % checkpoint_path)
            # for var in saver._var_list:
            #     print(var)

            saver.save(session, checkpoint_path)


def test(batch_size, data_set, log_dir='./', profile=SSD_300):

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/data/Workspace/data/VOCDebug', help='data directory')
    parser.add_argument('--data-parser', default='pascal-voc', help='data parser name')     # TODO:
    parser.add_argument('--dest-dir', default='checkpoints', help='output directory')
    parser.add_argument('--experiment', default='ssd', help='experiment name')
    parser.add_argument('--log-dir', default="tb", help='log directory')
    parser.add_argument('--vgg-checkpoint', default='vgg_graph', help='path to pretrained VGG16 model(checkpoint file)')
    parser.add_argument('--n-epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--lr', type=str, default='0.0001', help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for the optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='L2 normalization factor')

    args = parser.parse_args()

    checkpoints_dir = make_dir(os.path.join(args.dest_dir, args.experiment))

    ds = dataset.VocDataset('/data/Workspace/data/VOCDebug')
    # ds = dataset.VocDataset('/home/arthur/Workspace/data/VOC2007')

    train_from_scratch(10, 0.0001, 2, ds, '/data/Downloads/vgg_16_2016_08_28/vgg_16.ckpt', checkpoints_dir)
    # train_from_scratch(10, 0.0001, 2, ds, '/home/arthur/Workspace/projects/github/ssd.tf/vgg_16.ckpt', checkpoints_dir)