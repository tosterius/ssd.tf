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
        self.gt_samples = []
        self.detections = defaultdict(list)

    def add(self, gt_sample, detections):
        self.gt_samples.append(gt_sample)
        for rect, label, score in detections:
            self.detections[label] = [rect, label, score, len(self.gt_samples) - 1]

    def calc(self):
        precisions = {}

        label_counter = Counter()
        gt_map = defaultdict(dict)

        for sample_id, boxes in enumerate(self.gt_samples):
            boxes_by_class = defaultdict(list)
            for box in boxes:
                label_counter[box[1]] += 1
                boxes_by_class[box[1]].append(box[0])

            for k, v in boxes_by_class.items():
                arr = np.zeros((len(v), 4))
                match = np.zeros((len(v)), dtype=np.bool)
                for i, box in enumerate(v):
                    arr[i] = box.as_array()
                gt_map[k][sample_id] = (arr, match)

        return precisions

    def reset(self):
        self.detections.clear()
        self.gt_samples.clear()


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
                precision_metric.calc()

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