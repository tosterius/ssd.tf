import os
import argparse
import tensorflow as tf

import ssd
import dataset
import utils

from profiles import SSD_300


def initialize_variables(session):
    init_tensors = [tf.is_variable_initialized(var) for var in tf.global_variables()]
    init_flags = session.run(init_tensors)
    flag_tensor_pairs = zip(init_flags, tf.global_variables())
    flag_tensor_pairs = [var for init, var in flag_tensor_pairs if not init]
    session.run(tf.variables_initializer(flag_tensor_pairs))


class Summary:
    def __init__(self, session, writer, name, item_names):
        self.session = session
        self.writer = writer
        self.item_names = item_names
        self.placeholders = {}
        self.summary_ops = {}
        for item_name in self.item_names:
            item_name = str(item_name)
            summary_name = name + '_' + item_name
            ph = tf.placeholder(tf.float32, name=summary_name + '_ph')
            self.placeholders[item_name] = ph
            self.summary_ops[item_name] = tf.summary.scalar(summary_name, ph)

    def append(self, epoch, item_value_dict):
        feed = {}
        tensors = []
        for item, value in item_value_dict.items():
            item_name = str(item)
            ph = self.placeholders[item_name]
            feed[ph] = value
            tensors.append(self.summary_ops[item_name])

        summaries = self.session.run(tensors, feed_dict=feed)
        for summary in summaries:
            self.writer.add_summary(summary, epoch)


def init_summaries(session, writer, label_names):
    tps = Summary(session, writer, 'train_precision', label_names)
    vps = Summary(session, writer, 'val_precision', label_names)

    loss_names = ['loss', 'confidence_loss', 'localization_loss']
    tls = Summary(session, writer, 'train_loss', loss_names)
    vls = Summary(session, writer, 'val_loss', loss_names)
    return tps, vps, tls, vls


def train(n_epochs, lr, weight_decay, batch_size, train_dataset, checkpoint_path,
          checkpoints_dir, log_dir='./', profile=SSD_300, continue_training=False):

    tf.reset_default_graph()

    precision_metric_local = utils.PrecisionMetric()
    precision_metric_global = utils.PrecisionMetric()

    val_dataset = train_dataset.extract(0.05)

    def calc_and_print_stat(e, gt, result, batch_counter, total_loss, loc_loss, conf_loss):
        for i in range(result.shape[0]):
            detections = utils.get_filtered_result_bboxes(result[i], label_generator.default_boxes_rel, profile.imgsize)
            gt_objects = dataset.lo_to_abs_rects(profile.imgsize, gt[i])
            precision_metric_local.add(gt_objects, detections)
            precision_metric_global.add(gt_objects, detections)

        if batch_counter % 10 == 0:
            print("E{}/Batch[{}] loss: {}, {}, {}".format(e, batch_counter, total_loss, loc_loss, conf_loss))
            precisions, mean = precision_metric_local.calc_and_reset()
            print("Local prec: ", mean, precisions)

    with tf.Session() as session:

        net = ssd.SSD(session, profile, len(train_dataset.label_names))
        if continue_training:
            net.load_metagraph(checkpoint_path, continue_training=continue_training)
        else:
            net.build_with_vgg(checkpoint_path, weight_decay)
            global_step = tf.Variable(0, trainable=False)
            net.init_loss_and_optimizer(lr, global_step=global_step)

        # for op in graph.get_operations():
        #     print(op.name)

        # summary loggers initialization
        summary_writer = tf.summary.FileWriter(log_dir)
        train_precision_summary, val_precision_summary, train_loss_summary, val_loss_summary = \
            init_summaries(session, summary_writer, train_dataset.label_names)

        saver = tf.train.Saver()
        initialize_variables(session)

        label_generator = dataset.LabelGenerator(profile)
        image_loader = dataset.ImageLoader(profile.imgsize)

        print('Dataset size: {}'.format(len(ds.data_list)))
        for epoch in range(n_epochs):
            print('Epoch [{}/{}]'.format(epoch, n_epochs))
            # Train
            train_generator = label_generator.get(train_dataset, batch_size, image_loader)
            for batch_counter, (x, y, gt) in enumerate(train_generator):
                feed = {net.input: x, net.gt: y}
                result, total_loss, loc_loss, conf_loss, _ = session.run(
                    [net.output, net.loss, net.localization_loss, net.confidence_loss, net.optimizer],
                    feed_dict=feed)

                calc_and_print_stat(epoch, gt, result, batch_counter, total_loss, loc_loss, conf_loss)

            train_loss_summary.append(epoch, {'loss': total_loss,
                                              'confidence_loss': conf_loss,
                                              'localization_loss': loc_loss})

            precisions, mean = precision_metric_global.calc_and_reset()
            precisions = ds.decode_dict(precisions)
            print("[T] Global prec: ", mean, precisions)
            train_precision_summary.append(epoch, precisions)

            # Validation
            print('--Validation: ')
            val_generator = label_generator.get(val_dataset, batch_size, image_loader)
            for batch_counter, (x, y, gt) in enumerate(val_generator):
                feed = {net.input: x, net.gt: y}
                result, total_loss, loc_loss, conf_loss = session.run(
                    [net.output, net.loss, net.localization_loss, net.confidence_loss],
                    feed_dict=feed)

                calc_and_print_stat(epoch, gt, result, batch_counter, total_loss, loc_loss, conf_loss)

            val_loss_summary.append(epoch, {'loss': total_loss,
                                            'confidence_loss': conf_loss,
                                            'localization_loss': loc_loss})

            precisions, mean = precision_metric_global.calc_and_reset()
            precisions = ds.decode_dict(precisions)
            print("[V] Global prec: ", mean, precisions)
            val_precision_summary.append(epoch, precisions)

            summary_writer.flush()

            checkpoint_path = os.path.join(checkpoints_dir, 'checkpoint-epoch-%03d.ckpt' % epoch)
            saver.save(session, checkpoint_path)
            print('-Checkpoint "%s" was created' % checkpoint_path)


def test(batch_size, data_set, log_dir='./', profile=SSD_300):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/data/Workspace/data/VOCdevkit/VOC2012', help='data directory')
    parser.add_argument('--data-parser', default='pascal-voc', help='data parser name')     # TODO:
    parser.add_argument('--dest-dir', default='checkpoints', help='output directory')
    parser.add_argument('--experiment', default='ssd', help='experiment name')
    parser.add_argument('--log-dir', default="tb", help='log directory')
    parser.add_argument('--checkpoint', default='/data/Downloads/vgg_16_2016_08_28/vgg_16.ckpt',
                        help='path to pretrained VGG16 model(checkpoint file)')
    parser.add_argument('--continue', action='store_true', default=False)
    parser.add_argument('--n-epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=20, help='batch size')
    parser.add_argument('--lr', type=int, default=0.0001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for the optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='L2 normalization factor')
    parser.add_argument('--val-frac', type=float, default=0.05, help='the fraction of validation data')

    args = parser.parse_args()

    checkpoints_dir = utils.make_dir(os.path.join(args.dest_dir, args.experiment))
    data_dir = args.data_dir
    log_dir = args.log_dir
    experiment_name = args.experiment
    checkpoint = args.checkpoint
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay

    ds = dataset.VocDataset(data_dir, '/data/Workspace/data/VOCdevkit/vokdata.pkl')

    train(n_epochs, lr, weight_decay, batch_size, ds, checkpoint, checkpoints_dir)
