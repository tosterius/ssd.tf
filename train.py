import os
import argparse
import tensorflow as tf

import ssd
import dataset
from profiles import SSD_300


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def train_from_scratch(n_epochs, lr, batch_size, data_set, vgg_checkpoint_path, checkpoints_dir, log_dir='./', profile=SSD_300):
    tf.reset_default_graph()
    # graph = tf.get_default_graph()

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
                print(loss_batch)

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