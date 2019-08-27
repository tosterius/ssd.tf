import argparse
import tensorflow as tf

import ssd
import dataset
from profiles import SSD_300


def train_from_scratch(n_epochs, lr, batch_size, data_set, vgg_checkpoint_path=None, log_dir='./', profile=SSD_300):
    graph = tf.Graph()

    with tf.Session(graph=graph) as session:

        net = ssd.SSD(session, profile, data_set.get_labels_number())
        net.build_with_vgg(vgg_checkpoint_path)
        net.build_with_vgg('/data/Downloads/vgg_16_2016_08_28/vgg_16.ckpt')

        global_step = tf.Variable(0, trainable=False)
        net.init_loss_and_optimizer(lr, global_step=global_step)

        summary_writer = tf.summary.FileWriter(log_dir, graph)
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

            # TODO: save checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/data/Workspace/data/VOCDebug', help='data directory')
    parser.add_argument('--data-parser', default='pascal-voc', help='data parser name')     # TODO:
    parser.add_argument('--dest-dir', default='checkpoints', help='output directory')
    parser.add_argument('--log-dir', default="tb", help='log directory')
    parser.add_argument('--vgg-checkpoint', default='vgg_graph', help='path to pretrained VGG16 model(checkpoint file)')
    parser.add_argument('--n-epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--lr', type=str, default='0.0001', help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for the optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='L2 normalization factor')

    args = parser.parse_args()

    ds = dataset.VocDataset('/data/Workspace/data/VOCDebug')

    train_from_scratch(10, 0.0001, 2, ds)