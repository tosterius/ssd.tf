import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.framework import get_variables_to_restore

from profiles import voc_ssd_300
import dataset


def conv_with_l2_reg(tensor, depth, layer_hw, name):
    with tf.variable_scope(name):
        w = tf.get_variable("filter", shape=[3, 3, tensor.get_shape()[3], depth],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.zeros(depth), name='biases')
        x = tf.nn.conv2d(tensor, w, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        x = tf.reshape(x, [-1, layer_hw[0] * layer_hw[1], depth])
        l2_norm = tf.nn.l2_loss(w)
    return x, l2_norm


def smooth_l1_loss(x):
    """https://mohitjainweb.files.wordpress.com/2018/03/smoothl1loss.pdf"""
    with tf.variable_scope('smooth_l1_loss'):
        square_loss = 0.5 * tf.math.pow(x, 2)
        abs_x = tf.abs(x)
        return tf.where(tf.less(abs_x, 1.0), square_loss, abs_x - 0.5)


class SSD:
    def __init__(self, session, profile=voc_ssd_300):
        self.session = session
        self.net = None
        self.feature_maps = []
        self.profile = profile

        self.label_dim = self.profile.n_classes + 4  # 4 bbox coords

        # input tensors:
        with tf.variable_scope('image_input'):
            self.input = tf.placeholder(tf.float32, (None, 300, 300, 3))
        self.gt = None  # looks like [y_0, y_1, .., y_num_of_classes, box_xc, box_yx, box_w, box_h]

        # output tensors:
        self.logits = None
        self.classifier = None
        self.detections = None
        self.result = None

        self.optimizer = None
        self.l2_norm = 0
        self.loss = None
        self.confidence_loss = None
        self.localization_loss = None

        self.__init_vgg_16_part()
        self.__init_ssd_300_part()
        self.__init_detection_layers()

        exclude_layers = ['vgg_16/fc6',
                          'vgg_16/dropout6',
                          'vgg_16/fc7',
                          'vgg_16/dropout7',
                          'vgg_16/fc8'] + list(self.ssd_end_points.keys())

        self.saver = tf.train.Saver(get_variables_to_restore(exclude=exclude_layers))

    def build_with_vgg(self, session, vgg_ckpt_path):
        self.saver.restore(session, vgg_ckpt_path)

    def __init_vgg_16_part(self, scope='vgg_16'):
        with variable_scope.variable_scope(scope, 'vgg_16', [self.input]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d, slim.max_pool2d], outputs_collections=end_points_collection):
                self.net = slim.repeat(self.input, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                self.net = slim.max_pool2d(self.net, [2, 2], scope='pool1')  # 150x150
                self.net = slim.repeat(self.net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                self.net = slim.max_pool2d(self.net, [2, 2], scope='pool2')  # 75x75
                self.net = slim.repeat(self.net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                self.net = slim.max_pool2d(self.net, [2, 2], scope='pool3', padding='SAME')  # 38x38
                self.net = slim.repeat(self.net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                self.feature_maps.append(self.net)
                self.net = slim.max_pool2d(self.net, [2, 2], scope='pool4', padding='SAME')
                self.net = slim.repeat(self.net, 3, slim.conv2d, 512, [3, 3], scope='conv5')

                self.vgg_end_points = utils.convert_collection_to_dict(end_points_collection)

    def __init_ssd_300_part(self, scope='ssd_300', is_training=True, dropout_keep_prob=0.6):
        with variable_scope.variable_scope(scope, 'ssd_300', [self.net]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d, slim.max_pool2d], outputs_collections=end_points_collection):
                self.net = slim.max_pool2d(self.net, [3, 3], stride=1, scope='pool5', padding='SAME')

                self.net = slim.conv2d(self.net, 1024, [3, 3], rate=6, scope='conv6')
                self.net = tf.layers.dropout(self.net, rate=dropout_keep_prob, training=is_training)

                self.net = slim.conv2d(self.net, 1024, [1, 1], scope='conv7')
                self.net = tf.layers.dropout(self.net, rate=dropout_keep_prob, training=is_training)
                self.feature_maps.append(self.net)
                self.net = slim.conv2d(self.net, 256, [1, 1], scope='conv8_1')
                self.net = slim.conv2d(self.net, 512, [3, 3], stride=2, scope='conv8_2')
                self.feature_maps.append(self.net)
                self.net = slim.conv2d(self.net, 128, [1, 1], scope='conv9_1')
                self.net = slim.conv2d(self.net, 256, [3, 3], stride=2, scope='conv9_2')
                self.feature_maps.append(self.net)
                self.net = slim.conv2d(self.net, 128, [1, 1], scope='conv10_1')
                self.net = slim.conv2d(self.net, 256, [3, 3], scope='conv10_2', padding='VALID')
                self.feature_maps.append(self.net)
                self.net = slim.conv2d(self.net, 128, [1, 1], scope='conv11_1')
                self.net = slim.conv2d(self.net, 256, [3, 3], scope='conv11_2', padding='VALID')
                self.feature_maps.append(self.net)
                self.ssd_end_points = utils.convert_collection_to_dict(end_points_collection)

    def __init_detection_layers(self):
        output = []
        with tf.variable_scope('classifiers'):
            for i, feature_map in enumerate(self.feature_maps):
                mp = self.profile.maps[i]
                for j in range(mp.n_bboxes):
                    c, norm = conv_with_l2_reg(feature_map, self.label_dim, mp.size, 'classifier%d_%d' % (i, j))
                    output.append(c)
                    self.l2_norm += norm

        with tf.variable_scope('output'):
            output = tf.concat(output, axis=1, name='output')
            self.n_anchors = tf.shape(output, out_type=tf.int64)[0]
            self.logits = output[:, :, :self.profile.n_classes]
            self.classifier = tf.nn.softmax(self.logits)
            self.detections = output[:, :, self.profile.n_classes:]
            self.result = tf.concat([self.classifier, self.detections], axis=-1, name='result')

    def init_loss(self):
        self.gt = tf.placeholder(tf.float32, name='labels', shape=[None, None, self.label_dim])

        batch_size = tf.shape(self.gt)[0]

        with tf.variable_scope('gt'):
            gt_labels = self.gt[:, :, :self.profile.n_classes]
            gt_rects = self.gt[:, :, self.profile.n_classes:]

        with tf.variable_scope('counters'):
            n_total = tf.ones([batch_size], dtype=tf.int64) * self.n_anchors
            n_negative = tf.count_nonzero(gt_labels[:, :, -1], axis=1)
            n_positive = n_total - n_negative

        with tf.variable_scope('masks'):
            positives_mask = tf.equal(gt_labels[:, :, -1], 0)
            negatives_mask = tf.not_equal(gt_labels[:, :, -1], 0)

        with tf.variable_scope('confidence_loss'):
            ce = tf.nn.softmax_cross_entropy_with_logits_v2(gt_labels, self.logits)
            positive_losses = tf.where(positives_mask, ce, tf.zeros_like(ce))
            positive_total_loss = tf.reduce_sum(positive_losses, axis=-1)

            negative_losses = tf.where(negatives_mask, ce, tf.zeros_like(ce))
            negative_top_k = tf.nn.top_k(negative_losses, tf.cast(self.n_anchors, dtype=tf.int32))[0]

            # Instead of using all the negative examples, we sort them using the highest
            # confidence loss for each default box and pick the top ones so that the ratio
            # between the negatives and positives is at most 3:1.
            n_max_negative_per_sample = tf.minimum(n_negative, 3 * n_positive)

            n_max_negative_per_sample_t = tf.expand_dims(n_max_negative_per_sample, 1)
            rng = tf.range(0, self.n_anchors, 1)
            range_row = tf.to_int64(tf.expand_dims(rng, 0))
            negatives_max_mask = tf.less(range_row, n_max_negative_per_sample_t)

            negatives_max = tf.where(negatives_max_mask, negative_top_k, tf.zeros_like(negative_top_k))
            negatives_total_loss = tf.reduce_sum(negatives_max, axis=-1)

            confidence_loss = tf.add(positive_total_loss, negatives_total_loss)

            confidence_loss = tf.where(tf.equal(n_positive, 0),
                                       tf.zeros([batch_size]),
                                       tf.div_no_nan(confidence_loss, tf.cast(n_positive, dtype=tf.float32)))

            self.confidence_loss = tf.reduce_mean(confidence_loss, name='confidence_loss')

        with tf.variable_scope('localization_loss'):
            localization_loss = smooth_l1_loss(tf.subtract(self.detections, gt_rects))
            localisation_loss_for_anchor = tf.reduce_sum(localization_loss, axis=-1)

            positive_localisation_loss_for_anchor = tf.where(positives_mask, localisation_loss_for_anchor,
                                                             tf.zeros_like(localisation_loss_for_anchor))

            localization_loss = tf.reduce_sum(positive_localisation_loss_for_anchor, axis=-1)
            localization_loss = tf.where(tf.equal(n_positive, 0),
                                         tf.zeros([batch_size]),
                                         tf.div_no_nan(localization_loss, tf.cast(n_positive, dtype=tf.float32)))

            self.localization_loss = tf.reduce_mean(localization_loss, name='localization_loss')

            with tf.variable_scope('total_loss'):
                self.loss = tf.add(self.localization_loss, self.confidence_loss, name='loss')

        return self.loss

    def init_optimizer(self, lr, momentum=0.9, global_step=None):
        with tf.variable_scope('optimizer'):
            optimizer = tf.train.MomentumOptimizer(lr, momentum)
            optimizer = optimizer.minimize(self.loss, global_step=global_step, name='optimizer')
        self.optimizer = optimizer


def build_graph_train(checkpoint_load_path=None):
    graph = tf.Graph()
    with graph.as_default():
        input_x = tf.placeholder(tf.float32, (None, 300, 300, 3))

    with tf.Session(graph=graph) as session:
        ssd = SSD()
        ssd.init_loss()
        # ssd.build_with_vgg(session, checkpoint_load_path)
        for v in tf.get_default_graph().as_graph_def().node:
            print(v.name)
        pass


def train(n_epochs, lr, batch_size, data_set, checkpoint_load_path=None):
    graph = tf.Graph()

    global_step = tf.Variable(0, trainable=False)
    with tf.Session(graph=graph) as session:
        net = SSD(session)
        net.init_loss()
        net.init_optimizer(lr, global_step=global_step)

        session.run(tf.global_variables_initializer())

        lg = dataset.LabelGenerator(voc_ssd_300, True)
        loader = dataset.ImageLoader()
        generator = lg.get(ds, batch_size, loader)

        for x, y, gt in generator:
            feed = {net.input: x, net.gt: y}
            result, loss_batch, _ = session.run([net.result, net.loss, net.optimizer], feed_dict=feed)


if __name__ == '__main__':
    ds = dataset.VocDataset('/data/Workspace/data/VOCDebug')

    train(10, 0.00001, 2, ds)