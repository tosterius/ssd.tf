import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.framework import get_variables_to_restore


def conv_with_l2_reg(tensor, depth, layer_hw, name):
    with tf.variable_scope(name):
        weights = tf.get_variable("filter", shape=[3, 3, tensor.get_shape()[3], depth],
                            initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.Variable(tf.zeros(depth), name='biases')
        x = tf.nn.conv2d(tensor, weights, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, bias)
        x = tf.reshape(x, [-1, layer_hw[0] * layer_hw[1], depth])
        l2_norm = tf.nn.l2_loss(weights)
    return x, l2_norm


def smooth_l1_loss(x):
    """https://mohitjainweb.files.wordpress.com/2018/03/smoothl1loss.pdf"""
    with tf.variable_scope('smooth_l1_loss'):
        square_loss = 0.5 * tf.math.pow(x, 2)
        abs_x = tf.abs(x)
        return tf.where(tf.less(abs_x, 1.0), square_loss, abs_x - 0.5)


class SSD:
    def __init__(self, session, profile, n_classes):
        self.session = session
        self.profile = profile

        self.net = None
        self.feature_maps = []
        self.n_classes = n_classes
        self.label_dim = self.n_classes + 4  # 4 bbox coords

        # input tensors:
        self.input = None   # (?, img_h, img_w, img_ch)
        self.gt = None      # (y0, y1, .., y_num_of_classes, box_xc, box_yx, box_w, box_h)

        # output tensors:
        self.logits = None
        self.classifier = None
        self.detections = None
        self.result = None  # (?, 8652, 25)     for ssd 300 and default profile

        self.optimizer = None
        self.l2_norm = 0
        self.loss = None
        self.confidence_loss = None
        self.localization_loss = None

    def build_with_vgg(self, vgg_ckpt_path):
        with tf.variable_scope('image_input'):
            shape = (None, self.profile.imgsize[0], self.profile.imgsize[1], 3)
            self.input = tf.placeholder(tf.float32, shape, name='image_input')

        self.__init_vgg_16_part()
        exclude_layers = ['vgg_16/fc6',
                          'vgg_16/dropout6',
                          'vgg_16/fc7',
                          'vgg_16/dropout7',
                          'vgg_16/fc8']

        saver = tf.train.Saver(get_variables_to_restore(exclude=exclude_layers))
        saver.restore(self.session, vgg_ckpt_path)

        self.__init_ssd_part()
        self.__init_detection_layers()

    def load_metagraph(self, metagraph_path, chekpoint_path):
        saver = tf.train.import_meta_graph(metagraph_path)
        saver.restore(self.session, tf.train.latest_checkpoint(chekpoint_path))

        self.input = self.session.graph.get_tensor_by_name('image_input/image_input:0')
        self.result = self.session.graph.get_tensor_by_name('output/result:0')

    def load_metagraph_for_optimization(self):
        self.gt = self.session.graph.get_tensor_by_name('labels:0')
        self.optimizer = self.session.graph.get_tensor_by_name('optimizer/optimizer:0')
        self.l2_norm = self.session.graph.get_tensor_by_name('total_loss/l2_loss:0')
        self.loss = self.session.graph.get_tensor_by_name('total_loss/loss:0')
        self.confidence_loss = self.session.graph.get_tensor_by_name('confidence_loss/confidence_loss:0')
        self.localization_loss = self.session.graph.get_tensor_by_name('localization_loss/localization_loss:0')

    def __init_vgg_16_part(self, scope='vgg_16', reg_scale=0.0005):
        with variable_scope.variable_scope(scope, 'vgg_16', [self.input]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d, slim.max_pool2d], outputs_collections=end_points_collection):
                vgg_weights_reg = slim.l2_regularizer(reg_scale)
                self.net = slim.repeat(self.input, 2, slim.conv2d, 64, [3, 3], scope='conv1',
                                       weights_regularizer=vgg_weights_reg)
                self.net = slim.max_pool2d(self.net, [2, 2], scope='pool1')  # 150x150
                self.net = slim.repeat(self.net, 2, slim.conv2d, 128, [3, 3], scope='conv2',
                                       weights_regularizer=vgg_weights_reg)
                self.net = slim.max_pool2d(self.net, [2, 2], scope='pool2')  # 75x75
                self.net = slim.repeat(self.net, 3, slim.conv2d, 256, [3, 3], scope='conv3',
                                       weights_regularizer=vgg_weights_reg)
                self.net = slim.max_pool2d(self.net, [2, 2], scope='pool3', padding='SAME')  # 38x38
                self.net = slim.repeat(self.net, 3, slim.conv2d, 512, [3, 3], scope='conv4',
                                       weights_regularizer=vgg_weights_reg)
                self.feature_maps.append(self.net)
                self.net = slim.max_pool2d(self.net, [2, 2], scope='pool4', padding='SAME')
                self.net = slim.repeat(self.net, 3, slim.conv2d, 512, [3, 3], scope='conv5',
                                       weights_regularizer=vgg_weights_reg)
                self.net = slim.max_pool2d(self.net, [3, 3], stride=1, scope='pool5', padding='SAME')
                self.vgg_end_points = utils.convert_collection_to_dict(end_points_collection)

    def __init_ssd_part(self, scope='ssd_300', is_training=True, dropout_keep_prob=0.5, reg_scale=0.005):
        with variable_scope.variable_scope(scope, 'ssd_300', [self.net]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d, slim.max_pool2d], outputs_collections=end_points_collection,
                                weights_regularizer=slim.l2_regularizer(reg_scale)):
                self.net = slim.conv2d(self.net, 1024, [3, 3], scope='conv6') # rate=6
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
            self.n_anchors = tf.shape(output, out_type=tf.int64)[1]
            self.logits = output[:, :, :self.n_classes]
            self.classifier = tf.nn.softmax(self.logits)
            self.detections = output[:, :, self.n_classes:]
            self.result = tf.concat([self.classifier, self.detections], axis=-1, name='result')

    def init_loss_and_optimizer(self, lr, momentum=0.9, global_step=None, decay=0.005):
        self.gt = tf.placeholder(tf.float32, name='labels', shape=[None, None, self.label_dim])

        batch_size = tf.shape(self.gt)[0]

        with tf.variable_scope('gt'):
            gt_labels = self.gt[:, :, :self.n_classes]
            gt_rects = self.gt[:, :, self.n_classes:]

        with tf.variable_scope('counters'):
            n_total = tf.ones([batch_size], dtype=tf.int64) * self.n_anchors
            n_negative = tf.count_nonzero(gt_labels[:, :, -1], axis=1)
            n_positive = n_total - n_negative
            n_positives_div_safe = tf.where(tf.equal(n_positive, 0),
                                            tf.ones([batch_size]) * 10e-12,
                                            tf.to_float(n_positive))
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
                                       tf.div(confidence_loss, n_positives_div_safe))

            self.confidence_loss = tf.reduce_mean(confidence_loss, name='confidence_loss')

        with tf.variable_scope('localization_loss'):
            localization_loss = smooth_l1_loss(tf.subtract(self.detections, gt_rects))
            localisation_loss_for_anchor = tf.reduce_sum(localization_loss, axis=-1)

            positive_localisation_loss_for_anchor = tf.where(positives_mask, localisation_loss_for_anchor,
                                                             tf.zeros_like(localisation_loss_for_anchor))

            localization_loss = tf.reduce_sum(positive_localisation_loss_for_anchor, axis=-1)
            localization_loss = tf.where(tf.equal(n_positive, 0),
                                         tf.zeros([batch_size]),
                                         tf.div(localization_loss, n_positives_div_safe))

            self.localization_loss = tf.reduce_mean(localization_loss, name='localization_loss')

        with tf.variable_scope('total_loss'):
            reg_w = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            l2_det_loss = tf.multiply(decay, self.l2_norm, name='l2_loss')
            l2_loss = l2_det_loss + tf.reduce_sum(reg_w)
            loss = tf.add(self.localization_loss, self.confidence_loss, name='loc_conf_loss')
            self.loss = tf.add(loss, l2_loss, name='loss')

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.MomentumOptimizer(lr, momentum)
            self.optimizer = self.optimizer.minimize(self.loss, global_step=global_step, name='optimizer')
