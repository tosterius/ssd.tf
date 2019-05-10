import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.framework import get_variables_to_restore
from collections import namedtuple


#http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz


Profile = namedtuple('Profile', ['n_classes', 'maps'])
MapParams = namedtuple('MapParams', ['size', 'scale', 'n_bboxes', 'ratios'])


voc_ssd_300 = Profile(n_classes=20, maps=[MapParams((38, 38), 0.2, 4, [1.0, 1.0, 2.0, 1.0/2.0]),
                                          MapParams((19, 19), 0.34, 6, [1.0, 1.0, 2.0, 1.0/2.0, 3.0, 1.0/3.0]),
                                          MapParams((10, 10), 0.48, 6, [1.0, 1.0, 2.0, 1.0/2.0, 3.0, 1.0/3.0]),
                                          MapParams((5, 5), 0.62, 6, [1.0, 1.0, 2.0, 1.0/2.0, 3.0, 1.0/3.0]),
                                          MapParams((3, 3), 0.76, 6, [1.0, 1.0, 2.0, 1.0/2.0, 3.0, 1.0/3.0]),
                                          MapParams((1, 1), 0.9, 6, [1.0, 1.0, 2.0, 1.0/2.0, 3.0, 1.0/3.0])])


def conv_l2_norm(tensor, depth, layer_hw, name):
    with tf.variable_scope(name):
        w = tf.get_variable("filter", shape=[3, 3, tensor.get_shape()[3], depth],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.zeros(depth), name='biases')
        x = tf.layers.conv2d(tensor, w, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        x = tf.reshape(x, [-1, layer_hw[0] * layer_hw[1], depth])
        l2_norm = tf.nn.l2_loss(w)
    return x, l2_norm


class SSD:
    def __init__(self, input, profile=voc_ssd_300):
        self.net = None
        self.feature_maps = []
        self.profile = profile
        self.l2_norm = 0

        self.logits = None
        self.classifier = None
        self.detector = None
        self.result = None

        self.__init_vgg_16_part(input)
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

    def __init_vgg_16_part(self, inputs, scope='vgg_16'):
        with variable_scope.variable_scope(scope, 'vgg_16', [inputs]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d, slim.max_pool2d], outputs_collections=end_points_collection):
                self.net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                self.net = slim.max_pool2d(self.net, [2, 2], scope='pool1')   # 150x150
                self.net = slim.repeat(self.net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                self.net = slim.max_pool2d(self.net, [2, 2], scope='pool2')   # 75x75
                self.net = slim.repeat(self.net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                self.net = slim.max_pool2d(self.net, [2, 2], scope='pool3', padding='SAME')   # 38x38
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
        det_dim = self.profile.n_classes + 4    # 4 bbox coords

        output = []
        with tf.variable_scope('classifiers'):
            for i, feature_map in enumerate(self.feature_maps):
                mp = self.profile.maps[i]
                for j in range(mp.n_bboxes):
                    c, norm = conv_l2_norm(feature_map, det_dim, mp.size, 'classifier%d_%d' % (i, j))
                    output.append(c)
                    self.l2_norm += norm

        with tf.variable_scope('output'):
            output = tf.concat(output, axis=1, name='output')
            self.logits = output[:, :, :self.profile.n_classes]
            self.classifier = tf.nn.softmax(self.logits)
            self.detector = output[:, :, self.profile.n_classes:]
            self.result = tf.concat([self.classifier, self.detector], axis=-1, name='result')


def build_graph_train(checkpoint_load_path=None):
    graph = tf.Graph()
    with graph.as_default():
        input_x = tf.placeholder(tf.float32, (None, 300, 300, 3))

    with tf.Session(graph=graph) as session:
        ssd = SSD(input_x)
        ssd.build_with_vgg(session, checkpoint_load_path)
        for v in tf.get_default_graph().as_graph_def().node:
            print(v.name)
        pass



build_graph_train('pretraned/vgg_16.ckpt')