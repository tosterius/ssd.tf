import tensorflow as tf
import tensorflow.contrib.slim as slim
# from tensorflow.contrib.slim.nets import vgg
from tensorflow.contrib.layers.python.layers import layers as layers_lib
import tensorflow.contrib.layers as layers
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.framework import get_variables_to_restore


#http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz


class SSD:
    def __init__(self, input):
        self.net = None
        # self.logits, self.end_points = vgg.vgg_16(train_x)
        self.__init_vgg_16_part(input)
        self.__init_ssd_300_part()

        vars_to_restore = get_variables_to_restore(
            exclude=['vgg_16/fc6', 'vgg_16/dropout6', 'vgg_16/fc7', 'vgg_16/dropout7', 'vgg_16/fc8'])
        self.saver = tf.train.Saver(vars_to_restore)

    def build_with_vgg(self, session, vgg_ckpt_path):
        self.saver.restore(session, vgg_ckpt_path)

    def __init_vgg_16_part(self, inputs, scope='vgg_16'):
        with variable_scope.variable_scope(scope, 'vgg_16', [inputs]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([layers.conv2d, slim.max_pool2d], outputs_collections=end_points_collection):
                net = slim.repeat(inputs, 2, layers.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv5')

                end_points = utils.convert_collection_to_dict(end_points_collection)
                self.net = net
                self.end_points = end_points

    def __init_ssd_300_part(self, scope='ssd_300', is_training=True, dropout_keep_prob=0.6):
        with variable_scope.variable_scope(scope, 'ssd_300', [self.net]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([layers.conv2d, slim.max_pool2d], outputs_collections=end_points_collection):
                # this is the first difference with VGG
                self.net = slim.max_pool2d(self.net, [3, 3], stride=1, scope='pool5')

                self.net = slim.conv2d(self.net, 1024, [3, 3], rate=6, scope='conv6')
                self.net = tf.layers.dropout(self.net, rate=dropout_keep_prob, training=is_training)

                self.net = slim.conv2d(self.net, 1024, [1, 1], scope='conv7')
                self.net = tf.layers.dropout(self.net, rate=dropout_keep_prob, training=is_training)

                self.net = slim.conv2d(self.net, 256, [1, 1], scope='conv8_1')
                self.net = slim.conv2d(self.net, 512, [3, 3], stride=2, scope='conv8_2')


        pass



def build_graph_train(checkpoint_load_path=None):
    graph = tf.Graph()
    with graph.as_default():
        input_x = tf.placeholder(tf.float32, (64, 224, 224, 3))

    with tf.Session(graph=graph) as session:
        ssd = SSD(input_x)
        ssd.build_with_vgg(session, checkpoint_load_path)
        for v in tf.get_default_graph().as_graph_def().node:
            print(v.name)
        pass



build_graph_train('pretraned/vgg_16.ckpt')