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
            with slim.arg_scope([layers.conv2d, layers_lib.max_pool2d], outputs_collections=end_points_collection):
                net = layers_lib.repeat(inputs, 2, layers.conv2d, 64, [3, 3], scope='conv1')
                net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')
                net = layers_lib.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
                net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')
                net = layers_lib.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv3')
                net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')
                net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv4')
                net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')
                net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv5')
                # net = layers_lib.max_pool2d(net, [2, 2], scope='pool5')

                end_points = utils.convert_collection_to_dict(end_points_collection)
                self.net = net
                self.end_points = end_points

    def __init_ssd_300_part(self):
        self.net = layers_lib.max_pool2d(self.net, [3, 3], stride=1, scope='pool5')
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