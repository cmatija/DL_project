import pickle
import tensorflow as tf
import alexnet
import numpy as np
import vgg

class model_translator:

    def __init__(self, input1, input2, network='alexnet', mode='net', scope_suffix=''):
        self.slim = tf.contrib.slim
        self.network = network
        self.mode = mode
        #directly taken from torch script
        self.shift = tf.reshape(tf.constant([-.030, -.088, -.188]), [1, 1, 1, 3])
        self.scale = tf.reshape(tf.constant([.458, .448, .450]), [1, 1, 1, 3])
        network_input = tf.div(tf.concat([input1, input2], 0)-self.shift, self.scale)

        self.get_weights()
        if not scope_suffix == '':
            scope_suffix = '_' + scope_suffix
        if network == 'alexnet' or network == 'alex':
            with self.slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
                rets = alexnet.alexnet_v2(network_input, self.weights_transposed,
                                          scope='alexnet_v2'+scope_suffix)
        elif network == 'vgg':
            with self.slim.arg_scope(vgg.vgg_arg_scope()):
                rets = vgg.vgg_16(network_input, self.weights_transposed,scope='vgg_v2' + scope_suffix)
        self.net = self.get_diff(rets)

    def get_weights(self):

        if self.network == 'alexnet' and not self.mode=='net-lin':
            fpath = '/home/cmatija/code/python/DL_project_github/models/alex_net.pickle'
        elif self.network == 'vgg' and not self.mode=='net-lin':
            fpath = '/home/cmatija/code/python/DL_project_github/models/vgg_net.pickle'
        pickle_in = open(fpath, "rb")
        self.weights_original = pickle.load(pickle_in)

        weights_transposed = []
        permutation = [2,3,1,0]
        for weights in self.weights_original:
            #NCHW to NHWC
            if len(weights.shape)==4:
                weights_transposed.append(np.transpose(weights, permutation))
            else:
                weights_transposed.append(weights)

        self.weights_transposed = weights_transposed

    def get_diff(self, rets):
        batch_size = rets[0].shape[0].__int__() // 2
        for i, ret in enumerate(rets):
            inp = ret[:batch_size, ...]
            otp = ret[-batch_size:, ...]
            if i == 0:
                computed_loss = tf.losses.cosine_distance(tf.nn.l2_normalize(inp, axis=3),
                                                          tf.nn.l2_normalize(otp, axis=3), axis=3,
                                                          scope='cosine_distance')
            else:
                computed_loss += tf.losses.cosine_distance(tf.nn.l2_normalize(inp, axis=3),
                                                           tf.nn.l2_normalize(otp, axis=3), axis=3,
                                                           scope='cosine_distance')
        return computed_loss

    def get_alexnet_diff(self):
        a=2

    def get_vgg(inputs):
        a=2