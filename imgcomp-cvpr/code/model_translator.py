import pickle
import tensorflow as tf
import alexnet
import numpy as np
import vgg
from own_utils import prepare_imgs_for_lpips, get_img_patch_grid
import sys
#OUR CODE
#handles the translation from LPIPS/PyTorch models to TF models that can also be used for backprop
sys.path.insert(0, '../../lpips-tensorflow')
import lpips_tf
class model_translator:

    def __init__(self, input1, input2, network='alexnet', mode='net', scope_suffix='', datatype = 'NHWC', use_own=True):
        self.slim = tf.contrib.slim
        self.network = network
        self.mode = mode
        #directly taken from LPIPS torch script; they do this to all inputs to their networks
        self.shift = tf.reshape(tf.constant([-.030, -.088, -.188]), [1, 1, 1, 3])
        self.scale = tf.reshape(tf.constant([.458, .448, .450]), [1, 1, 1, 3])

        self.weights_transposed = {}
        self.weights_original = {}

        losses = []
        #Normalize all color channels to range -1,1
        inp_normalized = prepare_imgs_for_lpips(input1, None, datatype=datatype)
        otp_normalized = prepare_imgs_for_lpips(input2, None, datatype=datatype)
        ksizes = [1, 64, 64, 1]
        strides = [1, 64, 64, 1]
        rates = [1, 1, 1, 1]
        padding = 'SAME'
        patches_inp = get_img_patch_grid(inp_normalized, ksizes, strides, rates, padding)
        patches_otp = get_img_patch_grid(otp_normalized, ksizes, strides, rates, padding)

        if use_own:
            self.get_weights()
            network_input = tf.div(tf.concat([patches_inp, patches_otp], 0) - self.shift, self.scale)
            if not scope_suffix == '':
                scope_suffix = '_' + scope_suffix
            if 'alexnet' in network or 'alex' in network:
                with self.slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
                    rets_alex = alexnet.alexnet_v2(network_input, self.weights_transposed['alexnet'],
                                              scope='alexnet_v2'+scope_suffix)
                    loss_alex = self.get_diff(rets_alex)
                    losses += [loss_alex]

            if 'vgg' in network:
                with self.slim.arg_scope(vgg.vgg_arg_scope()):
                    rets_vgg = vgg.vgg_16(network_input, self.weights_transposed['vgg'], scope='vgg_v2' + scope_suffix)
                    loss_vgg = self.get_diff(rets_vgg)
                    losses += [loss_vgg]
            else:
                loss_vgg = None

            self.net = tf.reduce_sum(tf.stack(losses))
        else:
            if network == 'alexnet':
                lpips_network = 'alex'
            else:
                lpips_network = network
            self.net = lpips_tf.lpips(patches_inp, patches_otp, model=mode, net=lpips_network)

    def get_weights(self):

        possible_networks = ['alexnet', 'vgg']

        for possible_network in possible_networks:
            if possible_network in self.network:
                fpath = '../../models/' + possible_network + '_'+self.mode +\
                        '.pickle'
                pickle_in = open(fpath, "rb")
                self.weights_original[possible_network] = pickle.load(pickle_in)

                weights_transposed = []
                permutation = [2, 3, 1, 0]
                for weights in self.weights_original[possible_network]:
                    # NCHW to NHWC
                    if len(weights.shape) == 4:
                        weights_transposed.append(np.transpose(weights, permutation))
                    else:
                        weights_transposed.append(weights)

                self.weights_transposed[possible_network] = weights_transposed

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