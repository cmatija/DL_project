import tensorflow as tf
import numpy as np


def get_img_patch_grid(imgs, ksizes, strides, rates, padding):
    #inputs
    #imgs: NHWC tensor
    #patch_size: HW specification of patch size
    patches = tf.image.extract_image_patches(imgs, ksizes, strides, rates, padding)
    n_patches = patches.shape[1] * patches.shape[2]
    return tf.reshape(patches, [n_patches * patches.shape[0], ksizes[1], ksizes[2], 3])


def prepare_imgs_for_lpips(inp, axis, datatype='NHWC'):
    #normalize to -1/1
    inp_normalized = 2* tf.div(
        tf.subtract(
            inp,
            tf.reduce_min(inp, axis=axis)
        ),
        tf.subtract(
            tf.reduce_max(inp, axis=axis),
            tf.reduce_min(inp, axis=axis)
        )
    )-1

    if datatype == 'NCHW':
        return tf.transpose(inp_normalized, [0, 2, 3, 1])
    elif datatype == 'NHWC':
        return inp_normalized
    else:
        return -1