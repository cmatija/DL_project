import tensorflow as tf
import numpy as np


def get_img_patch_grid(imgs, ksizes, strides, rates, padding):
    #inputs
    #imgs: NHWC tensor
    #patch_size: HW specification of patch size
    patches = tf.image.extract_image_patches(imgs, ksizes, strides, rates, padding)
    n_patches = patches.shape[1] * patches.shape[2]
    return tf.reshape(patches, [n_patches * patches.shape[0], ksizes[1], ksizes[2], 3])