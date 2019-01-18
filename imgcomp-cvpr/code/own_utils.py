import tensorflow as tf
from os import listdir
#OUR CODE

def get_job_ids(logdir_root, mode='val'):

    #used for val.py when use_all is used; goes through all subfolders in logdir_root (which contain experiment checkpoints)
    # and returns that list to be used by val.py
    #NOTE: only truly guaranteed to work if there is a low/med/high experiment for al  configs stored in log_dir_root
    dirs = [f for f in listdir(logdir_root) if '@' in f]
    if mode == 'val':
        return ",".join([d.split()[0] for d in dirs])
    elif mode == 'plot':
        configs = ['vgg', 'alexnet', 'alexnet_vgg', 'ms_ssim']
        vgg_dirs = [d for d in dirs if 'vgg' in d and not 'alexnet' in d]
        alexnet_dirs = [d for d in dirs if 'vgg' not in d and 'alexnet' in d]
        alexnet_vgg_dirs = [d for d in dirs if 'vgg'  in d and 'alexnet' in d]
        baseline_list = [d for d in dirs if 'vgg'  not in d and 'alexnet' not in d and not 'ignore' in d]
        complete_list = [vgg_dirs, alexnet_dirs, alexnet_vgg_dirs, baseline_list]
        result_list1 = []
        result_list2 = []
        for i,l in enumerate(complete_list):
            if not l:
                continue
            low_config = [d for d in l if '@low' in d]
            med_config = [d for d in l if '@med' in d]
            hi_config = [d for d in l if '@hi' in d]
            if not low_config or not med_config or not hi_config:
                continue
            concat = low_config + med_config + hi_config
            result_list1.append((",".join([c.split()[0] for c in concat]), configs[i]))
        return result_list1

def get_img_patch_grid(imgs, ksizes, strides, rates, padding):
    #inputs
    #imgs: NHWC tensor
    #patch_size: HW specification of patch size
    patches = tf.image.extract_image_patches(imgs, ksizes, strides, rates, padding)
    n_patches = patches.shape[1] * patches.shape[2]
    return tf.reshape(patches, [n_patches * patches.shape[0], ksizes[1], ksizes[2], 3])


def prepare_imgs_for_lpips(inp, axis, datatype='NHWC'):
    #normalize to -1/1 in each channel
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