import tensorflow as tf
import numpy as np
import h5py, os
import math
import matplotlib
matplotlib.use('Agg')  ## to avoid "connect to display" error in plt.savefig
import matplotlib.pyplot as plt
import time


def normalize(img):
    return (img-np.min(img))/(np.max(img)-np.min(img))

    
def GANLoss(inputs, target_is_real):
    GAN_loss =  tf.nn.sigmoid_cross_entropy_with_logits
    if target_is_real:
        target_tensor = tf.ones_like(inputs)
    else:
        target_tensor = tf.zeros_like(inputs)     
    return tf.reduce_mean(GAN_loss(logits=inputs,  labels=target_tensor ))


def criterionCycle(a, b):
    return tf.reduce_mean(tf.abs(a-b))

def batch_norm(inputs, scope, is_train = True, ac_fn = tf.nn.relu):
    outputs = tf.contrib.layers.batch_norm(
        inputs, decay=0.9, scale=True, activation_fn=ac_fn,
        updates_collections=None, epsilon=1.1e-5, is_training=is_train,
        scope = scope+'/batch_norm')
    return outputs

def layer_norm(inputs, scope, is_train = True, ac_fn = tf.nn.relu):
    outputs = tf.contrib.layers.layer_norm(
        inputs, scale=True, activation_fn=ac_fn,
        reuse =tf.AUTO_REUSE,
        scope = scope+'/layer_norm')
    return outputs

def conv(inputs, out_num, kernel_size, scope, data_type='2D', stride=1):
    if data_type == '2D':
        outs = tf.layers.conv2d(
            inputs, out_num, kernel_size, padding='same', name=scope+'/conv', 
            strides=stride, kernel_initializer=tf.truncated_normal_initializer)
    else:
        shape = 3*[kernel_size] + [inputs.shape[-1].value, out_num]
        weights = tf.get_variable( scope+'/conv/weights', shape, initializer=tf.truncated_normal_initializer())
        outs = tf.nn.conv3d(
            inputs, weights, (1, stride, stride, stride, 1), padding='SAME',
            name=scope+'/conv')
    return outs

def leaky_relu(inputs, name='lrelu'):
    return tf.nn.leaky_relu(inputs, name=name)

def deconv(inputs, out_num, kernel_size, scope, data_type='2D', **kws):
    if data_type == '2D':
        outs = tf.layers.conv2d_transpose(
            inputs, out_num, kernel_size, (2, 2), padding='same', name=scope,
            kernel_initializer=tf.truncated_normal_initializer)
    else:
        shape = 3*[kernel_size] + [out_num, out_num]
        input_shape = inputs.shape.as_list()
        out_shape = [input_shape[0]] + \
            list(map(lambda x: x*2, input_shape[1:-1])) + [out_num]
        weights = tf.get_variable(
            scope+'/deconv/weights', shape,
            initializer=tf.truncated_normal_initializer())
        outs = tf.nn.conv3d_transpose(
            inputs, weights, out_shape, (1, 2, 2, 2, 1), name=scope+'/deconv')
    return outs
    
def log(x):
    return tf.log(x + 1e-8)

def pool(inputs, kernel_size, scope, data_type='2D'):
    if data_type == '2D':
        return tf.layers.max_pooling2d(inputs, kernel_size, (2, 2), name=scope)
    return tf.layers.max_pooling3d(inputs, kernel_size, (2, 2, 2), name=scope)

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    PIXEL_MAX = 1 
    res = 10. * math.log10(PIXEL_MAX / math.sqrt(mse) + 10e-8)
    return res

def ssim(img1, img2):
    from skimage.measure import compare_ssim as ssim
    res = ssim(img1, img2, data_range=img2.max() - img2.min())
    return res

def sharpness(img1, img2, data_type):
    if data_type == '3D':
        gx, gy, gz = np.array(np.gradient(img1)) - np.array(np.gradient(img2))
        gradients = abs(gx)+abs(gy)+abs(gz)
    else:
        gx, gy = np.array(np.gradient(img1)) - np.array(np.gradient(img2))
        gradients = abs(gx)+abs(gy)       
    PIXEL_MAX = 1 
    res = 10. * math.log10( PIXEL_MAX /np.average(gradients) )
    return res


def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap='gray', origin='lower')

def saveimg_png(filename, a_p, gen_a, b_p, gen_b, data_type):
    for i, (ap, ga, bp, gb) in enumerate(zip(a_p, gen_a, b_p, gen_b)):
        if data_type == '3D':
            if i>19:
                break
            center_z = ap.shape[2] // 2
            slice_A, slice_B, slice_C, slice_D = ap[:, :, center_z], ga[:, :, center_z], bp[:, :, center_z], gb[:, :, center_z]
        else:
            if i >19:
                break
            slice_A, slice_B, slice_C, slice_D = ap, ga, bp, gb
        show_slices([slice_A, slice_B, slice_C, slice_D])
        plt.savefig(filename+'_'+str(i))
        #time.sleep(1)
        plt.close()
