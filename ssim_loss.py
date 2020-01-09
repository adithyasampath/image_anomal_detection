from __future__ import absolute_import
import keras_contrib.backend as KC
from keras import backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Flatten, Activation, BatchNormalization, Conv2DTranspose, LeakyReLU, GlobalAveragePooling2D,concatenate
from keras.models import Sequential, Model
from keras import applications
import keras
import os
import cv2
import csv
from tensorflow.contrib import lite
from skimage.measure import compare_ssim
import time
import losses


def gaussian(x, mu, sigma):
    return np.exp(-(float(x) - float(mu)) ** 2 / (2 * sigma ** 2))


def make_kernel(sigma):
    # kernel radius = 2*sigma, but minimum 3x3 matrix
    kernel_size = max(3, int(2 * 2 * sigma + 1))
    mean = np.floor(0.5 * kernel_size)
    kernel_1d = np.array([gaussian(x, mean, sigma) for x in range(kernel_size)])
    # make 2D kernel
    np_kernel = np.outer(kernel_1d, kernel_1d).astype(dtype=K.floatx())
    # normalize kernel by sum of elements
    kernel = np_kernel / np.sum(np_kernel)
    kernel = np.reshape(kernel, (kernel_size, kernel_size, 1,1))    #height, width, in_channels, out_channel
    return kernel

def keras_SSIM_cs(y_true, y_pred):
    axis=None
    gaussian = make_kernel(1.5)
    x = tf.nn.conv2d(y_true, gaussian, strides=[1, 1, 1, 1], padding='SAME')
    y = tf.nn.conv2d(y_pred, gaussian, strides=[1, 1, 1, 1], padding='SAME')

    u_x=K.mean(x, axis=axis)
    u_y=K.mean(y, axis=axis)

    var_x=K.var(x, axis=axis)
    var_y=K.var(y, axis=axis)

    cov_xy=cov_keras(x, y, axis)

    K1=0.01
    K2=0.03
    L=1  # depth of image (255 in case the image has a differnt scale)

    C1=(K1*L)**2
    C2=(K2*L)**2
    C3=C2/2

    l = ((2*u_x*u_y)+C1) / (K.pow(u_x,2) + K.pow(u_x,2) + C1)
    c = ((2*K.sqrt(var_x)*K.sqrt(var_y))+C2) / (var_x + var_y + C2)
    s = (cov_xy+C3) / (K.sqrt(var_x)*K.sqrt(var_y) + C3)

    return [c,s,l]

def keras_MS_SSIM(y_true, y_pred):
    iterations = 5
    x=y_true
    y=y_pred
    weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    c=[]
    s=[]
    for i in range(iterations):
        cs=keras_SSIM_cs(x, y)
        c.append(cs[0])
        s.append(cs[1])
        l=cs[2]
        if(i!=4):
            x=tf.image.resize_images(x, (x.get_shape().as_list()[1]//(2**(i+1)), x.get_shape().as_list()[2]//(2**(i+1))))
            y=tf.image.resize_images(y, (y.get_shape().as_list()[1]//(2**(i+1)), y.get_shape().as_list()[2]//(2**(i+1))))
    c = tf.stack(c)
    s = tf.stack(s)
    cs = c*s

    #Normalize: suggestion from https://github.com/jorge-pessoa/pytorch-msssim/issues/2 last comment to avoid NaN values
    l=(l+1)/2
    cs=(cs+1)/2

    cs=cs**weight
    cs = tf.reduce_prod(cs)
    l=l**weight[-1]

    ms_ssim = l*cs
    ms_ssim = tf.where(tf.is_nan(ms_ssim), K.zeros_like(ms_ssim), ms_ssim)

    return K.mean(ms_ssim)


class DSSIMObjective:
    """Difference of Structural Similarity (DSSIM loss function).
    Clipped between 0 and 0.5
    Note : You should add a regularization term like a l2 loss in addition to this one.
    Note : In theano, the `kernel_size` must be a factor of the output size. So 3 could
           not be the `kernel_size` for an output of 32.
    # Arguments
        k1: Parameter of the SSIM (default 0.01)
        k2: Parameter of the SSIM (default 0.03)
        kernel_size: Size of the sliding window (default 3)
        max_value: Max value of the output (default 1.0)
    """

    def __init__(self, k1=0.01, k2=0.03, kernel_size=3, max_value=1.0):
        self.__name__ = 'DSSIMObjective'
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.c1 = (self.k1 * self.max_value) ** 2
        self.c2 = (self.k2 * self.max_value) ** 2
        self.dim_ordering = K.image_data_format()
        self.backend = K.backend()

    def __int_shape(self, x):
        return K.int_shape(x) if self.backend == 'tensorflow' else K.shape(x)

    def __call__(self, y_true, y_pred):
        # There are additional parameters for this function
        # Note: some of the 'modes' for edge behavior do not yet have a
        # gradient definition in the Theano tree
        #   and cannot be used for learning

        kernel = [self.kernel_size, self.kernel_size]
        y_true = K.reshape(y_true, [-1] + list(self.__int_shape(y_pred)[1:]))
        y_pred = K.reshape(y_pred, [-1] + list(self.__int_shape(y_pred)[1:]))

        patches_pred = KC.extract_image_patches(y_pred, kernel, kernel, 'valid',
                                                self.dim_ordering)
        patches_true = KC.extract_image_patches(y_true, kernel, kernel, 'valid',
                                                self.dim_ordering)

        # Reshape to get the var in the cells
        bs, w, h, c1, c2, c3 = self.__int_shape(patches_pred)
        patches_pred = K.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])
        patches_true = K.reshape(patches_true, [-1, w, h, c1 * c2 * c3])
        # Get mean
        u_true = K.mean(patches_true, axis=-1)
        u_pred = K.mean(patches_pred, axis=-1)
        # Get variance
        var_true = K.var(patches_true, axis=-1)
        var_pred = K.var(patches_pred, axis=-1)
        # Get std dev
        covar_true_pred = K.mean(patches_true * patches_pred, axis=-1) - u_true * u_pred

        ssim = (2 * u_true * u_pred + self.c1) * (2 * covar_true_pred + self.c2)
        denom = ((K.square(u_true)
                  + K.square(u_pred)
                  + self.c1) * (var_pred + var_true + self.c2))
        ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
        return K.mean((1.0 - ssim) / 2.0)