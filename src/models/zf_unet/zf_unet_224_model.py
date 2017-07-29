# coding: utf-8
'''
    - "ZF_UNET_224" Model based on UNET code from following paper:
          https://arxiv.org/abs/1505.04597
    - This model used to get 2nd place in DSTL competition:
          https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection
    - For training used DICE coefficient:
          https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    - Input shape for model is 224x224 (the same as for other popular CNNs like VGG or ResNet)
    - It has 3 input channels (to process standard RGB (BGR) images).
      You can change it with variable "INPUT_CHANNELS"
    - It trained on random image generator with
      random light shapes (ellipses) on dark background with noise (< 10%).
    - In most cases model ZF_UNET_224 is ok to be used without pretrained weights.
    - This code was checked for Theano backend only
'''

__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

from keras.models import Model
from keras.layers.core import Dropout, Activation
from keras.layers import (Input, Conv2D,
                          MaxPooling2D, UpSampling2D,
                          Concatenate)
from keras.layers.normalization import BatchNormalization
from keras import backend as K


# Number of image channels (e.g. 3 in case of RGB, or 1 for grayscale)
INPUT_CHANNELS = 3
# Number of output masks (1 in case you predict only one type of objects)
OUTPUT_MASK_CHANNELS = 1


def preprocess_batch(batch):
    batch /= 256
    batch -= 0.5
    return batch


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) /\
        (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) /\
        (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def double_conv_layer(x, filters, kernel_size, dropout, batch_norm):
    conv = Conv2D(filters, kernel_size=kernel_size,
                  strides=1, padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=-1)(conv)
    conv = Activation('relu')(conv)

    conv = Conv2D(filters, kernel_size=kernel_size,
                  strides=1, padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=-1)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = Dropout(dropout)(conv)
    return conv


def ZF_UNET_224(dropout_val=0.05, batch_norm=True):
    inputs = Input((224, 224, INPUT_CHANNELS))
    # inputs = Input((INPUT_CHANNELS, 224, 224))
    conv1 = double_conv_layer(inputs, 32, 3, dropout_val, batch_norm)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = double_conv_layer(pool1, 64, 3, dropout_val, batch_norm)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = double_conv_layer(pool2, 128, 3, dropout_val, batch_norm)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = double_conv_layer(pool3, 256, 3, dropout_val, batch_norm)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = double_conv_layer(pool4, 512, 3, dropout_val, batch_norm)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = double_conv_layer(pool5, 1024, 3, dropout_val, batch_norm)

    up6 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv6), conv5])
    conv7 = double_conv_layer(up6, 512, 3, dropout_val, batch_norm)

    up7 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv7), conv4])
    conv8 = double_conv_layer(up7, 256, 3, dropout_val, batch_norm)

    up8 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv8), conv3])
    conv9 = double_conv_layer(up8, 128, 3, dropout_val, batch_norm)

    up9 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv9), conv2])
    conv10 = double_conv_layer(up9, 64, 3, dropout_val, batch_norm)

    up10 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv10), conv1])
    conv11 = double_conv_layer(up10, 32, 3, 0, batch_norm)

    conv12 = Conv2D(OUTPUT_MASK_CHANNELS, kernel_size=1, strides=1)(conv11)
    conv12 = BatchNormalization(axis=-1)(conv12)
    conv12 = Activation('sigmoid')(conv12)

    model = Model(inputs=[inputs], outputs=[conv12])
    return model
