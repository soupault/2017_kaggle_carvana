# coding: utf-8

from keras.models import Model
from keras.layers.core import Dropout, Activation
from keras.layers import (Input, Conv2D,
                          MaxPooling2D, UpSampling2D,
                          Concatenate)
from keras.layers.normalization import BatchNormalization

from .layers import ConvOffset2D


# Number of image channels (e.g. 3 in case of RGB, or 1 for grayscale)
INPUT_CHANNELS = 3
# Number of output masks (1 in case you predict only one type of objects)
OUTPUT_MASK_CHANNELS = 1


def preprocess_batch(batch):
    batch /= 256
    batch -= 0.5
    return batch


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


def UNET_DEFORM_224(dropout_val=0.05, batch_norm=True):
    inputs = Input((224, 224, INPUT_CHANNELS))  # TensorFlow backend, dim_order
    # inputs = Input((INPUT_CHANNELS, 224, 224))
    conv1 = double_conv_layer(inputs, 32, 3, dropout_val, batch_norm)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    offset2 = ConvOffset2D(32, name='conv2_offset')(pool1)
    conv2 = double_conv_layer(offset2, 64, 3, dropout_val, batch_norm)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = double_conv_layer(pool2, 128, 3, dropout_val, batch_norm)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    offset4 = ConvOffset2D(128, name='conv4_offset')(pool3)
    conv4 = double_conv_layer(offset4, 256, 3, dropout_val, batch_norm)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = double_conv_layer(pool4, 512, 3, dropout_val, batch_norm)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    offset6 = ConvOffset2D(512, name='conv6_offset')(pool5)
    conv6 = double_conv_layer(offset6, 1024, 3, dropout_val, batch_norm)

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


def UNET_DEFORM_224_336(dropout_val=0.05, batch_norm=True):
    inputs = Input((224, 336, INPUT_CHANNELS))  # TensorFlow backend, dim_order
    # inputs = Input((INPUT_CHANNELS, 224, 224))
    conv1 = double_conv_layer(inputs, 32, 3, dropout_val, batch_norm)
    pool1 = MaxPooling2D(pool_size=(2, 3))(conv1)

    offset2 = ConvOffset2D(32, name='conv2_offset')(pool1)
    conv2 = double_conv_layer(offset2, 64, 3, dropout_val, batch_norm)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = double_conv_layer(pool2, 128, 3, dropout_val, batch_norm)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    offset4 = ConvOffset2D(128, name='conv4_offset')(pool3)
    conv4 = double_conv_layer(offset4, 256, 3, dropout_val, batch_norm)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = double_conv_layer(pool4, 512, 3, dropout_val, batch_norm)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = double_conv_layer(pool5, 1024, 3, dropout_val, batch_norm)

    up6 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv6), conv5])
    conv7 = double_conv_layer(up6, 512, 3, dropout_val, batch_norm)

    offset7 = ConvOffset2D(512, name='conv7_offset')(conv7)
    up7 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(offset7), conv4])
    conv8 = double_conv_layer(up7, 256, 3, dropout_val, batch_norm)

    up8 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv8), conv3])
    conv9 = double_conv_layer(up8, 128, 3, dropout_val, batch_norm)

    offset9 = ConvOffset2D(128, name='conv9_offset')(conv9)
    up9 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(offset9), conv2])
    conv10 = double_conv_layer(up9, 64, 3, dropout_val, batch_norm)

    up10 = Concatenate(axis=-1)([UpSampling2D(size=(2, 3))(conv10), conv1])
    conv11 = double_conv_layer(up10, 32, 3, 0, batch_norm)

    conv12 = Conv2D(OUTPUT_MASK_CHANNELS, kernel_size=1, strides=1)(conv11)
    conv12 = BatchNormalization(axis=-1)(conv12)
    conv12 = Activation('sigmoid')(conv12)

    model = Model(inputs=[inputs], outputs=[conv12])
    return model
