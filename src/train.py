import os
import argparse

import numpy as np

from keras.optimizers import Adam
from keras.callbacks import (ProgbarLogger, ModelCheckpoint,
                             EarlyStopping, LearningRateScheduler,
                             TensorBoard, ReduceLROnPlateau)

from h5py_shuffled_batch_gen import DatasetTrain
from models.unet_zf import UNET_ZF_224, UNET_320_480
from models.unet_deform import UNET_DEFORM_224, UNET_DEFORM_224_336
from models.losses_metrics import (dice_coef_loss, dice_coef,
                                   jacard_coef_loss, jacard_coef)


def parse_args():
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_cache', required=True)
    parser.add_argument('--path_imgs', required=True)
    parser.add_argument('--path_masks', required=True)
    parser.add_argument('--shape_row', required=False, type=int, default=224)
    parser.add_argument('--shape_col', required=False, type=int, default=224)
    parser.add_argument('--batch_size', required=False, type=int, default=64)
    parser.add_argument('--num_epochs', required=False, type=int, default=100)
    parser.add_argument('--ckpt_epochs', required=False, type=int, default=10)
    parser.add_argument('--weights_init', required=False, default=None)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = parse_args()

    datagen = DatasetTrain(cache_file=config.file_cache,
                           path_imgs=config.path_imgs,
                           path_masks=config.path_masks,
                           batch_size=config.batch_size)

    # model = ZF_UNET_224()
    # model = ZF_UNET_320_480()
    # model = UNET_DEFORM_224()
    model = UNET_DEFORM_224_336()
    if config.weights_init is not None:
        model.load_weights(config.weights_init)

    path_weights = './weights'
    path_logs = './logs'
    for p in (path_weights, path_logs):
        if not os.path.exists(p):
            os.makedirs(p)
            print('Created folder {}'.format(p))

    # optim = Adam()
    optim = Adam(decay=0.001)
    model.compile(optimizer=optim,
                  loss='binary_crossentropy',
                  # loss=dice_coef_loss,
                  metrics=[dice_coef, jacard_coef])

    callbs = [
        # ProgbarLogger(count_mode='steps'),
        ModelCheckpoint(
            filepath=path_weights+'/temp.{epoch:02d}-{val_loss:.2f}.hdf5',
            monitor='val_loss', period=config.ckpt_epochs),
        # EarlyStopping(),
        # LearningRateScheduler(),
        TensorBoard(log_dir=path_logs,
                    write_images=False),
        # ReduceLROnPlateau()
    ]

    model.fit_generator(
        generator=datagen.batch_iterator(subset='train',
                                         batch_size=config.batch_size,
                                         num_epochs=config.num_epochs),
        # steps_per_epoch=datagen.num_examples // config.batch_size,
        steps_per_epoch=(datagen.num_examples['train'] //
                         config.batch_size // 10),
        epochs=config.num_epochs,
        verbose=2,
        callbacks=callbs,
        validation_data=datagen.batch_iterator(subset='val',
                                               batch_size=config.batch_size,
                                               num_epochs=config.num_epochs),
        validation_steps=(datagen.num_examples['val'] //
                          config.batch_size // 10)
        # validation_steps=30
    )
