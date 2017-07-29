import argparse

import numpy as np

from keras.optimizers import Adam
from keras.callbacks import (ProgbarLogger, ModelCheckpoint,
                             EarlyStopping, LearningRateScheduler,
                             TensorBoard, ReduceLROnPlateau)

from h5py_shuffled_batch_gen import Dataset
from models.zf_unet.zf_unet_224_model import (ZF_UNET_224,
                                              dice_coef_loss, dice_coef)


def parse_args():
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_cache', required=True)
    parser.add_argument('--path_imgs', required=True)
    parser.add_argument('--path_masks', required=True)
    parser.add_argument('--shape', required=False, type=int, default=224)
    parser.add_argument('--batch_size', required=False, type=int, default=64)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = parse_args()

    datagen = Dataset(cache_file=config.file_cache,
                      path_imgs=config.path_imgs,
                      path_masks=config.path_masks,
                      batch_size=config.batch_size)

    model = ZF_UNET_224()
    # model.load_weights("zf_unet_224.h5")
    optim = Adam()
    model.compile(optimizer=optim, loss=dice_coef_loss, metrics=[dice_coef])

    callbs = [
        # ProgbarLogger(count_mode='steps'),
        ModelCheckpoint(
            filepath='./weights/temp.{epoch:02d}-{val_loss:.2f}.hdf5',
            monitor='val_loss', period=5),
        # EarlyStopping(),
        # LearningRateScheduler(),
        TensorBoard(log_dir='./logs',
                    write_images=True),
        # ReduceLROnPlateau()
    ]

    # model.fit(...)

    model.fit_generator(
        generator=datagen.batch_iterator(subset='train'),
        # steps_per_epoch=datagen.num_examples // config.batch_size,
        steps_per_epoch=datagen.num_examples['train'] // config.batch_size / 10,
        epochs=50,
        verbose=2,
        callbacks=callbs,
        validation_data=datagen.batch_iterator(subset='test'),
        validation_steps=100
        # validation_steps=100
    )

    # for idx_step, data_batch in enumerate(datagen.batch_iterator()):
    #     _, _, X_batch, y_batch, _ = data_batch
    #     loss = model.train_on_batch(
    #         np.asarray(X_batch[0]),
    #         np.asarray(y_batch[0]))
    #     print('Step: {}, loss: {}'.format(idx_step, loss))
