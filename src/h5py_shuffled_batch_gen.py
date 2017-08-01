"""
__author__ = '@arseni, Arseni Anisimovich'

https://www.kaggle.com/arseni/h5py-dataset-caching-with-shuffled-batch-generator
"""

import os
import glob
import logging

import h5py
import numpy as np
from skimage.io import imread
from dask import delayed, threaded, compute
from sklearn.model_selection import train_test_split

logger = logging.getLogger('dataset loader')
logger.setLevel(logging.DEBUG)


class DatasetTrain:
    def __init__(self, cache_file, path_imgs, path_masks, batch_size=64,
                 train_val_ratio=0.8, random_seed=44):
        """
        Parameters
        ----------
        file : str
        cache_file : str
        path_imgs : str
        path_masks : str
        batch_size : int
        """
        self.cache_file = cache_file
        self.path_imgs = path_imgs
        self.path_masks = path_masks
        self.batch_size = batch_size
        self.train_val_ratio = train_val_ratio
        self.random_seed = random_seed

        if not os.path.exists(self.cache_file):
            self.h5_file = h5py.File(self.cache_file, 'w')
            logger.info('Creating cache {}'.format(self.cache_file))
            self._build_cache()
        else:
            self.h5_file = h5py.File(self.cache_file, 'r')
            self.num_examples = dict()
            for subset in ('train', 'val'):
                self.num_examples[subset] = \
                    len(self.h5_file['names_{}'.format(subset)])
                logger.info('Loaded h5py dataset `{}` of {} elements.'
                            .format(subset, self.num_examples[subset]))

    @staticmethod
    def _read_img(filename):
        """
        Parameters
        ----------
        filename : str
        """
        return imread(filename)
        # return np.clip(imread(filename), 0, 255).astype(np.uint8)

    def _build_cache(self):
        """ """
        # img_r, img_c = 1280, 1918
        # img_r, img_c = 224, 224
        img_r, img_c = 320, 480

        logger.info('Caching files from {} | {}'
                    .format(self.path_imgs, self.path_masks))

        fnames_imgs = {'total': os.listdir(self.path_imgs)}
        tmp = train_test_split(fnames_imgs['total'],
                               train_size=self.train_val_ratio,
                               random_state=self.random_seed)
        fnames_imgs['train'] = tmp[0]
        fnames_imgs['val'] = tmp[1]

        self.num_examples = {
            'total': len(fnames_imgs['total']),
            'train': len(fnames_imgs['train']),
            'val': len(fnames_imgs['val']),
        }

        for subset in ('train', 'val'):
            X_temp = self.h5_file.create_dataset(
                'X_{}'.format(subset),
                shape=(self.num_examples[subset], img_r, img_c, 3),
                dtype=np.uint8)
            y_temp = self.h5_file.create_dataset(
                'y_{}'.format(subset),
                shape=(self.num_examples[subset], img_r, img_c, 1),
                dtype=np.uint8)
            names_temp = self.h5_file.create_dataset(
                'names_{}'.format(subset),
                shape=(self.num_examples[subset],),
                dtype=h5py.special_dtype(vlen=str))

            logger.info('Total files in subset `{}`: {}'
                        .format(subset, self.num_examples[subset]))

            for idx, fname in enumerate(fnames_imgs[subset]):
                img = self._read_img(os.path.join(self.path_imgs, fname))
                mask = self._read_img(os.path.join(
                    self.path_masks, fname.replace('.jpg', '_mask.png'))
                ).reshape(img_r, img_c, 1)

                X_temp[idx, :, :, :] = img
                y_temp[idx, :, :, :] = mask
                names_temp[idx] = fname
                if idx % 100 == 0:
                    logger.info('Processed {} files.'.format(idx))

    def batch_iterator(self, subset, number_of_examples=None, batch_size=None,
                       num_epochs=10, shuffle=False):
        """Generates a batch iterator for a dataset.

        Parameters
        ----------
        subset : {'train', 'val'}
        number_of_examples : int
        batch_size : int
        num_epochs : int
        shuffle : bool

        Yields
        ------
        epoch : int
            Index of the current epoch.
        batch_num :
            Index of the batch within current epoch.
        X : (I, M, N, C)
            Batch of images.
        y : (I, M, N, C)
            Batch of masks.
        names :
            
        """
        if batch_size is None:
            batch_size = self.batch_size

        if number_of_examples is not None:
            data_size = number_of_examples
        else:
            data_size = self.num_examples[subset]

        X = self.h5_file['X_{}'.format(subset)]
        y = self.h5_file['y_{}'.format(subset)]
        names = self.h5_file['names_{}'.format(subset)]
        num_batches_per_epoch = int((data_size - 1) / batch_size) + 1

        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            shuffle_idxs = np.arange(data_size)
            if shuffle:
                shuffle_idxs = np.random.permutation(shuffle_idxs)

            for batch_num in range(num_batches_per_epoch):
                start_idx = batch_num * batch_size
                end_idx = min((batch_num + 1) * batch_size, data_size)
                batch_idxs = sorted(list(shuffle_idxs[start_idx:end_idx]))

                yield (np.asarray(compute(
                           [delayed(X.__getitem__)(i)
                            for i in batch_idxs], get=threaded.get)[0]),
                       np.asarray(compute(
                           [delayed(y.__getitem__)(i)
                            for i in batch_idxs], get=threaded.get)[0]))

                # yield (epoch,
                #        batch_num,
                #        compute([delayed(X.__getitem__)(i)
                #                 for i in batch_idxs], get=threaded.get)[0],
                #        compute([delayed(y.__getitem__)(i)
                #                 for i in batch_idxs], get=threaded.get)[0],
                #        compute([delayed(names.__getitem__)(i)
                #                 for i in batch_idxs], get=threaded.get)[0])


# class DatasetTest:
#     def __init__(self, path_imgs, batch_size=64):
#         """
#         Parameters
#         ----------
#         file : str
#         path_imgs : str
#         path_masks : str
#         batch_size : int
#         """
#         self.path_imgs = path_imgs
#         self.batch_size = batch_size
#         self.subset = 'test'

#         self.fnames = self._find_files()
#         self.num_examples = {'test': len(self.fnames)}
#         logger.info('Found dataset `{}` of {} elements.'
#                     .format(self.subset, self.num_examples[self.subset]))

#     @staticmethod
#     def _read_img(filename):
#         """
#         Parameters
#         ----------
#         filename : str
#         """
#         return imread(filename)
#         # return np.clip(imread(filename), 0, 255).astype(np.uint8)

#     def _find_files(self):
#         """ """
#         return glob.glob(os.path.join(self.path_imgs, '*'))

#     def batch_iterator(self, number_of_examples=None, batch_size=None,
#                        shuffle=False):
#         """Generates a batch iterator for a dataset.

#         Parameters
#         ----------
#         number_of_examples : int
#             Number of files to return from the full dataset.
#         batch_size : int
#             Number of files in a batch.
#         shuffle : bool

#         Yields
#         ------
#         epoch : int
#             Index of the current epoch.
#         batch_num :
#             Index of the batch within current epoch.
#         X : (I, M, N, C)
#             Batch of images.
#         y : (I, M, N, C)
#             Batch of masks.
#         names :
            
#         """
#         subset = self.subset

#         if batch_size is None:
#             batch_size = self.batch_size

#         if number_of_examples is not None:
#             data_size = min(number_of_examples, self.num_examples[subset])
#         else:
#             data_size = self.num_examples[subset]

#         X = self.h5_file['X_{}'.format(subset)]
#         names = self.h5_file['names_{}'.format(subset)]
#         num_batches = int((data_size - 1) / batch_size) + 1

#         # Shuffle the data at each epoch
#         shuffle_idxs = np.arange(data_size)
#         if shuffle:
#             shuffle_idxs = np.random.permutation(shuffle_idxs)

#         for batch_num in range(num_batches):
#             start_idx = batch_num * batch_size
#             end_idx = min((batch_num + 1) * batch_size, data_size)
#             batch_idxs = sorted(list(shuffle_idxs[start_idx:end_idx]))

#             yield (np.asarray(compute(
#                        [delayed(X.__getitem__)(i)
#                         for i in batch_idxs], get=threaded.get)[0]))
#             # yield (batch_num,
#             #        compute([delayed(X.__getitem__)(i)
#             #                 for i in batch_idxs], get=threaded.get)[0],
#             #        compute([delayed(y.__getitem__)(i)
#             #                 for i in batch_idxs], get=threaded.get)[0],
#             #        compute([delayed(names.__getitem__)(i)
#             #                 for i in batch_idxs], get=threaded.get)[0])
