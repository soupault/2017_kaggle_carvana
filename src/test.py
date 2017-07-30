import os
import glob
import argparse
import pickle
from joblib import Parallel, delayed

import numpy as np
# from scipy.misc import imresize
from skimage import io

# from h5py_shuffled_batch_gen import Dataset
from models.zf_unet_224 import ZF_UNET_224
from utils import batch_upscale_encode


def parse_args():
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_cache', required=True)
    parser.add_argument('--path_imgs', required=True)
    parser.add_argument('--path_masks', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--shape', required=False, type=int, default=224)
    parser.add_argument('--batch_size', required=False, type=int, default=32)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = parse_args()

    # datagen = Dataset(cache_file=config.file_cache,
    #                   path_imgs=config.path_imgs,
    #                   path_masks=config.path_masks,
    #                   batch_size=config.batch_size)

    model = ZF_UNET_224()
    model.load_weights(config.weights)
    # model.load_weights('./weights_224_100epochs/temp.99--1.94.hdf5')

    fnames = glob.glob(os.path.join(config.path_imgs, '*'))

    # Create batches of filenames
    num_files_ceil = (len(fnames) // config.batch_size) * config.batch_size
    idxs_start = np.arange(0, num_files_ceil - config.batch_size,
                           config.batch_size)
    idxs_end = np.arange(config.batch_size, num_files_ceil,
                         config.batch_size)
    total_intvls = len(idxs_start)

    # XXX: debug
    # idxs_start = idxs_start[:2]
    # idxs_end = idxs_end[:2]
    # XXX: end

    results = dict()

    for idx_intvl, (idx_start, idx_end) in enumerate(zip(idxs_start, idxs_end)):
        # Get batch filenames
        batch_fnames = fnames[idx_start:idx_end]

        # Read batch of images to process
        def job(fname):
            return io.imread(fname)

        res = Parallel(n_jobs=4, verbose=5)(
            delayed(job)(fname) for fname in batch_fnames)
        batch_images = np.asarray(res)

        # Predict masks
        masks_downsz = model.predict_on_batch(batch_images)

        # Upscale and apply RLE
        batch_bnames = [os.path.basename(e) for e in batch_fnames]
        result = batch_upscale_encode(batch_bnames, masks_downsz)

        # Aggregate the results
        results.update(result)

        if idx_intvl % 10 == 0:
            print('Batch {} of {}'.format(idx_intvl, total_intvls))

    # XXX: DEBUG
    # print(results)
    # XXX: END

    with open('temp_submission.pkl', 'wb') as f:
        pickle.dump(results, f)
        print('Submission data has been pickled')

    # Write submission file
    with open('temp_submission.csv', 'w') as f:
        f.writelines(['img,rle_mask\n'])
        lines = ['{},{}\n'.format(k, v) for k, v in results.items()]
        f.writelines(lines)
        print('Submission file successfully written')
