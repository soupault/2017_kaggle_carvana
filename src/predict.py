import os
import glob
import argparse
from joblib import Parallel, delayed

import numpy as np
import scipy.ndimage as ndi
# from skimage import io
import cv2

# from h5py_shuffled_batch_gen import Dataset
from models.unet_zf import UNET_ZF_224, UNET_320_480
from models.unet_deform import UNET_DEFORM_224, UNET_DEFORM_224_336
from utils import (_resize_encode_mask,
                   _mask_to_rle_string)


def parse_args():
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_imgs', required=True,
                        help='Path to images to predict')
    parser.add_argument('--path_masks_low', required=True,
                        help='Where to save raw predicts')
    parser.add_argument('--weights', required=True,
                        help='DNN weights to load')
    parser.add_argument('--batch_size', required=False, type=int, default=32)

    parser.add_argument('--num_batches', required=False, default=None,
                        help=('DEBUG. Number of batches to select. '
                              'By default, uses all images.'))

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = parse_args()
    if not os.path.exists(config.path_masks_low):
        os.makedirs(config.path_masks_low)

    # model = UNET_ZF_224()
    # model = UNET_320_480()
    # model = UNET_DEFORM_224()
    model = UNET_DEFORM_224_336()
    model.load_weights(config.weights)

    fnames = glob.glob(os.path.join(config.path_imgs, '*'))

    # Create batches of filenames
    num_files_ceil = (((len(fnames) - 1) // config.batch_size + 1) *
                      config.batch_size)
    idxs_s = np.arange(0, num_files_ceil - config.batch_size + 1,
                       config.batch_size)
    idxs_e = np.arange(config.batch_size, num_files_ceil + 1,
                       config.batch_size)
    total_intvls = len(idxs_s)

    # Select a fraction of the data if specified
    if config.num_batches is not None:
        idxs_s = idxs_s[:config.num_batches]
        idxs_e = idxs_e[:config.num_batches]

    results = dict()
    shape_out = (1280, 1918)

    # Prepare jobs
    def job_read_mask(fname):
        return cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

    def job_read_img(fname):
        return cv2.imread(fname)[:, :, ::-1]

    def job_write_mask(fname, image):
        return cv2.imwrite(fname, image)

    struct = ndi.morphology.generate_binary_structure(2, 1)
    # struct2 = ndi.morphology.iterate_structure(struct, 2)
    struct_rep = ndi.morphology.iterate_structure(struct, 4)

    def job_postprocess(image_bin):
        tmp = image_bin
        # tmp = ndi.morphology.binary_erosion(image_bin, struct_rep)
        # tmp = ndi.morphology.binary_opening(image_bin, struct3)
        # image_filt = ndi.morphology.binary_closing(image_filt, struct3)
        return tmp

    def job_binarize_postproc_upsize(image):
        # For `dice loss`
        # image_bin = np.not_equal(image, 0)
        # For `binary cross-entropy loss`
        image_bin = image > 0.7
        image_filt = job_postprocess(image_bin)
        image_up = cv2.resize(image_filt.astype(np.uint8), shape_out[::-1])
        image_up = np.not_equal(image_up, 0)
        return image_up

    def job_mask_to_rle(mask):
        return _mask_to_rle_string(mask)

    def job_file_mask_low_to_rle(mask):
        tmp = job_read_mask(mask)
        tmp = job_binarize_postproc_upsize(tmp)
        tmp = job_mask_to_rle(tmp)
        return tmp

    def job_mask_low_to_rle(mask):
        tmp = job_binarize_postproc_upsize(mask)
        tmp = job_mask_to_rle(tmp)
        return tmp

    # Run prediction pipeline
    with Parallel(n_jobs=4, verbose=4) as parallel:
        for idx_intvl, (idx_s, idx_e) in enumerate(zip(idxs_s, idxs_e)):
            # Get batch filenames
            batch_fnames_in = fnames[idx_s:idx_e]
            batch_bnames = [os.path.basename(e) for e in batch_fnames_in]

            # *****************************************************************
            # Read images
            # print('...Batch read')
            ret = parallel(delayed(job_read_img)(e)
                           for e in batch_fnames_in)
            batch_images = np.asarray(ret)

            # Predict masks
            # print('...Batch predict')
            batch_masks_low = np.squeeze(model.predict_on_batch(batch_images))

            # Save predicts
            # print('...Batch save predicts')
            if False:
                batch_fnames_out = [os.path.join(config.path_masks_low, e)
                                    for e in batch_bnames]
                parallel(delayed(job_write_mask)(f, m)
                         for f, m in zip(batch_fnames_out,
                                         batch_masks_low))
            # *****************************************************************

            # *****************************************************************
            # Read, upscale, and apply RLE
            # print('...Batch read encode')
            if False:
                rles = parallel(delayed(job_file_mask_low_to_rle)(e)
                                for e in batch_fnames_out)
            rles = parallel(delayed(job_mask_low_to_rle)(e)
                            for e in batch_masks_low)

            # Aggregate the results
            batch_bnames_o = [e.replace('.jpg', '.png') for e in batch_bnames]
            results.update(zip(batch_bnames_o, rles))
            # *****************************************************************

            if idx_intvl % 10 == 0:
                print('Batch {} of {}'.format(idx_intvl, total_intvls))

    print('Processing finished')

    # *****************************************************************
    # Write submission file
    with open('temp_submission.csv', 'w') as f:
        print('Creating submission file')
        f.writelines(['img,rle_mask\n'])
        lines = ['{},{}\n'.format(k, v)
                 for k, v in sorted(results.items())]
        f.writelines(lines)
        print('Submission file successfully written')
    # *****************************************************************
