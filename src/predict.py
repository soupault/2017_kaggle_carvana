import os
import glob
import argparse
from joblib import Parallel, delayed

import numpy as np
import scipy.ndimage as ndi
# from skimage import io
import cv2

# from h5py_shuffled_batch_gen import Dataset
from models.zf_unet import ZF_UNET_224, ZF_UNET_320_480
from utils import (_resize_encode_mask,
                   _mask_to_rle_string)


def parse_args():
    """"""
    parser = argparse.ArgumentParser()
    # parser.add_argument('--file_cache', required=True)
    parser.add_argument('--path_imgs', required=True)
    parser.add_argument('--path_masks_low', required=True,
                        help='Where to save raw predicts')
    parser.add_argument('--weights', required=True)
    # parser.add_argument('--shape', required=False, type=int, default=224)
    parser.add_argument('--batch_size', required=False, type=int, default=32)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = parse_args()
    if not os.path.exists(config.path_masks_low):
        os.makedirs(config.path_masks_low)

    # datagen = Dataset(cache_file=config.file_cache,
    #                   path_imgs=config.path_imgs,
    #                   path_masks=config.path_masks,
    #                   batch_size=config.batch_size)

    # model = ZF_UNET_224()
    model = ZF_UNET_320_480()
    model.load_weights(config.weights)

    fnames = glob.glob(os.path.join(config.path_imgs, '*'))

    # Create batches of filenames
    num_files_ceil = ((len(fnames) - 1) // config.batch_size + 1) * config.batch_size
    idxs_s = np.arange(0, num_files_ceil - config.batch_size + 1,
                       config.batch_size)
    idxs_e = np.arange(config.batch_size, num_files_ceil + 1,
                       config.batch_size)
    total_intvls = len(idxs_s)

    # XXX: debug
    # idxs_s = idxs_s[:2]
    # idxs_e = idxs_e[:2]
    # idxs_s = idxs_s[:20]
    # idxs_e = idxs_e[:20]
    # XXX: end

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
    struct3 = ndi.morphology.iterate_structure(struct, 3)

    def job_binarize_clean_upsize(image):
        image_bin = np.not_equal(image, 0)
        image_filt = ndi.morphology.binary_opening(image_bin, struct3)
        # image_filt = ndi.morphology.binary_closing(image_filt, struct3)
        image_up = cv2.resize(image_filt.astype(np.uint8), shape_out[::-1])
        image_up = np.not_equal(image_up, 0)
        return image_up

    def job_mask_to_rle(mask):
        return _mask_to_rle_string(mask)

    def job_file_mask_low_to_rle(mask):
        tmp = job_read_mask(mask)
        tmp = job_binarize_clean_upsize(tmp)
        tmp = job_mask_to_rle(tmp)
        return tmp

    # Run prediction pipeline
    with Parallel(n_jobs=4, verbose=4) as parallel:
        for idx_intvl, (idx_s, idx_e) in enumerate(zip(idxs_s, idxs_e)):
            # Get batch filenames
            batch_fnames_in = fnames[idx_s:idx_e]
            batch_bnames = [os.path.basename(e) for e in batch_fnames_in]

            # if False:
            if True:
                # Read images
                ret = parallel(delayed(job_read_img)(e)
                               for e in batch_fnames_in)
                batch_images = np.asarray(ret)

                # Predict masks
                batch_masks_low = model.predict_on_batch(batch_images)[:, :, :, 0]

                # Save predicts
                batch_fnames_out = [os.path.join(config.path_masks_low, e)
                                    for e in batch_bnames]
                parallel(delayed(job_write_mask)(f, m)
                         for f, m in zip(batch_fnames_out,
                                         batch_masks_low))

            # if True:
            if False:
                # XXX
                batch_fnames_out = batch_fnames_in

                # tmp = job_read_mask(batch_fnames_out[0])
                # print(np.unique(tmp), np.min(tmp), np.max(tmp))
                # tmp = job_binarize_clean_upsize(tmp)
                # print(np.unique(tmp), np.min(tmp), np.max(tmp))
                # tmp = job_write_mask('./debug.jpg', tmp)
                # print('OK')
                # quit()
                
                # Upscale and apply RLE
                rles = parallel(delayed(job_file_mask_low_to_rle)(e)
                                for e in batch_fnames_out)
                # rles = parallel(delayed(_resize_encode_mask)(mask, shape_out)
                #                 for mask in masks_downsz)

                # Aggregate the results
                results.update(zip(batch_bnames, rles))

            if idx_intvl % 10 == 0:
                print('Batch {} of {}'.format(idx_intvl, total_intvls))

    print('Processing finished')

    if False:
        # Write submission file
        with open('temp_submission.csv', 'w') as f:
            print('Creating submission file')
            f.writelines(['img,rle_mask\n'])
            lines = ['{},{}\n'.format(k, v) for k, v in sorted(results.items())]
            f.writelines(lines)
            print('Submission file successfully written')
