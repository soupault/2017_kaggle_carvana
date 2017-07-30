import os
import glob
import argparse
from joblib import Parallel, delayed

import numpy as np
from skimage import io, transform


def parse_args():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', required=True)
    parser.add_argument('--path_out', required=True)
    parser.add_argument('--shape', required=False, type=int, default=224)
    parser.add_argument('--binarize', action='store_true', default=False)

    args = parser.parse_args()
    return args


def _roundtrip_rescale(fname_in, fname_out, shape_out, binarize):
    """
    """
    image = io.imread(fname_in)

    image_resh = transform.resize(
        image, shape_out, preserve_range=True).astype(np.uint8)
    if binarize:
        image_resh[image_resh > 127] = 255
        image_resh[image_resh <= 127] = 0

    io.imsave(fname_out, image_resh)

    
def batch_downscale(path_in, path_out, shape, binarize):
    """
    """
    # Create a list of files to process
    fnames_in = glob.glob(os.path.join(path_in, '*.*'))
    fnames_out = [os.path.join(path_out, os.path.basename(e))
                  for e in fnames_in]

    if not os.path.exists(path_out):
        os.makedirs(path_out)

    # Schedule jobs
    jobs_args = [(fin, fout, (shape, )*2, binarize)
                 for fin, fout in zip(fnames_in, fnames_out)]

    # Execute jobs
    Parallel(n_jobs=4, verbose=5)(delayed(_roundtrip_rescale)(*job_args)
                                  for job_args in jobs_args)


def _mask_to_rle_string(mask):
    """
    """
    # Mask to RLE
    pixels = mask.flatten()
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    # XXX: DEBUG
    # print('Runs: {}'.format(str(runs)))

    # RLE to string
    return ' '.join(str(x) for x in runs)


def _resize_encode_mask(fname, mask, shape_out):
    """
    """
    mask_resz = transform.resize(mask, shape_out, preserve_range=True)
    mask_resz = mask_resz > 0.5
    mask_rle = _mask_to_rle_string(mask_resz)
    return fname, mask_rle


def batch_upscale_encode(batch_fnames, batch_masks):
    """
    """
    SHAPE_OUT = (1280, 1913, 1)

    # Schedule jobs
    jobs_args = [(fname, mask, SHAPE_OUT)
                 for fname, mask in zip(batch_fnames, batch_masks)]

    # Execute jobs
    res = Parallel(n_jobs=4, verbose=5)(
        delayed(_resize_encode_mask)(*job_args) for job_args in jobs_args)

    # Aggregate results
    results = dict((res))

    return results


if __name__ == '__main__':
    args = parse_args()

    batch_downscale(args.path_in, args.path_out, args.shape, args.binarize)
