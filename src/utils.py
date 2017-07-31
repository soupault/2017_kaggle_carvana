import os
import glob
import argparse
from joblib import Parallel, delayed

import numpy as np
from skimage import transform
import cv2


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


def _roundtrip_resize(fname_in, fname_out, shape_out, binarize):
    """Job. Read, resize, and write image. Optionally, binarize."""
    image = cv2.imread(fname_in)[:, :, ::-1]

    image_resh = transform.resize(image, shape_out,
                                  preserve_range=True,
                                  mode='reflect').astype(np.uint8)
    if binarize:
        image_resh[image_resh > 127] = 255
        image_resh[image_resh <= 127] = 0

    cv2.imwrite(fname_out, image_resh[:, :, ::-1])

    
def batch_downscale(path_in, path_out, shape, binarize):
    """Parallel downscaling of all images within folder."""
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
    Parallel(n_jobs=4, verbose=4)(delayed(_roundtrip_resize)(*job_args)
                                  for job_args in jobs_args)


def _mask_to_rle_string(mask):
    """Convert boolean/`binary uint` mask to RLE string."""
    # Mask to RLE
    pixels = mask.T.flatten()
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]

    # RLE to string
    return ' '.join(str(x) for x in runs)


def _rle_string_to_mask(rle_string, shape=None):
    """Convert RLE string to uint8 binary mask."""
    if shape is None:
        shape = (1280, 1913)

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    # mask = np.zeros(shape[0] * shape[1], dtype=np.bool)

    rle = rle_string.split(' ')

    for i in range(0, len(rle) - 1, 2):
        idx_s, n = int(rle[i]) - 1, int(rle[i+1])
        idx_e = idx_s + n
        mask[idx_s:idx_e] = 255
        # mask[idx_s:idx_e] = True

    return mask.reshape(shape[::-1]).T


def _resize_encode_mask(mask, shape_out):
    """Job. Resize, binarize, and encode image into RLE string."""
    mask_resz = cv2.resize(mask, shape_out[::-1])
    mask_resz = mask_resz > 0.5
    mask_rle = _mask_to_rle_string(mask_resz)
    return mask_rle


def batch_upscale_encode(batch_masks):
    """Parallel upscale, binarize, and encode masks into RLE strings."""
    shape = (1280, 1913)

    # Execute jobs
    ret = Parallel(n_jobs=4, verbose=4)(
        delayed(_resize_encode_mask)(mask, shape) for mask in batch_masks)

    return ret


if __name__ == '__main__':
    pass
    # args = parse_args()

    # batch_downscale(args.path_in, args.path_out, args.shape, args.binarize)
