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
    pixels = mask.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    # pixels = mask.swapaxes(0, 1).flatten()
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]

    # RLE to string
    return ' '.join(str(x) for x in runs)


def _rle_string_to_mask(rle_string, shape=None):
    """Convert RLE string to uint8 binary mask."""
    if shape is None:
        shape = (1280, 1918)

    mask = np.zeros(shape[0] * shape[1], dtype=np.bool)

    rle = rle_string.split(' ')

    for i in range(0, len(rle) - 1, 2):
        idx_s, n = int(rle[i]) - 1, int(rle[i+1])
        idx_e = idx_s + n
        mask[idx_s:idx_e] = True

    return mask.reshape(shape)


def _resize_encode_mask(mask, shape_out):
    """Job. Resize, binarize, and encode image into RLE string."""
    mask_resz = cv2.resize(mask, shape_out[::-1])
    mask_resz = mask_resz > 0.5
    mask_rle = _mask_to_rle_string(mask_resz)
    return mask_rle


def batch_upscale_encode(batch_masks):
    """Parallel upscale, binarize, and encode masks into RLE strings."""
    shape = (1280, 1918)

    # Execute jobs
    ret = Parallel(n_jobs=4, verbose=4)(
        delayed(_resize_encode_mask)(mask, shape) for mask in batch_masks)

    return ret


# WIP
# def check_rle():

#     if 0: #check one mask file
#         #opencv does not read gif
#         mask_file = '/root/share/[data]/kaggle-carvana-cars-2017/annotations/train_masks/0cdf5b5d0ce1_01_mask.gif'
#         mask = PIL.Image.open(mask_file)  #cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
#         mask = np.array(mask)

#         #im_show('mask', mask*255, resize=0.25)
#         #cv2.waitKey(0)
#         mask1 = cv2.resize(mask,(0,0), fx=0.25, fy=0.25)
#         im_show('mask1', mask1*255, resize=1)
#         rle = run_length_encode(mask1)

#         cv2.waitKey(0)

#     if 1: #check with train_masks.csv given

#         csv_file  = CARVANA_DIR + '/masks_train.csv'  # read all annotations
#         mask_dir  = CARVANA_DIR + '/annotations/train'  # read all annotations
#         df  = pd.read_csv(csv_file)
#         for n in range(10):
#             shortname = df.values[n][0].replace('.jpg','')
#             rle_hat   = df.values[n][1]

#             mask_file = mask_dir + '/' + shortname + '_mask.gif'
#             mask = PIL.Image.open(mask_file)
#             mask = np.array(mask)
#             rle  = run_length_encode(mask)
#             #im_show('mask', mask*255, resize=0.25)
#             #cv2.waitKey(0)
#             match = rle == rle_hat
#             print('%d match=%s'%(n,match))


if __name__ == '__main__':
    pass
    # args = parse_args()

    # batch_downscale(args.path_in, args.path_out, args.shape, args.binarize)
