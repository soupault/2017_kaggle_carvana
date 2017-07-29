import os
import glob
import argparse
from joblib import Parallel, delayed

import numpy as np
from skimage import io, transform


def parse_args():
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', required=True)
    parser.add_argument('--path_out', required=True)
    parser.add_argument('--shape', required=False, type=int, default=224)
    parser.add_argument('--binarize', action='store_true', default=False)

    args = parser.parse_args()
    return args


def _roundtrip_rescale(fname_in, fname_out, shape_out, binarize):
    """"""
    image = io.imread(fname_in)

    image_resh = transform.resize(
        image, shape_out, preserve_range=True).astype(np.uint8)
    if binarize:
        image_resh[image_resh > 127] = 255
        image_resh[image_resh <= 127] = 0

    io.imsave(fname_out, image_resh)

    
def batch_downscale(args):
    """"""
    filenames_in = glob.glob(os.path.join(args.path_in, '*.*'))
    filenames_out = [os.path.join(args.path_out, os.path.basename(e))
                     for e in filenames_in]

    if not os.path.exists(args.path_out):
        os.makedirs(args.path_out)

    jobs = [(fin, fout, (args.shape, )*2, args.binarize)
            for fin, fout in zip(filenames_in, filenames_out)]

    Parallel(n_jobs=4, verbose=5)(delayed(_roundtrip_rescale)(*job)
                                  for job in jobs)


if __name__ == '__main__':
    args = parse_args()

    batch_downscale(args)
