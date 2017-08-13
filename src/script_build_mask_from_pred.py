import os
import argparse
from joblib import Parallel, delayed

# import numpy as np
import pandas as pd
import imageio

# import cv2

from utils import _rle_string_to_mask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename_in', required=False,
                        help='Submission .csv file')
    parser.add_argument('--path_out', required=False,
                        help='Where to save the prediction masks')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = parse_args()
    config.filename_in = '../submissions/17_224_336_200epoch_bce_nopostproc_fixed.csv'
    config.path_out = '../data/predicts_full_224_336'

    if not os.path.exists(config.path_out):
        os.makedirs(config.path_out)
    else:
        print('Output directory already exists')

    df = pd.read_csv(config.filename_in)

    def job(fname, rle):
        image = _rle_string_to_mask(rle)
        imageio.imwrite(fname, image)

    with Parallel(n_jobs=4, verbose=4) as parallel:
        fnames = [os.path.join(config.path_out, e) for e in df['img']]
        rles = df['rle_mask'].tolist()

        parallel(delayed(job)(f, r) for f, r in zip(fnames, rles))
