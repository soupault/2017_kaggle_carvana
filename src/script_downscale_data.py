import argparse

from utils import batch_downscale


def parse_args():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', required=True)
    parser.add_argument('--path_out', required=True)
    parser.add_argument('--shape_row', required=False, type=int, default=224)
    parser.add_argument('--shape_col', required=False, type=int, default=224)
    parser.add_argument('--binarize', action='store_true', default=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = parse_args()

    config.shape = (config.shape_row, config.shape_col)
    batch_downscale(path_in=config.path_in,
                    path_out=config.path_out,
                    shape=config.shape,
                    binarize=config.binarize)
