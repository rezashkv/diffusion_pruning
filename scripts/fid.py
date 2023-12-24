import argparse
import os

from PIL import Image
from cleanfid import fid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_dir', type=str, required=True)
    parser.add_argument('--ref_dir', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    # iterate over ref_dir and delete corrupted images
    for file in os.listdir(args.ref_dir):
        try:
            Image.open(os.path.join(args.ref_dir, file))
        except:
            os.remove(os.path.join(args.ref_dir, file))

    fid_value = fid.compute_fid(args.ref_dir, args.gen_dir, mode="clean")
    print('FID: ', fid_value)


if __name__ == '__main__':
    main()
