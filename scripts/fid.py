import argparse
import os

from PIL import Image
from cleanfid import fid
import logging

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, default="legacy_pytorch")
    return parser.parse_args()


def main():
    args = parse_args()
    # iterate over ref_dir and delete corrupted images
    for file in os.listdir(args.ref_dir):
        try:
            Image.open(os.path.join(args.ref_dir, file))
        except:
            os.remove(os.path.join(args.ref_dir, file))

    fid_value = fid.compute_fid(args.ref_dir, args.gen_dir, mode=args.mode)
    logging.info(f"FID value: {fid_value}")

if __name__ == '__main__':
    main()
