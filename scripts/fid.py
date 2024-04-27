import argparse
from cleanfid import fid
import logging

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default="coco-30k")
    parser.add_argument('--mode', type=str, default="legacy_pytorch")
    return parser.parse_args()


def main():
    args = parse_args()
    fid_value = fid.compute_fid(args.ref_dir, dataset_name=args.dataset, mode=args.mode, dataset_split="custom")
    logging.info(f"FID value: {fid_value}")

if __name__ == '__main__':
    main()
