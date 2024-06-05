import argparse
import os

from cleanfid import fid
import logging

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default="coco-30k")
    parser.add_argument('--mode', type=str, default="legacy_pytorch")
    parser.add_argument('--result_dir', type=str, required=True, help="Directory to save the results")
    return parser.parse_args()


def main():
    args = parse_args()
    fid_value = fid.compute_fid(args.gen_dir, dataset_name=args.dataset, mode=args.mode, dataset_split="custom")
    logging.info(f"FID: {fid_value}")

    os.makedirs(args.result_dir, exist_ok=True)

    with open(f"{args.result_dir}/fid.txt", "a") as f:
        f.write(f"{args.gen_dir} {fid_value}\n")


if __name__ == '__main__':
    main()
