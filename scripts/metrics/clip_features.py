import argparse
from pdm.utils.clip_utils import clip_features
import logging

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--clip_model', type=str, default="ViT-B/32")
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    clip_features(args.dataset_path, clip_model=args.clip_model, num_workers=args.num_workers,
                  batch_size=args.batch_size)


if __name__ == '__main__':
    main()
