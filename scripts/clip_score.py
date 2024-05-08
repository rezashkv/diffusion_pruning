import argparse
import os
from pdm.utils.clip_utils import clip_score
import logging

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_dir', type=str, required=True)
    parser.add_argument('--text_features_path', type=str, required=True)
    parser.add_argument('--clip_model', type=str, default="ViT-B/32")
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--result_dir', type=str, required=True, help="Directory to save the results")
    return parser.parse_args()


def main():
    args = parse_args()
    score = clip_score(args.text_features_path, args.gen_dir, clip_model=args.clip_model, num_workers=args.num_workers,
                       batch_size=args.batch_size)
    logging.info(f"CLIP score: {score}")

    os.makedirs(args.result_dir, exist_ok=True)

    with open(f"{args.result_dir}/clip_score.txt", "a") as f:
        f.write(f"{args.gen_dir} {score}\n")


if __name__ == '__main__':
    main()