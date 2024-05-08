import argparse
import os
from pdm.utils.clip_utils import clip_score
import logging

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_images_dir', type=str, required=True)
    parser.add_argument('--text_features_dir', type=str, required=True)
    parser.add_argument('--clip_model', type=str, default="ViT-B/32")
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--result_dir', type=str, required=True, help="Directory to save the results")
    parser.add_argument('--dataset_name', type=str, required=True, help="Dataset name")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.info(f"Calculating CLIP score for {args.gen_images_dir} using {args.text_features_dir} as text features.")
    score = clip_score(args.text_features_dir, args.gen_images_dir, clip_model=args.clip_model, num_workers=args.num_workers,
                       batch_size=args.batch_size)
    logging.info(f"CLIP score: {score}")

    os.makedirs(args.result_dir, exist_ok=True)

    with open(f"{args.result_dir}/clip_score_{args.dataset_name}.txt", "a") as f:
        f.write(f"{args.gen_images_dir} {score}\n")


if __name__ == '__main__':
    main()
