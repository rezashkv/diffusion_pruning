import argparse
from cmmd_utils import save_ref_embeds


def parse_args():
    parser = argparse.ArgumentParser(description="Compute and save embeddings for reference images.")
    parser.add_argument("--ref_dir", type=str, help="Path to the directory containing reference images.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size used in the CLIP embedding calculation.")
    parser.add_argument("--max_count", type=int, default=-1, help="Maximum number of images to use from each directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_ref_embeds(args.ref_dir, args.batch_size, args.max_count)
