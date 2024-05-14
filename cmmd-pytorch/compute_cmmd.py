import argparse
from cmmd_utils import compute_cmmd


def parse_args():
    parser = argparse.ArgumentParser(description="Compute and save embeddings for reference images.")
    parser.add_argument("--ref_file", type=str, help="Path to the file containing reference images embeddings.")
    parser.add_argument("--eval_dir", type=str, help="Path to the directory containing images to be evaluated.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size used in the CLIP embedding calculation.")
    parser.add_argument("--max_count", type=int, default=-1,
                        help="Maximum number of images to use from each directory.")
    parser.add_argument("--result_file", type=str, help="Path to the file to save the computed CMMD value.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cmmd = compute_cmmd(ref_dir=None, eval_dir=args.eval_dir, ref_embed_file=args.ref_file, batch_size=args.batch_size,
                        max_count=args.max_count)
    with open(args.result_file, "a") as f:
        f.write(f"{args.eval_dir}: {cmmd}\n")
