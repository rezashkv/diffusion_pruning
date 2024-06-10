import argparse

import numpy as np
from PIL import Image
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Resize images in a directory")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save resized images")
    parser.add_argument("--size", type=int, nargs=2, default=[256, 256], help="Size of the resized images")
    return parser.parse_args()


def resize_images_in_dir(data_dir, output_dir, size):
    for img_name in os.listdir(data_dir):
        img = Image.open(os.path.join(data_dir, img_name))
        img = img.resize(size)
        img = np.array(img)
        np.save(os.path.join(output_dir, img_name + ".npy"), img)


def main():
    args = parse_args()
    resize_images_in_dir(args.data_dir, args.output_dir, size=tuple(args.size))


if __name__ == "__main__":
    main()
