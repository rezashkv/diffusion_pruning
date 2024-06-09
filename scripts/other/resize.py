import argparse
import logging
import os
import cv2
import numpy as np

def resize_images_in_dir(directory, size=(256, 256)):
    logging.info(f"Resizing images in {directory} to {size}")
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            img = np.load(os.path.join(directory, filename))
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            np.save(os.path.join(directory, filename.replace(".jpg", "")), img)
            logging.info(f"Resized {filename}")


def parse_args():
    parser = argparse.ArgumentParser(description="Resize images in a directory")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing images")
    parser.add_argument("--size", type=int, nargs=2, default=[256, 256], help="Size of the resized images")
    return parser.parse_args()

def main():
    args = parse_args()
    resize_images_in_dir(args.directory, size=tuple(args.size))

if __name__ == "__main__":
    main()
