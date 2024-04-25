import os
import json
import numpy as np
import argparse

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the COCO dataset directory.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--num_samples", type=int, default=30000,
                        help="Number of samples to take from the COCO 2014 validation set.")
    return parser.parse_args()

def main(args):
    annotations_path = os.path.join(args.data_dir, "annotations", "captions_val2014.json")
    images_dir = os.path.join(args.data_dir, "images", "val2014")
    output_dir = os.path.join(args.data_dir, "images", "val2014_30k")
    output_annotations_path = os.path.join(args.data_dir, "annotations", "captions_val2014_30k.json")

    os.makedirs(output_dir, exist_ok=True)

    annotations = json.load(open(annotations_path))
    # deduplicate the annotations
    image_ids = set()
    deduplicated_annotations = []
    for ann in annotations['annotations']:
        if ann['image_id'] not in image_ids:
            deduplicated_annotations.append(ann)
            image_ids.add(ann['image_id'])

    annotations['annotations'] = deduplicated_annotations
    np.random.seed(args.seed)
    indices = np.random.choice(len(annotations['annotations']), args.num_samples, replace=False)
    selected_annotations = [annotations['annotations'][i] for i in indices]

    with open(output_annotations_path, "w") as f:
        json.dump({"annotations": selected_annotations}, f)

    for i in indices:
        image_id = annotations['annotations'][i]['image_id']
        # read the image and save it as a npy file
        image_path = os.path.join(images_dir, f"COCO_val2014_{image_id:012d}.jpg")
        img = Image.open(image_path)
        data = np.asarray(img)
        output_path = os.path.join(output_dir, f"COCO_val2014_{image_id:012d}.npy")
        np.save(output_path, data)

    print(f"Saved {args.num_samples} samples to {output_dir}.")
    print(f"Saved annotations to {output_annotations_path}.")
    print("Done.")

if __name__ == "__main__":
    args = parse_args()
    main(args)