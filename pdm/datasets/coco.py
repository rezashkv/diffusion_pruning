import json
import os

from datasets import Dataset


def load_coco_dataset(images_dir, annotations_file):
    captions_file = json.load(open(annotations_file))
    images = []
    captions = []
    for capt in captions_file['annotations']:
        if '2014' in images_dir:
            split_name = os.path.basename(images_dir)
            image_path = os.path.join(images_dir, f"COCO_{split_name}_%012d.jpg" % capt['image_id'])
        else:
            image_path = os.path.join(images_dir, "%012d.jpg" % capt['image_id'])
        caption = capt['caption']
        images.append(image_path)
        captions.append(caption)
    dataset = Dataset.from_dict({"image": images, "caption": captions})
    return dataset