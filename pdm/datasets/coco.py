import json
import os

from datasets import Dataset


def load_coco_dataset(images_dir, annotations_file):
    captions_file = json.load(open(annotations_file))
    images = []
    captions = []
    for capt in captions_file['annotations']:
        image_path = os.path.join(images_dir, "%012d.jpg" % capt['image_id'])
        caption = capt['caption']
        images.append(image_path)
        captions.append(caption)
    dataset = Dataset.from_dict({"image": images, "caption": captions})
    return dataset