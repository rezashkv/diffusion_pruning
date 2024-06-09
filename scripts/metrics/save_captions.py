import json
import os
from pdm.datasets.cc3m import load_cc3m_webdataset


def save_coco_captions(annotations_file):
    # annotations file's name is something like 'annotations/captions_val2014_30k.json'
    split_name = os.path.basename(annotations_file)[len('captions_'):-len('.json')]
    captions_file = json.load(open(annotations_file))
    captions_dir = os.path.dirname(annotations_file)
    save_dir = os.path.join(captions_dir, 'clip-captions')
    os.makedirs(save_dir, exist_ok=True)
    for capt in captions_file['annotations']:
        if '2014' in annotations_file:
            image_id = f"COCO_{split_name}_%012d" % capt['image_id']
        else:
            image_id = "%012d" % capt['image_id']

        caption = capt['caption']
        with open(os.path.join(save_dir, image_id + '.txt'), 'w') as f:
            f.write(caption)


def save_cc3m_captions(data_dir):
    split = "validation"
    dataset = load_cc3m_webdataset(data_dir, split=split)
    save_dir = os.path.join(data_dir, 'clip-captions')
    os.makedirs(save_dir, exist_ok=True)
    for sample in dataset:
        image_id = sample['__key__']
        caption = sample['caption']
        with open(os.path.join(save_dir, image_id + '.txt'), 'w') as f:
            f.write(caption)


if __name__ == '__main__':
    save_coco_captions('/home/rezashkv/scratch/research/data/coco/annotations/captions_val2014_30k.json')
    save_cc3m_captions('/home/rezashkv/scratch/research/data/cc3m')
