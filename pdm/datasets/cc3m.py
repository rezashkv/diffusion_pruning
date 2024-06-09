import glob
import logging
import os
import pickle
import PIL
from PIL import Image
import pandas as pd
from PIL import ImageFile
from datasets import Dataset
from webdataset import WebDataset
import webdataset as wds
from pdm.utils.dist_utils import nodesplitter

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_cc3m_dataset(data_dir, split="train", split_file="Train_GCC-training.tsv",
                      split_dir="training"):
    captions = pd.read_csv(os.path.join(data_dir, split_file),
                           sep="\t", header=None, names=["caption", "link"],
                           dtype={"caption": str, "link": str})

    names_file = os.path.join(os.getcwd(), "../data", f"{split}_cc3m_names.pkl")
    if os.path.exists(names_file):
        with open(names_file, 'rb') as file:
            images = pickle.load(file)

    else:
        images = os.listdir(os.path.join(data_dir, split_dir))
        with open(names_file, 'wb') as file:
            pickle.dump(images, file)

    images = [os.path.join(data_dir, split_dir, image) for image in images]

    image_indices = [int(os.path.basename(image).split("_")[0]) for image in images]
    captions = captions.iloc[image_indices].caption.values.tolist()
    dataset = Dataset.from_dict({"image": images, "caption": captions})
    return dataset


def load_cc3m_webdataset(data_dir, split="training", resampled=True):
    training = split == "training"
    data_files = glob.glob(os.path.join(data_dir, split, "*.tar"))
    data_files = sorted(data_files)
    dataset = (
        WebDataset(
            data_files,
            repeat=training,
            shardshuffle=1000 if training else False,
            resampled=resampled if training else False,
            handler=wds.ignore_and_continue,
            nodesplitter=None if (training and resampled) else nodesplitter,
        )
        .shuffle(5000 if training else 0)
        .decode("pil")
    )
    dataset = dataset.rename(caption="txt", image="jpg")
    dataset = dataset.map(lambda x: {"caption": x["caption"], "image": x["image"]})
    return dataset
