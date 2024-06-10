import os
import pickle
import pandas as pd
from PIL import ImageFile
from datasets import Dataset

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
