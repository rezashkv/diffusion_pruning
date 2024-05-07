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
from webdataset.filters import map, select

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_cc3m_dataset(data_dir,  split="train", split_file="Train_GCC-training.tsv",
                      split_dir="training", max_samples=1000, bad_images_path=None):

    captions = pd.read_csv(os.path.join(data_dir, split_file),
                           sep="\t", header=None, names=["caption", "link"],
                           dtype={"caption": str, "link": str})
    # get parent of getcwd() to get to projects/diffusion_pruning/pdm
    names_file = os.path.join(os.getcwd(), "../data", f"{split}_cc3m_names.pkl")
    if os.path.exists(names_file):
        with open(names_file, 'rb') as file:
            images = pickle.load(file)

    else:
        images = os.listdir(os.path.join(data_dir, split_dir))
        with open(names_file, 'wb') as file:
            pickle.dump(images, file)

    if max_samples is not None and max_samples < 1000:
        images = images[:max_samples * 5]

    images = [os.path.join(data_dir, split_dir, image) for image in images]
    if bad_images_path is None:
        bad_images_path = os.path.join(os.getcwd(), "../data", f"{split}_cc3m_bad_images.txt")
    if os.path.exists(bad_images_path):
        with open(os.path.join(bad_images_path), "r") as f:
            bad_images = f.readlines()
        bad_images = [image.strip() for image in bad_images]
        images = set(images) - set(bad_images)
        images = sorted(list(images))

    else:
        # remove images that cant be opened by PIL
        imgs = []
        bad_images = []
        PIL.Image.MAX_IMAGE_PIXELS = 933120000
        for image in images:
            try:
                with Image.open(image) as img:
                    imgs.append(image)
            except PIL.UnidentifiedImageError as e:
                bad_images.append(image)
                logging.info(
                    f"Image file `{image}` is corrupt and can't be opened. {e}"
                )
            except FileNotFoundError as e:
                bad_images.append(image)
                logging.info(
                    f"Image file `{image}` is not found. {e}"
                )
            except Exception as e:
                bad_images.append(image)
                logging.info(f"Unidentified exception {e}")
        images = imgs
        with open(bad_images_path, "w") as f:
            f.write("\n".join(bad_images))

    image_indices = [int(os.path.basename(image).split("_")[0]) for image in images]
    captions = captions.iloc[image_indices].caption.values.tolist()
    dataset = Dataset.from_dict({"image": images, "caption": captions})
    del images, captions, image_indices, bad_images
    return dataset


def load_cc3m_webdataset(data_dir, split="training", return_image=False):
    data_files = glob.glob(os.path.join(data_dir, split, "*.tar"))
    data_files = sorted(data_files)
    dataset = WebDataset(data_files).decode("pil")
    dataset = dataset.rename(caption="txt", image="jpg")
    if not return_image:
        dataset = dataset.map(lambda x: {"caption": x["caption"]})
    else:
        dataset = dataset.map(lambda x: {"caption": x["caption"], "image": x["image"]})

    return dataset
