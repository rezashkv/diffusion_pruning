#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import logging
import os
import random
import pickle

from accelerate.utils import set_seed
from omegaconf import OmegaConf

import PIL
import numpy as np
import torch
import torch.utils.checkpoint
from accelerate.logging import get_logger
from datasets import load_dataset, Dataset, concatenate_datasets
from pdm.datasets.coco import load_coco_dataset
from transformers import AutoTokenizer, AutoModel
from diffusers.utils import check_min_version, deprecate
from pdm.models import HyperStructure
from pdm.models import StructureVectorQuantizer
from pdm.utils.arg_utils import parse_args
from pdm.datasets.cc3m import load_cc3m_dataset, load_cc3m_webdataset
import torch._dynamo

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
    "cc3m": ("image", "caption"),
    "coco": ("image", "caption"),
}

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

logger = get_logger(__name__)


def main():
    PIL.Image.MAX_IMAGE_PIXELS = 933120000
    torch.autograd.set_detect_anomaly(True)
    torch._dynamo.config.suppress_errors = True
    args = parse_args()
    config = OmegaConf.load(args.base_config_path)
    config.update(vars(args))

    if config.seed is not None:
        set_seed(config.seed)

    if config.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # ########################### Hypernet and Quantizer for Dataset Preprocessing #####################################

    hyper_net = HyperStructure.from_pretrained(config.pruning_ckpt_dir, subfolder="hypernet")
    quantizer = StructureVectorQuantizer.from_pretrained(config.pruning_ckpt_dir, subfolder="quantizer")

    mpnet_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    mpnet_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    # #################################################### Datasets ####################################################

    logging.info("Loading datasets...")
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    dataset_name = getattr(config.data, "dataset_name", None)
    dataset_config_name = getattr(config.data, "dataset_config_name", None)
    data_dir = getattr(config.data, "data_dir", None)

    if dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            dataset_name,
            dataset_config_name,
            cache_dir=config.cache_dir,
            ignore_verifications=True
        )

    else:
        if "conceptual_captions" in data_dir or "cc3m" in data_dir:
            dataset_name = "cc3m"
            dataset = {"validation":
                           load_cc3m_webdataset(data_dir, split="validation", return_image=False)}

        elif "coco" in data_dir:
            dataset_name = "coco"
            year = config.data.year
            dataset = {"validation": load_coco_dataset(os.path.join(data_dir, "images", f"val{year}"),
                                                       os.path.join(data_dir, "annotations",
                                                                    f"captions_val{year}.json"))}

        else:
            raise ValueError(f"Dataset {data_dir} not supported.")

    caption_column = DATASET_NAME_MAPPING[dataset_name][1]

    def get_mpnet_embeddings(capts, is_train=True):
        # Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                      min=1e-9)

        captions = []
        for caption in capts:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )

        encoded_input = mpnet_tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
        encoded_input = {k: v.to(mpnet_model.device) for k, v in encoded_input.items()}
        # Compute token embeddings
        with torch.no_grad():
            model_output = mpnet_model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings

    def filter_cc3m_webdataset(dataset, validation_indices=None):
        dataset = dataset["validation"]
        if validation_indices is None:
            validation_filtering_dataloader = torch.utils.data.DataLoader(dataset, batch_size=2048,
                                                                          shuffle=False)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            hyper_net.to(device)
            quantizer.to(device)
            mpnet_model.to(device)
            hyper_net.eval()
            quantizer.eval()
            validation_indices = {}
            with torch.no_grad():
                for batch in validation_filtering_dataloader:
                    batch[caption_column] = get_mpnet_embeddings(batch[caption_column], is_train=False)
                    arch_v = hyper_net(batch[caption_column])
                    indices = quantizer.get_cosine_sim_min_encoding_indices(arch_v)
                    for i, idx in enumerate(indices):
                        idx = idx.item()
                        validation_indices[batch["__key__"][i]] = idx

            # save validation indices to disk as a pk file
            pickle.dump(validation_indices, open(os.path.join(config.finetuning_ckpt_dir, "cc3m_validation_mapped_indices.pkl"), "wb"))
        return dataset

    def filter_dataset(dataset, validation_indices=None):
        if validation_indices is None:
            validation_captions = dataset["validation"][caption_column]
            validation_filtering_dataloader = torch.utils.data.DataLoader(validation_captions, batch_size=2048,
                                                                          shuffle=False)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            hyper_net.to(device)
            quantizer.to(device)
            mpnet_model.to(device)
            hyper_net.eval()
            quantizer.eval()
            validation_indices = []
            with torch.no_grad():
                for batch in validation_filtering_dataloader:
                    batch = get_mpnet_embeddings(batch, is_train=False)
                    arch_v = hyper_net(batch)
                    indices = quantizer.get_cosine_sim_min_encoding_indices(arch_v)
                    validation_indices.append(indices)
            validation_indices = torch.cat(validation_indices, dim=0)
            torch.save(validation_indices, os.path.join(config.finetuning_ckpt_dir, "fid_validation_mapped_indices.pt"))

        return dataset

    if dataset_name == "coco":
        val_indices = None
        if os.path.exists(os.path.join(config.finetuning_ckpt_dir, f"{dataset_name}_validation_mapped_indices.pt")):
            logging.info("Skipping filtering dataset. Loading indices from disk.")
            val_indices = torch.load(
                os.path.join(config.finetuning_ckpt_dir, f"{dataset_name}_validation_mapped_indices.pt"),
                map_location="cpu")

        filter_dataset(dataset, validation_indices=val_indices)
    elif dataset_name == "cc3m":
        val_indices = None
        if os.path.exists(os.path.join(config.finetuning_ckpt_dir, "cc3m_validation_mapped_indices.pkl")):
            logging.info("Skipping filtering dataset. Loading indices from disk.")
            val_indices = pickle.load(
                open(os.path.join(config.finetuning_ckpt_dir, "cc3m_validation_mapped_indices.pkl"), "rb"))
        filter_cc3m_webdataset(dataset, validation_indices=val_indices)


if __name__ == "__main__":
    main()
