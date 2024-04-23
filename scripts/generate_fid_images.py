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


import os

from accelerate.utils import set_seed
from omegaconf import OmegaConf

import accelerate
import numpy as np
import torch
import torch.utils.checkpoint
from accelerate.logging import get_logger
from datasets import load_dataset
from pdm.datasets.coco import load_coco_dataset

from diffusers import DDIMScheduler
from diffusers.utils import check_min_version
from diffusers import StableDiffusionPipeline

from pdm.models.diffusion import UNet2DConditionModelPruned
from pdm.utils.arg_utils import parse_args
from pdm.datasets.cc3m import load_cc3m_dataset
import torch._dynamo

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

logger = get_logger(__name__)


def main():
    args = parse_args()
    config = OmegaConf.load(args.base_config_path)
    # add args to config
    config.update(vars(args))

    if config.seed is not None:
        set_seed(config.seed)

    # #################################################### Accelerator ####################################################
    accelerator = accelerate.Accelerator()

    # #################################################### Datasets ####################################################

    logger.info("Loading datasets...")
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    dataset_name = getattr(config.data, "dataset_name", None)
    dataset_config_name = getattr(config.data, "dataset_config_name", None)
    data_dir = getattr(config.data, "data_dir", None)

    validation_data_dir = getattr(config.data, "validation_data_dir", None)
    validation_data_file = getattr(config.data, "validation_data_file", None)
    validation_bad_images_path = getattr(config.data, "validation_bad_images_path", None)
    max_validation_samples = getattr(config.data, "max_validation_samples", None)

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
            dataset = {"validation": load_cc3m_dataset(data_dir,
                                                       split="validation",
                                                       split_file=validation_data_file,
                                                       split_dir=validation_data_dir,
                                                       max_samples=max_validation_samples,
                                                       bad_images_path=validation_bad_images_path)}


        elif "coco" in data_dir:
            year = config.data.year
            dataset = {
                "validation": load_coco_dataset(os.path.join(data_dir, "images", f"val{year}"),
                                                os.path.join(data_dir, "annotations", f"captions_val{year}.json"))}

        else:
            raise ValueError(f"Dataset {data_dir} not supported.")

    def filter_dataset(dataset, validation_indices):
        dataset["validation"] = dataset["validation"].select(validation_indices)
        return dataset


    assert os.path.exists(os.path.join(config.finetuning_ckpt_dir, "..", "filtered_validation_indices.pt")), \
        "filtered_validation_indices.pt must be present in the checkpoint parent directory"
    val_indices = torch.load(os.path.join(config.finetuning_ckpt_dir, "..", "filtered_validation_indices.pt"),
                             map_location="cpu")

    dataset = filter_dataset(dataset, validation_indices=val_indices)
    dataset = dataset["validation"]

    #deduplicate dataset based on image path
    dataset = dataset.unique("image")

    logger.info("Dataset of size %d loaded." % len(dataset))

    dataloader = torch.utils.data.DataLoader(
       dataset,
        shuffle=False,
        batch_size=config.data.dataloader.image_generation_batch_size * accelerator.num_processes,
        num_workers=config.data.dataloader.dataloader_num_workers,
    )

    dataloader = accelerator.prepare(dataloader)

    # #################################################### Models ####################################################

    unet = UNet2DConditionModelPruned.from_pretrained(
        config.finetuning_ckpt_dir,
        subfolder="unet",
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True

    )

    noise_scheduler = DDIMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        unet=unet,
        scheduler=noise_scheduler,
    )

    if config.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    pipeline.set_progress_bar_config(disable=not accelerator.is_main_process)

    pipeline.to(accelerator.device)

    image_output_dir = os.path.join(config.finetuning_ckpt_dir, "..", "..", "fid_images")
    os.makedirs(image_output_dir, exist_ok=True)

    for step, batch in enumerate(dataloader):
        if config.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=accelerator.device).manual_seed(config.seed)
        gen_images = pipeline(batch["caption"], num_inference_steps=config.training.num_inference_steps,
                                               generator=generator, output_type="np"
                                               ).images
        # save the images. save with caption as name
        for idx, caption in enumerate(batch["caption"]):
            image_path = os.path.join(image_output_dir, f"{caption}.png")
            np.save(image_path, gen_images[idx])


if __name__ == "__main__":
    main()
