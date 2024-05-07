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
import pickle

import safetensors
from accelerate.utils import set_seed
from omegaconf import OmegaConf

import cv2
import accelerate
import numpy as np
import torch
import torch.utils.checkpoint
from accelerate.logging import get_logger
from datasets import load_dataset
from pdm.datasets.coco import load_coco_dataset

from diffusers import PNDMScheduler
from diffusers.utils import check_min_version
from diffusers import StableDiffusionPipeline

from pdm.models.diffusion import UNet2DConditionModelPruned
from pdm.utils.arg_utils import parse_args
from pdm.datasets.cc3m import load_cc3m_dataset, load_cc3m_webdataset
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

    assert config.embedding_ind is not None, "embedding_ind must be provided"

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
            dataset = load_cc3m_webdataset(data_dir, split="validation", return_image=False)

            fid_val_indices_path = os.path.abspath(
                os.path.join(config.finetuning_ckpt_dir, "..", "..", f"{dataset_name}_validation_mapped_indices.pkl"))
            assert os.path.exists(fid_val_indices_path), \
                f"{dataset_name}_validation_mapped_indices.pkl must be present in two upper directory of the checkpoint directory {config.finetuning_ckpt_dir}"
            val_indices = pickle.load(open(fid_val_indices_path, "rb"))
            dataset = dataset.select(lambda x: val_indices[x["__key__"]] == config.embedding_ind)

        elif "coco" in data_dir:
            dataset_name = "coco"
            year = config.data.year
            dataset = {
                "validation": load_coco_dataset(os.path.join(data_dir, "images", f"val{year}"),
                                                os.path.join(data_dir, "annotations", f"captions_val{year}.json"))}

            def filter_dataset(dataset, validation_indices):
                dataset["validation"] = dataset["validation"].select(
                    torch.where(validation_indices == config.embedding_ind)[0])
                return dataset

            fid_val_indices_path = os.path.abspath(
                os.path.join(config.finetuning_ckpt_dir, "..", "..", f"{dataset_name}_validation_mapped_indices.pt"))
            assert os.path.exists(fid_val_indices_path), \
                f"{dataset_name}_validation_mapped_indices.pt must be present in two upper directory of the checkpoint directory {config.finetuning_ckpt_dir}"
            val_indices = torch.load(fid_val_indices_path, map_location="cpu")
            dataset = filter_dataset(dataset, validation_indices=val_indices)
            dataset = dataset["validation"]
            logger.info("Dataset of size %d loaded." % len(dataset))

        else:
            raise ValueError(f"Dataset {data_dir} not supported.")

    def collate_fn(examples):
        # get a list of images and captions from examples which is a list of dictionaries
        images = [example["image"] for example in examples]
        captions = [example["caption"] for example in examples]
        return {"image": images, "caption": captions}

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=config.data.dataloader.image_generation_batch_size * accelerator.num_processes,
        num_workers=config.data.dataloader.dataloader_num_workers,
        collate_fn=collate_fn
    )

    dataloader = accelerator.prepare(dataloader)

    # #################################################### Models ####################################################
    arch_v = torch.load(os.path.join(config.finetuning_ckpt_dir, "arch_vector.pt"), map_location="cpu")

    unet = UNet2DConditionModelPruned.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="unet",
        revision=config.revision,
        down_block_types=config.model.unet.unet_down_blocks,
        mid_block_type=config.model.unet.unet_mid_block,
        up_block_types=config.model.unet.unet_up_blocks,
        arch_vector=arch_v
    )

    state_dict = safetensors.torch.load_file(os.path.join(config.finetuning_ckpt_dir, "unet",
                                                          "diffusion_pytorch_model.safetensors"))
    unet.load_state_dict(state_dict)

    noise_scheduler = PNDMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        unet=unet,
        scheduler=noise_scheduler,
    )

    if config.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    pipeline.set_progress_bar_config(disable=not accelerator.is_main_process)

    pipeline.to(accelerator.device)

    image_output_dir = os.path.join(config.finetuning_ckpt_dir, "..", "..", f"fid_images_{dataset_name}")
    os.makedirs(image_output_dir, exist_ok=True)

    for batch in dataloader:
        if config.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=accelerator.device).manual_seed(config.seed)
        gen_images = pipeline(batch["caption"], num_inference_steps=config.training.num_inference_steps,
                              generator=generator, output_type="np"
                              ).images

        for idx, caption in enumerate(batch["caption"]):
            image_name = batch["image"][idx].split("/")[-1]
            image_path = os.path.join(image_output_dir, f"{image_name[:-4]}.npy")
            img = gen_images[idx]
            img = img * 255
            img = img.astype(np.uint8)
            img = cv2.resize(img, (256, 256))
            np.save(image_path, img)


if __name__ == "__main__":
    main()
