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

import argparse
import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path

import PIL
import accelerate
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, Dataset, concatenate_datasets
from datasets.utils.logging import set_verbosity_error, set_verbosity_warning
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from pdm.utils.op_counter import (add_flops_counting_methods)

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from pdm.pipelines import StableDiffusionPruningPipeline
from pdm.models.diffusion import UNet2DConditionModelGated
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from pdm.models import HyperStructure
from pdm.models import StructureVectorQuantizer
from pdm.losses import ClipLoss, ResourceLoss
from PIL import ImageFile

if is_wandb_available():
    import wandb

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

logger = get_logger(__name__)

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def save_model_card(
        args,
        repo_id: str,
        images=None,
        repo_folder=None,
):
    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(args.validation_prompts))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {args.pretrained_model_name_or_path}
datasets:
- {args.dataset_name}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
inference: true
---
    """
    model_card = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was pruned from **{args.pretrained_model_name_or_path}** on the **{args.dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: {args.validation_prompts}: \n
{img_str}

## Pipeline usage

You can use the pipeline like so:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
prompt = "{args.validation_prompts[0]}"
image = pipeline(prompt).images[0]
image.save("my_image.png")
```

## Training info

These are the key hyperparameters used during training:

* Epochs: {args.num_train_epochs}
* Hypernet Learning rate: {args.hypernet_learning_rate}
* Quantizer Learning rate: {args.quantizer_learning_rate}
* Batch size: {args.train_batch_size}
* Gradient accumulation steps: {args.gradient_accumulation_steps}
* Image resolution: {args.resolution}
* Mixed-precision: {args.mixed_precision}

"""
    wandb_info = ""
    if is_wandb_available():
        wandb_run_url = None
        if wandb.run is not None:
            wandb_run_url = wandb.run.url

    if wandb_run_url is not None:
        wandb_info = f"""
More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
"""

    model_card += wandb_info

    with open(os.path.join(repo_folder, "../README.md"), "w") as f:
        f.write(yaml + model_card)


def log_validation(hyper_net, quantizer, vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch):
    logger.info("Running validation... ")

    hyper_net.eval()
    quantizer.eval()
    pipeline = StableDiffusionPruningPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
        hyper_net=hyper_net,
        quantizer=quantizer,
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_output_dir = os.path.join(args.output_dir, "validation_images", f"epoch_{epoch}")
    os.makedirs(image_output_dir, exist_ok=True)
    images = []
    for step in range(0, len(args.validation_prompts), args.train_batch_size * accelerator.num_processes):
        batch = args.validation_prompts[step:step + args.train_batch_size * accelerator.num_processes]
        with torch.autocast("cuda"):
            with accelerator.split_between_processes(batch) as batch:
                gen_images = pipeline(batch, num_inference_steps=args.num_inference_steps, generator=generator).images
                for i, image in enumerate(gen_images):
                    try:
                        image.save(os.path.join(image_output_dir, f"{step + i}.png"))
                    except Exception as e:
                        logger.error(f"Error saving image {batch[i]}: {e}")

                if step == 0:
                    images = gen_images

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images


def interpret_quantizer_codebook(quantizer, args):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Dynamic Pruning of StableDiffusion-2.1")
    parser.add_argument(
        "--input_perturbation", type=float, default=0,
        help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--data_files",
        type=str,
        default=None,
        nargs="*",
        help=("The data_files to use in dataset loading. It can be a string pointing to a csv or a json file, or a"
              " list of strings pointing to files in JSON or CSV format.")
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument("--num_validation_samples", type=int, default=16,
                        help="The number of validation samples to generate.")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-pruned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=43, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16,
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--hypernet_learning_rate",
        type=float,
        default=1e-1,
        help="Initial hypernet learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--quantizer_learning_rate",
                        help="Initial quantizer learning rate (after the potential warmup period) to use.",
                        type=float, default=1e-1)
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
             "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2,
                        help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08,
                        help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None,
                        help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default='epsilon',
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or"
             " leave `None`. If left to `None` the default prediction type of the "
             "scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-dynamic-pruning",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="The `run_name` argument passed to Accelerator.init_trackers")
    parser.add_argument("--num_inference_steps", type=int, default=100,
                        help="The number of inference steps to run.")
    parser.add_argument("--contrastive_loss_temperature", type=float, default=1.0,
                        help="The temperature for the clip loss.")
    parser.add_argument(
        "--hypernet_T",
        type=float,
        default=0.4,
        required=False,
        help="The temperature for the hypernet.",
    )
    parser.add_argument(
        "--hypernet_base",
        type=float,
        default=3,
        required=False,
        help="The base for the hypernet.",
    )
    parser.add_argument(
        "--hypernet_sparsity",
        type=int,
        default=0,
        required=False,
        help="The sparsity for the hypernet.",
    )

    parser.add_argument(
        "--num_arch_vq_codebook_embeddings",
        type=int,
        default=256,
        required=False,
        help="The number of codebook embeddings for the quantizer.",
    )
    parser.add_argument(
        "--arch_vq_beta",
        type=float,
        default=0.25,
        required=False,

    )
    parser.add_argument(
        "--unet_down_blocks",
        type=str,
        nargs="*",
        default=("CrossAttnDownBlock2DHalfGated", "CrossAttnDownBlock2DHalfGated",
                 "CrossAttnDownBlock2DHalfGated", "DownBlock2DHalfGated"),
    )
    parser.add_argument(
        "--unet_up_blocks",
        type=str,
        nargs="*",
        default=("UpBlock2DHalfGated", "CrossAttnUpBlock2DHalfGated", "CrossAttnUpBlock2DHalfGated",
                 "CrossAttnUpBlock2DHalfGated"),
    )
    parser.add_argument("--pruning_target",
                        type=float,
                        required=False,
                        default=0.7,
                        help="The pruning probability for each layer.")
    parser.add_argument("--resource_loss_type",
                        type=str,
                        default="log",
                        required=False,
                        choices=["log", "mae", "mse"],
                        help="The resource loss type. Choose between `log`, `mae` and `mse`.")

    parser.add_argument("--resource_loss_weight",
                        type=float,
                        default=1.0,
                        required=False,
                        help="The resource loss weight.")
    parser.add_argument("--q_loss_weight",
                        type=float,
                        default=1.0,
                        required=False,
                        help="The quantizer loss weight.")
    parser.add_argument("--contrastive_loss_weight",
                        type=float,
                        default=1.0,
                        required=False,
                        help="The contrastive loss weight.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir,
                                                      total_limit=args.checkpoints_total_limit)

    # if validation_prompts is a file or a dir, load the prompts from there
    if args.validation_prompts is not None:
        if args.validation_prompts[0].endswith(".csv") or args.validation_prompts[0].endswith(".tsv"):
            validation_data = pd.read_csv(args.validation_prompts[0], sep="\t", header=None,
                                          names=[args.image_column, args.caption_column])[
                args.caption_column].values.astype(str).tolist()
            args.validation_prompts = validation_data

        elif os.path.isfile(args.validation_prompts[0]):
            with open(args.validation_prompts[0], "r") as f:
                args.validation_prompts = [line.strip() for line in f.readlines()]
        elif os.path.isdir(args.validation_prompts[0]):
            prompts = []
            for d in args.validation_prompts:
                files = [os.path.join(d, caption_file) for caption_file in os.listdir(d) if f.endswith(".txt")]
                for f in files:
                    with open(f, "r") as f:
                        prompts.extend([line.strip() for line in f.readlines()])
            args.validation_prompts = prompts

    if args.num_validation_samples is not None:
        args.validation_prompts = args.validation_prompts[:args.num_validation_samples]

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )
    unet = UNet2DConditionModelGated.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision,
        down_block_types=args.unet_down_blocks, up_block_types=args.unet_up_blocks
    )
    hyper_net = HyperStructure(input_dim=text_encoder.config.hidden_size,
                               seq_len=text_encoder.config.max_position_embeddings,
                               structure=unet.get_structure(),
                               T=args.hypernet_T, base=args.hypernet_base)

    quantizer = StructureVectorQuantizer(n_e=args.num_arch_vq_codebook_embeddings,
                                         vq_embed_dim=sum(unet.get_structure()), beta=args.arch_vq_beta,
                                         temperature=args.hypernet_T, base=args.hypernet_base)

    r_loss = ResourceLoss(p=args.pruning_target, loss_type=args.resource_loss_type)
    clip_loss = ClipLoss(temperature=args.contrastive_loss_temperature)

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModelGated.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision,
            down_block_types=args.unet_down_blocks, up_block_types=args.unet_up_blocks
        )
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModelGated,
                            model_config=ema_unet.config)

    unet.eval()
    unet.freeze()
    hyper_net.train()
    quantizer.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    if isinstance(model, (UNet2DConditionModel, UNet2DConditionModelGated)):
                        logger.info("Save UNet")
                        model.save_pretrained(os.path.join(output_dir, "unet"))
                    elif isinstance(model, HyperStructure):
                        logger.info(f"Saving HyperStructure")
                        model.save_pretrained(os.path.join(output_dir, "hypernet"))
                    elif isinstance(model, StructureVectorQuantizer):
                        logger.info(f"Saving Quantizer")
                        model.save_pretrained(os.path.join(output_dir, "quantizer"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                if isinstance(model, (UNet2DConditionModel, UNet2DConditionModelGated)):

                    load_model = UNet2DConditionModelGated.from_pretrained(input_dir, subfolder="unet")

                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model
                elif isinstance(model, HyperStructure):
                    model.from_pretrained(input_dir, subfolder="hypernet")
                elif isinstance(model, StructureVectorQuantizer):
                    model.from_pretrained(input_dir, subfolder="quantizer")
                else:
                    models.append(model)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.hypernet_learning_rate = (
                args.hypernet_learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        args.quantizer_learning_rate = (
                args.quantizer_learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    unet = add_flops_counting_methods(unet)
    unet.start_flops_count(ost=sys.stdout, verbose=False, ignore_list=[])

    optimizer = optimizer_cls(
        [
            {"params": hyper_net.parameters(), "lr": args.hypernet_learning_rate},
            {"params": quantizer.parameters(), "lr": args.quantizer_learning_rate},
        ],
        lr=args.hypernet_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    logger.info("Loading dataset...")
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            data_files=args.data_files,
            cache_dir=args.cache_dir,
            data_dir=args.train_data_dir,
            ignore_verifications=True
        )
    else:
        if "aesthetics" in args.train_data_dir:
            if args.data_files is None:
                # datafiles a list of 5-char strs from 00000 to 00200
                args.data_files = [f"{i:05d}" for i in range(250)]

            def load_dataset_dir(dataset_dir):
                dataset = []
                image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]
                for image_file in image_files:
                    image_path = os.path.join(dataset_dir, image_file)
                    caption_file = image_file.replace('.jpg', '.txt')
                    caption_path = os.path.join(dataset_dir, caption_file)
                    with open(caption_path, 'r') as caption_file:
                        caption = caption_file.read()
                    example = {
                        'image': image_path,
                        'caption': str(caption),
                    }
                    dataset.append(example)
                return dataset

            def load_main_dataset(main_dataset_dir, train_dirs):
                train_datasets = {}
                for subdir in train_dirs:
                    dataset_name = subdir
                    dataset_dir = os.path.join(main_dataset_dir, subdir)
                    dataset = load_dataset_dir(dataset_dir)
                    train_datasets[dataset_name] = dataset

                return train_datasets

            train_data = load_main_dataset(args.train_data_dir, list(args.data_files))

            # Convert the loaded data into a Hugging Face Dataset
            tr_datasets = []
            for dataset_name, dataset in train_data.items():
                tr_datasets.append(Dataset.from_list(dataset))

            dataset = {'train': concatenate_datasets(tr_datasets)}

        elif "captions" in args.train_data_dir:
            captions = pd.read_csv(os.path.join(args.train_data_dir, "Train_GCC-training.tsv"),
                                   sep="\t", header=None, names=["caption", "link"],
                                   dtype={"caption": str, "link": str})
            images = os.listdir(os.path.join(args.train_data_dir, "training"))
            images = [os.path.join(args.train_data_dir, "training", image) for image in images]
            if os.path.exists(os.path.join(args.output_dir, "cc3m_bad_images.txt")):
                with open(os.path.join(args.output_dir, "cc3m_bad_images.txt"), "r") as f:
                    bad_images = f.readlines()
                bad_images = [image.strip() for image in bad_images]
                images = set(images) - set(bad_images)
                images = list(images)

            else:
                # remove images that cant be opened by PIL
                imgs = []
                bad_images = []
                PIL.Image.MAX_IMAGE_PIXELS = 933120000
                for image in images:
                    try:
                        with PIL.Image.open(image) as img:
                            imgs.append(img)
                    except PIL.UnidentifiedImageError:
                        bad_images.append(image)
                        logger.info(
                            f"Image file `{image}` is corrupt and can't be opened."
                        )
                images = imgs

                # save the bad images to a file
                with open(os.path.join(args.output_dir, "cc3m_bad_images.txt"), "w") as f:
                    f.write("\n".join(bad_images))

            image_indices = [int(os.path.basename(image).split("_")[0]) for image in images]
            captions = captions.iloc[image_indices].caption.values.tolist()
            train_dataset = Dataset.from_dict({"image": images, "caption": captions})

            dataset = {"train": train_dataset}

        else:
            data_files = {}
            if args.train_data_dir is not None:
                data_files["train"] = os.path.join(args.train_data_dir, "**")
            dataset = load_dataset(
                "imagefolder",
                data_files=data_files,
                cache_dir=args.cache_dir,
            )
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # 6. Get the column names for input/target.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        # if image_column contains urls or paths to files, convert to bytes
        # check if image_column path exists:
        if isinstance(examples[image_column][0], str):
            if not os.path.exists(examples[image_column][0]):
                examples[image_column] = [requests.get(image).content for image in examples[image_column]]
            else:
                examples[image_column] = [PIL.Image.open(image) for image in examples[image_column]]

        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler, hyper_net, quantizer = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler, hyper_net, quantizer
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        if args.wandb_run_name is None:
            args.wandb_run_name = f"{args.dataset_name if args.dataset_name else args.train_data_dir.split('/')[-1]}-{args.max_train_samples}"
        accelerator.init_trackers(args.tracker_project_name, tracker_config,
                                  init_kwargs={"wandb": {"name": args.wandb_run_name}}
                                  )
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            hyper_net.train()
            quantizer.train()
            unet.reset_flops_count()
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                arch_vector = hyper_net(encoder_hidden_states)
                arch_vector_quantized, q_loss, _ = quantizer(arch_vector)

                # gather the arch_vector_quantized across all processes to get large batch for contrastive loss
                encoder_hidden_states_list = accelerator.gather(encoder_hidden_states)
                arch_vector_quantized_list = accelerator.gather(arch_vector_quantized)
                contrastive_loss = clip_loss(torch.sum(encoder_hidden_states_list, dim=1).squeeze(1), arch_vector_quantized_list)

                if not hasattr(unet, "total_flops"):
                    with torch.no_grad():
                        unet.set_structure(arch_vector_quantized)
                        unet(noisy_latents, timesteps, encoder_hidden_states)
                        unet.total_flops = unet.compute_average_flops_cost()[0]
                        unet.reset_flops_count()

                unet.set_structure(arch_vector_quantized)

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                            torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                curr_flops, _ = unet.compute_average_flops_cost()

                resource_ratio = (curr_flops / unet.total_flops)
                resource_loss = r_loss(resource_ratio)

                loss += args.resource_loss_weight * resource_loss
                loss += args.q_loss_weight * q_loss
                loss += args.contrastive_loss_weight * contrastive_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Back-propagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                accelerator.log({"resource_ratio": resource_ratio}, step=global_step)
                accelerator.log({"resource_loss": resource_loss}, step=global_step)
                accelerator.log({"commitment_loss": q_loss}, step=global_step)
                accelerator.log({"contrastive_loss": contrastive_loss}, step=global_step)
                accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=global_step)

                # log the pairwise cosine similarity of the embeddings of the quantizer:
                if accelerator.num_processes == 1:
                    # TODO apply gumble softmax to the embeddings before computing the similarity
                    quantizer_embeddings = quantizer.embedding.weight.data.cpu().numpy()
                else:
                    quantizer_embeddings = quantizer.module.embedding.weight.data.cpu().numpy()
                quantizer_embeddings = quantizer_embeddings / np.linalg.norm(quantizer_embeddings, axis=1, keepdims=True)
                quantizer_embeddings = quantizer_embeddings @ quantizer_embeddings.T
                accelerator.log({"quantizer embeddings pairwise similarity": wandb.Image(quantizer_embeddings)},
                                step=global_step)

                # log the pairwise cosine similarity of the embeddings of the architecture vectors quantized:
                arch_vector_ = arch_vector_quantized.data.cpu().numpy()
                arch_vector_ = arch_vector_ / np.linalg.norm(arch_vector_, axis=1, keepdims=True)
                arch_vector_ = arch_vector_ @ arch_vector_.T
                accelerator.log({"arch vector pairwise similarity": wandb.Image(arch_vector_)},
                                step=global_step)

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        # save architecture vector quantized
                        torch.save(arch_vector_quantized, os.path.join(args.output_dir,
                                                                       f"arch_vector_quantized-{global_step}.pt"))

                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0],
                    "q_loss": q_loss.detach().item(), "contrastive_loss": contrastive_loss.detach().item(),
                    "resource_loss": resource_loss.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompts is not None and (epoch % args.validation_epochs == 0 or
                                                        epoch == args.num_train_epochs - 1):
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                val_images = log_validation(
                    hyper_net,
                    quantizer,
                    vae,
                    text_encoder,
                    tokenizer,
                    unet,
                    args,
                    accelerator,
                    weight_dtype,
                    global_step,
                )
                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        if args.push_to_hub:
            save_model_card(args, repo_id, val_images, repo_folder=args.output_dir)
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()

#1520451_2413603269