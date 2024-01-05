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
from typing import Dict

import math
import os
import random
import shutil
import sys
import datetime

from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import PIL
import accelerate
import numpy as np
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
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel
from transformers.utils import ContextManagers
from pdm.utils.op_counter import (add_flops_counting_methods)

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from pdm.models.diffusion import UNet2DConditionModelGated
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from pdm.models import HyperStructure
from pdm.models import StructureVectorQuantizer
from pdm.losses import ClipLoss, ResourceLoss
from PIL import ImageFile
from pdm.utils.logging_utils import save_model_card, generate_samples_from_prompts, log_quantizer_embedding_samples
from pdm.utils.arg_utils import parse_args
from pdm.utils.metric_utils import compute_snr
from pdm.datasets.cc3m import load_cc3m_dataset
from pdm.datasets.laion_aes import load_main_laion_dataset
from pdm.training.validation import validation

from sentence_transformers import SentenceTransformer


if is_wandb_available():
    import wandb

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

logger = get_logger(__name__)

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def main():
    torch.autograd.set_detect_anomaly(True)
    args = parse_args()
    config = OmegaConf.load(args.base_config_path)
    # add args to config
    config.update(vars(args))

    if config.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    if config.name != "":
        nowname = now + f"_{config.name}"
    else:
        nowname = now

    config["training"]["logging"]["logging_dir"] = os.path.join(config["training"]["logging"]["logging_dir"],
                                                                os.getcwd().split('/')[-2],
                                                                config.base_config_path.split('/')[-1].split('.')[0],
                                                                nowname)
    logging_dir = config["training"]["logging"]["logging_dir"]

    accelerator_project_config = ProjectConfiguration(project_dir=logging_dir,
                                                      logging_dir=logging_dir,
                                                      total_limit=config["training"]["logging"][
                                                          "checkpoints_total_limit"])

    if os.path.isfile(config["data"]["prompts"][0]):
        with open(config["data"]["prompts"][0], "r") as f:
            config["data"]["prompts"] = [line.strip() for line in f.readlines()]
    elif os.path.isdir(config["data"]["prompts"][0]):
        validation_prompts_dir = config["data"]["prompts"][0]
        prompts = []
        for d in validation_prompts_dir:
            files = [os.path.join(d, caption_file) for caption_file in os.listdir(d) if f.endswith(".txt")]
            for f in files:
                with open(f, "r") as f:
                    prompts.extend([line.strip() for line in f.readlines()])

        config["data"]["prompts"] = prompts

    if config["data"]["max_generated_samples"] is not None:
        config["data"]["prompts"] = config["data"]["prompts"][:config["data"]["max_generated_samples"]]

    accelerator = Accelerator(
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        mixed_precision=config["training"]["mixed_precision"],
        log_with=config["training"]["logging"]["report_to"],
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
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        set_seed(config.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if config["training"]["logging"]["logging_dir"] is not None:
            os.makedirs(config["training"]["logging"]["logging_dir"], exist_ok=True)

            # dump the args to a yaml file
            logging.info("Project config")
            print(OmegaConf.to_yaml(config))
            OmegaConf.save(config, os.path.join(logging_dir, "config.yaml"))

        if config["training"]["hf_hub"]["push_to_hub"]:
            repo_id = create_repo(
                repo_id=config["training"]["hf_hub"]["hub_model_id"] or Path(logging_dir).name, exist_ok=True,
                token=config["training"]["hf_hub"]["hub_token"]
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="tokenizer", revision=config.revision
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
            config.pretrained_model_name_or_path, subfolder="text_encoder", revision=config.revision
        )

        text_projection = CLIPModel.from_pretrained(config.clip_model_name_or_path).text_projection

        vae = AutoencoderKL.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="vae", revision=config.revision
        )

    unet = UNet2DConditionModelGated.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="unet",
        revision=config.non_ema_revision,
        down_block_types=config["model"]["unet"]["unet_down_blocks"],
        mid_block_type=config["model"]["unet"]["unet_mid_block"],
        up_block_types=config["model"]["unet"]["unet_up_blocks"],
        gated_ff=config["model"]["unet"]["gated_ff"]
    )
    # unet_structure, unet_structure_widths = unet.get_structure()
    unet_structure = unet.get_structure()
    hyper_net = HyperStructure(input_dim=text_encoder.config.hidden_size,
                               seq_len=text_encoder.config.max_position_embeddings,
                               structure=unet_structure,
                               wn_flag=config["model"]["hypernet"]["weight_norm"],
                               inner_dim=config["model"]["hypernet"]["inner_dim"])

    quantizer = StructureVectorQuantizer(n_e=config["model"]["quantizer"]["num_arch_vq_codebook_embeddings"],
                                         structure=unet_structure,
                                         beta=config["model"]["quantizer"]["arch_vq_beta"],
                                         temperature=config["model"]["quantizer"]["quantizer_T"],
                                         base=config["model"]["quantizer"]["quantizer_base"],
                                         depth_order=config["model"]["quantizer"]["depth_order"],
                                         non_zero_width=config["model"]["quantizer"]["non_zero_width"])

    r_loss = ResourceLoss(p=config["training"]["losses"]["resource_loss"]["pruning_target"],
                          loss_type=config["training"]["losses"]["resource_loss"]["type"])

    clip_loss = ClipLoss(structure=unet_structure,
                         temperature=config["training"]["losses"]["contrastive_clip_loss"]["temperature"])

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_projection.requires_grad_(False)

    # Create EMA for the unet.
    if config["model"]["unet"]["use_ema"]:
        ema_unet = UNet2DConditionModelGated.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="unet",
            revision=config.revision,
            down_block_types=config["model"]["unet"]["unet_down_blocks"],
            up_block_types=config["model"]["unet"]["unet_up_blocks"]
        )
        ema_unet = EMAModel(ema_unet.parameters(),
                            model_cls=UNet2DConditionModelGated,
                            model_config=ema_unet.config)

    unet.eval()
    unet.freeze()
    hyper_net.train()
    quantizer.train()

    if config["training"]["enable_xformers_memory_efficient_attention"]:
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
                if config.use_ema:
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
            if config.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                if isinstance(model, (UNet2DConditionModel, UNet2DConditionModelGated)):
                    load_model = UNet2DConditionModelGated.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    model.total_flops = load_model.total_flops
                    del load_model
                elif isinstance(model, HyperStructure):
                    load_model = HyperStructure.from_pretrained(input_dir, subfolder="hypernet")
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model
                elif isinstance(model, StructureVectorQuantizer):
                    load_model = StructureVectorQuantizer.from_pretrained(input_dir, subfolder="quantizer")
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model
                else:
                    models.append(model)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if config["training"]["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config["training"]["allow_tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config["training"]["optim"]["scale_lr"]:
        config["training"]["optim"]["hypernet_learning_rate"] = (
                config["training"]["optim"]["hypernet_learning_rate"] *
                config["training"]["optim"]["gradient_accumulation_steps"] *
                config["data"]["dataloader"]["train_batch_size"] *
                accelerator.num_processes
        )
        config["training"]["optim"]["quantizer_learning_rate"] = (
                config["training"]["optim"]["quantizer_learning_rate"] *
                config["training"]["optim"]["gradient_accumulation_steps"] *
                config["data"]["dataloader"]["train_batch_size"] *
                accelerator.num_processes
        )

    # Initialize the optimizer
    if config["training"]["optim"]["use_8bit_adam"]:
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
            {"params": hyper_net.parameters(), "lr": config["training"]["optim"]["hypernet_learning_rate"]},
            {"params": quantizer.parameters(), "lr": config["training"]["optim"]["quantizer_learning_rate"]},
        ],
        lr=config["training"]["optim"]["hypernet_learning_rate"],
        betas=(config["training"]["optim"]["adam_beta1"], config["training"]["optim"]["adam_beta2"]),
        weight_decay=config["training"]["optim"]["adam_weight_decay"],
        eps=config["training"]["optim"]["adam_epsilon"],
    )

    # ##################################################################################################################
    # #################################################### Datasets ####################################################

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    dataset_name = getattr(config["data"], "dataset_name", None)
    dataset_config_name = getattr(config["data"], "dataset_config_name", None)
    data_files = getattr(config["data"], "data_files", None)
    data_dir = getattr(config["data"], "data_dir", None)

    train_data_dir = getattr(config["data"], "train_data_dir", None)
    train_data_file = getattr(config["data"], "train_data_file", None)
    train_bad_images_path = getattr(config["data"], "train_bad_images_path", None)
    max_train_samples = getattr(config["data"], "max_train_samples", None)

    validation_data_dir = getattr(config["data"], "validation_data_dir", None)
    validation_data_file = getattr(config["data"], "validation_data_file", None)
    validation_bad_images_path = getattr(config["data"], "validation_bad_images_path", None)
    max_validation_samples = getattr(config["data"], "max_validation_samples", None)

    logger.info("Loading dataset...")
    if dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            dataset_name,
            dataset_config_name,
            data_files=data_files,
            cache_dir=config.cache_dir,
            data_dir=train_data_dir,
            ignore_verifications=True
        )
    else:
        if "aesthetics" in data_dir:
            if data_files is None:
                # datafiles a list of 5-char strs from 00000 to 00200
                data_files = [f"{i:05d}" for i in range(250)]
            train_data = load_main_laion_dataset(data_dir, list(data_files))

            # Convert the loaded data into a Hugging Face Dataset
            tr_datasets = []
            for dataset_name, dataset in train_data.items():
                tr_datasets.append(Dataset.from_list(dataset))
            dataset = {'train': concatenate_datasets(tr_datasets), 'validation': None}
            del tr_datasets

        elif "conceptual_captions" in data_dir:
            dataset = {"train": load_cc3m_dataset(data_dir,
                                                  split="train",
                                                  split_file=train_data_file,
                                                  split_dir=train_data_dir,
                                                  max_samples=max_train_samples,
                                                  bad_images_path=train_bad_images_path)}
            if validation_data_dir is not None:
                dataset["validation"] = load_cc3m_dataset(data_dir,
                                                          split="validation",
                                                          split_file=validation_data_file,
                                                          split_dir=validation_data_dir,
                                                          max_samples=max_validation_samples,
                                                          bad_images_path=validation_bad_images_path)

        else:
            data_files = {}
            if config.data.data_dir is not None:
                data_files["train"] = os.path.join(config.data.data_dir, "**")
            dataset = load_dataset(
                "imagefolder",
                data_files=data_files,
                cache_dir=config.cache_dir,
            )
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # 6. Get the column names for input/target.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(config.data.dataset_name, None)
    if config.data.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = config.data.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{config.data.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if config.data.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = config.data.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{config.data.caption_column}' needs to be one of: {', '.join(column_names)}"
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
        
        for i, c in enumerate(captions):
            print(f"{i+1}: {c}")

        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(config.model.unet.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(
                config.model.unet.resolution) if config.data.dataloader.center_crop else transforms.RandomCrop(
                config.model.unet.resolution),
            transforms.RandomHorizontalFlip() if config.data.dataloader.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
        if max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=config.seed).select(range(config.data.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

        if max_validation_samples is not None:
            dataset["validation"] = dataset["validation"].shuffle(seed=config.seed).select(
                range(config.data.max_validation_samples))
        # Set the validation transforms
        validation_dataset = dataset["validation"].with_transform(preprocess_train)
        del dataset

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
        batch_size=config["data"]["dataloader"]["train_batch_size"],
        num_workers=config["data"]["dataloader"]["dataloader_num_workers"],
    )

    if validation_dataset is not None:
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=config["data"]["dataloader"]["validation_batch_size"] * accelerator.num_processes,
            num_workers=config["data"]["dataloader"]["dataloader_num_workers"],
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.training.gradient_accumulation_steps)
    if config.training.max_train_steps is None:
        config.training.max_train_steps = config.training.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        config.training.optim.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.training.optim.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.training.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler, hyper_net, quantizer, text_projection = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler, hyper_net, quantizer, text_projection
    )

    if config.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        config.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        config.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.training.gradient_accumulation_steps)
    if overrode_max_train_steps:
        config.training.max_train_steps = config.training.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    config.training.num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        def cfg2dict(cfg: DictConfig) -> Dict:
            """
            Recursively convert OmegaConf to vanilla dict
            :param cfg:
            :return:
            """
            cfg_dict = {}
            for k, v in cfg.items():
                if type(v) == DictConfig:
                    cfg_dict[k] = cfg2dict(v)
                else:
                    cfg_dict[k] = v
            return cfg_dict

        tracker_config = cfg2dict(config)
        if config.wandb_run_name is None:
            config.wandb_run_name = f"{config.data.dataset_name if config.data.dataset_name else config.data.data_dir.split('/')[-1]}-{config.data.max_train_samples}"
        # accelerator.init_trackers(config.tracker_project_name, tracker_config,
        #                           init_kwargs={"wandb": {"name": config.wandb_run_name}}  #TODO: uncomment later.
        #                           )
    
    # Train!
    total_batch_size = (config.data.dataloader.train_batch_size * accelerator.num_processes *
                        config.training.gradient_accumulation_steps)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config.training.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.data.dataloader.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.training.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if config.training.logging.resume_from_checkpoint:
        if config.training.logging.resume_from_checkpoint != "latest":
            path = os.path.basename(config.training.logging.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(logging_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{config.training.logging.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            config.training.logging.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(logging_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, config.training.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, config.training.num_train_epochs):
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
                if config.model.unet.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += config.model.unet.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if config.model.unet.input_perturbation:
                    new_noise = noise + config.model.unet.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if config.model.unet.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                # encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                text_outputs = text_encoder(batch["input_ids"])
                encoder_hidden_states = text_outputs[0]
                pooled_output = text_outputs[1]
                text_features = text_projection(pooled_output)

                arch_vector = hyper_net(encoder_hidden_states)
                arch_vector_quantized, q_loss, _ = quantizer(arch_vector)

                # gather the arch_vector_quantized across all processes to get large batch for contrastive loss
                # encoder_hidden_states_list = accelerator.gather(encoder_hidden_states)

                text_features_list = accelerator.gather(text_features)
                arch_vector_quantized_list = accelerator.gather(arch_vector_quantized)

                # contrastive_loss = clip_loss(torch.sum(encoder_hidden_states_list, dim=1).squeeze(1), arch_vector_quantized_list)
                contrastive_loss = clip_loss(text_features_list,
                                             arch_vector_quantized_list)

                arch_vectors_separated = hyper_net.transform_structure_vector(arch_vector_quantized)
                unet.set_structure(arch_vectors_separated)
                if unet.total_flops is None:
                    with torch.no_grad():
                        unet(noisy_latents, timesteps, encoder_hidden_states)
                    unet.total_flops = unet.compute_average_flops_cost()[0]
                    unet.reset_flops_count()

                # Get the target for loss depending on the prediction type
                if config.model.unet.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=config.model.unet.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if config.training.losses.diffusion_loss.snr_gamma is None:
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
                            torch.stack(
                                [snr, config.training.losses.diffusion_loss.snr_gamma * torch.ones_like(timesteps)],
                                dim=1).min(dim=1)[0] / snr
                    )

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                curr_flops, _ = unet.compute_average_flops_cost()

                resource_ratio = (curr_flops / unet.total_flops)
                resource_loss = r_loss(resource_ratio)

                loss += config.training.losses.resource_loss.weight * resource_loss
                loss += config.training.losses.quantization_loss.weight * q_loss
                loss += config.training.losses.contrastive_clip_loss.weight * contrastive_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(config.data.dataloader.train_batch_size)).mean()
                train_loss += avg_loss.item() / config.training.gradient_accumulation_steps

                # Back-propagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), config.training.optim.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if config.use_ema:
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
                if hasattr(quantizer, "module"):
                    quantizer_embeddings = quantizer.module.embedding.weight.data.cpu().numpy()
                else:
                    quantizer_embeddings = quantizer.embedding.weight.data.cpu().numpy()
                quantizer_embeddings = quantizer_embeddings / np.linalg.norm(quantizer_embeddings, axis=1,
                                                                             keepdims=True)
                quantizer_embeddings = quantizer_embeddings @ quantizer_embeddings.T
                accelerator.log({"quantizer embeddings pairwise similarity": wandb.Image(quantizer_embeddings)},
                                step=global_step)

                # log the pairwise cosine similarity of the embeddings of the architecture vectors quantized:
                arch_vector_ = accelerator.gather(arch_vector_quantized).data.cpu().numpy()
                arch_vector_ = arch_vector_ / np.linalg.norm(arch_vector_, axis=1, keepdims=True)
                arch_vector_ = arch_vector_ @ arch_vector_.T
                accelerator.log({"arch vector pairwise similarity": wandb.Image(arch_vector_)},
                                step=global_step)

                if global_step % config.training.logging.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if config.training.logging.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(logging_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= config.training.logging.checkpoints_total_limit:
                                num_to_remove = len(
                                    checkpoints) - config.training.logging.checkpoints_total_limit.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(logging_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(logging_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        # save architecture vector quantized
                        torch.save(arch_vector_quantized, os.path.join(logging_dir,
                                                                       f"arch_vector_quantized-{global_step}.pt"))

                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0],
                    "q_loss": q_loss.detach().item(), "contrastive_loss": contrastive_loss.detach().item(),
                    "resource_loss": resource_loss.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= config.training.max_train_steps:
                break

        if validation_dataset is not None:
            validation(validation_dataloader, hyper_net, quantizer, unet, vae, text_encoder, noise_scheduler, config, accelerator,
                   global_step, weight_dtype)

        if accelerator.is_main_process:

            # generate some images from prompts
            if config.data.prompts is not None and (epoch % config.training.validation_epochs == 0 or
                                                    epoch == config.training.num_train_epochs - 1):
                if config.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                val_images = generate_samples_from_prompts(
                    hyper_net,
                    quantizer,
                    vae,
                    text_encoder,
                    tokenizer,
                    unet,
                    config,
                    accelerator,
                    weight_dtype,
                    global_step,
                )
                if config.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())

            # log quantizer embeddings samples
            if epoch % config.training.validation_epochs == 0 or epoch == config.training.num_train_epochs - 1:
                if config.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                log_quantizer_embedding_samples(
                    hyper_net,
                    quantizer,
                    vae,
                    text_encoder,
                    tokenizer,
                    unet,
                    config,
                    accelerator,
                    weight_dtype,
                    global_step,
                )
                if config.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        if config.push_to_hub:
            save_model_card(config, repo_id, val_images, repo_folder=config.output_dir)
            upload_folder(
                repo_id=repo_id,
                folder_path=logging_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--base_config_path", type=str, required=True)
    # args = parser.parse_args()
    # main(args)
    main()
