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
import datetime

from accelerate.utils import set_seed
from omegaconf import OmegaConf

import PIL
import accelerate
import numpy as np
import requests
import torch
import torch.utils.checkpoint
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from datasets import load_dataset, Dataset, concatenate_datasets
from packaging import version
from pdm.datasets.coco import load_coco_dataset
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, AutoModel
from transformers.utils import ContextManagers

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available

from pdm.models.diffusion import UNet2DConditionModelPruned
from pdm.models import HyperStructure
from pdm.models import StructureVectorQuantizer
from pdm.losses import ClipLoss, ResourceLoss
from pdm.utils.arg_utils import parse_args
from pdm.datasets.cc3m import load_cc3m_dataset
from pdm.datasets.laion_aes import load_main_laion_dataset
from pdm.training.trainer import DiffPruningTrainer
import torch._dynamo

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
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
    # add args to config
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
    # get current date only
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H")

    if config.name != "":
        nowname = now + f"_{config.name}"
    else:
        nowname = now

    config.training.logging.logging_dir = os.path.join(config.training.logging.logging_dir,
                                                       os.getcwd().split('/')[-2],
                                                       config.base_config_path.split('/')[-2],
                                                       config.base_config_path.split('/')[-1].split('.')[0],
                                                       nowname)

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
    data_files = getattr(config.data, "data_files", None)
    data_dir = getattr(config.data, "data_dir", None)

    train_data_dir = getattr(config.data, "train_data_dir", None)
    train_data_file = getattr(config.data, "train_data_file", None)
    train_bad_images_path = getattr(config.data, "train_bad_images_path", None)
    max_train_samples = getattr(config.data, "max_train_samples", None)

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
        if "aesthetics" in data_dir:
            data_dir = os.path.join(data_dir, "data")
            if data_files is None:
                # datafiles a list of 5-char strs from 00000 to 00250.
                data_files = [f"{i:05d}" for i in range(max_train_samples // 10000 + 1)]
            train_data = load_main_laion_dataset(data_dir, list(data_files))
            val_data = load_main_laion_dataset(data_dir, ["01000"])
            # Convert the loaded data into a Hugging Face Dataset
            tr_datasets = []
            for dataset_name, dataset in train_data.items():
                tr_datasets.append(Dataset.from_list(dataset))

            val_datasets = []
            for dataset_name, dataset in val_data.items():
                val_datasets.append(Dataset.from_list(dataset))
            if max_train_samples is None:
                max_train_samples = len(tr_datasets[0]) * 10000
            if max_validation_samples is None:
                max_validation_samples = len(val_datasets[0])
            dataset = {'train': concatenate_datasets(tr_datasets).select(range(max_train_samples)),
                       'validation': concatenate_datasets(val_datasets).select(range(max_validation_samples))}
            del tr_datasets, val_datasets, train_data, val_data

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


        elif "coco" in data_dir:
            dataset = {"train": load_coco_dataset(os.path.join(data_dir, "images", "train2017"),

                                                  os.path.join(data_dir, "annotations", "captions_train2017.json")),

                       "validation": load_coco_dataset(os.path.join(data_dir, "images", "val2017"),

                                                       os.path.join(data_dir, "annotations", "captions_val2017.json"))}

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

        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

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
        # sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

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

    validation_transforms = transforms.Compose(
        [
            transforms.Resize(config.model.unet.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(
                config.model.unet.resolution) if config.data.dataloader.center_crop else transforms.RandomCrop(
                config.model.unet.resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    def preprocess_train(examples):
        # if image_column contains urls or paths to files, convert to bytes
        # check if image_column path exists:
        if isinstance(examples[image_column][0], str):
            if not os.path.exists(examples[image_column][0]):
                downloaded_images = []
                for image in examples[image_column]:
                    try:
                        # download image and convert it to a PIL image
                        downloaded_images.append(PIL.Image.open(requests.get(image, stream=True).raw))
                    except:
                        # remove the caption if the image is not found
                        downloaded_images.append(None)
                examples[image_column] = downloaded_images
            else:
                examples[image_column] = [PIL.Image.open(image) for image in examples[image_column]]

        images = [image.convert("RGB") if image is not None else image for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) if image is not None else image for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        examples["mpnet_embeddings"] = get_mpnet_embeddings(examples[caption_column], is_train=True)
        return examples

    def preprocess_validation(examples):
        # if image_column contains urls or paths to files, convert to bytes
        # check if image_column path exists:
        if isinstance(examples[image_column][0], str):
            if not os.path.exists(examples[image_column][0]):
                downloaded_images = []
                for image in examples[image_column]:
                    try:
                        # download image and convert it to a PIL image
                        downloaded_images.append(PIL.Image.open(requests.get(image, stream=True).raw))
                    except:
                        # remove the caption if the image is not found
                        downloaded_images.append(None)
                examples[image_column] = downloaded_images
            else:
                examples[image_column] = [PIL.Image.open(image) for image in examples[image_column]]

        images = [image.convert("RGB") if image is not None else image for image in examples[image_column]]
        examples["pixel_values"] = [validation_transforms(image) if image is not None else image for image in images]
        examples["input_ids"] = tokenize_captions(examples, is_train=False)
        examples["mpnet_embeddings"] = get_mpnet_embeddings(examples[caption_column], is_train=False)
        return examples

    def preprocess_prompts(examples):
        examples["mpnet_embeddings"] = get_mpnet_embeddings(examples["prompts"], is_train=False)
        return examples

    def collate_fn(examples):
        examples = [example for example in examples if example["pixel_values"] is not None]
        if len(examples) == 0:
            return {"pixel_values": torch.tensor([]), "input_ids": torch.tensor([]),
                    "mpnet_embeddings": torch.tensor([])}
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        mpnet_embeddings = torch.stack([example["mpnet_embeddings"] for example in examples])
        mpnet_embeddings = mpnet_embeddings.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values, "input_ids": input_ids, "mpnet_embeddings": mpnet_embeddings}

    def prompts_collate_fn(examples):
        prompts = [example["prompts"] for example in examples]
        prompt_embdeddings = torch.stack([example["mpnet_embeddings"] for example in examples])
        prompt_embdeddings = prompt_embdeddings.to(memory_format=torch.contiguous_format).float()
        return {"prompts": prompts, "mpnet_embeddings": prompt_embdeddings}

    def filter_dataset(dataset, train_indices=None, validation_indices=None):
        if train_indices is None:
            train_captions = dataset["train"][caption_column]
            validation_captions = dataset["validation"][caption_column]
            train_filtering_dataloader = torch.utils.data.DataLoader(train_captions, batch_size=4096, shuffle=False)
            validation_filtering_dataloader = torch.utils.data.DataLoader(validation_captions, batch_size=4096,
                                                                          shuffle=False)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            hyper_net.to(device)
            quantizer.to(device)
            mpnet_model.to(device)
            hyper_net.eval()
            quantizer.eval()
            train_indices = []
            validation_indices = []
            with torch.no_grad():
                for batch in train_filtering_dataloader:
                    batch = get_mpnet_embeddings(batch, is_train=True)
                    arch_v = hyper_net(batch)
                    indices = quantizer.get_cosine_sim_min_encoding_indices(arch_v)
                    train_indices.append(indices)
                for batch in validation_filtering_dataloader:
                    batch = get_mpnet_embeddings(batch, is_train=False)
                    arch_v = hyper_net(batch)
                    indices = quantizer.get_cosine_sim_min_encoding_indices(arch_v)
                    validation_indices.append(indices)
            train_indices = torch.cat(train_indices, dim=0)
            validation_indices = torch.cat(validation_indices, dim=0)
            torch.save(train_indices, os.path.join(config.pruning_ckpt_dir, "train_mapped_indices.pt"))
            torch.save(validation_indices, os.path.join(config.pruning_ckpt_dir, "validation_mapped_indices.pt"))
        if config.embedding_ind == -1:
            index = torch.mode(train_indices, 0).values.item()
        else:
            index = config.embedding_ind
        dataset["train"] = dataset["train"].select(torch.where(train_indices == index)[0])
        dataset["validation"] = dataset["validation"].select(torch.where(validation_indices == index)[0])
        return dataset

    tr_indices, val_indices = None, None
    if os.path.exists(os.path.join(config.pruning_ckpt_dir, "train_mapped_indices.pt")) and \
            os.path.exists(os.path.join(config.pruning_ckpt_dir, "validation_mapped_indices.pt")):
        tr_indices = torch.load(os.path.join(config.pruning_ckpt_dir, "train_mapped_indices.pt"))
        val_indices = torch.load(os.path.join(config.pruning_ckpt_dir, "validation_mapped_indices.pt"))

    dataset = filter_dataset(dataset, train_indices=tr_indices, validation_indices=val_indices)

    if config.data.prompts is None:
        config.data.prompts = dataset["validation"][caption_column][:config.data.max_generated_samples]

    # #################################################### Models ####################################################

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
        vae = AutoencoderKL.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="vae", revision=config.revision
        )

    # load embedding_gs from checkpoint_dir
    assert config.pruning_ckpt_dir is not None, "checkpoint_dir must be provided"
    assert config.embedding_ind is not None, "embedding_ind must be provided"

    embeddings_gs = torch.load(os.path.join(config.pruning_ckpt_dir, "quantizer_embeddings.pt"), map_location="cpu")

    if config.embedding_ind != -1:
        arch_v = embeddings_gs[config.embedding_ind % embeddings_gs.shape[0]].unsqueeze(0)
    else:
        arch_v = torch.ones(1, embeddings_gs.shape[1])

    teacher_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="unet",
        revision=config.revision,
    )

    unet = UNet2DConditionModelPruned.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="unet",
        revision=config.non_ema_revision,
        down_block_types=tuple(config.model.unet.unet_down_blocks),
        mid_block_type=config.model.unet.unet_mid_block,
        up_block_types=tuple(config.model.unet.unet_up_blocks),
        gated_ff=config.model.unet.gated_ff,
        ff_gate_width=config.model.unet.ff_gate_width,
        arch_vector=arch_v
    )

    r_loss = ResourceLoss(p=config.training.losses.resource_loss.pruning_target,
                          loss_type=config.training.losses.resource_loss.type)

    clip_loss = ClipLoss(
        arch_vector_temperature=config.training.losses.contrastive_clip_loss.arch_vector_temperature,
        prompt_embedding_temperature=config.training.losses.contrastive_clip_loss.prompt_embedding_temperature)

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    teacher_unet.requires_grad_(False)

    # Create EMA for the unet.
    if config.model.unet.use_ema:
        ema_unet = UNet2DConditionModelPruned.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="unet",
            revision=config.revision,
            down_block_types=config.model.unet.unet_down_blocks,
            mid_block_type=config.model.unet.unet_mid_block,
            up_block_types=config.model.unet.unet_up_blocks,
        )
        ema_unet = EMAModel(ema_unet.parameters(),
                            model_cls=UNet2DConditionModelPruned,
                            model_config=ema_unet.config)
    else:
        ema_unet = None

    unet.train()

    if config.training.enable_xformers_memory_efficient_attention:
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

    if config.training.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.training.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    del args, data_dir, data_files, dataset_config_name, dataset_name, dataset_columns, \
        train_data_dir, train_data_file, train_bad_images_path, max_train_samples, validation_data_dir, \
        validation_data_file, validation_bad_images_path, max_validation_samples

    # set hyper_net to an empty module
    hyper_net = torch.nn.Module()
    n_e = embeddings_gs.shape[0]
    quantizer = torch.nn.Module()
    quantizer.n_e = n_e

    trainer = DiffPruningTrainer(config=config,
                                 hyper_net=hyper_net,
                                 quantizer=quantizer,
                                 unet=unet,
                                 noise_scheduler=noise_scheduler,
                                 vae=vae,
                                 text_encoder=text_encoder,
                                 clip_loss=clip_loss,
                                 resource_loss=r_loss,
                                 train_dataset=dataset["train"],
                                 preprocess_train=preprocess_train,
                                 preprocess_eval=preprocess_validation,
                                 preprocess_prompts=preprocess_prompts,
                                 data_collator=collate_fn,
                                 prompts_collator=prompts_collate_fn,
                                 ema_unet=ema_unet,
                                 eval_dataset=dataset["validation"],
                                 tokenizer=tokenizer,
                                 finetuning_arch_vector=arch_v,
                                 teacher_model=teacher_unet,
                                 )

    trainer.finetune()


if __name__ == "__main__":
    main()
