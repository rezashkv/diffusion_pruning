import os
from functools import partial

from omegaconf import OmegaConf

import cv2
import numpy as np

import torch
import torch.utils.checkpoint

import accelerate
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from diffusers import PNDMScheduler, UNet2DConditionModel
from diffusers.utils import check_min_version
from diffusers import StableDiffusionPipeline

import safetensors

from transformers import CLIPTextModel

from pdm.models.unet import UNet2DConditionModelMagnitudePruned
from pdm.utils.arg_utils import parse_args
from pdm.utils.data_utils import get_dataset

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

    # #################################################### Accelerator #################################################
    accelerator = accelerate.Accelerator()

    # #################################################### Datasets ####################################################

    logger.info("Loading datasets...")
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    dataset_name = getattr(config.data, "dataset_name", None)
    img_col = getattr(config.data, "image_column", "image")
    capt_col = getattr(config.data, "caption_column", "caption")

    assert config.expert_id is not None, "expert index must be provided"
    assert config.finetuning_ckpt_dir is not None, "finetuning checkpoint directory must be provided"

    def collate_fn(examples, caption_column="caption", image_column="image"):
        captions = [example[caption_column] for example in examples]
        images = [example[image_column] for example in examples]
        return {"image": images, "caption": captions}

    dataset = get_dataset(config.data)

    dataset = dataset["validation"]
    logger.info("Dataset of size %d loaded." % len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=config.data.dataloader.image_generation_batch_size * accelerator.num_processes,
        num_workers=config.data.dataloader.dataloader_num_workers,
        collate_fn=partial(collate_fn, caption_column=capt_col, image_column=img_col),
    )

    dataloader = accelerator.prepare(dataloader)

    # #################################################### Models ####################################################
    # unet = UNet2DConditionModel.from_pretrained(
    #     config.pretrained_model_name_or_path,
    #     subfolder="unet",
    #     revision=config.revision,
    # )
    #
    # state_dict = safetensors.torch.load_file(os.path.join(config.finetuning_ckpt_dir, "unet",
    #                                                       "diffusion_pytorch_model.safetensors"))
    # unet.load_state_dict(state_dict)

    noise_scheduler = PNDMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        scheduler=noise_scheduler,
    )

    pipeline.load_lora_weights(config.finetuning_ckpt_dir)

    if config.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    pipeline.set_progress_bar_config(disable=not accelerator.is_main_process)

    pipeline.to(accelerator.device)

    image_output_dir = os.path.join(config.finetuning_ckpt_dir, f"{dataset_name}_fid_images")
    os.makedirs(image_output_dir, exist_ok=True)

    for batch in dataloader:
        if config.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=accelerator.device).manual_seed(config.seed)
        gen_images = pipeline(batch[capt_col], num_inference_steps=config.training.num_inference_steps,
                              generator=generator, output_type="np"
                              ).images

        for idx, caption in enumerate(batch[capt_col]):
            image_name = batch["image"][idx].split("/")[-1]
            if image_name.endswith(".jpg"):
                image_name = image_name[:-4]

            image_path = os.path.join(image_output_dir, f"{image_name}.npy")
            img = gen_images[idx]
            img = img * 255
            img = img.astype(np.uint8)
            img = cv2.resize(img, (256, 256))
            np.save(image_path, img)


if __name__ == "__main__":
    main()
