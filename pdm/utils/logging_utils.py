import os
import numpy as np
import torch
import torch.utils.checkpoint
from accelerate.logging import get_logger
from pdm.pipelines import StableDiffusionPruningPipeline
from diffusers.utils import is_wandb_available, make_image_grid

if is_wandb_available():
    import wandb

logger = get_logger(__name__)


def save_model_card(
        config,
        repo_id: str,
        images=None,
        repo_folder=None,
):
    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(config.data.prompts))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {config.pretrained_model_name_or_path}
datasets:
- {config.data.dataset_name}
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

This pipeline was pruned from **{config.pretrained_model_name_or_path}** on the **{config.data.dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: {config.data.prompts}: \n
{img_str}

## Pipeline usage

You can use the pipeline like so:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
prompt = "{config.data.prompts[0]}"
image = pipeline(prompt).images[0]
image.save("my_image.png")
```

## Training info

These are the key hyperparameters used during training:

* Epochs: {config.training.num_train_epochs}
* Hypernet Learning rate: {config.training.optim.hypernet_learning_rate}
* Quantizer Learning rate: {config.training.optim.quantizer_learning_rate}
* Batch size: {config.data.dataloader.train_batch_size}
* Gradient accumulation steps: {config.training.gradient_accumulation_steps}
* Image resolution: {config.model.unet.resolution}
* Mixed-precision: {config.mixed_precision}

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


def generate_samples_from_prompts(hyper_net, quantizer, vae, text_encoder, tokenizer, unet, config, accelerator,
                                  weight_dtype, epoch):
    logger.info("Generating samples from the given prompts... ")

    hyper_net.eval()
    quantizer.eval()
    pipeline = StableDiffusionPruningPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=config.revision,
        torch_dtype=weight_dtype,
        hyper_net=hyper_net,
        quantizer=quantizer,
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if config.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if config.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(config.seed)

    image_output_dir = os.path.join(config["training"]["logging"]["logging_dir"], "prompt_images", f"epoch_{epoch}")
    os.makedirs(image_output_dir, exist_ok=True)
    images = []
    for step in range(0, len(config.data.prompts), config.data.dataloader.validation_batch_size * accelerator.num_processes):
        batch = config.data.prompts[step:step + config.data.dataloader.validation_batch_size * accelerator.num_processes]
        with torch.autocast("cuda"):
            with accelerator.split_between_processes(batch) as batch:
                gen_images = pipeline(batch, num_inference_steps=config.training.num_inference_steps,
                                      generator=generator).images
                gen_images = accelerator.gather(gen_images)
                images += gen_images
                for i, image in enumerate(gen_images):
                    try:
                        image.save(os.path.join(image_output_dir, f"{batch[i]}.png"))
                    except Exception as e:
                        logger.error(f"Error saving image {batch[i]}: {e}")

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {config.data.prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images


def log_quantizer_embedding_samples(hyper_net, quantizer, vae, text_encoder, tokenizer, unet, config, accelerator,
                                    weight_dtype, epoch):
    logger.info("Sampling from quantizer... ")

    hyper_net.eval()
    quantizer.eval()
    pipeline = StableDiffusionPruningPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=config.revision,
        torch_dtype=weight_dtype,
        hyper_net=hyper_net,
        quantizer=quantizer,
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if config.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if config.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(config.seed)

    image_output_dir = os.path.join(config["training"]["logging"]["logging_dir"], "quantizer_embedding_images",
                                    f"epoch_{epoch}")
    os.makedirs(image_output_dir, exist_ok=True)

    images = []

    if hasattr(pipeline.quantizer, "module"):
        n_e = pipeline.quantizer.module.n_e
    else:
        n_e = pipeline.quantizer.n_e
    quantizer_embedding_gumbel_sigmoid = []
    for step in range(0, n_e, config.data.dataloader.validation_batch_size * accelerator.num_processes):
        indices = torch.arange(step, step + config.data.dataloader.validation_batch_size * accelerator.num_processes,
                               device=accelerator.device)
        with torch.autocast("cuda"):
            with accelerator.split_between_processes(indices) as indices:
                gen_images, quantizer_embed_gs = pipeline.quantizer_samples(indices=indices,
                                                                            num_inference_steps=config.training.num_inference_steps,
                                                                            generator=generator)
                gen_images = gen_images.images
                gen_images = accelerator.gather(gen_images)
                quantizer_embed_gs = accelerator.gather(quantizer_embed_gs)
                quantizer_embedding_gumbel_sigmoid.append(quantizer_embed_gs)
                images += gen_images
                for i, image in enumerate(gen_images):
                    try:
                        image.save(os.path.join(image_output_dir, f"code-{step + i}.png"))
                    except Exception as e:
                        logger.error(f"Error saving image from code {step + i}: {e}")
    quantizer_embedding_gumbel_sigmoid = torch.cat(quantizer_embedding_gumbel_sigmoid, dim=0)
    torch.save(quantizer_embedding_gumbel_sigmoid, os.path.join(image_output_dir,
                                                                "quantizer_embeddings_gumbel_sigmoid.pt"))
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("quantizer embedding images", np_images, epoch, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "quantizer embedding images": [
                        wandb.Image(image, caption=f"Code: {i}")
                        for i, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images
