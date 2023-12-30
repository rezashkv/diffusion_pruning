import logging
from pathlib import Path

import torch.nn.functional as F
import os
import shutil
from typing import Optional, Tuple, List, Dict, Callable

import diffusers
import math
import numpy as np
import torch
import transformers
import wandb
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import Dataset
from diffusers import UNet2DConditionModel, EMAModel, get_scheduler
from diffusers.utils import make_image_grid, is_wandb_available
from huggingface_hub import upload_folder, create_repo
from pdm.models import UNet2DConditionModelGated, HyperStructure, StructureVectorQuantizer
from pdm.pipelines import StableDiffusionPruningPipeline
from pdm.utils import compute_snr
from torch import nn
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase, DataCollator
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets.utils.logging import set_verbosity_error, set_verbosity_warning
from packaging import version
import accelerate

logger = get_logger(__name__)


class DiffPruningTrainer:
    def __init__(self,
                 config: DictConfig,
                 hyper_net: nn.Module,
                 quantizer: nn.Module,
                 unet: nn.Module,
                 noise_scheduler: nn.Module,
                 vae: nn.Module,
                 text_encoder: nn.Module,
                 clip_loss: nn.Module,
                 resource_loss: nn.Module,
                 train_dataset: Dataset,
                 preprocess_train: Optional[Callable] = None,
                 preprocess_eval: Optional[Callable] = None,
                 data_collator: Optional[DataCollator] = None,
                 ema_unet: nn.Module = None,
                 eval_dataset: Dataset = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None)):

        self.config = config
        self.accelerator = self.create_accelerator()
        self.hyper_net = hyper_net
        self.quantizer = quantizer
        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.vae = vae
        self.text_encoder = text_encoder
        self.train_dataset = train_dataset
        self.clip_loss = clip_loss
        self.resource_loss = resource_loss
        self.eval_dataset = eval_dataset
        self.prepare_datasets(preprocess_train, preprocess_eval)
        self.tokenizer = tokenizer
        self.configure_logging()

        # If passed along, set the training seed now.
        if config.seed is not None:
            set_seed(config.seed)
        self.create_repo()

        if config.use_ema:
            self.ema_unet = ema_unet
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            self.init_accelerate_customized_saving_hooks()

        self.train_dataloader, self.eval_dataloader = self.initialize_dataloaders(data_collator)
        self.overrode_max_train_steps = False
        self.update_config_params()

        if optimizers[0] is None:
            optimizer = self.initialize_optimizer()
        else:
            optimizer = optimizers[0]
        self.optimizer = optimizer

        if optimizers[1] is None:
            lr_scheduler = self.initialize_lr_scheduler()
        else:
            lr_scheduler = optimizers[1]
        self.lr_scheduler = lr_scheduler

        self.optimizers = optimizers
        self.init_prompts()
        self.prepare_with_accelerator()

    def create_accelerator(self):
        logging_dir = self.config.training.logging.logging_dir
        accelerator_project_config = ProjectConfiguration(project_dir=logging_dir,
                                                          logging_dir=logging_dir,
                                                          total_limit=self.config.training.logging.checkpoints_total_limit)

        return Accelerator(
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            mixed_precision=self.config.training.mixed_precision,
            log_with=self.config.training.logging.report_to,
            project_config=accelerator_project_config,
        )

    def configure_logging(self):
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

    def init_accelerate_customized_saving_hooks(self):

        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                if self.config.use_ema:
                    self.ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
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
            if self.config.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                self.ema_unet.load_state_dict(load_model.state_dict())
                self.ema_unet.to(self.accelerator.device)
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

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def prepare_with_accelerator(self):
        # Prepare everything with our `accelerator`.
        self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler, self.hyper_net, self.quantizer = (
            self.accelerator.prepare(self.unet, self.optimizer, self.train_dataloader,
                                     self.lr_scheduler, self.hyper_net, self.quantizer
                                     ))

        if self.config.use_ema:
            self.ema_unet.to(self.accelerator.device)

    def initialize_optimizer(self):
        if self.config.training.optim.scale_lr:
            self.config.training.optim.hypernet_learning_rate = (
                    self.config.training.optim.hypernet_learning_rate *
                    self.config.training.optim.gradient_accumulation_steps *
                    self.config.data.dataloader.train_batch_size *
                    self.accelerator.num_processes
            )
            self.config.raining.optim.quantizer_learning_rate = (
                    self.config.training.optim.quantizer_learning_rate *
                    self.config.training.optim.gradient_accumulation_steps *
                    self.config.data.dataloader.train_batch_size *
                    self.accelerator.num_processes
            )

        # Initialize the optimizer
        if self.config.training.optim.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        optimizer = optimizer_cls(
            [
                {"params": self.hyper_net.parameters(), "lr": self.config.training.optim.hypernet_learning_rate},
                {"params": self.quantizer.parameters(), "lr": self.config.training.optim.quantizer_learning_rate},
            ],
            lr=self.config.training.optim.hypernet_learning_rate,
            betas=(self.config.training.optim.adam_beta1, self.config.training.optim.adam_beta2),
            weight_decay=self.config.training.optim.adam_weight_decay,
            eps=self.config.training.optim.adam_epsilon,
        )
        return optimizer

    def initialize_dataloaders(self, collate_fn):
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=self.config.data.dataloader.train_batch_size,
            num_workers=self.config.data.dataloader.dataloader_num_workers,
        )

        if self.eval_dataset is not None:
            eval_dataloader = torch.utils.data.DataLoader(
                self.eval_dataset,
                shuffle=False,
                collate_fn=collate_fn,
                batch_size=self.config.data.dataloader.validation_batch_size * self.accelerator.num_processes,
                num_workers=self.config.data.dataloader.dataloader_num_workers,
            )
        else:
            eval_dataloader = None
        return train_dataloader, eval_dataloader

    def update_config_params(self):
        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.training.gradient_accumulation_steps)
        if self.config.training.max_train_steps is None:
            self.config.training.max_train_steps = self.config.training.num_train_epochs * self.num_update_steps_per_epoch
            self.overrode_max_train_steps = True

    def initialize_lr_scheduler(self):
        lr_scheduler = get_scheduler(
            self.config.training.optim.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.training.optim.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.config.training.max_train_steps * self.accelerator.num_processes,
        )
        return lr_scheduler

    def init_trackers(self):
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initialize automatically on the main process.
        if self.accelerator.is_main_process:
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

            tracker_config = cfg2dict(self.config)
            if self.config.wandb_run_name is None:
                self.config.wandb_run_name = f"{self.config.data.dataset_name if self.config.data.dataset_name else self.config.data.data_dir.split('/')[-1]}-{self.config.data.max_train_samples}"
            self.accelerator.init_trackers(self.config.tracker_project_name, tracker_config,
                                           init_kwargs={"wandb": {"name": self.config.wandb_run_name}})

    def load_checkpoint(self):
        first_epoch = 0
        logging_dir = self.config.training.logging.logging_dir
        # Potentially load in the weights and states from a previous save
        if self.config.training.logging.resume_from_checkpoint:
            if self.config.training.logging.resume_from_checkpoint != "latest":
                path = os.path.basename(self.config.training.logging.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(logging_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.config.training.logging.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.config.training.logging.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(logging_dir, path))
                global_step = int(path.split("-")[1])

                initial_global_step = global_step
                first_epoch = global_step // self.num_update_steps_per_epoch

        else:
            initial_global_step = 0

        return initial_global_step, first_epoch

    def pre_train_setup(self):
        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
            self.config.mixed_precision = self.accelerator.mixed_precision
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
            self.config.mixed_precision = self.accelerator.mixed_precision

        # Move text_encode and vae to gpu and cast to weight_dtype
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.training.gradient_accumulation_steps)
        if self.overrode_max_train_steps:
            self.config.training.max_train_steps = self.config.training.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.config.training.num_train_epochs = math.ceil(
            self.config.training.max_train_steps / num_update_steps_per_epoch)

        self.init_trackers()

    def train(self):
        self.pre_train_setup()
        # Train!
        logging_dir = self.config.training.logging.logging_dir
        total_batch_size = (self.config.data.dataloader.train_batch_size * self.accelerator.num_processes *
                            self.config.training.gradient_accumulation_steps)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.config.training.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.config.data.dataloader.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.config.training.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.config.training.max_train_steps}")
        global_step = 0
        initial_global_step, first_epoch = self.load_checkpoint()

        progress_bar = tqdm(
            range(0, self.config.training.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_local_main_process,
        )

        for epoch in range(first_epoch, self.config.training.num_train_epochs):
            train_loss = 0.0
            for step, batch in enumerate(self.train_dataloader):
                self.hyper_net.train()
                self.quantizer.train()
                self.unet.reset_flops_count()
                with self.accelerator.accumulate(self.unet):
                    # Convert images to latent space
                    latents = self.vae.encode(batch["pixel_values"].to(self.weight_dtype)).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    if self.config.model.unet.noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += self.config.model.unet.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                        )
                    if self.config.model.unet.input_perturbation:
                        new_noise = noise + self.config.model.unet.input_perturbation * torch.randn_like(noise)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,),
                                              device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    if self.config.model.unet.input_perturbation:
                        noisy_latents = self.noise_scheduler.add_noise(latents, new_noise, timesteps)
                    else:
                        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

                    arch_vector = self.hyper_net(encoder_hidden_states)
                    arch_vector_quantized, q_loss, _ = self.quantizer(arch_vector)

                    # gather the arch_vector_quantized across all processes to get large batch for contrastive loss
                    encoder_hidden_states_list = self.accelerator.gather(encoder_hidden_states)
                    arch_vector_quantized_list = self.accelerator.gather(arch_vector_quantized)
                    contrastive_loss = self.clip_loss(torch.sum(encoder_hidden_states_list, dim=1).squeeze(1),
                                                      arch_vector_quantized_list)

                    if self.unet.total_flops is None:
                        with torch.no_grad():
                            self.unet.set_structure(arch_vector_quantized)
                            self.unet(noisy_latents, timesteps, encoder_hidden_states)
                            self.unet.total_flops = self.unet.compute_average_flops_cost()[0]
                            self.unet.reset_flops_count()

                    self.unet.set_structure(arch_vector_quantized)

                    # Get the target for loss depending on the prediction type
                    if self.config.model.unet.prediction_type is not None:
                        # set prediction_type of scheduler if defined
                        self.noise_scheduler.register_to_config(prediction_type=self.config.model.unet.prediction_type)

                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

                    # Predict the noise residual and compute loss
                    model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    if self.config.training.losses.diffusion_loss.snr_gamma is None:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    else:
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = compute_snr(self.noise_scheduler, timesteps)
                        if self.noise_scheduler.config.prediction_type == "v_prediction":
                            # Velocity objective requires that we add one to SNR values before we divide by them.
                            snr = snr + 1
                        mse_loss_weights = (
                                torch.stack(
                                    [snr,
                                     self.config.training.losses.diffusion_loss.snr_gamma * torch.ones_like(timesteps)],
                                    dim=1).min(dim=1)[0] / snr
                        )

                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()

                    curr_flops, _ = self.unet.compute_average_flops_cost()

                    resource_ratio = (curr_flops / self.unet.total_flops)
                    resource_loss = self.resource_loss(resource_ratio)

                    loss += self.config.training.losses.resource_loss.weight * resource_loss
                    loss += self.config.training.losses.quantization_loss.weight * q_loss
                    loss += self.config.training.losses.contrastive_clip_loss.weight * contrastive_loss

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = self.accelerator.gather(loss.repeat(self.config.data.dataloader.train_batch_size)).mean()
                    train_loss += avg_loss.item() / self.config.training.gradient_accumulation_steps

                    # Back-propagate
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(),
                                                         self.config.training.optim.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    if self.config.use_ema:
                        self.ema_unet.step(self.unet.parameters())
                    progress_bar.update(1)
                    global_step += 1
                    self.accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0
                    self.accelerator.log({"resource_ratio": resource_ratio}, step=global_step)
                    self.accelerator.log({"resource_loss": resource_loss}, step=global_step)
                    self.accelerator.log({"commitment_loss": q_loss}, step=global_step)
                    self.accelerator.log({"contrastive_loss": contrastive_loss}, step=global_step)
                    self.accelerator.log({"lr": self.lr_scheduler.get_last_lr()[0]}, step=global_step)

                    # log the pairwise cosine similarity of the embeddings of the quantizer:
                    if hasattr(self.quantizer, "module"):
                        quantizer_embeddings = self.quantizer.module.embedding.weight.data.cpu().numpy()
                    else:
                        quantizer_embeddings = self.quantizer.embedding.weight.data.cpu().numpy()
                    quantizer_embeddings = quantizer_embeddings / np.linalg.norm(quantizer_embeddings, axis=1,
                                                                                 keepdims=True)
                    quantizer_embeddings = quantizer_embeddings @ quantizer_embeddings.T
                    self.accelerator.log(
                        {"quantizer embeddings pairwise similarity": wandb.Image(quantizer_embeddings)},
                        step=global_step)

                    # log the pairwise cosine similarity of the embeddings of the architecture vectors quantized:
                    arch_vector_ = self.accelerator.gather(arch_vector_quantized).data.cpu().numpy()
                    arch_vector_ = arch_vector_ / np.linalg.norm(arch_vector_, axis=1, keepdims=True)
                    arch_vector_ = arch_vector_ @ arch_vector_.T
                    self.accelerator.log({"arch vector pairwise similarity": wandb.Image(arch_vector_)},
                                         step=global_step)

                    if global_step % self.config.training.logging.checkpointing_steps == 0:
                        if self.accelerator.is_main_process:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if self.config.training.logging.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(logging_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= self.config.training.logging.checkpoints_total_limit:
                                    num_to_remove = len(
                                        checkpoints) - self.config.training.logging.checkpoints_total_limit.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(logging_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(logging_dir, f"checkpoint-{global_step}")
                            self.accelerator.save_state(save_path)

                            # save architecture vector quantized
                            torch.save(arch_vector_quantized, os.path.join(logging_dir,
                                                                           f"arch_vector_quantized-{global_step}.pt"))

                            logger.info(f"Saved state to {save_path}")

                logs = {"step_loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0],
                        "q_loss": q_loss.detach().item(), "contrastive_loss": contrastive_loss.detach().item(),
                        "resource_loss": resource_loss.detach().item()}
                progress_bar.set_postfix(**logs)

                if global_step >= self.config.training.max_train_steps:
                    break
            if self.eval_dataset is not None:
                self.validate(epoch)

            if self.accelerator.is_main_process:

                # generate some validation images
                if self.config.data.prompts is not None and (epoch % self.config.training.validation_epochs == 0 or
                                                             epoch == self.config.training.num_train_epochs - 1):
                    if self.config.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        self.ema_unet.store(self.unet.parameters())
                        self.ema_unet.copy_to(self.unet.parameters())
                    val_images = self.generate_samples_from_prompts(epoch)
                    if self.config.use_ema:
                        # Switch back to the original UNet parameters.
                        self.ema_unet.restore(self.unet.parameters())

                # log quantizer embeddings samples
                if epoch % self.config.training.validation_epochs == 0 or epoch == self.config.training.num_train_epochs - 1:
                    if self.config.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        self.ema_unet.store(self.unet.parameters())
                        self.ema_unet.copy_to(self.unet.parameters())
                    self.log_quantizer_embedding_samples(epoch)
                    if self.config.use_ema:
                        # Switch back to the original UNet parameters.
                        self.ema_unet.restore(self.unet.parameters())
        # Create the pipeline using the trained modules and save it.
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            if self.config.push_to_hub:
                self.save_model_card(self.repo_id, val_images, repo_folder=self.config.output_dir)
                upload_folder(
                    repo_id=self.repo_id,
                    folder_path=logging_dir,
                    commit_message="End of training",
                    ignore_patterns=["step_*", "epoch_*"],
                )

        self.accelerator.end_training()

    def validate(self, global_step):

        progress_bar = tqdm(
            range(0, len(self.eval_dataloader)),
            initial=0,
            desc="Val Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_local_main_process,
        )
        val_loss = 0.0
        for step, batch in enumerate(self.eval_dataloader):

            self.hyper_net.eval()
            self.quantizer.eval()
            self.unet.eval()
            self.unet.reset_flops_count()
            with self.accelerator.split_between_processes(batch) as batch:
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                with torch.no_grad():
                    # Convert images to latent space
                    latents = self.vae.encode(batch["pixel_values"].to(self.weight_dtype)).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    if self.config.model.unet.noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += self.config.model.unet.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                        )
                    if self.config.model.unet.input_perturbation:
                        new_noise = noise + self.config.model.unet.input_perturbation * torch.randn_like(noise)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,),
                                              device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    if self.config.model.unet.input_perturbation:
                        noisy_latents = self.noise_scheduler.add_noise(latents, new_noise, timesteps)
                    else:
                        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

                    arch_vector = self.hyper_net(encoder_hidden_states)
                    arch_vector_quantized, q_loss, _ = self.quantizer(arch_vector)

                    self.unet.set_structure(arch_vector_quantized)

                    # Get the target for loss depending on the prediction type
                    if self.config.model.unet.prediction_type is not None:
                        # set prediction_type of scheduler if defined
                        self.noise_scheduler.register_to_config(prediction_type=self.config.model.unet.prediction_type)

                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

                    # Predict the noise residual and compute loss
                    model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    if self.config.training.losses.diffusion_loss.snr_gamma is None:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    else:
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = compute_snr(self.noise_scheduler, timesteps)
                        if self.noise_scheduler.config.prediction_type == "v_prediction":
                            # Velocity objective requires that we add one to SNR values before we divide by them.
                            snr = snr + 1
                        mse_loss_weights = (
                                torch.stack(
                                    [snr,
                                     self.config.training.losses.diffusion_loss.snr_gamma * torch.ones_like(timesteps)],
                                    dim=1).min(dim=1)[0] / snr
                        )

                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = self.accelerator.gather(
                        loss.repeat(self.config.data.dataloader.validation_batch_size)).mean()
                    val_loss += avg_loss.item() / len(self.eval_dataloader)
                    progress_bar.update(1)
                    step += 1
                self.accelerator.log({"val_loss": val_loss})
                logs = {"val_step_loss": loss.detach().item()}
                progress_bar.set_postfix(**logs)

    def generate_samples_from_prompts(self, epoch):
        logger.info("Generating samples from the given prompts... ")

        self.hyper_net.eval()
        self.quantizer.eval()
        pipeline = StableDiffusionPruningPipeline.from_pretrained(
            self.config.pretrained_model_name_or_path,
            vae=self.accelerator.unwrap_model(self.vae),
            text_encoder=self.accelerator.unwrap_model(self.text_encoder),
            tokenizer=self.tokenizer,
            unet=self.accelerator.unwrap_model(self.unet),
            safety_checker=None,
            revision=self.config.revision,
            torch_dtype=self.weight_dtype,
            hyper_net=self.hyper_net,
            quantizer=self.quantizer,
        )

        pipeline = pipeline.to(self.accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        if self.config.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

        if self.config.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=self.accelerator.device).manual_seed(self.config.seed)

        image_output_dir = os.path.join(self.config["training"]["logging"]["logging_dir"], "prompt_images",
                                        f"epoch_{epoch}")
        os.makedirs(image_output_dir, exist_ok=True)
        images = []
        for step in range(0, len(self.config.data.prompts),
                          self.config.data.dataloader.validation_batch_size * self.accelerator.num_processes):
            batch = self.config.data.prompts[
                    step:step + self.config.data.dataloader.validation_batch_size * self.accelerator.num_processes]
            with torch.autocast("cuda"):
                with self.accelerator.split_between_processes(batch) as batch:
                    gen_images = pipeline(batch, num_inference_steps=self.config.training.num_inference_steps,
                                          generator=generator).images
                    gen_images = self.accelerator.gather(gen_images)
                    images += gen_images
                    for i, image in enumerate(gen_images):
                        try:
                            image.save(os.path.join(image_output_dir, f"{batch[i]}.png"))
                        except Exception as e:
                            logger.error(f"Error saving image {batch[i]}: {e}")

        for tracker in self.accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
            elif tracker.name == "wandb":
                tracker.log(
                    {
                        "validation": [
                            wandb.Image(image, caption=f"{i}: {self.config.data.prompts[i]}")
                            for i, image in enumerate(images)
                        ]
                    }
                )
            else:
                logger.warn(f"image logging not implemented for {tracker.name}")

        del pipeline
        torch.cuda.empty_cache()

        return images

    def log_quantizer_embedding_samples(self, epoch):
        logger.info("Sampling from quantizer... ")

        self.hyper_net.eval()
        self.quantizer.eval()
        pipeline = StableDiffusionPruningPipeline.from_pretrained(
            self.config.pretrained_model_name_or_path,
            vae=self.accelerator.unwrap_model(self.vae),
            text_encoder=self.accelerator.unwrap_model(self.text_encoder),
            tokenizer=self.tokenizer,
            unet=self.accelerator.unwrap_model(self.unet),
            safety_checker=None,
            revision=self.config.revision,
            torch_dtype=self.weight_dtype,
            hyper_net=self.hyper_net,
            quantizer=self.quantizer,
        )

        pipeline = pipeline.to(self.accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        if self.config.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

        if self.config.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=self.accelerator.device).manual_seed(self.config.seed)

        image_output_dir = os.path.join(self.config["training"]["logging"]["logging_dir"], "quantizer_embedding_images",
                                        f"epoch_{epoch}")
        os.makedirs(image_output_dir, exist_ok=True)

        images = []

        if hasattr(pipeline.quantizer, "module"):
            n_e = pipeline.quantizer.module.n_e
        else:
            n_e = pipeline.quantizer.n_e
        quantizer_embedding_gumbel_sigmoid = []
        for step in range(0, n_e, self.config.data.dataloader.validation_batch_size * self.accelerator.num_processes):
            indices = torch.arange(step,
                                   step + self.config.data.dataloader.validation_batch_size * self.accelerator.num_processes,
                                   device=self.accelerator.device)
            with torch.autocast("cuda"):
                with self.accelerator.split_between_processes(indices) as indices:
                    gen_images, quantizer_embed_gs = pipeline.quantizer_samples(indices=indices,
                                                                                num_inference_steps=self.config.training.num_inference_steps,
                                                                                generator=generator)
                    gen_images = gen_images.images
                    gen_images = self.accelerator.gather(gen_images)
                    quantizer_embed_gs = self.accelerator.gather(quantizer_embed_gs)
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
        for tracker in self.accelerator.trackers:
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

    def create_repo(self):
        logging_dir = self.config.training.logging.logging_dir
        # Handle the repository creation
        if self.accelerator.is_main_process:
            if self.config.training.logging.logging_dir is not None:
                os.makedirs(self.config.training.logging.logging_dir, exist_ok=True)

                # dump the args to a yaml file
                logging.info("Project config")
                print(OmegaConf.to_yaml(self.config))
                OmegaConf.save(self.config, os.path.join(self.config.training.logging.logging_dir, "config.yaml"))

            if self.config["training"]["hf_hub"]["push_to_hub"]:
                self.repo_id = create_repo(
                    repo_id=self.config["training"]["hf_hub"]["hub_model_id"] or Path(logging_dir).name, exist_ok=True,
                    token=self.config["training"]["hf_hub"]["hub_token"]
                ).repo_id

    def save_model_card(
            self,
            repo_id: str,
            images=None,
            repo_folder=None,
    ):
        img_str = ""
        if len(images) > 0:
            image_grid = make_image_grid(images, 1, len(self.config.data.prompts))
            image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
            img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

        yaml = f"""
    ---
    license: creativeml-openrail-m
    base_model: {self.config.pretrained_model_name_or_path}
    datasets:
    - {self.config.data.dataset_name}
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

    This pipeline was pruned from **{self.config.pretrained_model_name_or_path}** on the **{self.config.data.dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: {self.config.data.prompts}: \n
    {img_str}

    ## Pipeline usage

    You can use the pipeline like so:

    ```python
    from diffusers import DiffusionPipeline
    import torch

    pipeline = DiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
    prompt = "{self.config.data.prompts[0]}"
    image = pipeline(prompt).images[0]
    image.save("my_image.png")
    ```

    ## Training info

    These are the key hyperparameters used during training:

    * Epochs: {self.config.training.num_train_epochs}
    * Hypernet Learning rate: {self.config.training.optim.hypernet_learning_rate}
    * Quantizer Learning rate: {self.config.training.optim.quantizer_learning_rate}
    * Batch size: {self.config.data.dataloader.train_batch_size}
    * Gradient accumulation steps: {self.config.training.gradient_accumulation_steps}
    * Image resolution: {self.config.model.unet.resolution}
    * Mixed-precision: {self.config.mixed_precision}

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

    def init_prompts(self):
        if os.path.isfile(self.config["data"]["prompts"][0]):
            with open(self.config["data"]["prompts"][0], "r") as f:
                self.config["data"]["prompts"] = [line.strip() for line in f.readlines()]
        elif os.path.isdir(self.config["data"]["prompts"][0]):
            validation_prompts_dir = self.config["data"]["prompts"][0]
            prompts = []
            for d in validation_prompts_dir:
                files = [os.path.join(d, caption_file) for caption_file in os.listdir(d) if f.endswith(".txt")]
                for f in files:
                    with open(f, "r") as f:
                        prompts.extend([line.strip() for line in f.readlines()])

            self.config["data"]["prompts"] = prompts

        if self.config["data"]["max_generated_samples"] is not None:
            self.config["data"]["prompts"] = self.config["data"]["prompts"][
                                             :self.config["data"]["max_generated_samples"]]

    def prepare_datasets(self, preprocess_train, preprocess_eval):
        with self.accelerator.main_process_first():
            if self.config.data.max_train_samples is not None:
                self.train_dataset = self.train_dataset.shuffle(seed=self.config.seed).select(
                    range(self.config.data.max_train_samples))
            # Set the training transforms
            self.train_dataset = self.train_dataset.with_transform(preprocess_train)

            if self.eval_dataset is not None:
                if self.config.data.max_validation_samples is not None:
                    self.eval_dataset = self.eval_dataset.shuffle(seed=self.config.seed).select(
                        range(self.config.data.max_validation_samples))
                    # Set the validation transforms
                self.eval_dataset = self.eval_dataset.with_transform(preprocess_eval)
