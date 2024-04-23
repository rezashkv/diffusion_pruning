import logging
from pathlib import Path

import numpy as np
import safetensors
import torch.nn.functional as F
import os
import shutil
from typing import Optional, Tuple, Dict, Callable

import diffusers
import math
import torch
import torchvision
import transformers
import wandb
from accelerate.utils import ProjectConfiguration
from datasets import Dataset
from diffusers import UNet2DConditionModel, EMAModel, get_scheduler
from diffusers.utils import make_image_grid, is_wandb_available
from huggingface_hub import upload_folder, create_repo
from pdm.models import UNet2DConditionModelGated, HyperStructure, StructureVectorQuantizer
from pdm.models.diffusion import UNet2DConditionModelPruned
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


from pdm.utils.op_counter_orig import count_ops_and_params
from pdm.utils.logging_utils import create_image_grid_from_indices, create_heatmap

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
                 preprocess_prompts: Optional[Callable] = None,
                 data_collator: Optional[DataCollator] = None,
                 prompts_collator: Optional[DataCollator] = None,
                 ema_unet: nn.Module = None,
                 eval_dataset: Dataset = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 finetuning_arch_vector=None,
                 teacher_model: Optional[nn.Module] = None,
                 ):

        self.config = config
        self.accelerator = self.create_accelerator()
        self.hyper_net = hyper_net
        self.quantizer = quantizer
        self.unet = unet
        self.teacher_model = teacher_model
        self.noise_scheduler = noise_scheduler
        self.vae = vae
        self.text_encoder = text_encoder
        self.train_dataset = train_dataset

        self.clip_loss = clip_loss
        self.resource_loss = resource_loss
        self.eval_dataset = eval_dataset
        self.prepare_datasets(preprocess_train, preprocess_eval, preprocess_prompts)
        self.tokenizer = tokenizer
        self.configure_logging()

        self.create_logging_dir()

        if config.use_ema:
            self.ema_unet = ema_unet
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            self.init_accelerate_customized_saving_hooks()

        (self.train_dataloader, self.eval_dataloader,
         self.prompt_dataloader, self.quantizer_embeddings_dataloader) = self.initialize_dataloaders(data_collator,
                                                                                                     prompts_collator)

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

        self.finetuning_arch_vector = finetuning_arch_vector

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
                        # save the quantizer embeddings
                        torch.save(model.embedding_gs, os.path.join(output_dir, "quantizer_embeddings.pt"))

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
                if isinstance(model, UNet2DConditionModelPruned):
                    state_dict = safetensors.torch.load_file(os.path.join(input_dir, "unet",
                                                                          "diffusion_pytorch_model.safetensors"))
                    model.load_state_dict(state_dict)
                    del state_dict
                # load diffusers style into model
                elif isinstance(model, (UNet2DConditionModel, UNet2DConditionModelGated)):
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

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def init_trackers(self, resume=False):
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
                self.config.wandb_run_name = (
                    f"{self.config.data.dataset_name if self.config.data.dataset_name else self.config.data.data_dir.split('/')[-1]}-"
                    f"{self.config.data.max_train_samples}-steps:{self.config.training.max_train_steps}-"
                    f"h_lr:{self.config.training.optim.hypernet_learning_rate}-"
                    f"q_lr:{self.config.training.optim.quantizer_learning_rate}")
            self.accelerator.init_trackers(self.config.tracker_project_name, tracker_config,
                                           init_kwargs={"wandb": {"name": self.config.wandb_run_name,
                                                                  "dir": self.config.training.logging.wandb_log_dir,
                                                                  "resume": resume}})

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

    def prepare_with_accelerator(self):
        # Prepare everything with our `accelerator`.
        (self.unet, self.optimizer, self.train_dataloader, self.eval_dataloader, self.prompt_dataloader,
         self.quantizer_embeddings_dataloader, self.lr_scheduler, self.hyper_net,
         self.quantizer) = (self.accelerator.prepare(self.unet, self.optimizer, self.train_dataloader,
                                                     self.eval_dataloader, self.prompt_dataloader,
                                                     self.quantizer_embeddings_dataloader, self.lr_scheduler,
                                                     self.hyper_net,
                                                     self.quantizer
                                                     ))

        if self.config.use_ema:
            self.ema_unet.to(self.accelerator.device)

    def init_prompts(self):
        if os.path.isfile(self.config.data.prompts[0]):
            with open(self.config.data.prompts[0], "r") as f:
                self.config.data.prompts = [line.strip() for line in f.readlines()]
        elif os.path.isdir(self.config.data.prompts[0]):
            validation_prompts_dir = self.config.data.prompts[0]
            prompts = []
            for d in validation_prompts_dir:
                files = [os.path.join(d, caption_file) for caption_file in os.listdir(d) if f.endswith(".txt")]
                for f in files:
                    with open(f, "r") as f:
                        prompts.extend([line.strip() for line in f.readlines()])

            self.config.data.prompts = prompts

        if self.config.data.max_generated_samples is not None:
            self.config.data.prompts = self.config.data.prompts[
                                       :self.config.data.max_generated_samples]

    def prepare_datasets(self, preprocess_train, preprocess_eval, preprocess_prompts):
        with self.accelerator.main_process_first():
            if self.config.data.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(
                    range(min(self.config.data.max_train_samples, len(self.train_dataset))))
            # Set the training transforms
            self.train_dataset = self.train_dataset.with_transform(preprocess_train)

            if self.eval_dataset is not None:
                if self.config.data.max_validation_samples is not None:
                    self.eval_dataset = self.eval_dataset.select(
                        range(min(self.config.data.max_validation_samples, len(self.eval_dataset))))
                    # Set the validation transforms
                self.eval_dataset = self.eval_dataset.with_transform(preprocess_eval)

            if self.config.data.prompts is not None:
                self.prompt_dataset = Dataset.from_dict({"prompts": self.config.data.prompts}).with_transform(
                    preprocess_prompts)


    def initialize_dataloaders(self, data_collate_fn, prompts_collate_fn):
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=data_collate_fn,
            batch_size=self.config.data.dataloader.train_batch_size,
            num_workers=self.config.data.dataloader.dataloader_num_workers,
        )

        if self.eval_dataset is not None:
            eval_dataloader = torch.utils.data.DataLoader(
                self.eval_dataset,
                shuffle=False,
                collate_fn=data_collate_fn,
                batch_size=self.config.data.dataloader.validation_batch_size * self.accelerator.num_processes,
                num_workers=self.config.data.dataloader.dataloader_num_workers,
            )
        else:
            eval_dataloader = None

        if self.config.data.prompts is None:
            self.config.data.prompts = []

        # create a torch dataloader from given prompts
        prompt_dataloader = torch.utils.data.DataLoader(self.prompt_dataset,
                                                        batch_size=self.config.data.dataloader.image_generation_batch_size * self.accelerator.num_processes,
                                                        shuffle=False,
                                                        collate_fn=prompts_collate_fn,
                                                        num_workers=self.config.data.dataloader.dataloader_num_workers)

        n_e = self.quantizer.n_e

        q_embedding_dataloader = torch.utils.data.DataLoader(torch.arange(n_e),
                                                             batch_size=self.config.data.dataloader.image_generation_batch_size * self.accelerator.num_processes,
                                                             shuffle=False,
                                                             num_workers=self.config.data.dataloader.dataloader_num_workers)

        return train_dataloader, eval_dataloader, prompt_dataloader, q_embedding_dataloader

    def initialize_optimizer(self):
        # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
        scaling_factor = self.config.training.gradient_accumulation_steps * self.config.data.dataloader.train_batch_size * self.accelerator.num_processes

        if self.config.training.optim.scale_lr:
            self.config.training.optim.hypernet_learning_rate = (
                    self.config.training.optim.hypernet_learning_rate * math.sqrt(scaling_factor)
            )
            self.config.training.optim.quantizer_learning_rate = (
                    self.config.training.optim.quantizer_learning_rate * math.sqrt(scaling_factor)
            )
            self.config.training.optim.unet_learning_rate = (
                    self.config.training.optim.unet_learning_rate * math.sqrt(scaling_factor)
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
                {"params": self.hyper_net.parameters(), "lr": self.config.training.optim.hypernet_learning_rate,
                 "weight_decay": self.config.training.optim.hypernet_weight_decay},
                {"params": self.quantizer.parameters(), "lr": self.config.training.optim.quantizer_learning_rate,
                 "weight_decay": self.config.training.optim.quantizer_weight_decay},
                {"params": self.unet.parameters(), "lr": self.config.training.optim.unet_learning_rate,
                 "weight_decay": self.config.training.optim.unet_weight_decay},
            ],
            betas=(self.config.training.optim.adam_beta1, self.config.training.optim.adam_beta2),
            eps=self.config.training.optim.adam_epsilon,
        )
        return optimizer

    def initialize_lr_scheduler(self):
        lr_scheduler = get_scheduler(
            self.config.training.optim.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.training.optim.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.config.training.max_train_steps * self.accelerator.num_processes,
        )
        return lr_scheduler

    def update_config_params(self):
        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.training.gradient_accumulation_steps)
        if self.config.training.max_train_steps is None:
            self.config.training.max_train_steps = self.config.training.num_train_epochs * self.num_update_steps_per_epoch
            self.overrode_max_train_steps = True

    def save_checkpoint(self, logging_dir, global_step):
        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
        if self.config.training.logging.checkpoints_total_limit is not None:
            checkpoints = os.listdir(logging_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

            # before we save the new checkpoint, we need to have at _most_
            # `checkpoints_total_limit - 1` checkpoints
            if len(checkpoints) >= self.config.training.logging.checkpoints_total_limit:
                num_to_remove = len(
                    checkpoints) - self.config.training.logging.checkpoints_total_limit + 1
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
        logger.info(f"Saved state to {save_path}")

    def load_checkpoint(self):
        first_epoch = 0
        logging_dir = self.config.training.logging.logging_dir
        # Potentially load in the weights and states from a previous save
        if self.config.training.logging.resume_from_checkpoint:
            if self.config.training.logging.resume_from_checkpoint != "latest":
                path = self.config.training.logging.resume_from_checkpoint
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
                if self.config.training.logging.resume_from_checkpoint != "latest":
                    self.accelerator.load_state(path)
                else:
                    self.accelerator.load_state(os.path.join(logging_dir, path))
                global_step = int(os.path.basename(path).split("-")[1])

                initial_global_step = global_step
                first_epoch = global_step // self.num_update_steps_per_epoch

        else:
            initial_global_step = 0

        return initial_global_step, first_epoch

    def init_weight_dtype(self):
        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
            self.config.mixed_precision = self.accelerator.mixed_precision
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
            self.config.mixed_precision = self.accelerator.mixed_precision

    def pre_train_setup(self):
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.training.gradient_accumulation_steps)
        if self.overrode_max_train_steps:
            self.config.training.max_train_steps = self.config.training.num_train_epochs * num_update_steps_per_epoch
        # Afterward we recalculate our number of training epochs
        self.config.training.num_train_epochs = math.ceil(
            self.config.training.max_train_steps / num_update_steps_per_epoch)

    def train(self):
        self.init_weight_dtype()

        # Move text_encode and vae to gpu and cast to weight_dtype
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)

        self.pre_train_setup()
        # Train!
        logging_dir = self.config.training.logging.logging_dir
        total_batch_size = (self.config.data.dataloader.train_batch_size * self.accelerator.num_processes *
                            self.config.training.gradient_accumulation_steps)

        initial_global_step, first_epoch = self.load_checkpoint()
        global_step = initial_global_step

        if len(self.accelerator.trackers) == 0:
            if global_step == 0:
                self.init_trackers(resume=False)
            else:
                self.init_trackers(resume=True)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.config.training.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.config.data.dataloader.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.config.training.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.config.training.max_train_steps}")

        progress_bar = tqdm(
            range(0, self.config.training.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_main_process,
        )

        self.unet.module.freeze()

        self.block_activations = {}
        self.cast_block_act_hooks(self.unet, self.block_activations)

        for epoch in range(first_epoch, self.config.training.num_train_epochs):
            for step, batch in enumerate(self.train_dataloader):

                if batch["pixel_values"].numel() == 0:
                    continue

                train_loss = 0.0
                self.hyper_net.train()
                self.quantizer.train()
                self.unet.eval()

                # Calculating the MACs of each module of the model in the first iteration.
                if global_step == initial_global_step:
                    self.count_flops(batch)
                    self.quantizer.module.set_prunable_flops_template(self.unet.module.prunable_flops_list)

                # pruning target is for total flops. we calculate loss for prunable flops.
                if global_step == 0:
                    self.update_pruning_target()

                pretrain = self.config.training.hypernet_pretraining_steps and global_step < self.config.training.hypernet_pretraining_steps
                loss, diff_loss, distillation_loss, block_loss, q_loss, contrastive_loss, resource_loss, arch_vectors_similarity, resource_ratio, \
                    flops_dict, arch_vector_quantized, quantizer_embedding_pairwise_similarity, \
                    batch_resource_ratios = self.step(batch, pretrain=pretrain)

                # avg_loss = self.accelerator.reduce(loss, "mean")
                avg_loss = loss
                train_loss += avg_loss.item() / self.config.training.gradient_accumulation_steps

                # Back-propagate
                try:
                    self.accelerator.backward(loss)
                except RuntimeError as e:
                    if "returned nan values" in str(e):
                        logger.error("NaNs detected in the loss. Skipping batch.")
                        self.optimizer.zero_grad()
                        continue
                    else:
                        raise e

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    if self.config.use_ema:
                        self.ema_unet.step(self.unet.parameters())
                    progress_bar.update(1)
                    log_dict = {
                        "training/loss": train_loss,
                        "training/diffusion_loss": diff_loss,
                        "training/distillation_loss": distillation_loss.detach().item(),
                        "training/block_loss": block_loss.detach().item(),
                        "training/resource_loss": resource_loss.detach().item(),
                        "training/commitment_loss": q_loss.detach().item(),
                        "training/contrastive_loss": contrastive_loss.detach().item(),
                        "training/hyper_net_lr": self.lr_scheduler.get_last_lr()[0],
                        "training/quantizer_lr": self.lr_scheduler.get_last_lr()[1],
                        "training/resource_ratio": resource_ratio.detach().item(),
                    }
                    for k, v in flops_dict.items():
                        if isinstance(v, torch.Tensor):
                            log_dict[f"training/{k}"] = v.detach().mean().item()
                        else:
                            log_dict[f"training/{k}"] = v

                    self.accelerator.log(log_dict)

                    logs = {"diff_loss": diff_loss.detach().item(),
                            "dist_loss": distillation_loss.detach().item(),
                            "block_loss": block_loss.detach().item(),
                            "q_loss": q_loss.detach().item(),
                            "c_loss": contrastive_loss.detach().item(),
                            "r_loss": resource_loss.detach().item(),
                            "step_loss": loss.detach().item(),
                            "h_lr": self.lr_scheduler.get_last_lr()[0],
                            "q_lr": self.lr_scheduler.get_last_lr()[1],
                            }
                    progress_bar.set_postfix(**logs)

                    if global_step % self.config.training.validation_steps == 0:
                        if self.eval_dataset is not None:
                            self.validate(pretrain=pretrain)

                    if (global_step % self.config.training.image_logging_steps == 0 or
                            (epoch == self.config.training.num_train_epochs - 1 and step == len(
                                self.train_dataloader) - 1)):
                        img_log_dict = {}
                        img_log_dict["images/arch vector pairwise similarity image"] = wandb.Image(arch_vectors_similarity)
                        img_log_dict["images/quantizer_embedding_pairwise_similarity"] = wandb.Image(
                            quantizer_embedding_pairwise_similarity)

                        with torch.no_grad():
                            batch_resource_ratios = self.accelerator.gather_for_metrics(
                                batch_resource_ratios).cpu().numpy()

                        img_log_dict["images/batch resource ratio heatmap"] = wandb.Image(
                            create_heatmap(batch_resource_ratios, n_rows=16, n_cols=len(batch_resource_ratios) // 16))
                        self.accelerator.log(img_log_dict, log_kwargs={"wandb": {"commit": False}})

                        # generate some validation images
                        if self.config.data.prompts is not None:
                            val_images = self.generate_samples_from_prompts(global_step, pretrain=pretrain)

                        # visualize the quantizer embeddings
                        self.log_quantizer_embedding_samples(global_step)

                    global_step += 1

                if global_step >= self.config.training.max_train_steps:
                    break

            # checkpoint at the end of each epoch
            if self.accelerator.is_main_process:
                self.save_checkpoint(logging_dir, global_step)

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

    @torch.no_grad()
    def validate(self, pretrain=False):
        self.init_weight_dtype()
        # Move text_encode and vae to gpu and cast to weight_dtype
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        if len(self.accelerator.trackers) == 0:
            self.init_trackers()

        progress_bar = tqdm(
            range(0, len(self.eval_dataloader)),
            initial=0,
            desc="Val Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_main_process,
        )

        self.hyper_net.eval()
        self.quantizer.eval()
        self.unet.eval()
        (total_val_loss, total_diff_loss, total_distillation_loss, total_block_loss, total_q_loss, total_c_loss,
         total_r_loss) = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for step, batch in enumerate(self.eval_dataloader):
            if batch["pixel_values"].numel() == 0:
                continue
            loss, diff_loss, distillation_loss, block_loss, q_loss, contrastive_loss, resource_loss, _, _, _, _, _, _ = self.step(
                batch,
                pretrain=pretrain)
            # Gather the losses across all processes for logging (if we use distributed training).
            total_val_loss += loss.item()
            total_diff_loss += diff_loss.item()
            total_distillation_loss += distillation_loss.item()
            total_block_loss += block_loss.item()
            total_q_loss += q_loss.item()
            total_c_loss += contrastive_loss.item()
            total_r_loss += resource_loss.item()
            progress_bar.update(1)

        total_val_loss /= len(self.eval_dataloader)
        total_diff_loss /= len(self.eval_dataloader)
        total_q_loss /= len(self.eval_dataloader)
        total_c_loss /= len(self.eval_dataloader)
        total_distillation_loss /= len(self.eval_dataloader)
        total_block_loss /= len(self.eval_dataloader)
        total_r_loss /= len(self.eval_dataloader)

        total_val_loss = self.accelerator.reduce(torch.tensor(total_val_loss, device=self.accelerator.device),
                                                 "mean").item()
        total_diff_loss = self.accelerator.reduce(torch.tensor(total_diff_loss, device=self.accelerator.device),
                                                  "mean").item()
        total_q_loss = self.accelerator.reduce(torch.tensor(total_q_loss, device=self.accelerator.device),
                                               "mean").item()
        total_c_loss = self.accelerator.reduce(torch.tensor(total_c_loss, device=self.accelerator.device),
                                               "mean").item()
        total_r_loss = self.accelerator.reduce(torch.tensor(total_r_loss, device=self.accelerator.device),
                                               "mean").item()
        total_distillation_loss = self.accelerator.reduce(
            torch.tensor(total_distillation_loss, device=self.accelerator.device), "mean").item()
        total_block_loss = self.accelerator.reduce(torch.tensor(total_block_loss, device=self.accelerator.device),
                                                   "mean").item()

        self.accelerator.log({
            "validation/loss": total_val_loss,
            "validation/diffusion_loss": total_diff_loss,
            "validation/distillation_loss": total_distillation_loss,
            "validation/block_loss": total_block_loss,
            "validation/resource_loss": total_r_loss,
            "validation/commitment_loss": total_q_loss,
            "validation/contrastive_loss": total_c_loss,
        },
            log_kwargs={"wandb": {"commit": False}})

        del loss, diff_loss, q_loss, contrastive_loss, \
            resource_loss, total_val_loss, total_diff_loss, total_q_loss, total_c_loss, total_r_loss, \
            total_distillation_loss, total_block_loss
        torch.cuda.empty_cache()

    def step(self, batch, pretrain=False):
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
        if self.config.model.unet.max_scheduler_steps is None:
            self.config.model.unet.max_scheduler_steps = self.noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, self.config.model.unet.max_scheduler_steps, (bsz,),
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
        text_embeddings = batch["mpnet_embeddings"]

        arch_vector = self.hyper_net(text_embeddings)
        arch_vector_quantized, q_loss, _ = self.quantizer(arch_vector)

        arch_vector = self.quantizer.module.gumbel_sigmoid_trick(arch_vector)
        arch_vector_width_depth_normalized = self.quantizer.module.width_depth_normalize(arch_vector)
        with torch.no_grad():
            quantizer_embeddings = self.quantizer.module.get_codebook_entry_gumbel_sigmoid(
                torch.arange(self.quantizer.module.n_e, device=self.accelerator.device), hard=True).detach()
            quantizer_embeddings /= quantizer_embeddings.norm(dim=-1, keepdim=True)
            quantizer_embeddings_pairwise_similarity = quantizer_embeddings @ quantizer_embeddings.t()

            text_embeddings_list = [torch.zeros_like(text_embeddings) for _ in
                                    range(self.accelerator.num_processes)]
            arch_vector_list = [torch.zeros_like(arch_vector) for _ in
                                range(self.accelerator.num_processes)]
            torch.distributed.all_gather(text_embeddings_list, text_embeddings)
            torch.distributed.all_gather(arch_vector_list, arch_vector_width_depth_normalized)
        text_embeddings_list[self.accelerator.process_index] = text_embeddings
        arch_vector_list[self.accelerator.process_index] = arch_vector_width_depth_normalized
        text_embeddings_list = torch.cat(text_embeddings_list, dim=0)
        arch_vector_list = torch.cat(arch_vector_list, dim=0)

        # During hyper_net pretraining, we don't cluster the architecture vector and directly use it.
        if pretrain:
            arch_vectors_separated = self.hyper_net.module.transform_structure_vector(arch_vector)
        else:
            arch_vectors_separated = self.hyper_net.module.transform_structure_vector(arch_vector_quantized)

        contrastive_loss, arch_vectors_similarity = self.clip_loss(text_embeddings_list, arch_vector_list,
                                                                   return_similarity=True)

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

        with torch.no_grad():
            full_arch_vector = torch.ones_like(arch_vector)
            full_arch_vector = self.hyper_net.module.transform_structure_vector(full_arch_vector)
            self.unet.module.set_structure(full_arch_vector)
            full_model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample.detach()
            teacher_block_activations = self.block_activations.copy()

        self.unet.module.set_structure(arch_vectors_separated)
        # Predict the noise residual and compute loss
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        student_block_activations = self.block_activations.copy()

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

        distillation_loss = F.mse_loss(model_pred.float(), full_model_pred.float(), reduction="mean")

        block_loss = torch.tensor(0.0, device=self.accelerator.device)
        for key in student_block_activations.keys():
            block_loss += F.mse_loss(student_block_activations[key], teacher_block_activations[key].detach(),
                                     reduction="mean")
        block_loss /= len(student_block_activations)

        flops_dict = self.unet.module.calc_flops()
        curr_flops = flops_dict['cur_prunable_flops']

        # The reason is that sanity['prunable_flops'] does not have depth-related pruning flops
        # like skip connections of resnets in it.
        resource_ratios = (curr_flops / (self.unet.module.resource_info_dict['cur_prunable_flops'].squeeze()))
        resource_loss = self.resource_loss(resource_ratios.mean())

        max_loss = 1. - torch.max(resource_ratios)
        std_loss = -torch.std(resource_ratios)
        with torch.no_grad():
            batch_resource_ratios = flops_dict['cur_prunable_flops'] / (
                self.unet.module.resource_info_dict['cur_prunable_flops'].squeeze())

        diff_loss = loss.clone().detach().mean()
        loss += self.config.training.losses.resource_loss.weight * resource_loss
        loss += self.config.training.losses.contrastive_clip_loss.weight * contrastive_loss
        loss += self.config.training.losses.std_loss.weight * std_loss
        loss += self.config.training.losses.max_loss.weight * max_loss
        loss += self.config.training.losses.distillation_loss.weight * distillation_loss
        loss += self.config.training.losses.block_loss.weight * block_loss

        del latents, noise, timesteps, noisy_latents, encoder_hidden_states, text_embeddings, arch_vector, arch_vectors_separated, \
            quantizer_embeddings, text_embeddings_list, arch_vector_list, curr_flops, model_pred, target, \
            arch_vector_width_depth_normalized, full_arch_vector, full_model_pred, teacher_block_activations, \
            student_block_activations, batch

        torch.cuda.empty_cache()

        return (
            loss, diff_loss, distillation_loss, block_loss, q_loss, contrastive_loss, resource_loss,
            arch_vectors_similarity, resource_ratios.mean(),
            flops_dict, arch_vector_quantized, quantizer_embeddings_pairwise_similarity, batch_resource_ratios)

    def cast_block_act_hooks(self, unet, dicts):
        def get_activation(activation, name, residuals_present):
            if residuals_present:
                def hook(model, input, output):
                    activation[name] = output[0]
            else:
                def hook(model, input, output):
                    activation[name] = output
            return hook

        unet = self.accelerator.unwrap_model(unet)
        for i in range(len(unet.down_blocks)):
            unet.down_blocks[i].register_forward_hook(get_activation(dicts, 'd' + str(i), True))
        unet.mid_block.register_forward_hook(get_activation(dicts, 'm', False))
        for i in range(len(unet.up_blocks)):
            unet.up_blocks[i].register_forward_hook(get_activation(dicts, 'u' + str(i), False))

    def finetune(self):
        assert self.teacher_model is not None, "Teacher model is not provided for finetuning"

        self.init_weight_dtype()

        # Move text_encode and vae to gpu and cast to weight_dtype
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.teacher_model.to(self.accelerator.device, dtype=self.weight_dtype)

        self.pre_train_setup()
        # Train!
        logging_dir = self.config.training.logging.logging_dir
        total_batch_size = (self.config.data.dataloader.train_batch_size * self.accelerator.num_processes *
                            self.config.training.gradient_accumulation_steps)

        initial_global_step, first_epoch = self.load_checkpoint()
        global_step = initial_global_step

        if len(self.accelerator.trackers) == 0:
            if global_step == 0:
                self.init_trackers(resume=False)
            else:
                self.init_trackers(resume=True)

        logger.info("***** Running finetuning *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.config.training.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.config.data.dataloader.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.config.training.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.config.training.max_train_steps}")

        progress_bar = tqdm(
            range(0, self.config.training.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_main_process,
        )

        for param in self.hyper_net.parameters():
            param.requires_grad = False
        for param in self.quantizer.parameters():
            param.requires_grad = False

        if self.config.training.validation_steps > self.config.training.max_train_steps:
            self.config.training.validation_steps = self.config.training.max_train_steps // 10
        if self.config.training.image_logging_steps > self.config.training.max_train_steps:
            self.config.training.image_logging_steps = self.config.training.max_train_steps // 10
        if self.config.training.logging.auto_checkpoint_step:
            self.config.training.logging.checkpoint_step = self.config.training.num_train_epochs // 5
        else:
            self.config.training.logging.checkpoint_step = 1

        self.block_act_teacher = {}
        self.block_act_student = {}
        self.cast_block_act_hooks(self.unet, self.block_act_student)
        self.cast_block_act_hooks(self.teacher_model, self.block_act_teacher)

        for epoch in range(first_epoch, self.config.training.num_train_epochs):
            for step, batch in enumerate(self.train_dataloader):
                if batch["pixel_values"].numel() == 0:
                    continue
                train_loss = 0.0
                self.hyper_net.eval()
                self.quantizer.eval()
                self.teacher_model.eval()
                self.unet.train()

                loss, diff_loss, distillation_loss, block_loss = self.finetune_step(batch)
                # avg_loss = self.accelerator.reduce(loss, "mean")
                avg_loss = loss
                train_loss += avg_loss.item() / self.config.training.gradient_accumulation_steps

                # Back-propagate
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    if self.config.use_ema:
                        self.ema_unet.step(self.unet.parameters())
                    progress_bar.update(1)
                    log_dict = {
                        "finetuning/loss": train_loss,
                        "finetuning/diffusion_loss": diff_loss,
                        "finetuning/distillation_loss": distillation_loss.detach().item(),
                        "finetuning/block_loss": block_loss.detach().item(),
                        "finetuning/unet_lr": self.lr_scheduler.get_last_lr()[2],
                    }
                    self.accelerator.log(log_dict)

                    logs = {
                        "step_loss": loss.detach().item(),
                        "lr": self.lr_scheduler.get_last_lr()[2],
                    }
                    progress_bar.set_postfix(**logs)

                    if global_step % self.config.training.validation_steps == 0:
                        if self.eval_dataset is not None:
                            self.finetuning_validate()

                    if (global_step % self.config.training.image_logging_steps == 0 or
                            (epoch == self.config.training.num_train_epochs - 1 and step == len(
                                self.train_dataloader) - 1)):

                        # generate some validation images
                        if self.config.data.prompts is not None:
                            val_images = self.generate_samples_from_prompts_finetuning(global_step)

                    global_step += 1

                if global_step >= self.config.training.max_train_steps:
                    break

            # checkpoint at the end of each epoch
            if epoch % self.config.training.logging.checkpoint_step == 0 and self.accelerator.is_main_process:
                self.save_checkpoint(logging_dir, global_step)
                # copy arch_vector.pt to logging_dir
                shutil.copy(os.path.join(logging_dir, "arch_vector.pt"), os.path.join(logging_dir,
                                                                                      f"checkpoint-{global_step}", "unet"))

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

    def finetune_step(self, batch):
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
        if self.config.model.unet.max_scheduler_steps is None:
            self.config.model.unet.max_scheduler_steps = self.noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, self.config.model.unet.max_scheduler_steps, (bsz,),
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
        torch.cuda.empty_cache()
        with torch.no_grad():
            full_model_pred = self.teacher_model(noisy_latents, timesteps, encoder_hidden_states).sample.detach()
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

        diff_loss = loss.clone().detach().mean()
        loss *= self.config.training.losses.diffusion_loss.weight

        block_loss = torch.tensor(0.0, device=self.accelerator.device)
        for key in self.block_act_student.keys():
            block_loss += F.mse_loss(self.block_act_student[key], self.block_act_teacher[key].detach(),
                                     reduction="mean")
        block_loss /= len(self.block_act_student)
        loss += self.config.training.losses.block_loss.weight * block_loss

        distillation_loss = F.mse_loss(model_pred.float(), full_model_pred.float(), reduction="mean")
        loss += self.config.training.losses.distillation_loss.weight * distillation_loss

        del latents, noise, timesteps, noisy_latents, encoder_hidden_states, model_pred, target, full_model_pred, batch

        torch.cuda.empty_cache()

        return loss, diff_loss, distillation_loss, block_loss

    @torch.no_grad()
    def finetuning_validate(self):
        self.init_weight_dtype()
        # Move text_encode and vae to gpu and cast to weight_dtype
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        if len(self.accelerator.trackers) == 0:
            self.init_trackers()

        progress_bar = tqdm(
            range(0, len(self.eval_dataloader)),
            initial=0,
            desc="Val Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_main_process,
        )

        self.hyper_net.eval()
        self.quantizer.eval()
        self.unet.eval()
        total_val_loss = 0.0
        total_diff_loss = 0.0
        total_distillation_loss = 0.0
        total_block_loss = 0.0
        for step, batch in enumerate(self.eval_dataloader):
            if batch["pixel_values"].numel() == 0:
                continue
            loss, diff_loss, distillation_loss, block_loss = self.finetune_step(batch)
            # Gather the losses across all processes for logging (if we use distributed training).
            total_val_loss += loss.item()
            total_diff_loss += diff_loss.item()
            total_distillation_loss += distillation_loss.item()
            total_block_loss += block_loss.item()
            progress_bar.update(1)

        total_val_loss /= len(self.eval_dataloader)
        total_diff_loss /= len(self.eval_dataloader)
        total_distillation_loss /= len(self.eval_dataloader)
        total_block_loss /= len(self.eval_dataloader)

        total_val_loss = self.accelerator.reduce(torch.tensor(total_val_loss, device=self.accelerator.device),
                                                 "mean").item()
        total_diff_loss = self.accelerator.reduce(torch.tensor(total_diff_loss, device=self.accelerator.device),
                                                  "mean").item()
        total_distillation_loss = self.accelerator.reduce(torch.tensor(total_distillation_loss,
                                                                       device=self.accelerator.device),
                                                          "mean").item()
        total_block_loss = self.accelerator.reduce(torch.tensor(total_block_loss, device=self.accelerator.device),
                                                   "mean").item()

        self.accelerator.log({
            "validation/loss": total_val_loss,
            "validation/diffusion_loss": total_diff_loss,
            "validation/distillation_loss": total_distillation_loss,
            "validation/block_loss": total_block_loss
        },
            log_kwargs={"wandb": {"commit": False}})

        del loss, total_val_loss, total_diff_loss, total_distillation_loss, total_block_loss
        torch.cuda.empty_cache()

    @torch.no_grad()
    def count_flops(self, batch):

        arch_vecs_separated = self.hyper_net.module.transform_structure_vector(
            torch.ones((1, self.quantizer.module.vq_embed_dim),
                       device=self.accelerator.device))

        self.unet.module.set_structure(arch_vecs_separated)

        latents = self.vae.encode(batch["pixel_values"][:1].to(self.weight_dtype)).latent_dist.sample()
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (1,),
                                  device=self.accelerator.device).long()
        encoder_hidden_states = self.text_encoder(batch["input_ids"][:1])[0]

        flops, params = count_ops_and_params(self.unet,
                                             {'sample': latents,
                                              'timestep': timesteps,
                                              'encoder_hidden_states': encoder_hidden_states})

        logger.info(
            "UNet's Params/MACs calculated by OpCounter:\tparams: {:.3f}M\t MACs: {:.3f}G".format(
                params / 1e6, flops / 1e9))

        sanity_flops_dict = self.unet.module.calc_flops()
        prunable_flops_list = [[e / sanity_flops_dict['prunable_flops'] for e in elem] for elem in
                               self.unet.module.get_prunable_flops()]

        self.unet.module.prunable_flops_list = prunable_flops_list
        self.unet.module.resource_info_dict = sanity_flops_dict

        sanity_string = "Our MACs calculation:\t"
        for k, v in sanity_flops_dict.items():
            if isinstance(v, torch.Tensor):
                sanity_string += f" {k}: {v.item() / 1e9:.3f}\t"
            else:
                sanity_string += f" {k}: {v / 1e9:.3f}\t"
        logger.info(sanity_string)

        del latents, timesteps, encoder_hidden_states, arch_vecs_separated, flops, params, sanity_flops_dict, \
            sanity_string
        torch.cuda.empty_cache()

    @torch.no_grad()
    def update_pruning_target(self):
        p = self.config.training.losses.resource_loss.pruning_target
        p_actual = (1 - (1 - p) * self.unet.module.resource_info_dict['total_flops'] /
                    self.unet.module.resource_info_dict['cur_prunable_flops']).item()
        self.resource_loss.p = p_actual

        del p, p_actual
        torch.cuda.empty_cache()

    @torch.no_grad()
    def generate_samples_from_prompts_finetuning(self, global_step, save_images=False):
        logger.info("Generating samples from the given prompts... ")

        pipeline = self.get_pipeline()

        image_output_dir = os.path.join(self.config.training.logging.logging_dir, "prompt_images",
                                        f"step_{global_step}")
        os.makedirs(image_output_dir, exist_ok=True)
        images = []

        for step, batch in enumerate(self.prompt_dataloader):
            if self.config.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.accelerator.device).manual_seed(self.config.seed)
            gen_images = pipeline.generate_samples(batch["prompts"],
                                                   num_inference_steps=self.config.training.num_inference_steps,
                                                   generator=generator, output_type="pt"
                                                   ).images
            gen_images = self.accelerator.gather_for_metrics(gen_images)
            images += gen_images

        images = [torchvision.transforms.ToPILImage()(img) for img in images]
        images = make_image_grid(images[:4 * (len(images) // 4)], 4, len(images) // 4)

        if self.accelerator.is_main_process and save_images:
            images.save(os.path.join(image_output_dir, "prompt_images.png"))

        self.accelerator.log(
            {
                "images/prompt images": wandb.Image(images),
            },
            log_kwargs={"wandb": {"commit": False}}
        )

        del pipeline, gen_images
        torch.cuda.empty_cache()

        return images

    @torch.no_grad()
    def generate_samples_from_prompts(self, global_step, save_images=False, pretrain=False):
        logger.info("Generating samples from the given prompts... ")

        pipeline = self.get_pipeline()

        image_output_dir = os.path.join(self.config.training.logging.logging_dir, "prompt_images",
                                        f"step_{global_step}")
        os.makedirs(image_output_dir, exist_ok=True)
        images = []
        prompts_resource_ratios = []

        for step, batch in enumerate(self.prompt_dataloader):
            if self.config.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.accelerator.device).manual_seed(self.config.seed)
            gen_images, _, resource_ratios = pipeline(batch["prompts"],
                                                      num_inference_steps=self.config.training.num_inference_steps,
                                                      generator=generator, output_type="pt",
                                                      return_mapped_indices=True,
                                                      hyper_net_input=batch["mpnet_embeddings"],
                                                      pretrain=pretrain
                                                      )
            gen_images = gen_images.images
            gen_images = self.accelerator.gather_for_metrics(gen_images)
            resource_ratios = self.accelerator.gather_for_metrics(resource_ratios)
            images += gen_images
            prompts_resource_ratios += resource_ratios

        images = [torchvision.transforms.ToPILImage()(img) for img in images]
        images = make_image_grid(images[:4 * (len(images) // 4)], 4, len(images) // 4)
        prompts_resource_ratios = torch.cat(prompts_resource_ratios, dim=0).cpu().numpy()
        prompts_resource_ratios_images = create_heatmap(prompts_resource_ratios, n_rows=4,
                                                        n_cols=len(prompts_resource_ratios) // 4)

        if self.accelerator.is_main_process and save_images:
            images.save(os.path.join(image_output_dir, "prompt_images.png"))

        self.accelerator.log(
            {
                "images/prompt images": wandb.Image(images),
                "images/prompts resource ratio heatmap": wandb.Image(prompts_resource_ratios_images),
            },
            log_kwargs={"wandb": {"commit": False}}
        )

        del pipeline, gen_images
        torch.cuda.empty_cache()

        return images

    @torch.no_grad()
    def log_quantizer_embedding_samples(self, step, save_to_disk=False):
        logger.info("Sampling from quantizer... ")

        pipeline = self.get_pipeline()

        image_output_dir = os.path.join(self.config.training.logging.logging_dir, "quantizer_embedding_images",
                                        f"step_{step}")
        os.makedirs(image_output_dir, exist_ok=True)

        images = []
        quantizer_embedding_gumbel_sigmoid = []
        embeddings_resource_ratios = []

        for step, indices in enumerate(self.quantizer_embeddings_dataloader):
            if self.config.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.accelerator.device).manual_seed(self.config.seed)

            gen_images, quantizer_embed_gs, resource_ratios = pipeline.quantizer_samples(indices=indices,
                                                                                         num_inference_steps=self.config.training.num_inference_steps,
                                                                                         generator=generator,
                                                                                         output_type="pt")
            gen_images = gen_images.images
            gen_images = self.accelerator.gather_for_metrics(gen_images)
            quantizer_embed_gs = self.accelerator.gather_for_metrics(quantizer_embed_gs)
            resource_ratios = self.accelerator.gather_for_metrics(resource_ratios)
            quantizer_embedding_gumbel_sigmoid += quantizer_embed_gs
            images += gen_images
            embeddings_resource_ratios += resource_ratios

        quantizer_embedding_gumbel_sigmoid = torch.cat(quantizer_embedding_gumbel_sigmoid, dim=0)
        images = [torchvision.transforms.ToPILImage()(img) for img in images]
        images = make_image_grid(images[:4 * (len(images) // 4)], len(images) // 4, 4)
        embeddings_resource_ratios = torch.cat(embeddings_resource_ratios, dim=0).cpu().numpy()
        embeddings_resource_ratios_images = create_heatmap(embeddings_resource_ratios, n_rows=4,
                                                           n_cols=len(embeddings_resource_ratios) // 4)

        if self.accelerator.is_main_process and save_to_disk:
            images.save(os.path.join(image_output_dir, "quantizer_embedding_images.png"))
            torch.save(quantizer_embedding_gumbel_sigmoid, os.path.join(image_output_dir,
                                                                        "quantizer_embeddings_gumbel_sigmoid.pt"))

        self.accelerator.log({"images/quantizer embedding images": wandb.Image(images),
                              "images/embedding resource ratio heatmap": wandb.Image(
                                  embeddings_resource_ratios_images)},
                             log_kwargs={"wandb": {"commit": False}})

        del pipeline, quantizer_embedding_gumbel_sigmoid, gen_images, quantizer_embed_gs, images, \
            embeddings_resource_ratios, embeddings_resource_ratios_images
        torch.cuda.empty_cache()

    @torch.no_grad()
    def depth_analysis(self, n_consecutive_blocks=1):
        logger.info("Generating depth analysis samples from the given prompts... ")
        pipeline = self.get_pipeline()

        image_output_dir = os.path.join(self.config.training.logging.logging_dir, "depth_analysis_images")
        os.makedirs(image_output_dir, exist_ok=True)

        n_depth_pruned_blocks = sum([sum(d) for d in self.unet.module.get_structure()['depth']])

        # index n_depth_pruned_blocks is for no pruning
        images = {i: [] for i in range(n_depth_pruned_blocks + 1)}

        for d_block in range(n_depth_pruned_blocks + 1):
            logger.info(f"Generating samples for depth block {d_block}...")
            progress_bar = tqdm(
                range(0, len(self.config.data.prompts)),
                initial=0,
                desc="Depth Analysis Steps",
                # Only show the progress bar once on each machine.
                disable=not self.accelerator.is_main_process,
            )
            for step in range(0, len(self.config.data.prompts),
                              self.config.data.dataloader.image_generation_batch_size * self.accelerator.num_processes):
                batch = self.config.data.prompts[
                        step:step + self.config.data.dataloader.image_generation_batch_size * self.accelerator.num_processes]
                with self.accelerator.split_between_processes(batch) as batch:
                    if self.config.seed is None:
                        generator = None
                    else:
                        generator = torch.Generator(device=self.accelerator.device).manual_seed(self.config.seed)

                    if d_block == n_depth_pruned_blocks:
                        gen_images = pipeline.depth_analysis(batch,
                                                             num_inference_steps=self.config.training.num_inference_steps,
                                                             generator=generator, depth_index=None,
                                                             output_type="pt").images
                    else:
                        if n_consecutive_blocks > 1:
                            d_blocks = [(d_block + i) % n_depth_pruned_blocks for i in range(n_consecutive_blocks)]
                        gen_images = pipeline.depth_analysis(batch,
                                                             num_inference_steps=self.config.training.num_inference_steps,
                                                             generator=generator, depth_index=d_blocks,
                                                             output_type="pt").images

                    gen_images = self.accelerator.gather(gen_images)

                    # append gen_images to images dict at the same key
                    images[d_block] += gen_images

                progress_bar.update(self.config.data.dataloader.validation_batch_size * self.accelerator.num_processes)

        image_grids = {}
        for i, image in images.items():
            # convert image from tensor to PIL image
            image = [torchvision.transforms.ToPILImage()(img) for img in image]

            # make an image grid with 4 columns
            image_grid = make_image_grid(image[:4 * (len(images) // 4)], 4, len(images) // 4)
            image_grid.save(os.path.join(image_output_dir, f"depth_{i}.png"))
            image_grids[i] = image_grid

        self.accelerator.log(
            {"depth analysis": [wandb.Image(image_grid, caption=f"Depth: {i}") for i, image_grid in
                                image_grids.items()]}
        )

        del pipeline, images, image_grids
        torch.cuda.empty_cache()
        return gen_images

    def create_logging_dir(self):
        logging_dir = self.config.training.logging.logging_dir
        # Handle the repository creation
        if self.accelerator.is_main_process:
            if self.config.training.logging.logging_dir is not None:
                os.makedirs(self.config.training.logging.logging_dir, exist_ok=True)

                # dump the args to a yaml file
                logging.info("Project config")
                print(OmegaConf.to_yaml(self.config))
                OmegaConf.save(self.config, os.path.join(self.config.training.logging.logging_dir, "config.yaml"))

            if self.config.training.hf_hub.push_to_hub:
                self.repo_id = create_repo(
                    repo_id=self.config.training.hf_hub.hub_model_id or Path(logging_dir).name, exist_ok=True,
                    token=self.config.training.hf_hub.hub_token
                ).repo_id

    def get_pipeline(self):
        self.init_weight_dtype()
        # Move text_encode and vae to gpu and cast to weight_dtype
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        if len(self.accelerator.trackers) == 0:
            self.init_trackers()

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
            hyper_net=self.accelerator.unwrap_model(self.hyper_net),
            quantizer=self.accelerator.unwrap_model(self.quantizer),
        )
        pipeline = pipeline.to(self.accelerator.device)
        pipeline.set_progress_bar_config(disable=not self.accelerator.is_main_process)

        if self.config.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()
        return pipeline

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

        if self.config.wandb_run_url is not None:
            wandb_info = f"""
                More information on all the CLI arguments and the environment are available on your [`wandb` run page]({self.config.wandb_run_url}).
                """

        model_card += wandb_info

        with open(os.path.join(repo_folder, "../README.md"), "w") as f:
            f.write(yaml + model_card)
