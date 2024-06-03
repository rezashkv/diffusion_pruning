# skeleton: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py

import logging
from functools import partial

from accelerate.utils import set_seed
from omegaconf import OmegaConf

import PIL
import accelerate

import torch
import torch.utils.checkpoint
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from packaging import version

from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, AutoModel
from transformers.utils import ContextManagers

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available

from pdm.models.diffusion import UNet2DConditionModelGated
from pdm.models import HyperStructure
from pdm.models import StructureVectorQuantizer
from pdm.losses import ContrastiveLoss, ResourceLoss
from pdm.utils.arg_utils import parse_args
from pdm.utils.logging_utils import init_logging
from pdm.utils.data_utils import (get_dataset, get_transforms, preprocess_samples, collate_fn,
                                  preprocess_prompts, prompts_collate_fn)

from pdm.training.trainer import DiffPruningTrainer

logger = get_logger(__name__)


def main():
    PIL.Image.MAX_IMAGE_PIXELS = 933120000
    torch.autograd.set_detect_anomaly(True)

    args = parse_args()
    config = OmegaConf.load(args.base_config_path)
    config.update(vars(args))

    if config.seed is not None:
        set_seed(config.seed)

    init_logging(config)

    # ################################################### Models ####################################################
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

    unet = UNet2DConditionModelGated.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="unet",
        revision=config.non_ema_revision,
        down_block_types=tuple(config.model.unet.unet_down_blocks),
        mid_block_type=config.model.unet.unet_mid_block,
        up_block_types=tuple(config.model.unet.unet_up_blocks),
        gated_ff=config.model.unet.gated_ff,
        ff_gate_width=config.model.unet.ff_gate_width,

    )

    mpnet_tokenizer = AutoTokenizer.from_pretrained(config.prompt_encoder_model_name_or_path)
    mpnet_model = AutoModel.from_pretrained(config.prompt_encoder_model_name_or_path)

    unet_structure = unet.get_structure()
    hyper_net = HyperStructure(input_dim=mpnet_model.config.hidden_size,
                               structure=unet_structure,
                               wn_flag=config.model.hypernet.weight_norm,
                               linear_bias=config.model.hypernet.linear_bias,
                               single_arch_param=config.model.hypernet.single_arch_param
                               )

    quantizer = StructureVectorQuantizer(n_e=config.model.quantizer.num_arch_vq_codebook_embeddings,
                                         structure=unet_structure,
                                         beta=config.model.quantizer.arch_vq_beta,
                                         temperature=config.model.quantizer.quantizer_T,
                                         base=config.model.quantizer.quantizer_base,
                                         depth_order=list(config.model.quantizer.depth_order),
                                         non_zero_width=config.model.quantizer.non_zero_width,
                                         resource_aware_normalization=config.model.quantizer.resource_aware_normalization,
                                         optimal_transport=config.model.quantizer.optimal_transport
                                         )

    r_loss = ResourceLoss(p=config.training.losses.resource_loss.pruning_target,
                          loss_type=config.training.losses.resource_loss.type)

    contrastive_loss = ContrastiveLoss(
        arch_vector_temperature=config.training.losses.contrastive_loss.arch_vector_temperature,
        prompt_embedding_temperature=config.training.losses.contrastive_loss.prompt_embedding_temperature)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if config.model.unet.use_ema:
        ema_unet = UNet2DConditionModelGated.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="unet",
            revision=config.revision,
            down_block_types=config.model.unet.unet_down_blocks,
            mid_block_type=config.model.unet.unet_mid_block,
            up_block_types=config.model.unet.unet_up_blocks,
        )
        ema_unet = EMAModel(ema_unet.parameters(),
                            model_cls=UNet2DConditionModelGated,
                            model_config=ema_unet.config)
    else:
        ema_unet = None

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

    # #################################################### Datasets ####################################################

    logging.info("Loading datasets...")

    dataset = get_dataset(config)

    # 6. Get the column names for input/target.
    column_names = dataset["train"].column_names
    image_column = config.data.image_column
    if image_column not in column_names:
        raise ValueError(
            f"--image_column' value '{config.data.image_column}' needs to be one of: {', '.join(column_names)}"
        )

    caption_column = config.data.caption_column
    if caption_column not in column_names:
        raise ValueError(
            f"--caption_column' value '{config.data.caption_column}' needs to be one of: {', '.join(column_names)}"
        )

    # Preprocessors and transformers
    train_transform, validation_transform = get_transforms(config)
    preprocess_train = partial(preprocess_samples, tokenizer=tokenizer, mpnet_model=mpnet_model,
                               mpnet_tokenizer=mpnet_tokenizer, transform=train_transform,
                               image_column=image_column, caption_column=caption_column, is_train=True)
    preprocess_validation = partial(preprocess_samples, tokenizer=tokenizer, mpnet_model=mpnet_model,
                                    mpnet_tokenizer=mpnet_tokenizer, transform=validation_transform,
                                    image_column=image_column, caption_column=caption_column, is_train=False)

    preprocess_prompts_ = partial(preprocess_prompts, mpnet_model=mpnet_model, mpnet_tokenizer=mpnet_tokenizer)

    if config.data.prompts is None:
        config.data.prompts = dataset["validation"][caption_column][:config.data.max_generated_samples]

    del args

    trainer = DiffPruningTrainer(config=config,
                                 hyper_net=hyper_net,
                                 quantizer=quantizer,
                                 unet=unet,
                                 noise_scheduler=noise_scheduler,
                                 vae=vae,
                                 text_encoder=text_encoder,
                                 contrastive_loss=contrastive_loss,
                                 resource_loss=r_loss,
                                 train_dataset=dataset["train"],
                                 preprocess_train=preprocess_train,
                                 preprocess_eval=preprocess_validation,
                                 preprocess_prompts=preprocess_prompts_,
                                 data_collator=collate_fn,
                                 prompts_collator=prompts_collate_fn,
                                 ema_unet=ema_unet,
                                 eval_dataset=dataset["validation"],
                                 tokenizer=tokenizer,
                                 )

    trainer.train()


if __name__ == "__main__":
    main()
