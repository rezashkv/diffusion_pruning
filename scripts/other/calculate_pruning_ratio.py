import logging
import os

from omegaconf import OmegaConf

import torch
import torch.utils.checkpoint

from accelerate.utils import set_seed
from accelerate.logging import get_logger

from diffusers import UNet2DConditionModel
from diffusers.utils import check_min_version

from transformers import CLIPTextModel,AutoModel
from transformers.utils import ContextManagers

from pdm.models.unet import UNet2DConditionModelGated
from pdm.models import HyperStructure, StructureVectorQuantizer
from pdm.utils.arg_utils import parse_args
from pdm.utils.op_counter_orig import count_ops_and_params
from pdm.utils.dist_utils import deepspeed_zero_init_disabled_context_manager


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

logger = get_logger(__name__)


def main():
    args = parse_args()
    config = OmegaConf.load(args.base_config_path)
    config.update(vars(args))

    assert config.pruning_ckpt_dir is not None, "Please provide a path to the pruning checkpoint directory."

    if config.seed is not None:
        set_seed(config.seed)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # #################################################### Models ####################################################

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

    pretrained_config = UNet2DConditionModel.load_config(config.pretrained_model_name_or_path, subfolder="unet")

    sample_inputs = {'sample': torch.randn(1, pretrained_config["in_channels"], pretrained_config["sample_size"],
                                           pretrained_config["sample_size"]),
                     'timestep': torch.ones((1,)).long(),
                     'encoder_hidden_states': text_encoder(torch.tensor([[100]]))[0],
                     }

    unet = UNet2DConditionModelGated.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="unet",
        revision=config.non_ema_revision,
        down_block_types=tuple(config.model.unet.unet_down_blocks),
        mid_block_type=config.model.unet.unet_mid_block,
        up_block_types=tuple(config.model.unet.unet_up_blocks),
        gated_ff=config.model.unet.gated_ff,
        ff_gate_width=config.model.unet.ff_gate_width
    )

    mpnet_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    unet_structure = unet.get_structure()
    hyper_net = HyperStructure(input_dim=mpnet_model.config.hidden_size,
                               structure=unet_structure,
                               wn_flag=config.model.hypernet.weight_norm,
                               linear_bias=config.model.hypernet.linear_bias)

    if config.pruning_type == "multi-expert":
        embeddings_gs = torch.load(os.path.join(config.pruning_ckpt_dir, "quantizer_embeddings.pt"), map_location="cpu")
    else:
        quantizer = StructureVectorQuantizer.from_pretrained(config.pruning_ckpt_dir, subfolder="quantizer")
        embeddings_gs = quantizer.gumbel_sigmoid_trick(hyper_net.arch)

    arch_vecs_separated = hyper_net.transform_structure_vector(
        torch.ones((1, embeddings_gs.shape[1]), device=embeddings_gs.device))

    unet.set_structure(arch_vecs_separated)

    flops, params = count_ops_and_params(unet, sample_inputs)

    logging.info(
        "Full UNet's Params/MACs calculated by OpCounter:\tparams: {:.3f}M\t MACs: {:.3f}G".format(
            params / 1e6, flops / 1e9))

    sanity_flops_dict = unet.calc_flops()
    prunable_flops_list = [[e / sanity_flops_dict['prunable_flops'] for e in elem] for elem in
                           unet.get_prunable_flops()]

    unet.prunable_flops_list = prunable_flops_list
    unet.resource_info_dict = sanity_flops_dict

    sanity_string = "Our MACs calculation:\t"
    for k, v in sanity_flops_dict.items():
        if isinstance(v, torch.Tensor):
            sanity_string += f" {k}: {v.item() / 1e9:.3f}\t"
        else:
            sanity_string += f" {k}: {v / 1e9:.3f}\t"
    logging.info(sanity_string)

    arch_vectors_separated = hyper_net.transform_structure_vector(embeddings_gs)
    unet.set_structure(arch_vectors_separated)

    flops_dict = unet.calc_flops()
    resource_ratios = flops_dict['cur_total_flops'] / (unet.resource_info_dict['cur_total_flops'].squeeze())
    logging.info(f"Resource Ratios: {resource_ratios}")
    # save the resource ratios to the checkpoint directory
    torch.save(resource_ratios, os.path.join(config.pruning_ckpt_dir, "resource_ratios.pt"))


if __name__ == "__main__":
    main()
