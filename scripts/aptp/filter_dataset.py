import logging
import os

from omegaconf import OmegaConf

from PIL import Image

import torch
import torch.utils.checkpoint

from accelerate.logging import get_logger
from accelerate.utils import set_seed

from diffusers.utils import check_min_version

from transformers import AutoModel, AutoTokenizer

from pdm.models import HyperStructure, StructureVectorQuantizer
from pdm.utils.arg_utils import parse_args
from pdm.utils.data_utils import get_dataset, filter_dataset

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

logger = get_logger(__name__)


def main():
    Image.MAX_IMAGE_PIXELS = 933120000
    torch.autograd.set_detect_anomaly(True)

    args = parse_args()
    config = OmegaConf.load(args.base_config_path)
    # add args to config
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

    # ########################### Hypernet and Quantizer for Dataset Preprocessing #####################################

    hyper_net = HyperStructure.from_pretrained(config.pruning_ckpt_dir, subfolder="hypernet")
    quantizer = StructureVectorQuantizer.from_pretrained(config.pruning_ckpt_dir, subfolder="quantizer")

    mpnet_tokenizer = AutoTokenizer.from_pretrained(config.prompt_encoder_model_name_or_path)
    mpnet_model = AutoModel.from_pretrained(config.prompt_encoder_model_name_or_path)

    # #################################################### Datasets ####################################################

    logging.info("Loading datasets...")
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).
    dataset = get_dataset(config.data)
    column_names = dataset["train"].column_names

    caption_column = config.data.caption_column
    if caption_column not in column_names:
        raise ValueError(
            f"--caption_column' value '{config.data.caption_column}' needs to be one of: {', '.join(column_names)}"
        )

    if not (os.path.exists(os.path.join(config.pruning_ckpt_dir, "train_mapped_indices.pt")) and
            os.path.exists(os.path.join(config.pruning_ckpt_dir, "validation_mapped_indices.pt"))):
        tr_indices, val_indices = filter_dataset(dataset, hyper_net, quantizer, mpnet_model, mpnet_tokenizer,
                                                 caption_column=caption_column)
        torch.save(tr_indices, os.path.join(config.pruning_ckpt_dir, "train_mapped_indices.pt"))
        torch.save(val_indices, os.path.join(config.pruning_ckpt_dir, "validation_mapped_indices.pt"))


if __name__ == "__main__":
    main()
