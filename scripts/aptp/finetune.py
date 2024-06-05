from omegaconf import OmegaConf

import PIL.Image

import torch
import torch.utils.checkpoint

from accelerate.utils import set_seed
from accelerate.logging import get_logger

from pdm.utils.arg_utils import parse_args
from pdm.utils.logging_utils import init_logging
from pdm.training.trainer import FineTuner


logger = get_logger(__name__)


def main():
    PIL.Image.MAX_IMAGE_PIXELS = 933120000
    torch.autograd.set_detect_anomaly(True)

    args = parse_args()
    config = OmegaConf.load(args.base_config_path)
    config.update(vars(args))

    assert config.pruning_ckpt_dir is not None, "Please provide a path to the pruning checkpoint directory."
    assert config.expert_id is not None, "Please provide an expert index to finetune"

    if config.seed is not None:
        set_seed(config.seed)

    init_logging(config)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.training.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    trainer = FineTuner(config=config)
    trainer.train()


if __name__ == "__main__":
    main()
