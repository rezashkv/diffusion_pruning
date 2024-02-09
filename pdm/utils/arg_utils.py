import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Dynamic Pruning of StableDiffusion-2.1")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--clip_model_name_or_path",
        type=str,
        default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        required=False,
        help="Path to pretrained clip model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--base_config_path",
        type=str,
        required=True,
        help="Path to the model/data/training config file.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Path to the model/data/training config file.",
    )
    parser.add_argument(
        "--pruning_ckpt_dir",
        type=str,
        default=None,
        help="Path to the saved pruning checkpoint dir. used for finetuning.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Experiment name. Will be added to the name of the logging directory for easier tracking of experiments.",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use EMA model.",
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--seed",
        type=int, 
        default=43, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-dynamic-pruning",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--wandb_run_name", 
        type=str, 
        default=None, 
        help="The `run_name` argument passed to Accelerator.init_trackers"
    )
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=1,
        help="Number of consecutive blocks to prune for depth analysis.",
    )
    parser.add_argument(
        "--embedding_ind",
        type=int,
        default=0,
        help="Index of the embedding layer to prune for depth analysis.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Whether or not to push the model to the Hub.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")


    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    # if args.dataset_name is None and args.train_data_dir is None:
    #     raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    # return parser
    return args
