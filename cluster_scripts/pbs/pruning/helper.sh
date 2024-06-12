#!/bin/bash

MAIN_HOSTNAME=$1
MAIN_PORT=$2
NNODES=$3
NGPUS_PER_NODE=$4
CONFIG_PATH=$5
WANDB_RUN_NAME=$6

source ~/.bashrc
conda activate pdm

cd /path/to/diffusion_pruning/scripts/aptp || exit

torchrun \
  --nnodes $NNODES \
  --nproc_per_node=$NGPUS_PER_NODE \
  --rdzv_endpoint=$MAIN_HOSTNAME:$MAIN_PORT \
  --rdzv_id=12345 \
  --rdzv_backend=c10d \
  prune.py \
  --base_config_path $CONFIG_PATH \
  --cache_dir \
  /scratch/yucongd/rs/.cache/huggingface/ \
  --seed \
  43 \
  --wandb_run_name \
  $WANDB_RUN_NAME 2>&1