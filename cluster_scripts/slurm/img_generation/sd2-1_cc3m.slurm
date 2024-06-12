#!/bin/bash
#SBATCH --job-name=sd-gen-img-cc3m            # Specify a name for your job
#SBATCH --output=logs/out-%x-%j.log         # Specify the output log file
#SBATCH --error=logs/err-%x-%j.log          # Specify the error log file
#SBATCH --cpus-per-task=8                   # Number of CPU cores per task
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks-per-node=1                 # crucial - only 1 task per dist per node!
#SBATCH --gres=gpu:a100:2                   # Number of GPUs to request and specify the GPU type
#SBATCH --time=24:00:00                     # Maximum execution time (HH:MM:SS)
#SBATCH --mem=64G                           # Memory per node

set -x -e

# Some info
echo "start time: $(date)"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_JOB_PARTITION"=$SLURM_JOB_PARTITION
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE"=$SLURM_GPUS_ON_NODE
echo "SLURM_SUBMIT_DIR"s=$SLURM_SUBMIT_DIR

# Training setup
MAIN_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MAIN_PORT=5560
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID
WORLD_SIZE=$(($SLURM_GPUS_ON_NODE * $NNODES))

echo "MAIN_ADDR"=$MAIN_ADDR
echo "NNODES"=$NNODES
echo "NODE_RANK"=$NODE_RANK

export WANDB_CACHE_DIR=/path/to/wandb

export NCCL_DEBUG=INFO

source ~/.bashrc
conda activate pdm

cd /path/to/diffusion_pruning/scripts/metrics || exit

finetuning_ckpt_dir=$1

for arch_dir in "${finetuning_ckpt_dir}"/arch*
do
    expert_id=$(echo "$arch_dir" | grep -o '[0-9]*$')
    checkpoint_dir=$(find -d "${arch_dir}/checkpoint-[0-9]*" | sort -rV | head -n 1) # the last checkpoint

    echo "$checkpoint_dir"
    echo "$expert_id"

    CMD=" \
    generate_fid_images.py \
    --base_config_path \
    /path/to/diffusion_pruning/configs/img_generation/sd-2-1_cc3m.yaml \
    --cache_dir \
    /path/to/.cache/huggingface/ \
    --finetuning_ckpt_dir \
    $checkpoint_dir \
    --expert_id \
    $expert_id \
    "
    LAUNCHER="accelerate launch \
      --mixed_precision no \
      --dynamo_backend no \
      --multi_gpu \
      --num_machines $NNODES \
      --num_processes $WORLD_SIZE \
      --main_process_ip $MAIN_ADDR \
      --main_process_port $MAIN_PORT \
      --machine_rank \$SLURM_PROCID \
      --role $SLURMD_NODENAME: \
      --rdzv_conf rdzv_backend=c10d \
      --max_restarts 0 \
      --tee 3 \
    "
    SRUN_ARGS=" \
      --wait=60 \
      --kill-on-bad-exit=1 \
    "
    srun "$SRUN_ARGS" --jobid "$SLURM_JOB_ID" bash -c "$LAUNCHER $CMD" 2>&1
    echo "END TIME: $(date)"
done