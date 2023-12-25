#!/bin/bash
#SBATCH --job-name=sd-prune-cc # Specify a name for your job
#SBATCH --output=logs/out-%x-%j.log       # Specify the output log file
#SBATCH --error=logs/err-%x-%j.log         # Specify the error log file
#SBATCH --cpus-per-task=16
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --gres=gpu:rtxa5000:8    # Number of GPUs to request and specify the GPU type
#SBATCH --time=12:00:00           # Maximum execution time (HH:MM:SS)
#SBATCH --partition=scavenger     # Partition name
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --mem=48G                # Memory per node

set -x -e

# log the sbatch environment
echo "start time: $(date)"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_JOB_PARTITION"=$SLURM_JOB_PARTITION
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE"=$SLURM_GPUS_ON_NODE
echo "SLURM_SUBMIT_DIR"=$SLURM_SUBMIT_DIR

# Training setup
GPUS_PER_NODE=8
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=5560
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

echo "MASTER_ADDR"=$MASTER_ADDR
echo "NNODES"=$NNODES
echo "NODE_RANK"=$NODE_RANK

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

source ~/.bashrc
conda activate conda-env-name
cd /path/to/diffusion_pruning/scripts || exit

DATASET="conceptual_captions"

CMD=" \
  prune.py \
  --train_data_dir '/fs/vulcan-datasets/conceptual_captions' \
  --cache_dir '/path/to/cache' \
  --output_dir '/path/to/output' \
  --validation_prompts '/fs/vulcan-datasets/conceptual_captions/Validation_GCC-1.1.0-Validation.tsv' \
  --max_train_samples 2000 \
  --num_validation_samples 16 \
  --pruning_target 0.8 \
  --resource_loss_type log \
  --resource_loss_weight 1.0 \
  --q_loss_weight 1.0 \
  --contrastive_loss_weight 0.1 \
  --num_arch_vq_codebook_embeddings 32 \
  --num_train_epochs 50 \
  --train_batch_size 8 \
  --checkpointing_steps 500 \
  --validation_epochs 1 \
  --resume_from_checkpoint latest \
  --hypernet_learning_rate 5e-2 \
  --quantizer_learning_rate 1e-3 \
  --lr_scheduler linear \
"

LAUNCHER="accelerate launch \
    --multi_gpu \
    --num_machines $NNODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port $MASTER_PORT \
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

srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER $CMD" 2>&1

echo "END TIME: $(date)"
