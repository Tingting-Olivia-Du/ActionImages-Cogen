#!/bin/bash
# Reproduce ActionImages RLBench training on 2x RTX 6000 Ada (48GB).
#
# Differences vs. the reference scripts/train.sh (which assumes 8x 80GB):
#   * DeepSpeed ZeRO-2 with CPU optimizer offload (configs/zero2_offload.json),
#     otherwise the fp32 AdamW states of the 5B DiT do not fit in 48GB.
#   * gradient_accumulation_steps 4 with 2 GPUs -> effective batch 8, i.e. the
#     same effective batch as 8 GPUs x bs1 x accum1 in the reference script.
#   * checkpoints every 1000 steps, keep 2 (each checkpoint is ~75GB with the
#     DeepSpeed optimizer state; the volume only has ~450GB free).
# Everything else (lr, warmup, resolution, num_frames, full_param, ...) matches
# scripts/train.sh. train.py itself is unmodified.
#
# Usage: bash scripts/train_rlbench_2gpu.sh [num_gpus] [extra train.py args...]

set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate ttd_train
cd /workspace/ttdu/ActionImages   # train.py resolves `training.*` and ./checkpoints from here

export TOKENIZERS_PARALLELISM=false

NUM_GPUS=${1:-2}
shift || true

# GPUs to use. Override per launch, e.g. CUDA_VISIBLE_DEVICES=2,3 bash scripts/...
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5,6}
# Needed so DeepSpeed can JIT-build the CPUAdam op for optimizer offload.
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}

export OMP_NUM_THREADS=8
# Peak VRAM measured at ~48GB/49GB per card, so keep the allocator from
# fragmenting. Both GPUs must be fully free; a co-tenant using >1GB will OOM us.
# For more headroom add `--use_gradient_checkpointing_offload True` (slower).
# Note: `--tiled True` does NOT help here -- tile_size is in latent units and
# the default 34*16=544px already exceeds the 512px input, so it is a no-op.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# W&B credentials live in /workspace/ttdu/.env (WANDB_API / WANDB_ENTITY).
ENV_FILE=${ENV_FILE:-/workspace/ttdu/.env}
if [ -f "$ENV_FILE" ]; then
    export WANDB_API_KEY=$(grep -E '^WANDB_API=' "$ENV_FILE" | cut -d= -f2-)
    export WANDB_ENTITY=$(grep -E '^WANDB_ENTITY=' "$ENV_FILE" | cut -d= -f2-)
fi
export WANDB_PROJECT="actionimages"

model_id="Wan-AI/Wan2.2-TI2V-5B"

torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    train.py \
    --deepspeed ./configs/zero2_offload.json \
    --dataset_path ./data \
    --dataset_name rlbench@1.0 \
    --output_dir ./outputs/rlbench-wan2.2-2gpu \
    --height 512 \
    --width 512 \
    --full_param True \
    --num_frames 41 \
    --model_id $model_id \
    --steps_per_epoch 8000 \
    --num_train_epochs 10000 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 4 \
    --max_grad_norm 1.0 \
    --use_gradient_checkpointing \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 2 \
    --dataloader_pin_memory True \
    --checkpoint_every_n_steps 250 \
    --checkpoint_save_top_k 2 \
    --checkpoint_monitor "train_loss" \
    --remove_unused_columns False \
    --dataloader_drop_last True \
    --prediction_loss_only True \
    --bf16 True \
    --ddp_find_unused_parameters False \
    --report_to "wandb" \
    --save_safetensors False \
    --per_device_train_batch_size 1 \
    --logging_steps 1 \
    --lr_scheduler_type "constant_with_warmup" \
    --warmup_steps 1000 \
    --seed 42 \
    "$@"
