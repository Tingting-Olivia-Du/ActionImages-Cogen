#!/bin/bash
# ============================================================================
# 30-step smoke run for one TASK TEMPLATE. Proves the whole path end to end:
# warm-start checkpoint loads strict=True, the dataset serves the requested streams, and the
# DiT produces finite losses on the assembled sequence.
#
#   GPUS=4,7 bash scripts/smoke_arm.sh video+action        # the official recipe
#   GPUS=4,7 bash scripts/smoke_arm.sh video+depth         # perception: RGB given -> depth
#   GPUS=4,7 bash scripts/smoke_arm.sh video+segmentation
#   GPUS=4,7 bash scripts/smoke_arm.sh video+depth+action  # 6 segments, ~1.5x sequence
#   GPUS=4,7 bash scripts/smoke_arm.sh "video+action@0.6,video+depth@0.4"   # a whole mix
#
# Use the SAME GPU count and deepspeed config the real arm will use. full_param=True on a 5B
# model does not fit a single 48GB card without ZeRO-2 + CPU optimizer offload, so a
# single-process smoke would OOM for a reason that has nothing to do with the template.
#
# --strict_getitem True + --dataloader_num_workers 0 deliberately: a broken data path should
# surface as a traceback in the main process, not as retry spam that quietly serves a
# different sample.
#
# EXPECTED PEAK VRAM: video+action, video+depth and video+segmentation all assemble FOUR
# segments of identical shape, so their peaks should agree to within noise -- a large gap
# means something changed shape. video+depth+action is six segments: ~1.5x the sequence and
# ~2.25x the attention, so it is expected to be visibly higher.
# ============================================================================
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate ttd_train

TEMPLATE="${1:?usage: smoke_arm.sh <template_mix>, e.g. video+depth}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS="${GPUS:?set GPUS to the free device indices the real arm will use, e.g. GPUS=4,7}"
NPROC=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
PORT="${PORT:-29561}"
DS_CONFIG="${DS_CONFIG:-./configs/zero2_offload.json}"
STEPS="${STEPS:-30}"
VARIATIONS="${VARIATIONS:-0}"   # 训练只看 variation0;其余 variation 是留出测试集
SLUG="$(echo "$TEMPLATE" | tr -c '[:alnum:]+' '_')"
OUT="${OUT:-$REPO/outputs/smoke_${SLUG}}"
# Default: warm-start from the official checkpoint, matching train_arm.sh's stage-1 default.
# Pass INIT_CKPT= (empty) to smoke the from-Wan-base path instead.
INIT_CKPT="${INIT_CKPT-/workspace/ttdu/starVLA/playground/Pretrained_models/anyeZHY/ActionImages/step125750.ckpt}"

INIT_ARG=""
[ -n "$INIT_CKPT" ] && INIT_ARG="--init_ckpt_path $INIT_CKPT"
echo "init: ${INIT_CKPT:-<Wan base, no warm-start>}"

rm -rf "$OUT"   # find_latest_checkpoint would otherwise resume a previous smoke run
mkdir -p "$OUT"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader -i "$GPUS"

CUDA_VISIBLE_DEVICES=$GPUS torchrun --nnodes=1 --nproc_per_node=$NPROC --master_port $PORT \
  train.py \
  --deepspeed "$DS_CONFIG" \
  --dataset_path ./data \
  --dataset_name "rlbench_selfgen@1.0" \
  --template_mix "$TEMPLATE" \
  --variations "$VARIATIONS" \
  $INIT_ARG \
  --output_dir "$OUT" \
  --height 256 --width 256 --num_frames 41 \
  --full_param True \
  --model_id Wan-AI/Wan2.2-TI2V-5B \
  --max_steps "$STEPS" --num_train_epochs 1 --steps_per_epoch "$STEPS" \
  --learning_rate 5e-7 --warmup_steps 1000 --lr_scheduler_type constant_with_warmup \
  --gradient_accumulation_steps 1 --max_grad_norm 1.0 \
  --use_gradient_checkpointing \
  --strict_getitem True --dataloader_num_workers 0 \
  --remove_unused_columns False --dataloader_drop_last True \
  --prediction_loss_only True --bf16 True --ddp_find_unused_parameters False \
  --save_safetensors False --per_device_train_batch_size 1 \
  --logging_steps 1 --seed 42 --report_to none \
  --checkpoint_every_n_steps 999999 ${EXTRA_ARGS:-}
echo "SMOKE_EXIT=$?"
