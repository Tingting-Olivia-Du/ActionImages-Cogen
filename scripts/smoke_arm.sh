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
#   GPUS=4,7 SEG_MODE=scene_roles ACTION_MASK_MIX=A1 bash scripts/smoke_arm.sh \
#     "video+action@0.4,depth+action@0.2,segmentation+action@0.2,normal+action@0.2"   # arm7
#
# DATASET/RES: the default was `rlbench_selfgen@1.0` at 256 until 2026-09-04. That symlink has
# been DANGLING since the 256 v2 tree was deleted to free disk, so the script could not run at
# all -- it defaults to the live 512_aug tree now, with RES derived from the tree exactly as
# train_arm.sh does it. A smoke on a different tree/resolution than the arm proves less than it
# looks like it does, so the knobs below default to what train_arm.sh uses.
#
# Use the SAME GPU count and deepspeed config the real arm will use. TWO GPUs minimum:
# measured 2026-08-10, a single-GPU smoke loads and starts fine (~29GB resident) and then OOMs
# in the FIRST optimizer step --
#     stage_1_and_2.py:1891  fp32_partition.to(device)   tried to allocate 23.89 GiB
# because ZeRO-2 with world_size=1 has nothing to partition across, so the whole fp32 master
# copy has to land on the one device during the step. With 2 ranks each partition is half that
# and it fits. The failure is an artefact of the GPU COUNT, not of the template or the code, so
# do not read a single-GPU OOM as a problem with the arm under test.
#
# --strict_getitem True + --dataloader_num_workers 0 deliberately: a broken data path should
# surface as a traceback in the main process, not as retry spam that quietly serves a
# different sample.
#
# EXPECTED PEAK VRAM: every FOUR-segment template assembles identically-shaped segments, so
# their peaks should agree to within noise -- a large gap means something changed shape. That
# covers arm7's whole menu (video/depth/segmentation/normal + action) as well as the video+X
# perception templates. video+depth+action is six segments: ~1.5x the sequence and ~2.25x the
# attention, so it is expected to be visibly higher.
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
# Mirror train_arm.sh so a smoke exercises the arm's ACTUAL configuration, not a 256-resolution
# approximation of it. Resolution follows the tree for the same reason it does there: training
# at anything other than the rendered size is silently wasteful in one direction and lossy in
# the other.
DATASET="${DATASET:-rlbench_selfgen_512_aug}"
case "$DATASET" in
  rlbench_selfgen_512*) RES="${RES:-512}" ;;
  *)                    RES="${RES:-256}" ;;
esac
FRAME_INTERVAL="${FRAME_INTERVAL:-3}"
SEG_MODE="${SEG_MODE:-referring}"
# The two mask axes partition the menu; see MODE_MIX_MASKS.md 9.5. Defaults match train_arm.sh's
# library-level defaults, so pass the arm's own values when smoking a specific arm.
PERCEPTION_MASK_MIX="${PERCEPTION_MASK_MIX:-M2}"
ACTION_MASK_MIX="${ACTION_MASK_MIX:-A0}"
SLUG="$(echo "$TEMPLATE" | tr -c '[:alnum:]+' '_')"
OUT="${OUT:-$REPO/outputs/smoke_${SLUG}}"
# Default: warm-start from the official checkpoint, matching train_arm.sh's stage-1 default.
# Pass INIT_CKPT= (empty) to smoke the from-Wan-base path instead.
INIT_CKPT="${INIT_CKPT-/workspace/ttdu/starVLA/playground/Pretrained_models/anyeZHY/ActionImages/step125750.ckpt}"

INIT_ARG=""
[ -n "$INIT_CKPT" ] && INIT_ARG="--init_ckpt_path $INIT_CKPT"
echo "init: ${INIT_CKPT:-<Wan base, no warm-start>}"
echo "template='$TEMPLATE' dataset=$DATASET res=$RES fi=$FRAME_INTERVAL seg_mode=$SEG_MODE"
echo "mask axes: perception_mask_mix=$PERCEPTION_MASK_MIX action_mask_mix=$ACTION_MASK_MIX"

rm -rf "$OUT"   # find_latest_checkpoint would otherwise resume a previous smoke run
mkdir -p "$OUT"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader -i "$GPUS"

CUDA_VISIBLE_DEVICES=$GPUS torchrun --nnodes=1 --nproc_per_node=$NPROC --master_port $PORT \
  train.py \
  --deepspeed "$DS_CONFIG" \
  --dataset_path ./data \
  --dataset_name "${DATASET}@1.0" \
  --template_mix "$TEMPLATE" \
  --segmentation_mode "$SEG_MODE" \
  --perception_mask_mix "$PERCEPTION_MASK_MIX" \
  --action_mask_mix "$ACTION_MASK_MIX" \
  --variations "$VARIATIONS" \
  $INIT_ARG \
  --output_dir "$OUT" \
  --height "$RES" --width "$RES" --num_frames 41 --frame_interval "$FRAME_INTERVAL" \
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
