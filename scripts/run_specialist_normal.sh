#!/bin/bash
# Normal specialist (video+normal@1.0), 4000 steps -- same recipe/budget as the depth, seg and
# action specialists. Direct launch on explicitly-chosen GPUs (no waiter): the caller has already
# confirmed the cards are free. See scripts/run_specialist_depth.sh for the shared rationale.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
GPUS_USE="${GPUS_USE:-3,4}"
echo "[normal] $(date) launching video+normal@1.0 on GPUs ${GPUS_USE}"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
GPUS="$GPUS_USE" SEED=42 STEPS=4000 CKPT_EVERY=500 SAVE_TOP_K=-1 \
OUT="$REPO/outputs/specialist_normal__seed42_fi3_512_aug_sr" \
EXTRA_ARGS="--run_name specialist-normal-seed42" \
bash scripts/train_arm.sh "video+normal@1.0"
echo "[normal] $(date) train_arm.sh exited $?"
