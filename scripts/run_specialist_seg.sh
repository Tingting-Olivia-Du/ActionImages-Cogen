#!/bin/bash
# Segmentation specialist baseline (video+segmentation@1.0), warm-started fresh from the official
# step125750 checkpoint -- for Table 1's "cost of sharing" comparison. SEG_MODE=scene_roles is
# REQUIRED here: without it train_arm.sh's fallback branch (ARM is a raw mix string, not a named
# arm) defaults SEG_MODE to "referring", which is a different supervision signal (single
# target-only mask) than arm6's scene_roles (dense multi-role mask) -- a referring specialist
# would not be comparable to arm6's seg numbers at all.
# STEPS=4000 / CKPT_EVERY=500: same reasoning as the depth specialist (scripts/run_specialist_depth.sh).
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"
cd "$REPO"

echo "[specialist-seg] $(date) launching video+segmentation@1.0 specialist (scene_roles)"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

GPUS=5,7 \
SEED=42 \
STEPS=4000 \
CKPT_EVERY=500 \
SAVE_TOP_K=-1 \
SEG_MODE=scene_roles \
OUT="$REPO/outputs/specialist_seg__seed42_fi3_512_aug_sr" \
EXTRA_ARGS="--run_name specialist-seg-seed42" \
bash scripts/train_arm.sh "video+segmentation@1.0"

echo "[specialist-seg] $(date) train_arm.sh exited with $?"
