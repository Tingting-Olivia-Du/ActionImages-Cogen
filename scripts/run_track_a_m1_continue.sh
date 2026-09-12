#!/bin/bash
# Track A: does the plan-mixture trade-off (sec:modegrid) move if you continue training with a
# BALANCED conditioning mix (M1 = 45/10/45) instead of the shipped M2 (90/5/5)?
#
# M1 is a preset that already exists in training/templates.py and was flagged in
# SEGMENTATION_SCENE_ROLES_PLAN.md as "still worth sweeping as a midpoint" but was never run --
# the team picked M2 as the shipped default and stopped. This runs that missing point.
#
# INIT (not resume): warm-starts fresh optimizer state from arm6's own final checkpoint
# (step10000.ckpt, M2-trained), into a NEW output dir, with everything else byte-identical to
# arm6 except perception_mask_mix. Never point this at arm6's own OUT dir -- that would hit the
# resume branch and silently keep training under M2, which defeats the whole point.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"
cd "$REPO"

echo "[track-a] $(date) launching M1-continuation from arm6 step10000"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

GPUS=5,7 \
SEED=42 \
STEPS=3000 \
CKPT_EVERY=500 \
SAVE_TOP_K=-1 \
INIT_CKPT="$REPO/outputs/.grid_pin/step10000.ckpt" \
PERCEPTION_MASK_MIX=M1 \
OUT="$REPO/outputs/arm6m1__seed42_fi3_512_aug_sr" \
EXTRA_ARGS="--run_name arm6m1-M1-continue-seed42" \
bash scripts/train_arm.sh arm6

echo "[track-a] $(date) train_arm.sh exited with $?"
