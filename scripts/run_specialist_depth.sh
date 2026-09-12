#!/bin/bash
# Depth specialist baseline (video+depth@1.0), warm-started fresh from the official step125750
# checkpoint like every other arm -- for Table 1's "cost of sharing" comparison in
# paper/U-CoGen_experiment_plan.md. STEPS=4000 because the draft's own arm6 numbers already show
# depth AbsRel plateaus by then (0.087@1500 -> 0.086@3500 -> 0.085@4000); CKPT_EVERY=500 gives a
# step~2000 checkpoint (matched depth-update count vs arm6's own 0.2x10000=2000 depth updates,
# "Panel B") alongside the step4000 ceiling reading ("Panel A").
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"
cd "$REPO"

echo "[specialist-depth] $(date) launching video+depth@1.0 specialist"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

GPUS=3,4 \
SEED=42 \
STEPS=4000 \
CKPT_EVERY=500 \
SAVE_TOP_K=-1 \
OUT="$REPO/outputs/specialist_depth__seed42_fi3_512_aug_sr" \
EXTRA_ARGS="--run_name specialist-depth-seed42" \
bash scripts/train_arm.sh "video+depth@1.0"

echo "[specialist-depth] $(date) train_arm.sh exited with $?"
