#!/bin/bash
# Wait for a GPU with enough free VRAM, then IMMEDIATELY start the closed-loop rollout.
# Exists because the window between "a card frees" and "we use it" is when another tenant takes
# it -- which is exactly what happened after fusion0 was stopped.
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; export PYTHONPATH="$REPO"
CKPT="${CKPT:?}"; TAG="${TAG:?}"; NEED=${NEED:-30000}
TASKS="${TASKS:-push_buttons open_drawer meat_off_grill}"; TRIALS=${TRIALS:-10}
while true; do
  g=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
      | awk -F', ' -v n="$NEED" '($3-$2)>n {print $1; exit}')
  if [ -n "$g" ]; then
    echo "$(date +%H:%M:%S) claiming GPU $g"
    # Protocol copied verbatim from scripts/queue_fusion_closedloop.sh -- the arm trained at
    # 512/fi=3, and rollout.py defaults to 256/fi=1, so these are not optional.
    CUDA_VISIBLE_DEVICES=$g xvfb-run -a python -u eval/rollout.py \
      --ckpt "$CKPT" --tag "$TAG" --anchor-modality fusion --tasks $TASKS \
      --variation 0 --num-trials "$TRIALS" \
      --res 512 --cfg 7.5 --steps 50 --frame-interval 3 \
      --prompt-tag-style explicit --axis-solver sphere --arm-action-mode planning \
      --max-steps-factor 1.5 --skip-anchor-frames 4 --execution-horizon 41 --seed 42 \
      >> "$REPO/logs_cl_$TAG.txt" 2>&1
    rc=$?
    if [ -s "$REPO/reports/closedloop/rollout_$TAG.json" ]; then echo "ROLLOUT_DONE rc=$rc"; exit 0; fi
    # Report the REAL error. The queue script's own comment records why: a previous version
    # guessed "probably OOM" and sent the operator chasing GPU contention for 8 attempts while
    # the actual cause was a ValueError in policy.
    echo "ROLLOUT_FAILED rc=$rc: $(grep -oE '^[A-Za-z_.]*(Error|Exception):.*' "$REPO/logs_cl_$TAG.txt" | tail -1 | cut -c1-140)"
    exit 1
  fi
  sleep 60
done
