#!/bin/bash
# Run a closed-loop rollout campaign on GPU $GPU as soon as $WAIT_PID releases it.
#
#   GPU=6 WAIT_PID=<backfill pid> nohup bash scripts/queue_closedloop.sh > logs_closedloop_queue.txt 2>&1 &
#
# n=20, not n=5. The previous campaign scored 0/5 and that number carries no information: the
# official checkpoint gets 3/20 = 15% on this same harness, and P(0 successes | p=0.15, n=5) is
# 0.44. Twenty rollouts is the smallest grid that can distinguish "worse than official" from
# "we did not look long enough".
#
# RES AND INTERVAL MUST MATCH TRAINING (512 / 3). The loader resizes and re-strides silently, so
# a mismatch does not error -- it just turns the result into a measurement of domain shift.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
GPU="${GPU:-6}"; TRIALS="${TRIALS:-4}"
TASKS="${TASKS:-push_buttons open_drawer meat_off_grill slide_block_to_target turn_tap}"
CKPT="${CKPT:-}"

if [ -n "${WAIT_PID:-}" ]; then
  echo "[cl] waiting for pid $WAIT_PID to release GPU $GPU"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[cl] $(date +%T) it exited; taking GPU $GPU"
  sleep 20
fi

if [ -z "$CKPT" ]; then
  CK_DIR=$(ls -d "$REPO"/outputs/arm6__seed42_fi3_512_aug_sr/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  STEP=$(basename "$CK_DIR" | sed 's/checkpoint-//'); CKPT="$CK_DIR/step${STEP}.ckpt"
else
  STEP=$(basename "$CKPT" | sed 's/step//;s/\.ckpt//')
fi
[ -f "$CKPT" ] || { echo "[cl] no checkpoint at $CKPT"; exit 2; }

FREE=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits \
       | sed -n "$((GPU+1))p" | awk -F', *' '{print $2-$1}')
echo "[cl] checkpoint step $STEP, GPU $GPU has ${FREE} MiB free, trials=$TRIALS"
[ "$FREE" -gt 30000 ] || { echo "[cl] not enough free VRAM on GPU $GPU"; exit 3; }

source /opt/conda/etc/profile.d/conda.sh
source /workspace/ttdu/ttd/scripts/env_eval.rc
conda activate ttd_rollout
export PYTHONPATH="$REPO"
CUDA_VISIBLE_DEVICES="$GPU" xvfb-run -a python -u eval/rollout.py \
  --ckpt "$CKPT" --tag "arm6_${STEP}" --res 512 --cfg 7.5 --steps 50 \
  --frame-interval 3 --prompt-tag-style explicit --axis-solver sphere \
  --arm-action-mode planning --max-steps-factor 1.5 --record-video \
  --tasks $TASKS --num-trials "$TRIALS"
echo "[cl] rollout exit=$? -> reports/closedloop/rollout_arm6_${STEP}.json"
