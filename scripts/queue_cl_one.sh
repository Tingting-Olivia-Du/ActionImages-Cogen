#!/bin/bash
# One closed-loop campaign, leased and retried. Split out of the serial gaps2 queue so the
# three Table-2 campaigns (280 rollouts, ~28 GPU-hours serial) run on separate cards.
#   bash scripts/queue_cl_one.sh <name> <ckpt> <variation> <proto> <tasks...>
# proto: "ours" (cfg 7.5, fi 3, explicit) or "init" (cfg 10, fi 4, none -- the released
# checkpoint's own protocol; it never saw our prompt tags).
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
LEASE="bash $REPO/scripts/gpu_lease.sh"
source /opt/conda/etc/profile.d/conda.sh
source /workspace/ttdu/ttd/scripts/env_eval.rc
conda activate ttd_rollout
export PYTHONPATH="$REPO"
NAME=$1; CKPT=$2; VAR=$3; PROTO=$4; shift 4; TASKS="$*"
case "$PROTO" in
  init) P="--cfg 10.0 --frame-interval 4 --prompt-tag-style none" ;;
  *)    P="--cfg 7.5 --frame-interval 3 --prompt-tag-style explicit" ;;
esac
OUT="$REPO/reports/closedloop/rollout_${NAME}.json"
for a in 1 2 3; do
  [ -s "$OUT" ] && { echo "[cl:$NAME] already done"; break; }
  G=$($LEASE acquire 28000)
  echo "[cl:$NAME] $(date -u +%H:%M) START attempt $a on GPU $G"
  CUDA_VISIBLE_DEVICES=$G xvfb-run -a python -u eval/rollout.py --ckpt "$CKPT" \
    --res 512 --steps 50 --axis-solver sphere --arm-action-mode planning \
    --max-steps-factor 1.5 --skip-anchor-frames 4 --num-trials 10 \
    --variation "$VAR" $P --tag "$NAME" --tasks $TASKS >> "$REPO/logs_cl_${NAME}.txt" 2>&1
  rc=$?; $LEASE release "$G"
  [ -s "$OUT" ] && { echo "[cl:$NAME] OK (rc=$rc)"; break; }
  echo "[cl:$NAME] no output (rc=$rc) -- retry"; sleep 60
done
echo "[cl:$NAME] DONE $(date -u +%H:%M)"
