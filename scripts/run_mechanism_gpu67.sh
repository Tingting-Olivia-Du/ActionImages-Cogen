#!/bin/bash
# Paired Arm 0 / Arm 7 causal-mechanism evaluation on physical GPUs 6 and 7.
# Every job is resumable at the condition level. Override EPISODES for a smoke test, e.g.
#   EPISODES=0 STEPS=10 bash scripts/run_mechanism_gpu67.sh
set -uo pipefail

REPO=/workspace/1228_tingting/ActionImages-Cogen
source /workspace/1228_tingting/ttd/scripts/env_eval_ttd_eval.rc
cd "$REPO"

EPISODES="${EPISODES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}"
STEPS="${STEPS:-50}"
TASK="${TASK:-close_microwave}"
TARGET="${TARGET:-microwave_door}"
CONDITIONS="${CONDITIONS:-clean,delete_target,control_target}"
SUFFIX="${SUFFIX:-s${STEPS}}"
OUT="$REPO/reports/mechanism_roles"
LOGS="$REPO/logs/mechanism_roles"
mkdir -p "$OUT" "$LOGS"

ARM0="$REPO/outputs/arm0_joint2src_seed42_fi3_6k/checkpoint-4000/step4000.ckpt"
ARM7="$REPO/outputs/arm7_joint2src_seed42_fi3_6k/checkpoint-4000/step4000.ckpt"

run_one() {
  local gpu=$1 arm=$2 ckpt=$3
  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/mechanism_causal_roles.py \
    --ckpt "$ckpt" --tag "${arm}_4k_${SUFFIX}" --task "$TASK" \
    --target-instances "$TARGET" --episodes "$EPISODES" --steps "$STEPS" \
    --conditions "$CONDITIONS" > "$LOGS/${arm}_4k_${SUFFIX}.log" 2>&1
}

run_one 6 arm0 "$ARM0" & pid0=$!
run_one 7 arm7 "$ARM7" & pid7=$!
wait "$pid0"; rc0=$?
wait "$pid7"; rc7=$?
if [ "$rc0" -ne 0 ] || [ "$rc7" -ne 0 ]; then
  echo "mechanism run failed: arm0=$rc0 arm7=$rc7 (see $LOGS)" >&2
  exit 1
fi

python scripts/compare_mechanism_roles.py \
  --arm0 "$OUT/arm0_4k_${SUFFIX}/${TASK}_causal_roles.json" \
  --arm7 "$OUT/arm7_4k_${SUFFIX}/${TASK}_causal_roles.json" \
  --out "$OUT/${TASK}_${SUFFIX}_comparison"
