#!/bin/bash
# Closed-loop on the four ADDITIONAL never-trained tasks, paired between arm6 and the released
# checkpoint. tab:generalization's tier-3 closed-loop cell currently rests on two tasks
# (close_box, close_drawer) and carries a surprising claim -- that unseen TASKS are easier than
# unseen VARIATIONS of trained tasks. Two tasks is not enough to hang that on, and these four are
# already rendered and confirmed present in RLBench.
#
# Each model keeps its own training-time protocol, same as every other campaign here.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
source /opt/conda/etc/profile.d/conda.sh
source /workspace/ttdu/ttd/scripts/env_eval.rc
conda activate ttd_rollout
export PYTHONPATH="$REPO"
GPU="${GPU:?set GPU}"; MODEL="${MODEL:?arm6|official}"
if [ "$MODEL" = "arm6" ]; then
  CKPT="$REPO/outputs/.grid_pin/step10000.ckpt"; TAG="arm6_10000_unseen4"; CFG=7.5; FI=3; TS=explicit
else
  CKPT="/workspace/ttdu/starVLA/playground/Pretrained_models/anyeZHY/ActionImages/step125750.ckpt"
  TAG="official_125750_unseen4"; CFG=10.0; FI=4; TS=none
fi
echo "[cl-unseen4] $(date) $MODEL on GPU $GPU (cfg=$CFG fi=$FI tag=$TS)"
CUDA_VISIBLE_DEVICES="$GPU" xvfb-run -a python -u eval/rollout.py \
  --ckpt "$CKPT" --tag "$TAG" --res 512 --cfg "$CFG" --steps 50 \
  --frame-interval "$FI" --prompt-tag-style "$TS" --axis-solver sphere \
  --arm-action-mode planning --max-steps-factor 1.5 --record-video --variation 0 \
  --tasks close_microwave take_lid_off_saucepan toilet_seat_down basketball_in_hoop \
  --num-trials 10
echo "[cl-unseen4] $(date) $MODEL exited $? -> reports/closedloop/rollout_${TAG}.json"
