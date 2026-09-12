#!/bin/bash
# GT-replay ceiling on variation 1. REQUIRED CONTROL before reading the variation-1 collapse
# (arm6 35% -> 1.7%, official 24% -> 0%) as a model failure: if replaying the demo's OWN actions
# also fails on variation 1, the scenes are not solvable by this harness and the number says
# nothing about either policy.
#
# ⚠️ MODE must match the policy runs (planning). rollout.py defaults to `ik`, and an ik ceiling
# does not bound a planning rollout -- the first version of this measurement had that mismatch.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
source /opt/conda/etc/profile.d/conda.sh
source /workspace/ttdu/ttd/scripts/env_eval.rc
conda activate ttd_rollout
export PYTHONPATH="$REPO"
GPU="${GPU:?set GPU}"; VAR="${VAR:-1}"
echo "[gt] $(date) GT replay, variation $VAR, GPU $GPU"
CUDA_VISIBLE_DEVICES="$GPU" xvfb-run -a python -u eval/rollout.py \
  --ckpt /workspace/ttdu/starVLA/playground/Pretrained_models/anyeZHY/ActionImages/step125750.ckpt \
  --tag "gt_replay_var${VAR}_${MODE:-planning}" --res 512 --gt-replay \
  --arm-action-mode "${MODE:-planning}" --max-steps-factor 1.5 \
  --variation "$VAR" --tasks push_buttons meat_off_grill open_drawer --num-trials 10
echo "[gt] $(date) exited $?"
