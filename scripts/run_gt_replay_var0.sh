#!/bin/bash
# GT-replay ceilings on variation 0 for all five closed-loop tasks.
#
# WHY IT MATTERS FOR THE PAPER: open_drawer is 0% for BOTH models on variation 0, which is
# unreadable without knowing whether the harness can solve it at all. tab:closed_loop currently
# asserts a 100% ceiling for it from an older n=3 measurement; this replaces that with a proper
# per-task ceiling at n=10, measured under the current harness, for every task in the table.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
source /opt/conda/etc/profile.d/conda.sh
source /workspace/ttdu/ttd/scripts/env_eval.rc
conda activate ttd_rollout
export PYTHONPATH="$REPO"
GPU="${GPU:?set GPU}"
echo "[gt-var0] $(date) GPU $GPU"
CUDA_VISIBLE_DEVICES="$GPU" xvfb-run -a python -u eval/rollout.py \
  --ckpt /workspace/ttdu/starVLA/playground/Pretrained_models/anyeZHY/ActionImages/step125750.ckpt \
  --tag "gt_replay_var0_${MODE:-planning}" --res 512 --gt-replay --variation 0 \
  --arm-action-mode "${MODE:-planning}" --max-steps-factor 1.5 \
  --tasks close_box close_drawer push_buttons meat_off_grill open_drawer --num-trials 10
echo "[gt-var0] $(date) exited $?"
