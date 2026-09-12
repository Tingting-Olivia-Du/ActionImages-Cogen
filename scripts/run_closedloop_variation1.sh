#!/bin/bash
# Closed-loop on variation 1 -- the "same task, unseen configuration" level that tab:closed_loop
# is currently missing (EXPERIMENT_STATUS_zh_aug29.md sec.10.5).
#
# Only the three tasks that are IN the training tree are meaningful here: close_box and
# close_drawer are absent from the tree entirely, so for them every variation is equally unseen
# and variation 1 adds nothing over variation 0. push_buttons / meat_off_grill / open_drawer all
# have a variation 1 in the tree, so variation 1 is a configuration of a TRAINED task that the
# model never saw -- exactly the missing middle tier between "trained config" and "never-trained
# task".
#
#   MODEL=arm6|official GPU=7 bash scripts/run_closedloop_variation1.sh
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
source /opt/conda/etc/profile.d/conda.sh
source /workspace/ttdu/ttd/scripts/env_eval.rc
conda activate ttd_rollout
export PYTHONPATH="$REPO"
GPU="${GPU:?set GPU}"; MODEL="${MODEL:?set MODEL=arm6|official}"

# Each model keeps its OWN training-time protocol; forcing one into the other's conditioning
# measures distribution shift, not control (same reasoning as run_official_closedloop_postfix.sh).
if [ "$MODEL" = "arm6" ]; then
  CKPT="$REPO/outputs/.grid_pin/step10000.ckpt"
  TAG="arm6_10000_var1"; CFG=7.5; FI=3; TAGSTYLE=explicit
else
  CKPT="/workspace/ttdu/starVLA/playground/Pretrained_models/anyeZHY/ActionImages/step125750.ckpt"
  TAG="official_125750_var1"; CFG=10.0; FI=4; TAGSTYLE=none
fi

echo "[cl-v1] $(date) $MODEL variation1 on GPU $GPU (cfg=$CFG fi=$FI tag=$TAGSTYLE)"
CUDA_VISIBLE_DEVICES="$GPU" xvfb-run -a python -u eval/rollout.py \
  --ckpt "$CKPT" --tag "$TAG" --res 512 --cfg "$CFG" --steps 50 \
  --frame-interval "$FI" --prompt-tag-style "$TAGSTYLE" --axis-solver sphere \
  --arm-action-mode planning --max-steps-factor 1.5 --record-video \
  --variation 1 --tasks push_buttons meat_off_grill open_drawer --num-trials 20
echo "[cl-v1] $(date) $MODEL exited $? -> reports/closedloop/rollout_${TAG}.json"
