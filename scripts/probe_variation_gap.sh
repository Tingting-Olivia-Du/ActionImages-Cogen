#!/bin/bash
# Why is tier-2 (unseen variation) 1.7% when tier-1 is 35%?
#
# Two candidate explanations that the existing campaign cannot separate:
#   (a) the model genuinely fails when the instruction's referent changes;
#   (b) the harness's "abort after 5 consecutive unreachable commands" rule, which is a
#       hardcoded policy and not a model property, amplifies a small geometric degradation.
#       It killed 16/20 push_buttons rollouts on variation 1 against 5/20 on variation 0.
#
# A and B raise that threshold to 50 (effectively off) on BOTH tiers, paired by scene_seed
# against the existing streak=5 runs. C answers "try more variations" directly: variation 2
# exists for push_buttons (3 buttons) and open_drawer (top drawer).
#
# Everything else is byte-identical to the original campaign: cfg 7.5, fi 3, explicit tags,
# planning arm mode, sphere solver, max_steps_factor 1.5, skip_anchor_frames 4, 20 trials.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
source /opt/conda/etc/profile.d/conda.sh
source /workspace/ttdu/ttd/scripts/env_eval.rc
conda activate ttd_rollout
export PYTHONPATH="$REPO"
CKPT=outputs/.grid_pin/step10000.ckpt
COMMON="--ckpt $CKPT --res 512 --cfg 7.5 --steps 50 --frame-interval 3 \
  --prompt-tag-style explicit --axis-solver sphere --arm-action-mode planning \
  --max-steps-factor 1.5 --skip-anchor-frames 4 --num-trials 20"
G=$1; WHICH=$2
case "$WHICH" in
  A) CUDA_VISIBLE_DEVICES=$G xvfb-run -a python -u eval/rollout.py $COMMON \
       --variation 1 --max-ik-fail-streak 50 --tag arm6_var1_ikstreak50 \
       --tasks push_buttons meat_off_grill open_drawer ;;
  B) CUDA_VISIBLE_DEVICES=$G xvfb-run -a python -u eval/rollout.py $COMMON \
       --variation 0 --max-ik-fail-streak 50 --tag arm6_var0_ikstreak50 \
       --tasks push_buttons meat_off_grill open_drawer ;;
  C) CUDA_VISIBLE_DEVICES=$G xvfb-run -a python -u eval/rollout.py $COMMON \
       --variation 2 --tag arm6_var2 \
       --tasks push_buttons open_drawer ;;
esac
echo "[probe_var] $WHICH exited $?"
