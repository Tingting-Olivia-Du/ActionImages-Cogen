#!/bin/bash
# Sequential closed-loop eval over every checkpoint, on ONE GPU, one checkpoint at a time.
#
#   bash scripts/run_full_eval.sh <gpu>
#
# One at a time is deliberate on both axes:
#   * DISK  -- a 12.8 GB checkpoint is downloaded, evaluated, then deleted before the next is
#              fetched. Three separate runs have already been lost to NoSpaceLeftError midway
#              (truncated JSON at exactly 16384 / 122880 bytes), on a volume other tenants fill
#              without warning.
#   * GPU   -- a single device, so the rest of the machine stays free.
#
# Every group runs with --skip-anchor-frames 4, and the dropped slots are refilled with a ramp
# from the arm's current pose.
#
# The chunk's leading frames reconstruct the pose the robot is ALREADY in -- the conditioning
# anchor -- rather than predicting where to go. At the near-vertical home orientation that
# encode -> VAE -> decode round trip flips the approach axis: measured +0.993 world-z (pointing
# up) against ground truth -0.971 (pointing down), a 174 deg error that IK then executes
# faithfully. That is the "gripper points at the sky and wiggles" at the start of every rollout.
#
# 4 is measured, not the 4x VAE compression ratio. Perturbing the anchor latent moves pixel
# frames 0/1/2/3 by 59/74/100/63% of its peak and frame 4 by only 19% -- the decoder's temporal
# receptive field is wide and its peak is at frame 2, not frame 0. An earlier run used 2 and
# still showed residual flipping, which is exactly what frames 2-3 being anchor-dominated
# predicts.
set -uo pipefail
GPU="${1:?usage: run_full_eval.sh <gpu>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

run_group() {  # repo  fi  res  cfg  tagstyle  steps...
  local hf=$1 fi=$2 res=$3 cfg=$4 style=$5; shift 5
  TRIALS=4 FI=$fi RES=$res CFG=$cfg TAGSTYLE=$style SOLVER=sphere \
    bash scripts/run_one_ckpt.sh "$GPU" "$hf" "$@"
}

echo "===== arm4 (fi=3, 256, explicit) ====="
run_group TingtingDu/arm4__seed42_fi3 3 256 7.5 explicit 2500 5000 7500 10000
echo "===== arm0 (fi=1, 256, explicit) ====="
run_group TingtingDu/arm0__seed42     1 256 7.5 explicit 750 1250 1750 2500
echo "===== official (fi=4, 512, none) ====="
source /opt/conda/etc/profile.d/conda.sh
source /workspace/ttdu/ttd/scripts/env_eval.rc
export PYTHONPATH="$REPO_ROOT"
CUDA_VISIBLE_DEVICES=$GPU xvfb-run -a conda run -n ttd_rollout python -u eval/rollout.py \
  --ckpt /workspace/ttdu/starVLA/playground/Pretrained_models/anyeZHY/ActionImages/step125750.ckpt \
  --tag official_125750 --res 512 --cfg 10.0 --frame-interval 4 --prompt-tag-style none \
  --axis-solver sphere --arm-action-mode planning --max-steps-factor 1.5 \
  --skip-anchor-frames 4 --record-video \
  --tasks close_drawer close_box open_drawer push_buttons meat_off_grill --num-trials 4
echo "FULL_EVAL_DONE"
