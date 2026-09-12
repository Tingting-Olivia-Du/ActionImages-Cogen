#!/bin/bash
# Queued behind the main eval on the SAME GPU: arm4 again, but executing its chunk at the
# native 20 Hz instead of at its training stride.
#
# Why this is the fair control rather than matching replan counts. arm4 (frame_interval=3)
# emits waypoints 24.5 mm apart; arm0 (frame_interval=1) emits them 8.1 mm apart. Straight-line
# motion between the coarser waypoints cuts corners by up to 4.4 mm on real trajectories, and
# that alone drops the GROUND-TRUTH replay ceiling from 100% to 50% -- before any model is
# involved. Forcing equal replan counts would instead handicap arm0, whose 41 frames only cover
# 2.0 s against arm4's 6.0 s: it would be asked to finish a 4.65 s task with 43% of the horizon.
#
# --interpolate resamples the SAME predictions to 20 Hz (SLERP for rotation, held gripper), so
# execution granularity matches arm0 while the replan count -- arm4's genuine advantage -- is
# preserved and still reported.
set -uo pipefail
GPU="${1:?usage: queue_interp.sh <gpu>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
echo "[queue] waiting for the running eval to finish ..."
while pgrep -f "run_full_eval.sh|run_one_ckpt.sh|eval/rollout.py" > /dev/null; do sleep 60; done
echo "[queue] main eval done, starting arm4 --interpolate on GPU $GPU"
TRIALS=4 FI=3 RES=256 CFG=7.5 TAGSTYLE=explicit SOLVER=sphere \
  TAG_SUFFIX="_interp" EXTRA="--interpolate" \
  bash scripts/run_one_ckpt.sh "$GPU" TingtingDu/arm4__seed42_fi3 2500 5000 7500 10000
echo "QUEUE_INTERP_DONE"
