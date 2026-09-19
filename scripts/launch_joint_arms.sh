#!/bin/bash
# ICLR joint-source trainings: arm7-joint (GPUs 0-3) + arm0-joint (GPUs 4-7) in parallel.
# Mix rlbench_selfgen_512_aug_wide@0.4, maniskill3@0.2, behavior@0.4; 6k steps, ckpt every
# 1k (so both the 4k precedent budget and the 6k point come from one run); official
# step125750 warm start; everything else identical across the two arms so the contrast is
# purely the template menu. LAUNCH ONLY when data/behavior is complete (or run the
# dual-source fallback by exporting JOINT_MIX without behavior).
set -uo pipefail
REPO=/workspace/1228_tingting/ActionImages-Cogen
# Dual-source (user dropped BEHAVIOR from training on 2026-09-17 quality grounds).
# 0.7/0.3 equalizes per-episode exposure: 24k samples -> rlbench 4.2 draws/ep (4000 eps),
# maniskill3 4.1 draws/ep (1750 eps); also matches the approved 2:1 relative weighting.
JOINT_MIX="${JOINT_MIX:-rlbench_selfgen_512_aug_wide@0.7,maniskill3@0.3}"
INIT="${INIT:-$REPO/checkpoints/official/step125750.ckpt}"
STEPS="${STEPS:-6000}"
export TTD_ENV="${TTD_ENV:-/workspace/1228_tingting/.env}"   # WANDB_API lives here
LOGDIR=$REPO/outputs/joint_launch_logs
mkdir -p "$LOGDIR"

launch() {  # launch <arm> <gpus> <port>
  local arm=$1 gpus=$2 port=$3
  DATASET="$JOINT_MIX" GPUS="$gpus" PORT="$port" STEPS="$STEPS" CKPT_EVERY=1000 \
  SAVE_OPTIM="${SAVE_OPTIM:-False}" \
  INIT_CKPT="$INIT" OUT="$REPO/outputs/${arm}_joint2src_seed42_fi3_6k" \
    setsid nohup bash "$REPO/scripts/train_arm.sh" "$arm" \
    > "$LOGDIR/${arm}_joint.log" 2>&1 < /dev/null &
  echo "launched $arm on GPUs $gpus port $port pid $! -> $LOGDIR/${arm}_joint.log"
}

# setsid, NOT bare nohup: on 2026-09-18 03:51 both runs died at step ~1480 with
# `SignalException: got signal: 15`. nohup only ignores SIGHUP, so when the supervising
# shell's PROCESS GROUP was torn down the SIGTERM still reached torchrun and killed 480
# steps of progress. setsid puts each run in its own session/process group, so nothing
# upstream can signal it; `wait` is also gone below so this launcher returns immediately
# instead of being the parent that gets torn down.
launch arm7 0,1,2,3 29511
sleep 5
launch arm0 4,5,6,7 29512
sleep 3
echo "JOINT_ARMS_LAUNCHED (detached; follow $LOGDIR/*.log)"
