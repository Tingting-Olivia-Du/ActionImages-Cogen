#!/bin/bash
# Auto-resume guardian for the joint arms.
#
# WHY: on 2026-09-18 03:51 both runs died at step ~1480 with `SignalException: got
# signal: 15` -- the supervising shell's process group was torn down and SIGTERM reached
# torchrun. `setsid` in launch_joint_arms.sh prevents that specific cause; this watchdog
# covers everything else (node OOM-killer, driver fault, transient disk-full, human kill).
#
# Resume is safe and idempotent: train_arm.sh's find_latest_checkpoint picks the newest
# step*.ckpt in OUT and passes it as --resume_ckpt_path, so a restart continues from the
# last saved checkpoint (CKPT_EVERY=1000). Never relaunches a run that FINISHED.
#
# Run detached:  setsid nohup bash scripts/train_watchdog.sh >> <log> 2>&1 < /dev/null &
set -uo pipefail
REPO=/workspace/1228_tingting/ActionImages-Cogen
LOGDIR=$REPO/outputs/joint_launch_logs
STEPS="${STEPS:-6000}"
INTERVAL="${INTERVAL:-180}"
MIN_FREE_GB="${MIN_FREE_GB:-40}"   # refuse to restart into a full volume (it would die again)

alive() {  # alive <arm-tag-in-OUT-path>
  pgrep -f "train\.py .*${1}_joint2src" >/dev/null 2>&1
}
finished() {  # finished <arm>: newest ckpt already at target steps, or clean exit logged
  local arm=$1 out=$REPO/outputs/${arm}_joint2src_seed42_fi3_6k
  [ -f "$out/checkpoint-${STEPS}/step${STEPS}.ckpt" ] && return 0
  grep -q "^TRAIN_EXIT=0" "$LOGDIR/${arm}_joint.log" 2>/dev/null && return 0
  return 1
}
last_step() {  # last_step <arm>
  local out=$REPO/outputs/${1}_joint2src_seed42_fi3_6k
  ls -d "$out"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1
}
gpus_for() { [ "$1" = arm7 ] && echo "0,1,2,3" || echo "4,5,6,7"; }
port_for() { [ "$1" = arm7 ] && echo 29511 || echo 29512; }

echo "[watchdog] start $(date '+%F %T') interval=${INTERVAL}s target=${STEPS} steps"
while true; do
  for arm in arm7 arm0; do
    if finished "$arm"; then continue; fi
    if alive "$arm"; then continue; fi

    free_gb=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
    if [ "${free_gb:-0}" -lt "$MIN_FREE_GB" ]; then
      echo "[watchdog] $(date '+%F %T') $arm DOWN but only ${free_gb}G free (<${MIN_FREE_GB}G) -- NOT restarting"
      continue
    fi
    # keep the dead run's log instead of letting the relaunch truncate it
    [ -f "$LOGDIR/${arm}_joint.log" ] && \
      cp -f "$LOGDIR/${arm}_joint.log" "$LOGDIR/${arm}_joint.$(date +%m%d_%H%M%S).crashed.log"
    echo "[watchdog] $(date '+%F %T') $arm DOWN at ckpt-$(last_step "$arm") -- resuming (${free_gb}G free)"
    DATASET="${JOINT_MIX:-rlbench_selfgen_512_aug_wide@0.7,maniskill3@0.3}" \
    GPUS="$(gpus_for "$arm")" PORT="$(port_for "$arm")" STEPS="$STEPS" CKPT_EVERY=1000 \
    SAVE_OPTIM=False TTD_ENV=/workspace/1228_tingting/.env \
    INIT_CKPT="$REPO/checkpoints/official/step125750.ckpt" \
    OUT="$REPO/outputs/${arm}_joint2src_seed42_fi3_6k" \
      setsid nohup bash "$REPO/scripts/train_arm.sh" "$arm" \
      > "$LOGDIR/${arm}_joint.log" 2>&1 < /dev/null &
    sleep 60   # let it claim its GPUs before the next check
  done
  sleep "$INTERVAL"
done
