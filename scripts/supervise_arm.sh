#!/bin/bash
# Keep one arm alive on a machine that evicts jobs to hand its GPUs to other tenants.
#
#   ARM=arm6 CKPT_EVERY=500 nohup bash scripts/supervise_arm.sh > logs_arm6_supervisor.txt 2>&1 &
#
# WHY THIS EXISTS. arm6 was SIGKILLed at step 891 of 10000 after 5h13m -- loss healthy, no CUDA
# OOM, no NaN, `exitcode -9`. The two GPUs it held (3, 4) were occupied seconds later by a
# different PID at 47.9 GB each. Read: an outside tenant took the cards and the job was killed to
# make room. Nothing in our code failed, and nothing in our code can prevent it.
#
# What our code CAN do is stop that costing a full run:
#
#   * a smaller CKPT_EVERY, so an eviction costs hours instead of everything. At 2000 the first
#     checkpoint lands ~12 h in, and a 5 h eviction loses all of it -- which is exactly what
#     happened. 500 caps the loss at ~3 h.
#   * automatic RESUME. train_arm.sh already resumes from a non-empty output_dir
#     (find_latest_checkpoint, train.py:490); the previous waiter deliberately REFUSED to launch
#     into a non-empty directory to prevent a silent resume of a different experiment. Here the
#     resume is the point, so the refusal is replaced by an explicit, logged decision.
#   * re-waiting after every death, instead of exiting.
#
# MEMORY.USED, NOT UTILIZATION: a tenant holding 45 GB at 0% util still owns the card.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

ARM="${ARM:-arm6}"
FREE_MB="${FREE_MB:-1000}"
POLL_S="${POLL_S:-120}"
MAX_RESTARTS="${MAX_RESTARTS:-40}"
export CKPT_EVERY="${CKPT_EVERY:-500}"
# Rotation, not keep-everything. A 500-step cadence over 10000 steps is 20 checkpoints; at
# ~120 GB for the resume tip plus ~12 GB per older stepN.ckpt that is ~348 GB on a shared volume
# other tenants fill without warning. TOP_K=5 holds it at ~168 GB.
#
# The COST is real, and is why train_arm.sh defaults to -1: rotation deletes older stepN.ckpt
# files, and those are what an eval-versus-step curve is made of. With 5 kept at a 500 cadence,
# only the last 2500 steps of history survive to the end of the run. If that curve matters, copy
# stepN.ckpt out (12 GB each) as it lands, or evaluate it before it rotates.
export SAVE_TOP_K="${SAVE_TOP_K:-5}"
OUT="$REPO/outputs/${ARM}__seed42_fi3_512_aug_sr"
LOG="$REPO/logs_${ARM}.txt"

echo "[sup] arm=$ARM  CKPT_EVERY=$CKPT_EVERY  SAVE_TOP_K=$SAVE_TOP_K  free<${FREE_MB}MiB  poll=${POLL_S}s  max_restarts=$MAX_RESTARTS"

for attempt in $(seq 1 "$MAX_RESTARTS"); do
  # ---- wait for two genuinely free cards ----
  while true; do
    mapfile -t F < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                     | awk -F', *' -v t="$FREE_MB" '$2 < t {print $1}')
    if [ "${#F[@]}" -ge 2 ]; then
      sleep 10   # a gap that closes in 10s was never really free
      mapfile -t F2 < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                        | awk -F', *' -v t="$FREE_MB" '$2 < t {print $1}')
      [ "${#F2[@]}" -ge 2 ] && { G="${F2[0]},${F2[1]}"; break; }
      echo "[sup] $(date +%T) gap closed before launch, still waiting"
    fi
    sleep "$POLL_S"
  done

  # ---- fresh or resume? say which, out loud ----
  LAST=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  if [ -n "$LAST" ]; then
    echo "[sup] $(date +%T) attempt $attempt: RESUMING from $(basename "$LAST") on GPUS=$G"
  else
    echo "[sup] $(date +%T) attempt $attempt: fresh warm-start on GPUS=$G"
  fi
  df -BG --output=avail /workspace | tail -1 | xargs echo "[sup] /workspace avail:"

  GPUS="$G" bash scripts/train_arm.sh "$ARM" >> "$LOG" 2>&1
  RC=$?
  STEP=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 | sed 's/.*checkpoint-//')
  echo "[sup] $(date +%T) train_arm.sh exited $RC (last checkpoint: ${STEP:-none})"

  if [ "$RC" -eq 0 ]; then
    echo "[sup] finished cleanly"; exit 0
  fi
  # -9/137 = SIGKILL, the eviction signature. Anything else is more likely our bug: still retry,
  # but the exit code is in the log so a repeated non-eviction failure is visible as such.
  echo "[sup] will re-wait for GPUs and resume (exit $RC)"
  sleep 30
done
echo "[sup] gave up after $MAX_RESTARTS attempts"; exit 1
