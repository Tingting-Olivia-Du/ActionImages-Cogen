#!/bin/bash
# Launch arm6 the moment two GPUs are actually free, and not before.
#
#   nohup bash scripts/wait_and_launch_arm6.sh > logs_arm6_waiter.txt 2>&1 &
#
# WHY A WATCHER. /workspace's GPUs are shared with outside tenants who take a card without
# warning: 7 of 8 were claimed between one `nvidia-smi` and the `torchrun` a few seconds later,
# and the run then OOMed at DeepSpeed's optimizer step. So the check has to be immediately
# before the launch, and it has to be repeated.
#
# MEMORY.USED, NOT UTILIZATION. A tenant that holds 45 GB at 0% util still owns the card. This
# polls memory.used and requires two cards under FREE_MB.
#
# SINGLE GPU IS NOT AN OPTION at 512. ZeRO-2 shards the fp32 master weights + Adam moments
# across ranks; on one rank that partition is the whole 5B x 4 bytes = ~20 GB, and
# stage_1_and_2.py:1901 copies it back to the GPU every optimizer step. Measured: 26.08 GiB
# already resident, tried to allocate 23.90 GiB, on a 47.40 GiB card. Gradient accumulation does
# not help -- it changes how many forwards precede a step, not how the optimizer is partitioned.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

FREE_MB="${FREE_MB:-1000}"      # a card with less than this in use counts as free
POLL_S="${POLL_S:-120}"
ARM="${ARM:-arm6}"
OUT_DIR="$REPO/outputs/${ARM}__seed42_fi3_512_aug_sr"

echo "[waiter] watching for 2 GPUs with < ${FREE_MB} MiB used, polling every ${POLL_S}s"
while true; do
  mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                      | awk -F', *' -v t="$FREE_MB" '$2 < t {print $1}')
  if [ "${#FREE[@]}" -ge 2 ]; then
    G="${FREE[0]},${FREE[1]}"
    echo "[waiter] $(date +%T) GPUs ${G} are free -- re-checking in 10s to skip a transient gap"
    sleep 10
    mapfile -t FREE2 < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                         | awk -F', *' -v t="$FREE_MB" '$2 < t {print $1}')
    if [ "${#FREE2[@]}" -ge 2 ]; then
      G="${FREE2[0]},${FREE2[1]}"
      # A non-empty output_dir would make train.py RESUME instead of warm-starting (train.py:490).
      if [ -d "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
        echo "[waiter] !! $OUT_DIR is not empty -- refusing to launch, it would resume, not warm-start"
        exit 3
      fi
      echo "[waiter] $(date +%T) launching ${ARM} on GPUS=${G}"
      GPUS="$G" bash scripts/train_arm.sh "$ARM" > "$REPO/logs_${ARM}.txt" 2>&1
      echo "[waiter] train_arm.sh exited $? -- see logs_${ARM}.txt"
      exit 0
    fi
    echo "[waiter] $(date +%T) the gap closed before launch; still waiting"
  fi
  sleep "$POLL_S"
done
