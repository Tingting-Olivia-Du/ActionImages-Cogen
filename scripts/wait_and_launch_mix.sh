#!/bin/bash
# Wait for two USABLE GPUs, run a 12-step smoke, and only then launch the real arm.
#
# "Usable" is not what nvidia-smi says. GPU 7 on this host has been observed reporting a
# normal memory figure while `torch.zeros(..., device="cuda")` fails with
# "No CUDA GPUs are available", so every candidate is probed with a real allocation before
# it is used. Free-ness is judged on memory.used, not utilization -- a co-tenant holding
# 45GB at 0% util will still OOM us.
set -uo pipefail
REPO=/workspace/ttdu/ActionImages-Cogen
cd "$REPO"
LOG_DIR="${LOG_DIR:-/tmp/claude-0/-workspace-ttdu/5bec7b5c-3b4f-493f-b843-acde2cfc0aab/scratchpad}"
ARM="${ARM:-A2}"
POLL="${POLL:-300}"
NEED_FREE_MB="${NEED_FREE_MB:-46000}"   # peak was 44GB at 512^2; leave a little headroom
PY=/opt/conda/envs/ttd_train/bin/python

probe_gpu() {  # $1 = index; 0 if a real allocation succeeds
  CUDA_VISIBLE_DEVICES="$1" timeout 120 "$PY" -c \
    "import torch; torch.zeros(256*1024*1024//4, device='cuda'); torch.cuda.synchronize()" \
    >/dev/null 2>&1
}

echo "[wait] polling every ${POLL}s for two usable GPUs with >= ${NEED_FREE_MB}MB free"
while true; do
  CAND=()
  while IFS=', ' read -r idx used total; do
    free=$(( total - used ))
    [ "$free" -ge "$NEED_FREE_MB" ] || continue
    if probe_gpu "$idx"; then CAND+=("$idx"); else echo "[wait] GPU $idx looks free but failed the allocation probe -- skipping"; fi
    [ "${#CAND[@]}" -ge 2 ] && break
  done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)

  if [ "${#CAND[@]}" -ge 2 ]; then
    GPUS="${CAND[0]},${CAND[1]}"
    echo "[wait] $(date '+%F %T') usable pair found: $GPUS"
    break
  fi
  sleep "$POLL"
done

AVAIL_G=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
echo "[wait] disk avail ${AVAIL_G}G"
if [ "${AVAIL_G:-0}" -lt 300 ]; then
  echo "[wait] ABORT: under 300G free; a 10k-step run needs ~4x13G step ckpts + ~220G peak optimizer state."
  exit 3
fi

SMOKE_OUT="$LOG_DIR/smoke_${ARM}_gate"
rm -rf "$SMOKE_OUT"
echo "[smoke] 12 steps on GPUs $GPUS"
GPUS="$GPUS" STEPS=12 CKPT_EVERY=1000000 OUT="$SMOKE_OUT" WANDB_MODE=offline \
  bash scripts/train_mix.sh "$ARM" > "$LOG_DIR/smoke_gate_${ARM}.log" 2>&1
SMOKE_RC=$?
STEPS_DONE=$(grep -oE "12/12 \[" "$LOG_DIR/smoke_gate_${ARM}.log" | wc -l)
rm -rf "$SMOKE_OUT"        # the Trainer writes a full checkpoint at max_steps (~132G)
if [ "$SMOKE_RC" -ne 0 ] || [ "$STEPS_DONE" -eq 0 ]; then
  echo "[smoke] FAILED (rc=$SMOKE_RC, reached-12/12=$STEPS_DONE). NOT launching. See $LOG_DIR/smoke_gate_${ARM}.log"
  exit 1
fi
echo "[smoke] PASS"

echo "[launch] $(date '+%F %T') arm=$ARM gpus=$GPUS ckpt_every=3000"
GPUS="$GPUS" CKPT_EVERY=3000 bash scripts/train_mix.sh "$ARM" > "$LOG_DIR/run_${ARM}.log" 2>&1
echo "[launch] arm $ARM exited rc=$? -- see $LOG_DIR/run_${ARM}.log"
