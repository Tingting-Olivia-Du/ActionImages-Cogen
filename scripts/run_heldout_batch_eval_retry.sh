#!/bin/bash
# Wrapper around heldout_batch_eval.py that finds a GPU with a REAL CUDA allocation probe before
# launching, instead of relying on --gpu auto's internal pick_gpu()+reserve_vram() sequence.
#
# WHY THE REWRITE. The first version just retried `--gpu auto` on failure, but the race is
# INSIDE that one process: pick_gpu() reads nvidia-smi, decides a card is free, then
# reserve_vram() tries to actually claim it a few seconds later -- and on this shared node an
# outside tenant repeatedly won that gap. It happened on GPU1 twice, and after 30 retries (all
# hitting the same race on whatever pick_gpu chose) the wrapper gave up entirely. Probing for a
# real free GPU BEFORE invoking python, then passing an explicit --gpu, moves the check-then-use
# gap from "nvidia-smi read vs 40s of model loading" down to "one small tensor alloc vs the next
# line of this script" -- the same fix already applied to the wait_and_launch_*.sh scripts.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"
cd "$REPO"
source /opt/conda/etc/profile.d/conda.sh
conda activate ttd_train
export PYTHONPATH="$REPO"

FREE_MB="${FREE_MB:-2000}"
PY=/opt/conda/envs/ttd_train/bin/python

probe_gpu() {
  CUDA_VISIBLE_DEVICES="$1" timeout 60 "$PY" -c \
    "import torch; torch.zeros(int(27000*0.9*1024*1024//4), device='cuda'); torch.cuda.synchronize()" \
    >/dev/null 2>&1
}

find_free_gpu() {
  while IFS=', ' read -r idx used; do
    [ "$used" -lt "$FREE_MB" ] || continue
    if probe_gpu "$idx"; then echo "$idx"; return 0; fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  return 1
}

ATTEMPT=0
MAX_ATTEMPTS="${MAX_ATTEMPTS:-200}"
while [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; do
  ATTEMPT=$((ATTEMPT+1))
  GPU="$(find_free_gpu || true)"
  if [ -z "$GPU" ]; then
    echo "[heldout-retry] $(date) attempt $ATTEMPT: no GPU passed the real-allocation probe, waiting"
    sleep 60
    continue
  fi
  echo "[heldout-retry] $(date) attempt $ATTEMPT: GPU $GPU probed free, launching"
  python -u scripts/heldout_batch_eval.py \
    --ckpt outputs/.grid_pin/step10000.ckpt \
    --tag arm6_10000_heldout16 --gpu "$GPU"
  EXIT=$?
  echo "[heldout-retry] $(date) attempt $ATTEMPT exited $EXIT"
  if [ "$EXIT" -eq 0 ]; then
    echo "[heldout-retry] done"
    exit 0
  fi
  sleep 30
done
echo "[heldout-retry] gave up after $MAX_ATTEMPTS attempts"
exit 1
