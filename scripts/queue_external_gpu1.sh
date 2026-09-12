#!/bin/bash
# Second worker: the four external-baseline probes, which sit at the TAIL of
# queue_maintable.sh's strictly-sequential list and would otherwise not start
# for ~12h while GPU 1 sits idle.
#
# Safe to run alongside queue_maintable.sh: each probe now writes
# reports/external/<name>_n<N>.json and exits early if that file already exists,
# so whichever worker reaches a probe second just skips it. Do NOT edit
# queue_maintable.sh while it runs -- bash reads scripts incrementally.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
PY=/opt/conda/envs/ttd_train/bin/python
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
export PYTHONPATH="$REPO" N_EP_PER_TASK=5 CUDA_VISIBLE_DEVICES=1

# Real allocation probe, not nvidia-smi: a card can look free and still hold
# leaked memory from a dead process.
until timeout 60 "$PY" -c \
  "import torch;torch.zeros(int(28000*0.9*1024*1024//4),device='cuda');torch.cuda.synchronize()" \
  >/dev/null 2>&1; do
  echo "[gpu1] $(date -u +%H:%M) GPU 1 not claimable yet, retry in 3m"; sleep 180
done
echo "[gpu1] $(date -u +%H:%M) claimed GPU 1"

for job in "da2 $PY -u scripts/probe_depth_baselines.py da2 40" \
           "da3 $PY -u scripts/probe_depth_baselines.py da3 40" \
           "vggt $PY -u scripts/probe_vggt_vs_ours.py 40" \
           "sam $PY -u scripts/probe_sam_vs_ours.py 40"; do
  name=${job%% *}; cmd=${job#* }
  echo "[gpu1] $(date -u +%H:%M) START $name"
  $cmd >> "$REPO/logs_queue_${name}.txt" 2>&1
  echo "[gpu1] $(date -u +%H:%M) $name exited $?"
done
echo "[gpu1] ALL EXTERNAL PROBES DONE $(date -u +%H:%M)"
