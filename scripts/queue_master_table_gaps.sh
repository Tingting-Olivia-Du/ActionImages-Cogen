#!/bin/bash
# The cells still missing from the merged master table (tab:master).
#
# After the merge, every row is (output x tier). Two rows have a hole in the
# unseen-task tier, which is exactly the tier the merge exists to stop treating
# as an afterthought:
#   1. Action has no unseen-task column at all (neither unified nor specialist).
#   2. FIFI has no unseen-task column for depth / segmentation / normal.
# Both run on the same 66 evaluation-only episodes the other unseen-task numbers
# use, read from the tree that is deliberately kept out of training.
#
# Sequential, one card at a time, each waiting for a real 28 GB allocation --
# nvidia-smi can report a card free while it still holds a dead process's memory.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
PY=/opt/conda/envs/ttd_train/bin/python
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
export PYTHONPATH="$REPO"
DATA="$REPO/data/rlbench_unseen_tasks_512_aug"
EPS=$(cat /tmp/unseen_eps.txt)
SP="$REPO/outputs"

claim() {
  while :; do
    for i in 4 1 0 5 6 7 2 3; do
      if CUDA_VISIBLE_DEVICES=$i timeout 50 "$PY" -c \
         "import torch;torch.zeros(int(28000*0.9*1024*1024//4),device='cuda');torch.cuda.synchronize()" \
         >/dev/null 2>&1; then echo "$i"; return 0; fi
    done
    echo "[gaps] no card claimable, retry 5m" >&2; sleep 300
  done
}
run() {
  local name="$1"; shift
  local g; g=$(claim)
  echo "[gaps] $(date -u +%H:%M) START $name on GPU $g"
  CUDA_VISIBLE_DEVICES=$g "$@" >> "$REPO/logs_gaps_${name}.txt" 2>&1
  echo "[gaps] $(date -u +%H:%M) $name exited $?"
}

# 1) action on unseen tasks -- unified, then the matched-exposure specialist
run uni_action_task $PY -u scripts/heldout_batch_eval.py \
  --ckpt outputs/.grid_pin/step10000.ckpt --data "$DATA" \
  --tag arm6_10000_unseen_task --gpu 0 --modalities action --episodes "$EPS"

run spec_action_task $PY -u scripts/heldout_batch_eval.py \
  --ckpt "$SP/specialist_action__seed42_fi3_512_aug_sr/checkpoint-4000/step4000.ckpt" \
  --data "$DATA" --tag actionspec_4000_unseen_task --gpu 0 --modalities action --episodes "$EPS"

# 2) FIFI on unseen tasks
run fifi_task $PY -u scripts/heldout_batch_eval.py \
  --ckpt outputs/.grid_pin/step10000.ckpt --data "$DATA" \
  --tag arm6_10000_fifi_unseen_task --gpu 0 --mode fifi \
  --modalities depth,segmentation,normal --episodes "$EPS"

echo "[gaps] ALL DONE $(date -u +%H:%M)"
