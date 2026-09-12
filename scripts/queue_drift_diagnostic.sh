#!/bin/bash
# Does IIII's depth error measure geometry, or the video drifting away from the episode it
# is scored against? score() compares generated depth to the REAL episode's ground truth
# pixel-for-pixel with no correspondence step, so under IIII any divergence between the
# predicted future and the real one lands in the perception metric.
#
# Frame 0 of each segment is a given anchor, so the hypotheses predict opposite curve shapes:
#   drift        -> IIII error grows with t, FIFI stays flat
#   bad geometry -> both flat, IIII simply offset upward
# score() returns absrel_per_frame, so the existing metric is the t-average of exactly this
# curve -- no new measure, no new data.
#
# The first attempt reported "exited 0" one minute in after losing its card to a sibling
# queue between the probe and model load; heldout_batch_eval exits 0 on that path. Leases
# and an explicit output check close that hole.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
PY=/opt/conda/envs/ttd_train/bin/python
LEASE="bash $REPO/scripts/gpu_lease.sh"
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
export PYTHONPATH="$REPO"
EPS=$(cat /tmp/eps40.txt)
for M in iiii fifi; do
  out="$REPO/reports/heldout_batch_eval/drift_${M}/per_episode.json"
  for attempt in 1 2 3; do
    [ -s "$out" ] && break
    G=$($LEASE acquire 28000)
    echo "[drift] $(date -u +%H:%M) START $M attempt $attempt on GPU $G"
    CUDA_VISIBLE_DEVICES=$G $PY -u scripts/heldout_batch_eval.py \
      --ckpt outputs/.grid_pin/step10000.ckpt --tag drift_${M} --gpu 0 --mode $M \
      --modalities depth --episodes "$EPS" >> "$REPO/logs_drift_${M}.txt" 2>&1
    rc=$?; $LEASE release "$G"
    if [ -s "$out" ]; then echo "[drift] $M OK (rc=$rc)"; break
    else echo "[drift] $M produced no output (rc=$rc) -- retrying"; sleep 60; fi
  done
done
echo "[drift] ALL DONE $(date -u +%H:%M)"
