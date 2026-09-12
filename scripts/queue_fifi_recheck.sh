#!/bin/bash
# Re-run FIFI depth on the held-out variation with decode coverage recorded.
#
# The reported 0.385 is reproducible bit-for-bit across two independent runs, so it is not
# noise. But it is carried by five of eighty episodes that jump from ~0.05 to a plateau near
# 2.1 in a single frame and stay there -- the signature of a decode failure rather than
# drift, and IIII and the depth specialist score normally on those same episodes. Depth is
# encoded as a colour path and off-path pixels decode to NaN, so the hypothesis is that the
# generation leaves the path and the mean is then taken over a shrinking, unrepresentative
# subset. Coverage is now recorded per frame, which tests that directly.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
PY=/opt/conda/envs/ttd_train/bin/python
LEASE="bash $REPO/scripts/gpu_lease.sh"
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
export PYTHONPATH="$REPO"
OUT="$REPO/reports/heldout_batch_eval/fifi_recheck/per_episode.json"
for a in 1 2 3; do
  [ -s "$OUT" ] && break
  G=$($LEASE acquire 28000)
  echo "[recheck] $(date -u +%H:%M) START attempt $a on GPU $G"
  CUDA_VISIBLE_DEVICES=$G $PY -u scripts/heldout_batch_eval.py \
    --ckpt outputs/.grid_pin/step10000.ckpt --tag fifi_recheck --gpu 0 --mode fifi \
    --modalities depth --episodes "$(cat /tmp/eps40.txt)" \
    >> "$REPO/logs_fifi_recheck.txt" 2>&1
  rc=$?; $LEASE release "$G"
  [ -s "$OUT" ] && { echo "[recheck] OK (rc=$rc)"; break; }
  echo "[recheck] no output (rc=$rc) -- retry"; sleep 60
done
echo "[recheck] DONE $(date -u +%H:%M)"
