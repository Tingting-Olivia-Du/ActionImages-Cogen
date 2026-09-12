#!/bin/bash
# Recompute arm6's action cells, which the first sweep nulled.
#
# WHY. scripts/heldout_batch_eval.py used to overwrite score_action's correct median with None
# (it guarded on a key name that is the metric's NAME, not a key of the returned dict). The
# depth/segmentation/normal cells of that sweep are unaffected and valid. With the resume support
# added alongside the fix, re-running the SAME tag with --modalities action skips every already-
# scored cell and recomputes only the nulled action ones, then rewrites summary.json.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="${GPU:-5}"   # shell-level mask; --gpu is unreliable, see queue script

WAIT_PID="239922"   # arm6 held-out eval; refix reuses its GPU when it exits
if [ -n "$WAIT_PID" ]; then
  echo "[refix] waiting for eval queue pid $WAIT_PID to finish"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
  echo "[refix] $(date) queue done; starting action re-run"
  sleep 20
fi
python -u scripts/heldout_batch_eval.py \
  --ckpt outputs/.grid_pin/step10000.ckpt --tag arm6_10000 --gpu 0 --modalities action
echo "[refix] $(date) exited $?"
