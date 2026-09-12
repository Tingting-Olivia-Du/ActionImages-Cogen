#!/bin/bash
# Scale the matched-exposure evaluation from 8 held-out episodes to 40.
#
# WHY. Table 1's intervals at n=8 only exclude degradations larger than ~0.03 AbsRel, and this
# session produced three concrete cases where a small sample was actively misleading -- VGGT went
# 0.093 -> 0.206 between 3 and 8 episodes, and FIFI depth went 0.215 -> 1.538 between 6 and 8,
# because the missing episodes were the hard ones. The tree holds 240 held-out episodes; using 8
# of them was leaving the cheapest possible variance reduction on the table.
#
# Resume makes this incremental: the 8 episodes already scored are skipped, so this adds cells
# rather than repeating work.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="${GPU:?set GPU}"
echo "[scaleup] $(date) $TAG ($MODS) on physical GPU $GPU"
python -u scripts/heldout_batch_eval.py \
  --ckpt "$CKPT" --tag "$TAG" --gpu 0 --modalities "$MODS" \
  --episodes "$(cat /tmp/eps40.txt)"
echo "[scaleup] $(date) exited $?"
