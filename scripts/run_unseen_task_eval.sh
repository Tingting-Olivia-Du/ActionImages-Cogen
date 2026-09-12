#!/bin/bash
# Perception on tasks that were NEVER TRAINED ON -- the tier that was previously unmeasurable.
#
# The open-loop harness scores against on-disk GT, so it can only see tasks that exist in a
# rendered tree; close_box/close_drawer are absent from the training tree, which is why every
# earlier perception number stopped at "unseen variation of a trained task". This run uses the
# separate data/rlbench_unseen_tasks_512_aug tree, rendered at the closed-loop trial seeds.
#
# --split_label unseen_task is REQUIRED: these episodes live under variation0 (each task has only
# one variation), so the default variation-based tier inference would file them as training data.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="${GPU:?set GPU}"   # shell-level mask; --gpu alone is unreliable
echo "[unseen-task] $(date) $TAG on physical GPU $GPU"
python -u scripts/heldout_batch_eval.py \
  --ckpt "$CKPT" --tag "$TAG" --gpu 0 \
  --data "$REPO/data/rlbench_unseen_tasks_512_aug" \
  --episodes "$(cat /tmp/unseen_eps.txt)" \
  --split_label unseen_task --modalities "${MODS:-depth,segmentation,normal}"
echo "[unseen-task] $(date) exited $?"
