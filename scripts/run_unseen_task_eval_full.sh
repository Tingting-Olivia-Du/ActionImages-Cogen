#!/bin/bash
# Widen the unseen-task tier from 8 to all 20 rendered episodes per task. The extra 24 episodes
# are already on disk, and resume means only the missing cells are generated. Purely a
# CI-tightening run: tier 3's depth CI is [0.116, 0.235] at n=16, wide enough that "does not
# collapse" is the strongest readable claim.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="${GPU:?set GPU}"
echo "[unseen-full] $(date) GPU $GPU"
python -u scripts/heldout_batch_eval.py \
  --ckpt outputs/.grid_pin/step10000.ckpt --tag arm6_10000_unseen_task --gpu 0 \
  --data "$REPO/data/rlbench_unseen_tasks_512_aug" \
  --episodes "$(cat /tmp/unseen_eps_full.txt)" \
  --split_label unseen_task --modalities depth,segmentation,normal
echo "[unseen-full] $(date) exited $?"
