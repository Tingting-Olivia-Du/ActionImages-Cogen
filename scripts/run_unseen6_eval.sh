#!/bin/bash
# arm6 (or a specialist) on SIX never-trained tasks, up from two. Answers the weakness the
# generalization section admits about itself -- "two unseen tasks is a probe, not a survey".
# The four added tasks span different manipulation types on purpose: close_microwave is
# articulated like close_drawer (a within-family control), take_lid_off_saucepan reverses
# close_box's direction, toilet_seat_down is articulated with very different geometry, and
# basketball_in_hoop is free-space placement.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="${GPU:?set GPU}"
echo "[unseen6] $(date) $TAG on GPU $GPU"
python -u scripts/heldout_batch_eval.py \
  --ckpt "$CKPT" --tag "$TAG" --gpu 0 \
  --data "$REPO/data/rlbench_unseen_tasks_512_aug" \
  --episodes "$(cat /tmp/unseen_eps6.txt)" \
  --split_label unseen_task --modalities "${MODS:-depth,segmentation,normal}"
echo "[unseen6] $(date) exited $?"
