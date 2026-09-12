#!/bin/bash
# Offline action + RGB metrics for the RELEASED checkpoint -- the Initialization row of
# tab:wam_preservation, which is otherwise entirely empty (only its closed-loop Success is known).
#
# --prompt_tag_style none is REQUIRED and is the whole reason this needs its own script: the
# released checkpoint was trained without modality tags, so prefixing <video><action> would score
# it out of its own prompt distribution and understate it. Same reasoning as the closed-loop
# protocol split in scripts/run_official_closedloop_postfix.sh.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="${GPU:?set GPU}"
echo "[init-eval] $(date) released checkpoint, prompt_tag_style=none, GPU $GPU"
python -u scripts/heldout_batch_eval.py \
  --ckpt /workspace/ttdu/starVLA/playground/Pretrained_models/anyeZHY/ActionImages/step125750.ckpt \
  --tag init_125750 --gpu 0 --modalities action --prompt_tag_style none
echo "[init-eval] $(date) exited $?"
