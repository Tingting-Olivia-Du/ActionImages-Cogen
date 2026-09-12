#!/bin/bash
# A specialist on the NEVER-TRAINED tasks -- the control for sec:exp_generalization.
#
# THE QUESTION. The unified model's perception barely degrades on close_box / close_drawer
# (depth 0.154 -> 0.171, normal 0.892 -> 0.898). That is currently reported as a property of the
# unified model, but nothing in the section rules out the alternative: that this is the frozen
# VAE plus the released backbone generalizing, and unification has nothing to do with it. A
# single-modality specialist, trained on exactly the same tree for the same number of updates on
# that modality, separates the two:
#
#   specialist generalizes equally  -> cross-task perception is a backbone property; the paper
#                                      must NOT claim unification buys it.
#   specialist degrades more        -> training alongside the other streams is what makes the
#                                      perception head robust off-distribution. A real result.
#
# Matched exposure is what makes the comparison fair, so each specialist is read at
# checkpoint-2000 (0.2T), the same point tab:headline uses.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="${GPU:?set GPU}"   # shell-level mask; --gpu alone is unreliable
ARM="${ARM:?set ARM=depth|seg|normal}"; MODS="${MODS:?set MODS}"
CKPT="$REPO/outputs/specialist_${ARM}__seed42_fi3_512_aug_sr/checkpoint-2000/step2000.ckpt"
echo "[spec-unseen] $(date) $ARM specialist @2000 on never-trained tasks, GPU $GPU"
python -u scripts/heldout_batch_eval.py \
  --ckpt "$CKPT" --tag "${ARM}spec_2000_unseen_task" --gpu 0 \
  --data "$REPO/data/rlbench_unseen_tasks_512_aug" \
  --episodes "$(cat /tmp/unseen_eps.txt)" \
  --split_label unseen_task --modalities "$MODS"
echo "[spec-unseen] $(date) exited $?"
