#!/bin/bash
# Render close_box / close_drawer -- the two closed-loop tasks that are NOT in the training tree
# -- so perception (depth / normal / segmentation) can be measured on genuinely unseen TASKS,
# not just unseen variations. See EXPERIMENT_STATUS_zh_aug29.md sec.14.
#
# ⚠️ WRITES TO A SEPARATE TREE, NEVER data/rlbench_selfgen_512_aug. Training runs with
# `--variations 0` over the whole tree, so putting close_box/variation0 into it would silently
# fold these tasks into the next training run and destroy the very property being measured.
#
# SEEDS ARE THE CLOSED-LOOP SEEDS. gen_dataset.py calls np.random.seed(seed + v*1_000_003 +
# attempt*100003) before get_demos; at v=0, attempt=0 that is exactly np.random.seed(seed), which
# is what eval/rollout_env.py's reset_to_new_demo does with trial_seed(task, 0, trial). So passing
# the trial seeds reproduces the same object layouts the rollouts faced. (Pixels still differ:
# this tree is rendered WITH colosseum augmentation, matching the training distribution, whereas
# the closed-loop env runs unaugmented. Layout corresponds; appearance does not.)
#
# AUG/RES MATCH THE TRAINING TREE (colosseum, 512): the point of the comparison is to read these
# numbers against the 16 trained tasks, so the visual domain must be the same one the model was
# trained in. Rendering unaugmented would confound "unseen task" with "unseen visual condition".
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate ttd_eval
source /workspace/ttdu/ttd/scripts/env_eval.rc
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
# robot-colosseum is not installed into the env; gen_launch_512.sh:55 puts it on the path
export PYTHONPATH=/workspace/ttdu/robot-colosseum${PYTHONPATH:+:$PYTHONPATH}

OUT="${OUT:-/workspace/ttdu/ttd/data/rlbench_unseen_tasks_512_aug}"
RES="${RES:-512}"; AUG="${AUG:-colosseum}"
TASKS="${TASKS:-close_box,close_drawer}"
SEEDS="${SEEDS:?set SEEDS=comma list}"
export LP_NUM_THREADS="${LP_NUM_THREADS:-4}"   # 8.9x cheaper in core-seconds, see gen_launch_512.sh

echo "[gen-unseen] $(date) tasks=$TASKS res=$RES aug=$AUG out=$OUT"
echo "[gen-unseen] seeds=$SEEDS"
xvfb-run -a python -u scripts/gen_dataset.py \
  --tasks "$TASKS" --seeds "$SEEDS" --out "$OUT" \
  --variations 0 --res "$RES" --aug "$AUG"
echo "[gen-unseen] $(date) exited $?"
