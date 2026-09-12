#!/bin/bash
# Two follow-ups, run as GPUs free. Job 2 first: it is 30 minutes and fills a column that
# currently holds one number; job 1 is 24 GPU-hours and can absorb whatever is left.
#
# JOB 2 -- external baselines on the other two tiers.
#   DA2/DA3/VGGT/SAM have only ever run on variation 1, so the master table's External
#   column has a single populated cell. The probes now take DATA_TREE / EVAL_VARIATION /
#   EPS_FILE / SPLIT_TAG from the environment, so the same code covers all three tiers and
#   writes to distinct json names instead of colliding with the variation-1 results.
#
# JOB 1 -- the IK-abort control, redone on the 12-task basis.
#   max_ik_fail_streak=5 is a hardcoded harness policy that killed 16/20 push_buttons
#   rollouts on variation 1 against 5/20 on variation 0, so it can turn a small geometric
#   degradation into a near-total collapse. Raising it to 50 on BOTH variations separates
#   "the model fails" from "the harness gives up". The earlier attempt used the old 3-task
#   set; this uses the 12 tasks that passed the GT-replay gate (close_jar excluded -- its
#   ground-truth replay scores 0/10, so it measures the harness, not the model).
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
PY=/opt/conda/envs/ttd_train/bin/python
GATED="insert_onto_square_peg light_bulb_in meat_off_grill open_drawer \
place_shape_in_shape_sorter push_buttons put_groceries_in_cupboard put_item_in_drawer \
put_money_in_safe reach_and_drag stack_blocks turn_tap"

claim() {
  while :; do
    for i in 4 0 1 5 6 7 2 3; do
      if CUDA_VISIBLE_DEVICES=$i timeout 50 "$PY" -c \
         "import torch;torch.zeros(int(${1:-28000}*0.9*1024*1024//4),device='cuda');torch.cuda.synchronize()" \
         >/dev/null 2>&1; then echo "$i"; return 0; fi
    done
    sleep 300
  done
}

# ---------- JOB 2: external baselines on seen + unseen-task ----------
(
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
export PYTHONPATH="$REPO" N_EP_PER_TASK=5
for CFG in "seen:$REPO/data/rlbench_selfgen_512_aug:0:" \
           "task:$REPO/data/rlbench_unseen_tasks_512_aug::$REPO/reports/external/unseen_task_eps.txt"; do
  IFS=: read -r TAG TREE VAR EPSF <<< "$CFG"
  export DATA_TREE="$TREE" SPLIT_TAG="_$TAG"
  [ -n "$VAR" ] && export EVAL_VARIATION="$VAR" || unset EVAL_VARIATION
  [ -n "$EPSF" ] && export EPS_FILE="$EPSF" || unset EPS_FILE
  N=$([ "$TAG" = task ] && echo 66 || echo 40)
  for J in "da2:scripts/probe_depth_baselines.py da2 $N" \
           "da3:scripts/probe_depth_baselines.py da3 $N" \
           "vggt:scripts/probe_vggt_vs_ours.py $N" \
           "sam:scripts/probe_sam_vs_ours.py $N"; do
    name=${J%%:*}; args=${J#*:}
    G=$(claim 20000); echo "[ext2] $(date -u +%H:%M) START $name/$TAG on GPU $G"
    CUDA_VISIBLE_DEVICES=$G $PY -u $args >> "$REPO/logs_ext2_${name}_${TAG}.txt" 2>&1
    echo "[ext2] $(date -u +%H:%M) $name/$TAG exited $?"
  done
done
echo "[ext2] ALL DONE $(date -u +%H:%M)"
) 2>&1 | tee -a "$REPO/logs_queue_next2.txt"

# ---------- JOB 1: IK-abort control on the gated 12 ----------
(
source /opt/conda/etc/profile.d/conda.sh
source /workspace/ttdu/ttd/scripts/env_eval.rc
conda activate ttd_rollout
export PYTHONPATH="$REPO"
for V in 1 0; do
  G=$(claim 28000); echo "[ikctl] $(date -u +%H:%M) START variation $V on GPU $G"
  CUDA_VISIBLE_DEVICES=$G xvfb-run -a python -u eval/rollout.py \
    --ckpt outputs/.grid_pin/step10000.ckpt --res 512 --cfg 7.5 --steps 50 \
    --frame-interval 3 --prompt-tag-style explicit --axis-solver sphere \
    --arm-action-mode planning --max-steps-factor 1.5 --skip-anchor-frames 4 \
    --variation $V --num-trials 10 --max-ik-fail-streak 50 \
    --tag arm6_gated12_v${V}_ikstreak50 --tasks $GATED \
    >> "$REPO/logs_ikctl_v${V}.txt" 2>&1
  echo "[ikctl] $(date -u +%H:%M) variation $V exited $?"
done
echo "[ikctl] ALL DONE $(date -u +%H:%M)"
) 2>&1 | tee -a "$REPO/logs_queue_next2.txt"
