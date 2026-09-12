#!/bin/bash
# Everything still missing from Table 1 and Table 2.
#
# Table 1's bo-IoU row is the only one where the in-domain specialist column would be empty,
# and its unseen-task cell has a counterpart (SAM 0.670) but no number of ours -- the one
# place a reader could think we avoided a comparison.
#
# Table 2's Success column is worse than incomplete, it is misleading: all three campaigns
# ran five tasks, two of which (close_box, close_drawer) are absent from the training tree.
# Pooling them hides two opposite effects -- fine-tuning takes the released checkpoint from
# 6.7% to 35% on trained tasks while dropping it from 50% to 35% on untrained ones. The
# twelve-task campaign the unified model already has needs the same basis for the other two
# models before the row means anything.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
PY=/opt/conda/envs/ttd_train/bin/python
LEASE="bash $REPO/scripts/gpu_lease.sh"
SP="$REPO/outputs"
GATED="insert_onto_square_peg light_bulb_in meat_off_grill open_drawer \
place_shape_in_shape_sorter push_buttons put_groceries_in_cupboard put_item_in_drawer \
put_money_in_safe reach_and_drag stack_blocks turn_tap"

run() {  # name outfile mb cmd...
  local name="$1" out="$2" mb="$3"; shift 3
  for a in 1 2 3; do
    [ -s "$out" ] && { echo "[gaps2] $name already done"; return 0; }
    local g; g=$($LEASE acquire "$mb")
    echo "[gaps2] $(date -u +%H:%M) START $name attempt $a on GPU $g"
    CUDA_VISIBLE_DEVICES=$g "$@" >> "$REPO/logs_gaps2_${name}.txt" 2>&1
    local rc=$?; $LEASE release "$g"
    [ -s "$out" ] && { echo "[gaps2] $name OK (rc=$rc)"; return 0; }
    echo "[gaps2] $name no output (rc=$rc) -- retry"; sleep 60
  done
  echo "[gaps2] $name FAILED after 3 attempts"
}

# ---- A. Table 1 bo-IoU gaps ----
(
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
export PYTHONPATH="$REPO" N_EP_PER_TASK=5

# ours on the unseen-task tree
export DATA_TREE="$REPO/data/rlbench_unseen_tasks_512_aug" SPLIT_TAG="_task" \
       EPS_FILE="$REPO/reports/external/unseen_task_eps.txt" MODEL_TAG=ours
unset EVAL_VARIATION
run boiou_ours_task "$REPO/reports/external/ours_classagnostic_fifi_task_n66.json" 28000 \
  $PY -u scripts/probe_ours_class_agnostic_seg.py fifi,iiii 66

# the segmentation specialist, both tiers -- fills the specialist column for this row
export MODEL_TAG=segspec CKPT="$SP/specialist_seg__seed42_fi3_512_aug_sr/checkpoint-2000/step2000.ckpt"
run boiou_segspec_task "$REPO/reports/external/segspec_classagnostic_iiii_task_n66.json" 28000 \
  $PY -u scripts/probe_ours_class_agnostic_seg.py iiii 66
export DATA_TREE="$REPO/data/rlbench_selfgen_512_aug" SPLIT_TAG="_var1" EVAL_VARIATION=1
unset EPS_FILE
run boiou_segspec_var1 "$REPO/reports/external/segspec_classagnostic_iiii_var1_n40.json" 28000 \
  $PY -u scripts/probe_ours_class_agnostic_seg.py iiii 40
echo "[gaps2] TABLE-1 GAPS DONE $(date -u +%H:%M)"
) 2>&1 | tee -a "$REPO/logs_gaps2_queue.txt"

echo "[gaps2] ALL DONE $(date -u +%H:%M)"
