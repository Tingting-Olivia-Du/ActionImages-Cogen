#!/bin/bash
# Closed-loop on ALL trained tasks that have more than one variation, not just the three
# in DEFAULT_TASKS.
#
# Why: every closed-loop number in the draft rests on push_buttons / meat_off_grill /
# open_drawer. open_drawer is 0% in both tiers, so the headline "35% -> 1.7%" effectively
# rests on TWO tasks, and push_buttons dominates it. The training tree has 16 tasks, 13 of
# them with >=2 variations. Those 12 others were never tried -- only close_jar was ever
# ruled out, and that was on a GT-replay argument that was never applied to the rest.
#
# 13 tasks x 10 trials = 130 rollouts per variation. Ten and not five because the per-task
# breakdown is itself a paper table, and an x/5 cell can only take six distinct values.
# Protocol is byte-identical to the original campaign (cfg 7.5, fi 3, explicit, planning,
# sphere, factor 1.5, anchor 4, max_ik_fail_streak left at its default 5), so these numbers
# drop straight into the existing comparison.
#
# rollout.py dumps its json inside the trial loop, so per-task results are readable as they
# land rather than only at the end.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
source /opt/conda/etc/profile.d/conda.sh
source /workspace/ttdu/ttd/scripts/env_eval.rc
conda activate ttd_rollout
export PYTHONPATH="$REPO"
TASKS="close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer \
place_shape_in_shape_sorter push_buttons put_groceries_in_cupboard put_item_in_drawer \
put_money_in_safe reach_and_drag stack_blocks turn_tap"
G=$1; V=$2
CUDA_VISIBLE_DEVICES=$G xvfb-run -a python -u eval/rollout.py \
  --ckpt outputs/.grid_pin/step10000.ckpt --res 512 --cfg 7.5 --steps 50 \
  --frame-interval 3 --prompt-tag-style explicit --axis-solver sphere \
  --arm-action-mode planning --max-steps-factor 1.5 --skip-anchor-frames 4 \
  --variation $V --num-trials 10 --tag arm6_alltasks_v$V --tasks $TASKS
echo "[alltasks] variation $V exited $?"
