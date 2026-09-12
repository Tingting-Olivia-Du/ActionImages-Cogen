#!/bin/bash
# Which of the trained tasks can this harness measure at all?
#
# Every closed-loop number rests on DEFAULT_TASKS = 3 tasks. close_jar was excluded because
# GT replay scores 0% on it despite zero IK failures -- a harness floor, not a model result.
# That test was never applied to the other 12 tasks with >=2 variations. This runs it on all
# of them, so the all-task campaign can separate "the model failed" from "end-effector-pose
# + discrete-gripper replay cannot express this task".
#
# CPU-only: rollout.py skips build_pipeline entirely under --gt-replay, and CUDA_VISIBLE_DEVICES
# is emptied so it cannot take a card from the GPU campaign running alongside it.
# Thread caps: CoppeliaSim renders through llvmpipe, which grabs ~34 cores per worker if left
# alone; capping it keeps this off the back of the two rollout jobs sharing the box.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
source /opt/conda/etc/profile.d/conda.sh
source /workspace/ttdu/ttd/scripts/env_eval.rc
conda activate ttd_rollout
export PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES=""
export LP_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
TASKS="close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer \
place_shape_in_shape_sorter push_buttons put_groceries_in_cupboard put_item_in_drawer \
put_money_in_safe reach_and_drag stack_blocks turn_tap"
V=$1
xvfb-run -a python -u eval/rollout.py --gt-replay --res 512 --frame-interval 1 \
  --axis-solver sphere --arm-action-mode planning --max-steps-factor 1.5 \
  --variation $V --num-trials 10 --tag gt_gate_all_v$V --tasks $TASKS
echo "[gate] variation $V exited $?"
