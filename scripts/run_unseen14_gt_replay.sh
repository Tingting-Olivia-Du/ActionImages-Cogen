#!/bin/bash
# GT-replay ceiling on the same 14 unseen tasks and the SAME scene seeds as the model
# campaign (trial_seed() depends only on task/variation/trial, so the scenes are paired).
#
# WHY. A 0/20 cell is unreadable without it: basketball_in_hoop was 0/20 for five different
# models in the 2026-09-15 campaign, which is a statement about the actuation harness, not
# about any model. tab:pertask needs this column to exclude harness floors from the aggregate.
#
# No GPU and no model -- one CoppeliaSim per task, all 14 in parallel on CPU.
set -uo pipefail
REPO=/workspace/1228_tingting/ActionImages-Cogen
source /workspace/1228_tingting/ttd/scripts/env_eval_ttd_eval.rc
cd "$REPO"
TRIALS="${TRIALS:-20}"
TAGPFX="${TAGPFX:-u14_gtreplay}"
OUT="${OUT:-$REPO/reports/closedloop_unseen14}"
LOGS="${LOGS:-$REPO/logs/logs_cl_unseen14_gt}"
TASKS="${TASKS:-close_box close_drawer close_microwave take_lid_off_saucepan toilet_seat_down \
basketball_in_hoop meat_on_grill light_bulb_out take_item_out_of_drawer take_money_out_safe \
put_rubbish_in_bin stack_cups open_window wipe_desk}"
mkdir -p "$OUT" "$LOGS"
for t in $TASKS; do
  # Completeness, not existence: rollout.py writes its JSON incrementally, so a killed
  # sweep leaves short files that an `-f` test would skip forever.
  if python "$REPO/scripts/cl_json_complete.py" "$OUT/rollout_${TAGPFX}_${t}.json" "$TRIALS"; then
    echo "skip $t (complete)"; continue
  fi
  xvfb-run -a python -u eval/rollout.py --gt-replay \
    --tag "${TAGPFX}_${t}" --tasks "$t" --variation 0 --num-trials "$TRIALS" \
    --res 512 --arm-action-mode planning --max-steps-factor 1.5 \
    --no-record-video --out "$OUT" > "$LOGS/${t}.log" 2>&1 &
  sleep 3
done
wait
echo "GT_REPLAY_DONE $(date)"
