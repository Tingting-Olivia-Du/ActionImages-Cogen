#!/bin/bash
# Track B, GPU 0, two stages run back to back:
#
# Stage 1: full closed-loop campaign for arm6@step10000 (the headline four-stream checkpoint),
# same protocol as the arm4 table in the paper (tab:closedloop) -- 5 tasks x 20 trials = 100
# rollouts, paired scene seeds vs the official init. arm6 currently only has an n=4/task smoke
# test (reports/closedloop/rollout_arm6_9000.json, step 9000) -- the four-stream model has never
# had the same evidentiary bar as the two-stream arm4. This closes that gap at the FINAL
# checkpoint (10000, not 9000).
#
# Stage 2: tag-vs-anchor ablation (scripts/tag_swap_ablation.py) extended from the existing n=1
# episode (open_drawer only, step 9000) to the same 4 held-in episodes tab:crosstask already
# uses (open_drawer, push_buttons, stack_wine, turn_tap), at the final checkpoint. The existing
# result says the tag barely moves the metric (~1.7% spread) while removing the anchor entirely
# collapses it (f0f0) -- i.e. the ANCHOR selects the modality, not the prompt tag. That claim
# currently rests on one episode; this makes it four.
#
# Closed-loop runs first because it is the bigger, more obviously-missing piece of evidence and
# eval/rollout.py checkpoints its JSON after every trial (json.dump at line ~452/460), so partial
# progress is readable even if this script is still mid-campaign when checked.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"
cd "$REPO"

echo "[track-b] $(date) stage 1: arm6@10000 closed-loop campaign, n=20/task, GPU 0"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

# TASKS must match tab:closedloop's 5 tasks exactly (close_box, close_drawer, push_buttons,
# meat_off_grill, open_drawer) for the result to be comparable to the arm4/official table --
# queue_closedloop.sh's own default TASKS list is different (swaps in slide_block_to_target and
# turn_tap), which would silently produce a non-comparable set if left unoverridden.
GPU=0 TRIALS=20 TASKS="close_box close_drawer push_buttons meat_off_grill open_drawer" \
  bash scripts/queue_closedloop.sh
echo "[track-b] $(date) stage 1 (closed-loop) exited with $?"

echo "[track-b] $(date) stage 2: tag-swap ablation across 4 held-in episodes, GPU 0"
source /opt/conda/etc/profile.d/conda.sh
conda activate ttd_train
export PYTHONPATH="$REPO"

CKPT="$REPO/outputs/.grid_pin/step10000.ckpt"
for EP in open_drawer push_buttons stack_wine turn_tap; do
  echo "[track-b] $(date) tag-swap episode=$EP"
  python -u scripts/tag_swap_ablation.py \
    --ckpt "$CKPT" --tag "arm6_10000_${EP}" --gpu 0 \
    --episode "${EP}/variation0/episodes/episode0"
  echo "[track-b] $(date) tag-swap episode=$EP exit=$?"
done

echo "[track-b] $(date) stage 2 done -- run scripts/tag_swap_report.py per episode tag to summarize"
