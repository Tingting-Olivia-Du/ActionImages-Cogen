#!/bin/bash
# Phase 2 of the closed-loop campaign: ONE queue across both simulators, ordered by the
# priority the paper actually has -- every arm7 number first (RLBench then ManiSkill), then
# arm0, then the released checkpoint.
#
# Phase 1 (scripts/run_unseen14_campaign.sh) is RLBench-only and runs arm7's 14 tasks first.
# This script takes over once those are done: it re-lists every job, skips the ones already
# complete on disk, and dispatches the rest across the GPUs. Both backends need ~29 GB, so it
# is strictly one job per card.
#
#   bash scripts/run_cl_phase2.sh
set -uo pipefail
REPO=/workspace/1228_tingting/ActionImages-Cogen
source /workspace/1228_tingting/ttd/scripts/env_eval_ttd_eval.rc
export VK_ICD_FILENAMES=/workspace/1228_tingting/envs/ttd_eval/lib/python3.10/site-packages/sapien/vulkan_library/nvidia_icd.json
export GIT_PYTHON_REFRESH=quiet
cd "$REPO"

TRIALS="${TRIALS:-20}"
# GPU 0 was reserved for the user 2026-09-19 16:40 -- 2026-09-20 02:40, and is back in the
# pool now. Kept as a default rather than a launch-time GPUS= so a watchdog restart matches
# whatever the pool is supposed to be; to hand a card back, edit this line (via a temp file
# and mv -- a running dispatcher is still reading this script by byte offset).
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
# Order IS the priority, because a queue that never drains still has to produce the most
# important numbers first:
#   arm7_6k   -- its 14 RLBench jobs are already complete and will be skipped, so what runs
#                here is the missing half: ManiSkill.
#   arm0_6k   -- completes the headline paired comparison on both simulators.
#   arm7_4k   -- tab:ablation (b), the 4,000-against-6,000-step budget point. Note the row
#                also wants arm0@4k before it is a clean budget ablation rather than an
#                arm7-only curve; that checkpoint exists (checkpoint-4000 of the arm0 run).
#   official  -- tab:main's matched baseline.
# 2026-09-20: the 6k budget point is abandoned as the headline. arm7@6k finished all 14
# RLBench tasks at 52/280 = 18.6% with half of them a hard zero against a 96.8% GT gate,
# so the whole comparison moves to the 4,000-step checkpoints. The 6k numbers already on
# disk are kept: they are the other end of tab:ablation (b)'s budget row.
MODELS="${MODELS:-arm7_4k arm0_4k official}"
RLB_OUT="$REPO/reports/closedloop_unseen14"
MS_OUT="$REPO/reports/closedloop_maniskill"
LOGS="${LOGS:-$REPO/logs/logs_cl_phase2}"
mkdir -p "$RLB_OUT" "$MS_OUT" "$LOGS"

RLB_TASKS="${RLB_TASKS:-close_box close_drawer close_microwave take_lid_off_saucepan \
toilet_seat_down basketball_in_hoop meat_on_grill light_bulb_out take_item_out_of_drawer \
take_money_out_safe put_rubbish_in_bin stack_cups open_window wipe_desk}"
# Only tasks whose GT-replay gate passes: lift_peg_upright (0/20), peg_insertion_side (1/20)
# and plug_charger (0/20) are harness floors and are reported as such, not run against models.
MS_HELDOUT_TASKS="${MS_HELDOUT_TASKS:-pull_cube place_sphere}"
MS_SEED_TASKS="${MS_SEED_TASKS:-pick_cube stack_cube pull_cube_tool push_cube}"

ckpt_of() { case "$1" in
  arm7_6k)  echo "$REPO/outputs/arm7_joint2src_seed42_fi3_6k/checkpoint-6000/step6000.ckpt" ;;
  arm7_4k)  echo "$REPO/outputs/arm7_joint2src_seed42_fi3_6k/checkpoint-4000/step4000.ckpt" ;;
  arm0_4k)  echo "$REPO/outputs/arm0_joint2src_seed42_fi3_6k/checkpoint-4000/step4000.ckpt" ;;
  arm0_6k)  echo "$REPO/outputs/arm0_joint2src_seed42_fi3_6k/checkpoint-6000/step6000.ckpt" ;;
  official) echo "$REPO/checkpoints/official/step125750.ckpt" ;;
esac; }
cfg_of()      { [ "$1" = official ] && echo 10.0 || echo 7.5; }
rlb_fi_of()   { [ "$1" = official ] && echo 4    || echo 3; }
tagstyle_of() { [ "$1" = official ] && echo none || echo explicit; }

QUEUE="$LOGS/queue.txt"; IDX="$LOGS/queue.idx"
if [ "${FRESH_QUEUE:-1}" = 1 ]; then
  : > "$QUEUE"
  for m in $MODELS; do
    for t in $RLB_TASKS;        do echo "$m rlb $t"        >> "$QUEUE"; done
    for t in $MS_HELDOUT_TASKS; do echo "$m ms_heldout $t" >> "$QUEUE"; done
    for t in $MS_SEED_TASKS;    do echo "$m ms_seeds $t"   >> "$QUEUE"; done
  done
  echo 0 > "$IDX"
fi
TOTAL=$(wc -l < "$QUEUE")
echo "[phase2] $TOTAL jobs x $TRIALS trials, GPUs: $GPUS"

next_job() {
  local n; exec 9>"$LOGS/.queue.lock"; flock 9
  n=$(cat "$IDX"); [ "$n" -ge "$TOTAL" ] && { flock -u 9; return 1; }
  echo $((n + 1)) > "$IDX"; sed -n "$((n + 1))p" "$QUEUE"; flock -u 9
}

job_complete() {
  python - "$1" "$TRIALS" <<'PY' 2>/dev/null
import json, sys
try: d = json.load(open(sys.argv[1]))
except Exception: sys.exit(1)
sys.exit(0 if len(d.get("results", [])) >= int(sys.argv[2]) else 1)
PY
}

worker() {
  local gpu=$1 job model backend task tag json log
  while job=$(next_job); do
    read -r model backend task <<< "$job"
    case "$backend" in
      rlb)        tag="u14_${model}_${task}"; json="$RLB_OUT/rollout_${tag}.json" ;;
      ms_heldout) tag="ms_${model}_${task}";  json="$MS_OUT/rollout_${tag}.json" ;;
      ms_seeds)   tag="msv1_${model}_${task}"; json="$MS_OUT/rollout_${tag}.json" ;;
    esac
    log="$LOGS/${tag}.log"
    if job_complete "$json"; then echo "[gpu$gpu] skip $tag (complete)"; continue; fi
    echo "[gpu$gpu] $(date +%H:%M:%S) start $tag"
    if [ "$backend" = rlb ]; then
      CUDA_VISIBLE_DEVICES="$gpu" xvfb-run -a python -u eval/rollout.py \
        --ckpt "$(ckpt_of "$model")" --tag "$tag" --tasks "$task" \
        --variation 0 --num-trials "$TRIALS" --res 512 --cfg "$(cfg_of "$model")" --steps 50 \
        --frame-interval "$(rlb_fi_of "$model")" --prompt-tag-style "$(tagstyle_of "$model")" \
        --axis-solver sphere --arm-action-mode planning --max-steps-factor 1.5 \
        --skip-anchor-frames 4 --execution-horizon 41 --max-ik-fail-streak 5 \
        --anchor-modality video --seed 42 --record-video --out "$RLB_OUT" > "$log" 2>&1
    else
      local tree var
      if [ "$backend" = ms_heldout ]; then tree="$REPO/data/maniskill3_heldout"; var=0
      else tree="$REPO/data/maniskill3"; var=1; fi
      CUDA_VISIBLE_DEVICES="$gpu" python -u eval/rollout_maniskill.py \
        --ckpt "$(ckpt_of "$model")" --tag "$tag" --tasks "$task" \
        --tree "$tree" --variation "$var" --num-trials "$TRIALS" \
        --res 512 --cfg "$(cfg_of "$model")" --steps 50 --frame-interval 1 \
        --prompt-tag-style "$(tagstyle_of "$model")" --axis-solver sphere \
        --max-steps-factor 1.5 --skip-anchor-frames 4 --execution-horizon 41 \
        --max-ik-fail-streak 5 --seed 42 --record-video --out "$MS_OUT" > "$log" 2>&1
    fi
    echo "[gpu$gpu] $(date +%H:%M:%S) done  $tag rc=$?"
  done
  echo "[gpu$gpu] queue empty"
}

for g in $GPUS; do worker "$g" & sleep 2; done
wait
echo "PHASE2_DONE $(date)"
