#!/bin/bash
# Closed-loop campaign on the 14 never-trained RLBench tasks (the tree in
# data/rlbench_unseen_tasks_512_*), 20 trials each, for three models.
#
# WHY A QUEUE AND NOT A FIXED SHARDING. Per-trial cost spans 5x across these tasks (close_drawer
# 306 s, close_box 564 s measured over the 600-trial 2026-09-15 campaign), so a round-robin
# assignment of 42 task-jobs to 8 GPUs leaves cards idle for hours. Workers pull the next job
# under flock instead, and the queue is ordered by priority: every arm7 task first, then arm0,
# then the released checkpoint -- so the number the paper needs most is complete first even if
# the campaign is killed halfway.
#
# PROTOCOL. Identical to the 2026-09-15 20-trial campaign (cfg 7.5, fi 3, explicit tags, sphere
# solver, planning, factor 1.5, skip-anchor 4, ik-streak 5, seed 42, anchor video), which is
# what makes these numbers comparable with everything already in eval_doc/results.md. The
# released checkpoint runs under ITS OWN protocol (cfg 10.0, fi 4, no tags) because that is
# what it was trained with -- same exception as scripts/run_official_closedloop_postfix.sh.
#
#   bash scripts/run_unseen14_campaign.sh            # all 8 GPUs, resumes
#   MODELS="arm7_6k" bash scripts/run_unseen14_campaign.sh
set -uo pipefail
REPO=/workspace/1228_tingting/ActionImages-Cogen
source /workspace/1228_tingting/ttd/scripts/env_eval_ttd_eval.rc
cd "$REPO"

TRIALS="${TRIALS:-20}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
MODELS="${MODELS:-arm7_6k arm0_6k official}"
OUT="${OUT:-$REPO/reports/closedloop_unseen14}"
LOGS="${LOGS:-$REPO/logs/logs_cl_unseen14}"
TASKS="${TASKS:-close_box close_drawer close_microwave take_lid_off_saucepan toilet_seat_down \
basketball_in_hoop meat_on_grill light_bulb_out take_item_out_of_drawer take_money_out_safe \
put_rubbish_in_bin stack_cups open_window wipe_desk}"
mkdir -p "$OUT" "$LOGS"

ckpt_of() { case "$1" in
  arm7_6k)  echo "$REPO/outputs/arm7_joint2src_seed42_fi3_6k/checkpoint-6000/step6000.ckpt" ;;
  arm0_6k)  echo "$REPO/outputs/arm0_joint2src_seed42_fi3_6k/checkpoint-6000/step6000.ckpt" ;;
  official) echo "$REPO/checkpoints/official/step125750.ckpt" ;;
esac; }
cfg_of()      { [ "$1" = official ] && echo 10.0 || echo 7.5; }
fi_of()       { [ "$1" = official ] && echo 4    || echo 3; }
tagstyle_of() { [ "$1" = official ] && echo none || echo explicit; }

QUEUE="$LOGS/queue.txt"; IDX="$LOGS/queue.idx"
if [ "${FRESH_QUEUE:-1}" = 1 ]; then
  : > "$QUEUE"
  for m in $MODELS; do for t in $TASKS; do echo "$m $t" >> "$QUEUE"; done; done
  echo 0 > "$IDX"
fi
TOTAL=$(wc -l < "$QUEUE")
echo "[campaign] $TOTAL jobs x $TRIALS trials, GPUs: $GPUS -> $OUT"

next_job() {
  local n
  exec 9>"$LOGS/.queue.lock"
  flock 9
  n=$(cat "$IDX")
  if [ "$n" -ge "$TOTAL" ]; then flock -u 9; return 1; fi
  echo $((n + 1)) > "$IDX"
  sed -n "$((n + 1))p" "$QUEUE"
  flock -u 9
}

# A job is done when its JSON carries $TRIALS scored results. Anything short (a kill, a crash)
# is rerun from scratch rather than topped up: rollout.py has no partial-resume and a half file
# would silently shrink the denominator.
job_complete() {
  python - "$1" "$TRIALS" <<'PY' 2>/dev/null
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
sys.exit(0 if len(d.get("results", [])) >= int(sys.argv[2]) else 1)
PY
}

worker() {
  local gpu=$1 job model task tag json log
  while job=$(next_job); do
    model=${job%% *}; task=${job##* }
    tag="u14_${model}_${task}"; json="$OUT/rollout_${tag}.json"; log="$LOGS/${tag}.log"
    if job_complete "$json"; then echo "[gpu$gpu] skip $tag (complete)"; continue; fi
    echo "[gpu$gpu] $(date +%H:%M:%S) start $tag"
    CUDA_VISIBLE_DEVICES="$gpu" xvfb-run -a python -u eval/rollout.py \
      --ckpt "$(ckpt_of "$model")" --tag "$tag" --tasks "$task" \
      --variation 0 --num-trials "$TRIALS" \
      --res 512 --cfg "$(cfg_of "$model")" --steps 50 \
      --frame-interval "$(fi_of "$model")" --prompt-tag-style "$(tagstyle_of "$model")" \
      --axis-solver sphere --arm-action-mode planning --max-steps-factor 1.5 \
      --skip-anchor-frames 4 --execution-horizon 41 --max-ik-fail-streak 5 \
      --anchor-modality video --seed 42 --record-video --out "$OUT" \
      > "$log" 2>&1
    echo "[gpu$gpu] $(date +%H:%M:%S) done  $tag rc=$? $(grep -c 'success=True' "$log" 2>/dev/null)/$TRIALS"
  done
  echo "[gpu$gpu] queue empty, exiting"
}

for g in $GPUS; do worker "$g" & sleep 2; done
wait
echo "CAMPAIGN_DONE $(date)"
