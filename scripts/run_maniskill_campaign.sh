#!/bin/bash
# Closed-loop campaign on the two ManiSkill3 HELD-OUT tasks (data/maniskill3_heldout), 20
# trials each, for the same three models as the RLBench campaign.
#
# This is the first closed-loop number ManiSkill3 has in this project -- eval/rollout.py is
# RLBench-only, which is why tab:pertask's ManiSkill block reads "no rollout harness". The
# harness is eval/rollout_maniskill.py; its GT-replay gate is what licenses these numbers.
#
# frame_interval is 1 and NOT 3: training/dataset/base.py pins the ManiSkill source to stride
# 1, so a stride-3 rollout would score the checkpoint on a stride it never saw for this source.
#
#   bash scripts/run_maniskill_campaign.sh            # models on GPUs, 20 trials
#   MODELS=gt bash scripts/run_maniskill_campaign.sh  # the GT-replay ceiling (no model)
set -uo pipefail
REPO=/workspace/1228_tingting/ActionImages-Cogen
source /workspace/1228_tingting/ttd/scripts/env_eval_ttd_eval.rc
export VK_ICD_FILENAMES=/workspace/1228_tingting/envs/ttd_eval/lib/python3.10/site-packages/sapien/vulkan_library/nvidia_icd.json
cd "$REPO"

TRIALS="${TRIALS:-20}"
# GPU 0 IS RESERVED for the user's own work from 2026-09-19 16:40 on. The phase-1 wave
# that was already in flight keeps all eight; everything launched after it gets seven.
# Change this default rather than passing GPUS= at launch, so a watchdog restart cannot
# quietly take the card back.
GPUS="${GPUS:-1 2 3 4 5 6 7}"
MODELS="${MODELS:-arm7_6k arm0_6k official}"
TASKS="${TASKS:-pull_cube lift_peg_upright}"
OUT="${OUT:-$REPO/reports/closedloop_maniskill}"
LOGS="${LOGS:-$REPO/logs/logs_cl_maniskill}"
mkdir -p "$OUT" "$LOGS"

ckpt_of() { case "$1" in
  arm7_6k)  echo "$REPO/outputs/arm7_joint2src_seed42_fi3_6k/checkpoint-6000/step6000.ckpt" ;;
  arm0_6k)  echo "$REPO/outputs/arm0_joint2src_seed42_fi3_6k/checkpoint-6000/step6000.ckpt" ;;
  official) echo "$REPO/checkpoints/official/step125750.ckpt" ;;
esac; }
cfg_of()      { [ "$1" = official ] && echo 10.0 || echo 7.5; }
tagstyle_of() { [ "$1" = official ] && echo none || echo explicit; }

QUEUE="$LOGS/queue.txt"; IDX="$LOGS/queue.idx"
if [ "${FRESH_QUEUE:-1}" = 1 ]; then
  : > "$QUEUE"; for m in $MODELS; do for t in $TASKS; do echo "$m $t" >> "$QUEUE"; done; done
  echo 0 > "$IDX"
fi
TOTAL=$(wc -l < "$QUEUE")
echo "[ms-campaign] $TOTAL jobs x $TRIALS trials, GPUs: $GPUS -> $OUT"

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
  local gpu=$1 job model task tag json log extra
  while job=$(next_job); do
    model=${job%% *}; task=${job##* }
    tag="ms_${model}_${task}"; json="$OUT/rollout_${tag}.json"; log="$LOGS/${tag}.log"
    if job_complete "$json"; then echo "[gpu$gpu] skip $tag (complete)"; continue; fi
    extra=""
    [ "$model" = gt ] && extra="--gt-replay" || extra="--ckpt $(ckpt_of "$model")"
    echo "[gpu$gpu] $(date +%H:%M:%S) start $tag"
    CUDA_VISIBLE_DEVICES="$gpu" python -u eval/rollout_maniskill.py \
      $extra --tag "$tag" --tasks "$task" --num-trials "$TRIALS" \
      --res 512 --cfg "$(cfg_of "$model")" --steps 50 --frame-interval 1 \
      --prompt-tag-style "$(tagstyle_of "$model")" --axis-solver sphere \
      --max-steps-factor 1.5 --skip-anchor-frames 4 --execution-horizon 41 \
      --max-ik-fail-streak 5 --seed 42 --record-video --out "$OUT" > "$log" 2>&1
    echo "[gpu$gpu] $(date +%H:%M:%S) done  $tag rc=$?"
  done
  echo "[gpu$gpu] queue empty"
}

for g in $GPUS; do worker "$g" & sleep 2; done
wait
echo "MS_CAMPAIGN_DONE $(date)"
