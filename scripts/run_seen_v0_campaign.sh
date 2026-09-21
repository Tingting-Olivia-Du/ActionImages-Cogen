#!/bin/bash
# Closed-loop on the 16 TRAINED tasks at variation 0 with FRESH SCENE SEEDS.
#
# WHY THIS BLOCK EXISTS. The project has two closed-loop regimes and they sit at opposite ends:
# never-trained TASKS (hard: every checkpoint lands at 18-23%) and trained tasks at
# variation 1/2 (harder still: 12 of the 16 are a hard zero, the whole history is 2/63, 0/30,
# 0/60 ...). Neither separates checkpoints. This is the rung in between, and nothing has ever
# systematically run it: same task, same variation, same instruction -- only the object LAYOUT
# is new, because `trial_seed()` reseeds numpy before `get_demos(live_demos=True)` and RLBench
# samples placements from it. The handful of trials that exist in the archives at this setting
# (push_buttons 3/3, meat_off_grill 1/1, open_drawer 1/1) are all successes, which is the
# reason to expect this rung to have signal rather than floor out.
#
# A CONFOUND TO STATE IN THE PAPER, not to hide: the rollout harness runs VANILLA RLBench --
# fixed cameras, no Colosseum randomization -- while the training tree was rendered WITH it
# (camera pose, table colour/texture, background texture, light colour). So a trained task at a
# new seed changes the layout AND the visual domain at once. Separating them needs
# scripts/colosseum_aug.py wired into rollout_env (bind() + randomize(rng)); until then this
# block answers "new layout, clean domain" and must be labelled as such.
#
#   MODELS="arm7_4k" bash scripts/run_seen_v0_campaign.sh
set -uo pipefail
REPO=/workspace/1228_tingting/ActionImages-Cogen
source /workspace/1228_tingting/ttd/scripts/env_eval_ttd_eval.rc
cd "$REPO"

TRIALS="${TRIALS:-20}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
MODELS="${MODELS:-arm7_4k}"
OUT="${OUT:-$REPO/reports/closedloop_seen_v0}"
LOGS="${LOGS:-$REPO/logs/logs_cl_seen_v0}"
# Filled in from the GT gate: a task whose own demonstration cannot be replayed through this
# controller is a harness floor and is not worth a GPU. close_jar is the known one -- the gate
# scores it 0% with zero IK failures, which is why eval/rollout.py's DEFAULT_TASKS excludes it.
TASKS="${TASKS:-close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer \
place_shape_in_shape_sorter push_buttons put_groceries_in_cupboard put_item_in_drawer \
put_money_in_safe reach_and_drag slide_block_to_target stack_blocks stack_wine \
sweep_to_dustpan turn_tap}"
mkdir -p "$OUT" "$LOGS"

ckpt_of() { case "$1" in
  arm7_4k)  echo "$REPO/outputs/arm7_joint2src_seed42_fi3_6k/checkpoint-4000/step4000.ckpt" ;;
  arm7_6k)  echo "$REPO/outputs/arm7_joint2src_seed42_fi3_6k/checkpoint-6000/step6000.ckpt" ;;
  arm0_4k)  echo "$REPO/outputs/arm0_joint2src_seed42_fi3_6k/checkpoint-4000/step4000.ckpt" ;;
  arm0_6k)  echo "$REPO/outputs/arm0_joint2src_seed42_fi3_6k/checkpoint-6000/step6000.ckpt" ;;
  official) echo "$REPO/checkpoints/official/step125750.ckpt" ;;
esac; }
cfg_of()      { [ "$1" = official ] && echo 10.0 || echo 7.5; }
fi_of()       { [ "$1" = official ] && echo 4    || echo 3; }
tagstyle_of() { [ "$1" = official ] && echo none || echo explicit; }

QUEUE="$LOGS/queue.txt"; IDX="$LOGS/queue.idx"
if [ "${FRESH_QUEUE:-1}" = 1 ]; then
  : > "$QUEUE"; for m in $MODELS; do for t in $TASKS; do echo "$m $t" >> "$QUEUE"; done; done
  echo 0 > "$IDX"
fi
TOTAL=$(wc -l < "$QUEUE")
echo "[seenv0] $TOTAL jobs x $TRIALS trials, GPUs: $GPUS -> $OUT"

next_job() {
  local n; exec 9>"$LOGS/.queue.lock"; flock 9
  n=$(cat "$IDX"); [ "$n" -ge "$TOTAL" ] && { flock -u 9; return 1; }
  echo $((n + 1)) > "$IDX"; sed -n "$((n + 1))p" "$QUEUE"; flock -u 9
}

worker() {
  local gpu=$1 job model task tag json log
  while job=$(next_job); do
    model=${job%% *}; task=${job##* }
    tag="sv0_${model}_${task}"; json="$OUT/rollout_${tag}.json"; log="$LOGS/${tag}.log"
    if python "$REPO/scripts/cl_json_complete.py" "$json" "$TRIALS"; then
      echo "[gpu$gpu] skip $tag (complete)"; continue
    fi
    echo "[gpu$gpu] $(date +%H:%M:%S) start $tag"
    CUDA_VISIBLE_DEVICES="$gpu" xvfb-run -a python -u eval/rollout.py \
      --ckpt "$(ckpt_of "$model")" --tag "$tag" --tasks "$task" \
      --variation 0 --num-trials "$TRIALS" \
      --res 512 --cfg "$(cfg_of "$model")" --steps 50 \
      --frame-interval "$(fi_of "$model")" --prompt-tag-style "$(tagstyle_of "$model")" \
      --axis-solver sphere --arm-action-mode planning --max-steps-factor 1.5 \
      --skip-anchor-frames 4 --execution-horizon 41 --max-ik-fail-streak 5 \
      --anchor-modality video --seed 42 --record-video --out "$OUT" > "$log" 2>&1
    echo "[gpu$gpu] $(date +%H:%M:%S) done  $tag rc=$?"
  done
  echo "[gpu$gpu] queue empty"
}

for g in $GPUS; do worker "$g" & sleep 2; done
wait
echo "SEENV0_DONE $(date)"
