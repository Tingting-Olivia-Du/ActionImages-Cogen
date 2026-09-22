#!/bin/bash
# Closed-loop ManiSkill3 evaluation of the ManiSkill-ONLY pair (both arms warm-started from the
# released checkpoint, ACTION_MASK_MIX=A1, 4000 steps; scripts/launch_ms_fresh.sh).
#
# Same six tasks and the same protocol as the ManiSkill block of scripts/run_cl_phase2.sh, so
# these cells pair trial-for-trial with the joint (RLBench+ManiSkill) arm7 numbers already on
# disk: the scene of (task, trial) is the stored episode `trial`, identical for every model.
#   held-out TASKS         pull_cube  place_sphere            data/maniskill3_heldout  var 0
#   trained tasks, new seeds  pick_cube stack_cube pull_cube_tool push_cube  data/maniskill3  var 1
# lift_peg_upright, peg_insertion_side and plug_charger are harness floors (GT replay 0-1/20)
# and are not run against models.
#
# ORDER: every 2k job before any 4k job (the 2k read comes first, as asked); within a step the
# seen-task seed block before the unseen tasks; and within a block
# arm7 and arm0 alternate task by task so the paired comparison fills in together rather than
# one model finishing hours ahead of the other.
#
# TRIALS defaults to 10 for a first look. eval/rollout_maniskill.py resumes a tag, so rerunning
# with TRIALS=20 tops the same files up to 20 instead of starting over.
set -uo pipefail
REPO=/workspace/1228_tingting/ActionImages-Cogen
source /workspace/1228_tingting/ttd/scripts/env_eval_ttd_eval.rc
export VK_ICD_FILENAMES=/workspace/1228_tingting/envs/ttd_eval/lib/python3.10/site-packages/sapien/vulkan_library/nvidia_icd.json
cd "$REPO"
TRIALS="${TRIALS:-10}"
GPUS="${GPUS:-0 1 2 3}"
STEPS_LIST="${STEPS_LIST:-2000 4000}"
OUT="$REPO/reports/closedloop_msonly"
LOGS="$REPO/logs/logs_cl_msonly"; mkdir -p "$OUT" "$LOGS"

QUEUE="$LOGS/queue.txt"; IDX="$LOGS/queue.idx"
if [ "${FRESH_QUEUE:-1}" = 1 ]; then
  : > "$QUEUE"
  for st in $STEPS_LIST; do
    # Seen-task seed variations FIRST, unseen tasks second (as asked): the trained tasks at new
    # seeds are the nearer-transfer question, and both arms spent their whole budget on them.
    for spec in "seeds pick_cube" "seeds stack_cube" "seeds push_cube" "seeds pull_cube_tool" \
                "heldout pull_cube" "heldout place_sphere"; do
      for arm in arm7 arm0; do echo "$arm $st $spec" >> "$QUEUE"; done
    done
  done
  echo 0 > "$IDX"
fi
TOTAL=$(wc -l < "$QUEUE")
echo "[ms-only] $TOTAL jobs x $TRIALS trials, GPUs: $GPUS -> $OUT"

next_job() {
  local n; exec 9>"$LOGS/.queue.lock"; flock 9
  n=$(cat "$IDX"); [ "$n" -ge "$TOTAL" ] && { flock -u 9; return 1; }
  echo $((n + 1)) > "$IDX"; sed -n "$((n + 1))p" "$QUEUE"; flock -u 9
}

worker() {
  local gpu=$1 job arm st blk task tag tree var ck
  while job=$(next_job); do
    read -r arm st blk task <<< "$job"
    k=$((st / 1000))k
    tag="mso_${arm}_${k}_${task}"
    if python "$REPO/scripts/cl_json_complete.py" "$OUT/rollout_${tag}.json" "$TRIALS"; then
      echo "[gpu$gpu] skip $tag (complete)"; continue
    fi
    if [ "$blk" = heldout ]; then tree="$REPO/data/maniskill3_heldout"; var=0
    else tree="$REPO/data/maniskill3"; var=1; fi
    ck="$REPO/outputs/${arm}_msonly_fromofficial_a1_4k/checkpoint-${st}/step${st}.ckpt"
    echo "[gpu$gpu] $(date +%H:%M:%S) start $tag"
    CUDA_VISIBLE_DEVICES="$gpu" python -u eval/rollout_maniskill.py \
      --ckpt "$ck" --tag "$tag" --tasks "$task" --tree "$tree" --variation "$var" \
      --num-trials "$TRIALS" --res 512 --cfg 7.5 --steps 50 --frame-interval 1 \
      --prompt-tag-style explicit --axis-solver sphere --max-steps-factor 1.5 \
      --skip-anchor-frames 4 --execution-horizon 41 --max-ik-fail-streak 5 --seed 42 \
      --record-video --out "$OUT" >> "$LOGS/${tag}.log" 2>&1
    echo "[gpu$gpu] $(date +%H:%M:%S) done  $tag rc=$?"
  done
  echo "[gpu$gpu] queue empty"
}
for g in $GPUS; do worker "$g" & sleep 2; done
wait
echo "MS_ONLY_EVAL_DONE $(date)"
