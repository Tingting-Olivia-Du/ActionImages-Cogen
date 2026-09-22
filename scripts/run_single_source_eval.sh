#!/bin/bash
# Closed-loop evaluation of the single-source runs (scripts/launch_single_source.sh).
#
#   bash scripts/run_single_source_eval.sh gt       # harness ceilings: no model, run FIRST
#   bash scripts/run_single_source_eval.sh models   # the six checkpoints, one GPU per job
#   DRY_RUN=1 bash scripts/run_single_source_eval.sh models   # print the queue and exit
#
# Each model is scored in ITS OWN simulator (an rlbench-only run on RLBench, a maniskill-only run
# on ManiSkill), 20 trials per task, with exactly the protocol of the origin machine's campaigns:
#   RLBench    close_box close_drawer close_microwave toilet_seat_down meat_on_grill
#              (all absent from the RLBench training tree; scenes generated live, no data needed)
#   ManiSkill  unseen TASKS first:  pull_cube place_sphere lift_peg_upright   (maniskill3_heldout, var 0)
#              then seen tasks at NEW SEEDS: pick_cube stack_cube push_cube pull_cube_tool
#                                                                       (maniskill3, variation 1)
# Within a task the models alternate, so the paired comparison fills in together. The scene of
# (task, trial) is identical for every model -- RLBench seeds it from blake2b(task|var|trial),
# ManiSkill replays stored episode `trial` -- which is what licenses a paired McNemar test.
set -uo pipefail
MODE="${1:?gt|models}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 2
[ -n "${ENV_RC:-}" ] && source "$ENV_RC"
unset CUDA_VISIBLE_DEVICES
if [ -z "${VK_ICD_FILENAMES:-}" ]; then
  icd=$(python -c "import sapien,os;print(os.path.join(os.path.dirname(sapien.__file__),'vulkan_library','nvidia_icd.json'))" 2>/dev/null)
  [ -f "$icd" ] && export VK_ICD_FILENAMES="$icd"
fi
TRIALS="${TRIALS:-20}"; STEP="${STEP:-4000}"; GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
ARMS="${ARMS:-arm1 arm2 arm3}"; SOURCES="${SOURCES:-rlbench maniskill}"
OUT="$REPO/reports/closedloop_single_source"; LOGS="$REPO/logs/logs_single_source_eval"
mkdir -p "$OUT" "$LOGS"
RLB_TASKS="close_box close_drawer close_microwave toilet_seat_down meat_on_grill"
MS_SPECS="heldout:pull_cube heldout:place_sphere heldout:lift_peg_upright \
seeds:pick_cube seeds:stack_cube seeds:push_cube seeds:pull_cube_tool"

rlb_cmd() {  # rlb_cmd <tag> <task> [--ckpt X | --gt-replay]
  local tag=$1 task=$2; shift 2
  xvfb-run -a python -u eval/rollout.py "$@" --tag "$tag" --tasks "$task" --variation 0 \
    --num-trials "$TRIALS" --res 512 --cfg 7.5 --steps 50 --frame-interval 3 \
    --prompt-tag-style explicit --axis-solver sphere --arm-action-mode planning \
    --max-steps-factor 1.5 --skip-anchor-frames 4 --execution-horizon 41 \
    --max-ik-fail-streak 5 --anchor-modality video --seed 42 --out "$OUT"
}
ms_cmd() {   # ms_cmd <tag> <blk> <task> [--ckpt X | --gt-replay]
  local tag=$1 blk=$2 task=$3 tree var; shift 3
  if [ "$blk" = heldout ]; then tree="$REPO/data/maniskill3_heldout"; var=0
  else tree="$REPO/data/maniskill3"; var=1; fi
  python -u eval/rollout_maniskill.py "$@" --tag "$tag" --tasks "$task" --tree "$tree" \
    --variation "$var" --num-trials "$TRIALS" --res 512 --cfg 7.5 --steps 50 \
    --frame-interval 1 --prompt-tag-style explicit --axis-solver sphere \
    --max-steps-factor 1.5 --skip-anchor-frames 4 --execution-horizon 41 \
    --max-ik-fail-streak 5 --seed 42 --out "$OUT"
}
complete() { python "$REPO/scripts/cl_json_complete.py" "$OUT/rollout_$1.json" "$TRIALS"; }

if [ "$MODE" = gt ]; then
  # GT replay: each scene's own demonstration through the SAME controller. A task near zero here
  # is a limit of the actuation harness, not of any model -- read every model cell against it.
  # RLBench needs no GPU (one CoppeliaSim per task, in parallel); ManiSkill needs a sliver of one.
  for t in $RLB_TASKS; do
    complete "ssgt_rlbench_$t" && continue
    rlb_cmd "ssgt_rlbench_$t" "$t" --gt-replay --no-record-video > "$LOGS/ssgt_rlbench_$t.log" 2>&1 &
    sleep 2
  done
  for spec in $MS_SPECS; do
    blk=${spec%%:*}; t=${spec##*:}; complete "ssgt_maniskill_$t" && continue
    CUDA_VISIBLE_DEVICES="${GPUS%% *}" ms_cmd "ssgt_maniskill_$t" "$blk" "$t" --gt-replay \
      --no-record-video > "$LOGS/ssgt_maniskill_$t.log" 2>&1
  done
  wait; echo "GT_DONE"; exit 0
fi

# ---- models: build the queue
QUEUE="$LOGS/queue_${STEP}.txt"; IDX="$LOGS/queue_${STEP}.idx"
if [ "${FRESH_QUEUE:-1}" = 1 ]; then
  : > "$QUEUE"
  rl=($RLB_TASKS); ms=($MS_SPECS); n=$(( ${#rl[@]} > ${#ms[@]} ? ${#rl[@]} : ${#ms[@]} ))
  for ((i = 0; i < n; i++)); do
    for src in $SOURCES; do
      if [ "$src" = rlbench ] && [ $i -lt ${#rl[@]} ]; then
        for a in $ARMS; do echo "$a rlbench - ${rl[$i]}" >> "$QUEUE"; done
      elif [ "$src" = maniskill ] && [ $i -lt ${#ms[@]} ]; then
        for a in $ARMS; do echo "$a maniskill ${ms[$i]%%:*} ${ms[$i]##*:}" >> "$QUEUE"; done
      fi
    done
  done
  echo 0 > "$IDX"
fi
TOTAL=$(wc -l < "$QUEUE")
if [ "${DRY_RUN:-0}" = 1 ]; then nl "$QUEUE"; echo "($TOTAL jobs x $TRIALS trials, step $STEP)"; exit 0; fi
echo "[single-source eval] $TOTAL jobs x $TRIALS trials, step $STEP, GPUs: $GPUS -> $OUT"

next_job() {
  local k; exec 9>"$LOGS/.lock"; flock 9
  k=$(cat "$IDX"); [ "$k" -ge "$TOTAL" ] && { flock -u 9; return 1; }
  echo $((k + 1)) > "$IDX"; sed -n "$((k + 1))p" "$QUEUE"; flock -u 9
}
worker() {
  local gpu=$1 job a src blk t tag ck
  while job=$(next_job); do
    read -r a src blk t <<< "$job"
    tag="ss_${a}_${src}_$((STEP / 1000))k_${t}"
    complete "$tag" && { echo "[gpu$gpu] skip $tag (complete)"; continue; }
    ck="$REPO/outputs/${a}_${src}only_fromofficial_a1_4k/checkpoint-${STEP}/step${STEP}.ckpt"
    [ -f "$ck" ] || { echo "[gpu$gpu] MISSING $ck -- skipping $tag"; continue; }
    echo "[gpu$gpu] $(date +%H:%M:%S) start $tag"
    if [ "$src" = rlbench ]; then
      CUDA_VISIBLE_DEVICES="$gpu" rlb_cmd "$tag" "$t" --ckpt "$ck" --record-video >> "$LOGS/$tag.log" 2>&1
    else
      CUDA_VISIBLE_DEVICES="$gpu" ms_cmd "$tag" "$blk" "$t" --ckpt "$ck" --record-video >> "$LOGS/$tag.log" 2>&1
    fi
    echo "[gpu$gpu] $(date +%H:%M:%S) done  $tag rc=$?"
  done
}
for g in $GPUS; do worker "$g" & sleep 2; done
wait; echo "SINGLE_SOURCE_EVAL_DONE $(date)"
