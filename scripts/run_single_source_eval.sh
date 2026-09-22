#!/bin/bash
# Closed-loop evaluation of the single-source runs (scripts/launch_single_source.sh).
#
#   bash scripts/run_single_source_eval.sh gt       # harness ceilings: no model, run FIRST
#   bash scripts/run_single_source_eval.sh models   # the six checkpoints, one GPU per job
#   DRY_RUN=1 bash scripts/run_single_source_eval.sh models   # print the queue and exit
#
# Each model is scored in ITS OWN simulator (an rlbench-only run on RLBench, a libero-only run
# on LIBERO), 20 trials per task.
#   RLBench  close_box close_drawer close_microwave toilet_seat_down meat_on_grill
#            (all absent from the RLBench training tree; scenes generated live, no data needed)
#   LIBERO   LIB_SPECS, "suite:task_index" pairs; default one task per suite (index 0).
#            Trials are the benchmark's own FIXED init states 0..TRIALS-1, which are not the
#            demonstrations' initial states -- the standard LIBERO protocol.
#
# LIBERO HAS NO CLOSED-LOOP HARNESS YET. The LIBERO branch calls eval/rollout_libero.py with the
# CLI contract below; until that file exists it prints where to start and skips LIBERO, so the
# RLBench half still runs. See the guide (separate_train_eval.md, Sec. 5.3) for the spec.
#
# Within a task the models alternate, so the paired comparison fills in together. The scene of
# (task, trial) is identical for every model -- RLBench seeds it from blake2b(task|var|trial),
# LIBERO loads fixed init state `trial` -- which is what licenses a paired McNemar test.
set -uo pipefail
MODE="${1:?gt|models}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 2
[ -n "${ENV_RC:-}" ] && source "$ENV_RC"
unset CUDA_VISIBLE_DEVICES
export MUJOCO_GL="${MUJOCO_GL:-egl}"   # LIBERO offscreen rendering
TRIALS="${TRIALS:-20}"
# RLBench runs stop at 4k. LIBERO runs go to 10k; evaluate the step the held-out validation
# picked (scripts/run_libero_val.sh), the SAME step for all three arms.
RLB_STEP="${RLB_STEP:-4000}"; LIB_STEP="${LIB_STEP:-10000}"; GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
ARMS="${ARMS:-arm1 arm2 arm3}"; SOURCES="${SOURCES:-rlbench libero}"
OUT="$REPO/reports/closedloop_single_source"; LOGS="$REPO/logs/logs_single_source_eval"
mkdir -p "$OUT" "$LOGS"
RLB_TASKS="close_box close_drawer close_microwave toilet_seat_down meat_on_grill"
LIB_SPECS="${LIB_SPECS:-libero_spatial:0 libero_object:0 libero_goal:0 libero_10:0}"
HAVE_LIBERO_HARNESS=0; [ -f eval/rollout_libero.py ] && HAVE_LIBERO_HARNESS=1

rlb_cmd() {  # rlb_cmd <tag> <task> [--ckpt X | --gt-replay]
  local tag=$1 task=$2; shift 2
  xvfb-run -a python -u eval/rollout.py "$@" --tag "$tag" --tasks "$task" --variation 0 \
    --num-trials "$TRIALS" --res 512 --cfg 7.5 --steps 50 --frame-interval 3 \
    --prompt-tag-style explicit --axis-solver sphere --arm-action-mode planning \
    --max-steps-factor 1.5 --skip-anchor-frames 4 --execution-horizon 41 \
    --max-ik-fail-streak 5 --anchor-modality video --seed 42 --out "$OUT"
}
lib_cmd() {  # lib_cmd <tag> <suite> <task_index> [--ckpt X | --gt-replay]
  # CLI CONTRACT for eval/rollout_libero.py (to be written; mirror eval/rollout_maniskill.py).
  local tag=$1 suite=$2 ti=$3; shift 3
  python -u eval/rollout_libero.py "$@" --tag "$tag" --suite "$suite" --task-index "$ti" \
    --num-trials "$TRIALS" --res 512 --cfg 7.5 --steps 50 --frame-interval 1 \
    --prompt-tag-style explicit --axis-solver sphere --skip-anchor-frames 4 \
    --execution-horizon 41 --max-ik-fail-streak 5 --seed 42 --out "$OUT"
}
no_harness() {
  echo "!! eval/rollout_libero.py does not exist yet -- skipping LIBERO. Write it to the spec in"
  echo "   separate_train_eval.md Sec. 5.3 (start from eval/rollout_maniskill.py + scripts/libero_gen.py)."
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
  if [ "$HAVE_LIBERO_HARNESS" = 1 ]; then
    for spec in $LIB_SPECS; do
      su=${spec%%:*}; ti=${spec##*:}; complete "ssgt_libero_${su}_t$ti" && continue
      # GT replay on LIBERO replays the tree's own recorded TCP poses through the same absolute
      # controller the model drives -- it validates the controller conventions (Sec. 5.3).
      CUDA_VISIBLE_DEVICES="${GPUS%% *}" lib_cmd "ssgt_libero_${su}_t$ti" "$su" "$ti" --gt-replay \
        --tree "$REPO/data/$su" --no-record-video > "$LOGS/ssgt_libero_${su}_t$ti.log" 2>&1
    done
  else no_harness; fi
  wait; echo "GT_DONE"; exit 0
fi

# ---- models: build the queue
QUEUE="$LOGS/queue_r${RLB_STEP}_l${LIB_STEP}.txt"; IDX="$LOGS/queue_r${RLB_STEP}_l${LIB_STEP}.idx"
if [ "${FRESH_QUEUE:-1}" = 1 ]; then
  : > "$QUEUE"
  rl=($RLB_TASKS); lb=($LIB_SPECS); n=$(( ${#rl[@]} > ${#lb[@]} ? ${#rl[@]} : ${#lb[@]} ))
  for ((i = 0; i < n; i++)); do
    for src in $SOURCES; do
      if [ "$src" = rlbench ] && [ $i -lt ${#rl[@]} ]; then
        for a in $ARMS; do echo "$a rlbench - ${rl[$i]}" >> "$QUEUE"; done
      elif [ "$src" = libero ] && [ $i -lt ${#lb[@]} ]; then
        for a in $ARMS; do echo "$a libero ${lb[$i]%%:*} ${lb[$i]##*:}" >> "$QUEUE"; done
      fi
    done
  done
  echo 0 > "$IDX"
fi
TOTAL=$(wc -l < "$QUEUE")
if [ "${DRY_RUN:-0}" = 1 ]; then nl "$QUEUE"; echo "($TOTAL jobs x $TRIALS trials, RLBench step $RLB_STEP, LIBERO step $LIB_STEP)"; exit 0; fi
echo "[single-source eval] $TOTAL jobs x $TRIALS trials, RLBench step $RLB_STEP, LIBERO step $LIB_STEP, GPUs: $GPUS -> $OUT"

next_job() {
  local k; exec 9>"$LOGS/.lock"; flock 9
  k=$(cat "$IDX"); [ "$k" -ge "$TOTAL" ] && { flock -u 9; return 1; }
  echo $((k + 1)) > "$IDX"; sed -n "$((k + 1))p" "$QUEUE"; flock -u 9
}
worker() {
  local gpu=$1 job a src blk t tag ck
  while job=$(next_job); do
    read -r a src blk t <<< "$job"
    if [ "$src" = libero ]; then
      st=$LIB_STEP; run="${a}_liberoonly_fromofficial_a1_10k"
      tag="ss_${a}_libero_$((st / 1000))k_${blk}_t${t}"
      [ "$HAVE_LIBERO_HARNESS" = 1 ] || { echo "[gpu$gpu] skip $tag (no LIBERO harness)"; continue; }
    else
      st=$RLB_STEP; run="${a}_rlbenchonly_fromofficial_a1_4k"
      tag="ss_${a}_${src}_$((st / 1000))k_${t}"
    fi
    complete "$tag" && { echo "[gpu$gpu] skip $tag (complete)"; continue; }
    ck="$REPO/outputs/$run/checkpoint-${st}/step${st}.ckpt"
    [ -f "$ck" ] || { echo "[gpu$gpu] MISSING $ck -- skipping $tag"; continue; }
    echo "[gpu$gpu] $(date +%H:%M:%S) start $tag"
    if [ "$src" = rlbench ]; then
      CUDA_VISIBLE_DEVICES="$gpu" rlb_cmd "$tag" "$t" --ckpt "$ck" --record-video >> "$LOGS/$tag.log" 2>&1
    else
      CUDA_VISIBLE_DEVICES="$gpu" lib_cmd "$tag" "$blk" "$t" --ckpt "$ck" --record-video >> "$LOGS/$tag.log" 2>&1
    fi
    echo "[gpu$gpu] $(date +%H:%M:%S) done  $tag rc=$?"
  done
}
for g in $GPUS; do worker "$g" & sleep 2; done
wait; echo "SINGLE_SOURCE_EVAL_DONE $(date)"
