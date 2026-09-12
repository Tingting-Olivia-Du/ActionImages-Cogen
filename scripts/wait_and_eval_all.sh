#!/bin/bash
# Wait for a GPU, then run the FULL eval of one checkpoint: every visual modality + action
# open-loop + the mutual-completion sweep + action closed-loop.
#
#   OUT_DIR=outputs/fusion0__seed42_fi3_512_aug_wide_sr_F1 TAG=wide5000 \
#     bash scripts/wait_and_eval_all.sh
#
# EVERY STAGE WAITS AND RETRIES ON ITS OWN. An earlier version claimed a card once and ran the
# stages back to back; but a stage releases its memory when it exits, so a tenant can take the
# card in the gap before the next one starts and the whole eval dies on a transient. On this host
# every card is routinely held by outside tenants, so that is the common case, not the edge case.
#
# STAGE ORDER IS BY INFORMATION PER GPU-MINUTE, deliberately: the two offline stages are ~10
# min/row and cover all five modalities; closed-loop is hours for one number. If the machine only
# ever frees a card briefly, the cheap high-information rows are the ones that land.
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; export PYTHONPATH="$REPO"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_DIR="${OUT_DIR:?set OUT_DIR to the output dir of the arm}"
TAG="${TAG:?set TAG}"
NEED=${NEED:-31000}
TASKS="${TASKS:-push_buttons open_drawer meat_off_grill}"
TRIALS=${TRIALS:-10}
# ---- GENERALISATION ONLY: T2 and T3. Variation-0 (T1, trained on) is deliberately skipped ----
#
#   T2 unseen-var    wide tree, variation1   -- unseen layout, task WAS trained on
#   T3 unseen-task   separate tree           -- 6 tasks never trained on at all
#
# CONSEQUENCE OF DROPPING T1, stated so nobody reads the numbers as something they are not:
# there is no in-distribution baseline here, so T2/T3 are ABSOLUTE quality, not a degradation
# relative to seen data. What survives is (a) T2 vs T3, which separates layout generalisation
# from task generalisation, and (b) the mutual-completion matrix, which is a WITHIN-episode
# contrast (full_anchor vs absent) and therefore never needed T1 in the first place.
#
# The budget freed by dropping T1 is spent on the axis that is kept: every one of EVAL_PLAN.md's
# 8 paired tasks for T2, and all 6 unseen tasks for T3.
#
# T3 lives in a different data root, so it needs its own invocation (--data), not another episode
# in the list. Its tree carries scene_segments.json for all 72 episodes (checked) but no
# seg_targets.json -- fine, that file is only read under the `referring` protocol and every arm
# here is `scene_roles`.
N_TASK=${N_TASK:-8}          # T2: one variation1 episode per task
N_TASK3=${N_TASK3:-6}        # T3: one episode per never-trained task
DATA_MAIN="${DATA_MAIN:-$REPO/data/rlbench_selfgen_512_aug_wide}"
DATA_UNSEEN="${DATA_UNSEEN:-$REPO/data/rlbench_unseen_tasks_512_aug}"
# EVAL_PLAN.md Sec 3.3's task list, all of which have a variation1 in this tree (verified).
PAIR_TASKS="${PAIR_TASKS:-close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer push_buttons reach_and_drag put_item_in_drawer}"
EP_T2=""; i=0
for t in $PAIR_TASKS; do
  [ "$i" -ge "$N_TASK" ] && break
  EP_T2="$EP_T2,$t/variation1/episodes/episode0"
  i=$((i+1))
done
EP_T2="${EP_T2#,}"
# The unseen-task tree names episodes by RANDOM SEED (episode19627, episode309991, ...), not
# episode0 -- a hardcoded `episode0` silently matches nothing there. Take the first episode of
# each task so the sample spans TASKS rather than several episodes of one task.
EP_T3=""; i=0
for d in $(ls -d "$DATA_UNSEEN"/*/variation0/episodes 2>/dev/null); do
  [ "$i" -ge "$N_TASK3" ] && break
  e=$(ls "$d" 2>/dev/null | head -1)
  [ -n "$e" ] || continue
  EP_T3="$EP_T3,$(echo "$d" | sed "s#^$DATA_UNSEEN/##")/$e"
  i=$((i+1))
done
EP_T3="${EP_T3#,}"
MAX_TRY=${MAX_TRY:-6}
FUSION_TMPL="video+depth+segmentation+normal+action"

# Latest checkpoint, resolved once and PINNED. If the arm were still training, "latest" would mean
# a different file in a later stage and the halves of the report would describe different models.
N=$(ls -d "$OUT_DIR"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
[ -n "$N" ] || { echo "NO_CHECKPOINT in $OUT_DIR"; exit 2; }
CKPT="$OUT_DIR/checkpoint-$N/step$N.ckpt"
[ -s "$CKPT" ] || { echo "MISSING $CKPT"; exit 2; }
echo "$(date +%H:%M:%S) EVAL_TARGET step=$N file=$CKPT"

wait_for_gpu() {   # -> echoes an index once one has NEED MiB free
  while true; do
    local g
    g=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', ' -v n="$NEED" '($3-$2)>n {print $1; exit}')
    [ -n "$g" ] && { echo "$g"; return; }
    sleep 60
  done
}

# run_stage <name> <logfile> <success-test-cmd> <runner-fn>
# The success test is a COMMAND, not an exit code: a stage that dies after writing its report is
# a success, and one that exits 0 having written nothing is not.
run_stage() {
  local name="$1" log="$2" test_cmd="$3" runner="$4" try=1
  while [ "$try" -le "$MAX_TRY" ]; do
    local g; g=$(wait_for_gpu)
    echo "$(date +%H:%M:%S) ${name}_START gpu=$g try=$try"
    "$runner" "$g" "$log"
    if eval "$test_cmd"; then echo "$(date +%H:%M:%S) ${name}_DONE"; return 0; fi
    echo "$(date +%H:%M:%S) ${name}_RETRY try=$try: $(grep -oE '^[A-Za-z_.]*(Error|Exception):.*' "$log" 2>/dev/null | tail -1 | cut -c1-150)"
    try=$((try+1)); sleep 90
  done
  echo "$(date +%H:%M:%S) ${name}_GAVE_UP after $MAX_TRY tries"; return 1
}

# $1=gpu $2=log ; $G_TAG/$G_DATA/$G_EPS set by the caller so one runner serves every tier
grid_runner() {
  python -u scripts/modality_mode_grid.py \
    --ckpt "$CKPT" --tag "$G_TAG" --gpu "$1" --data "$G_DATA" \
    --only "$FUSION_TMPL" --segmentation_mode scene_roles \
    --episode "$G_EPS" --res 512 --frame_interval 3 --steps 50 --cfg 7.5 --seed 42 > "$2" 2>&1
}
oneout_runner() {
  python -u scripts/modality_mode_grid.py \
    --ckpt "$CKPT" --tag "$G_TAG" --gpu "$1" --data "$G_DATA" \
    --only "$FUSION_TMPL" --modes "full_anchor" \
    --withhold "depth,segmentation,normal,video,action" \
    --segmentation_mode scene_roles \
    --episode "$G_EPS" --res 512 --frame_interval 3 --steps 50 --cfg 7.5 --seed 42 > "$2" 2>&1
}
cl_runner() {     # action closed loop, asked through the 10-segment canvas under rgb_only
  # Protocol copied verbatim from scripts/queue_fusion_closedloop.sh. rollout.py defaults to
  # --res 256 --frame-interval 1 while this arm trained at 512/fi=3: leaving the defaults would
  # measure a resolution/horizon mismatch and call it a policy failure.
  CUDA_VISIBLE_DEVICES=$1 xvfb-run -a python -u eval/rollout.py \
    --ckpt "$CKPT" --tag "${TAG}_cl" --anchor-modality fusion --tasks $TASKS \
    --variation 0 --num-trials "$TRIALS" \
    --res 512 --cfg 7.5 --steps 50 --frame-interval 3 \
    --prompt-tag-style explicit --axis-solver sphere --arm-action-mode planning \
    --max-steps-factor 1.5 --skip-anchor-frames 4 --execution-horizon 41 --seed 42 > "$2" 2>&1
}

J() { echo "$REPO/reports/modality_mode_grid/$1/metrics.json"; }

echo "T2 episodes (unseen variation, seen task): $EP_T2"
echo "T3 episodes (never-trained task):          $EP_T3   (root $DATA_UNSEEN)"

# Order is by information per GPU-minute: the grid carries rgb_only (the deployment regime) and
# the one-out sweep is the headline matrix. T2 before T3 because an unseen LAYOUT is the weaker
# ask; if a modality already fails there, its T3 number needs no interpretation.
G_TAG="${TAG}_grid_t2";   G_DATA="$DATA_MAIN";   G_EPS="$EP_T2"
run_stage STAGE1_GRID_T2    "$REPO/logs_eval_${TAG}_grid_t2.txt"    "[ -s '$(J ${TAG}_grid_t2)' ]"    grid_runner
grep -E "\|(full_anchor|rgb_only|rgb_given|policy) |^  (rgb|full|policy|out:)" "$REPO/logs_eval_${TAG}_grid_t2.txt" 2>/dev/null | tail -30 || true

G_TAG="${TAG}_oneout_t2"; G_DATA="$DATA_MAIN";   G_EPS="$EP_T2"
run_stage STAGE1B_ONEOUT_T2 "$REPO/logs_eval_${TAG}_oneout_t2.txt" "[ -s '$(J ${TAG}_oneout_t2)' ]" oneout_runner
grep -E "\|(out:|full_anchor)|^  (full|out:)" "$REPO/logs_eval_${TAG}_oneout_t2.txt" 2>/dev/null | tail -30 || true

if [ -n "$EP_T3" ]; then
  G_TAG="${TAG}_grid_t3";   G_DATA="$DATA_UNSEEN"; G_EPS="$EP_T3"
  run_stage STAGE1C_GRID_T3   "$REPO/logs_eval_${TAG}_grid_t3.txt"   "[ -s '$(J ${TAG}_grid_t3)' ]"   grid_runner
  G_TAG="${TAG}_oneout_t3";  G_DATA="$DATA_UNSEEN"; G_EPS="$EP_T3"
  run_stage STAGE1D_ONEOUT_T3 "$REPO/logs_eval_${TAG}_oneout_t3.txt" "[ -s '$(J ${TAG}_oneout_t3)' ]" oneout_runner
else
  echo "SKIP T3: no episodes resolved under $DATA_UNSEEN"
fi

CL_JSON="$REPO/reports/closedloop/rollout_${TAG}_cl.json"
run_stage STAGE2_CLOSEDLOOP "$REPO/logs_cl_${TAG}.txt" "[ -s '$CL_JSON' ]" cl_runner
if [ -s "$CL_JSON" ]; then
  python - "$CL_JSON" <<'PY'
import json, sys, collections
d = json.load(open(sys.argv[1]))
rs = d.get("rollouts", d if isinstance(d, list) else [])
by = collections.defaultdict(lambda: [0, 0])
for r in rs:
    if not isinstance(r, dict): continue
    t = r.get("task", "?"); by[t][1] += 1; by[t][0] += int(bool(r.get("success")))
tot = [sum(v[0] for v in by.values()), sum(v[1] for v in by.values())]
for t, (s, n) in sorted(by.items()):
    print(f"  CLOSEDLOOP {t:24s} {s}/{n}" + (f"  {100*s/n:.1f}%" if n else ""))
if tot[1]: print(f"  CLOSEDLOOP {'TOTAL':24s} {tot[0]}/{tot[1]}  {100*tot[0]/tot[1]:.1f}%")
print("  errors:", d.get("errors", "n/a") if isinstance(d, dict) else "n/a")
PY
fi
echo "$(date +%H:%M:%S) EVAL_ALL_FINISHED step=$N"
