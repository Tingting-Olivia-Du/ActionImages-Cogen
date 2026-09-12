#!/bin/bash
# Evaluate ONE checkpoint two ways: what it can GENERATE, and whether it can ACT.
#
#   bash scripts/eval_ckpt.sh <ckpt> <tag> [gpu]
#   bash scripts/eval_ckpt.sh outputs/arm1__seed42_fi3_512_aug_sr/checkpoint-2000/step2000.ckpt arm1_sr_2000 6
#
# Stage 1 (generation, ttd_train env) -- one episode, one instruction, one seed, and ONLY the
# prompt tag varies between panels. Produces, per modality: the model's own RGB prediction
# (segments 0/2 of the sequence) and the modality stream (segments 1/3), as mp4 + a contact
# sheet + metrics.json.
#
# `normal` is included DELIBERATELY as a negative control. arm1's menu is action/depth/seg only,
# so the model has never seen `<normal>`. If its `<normal>` output looks like depth, the tags are
# not separating the tasks; if it looks like nothing in particular, they are. Reading the three
# trained modalities without that control cannot distinguish "the tag works" from "the model
# produces the same thing whatever you ask".
#
# Stage 2 (closed loop, ttd_rollout env) -- receding-horizon rollouts on the live simulator.
# Restricted to tasks whose GROUND-TRUTH replay succeeds on this harness
# (tests/test_rollout_env.py writes reports/closedloop/gt_replay.json); a task that its own demo
# cannot solve would score 0% for the harness's reasons, not the model's.
#
# RESOLUTION AND INTERVAL ARE NOT FREE KNOBS. They must match the checkpoint's training values
# (arm1: 512 / 3). A mismatch is silent -- the loader resizes and the window restrides without
# complaint -- and turns every number below into a measurement of domain shift.
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
export PYTHONPATH="$REPO"

CKPT="${1:?usage: eval_ckpt.sh <ckpt> <tag> [gpu]}"
TAG="${2:?usage: eval_ckpt.sh <ckpt> <tag> [gpu]}"
GPU="${3:-6}"

RES="${RES:-512}"
FI="${FI:-3}"
SEG_MODE="${SEG_MODE:-scene_roles}"
MODALITIES="${MODALITIES:-action,depth,segmentation,normal}"
EPISODE="${EPISODE:-open_drawer/variation0/episodes/episode0}"
TASKS="${TASKS:-push_buttons open_drawer meat_off_grill slide_block_to_target turn_tap}"
TRIALS="${TRIALS:-1}"
CFG="${CFG:-7.5}"
STEPS="${STEPS:-50}"

[ -f "$CKPT" ] || { echo "!! no such checkpoint: $CKPT"; exit 2; }
echo "=============================================================================="
echo "ckpt      : $CKPT"
echo "tag       : $TAG    gpu=$GPU  res=$RES  frame_interval=$FI  seg_mode=$SEG_MODE"
echo "modalities: $MODALITIES     (normal = untrained negative control for arm1)"
echo "tasks     : $TASKS   x $TRIALS trial(s)"
echo "=============================================================================="
nvidia-smi --query-gpu=index,memory.used --format=csv | sed -n "$((GPU+2))p"

# ---------------------------------------------------------------- stage 1: generation
echo; echo "### stage 1/2  four-modality generation"
conda activate ttd_train
CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/modality_demo.py \
  --ckpt "$CKPT" --tag "$TAG" --modalities "$MODALITIES" \
  --episode "$EPISODE" --res "$RES" --frame_interval "$FI" \
  --segmentation_mode "$SEG_MODE" --conditioning iiii \
  --cfg "$CFG" --steps "$STEPS"
GEN_EXIT=$?
echo "stage1 exit=$GEN_EXIT   -> reports/modality_demo/$TAG/"
conda deactivate

# ---------------------------------------------------------------- stage 2: closed loop
echo; echo "### stage 2/2  closed-loop rollout"
GATE="reports/closedloop/gt_replay.json"
if [ -f "$GATE" ]; then
  python - "$GATE" $TASKS <<'PY'
import json, sys
gate = json.load(open(sys.argv[1]))
ok, bad = set(gate.get("measurable_tasks", [])), set(gate.get("unmeasurable_tasks", []))
asked = sys.argv[2:]
unmeasurable = [t for t in asked if t in bad]
unchecked = [t for t in asked if t not in ok and t not in bad]
if unmeasurable:
    print(f"  !! these tasks FAIL their own GT replay on this harness: {unmeasurable}")
    print(f"     a 0% closed-loop rate on them would be the harness's number, not the model's.")
if unchecked:
    print(f"  ?? never GT-replayed, measurability unknown: {unchecked}")
print(f"  gate says measurable: {sorted(ok)}")
PY
else
  echo "  !! $GATE missing -- run tests/test_rollout_env.py first"
fi

source /workspace/ttdu/ttd/scripts/env_eval.rc
conda activate ttd_rollout
CUDA_VISIBLE_DEVICES="$GPU" xvfb-run -a python -u eval/rollout.py \
  --ckpt "$CKPT" --tag "$TAG" --res "$RES" --cfg "$CFG" --steps "$STEPS" \
  --frame-interval "$FI" --prompt-tag-style explicit --axis-solver sphere \
  --arm-action-mode planning --max-steps-factor 1.5 --record-video \
  --tasks $TASKS --num-trials "$TRIALS"
RO_EXIT=$?
echo "stage2 exit=$RO_EXIT   -> reports/closedloop/rollout_${TAG}.json"

echo; echo "EVAL_DONE tag=$TAG  stage1=$GEN_EXIT stage2=$RO_EXIT"
