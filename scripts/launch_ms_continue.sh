#!/bin/bash
# Continue arm7 and arm0 on ManiSkill3 ALONE for 4,000 more steps, two GPUs each.
#
# WHAT THIS ASKS. Both arms were trained on rlbench@0.7 + maniskill3@0.3 and both are near zero
# on ManiSkill closed-loop -- including pick_cube and stack_cube, tasks they WERE trained on
# with only the scene seed changed (0/20 each against a 20/20 GT ceiling). This run gives the
# ManiSkill source the whole budget instead of 30% of it, so "not enough ManiSkill exposure"
# stops being an explanation for that.
#
# THE PARAMETERS, and which of them are not free choices:
#
#   OUT is a NEW directory, warm-started from each run's step6000.ckpt, NOT a resume in place.
#     outputs/arm{7,0}_joint2src_seed42_fi3_6k hold checkpoint-4000 and -6000, which are the
#     artifacts the closed-loop campaigns are scoring right now. Writing steps 6001-10000 into
#     the same tree would leave "which checkpoint is which" ambiguous for every report already
#     referring to them. A fresh directory keeps the evaluated artifacts frozen.
#
#   GRAD ACCUM 2 -- the single most important line here. The original arms ran 4 GPUs x bs1 x
#     accum1 = effective batch 4. On two GPUs, accum 1 would give batch 2, and every difference
#     this run produces would confound the data change with a halved batch. accum 2 restores
#     batch 4 exactly.
#
#   WARMUP 200, not the script's 1000. 1000 is sized for a warm start from a checkpoint trained
#     on other data; this continues from the arm's own weights for only 4000 steps, so 1000
#     would spend a quarter of the run at reduced LR. 200 is enough to absorb the shock of
#     restarting Adam with fresh moments (SAVE_OPTIM=False, as asked).
#
#   LR 5e-7 UNCHANGED. Tempting to raise it on a smaller source, but exposure goes the other
#     way: 4000 steps x batch 4 = 16,000 samples over 1,750 episodes is 9.1 draws/episode,
#     against 4.11 in the original mix. This run already sees each episode twice as often, so
#     the overfitting risk argues for holding the rate, not raising it.
#
#   ACTION_MASK_MIX AND SEG_MODE ARE PINNED PER ARM to what each originally trained under:
#     arm7 = A1 + scene_roles, arm0 = A0 + referring (outputs/joint_launch_logs/*.log). The
#     script's current default would give arm0 A1, which would make this a different experiment
#     wearing the same name. This is also the confound tab:ablation (a) already names.
#
#   FRAME_INTERVAL=1 because training/dataset/base.py PINS the ManiSkill source to stride 1
#     regardless of this flag; passing 3 only makes the startup banner claim a 6.0 s window
#     when the real one is 2.05 s.
#
#   CKPT_EVERY=1000, SAVE_TOP_K=-1, SAVE_OPTIM=False -> 4 x 12.8 GB per arm, ~103 GB for both,
#     against 438 GB free. Keeping all four matters: if this helps, the interesting question is
#     immediately "at which step", and a coarser cadence cannot answer it.
set -uo pipefail
REPO=/workspace/1228_tingting/ActionImages-Cogen
# train_arm.sh does NOT activate an environment -- it runs whatever `python` is on PATH. The
# first attempt at this run died at startup with `ModuleNotFoundError: No module named
# 'transformers'` because that was the base conda python. Source the rc so the launcher does
# not depend on the caller's shell having done it.
source /workspace/1228_tingting/ttd/scripts/env_eval_ttd_eval.rc
export TTD_ENV="${TTD_ENV:-/workspace/1228_tingting/.env}"
# Preflight, run FROM THE REPO ROOT because that is train.py's cwd and the repo contains a
# stray `wandb/` directory that shadowed the real package as a namespace package -- an earlier
# check from a different directory reported wandb present while the run died on
# `WandbCallback requires wandb to be installed`. Ask the question exactly the way transformers
# asks it, not an approximation of it.
cd "$REPO" || exit 2
python - <<'PY' || { echo "!! preflight failed -- not launching"; exit 2; }
import sys
import transformers, deepspeed, diffsynth  # noqa: F401
from transformers.integrations import is_wandb_available
print(f"preflight OK: transformers {transformers.__version__}, deepspeed {deepspeed.__version__}, "
      f"wandb_available={is_wandb_available()}")
PY
LOGDIR=$REPO/outputs/ms_continue_logs
mkdir -p "$LOGDIR"

launch() {  # launch <arm> <gpus> <port> <action_mask_mix> <seg_mode>
  local arm=$1 gpus=$2 port=$3 amm=$4 seg=$5
  DATASET="maniskill3" VARIATIONS=0 FRAME_INTERVAL=1 RES=512 \
  GPUS="$gpus" PORT="$port" SEED=42 STEPS=4000 CKPT_EVERY=1000 SAVE_TOP_K=-1 \
  SAVE_OPTIM=False ACTION_MASK_MIX="$amm" SEG_MODE="$seg" \
  INIT_CKPT="$REPO/outputs/${arm}_joint2src_seed42_fi3_6k/checkpoint-6000/step6000.ckpt" \
  OUT="$REPO/outputs/${arm}_msonly_from6k_4k" \
  EXTRA_ARGS="--gradient_accumulation_steps 2 --warmup_steps 200 --allow_step_restart --report_to none" \
    setsid nohup bash "$REPO/scripts/train_arm.sh" "$arm" \
    > "$LOGDIR/${arm}_msonly.log" 2>&1 < /dev/null &
  echo "launched $arm on GPUs $gpus port $port (A=$amm seg=$seg) pid $! -> $LOGDIR/${arm}_msonly.log"
}

launch arm7 4,5 29521 A1 scene_roles
sleep 5
launch arm0 6,7 29522 A0 referring
sleep 3
echo "MS_CONTINUE_LAUNCHED (detached; follow $LOGDIR/*.log)"
