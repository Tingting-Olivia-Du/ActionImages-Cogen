#!/bin/bash
# arm7 and arm0 trained on ManiSkill3 ALONE, warm-started from the released ActionImages
# checkpoint (not from our own 6k runs), 4,000 steps, four GPUs each, in parallel.
#
# WHY FROM THE RELEASED CHECKPOINT. Both 6k runs carry 4,200 steps of RLBench history. Starting
# both arms from step125750 instead makes ManiSkill the ONLY thing either arm has been
# fine-tuned on, so any difference between them is the observation-space menu and nothing else.
#
# THE ONE DELIBERATE CHANGE FROM THE ORIGINAL PAIR: BOTH ARMS USE ACTION_MASK_MIX=A1.
#   The 6k pair trained arm7 under A1 and arm0 under A0 (outputs/joint_launch_logs/*.log), and
#   A1 spends more probability on exactly the single-anchor plan a closed-loop rollout supplies
#   -- so it confounds the headline comparison. Tables/tab_ablation.tex (a) names this confound
#   and records "the A1 baseline still has to be trained". A fresh pair removes it for free,
#   and ABLATION_HANDOFF.md already states "every arm uses A1". arm7 - arm0 is now the menu.
#
# EVERYTHING ELSE MATCHES THE ORIGINAL RUNS:
#   4 GPUs x bs1 x accum1 = effective batch 4  (the 2-GPU continuation needed accum 2 to get there)
#   lr 5e-7, constant_with_warmup, warmup 1000 -- the warmup IS right here: this is a warm start
#     from a checkpoint trained on other data, which is what that 1000 was sized for.
#   seed 42, 512^2, 41 frames, full_param, ZeRO-2 + CPU Adam offload.
#   FRAME_INTERVAL=1 because the loader pins ManiSkill to stride 1 (base.py); 3 only mislabels
#     the startup banner.
#
# W&B IS ON (project actionimages-cogen, entity from TTD_ENV). The run name is explicit because
# train_arm.sh's default would be "arm7-a1-seed42" -- the same name as older runs trained on
# other data -- and two curves under one name are how a comparison gets read off the wrong run.
#
# EXPOSURE, for whoever reads the result: 4000 steps x batch 4 = 16,000 samples over 1,750
# variation-0 episodes = 9.1 draws per episode, against 4.11 in the joint mix. If a model that
# has seen each training episode nine times still fails on those same scenes closed-loop, the
# bottleneck is not data exposure.
set -uo pipefail
REPO=/workspace/1228_tingting/ActionImages-Cogen
source /workspace/1228_tingting/ttd/scripts/env_eval_ttd_eval.rc
export TTD_ENV="${TTD_ENV:-/workspace/1228_tingting/.env}"
cd "$REPO" || exit 2
python - <<'PY' || { echo "!! preflight failed -- not launching"; exit 2; }
import transformers, deepspeed, diffsynth  # noqa: F401
from transformers.integrations import is_wandb_available
print(f"preflight OK: transformers {transformers.__version__}, deepspeed {deepspeed.__version__}, "
      f"wandb_available={is_wandb_available()}")
PY
LOGDIR=$REPO/outputs/ms_fresh_logs
mkdir -p "$LOGDIR"

launch() {  # launch <arm> <gpus> <port>
  local arm=$1 gpus=$2 port=$3
  DATASET="maniskill3" VARIATIONS=0 FRAME_INTERVAL=1 RES=512 \
  GPUS="$gpus" PORT="$port" SEED=42 STEPS=4000 CKPT_EVERY=1000 SAVE_TOP_K=-1 \
  SAVE_OPTIM=False ACTION_MASK_MIX=A1 \
  INIT_CKPT="$REPO/checkpoints/official/step125750.ckpt" \
  OUT="$REPO/outputs/${arm}_msonly_fromofficial_a1_4k" \
  EXTRA_ARGS="--allow_step_restart --run_name ${arm}-msonly-fromofficial-a1-4k" \
    setsid nohup bash "$REPO/scripts/train_arm.sh" "$arm" \
    > "$LOGDIR/${arm}.log" 2>&1 < /dev/null &
  echo "launched $arm on GPUs $gpus port $port pid $! -> $LOGDIR/${arm}.log"
}

launch arm7 0,1,2,3 29531
sleep 5
launch arm0 4,5,6,7 29532
sleep 3
echo "MS_FRESH_LAUNCHED"
