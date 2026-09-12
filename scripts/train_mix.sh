#!/bin/bash
# Three-dataset co-training (rlbench_selfgen_512_aug + droid + bridge), optionally with the
# depth auxiliary stream. See MULTIDATASET_DEPTH_PLAN.md for why each default is what it is.
#
# ARMS (MULTIDATASET_DEPTH_PLAN.md 4.5) -- the two differ by ONE flag, the template menu:
#   A1  co-training only        : every tree serves video+action (bridge: video)
#   A2  co-training + depth aux : selfgen additionally serves video+depth at 1/3
# A2 - A1 is the net effect of the depth stream. A1 - arm0 is the net effect of co-training.
# Running only A2 cannot separate the two.
#
# WHY NOT the official HF `rlbench` tree: every eval in this repo runs on selfgen
# (eval/rollout.py DEFAULT_TASKS, eval/rollout_env.py reproduces selfgen scenes,
# eval_action/eval_perception default --data to data/rlbench_selfgen) and every existing arm
# trained on selfgen. The official tree's 5 tasks are disjoint from selfgen's 16, so training
# on them dilutes the budget with tasks that are never evaluated. See plan 2.2.
#
# Usage:  bash scripts/train_mix.sh A2 [extra train.py args...]
# Env:    GPUS, SEED, STEPS, CKPT_EVERY, SAVE_TOP_K, OUT, PORT, INIT_CKPT, DATASET_SPECS,
#         TEMPLATE_MIX_PER_DATASET, PERCEPTION_MASK_MIX, FRAME_INTERVAL, VARIATIONS, DS_CONFIG

set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate ttd_train
REPO=/workspace/ttdu/ActionImages-Cogen
cd "$REPO"

ARM="${1:-A2}"; shift || true
# Everything after the arm name is forwarded to train.py (the usage line promises
# it). Without "$@" below they were silently dropped -- a smoke run passing
# --warmup_steps 5 trained with warmup_steps=1000 and nothing said so.
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_PROJECT="${WANDB_PROJECT:-actionimages-cogen}"
# Ported verbatim from train_arm.sh. wandb.init() runs AFTER the 12.8GB checkpoint has loaded
# (~3 min in), so a missing key does not fail fast -- it wastes the load and then dies. This
# launcher was missing the block, which is exactly how it failed on its first real invocation.
TTD_ENV="${TTD_ENV:-/workspace/ttdu/.env}"
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$TTD_ENV" ]; then
  WANDB_API_KEY="$(sed -n 's/^WANDB_API=//p' "$TTD_ENV" | tr -d '"'"'"' \r')"
  [ -n "$WANDB_API_KEY" ] && export WANDB_API_KEY
  WANDB_ENTITY_FROM_ENV="$(sed -n 's/^WANDB_ENTITY=//p' "$TTD_ENV" | tr -d '"'"'"' \r')"
  [ -n "$WANDB_ENTITY_FROM_ENV" ] && export WANDB_ENTITY="${WANDB_ENTITY:-$WANDB_ENTITY_FROM_ENV}"
fi
# Only a hard error when logging is actually requested; WANDB_MODE=offline / --report_to none
# are legitimate ways to run without a key (the smoke path uses the former).
if [ -z "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-online}" = "online" ] \
   && [[ "${EXTRA_ARGS:-} $*" != *"--report_to none"* ]]; then
  echo "!! no WANDB_API_KEY (looked in \$WANDB_API_KEY and $TTD_ENV:WANDB_API)."
  echo "   Export it, or use WANDB_MODE=offline, or pass --report_to none."
  exit 4
fi

GPUS="${GPUS:-6,7}"
SEED="${SEED:-42}"
RES="${RES:-512}"                 # selfgen_512_aug renders natively at 512
FRAME_INTERVAL="${FRAME_INTERVAL:-3}"
VARIATIONS="${VARIATIONS:-0}"     # selfgen only; droid/bridge ignore it

# --- the mix (plan 4.1 / 4.2) -------------------------------------------------------------
# Ratios follow Action-Images Tab.1's trajectory counts (RLBench 180k / DROID 80k / Bridge 30k
# -> .62/.28/.10). Our absolute counts are the OPPOSITE ordering (1,028 / 9,437 / 17,709), so
# these ratios are a deliberate re-weighting, not a consequence of how much data we have.
DATASET_SPECS="${DATASET_SPECS:-rlbench_selfgen_512_aug@0.62,droid@0.28,bridge@0.10}"
case "$ARM" in
  A1) DEFAULT_MENU="rlbench_selfgen_512_aug=video+action@1.0;droid=video+action@1.0;bridge=video@1.0" ;;
  A2) DEFAULT_MENU="rlbench_selfgen_512_aug=video+action@0.67,video+depth@0.33;droid=video+action@1.0;bridge=video@1.0" ;;
  *)  DEFAULT_MENU="" ;;
esac
TEMPLATE_MIX_PER_DATASET="${TEMPLATE_MIX_PER_DATASET:-$DEFAULT_MENU}"
if [ -z "$TEMPLATE_MIX_PER_DATASET" ]; then
  echo "unknown ARM '$ARM' and no TEMPLATE_MIX_PER_DATASET= given. Use A1 or A2."; exit 2
fi
# Plan 4.3 / decision D5: depth uses the SAME mask split as video+action, which is what
# plan_segments already hard-codes for the action branch -- (iiii .81, fiii .045, fifi .045,
# policy .10). NOT train_arm.sh's M2: this experiment deliberately puts both streams on one
# distribution so template identity does not predict conditioning level.
PERCEPTION_MASK_MIX="${PERCEPTION_MASK_MIX:-0.81,0.045,0.045,0.10}"

if [ -z "${PORT:-}" ]; then
  PORT="$(python - <<'PY'
import random, socket, sys
for _ in range(500):
    p = random.randint(20000, 29999)
    with socket.socket() as s:
        try: s.bind(("", p))
        except OSError: continue
        print(p); break
else: sys.exit("train_mix.sh: no free port in 20000-29999")
PY
)" || exit 5
fi

NPROC=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
INIT_CKPT="${INIT_CKPT-/workspace/ttdu/starVLA/playground/Pretrained_models/anyeZHY/ActionImages/step125750.ckpt}"
if [ -n "$INIT_CKPT" ]; then STEPS="${STEPS:-10000}"; else STEPS="${STEPS:-125000}"; fi
# 3000 requested by the operator. Each step*.ckpt is ~12.8G (DiT-only bf16); _prune_resume_state
# keeps every one of those but only the LATEST global_step*/ (~108G of ZeRO-2 optimizer state).
# Peak during a save is transiently 2x that dir, because HF writes the new one before the old is
# pruned -- budget ~220G of headroom on top of the step ckpts.
CKPT_EVERY="${CKPT_EVERY:-1000}"
SAVE_TOP_K="${SAVE_TOP_K:--2}"
# The config identity goes in the path, for the reason train_arm.sh spells out: a non-empty
# output_dir means RESUME, so two runs that differ in datasets/menu/resolution/interval but
# share a name will silently continue each other. Hash the fields that change what the arm IS.
CFG_ID=$(printf '%s|%s|%s|%s|%s' "$DATASET_SPECS" "$TEMPLATE_MIX_PER_DATASET" \
         "$PERCEPTION_MASK_MIX" "$RES" "$FRAME_INTERVAL" | sha1sum | cut -c1-8)
OUT="${OUT:-$REPO/outputs/mix_${ARM}_seed${SEED}_fi${FRAME_INTERVAL}_${RES}_${CFG_ID}}"
DS_CONFIG="${DS_CONFIG:-./configs/zero2_offload.json}"

if [ "$SAVE_TOP_K" -gt 0 ]; then KEPT="$SAVE_TOP_K"; else KEPT=$(( STEPS / CKPT_EVERY )); fi
AVAIL_G=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
PROJECTED_G=$(( KEPT * 13 + 220 ))
echo "df /workspace avail = ${AVAIL_G}G ; projected ${PROJECTED_G}G (${KEPT} x 13G step ckpts + 220G peak optimizer state)"
[ "${AVAIL_G:-0}" -lt "$PROJECTED_G" ] && echo "   NOTE: below projection -- prune as you go or this run fills the volume."

RESUME_FROM=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -n "$RESUME_FROM" ]; then
  echo "!! $OUT already contains $RESUME_FROM -- this will RESUME from it, NOT warm-start."
  echo "   Ctrl-C within 10s if you meant a fresh arm (then use a new OUT=)."
  sleep 10
fi
mkdir -p "$OUT"
INIT_ARG=""; [ -n "$INIT_CKPT" ] && INIT_ARG="--init_ckpt_path $INIT_CKPT"

echo "arm=$ARM GPUS=$GPUS NPROC=$NPROC SEED=$SEED STEPS=$STEPS RES=$RES PORT=$PORT"
echo "  datasets : $DATASET_SPECS"
echo "  templates: $TEMPLATE_MIX_PER_DATASET"
echo "  mask mix : $PERCEPTION_MASK_MIX (perception); action branch uses its hard-coded .81/.045/.045/.10"
echo "  out      : $OUT"

CUDA_VISIBLE_DEVICES=$GPUS torchrun --nnodes=1 --nproc_per_node=$NPROC --master_port $PORT \
  train.py \
  --deepspeed "$DS_CONFIG" \
  --dataset_path ./data \
  --dataset_name "$DATASET_SPECS" \
  --template_mix_per_dataset "$TEMPLATE_MIX_PER_DATASET" \
  --variations "$VARIATIONS" \
  $INIT_ARG \
  --output_dir "$OUT" \
  --height "$RES" --width "$RES" --num_frames 41 --frame_interval "$FRAME_INTERVAL" \
  --perception_mask_mix "$PERCEPTION_MASK_MIX" \
  --full_param True \
  --model_id Wan-AI/Wan2.2-TI2V-5B \
  --max_steps "$STEPS" --num_train_epochs 1 --steps_per_epoch "$STEPS" \
  --learning_rate 5e-8 --warmup_steps 1000 --lr_scheduler_type constant_with_warmup \
  --gradient_accumulation_steps 4 --max_grad_norm 1.0 \
  --use_gradient_checkpointing \
  --dataloader_num_workers 4 --dataloader_prefetch_factor 2 --dataloader_pin_memory True \
  --checkpoint_every_n_steps "$CKPT_EVERY" --checkpoint_save_top_k "$SAVE_TOP_K" \
  --remove_unused_columns False --dataloader_drop_last True \
  --prediction_loss_only True --bf16 True --ddp_find_unused_parameters False \
  --save_safetensors False --per_device_train_batch_size 1 \
  --logging_steps 10 --seed "$SEED" \
  --report_to wandb --run_name "mix-${ARM}-seed${SEED}" ${EXTRA_ARGS:-} "$@"
TRAIN_EXIT=$?
echo "TRAIN_EXIT=$TRAIN_EXIT"
# Propagate it. `echo` was the last statement, so the script exited 0 even when torchrun
# died -- any `bash train_mix.sh && next-step` chain treated a crashed run as success.
exit "$TRAIN_EXIT"
