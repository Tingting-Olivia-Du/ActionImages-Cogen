#!/bin/bash
# ============================================================================
# One experimental arm of the perception/action co-generation study.
# ----------------------------------------------------------------------------
# An arm is defined by ONE thing: --template_mix. Everything else -- data tree, seed, steps,
# lr, warm-start checkpoint, prompt tag style -- is identical across arms, so any difference
# in r_peak is attributable to the task menu and nothing else.
#
#   bash scripts/train_arm.sh arm0     # video+action@1.0                      the CONTROL
#   bash scripts/train_arm.sh arm1     # 60% action / 20% depth / 20% seg      low-ratio perception
#   bash scripts/train_arm.sh arm2     # 50% action / 50% co-generation        co-supervision
#   bash scripts/train_arm.sh "video+depth@1.0"      # any explicit mix also works
#
# WHY Arm-0 IS NOT OPTIONAL. The variation0 train split is only ~470 episodes, and a pure
# action run on it has previously gone r_peak 500步=98.5 -> 2500步=47.0 -> 3000步=3.7 while
# the loss curve looked fine (ttd plan/core/TODO_post_3k_overfit.md). Without a same-recipe
# no-perception arm there is nothing to attribute an r_peak change to: continued-training
# drift and perception interference are indistinguishable. A zero-shot reading of the official
# checkpoint does NOT serve this role -- it measures a different thing (how strong the prior
# is), not where the prior drifts to on this data.
#
# INITIALISATION: stage 1 warm-starts from anyeZHY's step125750, because the main research
# question is "does adding perception damage an ALREADY-TRAINED action prior". Pass
# INIT_CKPT= (empty) for the stage-2 from-Wan-base run, which answers a different question
# ("which modality is easier to learn") and costs ~21 days/arm at 125k steps -- do not start
# that until the code is frozen and stage 1 has a result.
#
# RESOLUTION: 256, because the selfgen tree is RENDERED at 256x256 (both rgb and depth.npz).
# Training at the official 512 would upsample 256 -> 512: ~4x the attention cost for zero extra
# information. NOTE the official checkpoint was trained at 512, so every arm pays the same
# 512-prior -> 256-domain adaptation; it is common-mode and cancels between arms, but absolute
# numbers are not comparable to the paper.
#
# Env overrides:  GPUS, SEED, STEPS, RES, OUT, PORT, INIT_CKPT, VARIATIONS, EXTRA_ARGS
# Resume: re-run the same command; find_latest_checkpoint picks up output_dir's newest ckpt.
#         A FRESH arm therefore needs an empty output_dir, or the warm-start is silently
#         replaced by that directory's latest checkpoint (train.py:490).
# ============================================================================
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate ttd_train

ARM="${1:?usage: train_arm.sh <arm0|arm1|arm2|template_mix>}"
case "$ARM" in
  arm0) MIX="video+action@1.0" ;;
  arm1) MIX="video+action@0.6,video+depth@0.2,video+segmentation@0.2" ;;
  arm2) MIX="video+action@0.5,video+depth+action@0.5" ;;
  *)    MIX="$ARM" ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
# Required: an editable install of a distribution ALSO named "actionimages" maps `training`
# to /workspace/ttdu/ActionImages. train.py asserts on this, but set it correctly up front.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="${WANDB_PROJECT:-actionimages-cogen}"
# wandb.init() runs AFTER the 12.8GB checkpoint has loaded (~3 min in), so a missing key does
# not fail fast -- it wastes the load and then dies. Pick the key up from the project .env
# (D-008) rather than relying on whatever `wandb login` state this shell happens to have.
TTD_ENV="${TTD_ENV:-/workspace/ttdu/.env}"
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$TTD_ENV" ]; then
  WANDB_API_KEY="$(sed -n 's/^WANDB_API=//p' "$TTD_ENV" | tr -d '"'"'"' \r')"
  [ -n "$WANDB_API_KEY" ] && export WANDB_API_KEY
  WANDB_ENTITY_FROM_ENV="$(sed -n 's/^WANDB_ENTITY=//p' "$TTD_ENV" | tr -d '"'"'"' \r')"
  [ -n "$WANDB_ENTITY_FROM_ENV" ] && export WANDB_ENTITY="${WANDB_ENTITY:-$WANDB_ENTITY_FROM_ENV}"
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "!! no WANDB_API_KEY (looked in \$WANDB_API_KEY and $TTD_ENV:WANDB_API)."
  echo "   Either export it, or add EXTRA_ARGS='--report_to none' to run without logging."
  exit 4
fi

GPUS="${GPUS:-1,4}"
SEED="${SEED:-42}"
RES="${RES:-256}"
PORT="${PORT:-29556}"
VARIATIONS="${VARIATIONS:-0}"   # 训练只看 variation0;其余 variation 是留出测试集
SLUG="$(echo "$ARM" | tr -c '[:alnum:]+' '_')"
OUT="${OUT:-$REPO/outputs/${SLUG}_seed${SEED}}"
NPROC=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
# Stage 1 default. Empty string = from the Wan base (stage 2, see header).
INIT_CKPT="${INIT_CKPT-/workspace/ttdu/starVLA/playground/Pretrained_models/anyeZHY/ActionImages/step125750.ckpt}"
# Warm-start needs ~10k steps; from-scratch needs the official 125k order of magnitude.
if [ -n "$INIT_CKPT" ]; then STEPS="${STEPS:-10000}"; else STEPS="${STEPS:-125000}"; fi
# Checkpoint cadence is a DISK vs RESOLUTION trade and it is expensive to get wrong BOTH ways.
# Each DiT-only bf16 checkpoint is ~12.8GB.
#   SAVE_TOP_K=-1 (keep all) is deliberate: the deliverable is the r_peak-versus-step CURVE.
#   With rotation on, intermediate checkpoints are deleted before the eval scripts (not yet
#   ported) ever see them, and the whole 1.4-day arm has to be repeated to get them back.
#   CKPT_EVERY=250 over 10k steps = 40 checkpoints = ~540GB per arm if none are ever deleted.
#   That is roughly the whole free volume, so this cadence ASSUMES the operator evaluates and
#   prunes as the run proceeds. The preflight below only enforces a floor, not the projection.
#   250 is worth the bookkeeping: the A1 diagnostic went r_peak 98.5 -> 47.0 -> 3.7 between
#   steps 500 and 3000, and a coarser cadence cannot locate the turn.
# If you would rather not babysit it: CKPT_EVERY=500 (~280GB) still resolves the curve.
# CKPT_EVERY=250 with SAVE_TOP_K=8 does NOT -- it keeps only the last 2k steps.
CKPT_EVERY="${CKPT_EVERY:-250}"
SAVE_TOP_K="${SAVE_TOP_K:--3}"

# /workspace is a shared 17T volume that outside tenants fill without warning, and it has been
# observed swinging between 37GB and 265GB free within minutes (ttd DECISIONS.md D-038).
# The hard gate is deliberately only "can this run write its next few checkpoints" -- the
# operator prunes evaluated checkpoints as the run proceeds, so demanding the full projection
# up front would block a run that is actually fine. The projection is still printed, loudly,
# because running out at step 7000 costs a day.
if [ "$SAVE_TOP_K" -gt 0 ]; then KEPT="$SAVE_TOP_K"; else KEPT=$(( STEPS / CKPT_EVERY )); fi
PROJECTED_G=$(( KEPT * 13 + 20 ))
FLOOR_G="${FLOOR_G:-80}"        # ~6 checkpoints of headroom before pruning becomes urgent
AVAIL_G=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
echo "df /workspace avail = ${AVAIL_G}G"
echo "checkpoints: every ${CKPT_EVERY} steps, keep ${SAVE_TOP_K} -> up to ${KEPT} x ~13G = ~${PROJECTED_G}G if never pruned"
# Hard floor DISABLED by the operator -- they prune manually. Re-enable by uncommenting.
# if [ "${AVAIL_G:-0}" -lt "$FLOOR_G" ]; then
#   echo "!! only ${AVAIL_G}G free, below the ${FLOOR_G}G floor -- not starting."
#   echo "   Free space, or lower the count: CKPT_EVERY=1000 / SAVE_TOP_K=8."
#   exit 3
# fi
if [ "${AVAIL_G:-0}" -lt "$PROJECTED_G" ]; then
  echo "   NOTE: ${AVAIL_G}G < ${PROJECTED_G}G projected. Fine if you prune as you go --"
  echo "   but this run WILL fill the volume around step $(( (AVAIL_G - 20) / 13 * CKPT_EVERY )) if you do not."
fi

# A non-empty output_dir means find_latest_checkpoint will RESUME from it and the warm-start
# checkpoint is ignored (train.py:490). That is the documented way to continue an interrupted
# arm -- and a silent disaster if the directory belongs to a different experiment.
RESUME_FROM=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -n "$RESUME_FROM" ]; then
  echo "!! $OUT already contains $RESUME_FROM -- this will RESUME from it, NOT warm-start."
  echo "   Ctrl-C within 10s if you meant to start a fresh arm (then use a new OUT=)."
  sleep 10
fi

mkdir -p "$OUT"
# Only pass --init_ckpt_path when one was explicitly requested; passing an empty string would
# make HfArgumentParser set it to "" rather than None, and torch.load("") then fails.
INIT_ARG=""
[ -n "$INIT_CKPT" ] && INIT_ARG="--init_ckpt_path $INIT_CKPT"
echo "arm=$ARM mix='$MIX' GPUS=$GPUS NPROC=$NPROC SEED=$SEED STEPS=$STEPS RES=$RES OUT=$OUT"
echo "init: ${INIT_CKPT:-<Wan base, no warm-start (stage 2)>}"

# zero2_offload, not zero.json: full_param=True on a 5B model needs the Adam states on CPU to
# fit two 48GB cards at 256^2. This is the config the 14.5 s/it throughput figure was measured
# with (ttd runs/A4_stage2_full_v12seg, same GPUs 4+7). Plain ZeRO-2 keeps fp32 master weights
# and Adam moments resident and does not fit.
DS_CONFIG="${DS_CONFIG:-./configs/zero2_offload.json}"

CUDA_VISIBLE_DEVICES=$GPUS torchrun --nnodes=1 --nproc_per_node=$NPROC --master_port $PORT \
  train.py \
  --deepspeed "$DS_CONFIG" \
  --dataset_path ./data \
  --dataset_name "rlbench_selfgen@1.0" \
  --template_mix "$MIX" \
  --variations "$VARIATIONS" \
  $INIT_ARG \
  --output_dir "$OUT" \
  --height "$RES" --width "$RES" --num_frames 41 \
  --full_param True \
  --model_id Wan-AI/Wan2.2-TI2V-5B \
  --max_steps "$STEPS" --num_train_epochs 1 --steps_per_epoch "$STEPS" \
  --learning_rate 5e-7 --warmup_steps 1000 --lr_scheduler_type constant_with_warmup \
  --gradient_accumulation_steps 1 --max_grad_norm 1.0 \
  --use_gradient_checkpointing \
  --dataloader_num_workers 4 --dataloader_prefetch_factor 2 --dataloader_pin_memory True \
  --checkpoint_every_n_steps "$CKPT_EVERY" --checkpoint_save_top_k "$SAVE_TOP_K" \
  --remove_unused_columns False --dataloader_drop_last True \
  --prediction_loss_only True --bf16 True --ddp_find_unused_parameters False \
  --save_safetensors False --per_device_train_batch_size 1 \
  --logging_steps 10 --seed "$SEED" \
  --report_to wandb --run_name "${SLUG}-seed${SEED}" ${EXTRA_ARGS:-}
echo "TRAIN_EXIT=$?"
