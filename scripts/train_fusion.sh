#!/bin/bash
# ============================================================================
# The FUSION arm: all five modalities co-present in ONE latent sequence.
# ----------------------------------------------------------------------------
#   V0 D0 S0 N0 A0 | V1 D1 S1 N1 A1        10 segments, 110 latent frames
#
# Every arm before this one packs FOUR segments -- one visual modality plus action, two views
# -- and `train.py:_batch_template` refuses a batch containing more than one template, so two
# modalities NEVER co-exist in a training step. arm6/arm7 therefore learn "one set of weights
# taking turns at four jobs", not "modalities supervising each other".
# CROSSMODAL_FUSION_PLAN.md §4.1 lists that as arm8's blocking constraint and files the
# co-generation template as option B ("真融合"), deferred. This is option B taken to its limit.
#
#   bash scripts/train_fusion.sh fusion0          # the F1 modality-dropout arm  (Stage 1)
#   bash scripts/train_fusion.sh fusion0-anchor   # F0 control: the all-anchor behaviour
#   bash scripts/train_fusion.sh "video+depth+action"   # any explicit template also works
#
# WHY THIS IS A SEPARATE FILE FROM train_arm.sh. Not taste -- bash reads a script by byte
# offset as it executes it, so editing a running .sh makes the live shell resume mid-comment
# and, on 2026-09-06, silently restarted a job. train_arm.sh is held open by every arm
# currently training. A new file cannot do that to them.
#
# THE ONE THING THAT HAD TO BE WRITTEN: the F axis (--fusion_mask_mix, templates.py
# FUSION_MASK_MIX_PRESETS). `assemble` hands every non-fully-given segment a free first latent
# frame; on a 10-segment canvas that is TEN free anchors. The tag-swap ablation
# (paper/EXPERIMENT_STATUS_zh_aug29.md) measured what that means: swapping the prompt tag moves
# metrics 1-2%, removing the anchor costs ~9x. So with all ten anchors present, "the model
# completes depth from the other modalities" is indistinguishable from "the model copies depth's
# own anchor", AND the model only ever trains in a regime deployment cannot supply -- at rollout
# there is no depth/segmentation/normal frame 0 to hand it. F1 spends 55% of samples with the
# non-RGB modalities fully ABSENT (no anchor at all), which is both the deployment condition and
# the only setting in which cross-modal completion is a real prediction problem.
#
# COST, measured not guessed. arm7u runs 4 segments / 11,264 tokens at 20.5 s/it on 2 cards.
# This canvas is 110 latent frames x 16 x 16 = 28,160 tokens: ~2.5x the sequence, ~3.6x the DiT
# FLOPs (attention is quadratic), plus 14 VAE encodes per step instead of 8.
#
# WHY 4 GPUS, AND WHY 5000 STEPS. Data parallel does NOT shorten a step -- sequence parallelism
# is wired for inference only (wan_video_action_images.py enable_usp), so 4 cards buy batch, not
# speed. They buy two other things:
#   1. MEMORY, and this arm does not fit without it. ZeRO-2 shards gradients across ranks: 5B
#      fp32 gradients are 20G total, i.e. 10G/card on 2 cards but 5G/card on 4. That 5G is what
#      pays for the extra ~6G of activation the longer sequence needs. arm7u already sits at
#      41-43G of 48G at HALF this sequence length.
#   2. Twice the sample exposure per wall-clock hour -- so 5000 steps at batch 4 is exactly
#      arm7u@10000's 20k samples, in half the time.
# 5000 IS THE DEFAULT FOR A REASON. ARM7_RESULTS.md retracted a headline number over exactly
# this: "spec@4000 is the control for arm7@10000, not arm7@4000". Matching EXPOSURE rather than
# step count removes that failure mode by construction. Do not raise STEPS without also saying
# what arm7u checkpoint it is then comparable to.
#
# CONTROLS. fusion0's control is arm7u, NOT arm6: same warm start, tree, seed, four modalities
# and A1, differing only in "one modality per sample" vs "all of them". fusion0-anchor is the
# control for the F axis itself -- identical template, no --fusion_mask_mix, so multi-modality
# samples fall back to the M/A axes and every segment keeps its anchor.
#
# Env overrides: GPUS, SEED, STEPS, RES, OUT, PORT, INIT_CKPT, DATASET, VARIATIONS,
#                FRAME_INTERVAL, SEG_MODE, FUSION_MASK_MIX, ACTION_DROPOUT, CKPT_EVERY,
#                SAVE_TOP_K, DS_CONFIG, EXTRA_ARGS
# Resume: re-run the same command. A FRESH arm needs an empty output_dir or the warm start is
#         silently replaced by that directory's latest checkpoint (train.py find_latest_checkpoint).
#
# The environment/launch block below is transcribed from scripts/train_arm.sh (the aligned
# source) as of 2026-09-08, with the arm-selection and mask-axis logic replaced.
# ============================================================================
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate ttd_train

ARM="${1:?usage: train_fusion.sh <fusion0|fusion0-anchor|template>}"
FULL_MIX="video+depth+segmentation+normal+action@1.0"
case "$ARM" in
  fusion0)        MIX="$FULL_MIX"; FUSION_MASK_MIX_DEFAULT=F1 ;;
  # No F axis: the template still packs 10 segments, but plan_segments routes it back to the
  # M/A axes and every segment keeps its first frame. This is the "what did modality dropout
  # actually buy" control, and it is obtained by DECLARING F0 rather than by omitting a flag,
  # so the run's config records which of the two it meant.
  fusion0-anchor) MIX="$FULL_MIX"; FUSION_MASK_MIX_DEFAULT=F0 ;;
  # Drops full_anchor entirely -- "is the all-anchor regime load-bearing, or just diluting?"
  fusion0-f2)     MIX="$FULL_MIX"; FUSION_MASK_MIX_DEFAULT=F2 ;;
  *)              MIX="$ARM" ;;
esac
FUSION_MASK_MIX="${FUSION_MASK_MIX:-${FUSION_MASK_MIX_DEFAULT:-F1}}"

# scene_roles, not referring: `referring` is a target-only mask (median 0.18% non-black) and
# `scene_roles` a dense role map (11.8%). They are different supervision signals sharing a
# modality name, and arm6/arm7/arm7u -- everything this arm is compared against -- are all
# scene_roles. Part of the arm DEFINITION, not a preference.
SEG_MODE="${SEG_MODE:-scene_roles}"
case "$SEG_MODE" in
  referring|scene_roles) ;;
  *) echo "!! SEG_MODE=$SEG_MODE; expected 'referring' or 'scene_roles'"; exit 6 ;;
esac

# 0, not the 0.1 every other arm uses. Action already falls from 2/4 to 2/10 of the predicted
# latent positions on this canvas -- the loss is a uniform mean over them (train.py) -- so
# dropping <action> from another 10% of samples would compound a dilution this arm is already
# paying. It also keeps every sample the same shape, which is what makes the per-segment loss
# table (segloss_pos/*) comparable across steps.
ACTION_DROPOUT="${ACTION_DROPOUT:-0.0}"


REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
# Required: an editable install of a distribution ALSO named "actionimages" maps `training`
# to /workspace/ttdu/ActionImages. train.py asserts on this, but set it correctly up front.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="${WANDB_PROJECT:-actionimages-cogen}"
# wandb.init() runs AFTER the 12.8GB checkpoint has loaded (~3 min in), so a missing key does
# not fail fast -- it wastes the load and then dies.
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

# Four cards by default -- see the header: this arm needs the ZeRO-2 gradient sharding to fit,
# not just the throughput. Re-check nvidia-smi before launching; do not trust this list.
GPUS="${GPUS:-0,1,2,6}"
SEED="${SEED:-42}"
DATASET="${DATASET:-rlbench_selfgen_512_aug}"
case "$DATASET" in
  rlbench_selfgen_512*|rlbench_unseen_tasks_512*) RES="${RES:-512}" ;;
  *)                                              RES="${RES:-256}" ;;
esac
FRAME_INTERVAL="${FRAME_INTERVAL:-3}"
VARIATIONS="${VARIATIONS:-0}"   # 训练只看 variation0;其余 variation 是留出测试集

# Probe for a free rendezvous port rather than defaulting to one: a collision surfaces as a
# rendezvous timeout several MINUTES in, after the checkpoint has already loaded. 20000-29999
# sits below Linux's ephemeral range so nothing can take it between the probe and the bind.
if [ -z "${PORT:-}" ]; then
  PORT="$(python - <<'PY'
import random, socket, sys
for _ in range(500):
    p = random.randint(20000, 29999)
    with socket.socket() as s:
        try:
            s.bind(("", p))
        except OSError:
            continue
        print(p)
        break
else:
    sys.exit("train_fusion.sh: no free port in 20000-29999")
PY
)" || exit 5
fi

SLUG="$(echo "$ARM" | tr -c '[:alnum:]+' '_')"
FI_SUFFIX=""
[ "$FRAME_INTERVAL" != "1" ] && FI_SUFFIX="_fi${FRAME_INTERVAL}"
DS_SUFFIX=""
[ "$DATASET" != "rlbench_selfgen" ] && DS_SUFFIX="_${DATASET#rlbench_selfgen_}"
SEG_SUFFIX=""
[ "$SEG_MODE" = "scene_roles" ] && SEG_SUFFIX="_sr"
# The F axis goes in the OUTPUT PATH, not just the run name. Same reason the tree and the
# interval do in train_arm.sh: it changes what the arm IS. fusion0 and fusion0-anchor share a
# template, so without it the control would land in fusion0's directory, hit the resume branch
# below, and silently continue from F1-trained weights -- a hybrid no metric can be attributed
# to anything. (train_arm.sh puts the A axis in the run name only, and its own comment tells
# you to pass an explicit OUT= when sweeping it. This avoids needing that.)
FMM_SUFFIX="_$(printf '%s' "$FUSION_MASK_MIX" | tr -c '[:alnum:]' '-')"
OUT="${OUT:-$REPO/outputs/${SLUG}_seed${SEED}${FI_SUFFIX}${DS_SUFFIX}${SEG_SUFFIX}${FMM_SUFFIX}}"
NPROC=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
# Global batch must stay 4 no matter how many cards the scheduler hands out, because batch is part
# of the recipe: steps 1-5000 of this arm ran at 4. On a cluster where 4 GPUs on one node is a
# rare allocation, accumulation is what decouples the recipe from the allocation.
#   4 GPUs -> accum 1      2 GPUs -> accum 2      1 GPU -> accum 4
# Derived rather than passed so the two cannot disagree; override GRAD_ACCUM only to break the
# invariant on purpose, and say so in the run name if you do.
GRAD_ACCUM="${GRAD_ACCUM:-$(( 4 / NPROC ))}"
if [ $(( GRAD_ACCUM * NPROC )) -ne 4 ]; then
  echo "!! global batch = GRAD_ACCUM($GRAD_ACCUM) x NPROC($NPROC) = $(( GRAD_ACCUM * NPROC )), not 4."
  echo "   steps 1-5000 of this arm ran at 4; a different batch is a different recipe."
  echo "   Set GRAD_ACCUM explicitly if that is intended."
fi

INIT_CKPT="${INIT_CKPT-/workspace/ttdu/starVLA/playground/Pretrained_models/anyeZHY/ActionImages/step125750.ckpt}"
# 5000 warm-start (= arm7u@10000's exposure at batch 4; see header) vs the 125k order of
# magnitude a from-Wan-base run needs.
if [ -n "$INIT_CKPT" ]; then STEPS="${STEPS:-5000}"; else STEPS="${STEPS:-125000}"; fi
CKPT_EVERY="${CKPT_EVERY:-1000}"
SAVE_TOP_K="${SAVE_TOP_K:--1}"

if [ "$SAVE_TOP_K" -gt 0 ]; then KEPT="$SAVE_TOP_K"; else KEPT=$(( STEPS / CKPT_EVERY )); fi
# A checkpoint directory is ~120G while it is the resume tip and ~13G after
# `_prune_resume_state` drops its global_step*/ (on by default). So the steady state is
# (KEPT-1) pruned + 1 tip, not KEPT tips.
CKPT_G="${CKPT_G:-13}"; TIP_G="${TIP_G:-120}"
PROJECTED_G=$(( (KEPT - 1) * CKPT_G + TIP_G + 20 ))
AVAIL_G=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
echo "df /workspace avail = ${AVAIL_G}G"
echo "checkpoints: every ${CKPT_EVERY} steps, keep ${SAVE_TOP_K} -> ${KEPT} total, ~${PROJECTED_G}G"
if [ "${AVAIL_G:-0}" -lt "$PROJECTED_G" ]; then
  echo "   NOTE: ${AVAIL_G}G < ${PROJECTED_G}G projected -- fine if you prune as you go."
fi

RESUME_FROM=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -n "$RESUME_FROM" ]; then
  echo "!! $OUT already contains $RESUME_FROM -- this will RESUME from it, NOT warm-start."
  echo "   Ctrl-C within 10s if you meant to start a fresh arm (then use a new OUT=)."
  sleep 10
fi
mkdir -p "$OUT"

# Only pass --init_ckpt_path when one was requested: an empty string makes HfArgumentParser set
# it to "" rather than None, and torch.load("") then fails.
INIT_ARG=""
[ -n "$INIT_CKPT" ] && INIT_ARG="--init_ckpt_path $INIT_CKPT"

SEGS=$(( $(printf '%s' "${MIX%%@*}" | tr '+' '\n' | grep -c .) * 2 ))
echo "arm=$ARM mix='$MIX' -> ${SEGS} segments"
echo "seg_mode=$SEG_MODE fusion_mask_mix=$FUSION_MASK_MIX action_dropout=$ACTION_DROPOUT"
echo "dataset=$DATASET GPUS=$GPUS NPROC=$NPROC SEED=$SEED STEPS=$STEPS RES=$RES PORT=$PORT"
echo "global batch = NPROC($NPROC) x per_device(1) x accum($GRAD_ACCUM) = $(( NPROC * GRAD_ACCUM )); ${STEPS} steps = $(( STEPS * NPROC * GRAD_ACCUM )) samples"
echo "OUT=$OUT"
# Report what the weights ACTUALLY come from: train.py prefers a checkpoint found in output_dir
# over init_ckpt_path, so printing INIT_CKPT unconditionally reads as "warm-starting" even when
# the run is really continuing from its own latest checkpoint.
RESUME_CKPT=""
[ -n "$RESUME_FROM" ] && RESUME_CKPT=$(ls "$RESUME_FROM"/step*.ckpt 2>/dev/null | head -1)
if [ -n "$RESUME_CKPT" ]; then
  echo "weights: RESUME from $RESUME_CKPT"
  echo "         (--init_ckpt_path ${INIT_CKPT:-<none>} is passed but IGNORED -- resume wins)"
else
  echo "weights: warm-start from ${INIT_CKPT:-<Wan base, no warm-start (stage 3)>}"
fi
# The M and A axes are INERT here: --fusion_mask_mix replaces both for any template it applies
# to. Saying so beats printing them unqualified, which reads as though they were in effect.
echo "mask axes: fusion_mask_mix=$FUSION_MASK_MIX (M and A axes INERT on this template)"

# zero2_offload, not zero.json: full_param=True on a 5B model needs the Adam states on CPU.
DS_CONFIG="${DS_CONFIG:-./configs/zero2_offload.json}"

CUDA_VISIBLE_DEVICES=$GPUS torchrun --nnodes=1 --nproc_per_node=$NPROC --master_port $PORT \
  train.py \
  --deepspeed "$DS_CONFIG" \
  --dataset_path ./data \
  --dataset_name "${DATASET}@1.0" \
  --template_mix "$MIX" \
  --segmentation_mode "$SEG_MODE" \
  --fusion_mask_mix "$FUSION_MASK_MIX" \
  --action_dropout_prob "$ACTION_DROPOUT" \
  --variations "$VARIATIONS" \
  $INIT_ARG \
  --output_dir "$OUT" \
  --height "$RES" --width "$RES" --num_frames 41 --frame_interval "$FRAME_INTERVAL" \
  --full_param True \
  --model_id Wan-AI/Wan2.2-TI2V-5B \
  --max_steps "$STEPS" --num_train_epochs 1 --steps_per_epoch "$STEPS" \
  --learning_rate 5e-7 --warmup_steps 1000 --lr_scheduler_type constant_with_warmup \
  --gradient_accumulation_steps "$GRAD_ACCUM" --max_grad_norm 1.0 \
  --use_gradient_checkpointing \
  --dataloader_num_workers 4 --dataloader_prefetch_factor 2 --dataloader_pin_memory True \
  --checkpoint_every_n_steps "$CKPT_EVERY" --checkpoint_save_top_k "$SAVE_TOP_K" \
  --remove_unused_columns False --dataloader_drop_last True \
  --prediction_loss_only True --bf16 True --ddp_find_unused_parameters False \
  --save_safetensors False --per_device_train_batch_size 1 \
  --logging_steps 10 --seed "$SEED" \
  --report_to wandb --run_name "${SLUG}-seed${SEED}${DS_SUFFIX//_/-}${FMM_SUFFIX}" ${EXTRA_ARGS:-}
TRAIN_EXIT=$?
echo "TRAIN_EXIT=$TRAIN_EXIT"
exit "$TRAIN_EXIT"
