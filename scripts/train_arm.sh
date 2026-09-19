#!/bin/bash
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate ttd_train

ARM="${1:?usage: train_arm.sh <arm0..arm8|template_mix>   (plan: ACTION_MASK_MIX=A0|A1|A2)}"
# Each case sets MIX, and may declare the two things that are part of the arm's DEFINITION
# rather than a preference: which segmentation protocol it means, and which conditioning plan
# it is canonically trained under. Both are overridable per run (SEG_MODE=, ACTION_MASK_MIX=)
# and both reach the output path, so an override cannot silently resume the other variant.
# `*)` must stay LAST -- it matches everything, and any branch written after it is dead.
case "$ARM" in
  # --- the observation-space ladder. video+action pinned at 0.4 on every rung with
  #     perception; ratios sum to 1.0 because parse_template_mix normalises (see header).
  arm0) MIX="video+action@1.0"
        ACTION_MASK_MIX_DEFAULT=A1 ;;
  arm1) MIX="video+action@0.4,depth+action@0.6"
        ACTION_MASK_MIX_DEFAULT=A1 ;;
  arm2) MIX="video+action@0.4,segmentation+action@0.6" 
        SEG_MODE_DEFAULT=scene_roles
        ACTION_MASK_MIX_DEFAULT=A1 ;;
  arm3) MIX="video+action@0.4,normal+action@0.6"
        ACTION_MASK_MIX_DEFAULT=A1 ;;
  arm4) MIX="video+action@0.4,depth+action@0.3,segmentation+action@0.3"
        SEG_MODE_DEFAULT=scene_roles
        ACTION_MASK_MIX_DEFAULT=A1 ;;
  arm5) MIX="video+action@0.4,normal+action@0.3,segmentation+action@0.3"
        SEG_MODE_DEFAULT=scene_roles
        ACTION_MASK_MIX_DEFAULT=A1 ;;
  arm6) MIX="video+action@0.4,normal+action@0.3,depth+action@0.3"
        SEG_MODE_DEFAULT=scene_roles
        ACTION_MASK_MIX_DEFAULT=A1 ;;
  arm7) MIX="video+action@0.4,depth+action@0.2,segmentation+action@0.2,normal+action@0.2"
        SEG_MODE_DEFAULT=scene_roles
        ACTION_MASK_MIX_DEFAULT=A1 ;;
  # arm7's ratio control: same four templates, uniform, so every stream gets the same number of
  # updates. Formerly `arm7u` -- that name still resolves below, to its own legacy directory.
  arm8) MIX="video+action@0.25,depth+action@0.25,segmentation+action@0.25,normal+action@0.25"
        SEG_MODE_DEFAULT=scene_roles
        ACTION_MASK_MIX_DEFAULT=A1 ;;
  *)    MIX="$ARM" ;;
esac
SEG_MODE="${SEG_MODE:-${SEG_MODE_DEFAULT:-referring}}"
# The A axis (training/templates.py ACTION_MASK_MIX_PRESETS): how a template CONTAINING <action>
# splits its conditioning, as (iiii, fiii, fifi, policy). A0 = the upstream literals, which is what
# arm0-arm3, arm5, arm6 and the official step125750 were all trained under; the library default in
# templates.py stays A0 so those stay reproducible and test_forward_unchanged.py keeps its
# meaning. An arm declares its CANONICAL plan in its case above -- same split of
# responsibilities as FRAME_INTERVAL (3 here, 1 in args.py) and PERCEPTION_MASK_MIX (M2/M0) --
# and this is the one axis you are EXPECTED to override, because flipping it is the ablation.
ACTION_MASK_MIX="${ACTION_MASK_MIX:-${ACTION_MASK_MIX_DEFAULT:-A0}}"
case "$SEG_MODE" in
  referring|scene_roles) ;;
  *) echo "!! SEG_MODE=$SEG_MODE; expected 'referring' or 'scene_roles'"; exit 6 ;;
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

GPUS="${GPUS:-6,7}"
SEED="${SEED:-42}"
DATASET="${DATASET:-rlbench_selfgen_512_aug}"
# Resolution follows the tree's native render size unless RES= overrides it. Getting this wrong
# is silent -- the loader resizes without complaint -- so it is derived rather than defaulted.
case "$DATASET" in
  rlbench_selfgen_512*) RES="${RES:-512}" ;;
  *)                    RES="${RES:-256}" ;;
esac
# torchrun's rendezvous port. A fixed default collides as soon as two arms run side by side,
# which is the normal case here (one arm per GPU pair). A bare $RANDOM is not enough: it still
# collides eventually, and the failure surfaces as a rendezvous timeout several MINUTES into the
# run, after the 12.8GB checkpoint has already loaded. So probe until a port actually binds.
# Range 20000-29999 sits below Linux's default ephemeral range (32768-60999), so it cannot be
# taken by an unrelated outbound connection between the probe and torchrun's bind.
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
    sys.exit("train_arm.sh: no free port in 20000-29999")
PY
)" || exit 5
fi
VARIATIONS="${VARIATIONS:-0}"   # 训练只看 variation0;其余 variation 是留出测试集
# Temporal stride within a 41-frame window. 1 = the stage-1 arms (arm0/arm1), 2.0 s of motion.
# The official release effectively trains at 4 (its video is stored 4x-downsampled and actions
# are realigned with actions[::4]), i.e. 8.0 s. Raising this costs window diversity because a
# window then needs 1+(41-1)*FI native steps to fit and 41*FI to have more than one legal start.
# Measured over the variation0 split that training actually sees:
#   FI =                      1       2       3       4
#   512_aug (788 ep, med 168) 100%   94.2%   73.4%   52.4%
#   v2 256  (1398 ep, med 119) 100%   98.4%   40.6%   18.6%
# 3 is a far better fit on 512_aug than it ever was on v2 -- the augmented tree's episodes are
# longer (median 168 vs 119 steps), so the same interval keeps 73% of episodes multi-window
# instead of 41%. See EVAL_PLAN_CLOSEDLOOP.md Sec. 2.2.1.
FRAME_INTERVAL="${FRAME_INTERVAL:-3}"
# M axis (SEGMENTATION_SCENE_ROLES_PLAN.md §10.3), as (iiii, fiii, fifi, single_frame).
# M2 = 90/5/5 is the DELIBERATE default for new arms: it matches the official video+action split,
# so template identity stops predicting conditioning level, and -- the reason that matters here --
# the closed-loop rollout runs `video+action` under i2va (= IIII), so an auxiliary stream trained
# 90% under FIFI is learning a discriminative RGB->depth/seg map in a regime deployment never uses.
#
# The LIBRARY default in templates.py stays M0 (the historical 10/0/90) on purpose: that is what
# tests/test_forward_unchanged.py pins, and it must keep meaning "we can still reproduce the old
# behaviour". The experiment's choice belongs in the launcher, where it lands in the command line
# and in the wandb config -- same split of responsibilities as FRAME_INTERVAL (3 here, 1 in args.py).
# Set PERCEPTION_MASK_MIX=M0 to reproduce a pre-2026-08-14 arm.
PERCEPTION_MASK_MIX="${PERCEPTION_MASK_MIX:-M2}"
# printf, not echo: `echo | tr -c` turns the trailing newline into a '_' as well, which is
# where the double underscore in the historical outputs/arm0__seed42_fi3 came from. The A
# suffix already breaks byte-identity with those directories (see the legacy note under OUT),
# so the stray separator is dropped in the same breath rather than carried forever.
SLUG="$(printf '%s' "$ARM" | tr -c '[:alnum:]+' '_')"
LEGACY_SLUG="$(echo "$ARM" | tr -c '[:alnum:]+' '_')"   # pre-2026-09-19 spelling, for the note below
# The A axis goes in the OUTPUT PATH and the run name, on EVERY run including A0 -- for exactly
# the reason the interval, the tree and the seg protocol do: it changes what the arm IS. It is
# also the axis most likely to be flipped, since flipping it IS ablation (a). It used to be
# suppressed at A0 and kept out of the path, to preserve byte-identical arm0-arm6 run names;
# the cost of that was `ACTION_MASK_MIX=A0 train_arm.sh arm7` landing in arm7's own directory
# and resuming its A1 weights under an A0 label -- a hybrid no metric can be attributed to.
# The byte-identity is gone instead: see the legacy-directory note under OUT.
A_LOWER="$(printf '%s' "$ACTION_MASK_MIX" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]' '_')"
A_SUFFIX="_${A_LOWER}"        # outputs/arm7_a1_seed42_...
A_SUFFIX_DASH="-${A_LOWER}"   # wandb run name arm7-a1-seed42-...
# The interval goes in the output path, because it changes what the arm IS. Without it,
# FRAME_INTERVAL=3 on an existing arm0 directory would hit the resume path below and silently
# continue the 20 Hz run instead of starting the 6.7 Hz one -- the exact failure the resume
# warning further down exists to prevent. FI=1 keeps the historical paths byte-identical.
FI_SUFFIX=""
[ "$FRAME_INTERVAL" != "1" ] && FI_SUFFIX="_fi${FRAME_INTERVAL}"
# The tree goes in the path for the same reason the interval does: it changes what the arm IS.
# Without it a 512_aug run lands in arm0/arm1's existing 256 directories, hits the resume branch
# below, and silently continues from 256-trained weights -- producing a hybrid that no r_peak
# reading can be attributed to anything. The historical 256 tree keeps the bare path so the
# existing arm0/arm1 directories still resume.
DS_SUFFIX=""
[ "$DATASET" != "rlbench_selfgen" ] && DS_SUFFIX="_${DATASET#rlbench_selfgen_}"
# Multi-source joint runs pass DATASET as a full CombDataset spec ("a@0.4,b@0.2,c@0.4").
# The @ratio must reach --dataset_name untouched (no @1.0 appended), and the raw spec is
# unusable as a path/run-name fragment -- stamp a fixed _joint marker instead (set OUT=
# explicitly for anything fancier).
if [[ "$DATASET" == *"@"* ]]; then
  DATASET_ARG="$DATASET"
  DS_SUFFIX="_joint"
else
  DATASET_ARG="${DATASET}@1.0"
fi
# The seg protocol goes in the path for exactly the reason the interval and the tree do: it
# changes what the arm IS. Without it a scene_roles arm1 lands in a referring arm1's directory,
# hits the resume branch below, and silently continues from weights trained against a different
# target -- a hybrid no metric can be attributed to anything. `referring` keeps the bare path so
# the existing arm0/arm1 directories still resume.
SEG_SUFFIX=""
[ "$SEG_MODE" = "scene_roles" ] && SEG_SUFFIX="_sr"
OUT_WAS_SET="${OUT:+1}"
OUT="${OUT:-$REPO/outputs/${SLUG}${A_SUFFIX}_seed${SEED}${FI_SUFFIX}${DS_SUFFIX}${SEG_SUFFIX}}"
# Every directory written before the A suffix existed (2026-09-19) lacks it, so the default OUT
# no longer points at them: re-running such an arm STARTS A NEW RUN rather than resuming. That
# is the safe direction -- a fresh directory is recoverable, a silent hybrid is not -- but it is
# invisible, so say it out loud. The two in-flight joint runs pass OUT= explicitly and are
# unaffected.
if [ -z "$OUT_WAS_SET" ]; then
  LEGACY_OUT="$REPO/outputs/${LEGACY_SLUG}_seed${SEED}${FI_SUFFIX}${DS_SUFFIX}${SEG_SUFFIX}"
  if [ ! -d "$OUT" ] && [ -d "$LEGACY_OUT" ]; then
    echo "note: $LEGACY_OUT exists and predates the _${A_LOWER} suffix -- this run will NOT resume it."
    echo "      To continue that one instead:  OUT=$LEGACY_OUT bash scripts/train_arm.sh $ARM"
  fi
fi
NPROC=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
# Stage 1 default. Empty string = from the Wan base (stage 2, see header).
# Warm start. Default is repo-relative so a fresh clone works; fetch it once with
#   huggingface-cli download anyeZHY/ActionImages step125750.ckpt --local-dir checkpoints/official
INIT_CKPT="${INIT_CKPT-$REPO/checkpoints/official/step125750.ckpt}"
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
CKPT_EVERY="${CKPT_EVERY:-2000}"
SAVE_TOP_K="${SAVE_TOP_K:--1}"
# SAVE_OPTIM=False drops DeepSpeed's global_step*/ so a checkpoint is 12GB instead of 120GB
# and a save can never fill the volume mid-write (which killed both joint runs on
# 2026-09-17). Crash-resume then restarts Adam moments and needs --allow_step_restart.
SAVE_OPTIM="${SAVE_OPTIM:-True}"

# /workspace is a shared 17T volume that outside tenants fill without warning, and it has been
# observed swinging between 37GB and 265GB free within minutes (ttd DECISIONS.md D-038).
# The hard gate is deliberately only "can this run write its next few checkpoints" -- the
# operator prunes evaluated checkpoints as the run proceeds, so demanding the full projection
# up front would block a run that is actually fine. The projection is still printed, loudly,
# because running out at step 7000 costs a day.
if [ "$SAVE_TOP_K" -gt 0 ]; then KEPT="$SAVE_TOP_K"; else KEPT=$(( STEPS / CKPT_EVERY )); fi
# A checkpoint DIRECTORY is 120G while it is the resume tip and ~12.8G afterwards, so the
# projection has to model the pruner, not just multiply:
#   stepN.ckpt      12.8G   DiT-only bf16 weights -- what eval and warm-start read
#   global_stepN/  108.0G   ZeRO-2 fp32 master + Adam m/v -- resume state, ONLY for that step
# train.py's `_prune_resume_state` (on by default via --keep_optimizer_last_only) drops
# `global_step*/` from every checkpoint except the newest as soon as the next one lands, so the
# steady state is (KEPT-1) small + 1 full, NOT KEPT full. Measured 2026-09-04 on
# arm7/checkpoint-1000 and specialist_action/checkpoint-4000; tests/test_checkpoint_pruning.py
# pins which files survive.
#
# The old formula (KEPT * 13 + 20) undercounted anyway: it used 13G for every checkpoint and so
# ignored the 108G tip entirely.
CKPT_G="${CKPT_G:-13}"          # a pruned (non-tip) checkpoint
TIP_G="${TIP_G:-120}"           # the newest checkpoint, which still carries the optimizer state
PROJECTED_G=$(( (KEPT - 1) * CKPT_G + TIP_G + 20 ))
FLOOR_G="${FLOOR_G:-80}"        # ~6 checkpoints of headroom before pruning becomes urgent
AVAIL_G=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
echo "df /workspace avail = ${AVAIL_G}G"
echo "checkpoints: every ${CKPT_EVERY} steps, keep ${SAVE_TOP_K} -> ${KEPT} total"
echo "             ~${PROJECTED_G}G = $(( KEPT - 1 )) x ~${CKPT_G}G (pruned) + 1 x ~${TIP_G}G (resume tip)"
echo "             pass --keep_optimizer_last_only False in EXTRA_ARGS to keep every checkpoint"
echo "             independently resumable instead -- that costs ${KEPT} x ~${TIP_G}G."
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
echo "arm=$ARM mix='$MIX' seg_mode=$SEG_MODE dataset=$DATASET GPUS=$GPUS NPROC=$NPROC SEED=$SEED STEPS=$STEPS RES=$RES PORT=$PORT OUT=$OUT"
# The two mask axes partition the menu: M applies only to action-FREE templates, A only to
# templates containing <action>. A menu made entirely of one kind leaves the other flag doing
# nothing at all, and printing it unqualified reads as though it were in effect -- the same class
# of misleading log line the "weights: RESUME" branch below exists to prevent.
# Counted in pure bash rather than with `grep -qv`: /usr/bin/grep on this box is ugrep, whose
# `-q -v` returns 1 even when non-matching lines exist, so the obvious one-liner reports arm6's
# mixed menu as "every template contains <action>". Verified 2026-09-04 (ugrep 7.5.0).
PERCEP_NOTE="" ; ACTION_NOTE=""
n_total=0 ; n_action=0
for _t in ${MIX//,/ }; do
  _t="${_t%%@*}"
  [ -z "$_t" ] && continue
  n_total=$((n_total + 1))
  case "$_t" in *action*) n_action=$((n_action + 1)) ;; esac
done
[ "$n_action" -eq "$n_total" ] && PERCEP_NOTE=" (INERT: all $n_total templates contain <action>)"
[ "$n_action" -eq 0 ] && ACTION_NOTE=" (INERT: no template in this menu contains <action>)"
echo "mask axes: perception_mask_mix=$PERCEPTION_MASK_MIX$PERCEP_NOTE  action_mask_mix=$ACTION_MASK_MIX$ACTION_NOTE"
echo "frame_interval=$FRAME_INTERVAL -> window spans $(python -c "print(f'{40*$FRAME_INTERVAL/20:.1f}')")s of motion at $(python -c "print(f'{20/$FRAME_INTERVAL:.1f}')") Hz"
# Report what the weights will ACTUALLY come from, not just what got passed on the command
# line. train.py prefers resume_ckpt_path (found in output_dir) over init_ckpt_path, so
# printing INIT_CKPT unconditionally reads as "starting from the warm-start checkpoint" even
# when the run is really continuing from its own latest checkpoint.
RESUME_CKPT=""
[ -n "$RESUME_FROM" ] && RESUME_CKPT=$(ls "$RESUME_FROM"/step*.ckpt 2>/dev/null | head -1)
if [ -n "$RESUME_CKPT" ]; then
  echo "weights: RESUME from $RESUME_CKPT"
  echo "         (--init_ckpt_path ${INIT_CKPT:-<none>} is passed but IGNORED -- resume wins)"
else
  echo "weights: warm-start from ${INIT_CKPT:-<Wan base, no warm-start (stage 2)>}"
fi

# zero2_offload, not zero.json: full_param=True on a 5B model needs the Adam states on CPU to
# fit two 48GB cards at 256^2. This is the config the 14.5 s/it throughput figure was measured
# with (ttd runs/A4_stage2_full_v12seg, same GPUs 4+7). Plain ZeRO-2 keeps fp32 master weights
# and Adam moments resident and does not fit.
DS_CONFIG="${DS_CONFIG:-./configs/zero2_offload.json}"

CUDA_VISIBLE_DEVICES=$GPUS torchrun --nnodes=1 --nproc_per_node=$NPROC --master_port $PORT \
  train.py \
  --deepspeed "$DS_CONFIG" \
  --dataset_path ./data \
  --dataset_name "$DATASET_ARG" \
  --template_mix "$MIX" \
  --segmentation_mode "$SEG_MODE" \
  --variations "$VARIATIONS" \
  $INIT_ARG \
  --output_dir "$OUT" \
  --height "$RES" --width "$RES" --num_frames 41 --frame_interval "$FRAME_INTERVAL" \
  --perception_mask_mix "$PERCEPTION_MASK_MIX" \
  --action_mask_mix "$ACTION_MASK_MIX" \
  --full_param True \
  --model_id Wan-AI/Wan2.2-TI2V-5B \
  --max_steps "$STEPS" --num_train_epochs 1 --steps_per_epoch "$STEPS" \
  --learning_rate 5e-7 --warmup_steps 1000 --lr_scheduler_type constant_with_warmup \
  --gradient_accumulation_steps 1 --max_grad_norm 1.0 \
  --use_gradient_checkpointing \
  --dataloader_num_workers 4 --dataloader_prefetch_factor 2 --dataloader_pin_memory True \
  --checkpoint_every_n_steps "$CKPT_EVERY" --checkpoint_save_top_k "$SAVE_TOP_K" \
  --save_optimizer_state "$SAVE_OPTIM" \
  --remove_unused_columns False --dataloader_drop_last True \
  --prediction_loss_only True --bf16 True --ddp_find_unused_parameters False \
  --save_safetensors False --per_device_train_batch_size 1 \
  --logging_steps 10 --seed "$SEED" \
  --report_to wandb --run_name "${SLUG}${A_SUFFIX_DASH}-seed${SEED}${DS_SUFFIX//_/-}" ${EXTRA_ARGS:-}
TRAIN_EXIT=$?
echo "TRAIN_EXIT=$TRAIN_EXIT"
exit "$TRAIN_EXIT"
