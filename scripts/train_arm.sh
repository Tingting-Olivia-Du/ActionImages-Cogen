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
#   bash scripts/train_arm.sh arm4     # 60% action / 40% depth
#   bash scripts/train_arm.sh arm5     # 60% action / 40% surface normal
#   bash scripts/train_arm.sh arm6     # 40% action / 20% each depth+seg+normal (4 streams)
#   bash scripts/train_arm.sh arm7     # action in EVERY template, observation in 4 modalities
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
# DATA TREE: selected with DATASET=, and it is NOT a free knob -- it changes what the arm is
# comparable to. Arms may only be compared within one tree.
#
#   rlbench_selfgen_512_aug  (default) 512x512, Colosseum-style domain randomisation: camera
#                            pose, table colour/texture, background texture, light colour.
#                            1028 episodes / 788 at variation0. The only self-gen tree whose
#                            visual diversity approaches the official release.
#   rlbench_selfgen          256x256, the v2 tree arm0/arm1 were trained on. Background wall
#                            only -- fixed cameras, fixed table, fixed lighting.
#   rlbench_selfgen_512      512x512 un-augmented. DELETED to free disk; the symlink dangles.
#
# RESOLUTION follows the tree and is derived below, because training at anything other than the
# rendered resolution is silently wasteful in one direction (256 tree at 512 upsamples for ~4x
# the attention cost and zero extra information) and lossy in the other. Override with RES= only
# to test that claim. NOTE the official checkpoint was trained at 512, so a 256 arm pays a
# 512-prior -> 256-domain adaptation that a 512 arm does not: absolute r_peak is not comparable
# across trees, which is why the tree is stamped into OUT and the run name.
#
# TEMPORAL SPAN: FRAME_INTERVAL controls how much MOTION a 41-frame window covers, which
# --num_frames alone does not. The selfgen tree is written at native 20 Hz 1:1, so the default
# FRAME_INTERVAL=1 gives 2.0 s per window; the official release is effectively 4 (its video is
# stored 4x-downsampled, actions realigned with actions[::4]) and covers 8.0 s. arm0/arm1 were
# trained at 1 -- do not change it on those. A different interval writes to a different OUT
# (suffix _fi<N>) so it cannot silently resume an arm trained at another rate.
# no, this is not right
#   FRAME_INTERVAL=3 bash scripts/train_arm.sh arm0   # 6.0 s windows at 6.7 Hz
#
# Env overrides:  DATASET, GPUS, SEED, STEPS, RES, OUT, PORT, INIT_CKPT, VARIATIONS,
#                 FRAME_INTERVAL, SEG_MODE, PERCEPTION_MASK_MIX, ACTION_MASK_MIX, CKPT_EVERY,
#                 SAVE_TOP_K, DS_CONFIG, EXTRA_ARGS
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
  # SEG_MODE_DEFAULT is part of the arm DEFINITION, not a preference: `referring` (a target-only
  # mask, median 0.18% non-black) and `scene_roles` (a dense role map, 11.8%) are different
  # supervision signals that happen to share a modality name. An arm that does not record which
  # one it meant is not reproducible. Override for one run with SEG_MODE=.
  arm1) MIX="video+action@0.6,video+depth@0.2,video+segmentation@0.2"; SEG_MODE_DEFAULT=scene_roles ;;
  arm2) MIX="video+action@0.5,video+depth+action@0.5" ;;
  arm4) MIX="video+action@0.6,video+depth@0.4" ;;
  # arm5 mirrors arm4 with normal in depth's place, so the pair isolates one modality swap at a
  # fixed action ratio. NOTE outputs/arm4__seed42_fi3 was trained on the DELETED 256 v2 tree (no
  # _512_aug in its path), so it is NOT comparable to a 512_aug arm5 -- that reading needs a
  # depth-only arm rerun on this tree first.
  arm5) MIX="video+action@0.6,video+normal@0.4" ;;
  # arm6 is the headline versatile model: one checkpoint, four streams selected by prompt tag.
  # Four auxiliary streams is the CEILING on 788 variation0 episodes -- Argus (CVPR 2025) Tab.13
  # loses accuracy on every task once its core task set outgrows its data, and we have three
  # orders of magnitude less data than it does.
  arm6) MIX="video+action@0.4,video+depth@0.2,video+segmentation@0.2,video+normal@0.2"; SEG_MODE_DEFAULT=scene_roles ;;
  # arm7 inverts arm6's bet. arm6 spends 60% of its samples on PERCEPTION templates, which carry
  # no action segment at all -- so counting action dropout only 0.4*0.9 = 36% of its steps produce
  # any action gradient, and it has already plateaued (6000->8000 flat on train AND held-out).
  # arm7 keeps <action> in EVERY template and varies only which visual space the observation lives
  # in: 90% of its steps carry action, at IDENTICAL cost, because all four templates are still
  # four segments of the same shape (V0|A0|V1|A1, D0|A0|D1|A1, S0|A0|S1|A1, N0|A0|N1|A1).
  #
  # THE CONTROL IS arm0, NOT arm6. outputs/specialist_action__seed42_fi3_512_aug_sr is
  # video+action@1.0 on the same warm start / seed / tree / interval / lr, so it has the SAME 90%
  # action-sample rate with one modality; arm7 - arm0 is "four observation modalities vs one at a
  # fixed action budget". arm7 - arm6 confounds the menu with a 2.5x action-budget change.
  #
  # WHAT arm7 GIVES UP: there is no video+X template left, so this checkpoint cannot be asked
  # RGB->depth/seg/normal. eval/eval_perception.py and eval/eval_all_masks.py would be reading it
  # off-distribution, and --perception_mask_mix is INERT here (nothing reaches the action-free
  # branch of plan_segments). Read it with eval/eval_action.py --template <X>+action instead.
  # arm7u = arm7 的均匀采样对照。只改一个东西:四个模板等比例(0.25 x 4)而不是 0.4/0.2/0.2/0.2。
  # 为什么要它:arm7@8000 实测四条路径【不】等价 —— normal 比 RGB 差 1.36x(n=80, p=0.005),
  # depth 1.25x / seg 1.18x(p=0.093)。最朴素的解释是【更新次数不平衡】:arm7 里 RGB 占 0.4,
  # 其余各 0.2,RGB 的更新次数是它们的两倍。均匀采样把这个解释直接消掉。
  # 这是 CROSSMODAL_SUPERVISION_RESEARCH.md §6.1 排在所有机制之前的对照 —— 零代码,一个参数,
  # 必须先排除它,才轮得到梯度调制/互教那些更贵的方案。
  # 其余一切(A1 mask 轴、scene_roles、fi=3、512、variations 0、seed、warm start、lr)与 arm7 相同。
  arm7u) MIX="video+action@0.25,depth+action@0.25,segmentation+action@0.25,normal+action@0.25"
        SEG_MODE_DEFAULT=scene_roles
        ACTION_MASK_MIX_DEFAULT=A1 ;;
  arm7) MIX="video+action@0.4,depth+action@0.2,segmentation+action@0.2,normal+action@0.2"
        SEG_MODE_DEFAULT=scene_roles
        # Part of the arm DEFINITION, for the same reason SEG_MODE_DEFAULT is: A1 spends FIII (the
        # cross-view mode, least relevant of the four to action quality) on policy mode, whose loss
        # lands entirely on the action segments under the same single-frame conditioning the
        # closed-loop rollout provides. Override with ACTION_MASK_MIX= to sweep the axis -- and
        # use an explicit OUT= when you do, because the mix is NOT in the output path.
        ACTION_MASK_MIX_DEFAULT=A1 ;;
  *)    MIX="$ARM" ;;
esac
SEG_MODE="${SEG_MODE:-${SEG_MODE_DEFAULT:-referring}}"
# The A axis (training/templates.py ACTION_MASK_MIX_PRESETS): how a template CONTAINING <action>
# splits its conditioning, as (iiii, fiii, fifi, policy). A0 = the upstream literals, which is what
# arm0-arm6 and the official step125750 were all trained under; the library default in
# templates.py stays A0 so those stay reproducible and test_forward_unchanged.py keeps its
# meaning. An arm that wants something else declares it in its case above -- same split of
# responsibilities as FRAME_INTERVAL (3 here, 1 in args.py) and PERCEPTION_MASK_MIX (M2/M0).
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
SLUG="$(echo "$ARM" | tr -c '[:alnum:]+' '_')"
# The A axis goes in the RUN NAME (not the output path) when it is not the default, so an
# A-axis sweep is distinguishable in wandb without renaming the arm's directory. Suppressed at
# A0 so every arm0-arm6 run name stays byte-identical to what it was before this axis existed.
AMM_SUFFIX=""
[ "$ACTION_MASK_MIX" != "A0" ] && \
  AMM_SUFFIX="-$(printf '%s' "$ACTION_MASK_MIX" | tr -c '[:alnum:]' '-')"
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
# The seg protocol goes in the path for exactly the reason the interval and the tree do: it
# changes what the arm IS. Without it a scene_roles arm1 lands in a referring arm1's directory,
# hits the resume branch below, and silently continues from weights trained against a different
# target -- a hybrid no metric can be attributed to anything. `referring` keeps the bare path so
# the existing arm0/arm1 directories still resume.
SEG_SUFFIX=""
[ "$SEG_MODE" = "scene_roles" ] && SEG_SUFFIX="_sr"
OUT="${OUT:-$REPO/outputs/${SLUG}_seed${SEED}${FI_SUFFIX}${DS_SUFFIX}${SEG_SUFFIX}}"
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
CKPT_EVERY="${CKPT_EVERY:-2000}"
SAVE_TOP_K="${SAVE_TOP_K:--1}"

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
  --dataset_name "${DATASET}@1.0" \
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
  --remove_unused_columns False --dataloader_drop_last True \
  --prediction_loss_only True --bf16 True --ddp_find_unused_parameters False \
  --save_safetensors False --per_device_train_batch_size 1 \
  --logging_steps 10 --seed "$SEED" \
  --report_to wandb --run_name "${SLUG}-seed${SEED}${DS_SUFFIX//_/-}${AMM_SUFFIX}" ${EXTRA_ARGS:-}
TRAIN_EXIT=$?
echo "TRAIN_EXIT=$TRAIN_EXIT"
exit "$TRAIN_EXIT"
