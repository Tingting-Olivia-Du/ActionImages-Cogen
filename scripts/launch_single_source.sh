#!/bin/bash
# Launch ONE single-source run: one arm, one simulator's data only, warm-started from the
# released ActionImages checkpoint, ACTION_MASK_MIX=A1, 4000 steps, a checkpoint every 2000.
#
#   bash scripts/launch_single_source.sh <arm1|arm2|arm3> <rlbench|maniskill> <gpus> [port]
#   bash scripts/launch_single_source.sh arm1 rlbench   0,1,2,3
#   bash scripts/launch_single_source.sh arm1 maniskill 4,5,6,7
#
# SINGLE SOURCE, NOT A MIX. The earlier runs trained on rlbench@0.7 + maniskill@0.3 in one
# stream (ABLATION_HANDOFF.md); these train on exactly one tree at 100%, so a run's result can be
# attributed to one simulator's data. DATASET is therefore a bare tree name, which train_arm.sh
# turns into "<tree>@1.0".
#
# What this wrapper adds over calling train_arm.sh directly -- each line is a failure that has
# already happened on the origin machine:
#   * train_arm.sh does not activate an environment; it runs whatever `python` is on PATH.
#     The preflight below imports what training needs FROM THE REPO ROOT (train.py's cwd), where
#     a stale `wandb/` log directory once shadowed a missing wandb package and made a check pass.
#   * a non-empty OUT makes train_arm.sh RESUME from it and silently ignore INIT_CKPT, so an OUT
#     that already holds checkpoints is refused unless RESUME=1 is set on purpose.
#   * ACTION_MASK_MIX is passed explicitly rather than trusted to the per-arm default.
#   * the ManiSkill loader PINS stride 1 whatever --frame_interval says; passing 1 keeps the
#     startup banner honest (it otherwise claims a 6.0 s window that is really 2.05 s).
#   * detached with setsid, not nohup: a torn-down parent process group SIGTERMs torchrun.
set -uo pipefail
ARM="${1:?arm1|arm2|arm3}"; SRC="${2:?rlbench|maniskill}"; GPUS_ARG="${3:?e.g. 0,1,2,3}"
PORT_ARG="${4:-$(( 29600 + RANDOM % 300 ))}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 2
[ -n "${ENV_RC:-}" ] && source "$ENV_RC"
unset CUDA_VISIBLE_DEVICES

case "$ARM" in arm1|arm2|arm3) ;; *) echo "!! arm must be arm1, arm2 or arm3"; exit 2 ;; esac
case "$SRC" in
  rlbench)   DATASET=rlbench_selfgen_512_aug_wide; FI=3; CENSUS="3960/4200 episodes across 16 tasks" ;;
  maniskill) DATASET=maniskill3;                  FI=1; CENSUS="1750/1925 episodes across 7 tasks" ;;
  *) echo "!! source must be rlbench or maniskill"; exit 2 ;;
esac
OUT="$REPO/outputs/${ARM}_${SRC}only_fromofficial_a1_4k"
INIT="$REPO/checkpoints/official/step125750.ckpt"

# ---- preflight: refuse to start rather than die after the 12.8 GB load
fail() { echo "!! preflight: $*"; exit 3; }
python - <<'PY' || fail "the python on PATH cannot import the training stack (see above)"
import transformers, deepspeed, diffsynth  # noqa: F401
from transformers.integrations import is_wandb_available
print(f"preflight: transformers {transformers.__version__}  deepspeed {deepspeed.__version__}  "
      f"wandb_available={is_wandb_available()}")
PY
[ -f "$INIT" ] || fail "missing warm-start checkpoint $INIT"
ls checkpoints/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-0000*-of-00003.safetensors >/dev/null 2>&1 \
  || fail "missing Wan2.2-TI2V-5B base model under checkpoints/Wan-AI/ (see guide, weights)"
[ -d "data/$DATASET" ] || fail "missing data/$DATASET"
if ls -d "$OUT"/checkpoint-* >/dev/null 2>&1 && [ "${RESUME:-0}" != 1 ]; then
  fail "$OUT already has checkpoints -- train_arm.sh would RESUME and ignore the warm start. "\
"Delete it, or set RESUME=1 if continuing it is what you mean."
fi

# ---- W&B: OFFLINE by default. This machine has no online W&B access, so runs are written
# locally and uploaded afterwards from a machine that has a key:
#     wandb sync outputs/wandb_offline/wandb/offline-run-*
# WANDB_DIR is a dedicated directory rather than the repo root, because the zipped repo may
# carry an old `wandb/` log directory and the new offline runs must not get mixed into it.
# Set WANDB_MODE=online (with a key) or WANDB_MODE=disabled to override.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$REPO/outputs/wandb_offline}"; mkdir -p "$WANDB_DIR"
export WANDB_PROJECT="${WANDB_PROJECT:-actionimages-cogen}"
RUN="${ARM}-${SRC}only-fromofficial-a1-4k"
if [ "$WANDB_MODE" = disabled ]; then WB="--report_to none"; else WB="--run_name $RUN"; fi
# EFFECTIVE BATCH MUST STAY 4 (= GPUs x 1 x accum). With fewer than four cards per run, set
# ACCUM so that n_gpus * ACCUM = 4 -- a run at batch 2 is not comparable with the others.
NG=$(echo "$GPUS_ARG" | tr ',' '\n' | grep -c .)
ACCUM="${ACCUM:-$(( 4 / NG ))}"
[ $(( NG * ACCUM )) -eq 4 ] || fail "GPUs ($NG) x ACCUM ($ACCUM) = $(( NG * ACCUM )), must be 4"
[ "$ACCUM" -gt 1 ] && WB="$WB --gradient_accumulation_steps $ACCUM"
echo "W&B: mode=$WANDB_MODE dir=$WANDB_DIR run=$RUN"

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "DRY_RUN: preflight passed. Would launch:"
  echo "  DATASET=$DATASET FRAME_INTERVAL=$FI GPUS=$GPUS_ARG (x accum $ACCUM = batch 4) STEPS=4000 CKPT_EVERY=2000 SAVE_OPTIM=False"
  echo "  ACTION_MASK_MIX=A1 INIT_CKPT=$INIT"
  echo "  OUT=$OUT"
  echo "  EXTRA_ARGS=--allow_step_restart $WB"
  exit 0
fi
LOGDIR="$REPO/outputs/single_source_logs"; mkdir -p "$LOGDIR"
DATASET="$DATASET" VARIATIONS=0 FRAME_INTERVAL="$FI" RES=512 \
GPUS="$GPUS_ARG" PORT="$PORT_ARG" SEED=42 STEPS=4000 CKPT_EVERY=2000 SAVE_TOP_K=-1 \
SAVE_OPTIM=False ACTION_MASK_MIX=A1 INIT_CKPT="$INIT" OUT="$OUT" \
EXTRA_ARGS="--allow_step_restart $WB" \
  setsid nohup bash "$REPO/scripts/train_arm.sh" "$ARM" > "$LOGDIR/${ARM}_${SRC}.log" 2>&1 < /dev/null &
echo "launched $ARM on $SRC (GPUs $GPUS_ARG, port $PORT_ARG) -> $OUT"
echo "  log:  $LOGDIR/${ARM}_${SRC}.log"
echo "  expect in the log: action_mask_mix=A1 | frame_interval=$FI | warm-start from $INIT"
echo "                     [selfgen] variations='0': kept $CENSUS | then a {'loss': ...} line"
