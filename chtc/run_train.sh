#!/bin/bash
# In-job wrapper for the ActionImages-Cogen fusion arm on CHTC.
#   run_train.sh <arm> <exp_name> <steps>
#
# Responsibilities, in order of how badly each bites if omitted:
#  1. SELF-RESUME. GPU Lab jobs get evicted. On start this looks for a previous bundle for the
#     same exp_name in /staging and unpacks it, so a 7-day run that is evicted on day 3 continues
#     instead of restarting. Without this, long runs never finish on a shared cluster.
#  2. BUNDLE ON TERM. Checkpoints live in job scratch, which is destroyed on exit. The trap
#     packages them so HTCondor can transfer them out on eviction as well as on success.
#  3. Stage inputs out of the transferred tarballs and into the layout the repo expects.
#  4. Apply the DeepSpeed bf16 non-finite-gradient patch, which a pip install does not carry.
set -uo pipefail

ARM="${1:?usage: run_train.sh <arm> <exp_name> <steps>}"
EXP="${2:?usage: run_train.sh <arm> <exp_name> <steps>}"
STEPS="${3:-20000}"
SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
STAGING="${STAGING_DIR:-/staging/tdu35}"
BUNDLE="checkpoint_bundle.tar"

cd /app
export PYTHONPATH=/app HOME="$SCRATCH"
export HF_HOME="$SCRATCH/.cache/hf" XDG_CACHE_HOME="$SCRATCH/.cache"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME"

echo "=== fusion arm on CHTC: arm=$ARM exp=$EXP steps=$STEPS ==="
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv || true

NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
GPUS=$(seq -s, 0 $((NGPU-1)))
echo "visible GPUs: $NGPU ($GPUS)"

# ---- 1. inputs -----------------------------------------------------------------------------
# Tarballs arrive in scratch via transfer_input_files. Extract then delete, so peak disk is one
# copy rather than two -- a 137 GB dataset makes that the difference between fitting and evicting.
for tb in *.tar.gz *.tar; do
  [ -f "$tb" ] || continue
  case "$tb" in "$BUNDLE") continue ;; esac
  echo "extracting $tb"
  case "$tb" in *.tar.gz) tar -xzf "$tb" ;; *) tar -xf "$tb" ;; esac
  rm -f "$tb"
done

# The repo reads data and the base model through these two paths.
mkdir -p /app/data /app/checkpoints/Wan-AI
for d in rlbench_selfgen_512_aug_wide rlbench_unseen_tasks_512_aug; do
  [ -d "$SCRATCH/$d" ] && ln -sfn "$SCRATCH/$d" "/app/data/$d"
done
[ -d "$SCRATCH/Wan2.2-TI2V-5B" ] && ln -sfn "$SCRATCH/Wan2.2-TI2V-5B" /app/checkpoints/Wan-AI/Wan2.2-TI2V-5B
echo "data: $(ls -l /app/data 2>/dev/null | wc -l) entries; base model: $([ -e /app/checkpoints/Wan-AI/Wan2.2-TI2V-5B ] && echo present || echo MISSING)"

# ---- 2. the DeepSpeed bf16 patch ------------------------------------------------------------
# Upstream DeepSpeed's bf16 path does not skip non-finite gradients, which surfaces as a
# "Got Long" crash rather than as a skipped step. The repo ships the patch; it must be re-applied
# after any pip install of deepspeed, which includes this image.
# Copied into chtc/ from /workspace/ttdu/ActionImages/scripts/ (the aligned source, where it was
# originally verified) -- ActionImages-Cogen did not carry it.
if [ -f chtc/patch_deepspeed_bf16_overflow.py ]; then
  python chtc/patch_deepspeed_bf16_overflow.py || { echo "!! deepspeed bf16 patch FAILED"; exit 5; }
else
  echo "!! chtc/patch_deepspeed_bf16_overflow.py missing -- bf16 NaN grads will not be skipped"; exit 5
fi

# ---- 3. self-resume from a previous eviction ------------------------------------------------
OUT="/app/outputs/${EXP}"
PREV="$STAGING/fusion/${EXP}.tar"
if [ -f "$PREV" ]; then
  echo "found previous bundle $PREV -- resuming instead of restarting"
  mkdir -p /app/outputs && tar -xf "$PREV" -C /app/outputs
  ls -d "$OUT"/checkpoint-* 2>/dev/null | tail -3
else
  echo "no previous bundle; this is a fresh start for exp=$EXP"
fi

package() {
  [ -d "$OUT" ] || { echo "no $OUT to package"; return 0; }
  [ -f "$BUNDLE" ] && return 0
  echo "packaging $OUT -> $BUNDLE"
  # Keep ONLY the newest global_step*: the resume tip is ~108 GB and older ones can never be
  # resumed from anyway. Without this the bundle is untransferable.
  keep=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
  for d in "$OUT"/checkpoint-*/; do
    n=$(basename "$d" | sed 's/checkpoint-//')
    [ "$n" = "$keep" ] || rm -rf "$d"global_step* 2>/dev/null
  done
  tar -cf "$BUNDLE" -C /app/outputs "$(basename "$OUT")"
  echo "bundle: $(du -sh "$BUNDLE" | cut -f1)"
}
DONE=0
trap '[ "$DONE" -eq 1 ] || package' EXIT TERM INT

# ---- 4. train ------------------------------------------------------------------------------
# GPUS/STEPS/OUT are passed through; GRAD_ACCUM is derived inside train_fusion.sh so the global
# batch stays 4 whether CHTC gave us 1, 2 or 4 cards.
GPUS="$GPUS" \
DATASET="${DATASET:-rlbench_selfgen_512_aug_wide}" \
STEPS="$STEPS" \
OUT="$OUT" \
INIT_CKPT="${INIT_CKPT:-$SCRATCH/step5000.ckpt}" \
CKPT_EVERY="${CKPT_EVERY:-1000}" \
EXTRA_ARGS="${EXTRA_ARGS:-}" \
  bash scripts/train_fusion.sh "$ARM"
rc=$?
echo "TRAIN_RC=$rc"

package
DONE=1
trap - EXIT TERM INT
exit "$rc"
