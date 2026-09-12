#!/bin/bash
# Download -> rollout -> delete, one checkpoint at a time.
#
# A 12.8 GB checkpoint times six does not fit next to everything else on a shared 18T volume
# that other tenants fill without warning (a previous run died with NoSpaceLeftError at
# rollout 78 of 100). Holding one checkpoint at a time keeps the campaign unattended-safe.
#
#   bash scripts/rollout_campaign.sh <gpu> <repo> <steps...>
#   bash scripts/rollout_campaign.sh 1 TingtingDu/arm4__seed42_fi3 5000 10000
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"; export PYTHONPATH="$REPO_ROOT"
source /workspace/ttdu/ttd/scripts/env_eval.rc

GPU="${1:?gpu}"; HFREPO="${2:?hf repo}"; shift 2
TASKS="${TASKS:-close_drawer close_box open_drawer push_buttons meat_off_grill}"
TRIALS="${TRIALS:-20}"
RES="${RES:-256}"; CFG="${CFG:-7.5}"; FI="${FI:-3}"; TAGSTYLE="${TAGSTYLE:-explicit}"
SOLVER="${SOLVER:-sphere}"
NAME="$(basename "$HFREPO")"

for STEP in "$@"; do
  CK="outputs/${NAME}/checkpoint-${STEP}/step${STEP}.ckpt"
  if [ ! -f "$CK" ]; then
    echo "[campaign] downloading ${HFREPO} checkpoint-${STEP} ..."
    conda run -n ttd_train python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('${HFREPO}','checkpoint-${STEP}/step${STEP}.ckpt',local_dir='outputs/${NAME}')" || { echo "[campaign] download FAILED"; continue; }
  fi
  AVAIL=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
  echo "[campaign] ${NAME}@${STEP} solver=${SOLVER} fi=${FI} res=${RES}  (disk ${AVAIL}G free)"
  CUDA_VISIBLE_DEVICES=$GPU xvfb-run -a conda run -n ttd_rollout python -u eval/rollout.py \
    --ckpt "$CK" --tag "${NAME}_${STEP}${TAG_SUFFIX:-}" --res "$RES" --cfg "$CFG" \
    --frame-interval "$FI" --prompt-tag-style "$TAGSTYLE" --axis-solver "$SOLVER" \
    --arm-action-mode planning --max-steps-factor 1.5 --skip-anchor-frames 4 --record-video \
    --tasks $TASKS --num-trials "$TRIALS" ${EXTRA:-}
  echo "[campaign] done ${NAME}@${STEP} (exit $?)"
  # Keep only what the next iteration needs.
  if [ "${KEEP_CKPT:-0}" != "1" ]; then rm -rf "outputs/${NAME}/checkpoint-${STEP}"; echo "[campaign] removed $CK"; fi
done
echo "CAMPAIGN_DONE ${NAME}"
