#!/bin/bash
# Evaluate a specialist at its MATCHED-EXPOSURE checkpoint as soon as that checkpoint lands.
#
#   ARM=normal MODS=normal STEP=2000 nohup bash scripts/wait_and_eval_specialist.sh &
#
# STEP is the matched-exposure point, not the end of training (EXPERIMENT_OUTLINE.md E2): the
# unified model gets 0.2T updates on each perception modality over its 10000 steps, so a
# perception specialist is compared at its step-2000 checkpoint. That checkpoint arrives long
# before the 4000-step run ends, so tab:headline does not have to wait for training to finish.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
ARM="${ARM:?set ARM}"; MODS="${MODS:?set MODS}"; STEP="${STEP:?set STEP}"
CKPT="$REPO/outputs/specialist_${ARM}__seed42_fi3_512_aug_sr/checkpoint-${STEP}/step${STEP}.ckpt"
TAG="${TAG:-${ARM}spec_${STEP}_matched}"
PY=/opt/conda/envs/ttd_train/bin/python

echo "[eval-$ARM] waiting for $CKPT"
while [ ! -f "$CKPT" ]; do sleep 120; done
# the file appears before it is fully written; wait for the size to settle above 1GB
prev=0; while :; do cur=$(stat -c%s "$CKPT"); [ "$cur" = "$prev" ] && [ "$cur" -gt 1000000000 ] && break; prev=$cur; sleep 30; done
echo "[eval-$ARM] $(date) checkpoint complete ($(du -h "$CKPT" | cut -f1)); finding a GPU"

probe() { CUDA_VISIBLE_DEVICES="$1" timeout 60 "$PY" -c \
  "import torch; torch.zeros(int(28000*0.9*1024*1024//4), device='cuda'); torch.cuda.synchronize()" \
  >/dev/null 2>&1; }
while :; do
  while IFS=', ' read -r idx used; do
    [ "$used" -lt 2000 ] || continue
    if probe "$idx"; then
      echo "[eval-$ARM] $(date) GPU $idx -- evaluating $TAG ($MODS)"
      GPU="$idx" CKPT="$CKPT" TAG="$TAG" MODS="$MODS" bash scripts/run_one_heldout_eval.sh
      echo "[eval-$ARM] $(date) exited $?"; exit 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  sleep 120
done
