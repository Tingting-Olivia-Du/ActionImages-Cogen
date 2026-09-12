#!/bin/bash
# Closed-loop for a specialist, launched once its training finishes and a GPU frees.
#
#   ARM=action nohup bash scripts/wait_and_launch_specialist_closedloop.sh &
#   ARM=normal nohup bash scripts/wait_and_launch_specialist_closedloop.sh &
#
# WHAT EACH ONE MEASURES -- they are not the same experiment:
#
#   action  -> the middle row of tab:wam_preservation. Init -> action-specialist isolates domain
#              adaptation; action-specialist -> unified isolates the cost of adding perception.
#              Without it the preservation table cannot separate those two effects.
#
#   normal  -> NOT part of any planned table. A normal specialist never trains the action
#              template, so this measures how much of the RELEASED checkpoint's action prior
#              survives 4000 steps of perception-only fine-tuning. That is a forgetting
#              measurement, and it is the counterfactual that makes the unified model's
#              preservation non-trivial: if perception-only training leaves control intact, then
#              "the unified model preserves control" says nothing about the mixture.
#              Worth having, but read it as forgetting, never as "the normal specialist's policy".
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
ARM="${ARM:?set ARM=action|normal}"
OUTDIR="$REPO/outputs/specialist_${ARM}__seed42_fi3_512_aug_sr"
STEP="${STEP:-4000}"
CKPT="$OUTDIR/checkpoint-${STEP}/step${STEP}.ckpt"
PY=/opt/conda/envs/ttd_train/bin/python

echo "[cl-$ARM] waiting for $CKPT"
while [ ! -f "$CKPT" ]; do sleep 300; done
echo "[cl-$ARM] $(date) checkpoint present; waiting for it to stop growing"
prev=0; while :; do cur=$(stat -c%s "$CKPT"); [ "$cur" = "$prev" ] && [ "$cur" -gt 1000000000 ] && break; prev=$cur; sleep 60; done

probe() { CUDA_VISIBLE_DEVICES="$1" timeout 60 "$PY" -c \
  "import torch; torch.zeros(int(30000*0.9*1024*1024//4), device='cuda'); torch.cuda.synchronize()" \
  >/dev/null 2>&1; }
echo "[cl-$ARM] $(date) polling for a free GPU"
while :; do
  while IFS=', ' read -r idx used; do
    [ "$used" -lt 2000 ] || continue
    if probe "$idx"; then
      echo "[cl-$ARM] $(date) GPU $idx -- launching n=20/task closed-loop"
      source /opt/conda/etc/profile.d/conda.sh
      source /workspace/ttdu/ttd/scripts/env_eval.rc
      conda activate ttd_rollout; export PYTHONPATH="$REPO"
      # Same protocol as the unified model's campaign (fine-tuned arm: fi=3, tagged, cfg 7.5)
      # and the same 5 tasks, so the runs are paired against rollout_arm6_10000.json.
      CUDA_VISIBLE_DEVICES="$idx" xvfb-run -a python -u eval/rollout.py \
        --ckpt "$CKPT" --tag "spec_${ARM}_${STEP}" --res 512 --cfg 7.5 --steps 50 \
        --frame-interval 3 --prompt-tag-style explicit --axis-solver sphere \
        --arm-action-mode planning --max-steps-factor 1.5 --record-video \
        --tasks close_box close_drawer push_buttons meat_off_grill open_drawer --num-trials 20
      echo "[cl-$ARM] $(date) exited $? -> reports/closedloop/rollout_spec_${ARM}_${STEP}.json"
      exit 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  sleep 180
done
