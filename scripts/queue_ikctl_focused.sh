#!/bin/bash
# Is the 1.7%/1.1% a model limit or the harness giving up?
#
# max_ik_fail_streak=5 is a hardcoded abort, and it fires on 37% of variation-0 rollouts and
# 31% of variation-1 rollouts across the twelve-task set -- not a marginal effect. It lands
# hardest on tasks the model scores zero on, which is exactly the confound: a rollout that is
# merely bad gets terminated before it can finish, and the zero is then read as inability.
#
# The full twelve-task control is 21 GPU-hours because raising the threshold makes the very
# rollouts it was killing run to timeout instead. This is the informative subset -- the two
# conditions with the highest abort rate AND zero successes, where a change would be
# unambiguous:
#     light_bulb_in variation 0   8/10 aborted, 0/10 success
#     push_buttons  variation 1   8/10 aborted, 0/10 success
# Same scene seeds as the streak=5 runs, so the comparison is paired per trial.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
LEASE="bash $REPO/scripts/gpu_lease.sh"
source /opt/conda/etc/profile.d/conda.sh
source /workspace/ttdu/ttd/scripts/env_eval.rc
conda activate ttd_rollout
export PYTHONPATH="$REPO"
run_one() {   # task variation
  local T=$1 V=$2
  local out="$REPO/reports/closedloop/rollout_ikctl_${T}_v${V}.json"
  for a in 1 2 3; do
    [ -s "$out" ] && return 0
    G=$($LEASE acquire 28000)
    echo "[ikctl] $(date -u +%H:%M) START $T var$V attempt $a on GPU $G"
    CUDA_VISIBLE_DEVICES=$G xvfb-run -a python -u eval/rollout.py \
      --ckpt outputs/.grid_pin/step10000.ckpt --res 512 --cfg 7.5 --steps 50 \
      --frame-interval 3 --prompt-tag-style explicit --axis-solver sphere \
      --arm-action-mode planning --max-steps-factor 1.5 --skip-anchor-frames 4 \
      --variation $V --num-trials 10 --max-ik-fail-streak 50 \
      --tag ikctl_${T}_v${V} --tasks $T >> "$REPO/logs_ikctl_${T}_v${V}.txt" 2>&1
    rc=$?; $LEASE release "$G"
    [ -s "$out" ] && { echo "[ikctl] $T var$V OK (rc=$rc)"; return 0; }
    echo "[ikctl] $T var$V no output (rc=$rc) -- retry"; sleep 60
  done
}
run_one light_bulb_in 0
run_one push_buttons  1
echo "[ikctl] ALL DONE $(date -u +%H:%M)"
