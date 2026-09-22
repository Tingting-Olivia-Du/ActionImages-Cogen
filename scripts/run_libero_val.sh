#!/bin/bash
# Held-out VALIDATION of the LIBERO-only runs: open-loop action error on LIBERO variation1
# (5 demos per task, never trained on), at every saved checkpoint. It is how the LIBERO step to
# evaluate closed-loop is chosen -- LIBERO trains to 10k because nobody knows where it converges,
# and the diffusion loss is too noisy to say.
#
#   bash scripts/run_libero_val.sh <arm1|arm2|arm3|official> <gpu> [steps...]
#   bash scripts/run_libero_val.sh arm2 4                      # every 1k: 1000 .. 10000
#   bash scripts/run_libero_val.sh arm2 5 6000 7000 8000       # or a subset, to split over GPUs
#   bash scripts/run_libero_val.sh official 4                  # the untrained baseline, once
#   python scripts/libero_val_table.py                         # the table + the chosen step
#
# FIXED EPISODE SET, identical for every arm and step: for each suite, tasks 0,2,4,6,8 in sorted
# directory order, each task's first variation1 episode -> 20 episodes per checkpoint. Scored with
# eval/eval_action.py, template video+action (the one template all three arms share), protocol
# i2va (one real anchor frame in, the regime closed-loop rollout uses), 512^2, frame_interval 1.
# ~165 s per episode -> ~1 h per checkpoint on one GPU, ~10 GPU-hours for one arm's 10 checkpoints. Checkpoints that do not exist yet are
# skipped, and finished (arm, step, suite) cells are not redone, so rerun it as training goes.
set -uo pipefail
ARM="${1:?arm1|arm2|arm3|official}"; GPU="${2:?gpu index}"; shift 2
STEPS="${*:-1000 2000 3000 4000 5000 6000 7000 8000 9000 10000}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 2
[ -n "${ENV_RC:-}" ] && source "$ENV_RC"
OUT="$REPO/reports/libero_val"; mkdir -p "$OUT/logs"
[ "$ARM" = official ] && STEPS=0

for st in $STEPS; do
  if [ "$ARM" = official ]; then
    ck="$REPO/checkpoints/official/step125750.ckpt"; tag="official"
  else
    ck="$REPO/outputs/${ARM}_liberoonly_fromofficial_a1_10k/checkpoint-${st}/step${st}.ckpt"; tag="${ARM}_${st}"
  fi
  [ -f "$ck" ] || { echo "[$tag] no checkpoint yet ($ck), skip"; continue; }
  for su in libero_spatial libero_object libero_goal libero_10; do
    rep="$OUT/$su/$tag/report_video-action_i2va.json"
    [ -f "$rep" ] && { echo "[$tag] $su done"; continue; }
    mapfile -t tasks < <(ls "data/$su" | sort)
    eps=()
    for i in 0 2 4 6 8; do
      e=$(ls "data/$su/${tasks[$i]}/variation1/episodes" | sort -V | head -1)
      eps+=("${tasks[$i]}/variation1/episodes/$e")
    done
    echo "[$tag] $(date +%H:%M:%S) $su (${#eps[@]} episodes)"
    CUDA_VISIBLE_DEVICES="$GPU" python -u eval/eval_action.py --ckpt "$ck" --data "data/$su" \
      --episodes "${eps[@]}" --output "$OUT/$su" --tag "$tag" --res 512 --frame_interval 1 \
      --template video+action --protocol i2va --axis-solver sphere --no-video \
      > "$OUT/logs/${tag}_${su}.log" 2>&1 || echo "[$tag] $su FAILED, see $OUT/logs/${tag}_${su}.log"
  done
done
echo "LIBERO_VAL_DONE $ARM"
