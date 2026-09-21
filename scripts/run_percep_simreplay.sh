#!/bin/bash
# Sim-replay perception campaign: score what a model IMAGINED against what the SIMULATOR
# renders after executing that model's OWN decoded actions.
#
# See PERCEP_SIMREPLAY.md for why the demonstration is not a valid target, what the numbers
# mean, and the gotchas. Read that before quoting anything this produces.
#
# A JOB IS  <arm>_<anchor>  x  <task>.  The anchor decides the template the policy is prompted
# with, so it must be a modality that arm ACTUALLY TRAINED ON -- prompting arm1 for normals
# measures a task it never saw and reports it as a bad score. NATIVE below is that check, and
# it is a hard refusal rather than a warning.
#
#   bash scripts/run_percep_simreplay.sh                       # everything in CONFIGS x TASKS
#   CONFIGS="arm1_depth arm1_video" TASKS=close_box bash scripts/run_percep_simreplay.sh
#   CKPT_MANIFEST=/path/ckpts.txt bash scripts/run_percep_simreplay.sh
#   DRY_RUN=1 bash scripts/run_percep_simreplay.sh              # print the queue, launch nothing
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${EVAL_RC:-/workspace/1228_tingting/ttd/scripts/env_eval_ttd_eval.rc}"
cd "$REPO"

TRIALS="${TRIALS:-5}"
GPUS="${GPUS:-0 1 2 3}"
TASKS="${TASKS:-close_microwave toilet_seat_down close_box meat_on_grill}"
OUT="${OUT:-$REPO/reports/percep_simreplay}"
LOGS="${LOGS:-$REPO/logs/logs_percep_simreplay}"
# One line per arm:  <arm> <absolute path to stepN.ckpt>
CKPT_MANIFEST="${CKPT_MANIFEST:-$REPO/ckpts.txt}"
mkdir -p "$OUT" "$LOGS"

# Which output spaces each arm was TRAINED on (scripts/train_arm.sh). `video` is in every arm
# because video+action is pinned at 0.4 on every rung -- arm0 is video ONLY.
declare -A NATIVE=(
  [arm0]="video"
  [arm1]="video depth"
  [arm2]="video segmentation"
  [arm3]="video normal"
  [arm4]="video depth segmentation"
  [arm5]="video normal segmentation"
  [arm6]="video depth normal"
  [arm7]="video depth normal segmentation"
)
# Arms trained with a segmentation stream were trained under scene_roles, which is part of the
# arm's DEFINITION, not a preference (train_arm.sh SEG_MODE_DEFAULT).
if [ ! -f "$CKPT_MANIFEST" ]; then
  echo "!! no checkpoint manifest at $CKPT_MANIFEST"
  echo "   Write one line per arm:   arm1 /abs/path/to/step4000.ckpt"
  exit 2
fi
ckpt_of() { awk -v a="$1" '$1==a {print $2; exit}' "$CKPT_MANIFEST"; }
# Default = every native (arm, anchor) pair of the arms LISTED IN THE MANIFEST. Defaulting to all
# eight arms would make one missing line (say arm7 before its download finishes) refuse the
# whole campaign. An explicit CONFIGS still gets the full check below.
DEFAULT_CONFIGS=""
for a in arm0 arm1 arm2 arm3 arm4 arm5 arm6 arm7; do
  [ -n "$(ckpt_of "$a")" ] || continue
  for m in ${NATIVE[$a]}; do DEFAULT_CONFIGS="$DEFAULT_CONFIGS ${a}_${m}"; done
done
CONFIGS="${CONFIGS:-$DEFAULT_CONFIGS}"
[ -n "${CONFIGS// /}" ] || { echo "!! $CKPT_MANIFEST lists no known arm (arm0..arm7)"; exit 2; }

# Refuse impossible jobs UP FRONT rather than 25 minutes into a rollout.
BAD=0
for cfg in $CONFIGS; do
  arm="${cfg%_*}"; anchor="${cfg##*_}"
  if [ -z "${NATIVE[$arm]:-}" ]; then echo "!! unknown arm '$arm' in config '$cfg'"; BAD=1; continue; fi
  case " ${NATIVE[$arm]} " in *" $anchor "*) ;; *)
    echo "!! $cfg: arm '$arm' never trained '$anchor' (native: ${NATIVE[$arm]}). Refusing."; BAD=1 ;;
  esac
  c="$(ckpt_of "$arm")"
  if [ -z "$c" ] || [ ! -f "$c" ]; then echo "!! $cfg: no checkpoint for '$arm' in $CKPT_MANIFEST"; BAD=1; fi
done
[ "$BAD" -ne 0 ] && exit 3

QUEUE="$LOGS/queue.txt"; : > "$QUEUE"
for cfg in $CONFIGS; do for t in $TASKS; do echo "$cfg $t" >> "$QUEUE"; done; done
LOCK="$LOGS/queue.lock"; : > "$LOCK"
echo "queued $(wc -l < "$QUEUE") jobs over GPUS='$GPUS' at $TRIALS trials each"
# DRY_RUN=1: show exactly what would run, then stop -- nothing is launched, no GPU is touched.
# Use it before every real launch; the checks above (native modality, checkpoint present) have
# already run by this point, so a clean dry run means the real one will start.
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN=1 -- not launching. Jobs (config task):"; cat "$QUEUE"
  for cfg in $CONFIGS; do echo "  ${cfg%_*} -> $(ckpt_of "${cfg%_*}")"; done | sort -u
  exit 0
fi

worker() {
  local gpu="$1"
  while true; do
    local job; job="$(flock "$LOCK" bash -c 'head -1 '"$QUEUE"'; sed -i 1d '"$QUEUE")"
    [ -z "$job" ] && break
    local cfg task arm anchor tag
    cfg="${job%% *}"; task="${job##* }"; arm="${cfg%_*}"; anchor="${cfg##*_}"
    tag="percep_${cfg}_${task}"
    [ -f "$OUT/rollout_${tag}.json" ] && { echo "[gpu$gpu] SKIP $tag (done)"; continue; }
    # No --segmentation-mode flag: eval/rollout.py hardcodes scene_roles (lines 206-219), which
    # is what every seg-bearing arm was trained under. Do not "fix" this by adding a flag
    # without also checking the arm's SEG_MODE in train_arm.sh.
    echo "[gpu$gpu] START $tag $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES="$gpu" xvfb-run -a python -u eval/rollout.py \
      --ckpt "$(ckpt_of "$arm")" --tag "$tag" --tasks "$task" --variation 0 \
      --num-trials "$TRIALS" --anchor-modality "$anchor" \
      --num-frames 41 --execution-horizon 41 --frame-interval 3 --res 512 \
      --cfg 7.5 --steps 50 --seed 42 --prompt-tag-style explicit \
      --axis-solver sphere --gripper-mode paper --arm-action-mode planning \
      --max-steps-factor 1.5 --skip-anchor-frames 4 --max-ik-fail-streak 5 \
      --record-video --dump-percep \
      --out "$OUT" > "$LOGS/$tag.log" 2>&1
    echo "[gpu$gpu] DONE  $tag rc=$? $(date +%H:%M:%S)"
  done
}
for g in $GPUS; do worker "$g" & done
wait
echo "ALL DONE"
