#!/bin/bash
# Offline (open-loop) multimodal quality on the held-out trees, for tab:main's perception
# rows, tab:obs and the LPIPS row.
#
# ONE GENERATION SERVES FOUR TABLE CELLS. heldout_batch_eval.py scores the ANCHOR segment of
# arm7's substitution templates as well as the action segment, so `--modalities
# action,action_from_depth,action_from_seg,action_from_normal` yields, per episode:
#   * action error in each of the four observation spaces        -> tab:obs
#   * depth AbsRel / normal cos / seg mIoU from depth+action etc -> tab:main perception rows
#   * LPIPS/PSNR/SSIM of the future RGB inside video+action      -> tab:main visual row
# That is 4 generations per episode, not 7, and the perception numbers come from arm7's OWN
# templates -- arm7 has no video+depth template, so scoring it with one would be asking the
# checkpoint a question it was never trained on.
#
# The RGB-ablated control (arm0) and the released checkpoint have only `video+action`, so they
# run `--modalities action` alone.
set -uo pipefail
REPO=/workspace/1228_tingting/ActionImages-Cogen
source /workspace/1228_tingting/ttd/scripts/env_eval_ttd_eval.rc
cd "$REPO"

# GPU 0 IS RESERVED for the user's own work from 2026-09-19 16:40 on. The phase-1 wave
# that was already in flight keeps all eight; everything launched after it gets seven.
# Change this default rather than passing GPUS= at launch, so a watchdog restart cannot
# quietly take the card back.
GPUS="${GPUS:-1 2 3 4 5 6 7}"
MODELS="${MODELS:-arm7_6k arm0_6k official}"
EPS_PER_TASK="${EPS_PER_TASK:-5}"
OUT="${OUT:-$REPO/reports/heldout_batch_eval}"
LOGS="${LOGS:-$REPO/logs/logs_offline_unseen14}"
mkdir -p "$OUT" "$LOGS"

ckpt_of() { case "$1" in
  arm7_6k)  echo "$REPO/outputs/arm7_joint2src_seed42_fi3_6k/checkpoint-6000/step6000.ckpt" ;;
  arm0_6k)  echo "$REPO/outputs/arm0_joint2src_seed42_fi3_6k/checkpoint-6000/step6000.ckpt" ;;
  official) echo "$REPO/checkpoints/official/step125750.ckpt" ;;
esac; }
mods_of()     { [ "$1" = arm7_6k ] && echo "action,action_from_depth,action_from_seg,action_from_normal" || echo "action"; }
tagstyle_of() { [ "$1" = official ] && echo none || echo explicit; }
cfg_of()      { [ "$1" = official ] && echo 10.0 || echo 7.5; }

# The four splits of tab:main, each with its own tree, stride and tier label.
#   name | tree | frame_interval | variation | split_label
SPLITS="${SPLITS:-rlb_aug|data/rlbench_unseen_tasks_512_aug|3|0|unseen_task \
rlb_clean|data/rlbench_unseen_tasks_512_clean|3|0|unseen_task \
ms_heldout|data/maniskill3_heldout|1|0|unseen_task \
ms_seeds|data/maniskill3|1|1|unseen}"

episodes_for() {   # <tree> <variation> <n per task> -> comma list of relative episode paths
  python - "$1" "$2" "$3" <<'PY'
import sys, os, glob
tree, var, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
out = []
for task in sorted(os.listdir(tree)):
    eps = sorted(glob.glob(f"{tree}/{task}/variation{var}/episodes/episode*"),
                 key=lambda p: int(os.path.basename(p).replace("episode", "")))[:n]
    out += [os.path.relpath(e, tree) for e in eps]
print(",".join(out))
PY
}

QUEUE="$LOGS/queue.txt"; IDX="$LOGS/queue.idx"
if [ "${FRESH_QUEUE:-1}" = 1 ]; then
  : > "$QUEUE"; for m in $MODELS; do for s in $SPLITS; do echo "$m $s" >> "$QUEUE"; done; done
  echo 0 > "$IDX"
fi
TOTAL=$(wc -l < "$QUEUE")
echo "[offline] $TOTAL jobs, GPUs: $GPUS -> $OUT"

next_job() {
  local n; exec 9>"$LOGS/.queue.lock"; flock 9
  n=$(cat "$IDX"); [ "$n" -ge "$TOTAL" ] && { flock -u 9; return 1; }
  echo $((n + 1)) > "$IDX"; sed -n "$((n + 1))p" "$QUEUE"; flock -u 9
}

worker() {
  local gpu=$1 job model spec name tree fi var label tag eps
  while job=$(next_job); do
    model=${job%% *}; spec=${job##* }
    IFS='|' read -r name tree fi var label <<< "$spec"
    tag="off_${model}_${name}"
    eps="$(episodes_for "$tree" "$var" "$EPS_PER_TASK")"
    echo "[gpu$gpu] $(date +%H:%M:%S) start $tag ($(tr ',' '\n' <<< "$eps" | wc -l) episodes)"
    CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/heldout_batch_eval.py \
      --ckpt "$(ckpt_of "$model")" --tag "$tag" --gpu 0 \
      --data "$REPO/$tree" --frame_interval "$fi" --res 512 --num_frames 41 \
      --cfg "$(cfg_of "$model")" --steps 50 --seed 42 \
      --prompt_tag_style "$(tagstyle_of "$model")" --segmentation_mode scene_roles \
      --split_label "$label" --modalities "$(mods_of "$model")" \
      --episodes "$eps" --out "$OUT" > "$LOGS/${tag}.log" 2>&1
    echo "[gpu$gpu] $(date +%H:%M:%S) done  $tag rc=$?"
  done
  echo "[gpu$gpu] queue empty"
}

for g in $GPUS; do worker "$g" & sleep 2; done
wait
echo "OFFLINE_DONE $(date)"
