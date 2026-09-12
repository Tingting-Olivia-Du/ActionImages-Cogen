#!/bin/bash
# Supervisor for the unattended window: relaunch anything that dies, and start the follow-up
# work as GPUs free. Idempotent -- every action checks whether its output already exists first,
# so re-running the supervisor never duplicates work.
#
#   nohup bash scripts/supervise_8h.sh > logs_supervisor_8h.txt 2>&1 &
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
PY=/opt/conda/envs/ttd_train/bin/python
POLL="${POLL:-600}"

probe() { CUDA_VISIBLE_DEVICES="$1" timeout 60 "$PY" -c \
  "import torch; torch.zeros(int(28000*0.9*1024*1024//4), device='cuda'); torch.cuda.synchronize()" \
  >/dev/null 2>&1; }
free_gpu() {
  while IFS=', ' read -r idx used; do
    [ "$used" -lt 2000 ] || continue
    probe "$idx" && { echo "$idx"; return 0; }
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  return 1
}
# NOTE: pgrep -f matches with EXTENDED regex, so a literal '+' in a template name must be
# escaped -- "video+action" as ERE means "vide" + one-or-more "o" + "action", which matches
# nothing and made this helper report every live training run as dead.
running() { pgrep -f "$1" 2>/dev/null | grep -qv "^$$\$"; }

log() { echo "[sup] $(date -u +%H:%M) $*"; }
log "supervisor up"

while :; do
  # ---- 1. gripper_acc backfill: arm6 + init action cells predate the metric ----------------
  for spec in "arm6_10000:outputs/.grid_pin/step10000.ckpt:explicit" \
              "init_125750:/workspace/ttdu/starVLA/playground/Pretrained_models/anyeZHY/ActionImages/step125750.ckpt:none"; do
    tag="${spec%%:*}"; rest="${spec#*:}"; ck="${rest%%:*}"; style="${rest##*:}"
    done_marker="$REPO/reports/heldout_batch_eval/$tag/.gripper_backfilled"
    [ -f "$done_marker" ] && continue
    has=$($PY -c "
import json
try:
    d=json.load(open('reports/heldout_batch_eval/$tag/per_episode.json'))['results']
    a=[r for r in d.values() if r['modality']=='action']
    print(int(bool(a) and all('gripper_acc' in r for r in a)))
except Exception: print(0)" 2>/dev/null)
    if [ "$has" = "1" ]; then touch "$done_marker"; continue; fi
    running "heldout_batch_eval.py.*--tag $tag" && continue
    running "run_one_heldout_eval.sh" && { sleep 5; continue; }
    g=$(free_gpu) || break
    log "gripper_acc backfill for $tag on GPU $g"
    # drop the action cells so resume recomputes them WITH the new metric
    $PY - <<PYEOF
import json
p='reports/heldout_batch_eval/$tag/per_episode.json'
b=json.load(open(p)); b['results']={k:v for k,v in b['results'].items() if v['modality']!='action'}
json.dump(b, open(p,'w'), indent=1)
PYEOF
    GPU="$g" CKPT="$ck" TAG="$tag" MODS=action \
      nohup bash -c "source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train;
        export PYTHONPATH=$REPO CUDA_VISIBLE_DEVICES=$g;
        python -u scripts/heldout_batch_eval.py --ckpt '$ck' --tag '$tag' --gpu 0 \
          --modalities action --prompt_tag_style $style && touch '$done_marker'" \
      >> "$REPO/logs_gripper_backfill_${tag}.txt" 2>&1 &
    sleep 90
  done

  # ---- 2. health: report anything that died without finishing ------------------------------
  # normal is deliberately stopped at step 2000: matched exposure for a perception modality is
  # 0.2T = 2000 updates, and the user scoped the normal specialist to exactly one number
  # (tab:headline's normal cell). Training it to 4000 would produce nothing anyone reads, so its
  # expected checkpoint is 2000, not 4000 -- otherwise this check false-alarms every cycle.
  for j in "specialist_action:train_arm.sh video\\+action@1.0:4000" \
           "specialist_normal:train_arm.sh video\\+normal@1.0:2000"; do
    n="${j%%:*}"; rest="${j#*:}"; pat="${rest%:*}"; want="${rest##*:}"
    ck="$REPO/outputs/${n}__seed42_fi3_512_aug_sr/checkpoint-${want}/step${want}.ckpt"
    [ -f "$ck" ] && continue
    running "$pat" || log "!! $n is NOT running and has no checkpoint-4000 -- needs attention"
  done

  sleep "$POLL"
done
