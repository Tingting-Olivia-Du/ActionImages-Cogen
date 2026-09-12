#!/bin/bash
# Hand the training GPUs to an eval and hand them straight back.
#
#   nohup bash scripts/eval_handoff.sh > logs_handoff.txt 2>&1 &
#
# On a node where outside tenants claim a card within seconds of it going idle, "stop training,
# run eval, resume training" cannot be three separate manual steps -- the gap between them is
# exactly when the cards get taken. This does the whole handoff in one process:
#
#   1. STOP THE SUPERVISOR FIRST. It polls every 120 s for two free GPUs and relaunches
#      training; leaving it up means it races the eval for the very cards the eval just freed.
#   2. Kill the trainer, then WAIT for the memory to actually be released -- `nvidia-smi` keeps
#      reporting the allocation for a few seconds after SIGKILL, and launching into that window
#      OOMs part-way through a 12.8 GB model load.
#   3. Occupy BOTH freed cards immediately: the full grid on one, the multi-episode fifi probe
#      on the other. One eval only needs ~28 GB, so a single-card eval would leave the second
#      card idle and stealable -- and the supervisor needs TWO to resume.
#   4. Restart the supervisor the moment both finish. Training resumes from the newest
#      checkpoint (train_arm.sh -> find_latest_checkpoint), losing only the steps since it.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
export PYTHONPATH="$REPO"

OUT="$REPO/outputs/arm6__seed42_fi3_512_aug_sr"
STEPS="${STEPS:-50}"
PROBE_EPISODES="${PROBE_EPISODES:-push_buttons/variation0/episodes/episode0 stack_wine/variation0/episodes/episode0 turn_tap/variation0/episodes/episode0}"

say() { echo "[handoff $(date +%T)] $*"; }

CK_DIR=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
[ -n "$CK_DIR" ] || { say "no checkpoint under $OUT"; exit 2; }
STEP=$(basename "$CK_DIR" | sed 's/checkpoint-//')
CK="$CK_DIR/step${STEP}.ckpt"
[ -f "$CK" ] || { say "missing $CK"; exit 2; }
say "latest checkpoint: step $STEP"
say "training is at: $(tail -c 200 "$REPO/logs_arm6.txt" | tr '\r' '\n' | grep -oE '[0-9]+/10000' | tail -1)"

# ---- 1. supervisor down, so it cannot grab the cards back mid-eval -------------------------
for p in $(ps -eo pid,args | grep "[s]upervise_arm.sh" | awk '{print $1}'); do
  say "stopping supervisor pid $p"; kill "$p" 2>/dev/null
done
# any queued eval that is merely WAITING for a card: replace it, we are about to pin cards
for p in $(ps -eo pid,args | grep "[m]odality_mode_grid.py" | awk '{print $1}'); do
  say "stopping queued eval pid $p"; kill -9 "$p" 2>/dev/null
done
sleep 3

# ---- 2/3. WARM FIRST, THEN TAKE ------------------------------------------------------------
# Killing the trainer and then starting a cold eval loses the card: python + torch import is
# ~10 s, and this node's tenants claim a freed GPU inside 9 s (measured twice, both handoffs
# lost). So the order is inverted. Both eval processes start while training is still running,
# pay the import cost, open a CUDA context on the target card (~430 MiB, granted even on a full
# card), and only THEN does the grid process kill the trainer and grab the memory in a 0.2 s
# retry loop. The probe process claims the second card the same way, without killing anything.
GPUS=$(grep -oE "GPUS=[0-9,]+" "$REPO/logs_arm6.txt" | tail -1 | cut -d= -f2)
G1=$(echo "$GPUS" | cut -d, -f1); G2=$(echo "$GPUS" | cut -d, -f2)
[ -n "$G1" ] || { say "cannot tell which GPUs training is on"; exit 3; }
say "training on GPUs $GPUS -- taking them over in place (no idle gap)"

python -u scripts/modality_mode_grid.py --ckpt "$CK" --tag "arm6_${STEP}" \
  --claim_gpu "$G1" --kill_first "train\.py --deepspeed" --steps "$STEPS" \
  > "$REPO/logs_eval_grid_${STEP}.txt" 2>&1 &
P1=$!
(
  sleep 25   # let the grid do the killing; this one only needs the card to come free
  for ep in $PROBE_EPISODES; do
    slug=$(echo "$ep" | cut -d/ -f1)
    kill -0 $P1 2>/dev/null || { echo "[probe] grid finished, stopping"; break; }
    python -u scripts/modality_mode_grid.py --ckpt "$CK" --tag "arm6_${STEP}_ep_${slug}" \
      --claim_gpu "$G2" --steps "$STEPS" --episode "$ep" --only "video+depth"
  done
  # Whatever time is left on GPU 2 goes to the older checkpoints. SAVE_TOP_K=5 rotates them
  # away, so anything not measured before it rotates is unmeasurable forever -- and the fifi
  # question (fiii 0.062 vs fifi 0.656 at step 1500, on an EQUAL 5% share) needs the trend
  # across checkpoints to separate "converging slowly" from "structurally broken".
  for OLD in $(ls -d "$OUT"/checkpoint-* | sort -t- -k2 -nr | tail -n +2); do
    kill -0 $P1 2>/dev/null || { echo "[backfill] grid finished, stopping"; break; }
    OSTEP=$(basename "$OLD" | sed 's/checkpoint-//')
    OCK="$OLD/step${OSTEP}.ckpt"
    [ -f "$OCK" ] || continue
    [ -f "$REPO/reports/modality_mode_grid/arm6_${OSTEP}/metrics.json" ] && continue
    echo "[backfill] step $OSTEP on GPU $G2"
    python -u scripts/modality_mode_grid.py --ckpt "$OCK" --tag "arm6_${OSTEP}" \
      --claim_gpu "$G2" --steps "$STEPS" --only "video+depth,video+segmentation"
  done
) > "$REPO/logs_eval_probe_${STEP}.txt" 2>&1 &
P2=$!
wait $P1; R1=$?
say "grid done (exit $R1); GPU $G2 stops taking new work"
wait $P2; R2=$?
say "grid exit=$R1  probes+backfill exit=$R2"

# ---- 4. training back on ------------------------------------------------------------------
say "restarting supervisor (will resume from the newest checkpoint)"
ARM=arm6 CKPT_EVERY=500 SAVE_TOP_K=5 nohup bash scripts/supervise_arm.sh \
  >> "$REPO/logs_arm6_supervisor.txt" 2>&1 &
say "HANDOFF_DONE step=$STEP grid=$R1 probes=$R2"
