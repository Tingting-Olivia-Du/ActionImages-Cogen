#!/bin/bash
# Action specialist (video+action@1.0), 4000 steps -- SAME recipe/budget as the depth,
# segmentation, and normal specialists (scripts/run_specialist_depth.sh,
# run_specialist_seg.sh, wait_and_launch_normal_specialist.sh), completing a
# consistent four-specialist set for tab:headline / tab:matched_exposure in
# paper/draft_aug_29.md.
#
# REPLACES the earlier plan to train a fresh 10000-step "arm0" on this tree: that would have
# given action 2.5x the training budget every other specialist gets, which is exactly the kind
# of asymmetry sec:exp_sharing exists to rule out. The zero-shot, 125,750-step-pretrained
# official checkpoint remains the "Action Images init" reference point wherever the draft wants
# a true zero-shot baseline (tab:offline_preservation, tab:closed_loop) -- it is not replaced by
# this specialist, which instead stands in for what the draft called "arm0": the domain-adapted,
# action-only continuation, now named consistently with the other three specialists.
#
#   nohup bash scripts/wait_and_launch_action_specialist.sh > logs_action_specialist_waiter.txt 2>&1 &
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"
cd "$REPO"

FREE_MB="${FREE_MB:-2000}"
POLL_S="${POLL_S:-120}"
OUT_DIR="$REPO/outputs/specialist_action__seed42_fi3_512_aug_sr"
PY=/opt/conda/envs/ttd_train/bin/python

probe_gpu() {
  CUDA_VISIBLE_DEVICES="$1" timeout 60 "$PY" -c \
    "import torch; torch.zeros(256*1024*1024//4, device='cuda'); torch.cuda.synchronize()" \
    >/dev/null 2>&1
}

echo "[waiter-action] watching for 2 GPUs with < ${FREE_MB} MiB used AND a real alloc probe, polling every ${POLL_S}s"
while true; do
  CAND=()
  while IFS=', ' read -r idx used; do
    [ "$used" -lt "$FREE_MB" ] || continue
    if probe_gpu "$idx"; then CAND+=("$idx"); else echo "[waiter-action] GPU $idx looked free but failed the probe"; fi
    [ "${#CAND[@]}" -ge 2 ] && break
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

  if [ "${#CAND[@]}" -ge 2 ]; then
    G="${CAND[0]},${CAND[1]}"
    echo "[waiter-action] $(date +%T) GPUs ${G} probed free -- re-checking in 10s to skip a transient gap"
    sleep 10
    if probe_gpu "${CAND[0]}" && probe_gpu "${CAND[1]}"; then
      if [ -d "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
        echo "[waiter-action] !! $OUT_DIR is not empty -- refusing to launch, it would resume, not warm-start"
        exit 3
      fi
      echo "[waiter-action] $(date +%T) launching action specialist on GPUS=${G}"
      GPUS="$G" SEED=42 STEPS=4000 CKPT_EVERY=500 SAVE_TOP_K=-1 OUT="$OUT_DIR" \
        EXTRA_ARGS="--run_name specialist-action-seed42" \
        bash scripts/train_arm.sh "video+action@1.0" > "$REPO/logs_specialist_action.txt" 2>&1
      echo "[waiter-action] train_arm.sh exited $? -- see logs_specialist_action.txt"
      exit 0
    fi
    echo "[waiter-action] $(date +%T) the gap closed before launch; still waiting"
  fi
  sleep "$POLL_S"
done
