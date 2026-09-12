#!/bin/bash
# Normal specialist (video+normal@1.0), completing the three-specialist set for tab:headline in
# paper/draft_aug_29.md alongside the depth and segmentation specialists already training
# (scripts/run_specialist_depth.sh, scripts/run_specialist_seg.sh). Same recipe: fresh warm-start
# from official step125750, 512_aug/fi3 tree, 4000 steps (arm6's own depth AbsRel already
# plateaus by step 4000: 0.087@1500 -> 0.086@3500 -> 0.085@4000), checkpointed every 500 steps.
#
#   nohup bash scripts/wait_and_launch_normal_specialist.sh > logs_normal_specialist_waiter.txt 2>&1 &
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"
cd "$REPO"

FREE_MB="${FREE_MB:-2000}"
POLL_S="${POLL_S:-120}"
OUT_DIR="$REPO/outputs/specialist_normal__seed42_fi3_512_aug_sr"
PY=/opt/conda/envs/ttd_train/bin/python

probe_gpu() {
  CUDA_VISIBLE_DEVICES="$1" timeout 60 "$PY" -c \
    "import torch; torch.zeros(256*1024*1024//4, device='cuda'); torch.cuda.synchronize()" \
    >/dev/null 2>&1
}

echo "[waiter-normal] watching for 2 GPUs with < ${FREE_MB} MiB used AND a real alloc probe, polling every ${POLL_S}s"
while true; do
  CAND=()
  while IFS=', ' read -r idx used; do
    [ "$used" -lt "$FREE_MB" ] || continue
    if probe_gpu "$idx"; then CAND+=("$idx"); else echo "[waiter-normal] GPU $idx looked free but failed the probe"; fi
    [ "${#CAND[@]}" -ge 2 ] && break
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

  if [ "${#CAND[@]}" -ge 2 ]; then
    G="${CAND[0]},${CAND[1]}"
    echo "[waiter-normal] $(date +%T) GPUs ${G} probed free -- re-checking in 10s to skip a transient gap"
    sleep 10
    if probe_gpu "${CAND[0]}" && probe_gpu "${CAND[1]}"; then
      if [ -d "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
        echo "[waiter-normal] !! $OUT_DIR is not empty -- refusing to launch, it would resume, not warm-start"
        exit 3
      fi
      echo "[waiter-normal] $(date +%T) launching normal specialist on GPUS=${G}"
      GPUS="$G" SEED=42 STEPS=4000 CKPT_EVERY=500 SAVE_TOP_K=-1 OUT="$OUT_DIR" \
        EXTRA_ARGS="--run_name specialist-normal-seed42" \
        bash scripts/train_arm.sh "video+normal@1.0" > "$REPO/logs_specialist_normal.txt" 2>&1
      echo "[waiter-normal] train_arm.sh exited $? -- see logs_specialist_normal.txt"
      exit 0
    fi
    echo "[waiter-normal] $(date +%T) the gap closed before launch; still waiting"
  fi
  sleep "$POLL_S"
done
