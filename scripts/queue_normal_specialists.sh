#!/bin/bash
# Dedicated surface-normal counterparts on both tiers the depth counterpart covers.
#
# Lotus-G and Marigold-Normals are both run and both reported; the stronger becomes the
# main-table counterpart. Running only one, or only the weaker, would make the counterpart
# column a choice rather than a measurement. A 2-episode smoke test already puts Lotus at
# 0.964 against Marigold's 0.876, with the coordinate convention resolved unambiguously
# (identity +0.96 against flip_yz -0.88), so the calibration is not a close call.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
PY=/opt/conda/envs/ttd_train/bin/python
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
export PYTHONPATH="$REPO" N_EP_PER_TASK=5
claim() {
  while :; do
    for i in 5 4 0 1 6 7 2 3; do
      if CUDA_VISIBLE_DEVICES=$i timeout 50 "$PY" -c \
         "import torch;torch.zeros(int(16000*0.9*1024*1024//4),device='cuda');torch.cuda.synchronize()" \
         >/dev/null 2>&1; then echo "$i"; return 0; fi
    done
    sleep 300
  done
}
for CFG in "var1:$REPO/data/rlbench_selfgen_512_aug:1::40" \
           "task:$REPO/data/rlbench_unseen_tasks_512_aug::$REPO/reports/external/unseen_task_eps.txt:66"; do
  IFS=: read -r TAG TREE VAR EPSF N <<< "$CFG"
  export DATA_TREE="$TREE" SPLIT_TAG="_$TAG"
  [ -n "$VAR" ] && export EVAL_VARIATION="$VAR" || unset EVAL_VARIATION
  [ -n "$EPSF" ] && export EPS_FILE="$EPSF" || unset EPS_FILE
  for M in lotus marigold; do
    G=$(claim); echo "[normspec] $(date -u +%H:%M) START $M/$TAG on GPU $G"
    CUDA_VISIBLE_DEVICES=$G $PY -u scripts/probe_normal_specialists.py $M $N \
      >> "$REPO/logs_normspec_${M}_${TAG}.txt" 2>&1
    echo "[normspec] $(date -u +%H:%M) $M/$TAG exited $?"
  done
done
echo "[normspec] ALL DONE $(date -u +%H:%M)"
