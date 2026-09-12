#!/bin/bash
# Surface-normal counterpart, on the same two tiers the depth counterpart covers.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
PY=/opt/conda/envs/ttd_train/bin/python
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
export PYTHONPATH="$REPO" N_EP_PER_TASK=5
claim() {
  while :; do
    for i in 4 5 0 1 6 7 2 3; do
      if CUDA_VISIBLE_DEVICES=$i timeout 50 "$PY" -c \
         "import torch;torch.zeros(int(20000*0.9*1024*1024//4),device='cuda');torch.cuda.synchronize()" \
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
  G=$(claim); echo "[normal] $(date -u +%H:%M) START $TAG on GPU $G"
  CUDA_VISIBLE_DEVICES=$G $PY -u scripts/probe_normal_baseline.py $N \
    >> "$REPO/logs_normal_baseline_${TAG}.txt" 2>&1
  echo "[normal] $(date -u +%H:%M) $TAG exited $?"
done
echo "[normal] ALL DONE $(date -u +%H:%M)"
