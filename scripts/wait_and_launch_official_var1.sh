#!/bin/bash
# The official-checkpoint half of the variation-1 closed-loop. NOT optional: the arm6 variation-1
# run is a PAIRED comparison and is uninterpretable on its own -- scene seeds are shared, and
# eval/aggregate_closedloop.py refuses to compare unpaired runs. Launches on the first GPU that
# passes a real allocation probe.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
FREE_MB="${FREE_MB:-2000}"; POLL_S="${POLL_S:-120}"
PY=/opt/conda/envs/ttd_train/bin/python
probe() { CUDA_VISIBLE_DEVICES="$1" timeout 60 "$PY" -c \
  "import torch; torch.zeros(int(28000*0.9*1024*1024//4), device='cuda'); torch.cuda.synchronize()" \
  >/dev/null 2>&1; }
echo "[wait-official-v1] polling for one GPU with a real ~25GB allocation free"
while true; do
  while IFS=', ' read -r idx used; do
    [ "$used" -lt "$FREE_MB" ] || continue
    if probe "$idx"; then
      echo "[wait-official-v1] $(date +%T) GPU $idx free -- launching"
      MODEL=official GPU="$idx" bash scripts/run_closedloop_variation1.sh \
        > "$REPO/logs_cl_var1_official.txt" 2>&1
      echo "[wait-official-v1] exited $? -- see logs_cl_var1_official.txt"; exit 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  sleep "$POLL_S"
done
