#!/bin/bash
# Render the four standard LIBERO suites (40 tasks x 50 demos) into our tree layout, one process
# per task, spread over the GPUs for MuJoCo EGL. Each process reuses one env across its 50 demos
# and resumes: a demo whose meta.json exists is skipped (meta.json is written LAST).
#
# The GPUs are shared with training. MuJoCo EGL needs ~0.5 GB and little compute per process,
# so PER_GPU processes per card sit in the ~16 GB training leaves free; watch the training
# s/it before and after launch and lower PER_GPU if it degrades.
set -uo pipefail
REPO=/workspace/1228_tingting/ActionImages-Cogen
PY=/workspace/1228_tingting/envs/libero/bin/python
cd "$REPO"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
LOGS="$REPO/logs/logs_libero_gen"; mkdir -p "$LOGS"
jobs=()
for s in libero_spatial libero_object libero_goal libero_10; do
  for t in 0 1 2 3 4 5 6 7 8 9; do jobs+=("$s $t"); done
done
gpus=($GPUS); i=0
for j in "${jobs[@]}"; do
  read -r s t <<< "$j"; g=${gpus[$((i % ${#gpus[@]}))]}
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$g setsid nohup "$PY" -u scripts/libero_gen.py \
    --suite "$s" --task-index "$t" --demos 0-49 --out "$REPO/data/$s" \
    > "$LOGS/${s}_${t}.log" 2>&1 < /dev/null &
  i=$((i + 1)); sleep 2
done
echo "launched ${#jobs[@]} tasks over GPUs: $GPUS"
