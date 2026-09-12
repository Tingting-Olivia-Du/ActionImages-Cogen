#!/bin/bash
# Parallel render of the two never-trained closed-loop tasks, one worker per (task, seed).
# Seeds are the closed-loop trial seeds, so object layouts correspond 1:1 with the rollouts.
# gen_dataset.py skips any episode that already has meta.json, so re-running is a safe resume.
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate ttd_eval
source /workspace/ttdu/ttd/scripts/env_eval.rc
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
export PYTHONPATH=/workspace/ttdu/robot-colosseum${PYTHONPATH:+:$PYTHONPATH}
export LP_NUM_THREADS="${LP_NUM_THREADS:-4}"

OUT="${OUT:-/workspace/ttdu/ttd/data/rlbench_unseen_tasks_512_aug}"
NPAR="${NPAR:-16}"
LOGD="$REPO/logs_gen_unseen"; mkdir -p "$LOGD"

run_one() {
  local task="$1" seed="$2"
  sleep $(( RANDOM % 20 ))          # stagger: concurrent `xvfb-run -a` race for a display number
  for try in 1 2 3; do
    xvfb-run -a python -u scripts/gen_dataset.py --tasks "$task" --seeds "$seed" \
      --out "$OUT" --variations 0 --res 512 --aug colosseum \
      >> "$LOGD/${task}_${seed}.log" 2>&1 && return 0
    echo "[retry $try]" >> "$LOGD/${task}_${seed}.log"; sleep 5
  done
  echo "GIVE_UP $task $seed" >> "$LOGD/${task}_${seed}.log"; return 1
}
export -f run_one; export OUT LOGD

python3 - <<'PY' > /tmp/unseen_jobs.txt
import hashlib
def ts(t,v,i):
    return int.from_bytes(hashlib.blake2b(f"{t}|{v}|{i}".encode(),digest_size=8).digest(),'big')%1_000_000+1
import os
for t in os.environ.get("UNSEEN_TASKS","close_box,close_drawer").split(","):
    for i in range(int(os.environ.get("N_SEEDS","20"))): print(t, ts(t,0,i))
PY
echo "[gen-par] $(date) $(wc -l < /tmp/unseen_jobs.txt) episodes, NPAR=$NPAR -> $OUT"
xargs -a /tmp/unseen_jobs.txt -n2 -P "$NPAR" bash -c 'run_one "$0" "$1"'
echo "[gen-par] $(date) all workers returned"
find "$OUT" -name meta.json | wc -l | xargs echo "[gen-par] episodes with meta.json:"
