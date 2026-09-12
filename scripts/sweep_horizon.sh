#!/bin/bash
# How much does replanning MORE often help? Sweep the execution horizon on a fixed cell.
#
# The chunk is always the same 41 generated poses; execution_horizon decides how many of them
# are executed before the next generation. 41 = execute the whole chunk (cheapest, most
# open-loop drift); 1 = regenerate every control step (fully closed-loop, ~30x the GPU cost).
#
# Held fixed so the horizon is the only variable: one checkpoint, one task, the same scene
# seeds, same solver, same anchor handling.
set -uo pipefail
GPU="${1:?usage: sweep_horizon.sh <gpu>}"
CK="${CK:-outputs/arm4__seed42_fi3/checkpoint-10000/step10000.ckpt}"
TASK="${TASK:-close_drawer}"
TRIALS="${TRIALS:-5}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"; export PYTHONPATH="$REPO_ROOT"
source /opt/conda/etc/profile.d/conda.sh
source /workspace/ttdu/ttd/scripts/env_eval.rc

if [ ! -f "$CK" ]; then
  echo "[sweep] fetching checkpoint-10000 ..."
  conda run -n ttd_train python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('TingtingDu/arm4__seed42_fi3','checkpoint-10000/step10000.ckpt',
                local_dir='outputs/arm4__seed42_fi3')" || exit 1
fi

for EH in "${@:2}"; do
  echo "===== execution_horizon=$EH ====="
  CUDA_VISIBLE_DEVICES=$GPU xvfb-run -a conda run -n ttd_rollout python -u eval/rollout.py \
    --ckpt "$CK" --tag "horizon_EH${EH}" --res 256 --cfg 7.5 --frame-interval 3 \
    --prompt-tag-style explicit --axis-solver sphere --arm-action-mode planning \
    --max-steps-factor 1.5 --skip-anchor-frames 4 --execution-horizon "$EH" \
    --tasks "$TASK" --num-trials "$TRIALS" 2>&1 | grep -E "trial |succ%|Error"
done
echo "HORIZON_SWEEP_DONE"
