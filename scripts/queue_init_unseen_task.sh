#!/bin/bash
# The released ActionImages checkpoint on the never-trained tasks.
#
# It is the counterpart for both action rows of the main table, but it was only ever run on
# the eight-task tree, so the unseen-task cell had no counterpart at all -- which would read
# as "no one else can do this" when we simply had not run it there. Same 66 episodes, same
# seed, same prompt-tag style the released model was trained with (none).
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
PY=/opt/conda/envs/ttd_train/bin/python
LEASE="bash $REPO/scripts/gpu_lease.sh"
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
export PYTHONPATH="$REPO"
OFFICIAL=/workspace/ttdu/starVLA/playground/Pretrained_models/anyeZHY/ActionImages/step125750.ckpt
out="$REPO/reports/heldout_batch_eval/init_125750_unseen_task/per_episode.json"
for a in 1 2 3; do
  [ -s "$out" ] && break
  G=$($LEASE acquire 28000)
  echo "[init-task] $(date -u +%H:%M) START attempt $a on GPU $G"
  CUDA_VISIBLE_DEVICES=$G $PY -u scripts/heldout_batch_eval.py --ckpt "$OFFICIAL" \
    --data "$REPO/data/rlbench_unseen_tasks_512_aug" --tag init_125750_unseen_task \
    --gpu 0 --modalities action --prompt_tag_style none \
    --episodes "$(cat /tmp/unseen_eps.txt)" >> "$REPO/logs_init_unseen_task.txt" 2>&1
  rc=$?; $LEASE release "$G"
  [ -s "$out" ] && { echo "[init-task] OK (rc=$rc)"; break; }
  echo "[init-task] no output (rc=$rc) -- retry"; sleep 60
done
echo "[init-task] ALL DONE $(date -u +%H:%M)"
