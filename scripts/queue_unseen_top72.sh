#!/bin/bash
# Extend every unseen-task evaluation from 66 episodes to all 72.
#
# The 66 was not a design choice. The episode list was built from an earlier run's results
# and happened to omit six close_drawer episodes; the loader sees all twenty and none of the
# six is missing an annotation. An arbitrary subset is the kind of thing a reader cannot
# check, so it gets closed rather than explained. Resume is per-episode, so only the six new
# ones are generated.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
PY=/opt/conda/envs/ttd_train/bin/python
LEASE="bash $REPO/scripts/gpu_lease.sh"
SP="$REPO/outputs"
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
export PYTHONPATH="$REPO"
DATA="$REPO/data/rlbench_unseen_tasks_512_aug"
EPS=$(cat /tmp/unseen_eps_full.txt)

one() {  # tag ckpt mode modalities
  local tag=$1 ck=$2 mode=$3 mods=$4
  local G; G=$($LEASE acquire 28000)
  echo "[t72] $(date -u +%H:%M) START $tag on GPU $G"
  CUDA_VISIBLE_DEVICES=$G $PY -u scripts/heldout_batch_eval.py --ckpt "$ck" --data "$DATA" \
    --tag "$tag" --gpu 0 --mode "$mode" --modalities "$mods" --episodes "$EPS" \
    >> "$REPO/logs_t72_${tag}.txt" 2>&1
  local rc=$?; $LEASE release "$G"
  echo "[t72] $(date -u +%H:%M) $tag exited $rc"
}
one arm6_10000_unseen_task            outputs/.grid_pin/step10000.ckpt iiii depth,segmentation,normal,action
one arm6_10000_fifi_unseen_task       outputs/.grid_pin/step10000.ckpt fifi depth,segmentation,normal
one depthspec_2000_unseen_task  "$SP/specialist_depth__seed42_fi3_512_aug_sr/checkpoint-2000/step2000.ckpt"  iiii depth
one segspec_2000_unseen_task    "$SP/specialist_seg__seed42_fi3_512_aug_sr/checkpoint-2000/step2000.ckpt"    iiii segmentation
one normalspec_2000_unseen_task "$SP/specialist_normal__seed42_fi3_512_aug_sr/checkpoint-2000/step2000.ckpt" iiii normal
one actionspec_4000_unseen_task "$SP/specialist_action__seed42_fi3_512_aug_sr/checkpoint-4000/step4000.ckpt" iiii action
one init_125750_unseen_task /workspace/ttdu/starVLA/playground/Pretrained_models/anyeZHY/ActionImages/step125750.ckpt iiii action
echo "[t72] ALL DONE $(date -u +%H:%M)"
