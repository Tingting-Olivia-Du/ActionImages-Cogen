#!/bin/bash
# Fresh official-checkpoint closed-loop campaign under the CURRENT (post-anchor-fix) eval/rollout.py.
#
# Deliberately NOT scripts/queue_closedloop.sh -- that script hardcodes
# --prompt-tag-style explicit --cfg 7.5 --frame-interval 3, which is the protocol for OUR
# fine-tuned arms (they were trained on tag-prefixed prompts at frame_interval=3). The official
# checkpoint was never trained with modality tags and its own established protocol
# (CAMPAIGN_200.md) is cfg=10.0, frame_interval=4, prompt_tag_style=none, zero-shot on this rig.
# Using the fine-tuned-arm protocol on the official checkpoint would silently run it out of
# distribution and bias the comparison against it.
#
# Every existing official/arm4 closed-loop reference json in this repo predates the anchor-frame
# fix (reports/closedloop/archive_pre_anchorfix/, mtime 2026-08-13, vs eval/rollout.py's last
# edit 2026-08-22 which is where --skip-anchor-frames defaulting to 4 came from) -- so this run
# is not just "official instead of arm0", it's also the first POST-fix official baseline at
# n=20/task, needed to validly pair against reports/closedloop/rollout_arm6_10000.json.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"
cd "$REPO"

echo "[official-cl] $(date) launching official step125750 closed-loop, n=20/task, GPU 0"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

source /opt/conda/etc/profile.d/conda.sh
source /workspace/ttdu/ttd/scripts/env_eval.rc
conda activate ttd_rollout
export PYTHONPATH="$REPO"

CUDA_VISIBLE_DEVICES=0 xvfb-run -a python -u eval/rollout.py \
  --ckpt /workspace/ttdu/starVLA/playground/Pretrained_models/anyeZHY/ActionImages/step125750.ckpt \
  --tag official_125750_postfix_n20 --res 512 --cfg 10.0 --steps 50 \
  --frame-interval 4 --prompt-tag-style none --axis-solver sphere \
  --arm-action-mode planning --max-steps-factor 1.5 --record-video \
  --tasks close_box close_drawer push_buttons meat_off_grill open_drawer \
  --num-trials 20
CL_EXIT=$?
echo "[official-cl] $(date) rollout exit=$CL_EXIT -> reports/closedloop/rollout_official_125750_postfix_n20.json"

if [ "$CL_EXIT" -eq 0 ]; then
  echo "[official-cl] $(date) pairing against arm6@10000"
  python eval/aggregate_closedloop.py --compare \
    reports/closedloop/rollout_arm6_10000.json \
    reports/closedloop/rollout_official_125750_postfix_n20.json \
    --out reports/closedloop/compare_arm6_vs_official_postfix.json
  echo "[official-cl] $(date) compare exit=$? -> reports/closedloop/compare_arm6_vs_official_postfix.json"
fi

echo "[official-cl] $(date) stage 1 (official closed-loop) done -- GPU 0 free for held-out batch eval"
