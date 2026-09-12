#!/bin/bash
# One held-out batch eval on one explicitly-masked GPU.
#   GPU=1 CKPT=... TAG=... MODS=depth bash scripts/run_one_heldout_eval.sh
# CUDA_VISIBLE_DEVICES is exported HERE, not passed via --gpu: `from inference import
# build_pipeline` initialises CUDA at import time, so --gpu is silently ignored and the process
# lands on physical GPU 0. See scripts/run_heldout_eval_queue.sh for the full explanation.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="${GPU:?set GPU}"
echo "[eval] $(date) $TAG ($MODS) on physical GPU $GPU"
python -u scripts/heldout_batch_eval.py --ckpt "$CKPT" --tag "$TAG" --gpu 0 --modalities "$MODS"
echo "[eval] $(date) $TAG exited $?"
