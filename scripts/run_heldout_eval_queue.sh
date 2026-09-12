#!/bin/bash
# Sequential held-out batch-eval queue on ONE GPU (E1 in paper/EXPERIMENT_OUTLINE.md).
#
# ⚠️ CUDA_VISIBLE_DEVICES IS SET HERE, IN THE SHELL, NOT VIA --gpu. `from inference import
# build_pipeline` initialises CUDA at IMPORT time (verified: torch.cuda.is_initialized() flips
# to True on that import alone), and every eval script in this repo does that import at module
# level, before main() gets a chance to set os.environ["CUDA_VISIBLE_DEVICES"]. So --gpu N is
# silently ignored and the process lands on physical GPU 0 while cheerfully printing "using GPU
# N" -- exactly the failure modality_mode_grid.py's own comments describe. Masking in the shell
# is the only placement that cannot lose this race. After masking, the target card is index 0
# inside the process, which is why --gpu 0 is passed below.
#
# Each entry is "ckpt|tag|modalities". Restricting --modalities matters for the specialists: a
# depth specialist only ever produces video+depth, so evaluating four modalities would waste ~3h
# per checkpoint generating templates it was never trained on.
#
# CHECKPOINT CHOICE IS NOT ARBITRARY (EXPERIMENT_OUTLINE.md E2): arm6 gets ~2000 updates on each
# perception modality over its 10000 steps (p=0.2), so the matched-exposure comparison uses each
# perception specialist's checkpoint-2000, NOT its final checkpoint. The final checkpoint is a
# separate reading -- the specialist's ceiling at roughly twice arm6's exposure.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
export PYTHONPATH="$REPO"
GPU="${GPU:?set GPU=<physical index>}"
export CUDA_VISIBLE_DEVICES="$GPU"

QUEUE=(
  "outputs/.grid_pin/step10000.ckpt|arm6_10000|depth,segmentation,normal,action"
  "outputs/specialist_depth__seed42_fi3_512_aug_sr/checkpoint-2000/step2000.ckpt|depthspec_2000_matched|depth"
  "outputs/specialist_seg__seed42_fi3_512_aug_sr/checkpoint-2000/step2000.ckpt|segspec_2000_matched|segmentation"
  "outputs/specialist_depth__seed42_fi3_512_aug_sr/checkpoint-2500/step2500.ckpt|depthspec_2500_ceiling|depth"
  "outputs/specialist_seg__seed42_fi3_512_aug_sr/checkpoint-3000/step3000.ckpt|segspec_3000_ceiling|segmentation"
)

echo "[queue] physical GPU $GPU masked to index 0 for every child process"
for entry in "${QUEUE[@]}"; do
  IFS='|' read -r CK TAG MODS <<< "$entry"
  if [ ! -f "$CK" ]; then echo "[queue] SKIP $TAG: no checkpoint at $CK"; continue; fi
  if [ -f "reports/heldout_batch_eval/$TAG/summary.json" ]; then
    echo "[queue] SKIP $TAG: already has summary.json"; continue
  fi
  echo "[queue] $(date) === $TAG ($MODS) ==="
  python -u scripts/heldout_batch_eval.py --ckpt "$CK" --tag "$TAG" --gpu 0 --modalities "$MODS"
  echo "[queue] $(date) $TAG exited $?"
done
echo "[queue] ALL DONE"
