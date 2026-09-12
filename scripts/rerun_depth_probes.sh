#!/bin/bash
# Re-run da2/da3 now that the probe decides output space from rho(pred, GT) instead
# of a hardcoded per-model assumption. ~7 min each. Waits for a genuinely claimable card.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
PY=/opt/conda/envs/ttd_train/bin/python
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
export PYTHONPATH="$REPO" N_EP_PER_TASK=5
for i in 1 6 5 7 0 2 3 4; do
  if CUDA_VISIBLE_DEVICES=$i timeout 50 "$PY" -c \
     "import torch;torch.zeros(int(20000*0.9*1024*1024//4),device='cuda');torch.cuda.synchronize()" \
     >/dev/null 2>&1; then G=$i; break; fi
done
[ -z "${G:-}" ] && { echo "[rerun] no card free, aborting"; exit 1; }
echo "[rerun] using GPU $G"
for w in da2 da3; do  # vggt handled separately below (different script)
  echo "[rerun] START $w"
  CUDA_VISIBLE_DEVICES=$G $PY -u scripts/probe_depth_baselines.py $w 40 >> "$REPO/logs_rerun_${w}.txt" 2>&1
  echo "[rerun] $w exited $?"
done
echo "[rerun] DONE"
mv -f "$REPO/reports/external/vggt_n40.json" "$REPO/reports/external/superseded/" 2>/dev/null
echo "[rerun] START vggt"
CUDA_VISIBLE_DEVICES=$G $PY -u scripts/probe_vggt_vs_ours.py 40 >> "$REPO/logs_rerun_vggt.txt" 2>&1
echo "[rerun] vggt exited $?"
echo "[rerun] ALL DONE"
