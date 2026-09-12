#!/bin/bash
# Re-run every external baseline with the seeding fix.
#
# Two bugs made the previous numbers unusable:
#  1) getitem draws the camera view and temporal window at RANDOM and records them as
#     provenance. Our own eval seeds with 42; these probes never did, so each baseline
#     was scored on different pixels than our model. Two unseeded runs of the same probe
#     differed by up to 0.12 AbsRel on a single episode.
#  2) probe_depth_baselines/vggt called _save() before the block defining `chosen`
#     and `mean_rho`, so they crashed at write time (UnboundLocalError).
# Both fixed; every probe now seeds 42 immediately before ds.getitem, and decides
# depth-vs-disparity from rho(pred, GT) rather than from a hardcoded assumption.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
PY=/opt/conda/envs/ttd_train/bin/python
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
export PYTHONPATH="$REPO" N_EP_PER_TASK=5

claim() {
  while :; do
    for i in 1 6 5 7 0 2 3 4; do
      if CUDA_VISIBLE_DEVICES=$i timeout 50 "$PY" -c \
         "import torch;torch.zeros(int(20000*0.9*1024*1024//4),device='cuda');torch.cuda.synchronize()" \
         >/dev/null 2>&1; then echo "$i"; return 0; fi
    done
    echo "[ext] no card claimable, retry 3m" >&2; sleep 180
  done
}
for job in "da2:scripts/probe_depth_baselines.py da2 40" \
           "da3:scripts/probe_depth_baselines.py da3 40" \
           "vggt:scripts/probe_vggt_vs_ours.py 40" \
           "sam:scripts/probe_sam_vs_ours.py 40"; do
  name=${job%%:*}; args=${job#*:}
  G=$(claim); echo "[ext] $(date -u +%H:%M) START $name on GPU $G"
  CUDA_VISIBLE_DEVICES=$G $PY -u $args >> "$REPO/logs_ext_${name}.txt" 2>&1
  echo "[ext] $(date -u +%H:%M) $name exited $?"
done
echo "[ext] ALL EXTERNAL BASELINES DONE (seeded) $(date -u +%H:%M)"
