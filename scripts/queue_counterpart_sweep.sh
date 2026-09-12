#!/bin/bash
# Every counterpart we can run for every row of Table 1, on both evaluation tiers.
#
# The main table reports the best of these per row; all of them are written to
# reports/external/ for the appendix. Running one model per row and calling it "best
# counterpart" would make that column a choice rather than a measurement -- in particular
# the metric-depth models below are the ones that can beat us on the zero-free-parameter
# asymmetry we lean on, so leaving them out would be self-serving.
#
# Leases via scripts/gpu_lease.sh: the probe and the claim happen under one flock, closing
# the window where a card passes a probe and is taken before the model finishes loading.
# heldout_batch_eval exits 0 on that failure, so the race was producing silent no-ops.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
PY=/opt/conda/envs/ttd_train/bin/python
LEASE="bash $REPO/scripts/gpu_lease.sh"
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
export PYTHONPATH="$REPO" N_EP_PER_TASK=5

run() {   # name  mb  cmd...
  local name="$1" mb="$2"; shift 2
  local out="$REPO/reports/external"
  local g; g=$($LEASE acquire "$mb")
  echo "[sweep] $(date -u +%H:%M) START $name on GPU $g"
  CUDA_VISIBLE_DEVICES=$g "$@" >> "$REPO/logs_sweep_${name}.txt" 2>&1
  local rc=$?
  $LEASE release "$g"
  echo "[sweep] $(date -u +%H:%M) $name exited $rc"
}

for CFG in "var1:$REPO/data/rlbench_selfgen_512_aug:1::40" \
           "task:$REPO/data/rlbench_unseen_tasks_512_aug::$REPO/reports/external/unseen_task_eps.txt:66"; do
  IFS=: read -r TAG TREE VAR EPSF N <<< "$CFG"
  export DATA_TREE="$TREE" SPLIT_TAG="_$TAG"
  [ -n "$VAR" ] && export EVAL_VARIATION="$VAR" || unset EVAL_VARIATION
  [ -n "$EPSF" ] && export EPS_FILE="$EPSF" || unset EPS_FILE

  # --- depth: metric (0 free params, same as ours) then relative (1-2 free params) ---
  for M in depthpro da2metric unidepth lotusdepth marigolddepth da2 da3; do
    [ "$M" = unidepth ] && continue            # needs a source install; handled separately
    run "depth_${M}_${TAG}" 20000 $PY -u scripts/probe_depth_baselines.py $M $N
  done
  run "depth_vggt_${TAG}" 20000 $PY -u scripts/probe_vggt_vs_ours.py $N

  # --- normals ---
  for M in lotus marigold; do
    run "normal_${M}_${TAG}" 16000 $PY -u scripts/probe_normal_specialists.py $M $N
  done

  # --- segmentation: class-agnostic, and the named row that currently reads n/a ---
  run "seg_sam_${TAG}"     16000 $PY -u scripts/probe_sam_vs_ours.py $N
  run "seg_clipseg_${TAG}" 12000 $PY -u scripts/probe_named_seg_baseline.py $N
done
echo "[sweep] ALL DONE $(date -u +%H:%M)"
