#!/bin/bash
# Everything still missing from the merged main table, run one job at a time as a GPU frees.
#
# Sequential on purpose: eight cards are already committed to the n=40 scale-up, and the failure
# this queue exists to avoid is the one that just happened -- two jobs launched onto cards that
# looked free by nvidia-smi but held leaked memory from dead processes, and both OOMed at model
# load. Each job here waits for a card that passes a real 28 GB allocation before starting.
#
# ORDER MATTERS. actionspec and init complete Table 1's action row and Table 2's top row, which
# are already-designed cells; the external baselines extend rows that currently exist at n=8 and
# would otherwise sit in the merged table at a different sample size from everything else.
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
PY=/opt/conda/envs/ttd_train/bin/python
SP="$REPO/outputs"
OFFICIAL=/workspace/ttdu/starVLA/playground/Pretrained_models/anyeZHY/ActionImages/step125750.ckpt

free_gpu() {
  while :; do
    for i in 0 1 2 3 4 5 6 7; do
      if CUDA_VISIBLE_DEVICES=$i timeout 50 "$PY" -c \
         "import torch;torch.zeros(int(28000*0.9*1024*1024//4),device='cuda');torch.cuda.synchronize()" \
         >/dev/null 2>&1; then echo "$i"; return 0; fi
    done
    sleep 180
  done
}
run() {  # name, command...
  local name="$1"; shift
  local g; g=$(free_gpu)
  echo "[queue] $(date -u +%H:%M) $name on GPU $g"
  CUDA_VISIBLE_DEVICES="$g" "$@" >> "$REPO/logs_queue_${name}.txt" 2>&1
  echo "[queue] $(date -u +%H:%M) $name exited $?"
}
export -f free_gpu

source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
export PYTHONPATH="$REPO" N_EP_PER_TASK=5

run actionspec $PY -u scripts/heldout_batch_eval.py \
  --ckpt "$SP/specialist_action__seed42_fi3_512_aug_sr/checkpoint-4000/step4000.ckpt" \
  --tag actionspec_4000_matched --gpu 0 --modalities action --episodes "$(cat /tmp/eps40.txt)"

run init $PY -u scripts/heldout_batch_eval.py --ckpt "$OFFICIAL" \
  --tag init_125750 --gpu 0 --modalities action --prompt_tag_style none \
  --episodes "$(cat /tmp/eps40.txt)"

run oursfifi $PY -u scripts/heldout_batch_eval.py --ckpt outputs/.grid_pin/step10000.ckpt \
  --tag arm6_10000_fifi --gpu 0 --mode fifi --modalities depth,normal,segmentation \
  --episodes "$(cat /tmp/eps40.txt)"

run vggt   $PY -u scripts/probe_vggt_vs_ours.py 40
run da2    $PY -u scripts/probe_depth_baselines.py da2 40
run da3    $PY -u scripts/probe_depth_baselines.py da3 40
run sam    $PY -u scripts/probe_sam_vs_ours.py 40

echo "[queue] ALL DONE $(date -u +%H:%M)"
