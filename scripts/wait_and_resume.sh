#!/bin/bash
# Wait for FOUR free GPUs, then resume a finished fusion arm to a higher step count.
#
#   OUT_DIR=outputs/fusion0__seed42_fi3_512_aug_wide_sr_F1 STEPS=10000 \
#     bash scripts/wait_and_resume.sh
#
# WHY EXACTLY FOUR, and why this is not tunable: ZeRO-2 shards the optimiser state per rank, and
# checkpoint-5000/global_step5000/ contains bf16_zero_pp_rank_0..3 -- four shards. DeepSpeed cannot
# load four shards into three ranks. Resuming with a different world size fails, and resuming with
# a different BATCH (which is what a different rank count means here, since
# per_device_train_batch_size is 1) would silently change the optimisation problem halfway through
# a single curve.
#
# RESUME MECHANICS. A non-empty --output_dir makes train.py's find_latest_checkpoint take over and
# IGNORE --init_ckpt_path; train_fusion.sh prints which of the two the weights really come from.
# That is the documented way to continue an arm. Two things must not change or the resume breaks:
#   * DS_CONFIG -- configs/zero2_offload.json, the same file the first 5,000 steps used. Swapping
#     to a config whose `optimizer` block differs makes the checkpoint unloadable in BOTH
#     directions (the LR scheduler's state moves between DeepSpeed and HF).
#   * CKPT_EVERY -- kept at 1000 so the checkpoint-versus-step curve stays evenly sampled.
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

OUT_DIR="${OUT_DIR:?set OUT_DIR to the output dir of the arm}"
STEPS="${STEPS:-10000}"
NEED=${NEED:-31000}
NGPU=${NGPU:-4}
MIN_DISK_G=${MIN_DISK_G:-140}

FROM=$(ls -d "$OUT_DIR"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
[ -n "$FROM" ] || { echo "NO_CHECKPOINT in $OUT_DIR"; exit 2; }
TIP="$OUT_DIR/checkpoint-$FROM/global_step$FROM"
if [ ! -d "$TIP" ]; then
  echo "NOT_RESUMABLE: $OUT_DIR/checkpoint-$FROM has no global_step$FROM/."
  echo "  That directory is the ONLY thing that lets a run continue; weights alone are refused by"
  echo "  _check_resume_consistency (it would restart the step counter at 0 and overwrite the"
  echo "  existing checkpoint dirs). Nothing to do but retrain or pass --allow_step_restart True"
  echo "  with a fresh --output_dir."
  exit 3
fi
SHARDS=$(ls "$TIP"/bf16_zero_pp_rank_*_optim_states.pt 2>/dev/null | wc -l)
if [ "$SHARDS" -ne "$NGPU" ]; then
  echo "RANK_MISMATCH: $TIP has $SHARDS optimiser shards but NGPU=$NGPU."
  echo "  ZeRO-2 cannot redistribute shards across a different world size. Use NGPU=$SHARDS."
  exit 4
fi
[ "$FROM" -ge "$STEPS" ] && { echo "ALREADY_AT_OR_PAST: checkpoint-$FROM >= STEPS=$STEPS"; exit 0; }
echo "$(date +%H:%M:%S) RESUME_TARGET from=$FROM to=$STEPS shards=$SHARDS out=$OUT_DIR"

while true; do
  FREE=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
         | awk -F', ' -v n="$NEED" '($3-$2)>n {print $1}' | head -"$NGPU" | paste -sd, -)
  CNT=$(echo "$FREE" | tr ',' '\n' | grep -c .)
  DISK=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
  if [ "$CNT" -eq "$NGPU" ]; then
    if [ "${DISK:-0}" -lt "$MIN_DISK_G" ]; then
      # Refuse rather than start: a run that fills the volume mid-checkpoint corrupts the tip it
      # was writing AND loses the one it was about to prune.
      echo "$(date +%H:%M:%S) HOLDING: ${NGPU} GPUs free ($FREE) but only ${DISK}G disk < ${MIN_DISK_G}G"
      sleep 300; continue
    fi
    echo "$(date +%H:%M:%S) CLAIMING GPUS $FREE  disk=${DISK}G"
    GPUS="$FREE" DATASET=rlbench_selfgen_512_aug_wide STEPS="$STEPS" OUT="$OUT_DIR" \
      bash scripts/train_fusion.sh fusion0 >> "$REPO/logs_fusion0_wide_resume.txt" 2>&1
    rc=$?
    echo "$(date +%H:%M:%S) TRAIN_RETURNED rc=$rc"
    NOW=$(ls -d "$OUT_DIR"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
    echo "$(date +%H:%M:%S) latest checkpoint now $NOW (was $FROM, target $STEPS)"
    if [ "${NOW:-0}" -ge "$STEPS" ]; then echo "RESUME_DONE"; exit 0; fi
    # Progress but not finished -> a tenant probably took a card. Re-wait and continue from the
    # new tip; the next pass re-reads the latest checkpoint, so no progress is repeated.
    echo "$(date +%H:%M:%S) RESUME_INCOMPLETE -- re-waiting"
    FROM="${NOW:-$FROM}"
    sleep 120
  else
    sleep 60
  fi
done
