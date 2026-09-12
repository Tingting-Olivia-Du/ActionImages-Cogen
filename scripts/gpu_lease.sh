#!/bin/bash
# Serialised GPU leasing for the eval queues.
#
# Five independent queues were each probing cards with their own allocation test and then
# starting a job. A card that passed the probe could be taken by another queue before the
# model finished loading, and heldout_batch_eval exits 0 on that path -- so the failure
# looked like a completed run. Leasing under flock closes the window: the probe and the
# claim happen while holding the lock, and the lease file is removed when the job exits.
#
#   G=$(bash scripts/gpu_lease.sh acquire 28000) ; ... ; bash scripts/gpu_lease.sh release $G
LEASE_DIR="/workspace/ttdu/ActionImages-Cogen/.gpulease"
LOCK="$LEASE_DIR/.lock"
PY=/opt/conda/envs/ttd_train/bin/python
mkdir -p "$LEASE_DIR"

case "$1" in
acquire)
  MB=${2:-28000}
  # mkdir is atomic and needs no file descriptor. flock kept reporting "9: Bad file
  # descriptor" from inside the sweep's nested command substitution even after the fd was
  # opened in the subshell, and the failure mode is silent starvation, so this drops the fd
  # entirely rather than keep guessing at inheritance rules.
  while :; do
    if mkdir "$LOCK.d" 2>/dev/null; then
      trap 'rmdir "$LOCK.d" 2>/dev/null' EXIT
      G=""
      for i in 4 5 6 7 0 1 2 3; do
        [ -e "$LEASE_DIR/$i" ] && continue
        if CUDA_VISIBLE_DEVICES=$i timeout 60 "$PY" -c \
           "import torch;torch.zeros(int($MB*0.95*1024*1024//4),device='cuda');torch.cuda.synchronize()" \
           >/dev/null 2>&1; then
          echo $PPID > "$LEASE_DIR/$i"; G=$i; break
        fi
      done
      rmdir "$LOCK.d" 2>/dev/null; trap - EXIT
      [ -n "$G" ] && { echo "$G"; exit 0; }
    fi
    sleep 120
  done ;;
release)
  rm -f "$LEASE_DIR/${2:?}" ;;
reap)                                                          # drop leases whose owner died
  for f in "$LEASE_DIR"/[0-7]; do
    [ -e "$f" ] || continue
    kill -0 "$(cat "$f" 2>/dev/null)" 2>/dev/null || rm -f "$f"
  done ;;
status)
  for i in 0 1 2 3 4 5 6 7; do
    [ -e "$LEASE_DIR/$i" ] && echo "  GPU $i leased by pid $(cat $LEASE_DIR/$i)"
  done ;;
esac
