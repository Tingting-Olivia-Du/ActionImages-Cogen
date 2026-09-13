#!/bin/bash
# Bring a finished (or evicted) run's checkpoints back for evaluation.
#   bash chtc/pull_checkpoints.sh <exp_name> [local_dir]
set -euo pipefail
EXP="${1:?usage: pull_checkpoints.sh <exp_name> [local_dir]}"
DEST="${2:-./outputs}"
NETID="${CHTC_USER:-tdu35}"
HOST="${CHTC_TRANSFER:-transfer.chtc.wisc.edu}"   # transfer node, not the access point
REMOTE="${STAGING_DIR:-/staging/$NETID}/fusion/${EXP}.tar"
mkdir -p "$DEST"
echo "scp ${NETID}@${HOST}:${REMOTE}  ->  $DEST"
scp "${NETID}@${HOST}:${REMOTE}" "$DEST/${EXP}.tar"
tar -xf "$DEST/${EXP}.tar" -C "$DEST" && rm -f "$DEST/${EXP}.tar"
echo "checkpoints:"; ls -d "$DEST/$EXP"/checkpoint-* 2>/dev/null
echo
echo "Next: evaluate it. The eval harness takes the output dir, not the .ckpt:"
echo "  OUT_DIR=$DEST/$EXP TAG=${EXP} bash scripts/wait_and_eval_all.sh"
