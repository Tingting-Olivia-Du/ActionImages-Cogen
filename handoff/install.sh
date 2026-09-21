#!/bin/bash
# Install the sim-replay perception code into an ActionImages-Cogen checkout, as WHOLE FILES,
# then prove it by hash and by running it.
#
#   bash install.sh /path/to/ActionImages-Cogen              # everything
#   bash install.sh /path/to/ActionImages-Cogen --scorer-only
#
# --scorer-only installs the scorer, its oracle test and the doc, and leaves eval/rollout*.py
# ALONE. Use it on a machine that is MID-CAMPAIGN: replacing rollout.py under a running queue
# makes the next job launched pick up different code from the previous one, and the dumps of a
# single table would then come from two writers.
#
# Every file that is replaced is first copied to <file>.orig.<timestamp>, so a local edit is
# never lost -- diff against it if the target had changes of its own.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${1:?usage: install.sh <repo root> [--scorer-only]}"
MODE="${2:-all}"
[ -f "$DST/train.py" ] && [ -d "$DST/eval" ] || { echo "!! $DST does not look like the repo root"; exit 2; }
TS="$(date +%Y%m%d_%H%M%S)"
( cd "$HERE/files" && sha256sum -c ../MANIFEST.sha256 --quiet ) || { echo "!! bundle is corrupt (hash mismatch before install)"; exit 3; }
while read -r _hash rel; do
  case "$rel" in eval/rollout*.py) [ "$MODE" = "--scorer-only" ] && { echo "  skip $rel (--scorer-only)"; continue; } ;; esac
  mkdir -p "$DST/$(dirname "$rel")"
  if [ -f "$DST/$rel" ] && ! cmp -s "$DST/$rel" "$HERE/files/$rel"; then
    cp -p "$DST/$rel" "$DST/$rel.orig.$TS"; echo "  backup $rel -> $rel.orig.$TS"
  fi
  cp "$HERE/files/$rel" "$DST/$rel"; echo "  installed $rel"
done < "$HERE/MANIFEST.sha256"
chmod +x "$DST/scripts/run_percep_simreplay.sh" 2>/dev/null || true
echo "now run:  bash $HERE/verify.sh $DST $MODE"
