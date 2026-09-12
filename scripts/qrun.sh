#!/bin/bash
# Launch a queue in its own process group so it can be stopped whole.
#
# Killing a queue's top-level pid leaves its `gpu_lease.sh acquire` child alive, and that
# child has already parsed the old script into memory -- so it keeps retrying with stale
# code and writing to the inherited log fd at its pre-truncation offset. Two such orphans
# survived for an hour and made a fixed lease look broken. setsid + `kill -- -PGID` ends
# the whole group instead.
#
#   bash scripts/qrun.sh start <name> <script> [args...]
#   bash scripts/qrun.sh stop  <name>
#   bash scripts/qrun.sh list
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
PIDDIR="$REPO/.qrun"; mkdir -p "$PIDDIR"
case "$1" in
start)
  name="$2"; shift 2
  f="$PIDDIR/$name.pgid"
  # Refuse to start over a live queue. Overwriting the pgid file orphans the running group:
  # it keeps working, keeps its GPU lease and keeps appending to the same log, but `stop`
  # and `list` now point at the new one. That happened once -- two copies of the same queue
  # raced for the same card while only one was visible.
  if [ -f "$f" ] && kill -0 -- -"$(cat "$f")" 2>/dev/null; then
    echo "[qrun] REFUSING: '$name' is already running (pgid $(cat "$f"), $(pgrep -g "$(cat "$f")" | wc -l) procs)." >&2
    echo "[qrun]   stop it first:  bash scripts/qrun.sh stop $name" >&2
    echo "[qrun]   or use a different name." >&2
    exit 1
  fi
  [ -f "$f" ] && rm -f "$f"          # stale file from a dead group
  setsid nohup bash "$@" > "$REPO/logs_${name}_queue.txt" 2>&1 &
  echo $! > "$f"
  echo "[qrun] $name started, pgid $(cat "$f")" ;;
stop)
  name="$2"; f="$PIDDIR/$name.pgid"
  [ -f "$f" ] || { echo "[qrun] no such queue: $name"; exit 1; }
  pg=$(cat "$f"); kill -- -"$pg" 2>/dev/null
  sleep 2; kill -9 -- -"$pg" 2>/dev/null
  rm -f "$f"; echo "[qrun] $name stopped (pgid $pg)" ;;
prune)                                                       # drop pgid files of dead groups
  for f in "$PIDDIR"/*.pgid; do
    [ -e "$f" ] || continue
    kill -0 -- -"$(cat "$f")" 2>/dev/null || { echo "  pruned $(basename "$f" .pgid)"; rm -f "$f"; }
  done ;;
list)
  for f in "$PIDDIR"/*.pgid; do
    [ -e "$f" ] || continue
    pg=$(cat "$f"); n=$(basename "$f" .pgid)
    if kill -0 -- -"$pg" 2>/dev/null; then
      echo "  $n  pgid=$pg  alive  ($(pgrep -g "$pg" 2>/dev/null | wc -l) procs)"
    else echo "  $n  pgid=$pg  dead"; fi
  done ;;
esac
