#!/bin/bash
# ============================================================================
# Stop every job THIS repo is running, then immediately claim the GPUs.
# ----------------------------------------------------------------------------
#   bash scripts/stop_and_claim.sh                      # dry run: list, kill nothing
#   bash scripts/stop_and_claim.sh --yes                # stop everything, leave GPUs idle
#   bash scripts/stop_and_claim.sh --yes --claim 'GPUS=0,1,2,6 bash scripts/train_fusion.sh fusion0'
#
# ORDER IS THE WHOLE POINT, and it is not the obvious one.
#
#   1. WATCHDOGS FIRST.  scripts/queue_*.sh and supervise_*.sh exist to grab a GPU the moment
#      one frees. Kill training first and they re-launch it inside the gap -- so every later
#      step gets silently undone and the run you meant to start finds no free card.
#   2. EVAL SECOND.  rollout.py under xvfb-run: kill the wrapper and the child, in that order,
#      or xvfb-run's shell respawns nothing but does leave the X server holding a lock.
#   3. TRAINING LAST.  torchrun, then its workers explicitly -- torchrun does not always take
#      them with it, and a surviving worker keeps ~40G of VRAM allocated.
#   4. Then wait on nvidia-smi and hand the cards straight to --claim, with nothing in between.
#
# WHY PRECISE PIDS, NEVER `pkill -f python`. GPU 5 (and whatever else shows up) belongs to
# another tenant whose processes are not even in our PID namespace -- nvidia-smi reports "no
# running processes" for them. A pattern kill either misses them (fine) or catches a colleague's
# job (not fine). Everything below is matched on /proc/<pid>/cmdline CONTAINING THIS REPO PATH,
# so it can only ever hit our own work.
#
# WHY "memory.used" AND NOT "utilization.gpu". A dying rank drops util to 0% instantly but holds
# its allocation until the process is reaped. Polling util reports the card free while 40G is
# still committed, and the claim then OOMs on launch.
# ============================================================================
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DO_IT=0
CLAIM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --yes)   DO_IT=1; shift ;;
    --claim) CLAIM="${2:?--claim needs a command}"; shift 2 ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
done

# ---- enumerate our own processes, tiered ----------------------------------------------
# Tier is decided by the cmdline: watchdog scripts, then eval, then training.
declare -a T1 T2 T3
for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
  [ -r "/proc/$pid/cmdline" ] || continue
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
  [ -z "$cmd" ] && continue
  # Only ever our own repo. This single test is what makes the script safe to run on a shared box.
  case "$cmd" in *"$REPO"*|*ActionImages-Cogen*) ;; *) continue ;; esac
  # Never kill this script, its shell, or the agent harness that launched it.
  [ "$pid" = "$$" ] && continue
  case "$cmd" in *stop_and_claim.sh*|*claude*|*shell-snapshots*) continue ;; esac
  case "$cmd" in
    *queue_*.sh*|*supervise*.sh*|*waiter*)          T1+=("$pid") ;;
    *rollout.py*|*eval/*|*eval_*.py*|*xvfb-run*)    T2+=("$pid") ;;
    *torchrun*|*train.py*)                          T3+=("$pid") ;;
  esac
done

show() {  # show <label> <pid...>
  local label="$1"; shift
  if [ $# -eq 0 ]; then echo "  ($label) none"; return; fi
  for p in "$@"; do
    printf "  (%s) %-8s %s\n" "$label" "$p" "$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-110)"
  done
}
echo "=== processes belonging to $REPO ==="
show watchdog ${T1[@]+"${T1[@]}"}
show eval     ${T2[@]+"${T2[@]}"}
show training ${T3[@]+"${T3[@]}"}
echo
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

if [ "$DO_IT" -ne 1 ]; then
  echo
  echo "DRY RUN -- nothing killed. Re-run with --yes to stop these, and add"
  echo "  --claim '<command>' to launch straight into the freed GPUs."
  exit 0
fi

kill_tier() {  # kill_tier <label> <pid...>
  local label="$1"; shift
  [ $# -eq 0 ] && { echo "-- $label: nothing to stop"; return; }
  echo "-- $label: TERM $*"
  kill -TERM "$@" 2>/dev/null
  # Give them a moment to exit cleanly; DeepSpeed ranks in particular need to release VRAM.
  for _ in $(seq 1 15); do
    local alive=0
    for p in "$@"; do [ -d "/proc/$p" ] && alive=1; done
    [ "$alive" -eq 0 ] && break
    sleep 1
  done
  for p in "$@"; do
    if [ -d "/proc/$p" ]; then echo "-- $label: KILL $p (did not exit)"; kill -KILL "$p" 2>/dev/null; fi
  done
}

kill_tier watchdog ${T1[@]+"${T1[@]}"}   # FIRST, or everything below gets relaunched
kill_tier eval     ${T2[@]+"${T2[@]}"}
kill_tier training ${T3[@]+"${T3[@]}"}

# ---- wait for the VRAM to actually come back -------------------------------------------
# Bounded: if a card is still held after 3 minutes it is not ours to wait for, and the claim
# should fail loudly rather than sit here forever.
echo "-- waiting for VRAM release (memory.used, not utilization)"
for i in $(seq 1 90); do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' '$2 > 1000 {printf "%s(%sMiB) ", $1, $2}')
  [ -z "$busy" ] && break
  [ $((i % 10)) -eq 0 ] && echo "   still held: $busy"
  sleep 2
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
       | awk -F', ' '$2 > 1000 {printf "%s ", $1}')
[ -n "$busy" ] && echo "!! GPUs still occupied: $busy  (another tenant? check before claiming)"

[ -z "$CLAIM" ] && { echo "no --claim given; GPUs left idle."; exit 0; }
# Straight into the claim, with nothing in between -- any gap is an invitation.
echo "-- claiming: $CLAIM"
cd "$REPO" && exec bash -c "$CLAIM"
