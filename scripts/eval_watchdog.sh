#!/bin/bash
# Auto-resume guardian for the evaluation campaigns, the counterpart of train_watchdog.sh.
#
# WHY: the closed-loop program is ~20 h of 8-GPU work split over several dispatchers, and
# nothing was restarting them. `setsid` (verified: every dispatcher is its own session leader
# with init as its ancestor) protects them from the launching shell going away; this covers
# what setsid cannot -- an OOM kill, a driver fault, a transient disk-full, a stray kill.
#
# THE ONE NON-OBVIOUS RULE: A RESTART REWINDS THE QUEUE INDEX TO 0.
# The dispatchers hand out jobs by advancing a shared index under flock, so a worker killed
# mid-job leaves an index that has already moved PAST that job -- a naive restart would skip
# it forever and the campaign would quietly finish with a missing task. Rewinding to 0 and
# letting `cl_json_complete.py` skip the finished jobs makes a restart lossless and
# idempotent, at the cost of one cheap rescan.
#
# It never starts work it should not:
#   * refuses to launch anything with less than MIN_FREE_GB free (it would just die again);
#   * leaves a dispatcher alone while its workers are alive, even if they are idle;
#   * does not resurrect phase 1 once phase 2 has taken over (phase 2's queue file is the
#     handover marker), and does not start phase 2 before arm7's RLBench jobs are done --
#     that ordering is the user's priority, not an accident;
#   * only supervises the offline campaign after someone has started it once (its queue file
#     is the opt-in marker), so the watchdog never decides on its own to spend GPUs on it.
#
# Run detached:
#   cd /workspace/1228_tingting/ActionImages-Cogen
#   setsid nohup bash scripts/eval_watchdog.sh >> logs/eval_watchdog.log 2>&1 < /dev/null &
set -uo pipefail
REPO=/workspace/1228_tingting/ActionImages-Cogen
cd "$REPO"
INTERVAL="${INTERVAL:-180}"
MIN_FREE_GB="${MIN_FREE_GB:-40}"
TRIALS="${TRIALS:-20}"
LOGDIR="$REPO/logs"
P1_LOGS="$LOGDIR/logs_cl_unseen14"
P2_LOGS="$LOGDIR/logs_cl_phase2"
OFF_LOGS="$LOGDIR/logs_offline_unseen14"
GT_TASKS="close_box close_drawer close_microwave take_lid_off_saucepan toilet_seat_down \
basketball_in_hoop meat_on_grill light_bulb_out take_item_out_of_drawer take_money_out_safe \
put_rubbish_in_bin stack_cups open_window wipe_desk"

ts() { date '+%F %T'; }

MY_PGID=$(ps -o pgid= -p $$ | tr -d ' ')
# `pgrep -f` matches ANY command line containing the pattern, including the watchdog's own
# subshells and the `bash -c` wrapper of whoever is inspecting it. That is not hypothetical:
# an earlier waiter in this campaign matched its own heredoc and never fired. So: ignore
# ourselves and anything in our process group. Every dispatcher is launched with setsid, so
# it lives in a different group and is never filtered out by mistake.
alive() {
  local pat="$1" pid pgid a1
  for pid in $(pgrep -f -- "$pat" 2>/dev/null); do
    [ "$pid" = "$$" ] && continue
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ "$pgid" = "$MY_PGID" ] && continue
    # Skip shells running an INLINE command string (`bash -c '...'`). Every Claude Code tool
    # call is one of those, its string quotes whole scripts verbatim, and the orphaned
    # wrappers outlive the call -- so `pgrep -f` matches them for essentially ANY pattern.
    # Measured: `pgrep -f run_cl_phase2.sh` returned a wrapper from an unrelated call while
    # phase 2 had never started, which would have made the watchdog believe a dead campaign
    # was alive. Our dispatchers are always launched as `bash <path>`, never `bash -c`.
    a1=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | sed -n 2p)
    [ "$a1" = "-c" ] && continue
    return 0
  done
  return 1
}
free_gb() { df -BG --output=avail /workspace | tail -1 | tr -dc '0-9'; }

disk_ok() {
  local g; g=$(free_gb)
  if [ "${g:-0}" -lt "$MIN_FREE_GB" ]; then
    echo "[watchdog] $(ts) only ${g}G free (<${MIN_FREE_GB}G) -- NOT starting anything"
    return 1
  fi
  return 0
}

# done/total for a queue file, via the same completeness test the dispatchers use
status() { python "$REPO/scripts/cl_campaign_status.py" "$1" "$TRIALS" 2>/dev/null || echo "0 0"; }

rewind_and_keep_log() {  # rewind_and_keep_log <queue-dir> <log-to-preserve>
  [ -f "$1/queue.txt" ] && echo 0 > "$1/queue.idx"
  [ -f "$2" ] && cp -f "$2" "${2%.log}.$(date +%m%d_%H%M%S).crashed.log"
}

arm7_rlb_done() {
  local n=0 t
  for t in $GT_TASKS; do
    python "$REPO/scripts/cl_json_complete.py" \
      "$REPO/reports/closedloop_unseen14/rollout_u14_arm7_6k_$t.json" "$TRIALS" && n=$((n+1))
  done
  [ "$n" -ge 14 ]
}

gt_remaining() {  # how many of the 14 corrected-protocol GT gates are still missing
  local n=0 t
  for t in $GT_TASKS; do
    python "$REPO/scripts/cl_json_complete.py" \
      "$REPO/reports/closedloop_unseen14/rollout_u14_gtreplay2_$t.json" "$TRIALS" || n=$((n+1))
  done
  echo "$n"
}

echo "[watchdog] start $(ts) interval=${INTERVAL}s trials=${TRIALS} min_free=${MIN_FREE_GB}G"
while true; do
  # ---------------------------------------------------------------- closed loop
  if [ -f "$P2_LOGS/queue.txt" ]; then
    # Phase 2 owns the GPUs from here on.
    read -r done total <<< "$(status "$P2_LOGS/queue.txt")"
    if [ "$total" -gt 0 ] && [ "$done" -lt "$total" ] && ! alive "run_cl_phase2.sh"; then
      if disk_ok; then
        echo "[watchdog] $(ts) phase2 DOWN at ${done}/${total} jobs -- restarting (rewound)"
        rewind_and_keep_log "$P2_LOGS" "$LOGDIR/logs_cl_phase2_dispatch.log"
        FRESH_QUEUE=0 setsid nohup bash "$REPO/scripts/run_cl_phase2.sh" \
          > "$LOGDIR/logs_cl_phase2_dispatch.log" 2>&1 < /dev/null &
        sleep 60
      fi
    fi
  else
    read -r done total <<< "$(status "$P1_LOGS/queue.txt")"
    if arm7_rlb_done; then
      # Handover time. chain_phase2.sh drains phase 1 and starts phase 2; keep IT alive
      # rather than racing it with a second drain.
      if ! alive "chain_phase2.sh" && ! alive "run_cl_phase2.sh" && disk_ok; then
        echo "[watchdog] $(ts) arm7 RLBench done but no handover running -- starting phase 2"
        wc -l < "$P1_LOGS/queue.txt" > "$P1_LOGS/queue.idx"   # drain phase 1
        while alive "run_unseen14_campaign.sh"; do sleep 30; done
        # RE-CHECK after the drain. The drain can take an hour (workers finish their current
        # 20-trial job rather than being killed), and phase 2 may well have been started by
        # hand in the meantime on the cards that freed up early. Starting a second instance
        # here would reset the shared queue index to 0 and put two workers on one card.
        if alive "run_cl_phase2.sh"; then
          echo "[watchdog] $(ts) phase 2 already running after the drain -- not starting a second"
        else
          FRESH_QUEUE="${FRESH_QUEUE_P2:-1}" setsid nohup bash "$REPO/scripts/run_cl_phase2.sh" \
            > "$LOGDIR/logs_cl_phase2_dispatch.log" 2>&1 < /dev/null &
          sleep 60
        fi
      fi
    elif [ "$total" -gt 0 ] && [ "$done" -lt "$total" ] && ! alive "run_unseen14_campaign.sh"; then
      if disk_ok; then
        echo "[watchdog] $(ts) phase1 DOWN at ${done}/${total} jobs -- restarting (rewound)"
        rewind_and_keep_log "$P1_LOGS" "$LOGDIR/logs_cl_unseen14_dispatch.log"
        FRESH_QUEUE=0 setsid nohup bash "$REPO/scripts/run_unseen14_campaign.sh" \
          > "$LOGDIR/logs_cl_unseen14_dispatch.log" 2>&1 < /dev/null &
        sleep 60
      fi
    fi
  fi

  # ---------------------------------------------------------------- GT gate (CPU only)
  n=$(gt_remaining)
  if [ "$n" -gt 0 ] && ! alive "run_unseen14_gt_replay.sh" && ! alive "chain_gt2.sh"; then
    if disk_ok; then
      echo "[watchdog] $(ts) GT gate (corrected protocol) missing $n/14 -- restarting"
      TAGPFX=u14_gtreplay2 setsid nohup bash "$REPO/scripts/run_unseen14_gt_replay.sh" \
        > "$LOGDIR/logs_cl_unseen14_gt_dispatch2.log" 2>&1 < /dev/null &
      sleep 30
    fi
  fi

  # ---------------------------------------------------------------- offline (opt-in)
  if [ -f "$OFF_LOGS/queue.txt" ] && ! alive "run_offline_unseen14.sh"; then
    idx=$(cat "$OFF_LOGS/queue.idx" 2>/dev/null || echo 0)
    tot=$(wc -l < "$OFF_LOGS/queue.txt")
    if [ "$idx" -lt "$tot" ] && disk_ok; then
      echo "[watchdog] $(ts) offline DOWN at ${idx}/${tot} jobs -- restarting"
      # heldout_batch_eval.py resumes per (episode, modality) cell from its own
      # per_episode.json, so re-running a job repeats no generation.
      echo 0 > "$OFF_LOGS/queue.idx"
      FRESH_QUEUE=0 setsid nohup bash "$REPO/scripts/run_offline_unseen14.sh" \
        > "$LOGDIR/logs_offline_dispatch.log" 2>&1 < /dev/null &
      sleep 60
    fi
  fi

  sleep "$INTERVAL"
done
