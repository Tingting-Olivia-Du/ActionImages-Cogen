#!/bin/bash
# ============================================================================
# 512x512 generation into a NEW tree. Does NOT touch data/rlbench_selfgen_v2.
# ----------------------------------------------------------------------------
#   bash scripts/gen_launch_512.sh
#
# WHY A SEPARATE LAUNCHER. gen_launch.sh shards TASKS round-robin across workers,
# so with 16 tasks it cannot use more than 16 workers -- ask for 48 and 32 sit idle.
# This one shards (task, variation, seed-range) instead, which is what makes a wide
# pool possible.
#
# WHY LP_NUM_THREADS. CoppeliaSim under plain xvfb-run renders through Mesa llvmpipe
# (software GL: VirtualGL is not installed, /usr/lib64/dri is empty). llvmpipe fans out
# over every core it can see -- measured 2026-08-15, ONE default worker takes ~34 cores
# (3400% CPU, 133 threads) and gets very little for them:
#
#   config                 res   cores/worker   s/episode   core-seconds/episode
#   default xvfb-run       256       ~34            107            ~3,640
#   default xvfb-run       512       ~34            149            ~5,070
#   LP_NUM_THREADS=4       512       2.7            211              ~570   <-- 8.9x cheaper
#
# So the v2 run's 12 default workers demanded ~408 cores on a 256-core box and spent most
# of their time fighting each other. Thin workers are 1.4x slower per episode and ~9x
# cheaper in core-seconds; that trade is why NPAR can be 48 instead of 12.
#
# 512 vs 256 costs only 1.40x wall time, not 4x -- the bottleneck is motion planning and
# physics stepping, not rasterisation. Disk is ~2.6x, not 4x (compression scales sublinearly):
# measured per-component 2.61x rgb / 2.50x depth / 2.71x mask.
#
# Env overrides: OUT, RES, NPAR, TRAIN_SEEDS, HELDOUT_SEEDS, HELDOUT_VARS, SHARD, TASKS
# Resume: just re-run. gen_dataset.py skips any episode that already has meta.json.
# ============================================================================
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate ttd_eval
source /workspace/ttdu/ttd/scripts/env_eval.rc

OUT=${OUT:-/workspace/ttdu/ttd/data/rlbench_selfgen_512_aug}
RES=${RES:-512}
AUG=${AUG:-colosseum}
NPAR=${NPAR:-48}
TRAIN_SEEDS=${TRAIN_SEEDS:-50}        # per task, variation0 (the TRAIN split)
HELDOUT_SEEDS=${HELDOUT_SEEDS:-10}    # per task per held-out variation (the TEST split)
HELDOUT_VARS=${HELDOUT_VARS:-1-2}     # inclusive range; gen_dataset.py intersects with
                                      # variation_count(), so tasks with fewer just skip
SHARD=${SHARD:-5}                     # seeds per job -- smaller = better tail balance,
                                      # but each job pays a ~50s CoppeliaSim launch
LOGD=/workspace/ttdu/ActionImages-Cogen/logs/gen_512_aug
TASKS=${TASKS:-"open_drawer,slide_block_to_target,sweep_to_dustpan,meat_off_grill,turn_tap,put_item_in_drawer,close_jar,reach_and_drag,stack_blocks,light_bulb_in,put_money_in_safe,stack_wine,put_groceries_in_cupboard,place_shape_in_shape_sorter,push_buttons,insert_onto_square_peg"}

# Thin worker. 4 is measured-good, not tuned -- it only reached 270% CPU, so the optimum
# is somewhere in 2..6 and nobody has swept it.
export LP_NUM_THREADS=${LP_NUM_THREADS:-4}
export OMP_NUM_THREADS=1
export PYTHONPATH=/workspace/ttdu/robot-colosseum${PYTHONPATH:+:$PYTHONPATH}

mkdir -p "$LOGD" "$OUT"

# --- refuse to write into the 256 tree -------------------------------------------------
# The whole point of this run is that v2 stays untouched and the two currently-running
# 256 arms keep reading it. A mistyped OUT= would quietly interleave 512 episodes into it.
case "$(readlink -f "$OUT")" in
  *rlbench_selfgen_v2*|*rlbench_selfgen_512) echo "!! OUT resolves into an existing no-aug tree -- refusing."; exit 2 ;;
esac

JOBS="$LOGD/jobs.txt"
: > "$JOBS"
IFS=',' read -ra TARR <<< "$TASKS"
for t in "${TARR[@]}"; do
  for ((a=0; a<TRAIN_SEEDS; a+=SHARD)); do
    b=$((a + SHARD - 1)); [ $b -ge $TRAIN_SEEDS ] && b=$((TRAIN_SEEDS - 1))
    echo "$t:0:$a-$b" >> "$JOBS"
  done
  for ((v=${HELDOUT_VARS%-*}; v<=${HELDOUT_VARS#*-}; v++)); do
    echo "$t:$v:0-$((HELDOUT_SEEDS - 1))" >> "$JOBS"
  done
done

NJOBS=$(wc -l < "$JOBS")
EPS=$(( ${#TARR[@]} * (TRAIN_SEEDS + (${HELDOUT_VARS#*-} - ${HELDOUT_VARS%-*} + 1) * HELDOUT_SEEDS) ))
AVAIL_G=$(df -BG --output=avail "$(dirname "$OUT")" | tail -1 | tr -dc '0-9')
# ~29 MB/episode at 512 (measured: 27 MB for a 157-step episode, 4 views).
PROJ_G=$(( EPS * 29 / 1000 + 1 ))
echo "out=$OUT res=$RES npar=$NPAR"
echo "jobs=$NJOBS  episodes<=$EPS (held-out shrinks where variation_count() < ${HELDOUT_VARS#*-}+1)"
echo "disk: ~${PROJ_G}G projected, ${AVAIL_G}G available"
echo "eta:  ~$(( EPS * 211 / NPAR / 60 )) min at 211 s/episode with $NPAR workers"
if [ "${AVAIL_G:-0}" -lt "$(( PROJ_G + 20 ))" ]; then
  echo "!! only ${AVAIL_G}G free for a ~${PROJ_G}G run -- not starting."
  exit 3
fi

run_one() {
  local spec="$1"
  local t="${spec%%:*}"; local rest="${spec#*:}"
  local v="${rest%%:*}"; local s="${rest#*:}"
  local log="$LOGD/${t}_v${v}_${s}.log"
  # Stagger: 48 workers calling `xvfb-run -a` at once race for a free display number.
  sleep $(( RANDOM % 25 ))
  local tries=0
  while [ $tries -lt 5 ]; do
    xvfb-run -a python /workspace/ttdu/ActionImages-Cogen/scripts/gen_dataset.py \
      --tasks "$t" --variations "$v" --seeds "$s" --out "$OUT" --res "$RES" --aug "$AUG" >> "$log" 2>&1
    local rc=$?
    [ $rc -eq 0 ] && return 0
    tries=$(( tries + 1 ))
    echo "[launcher] rc=$rc restart $tries" >> "$log"
    sleep 5
  done
  echo "[launcher] GIVING UP on $spec" >> "$log"
  return 1
}
export -f run_one
export OUT RES LOGD AUG

# shuf so the long variation0 shards and the short held-out jobs interleave rather than
# leaving all the long ones for the tail.
shuf "$JOBS" | xargs -P "$NPAR" -I{} bash -c 'run_one "$@"' _ {}

echo "GEN_512_DONE"
echo -n "episodes in $OUT: "; find "$OUT" -name meta.json | wc -l
