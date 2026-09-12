#!/bin/bash
# Simulator-backed self-checks for the closed-loop rollout harness.
#
#   bash scripts/run_rollout_tests.sh                    # default 4 tasks x 2 trials, ~8 min
#   bash scripts/run_rollout_tests.sh --tasks open_drawer --trials 1
#
# Separate from scripts/run_tests.sh on purpose: that suite is pure CPU with no simulator and
# runs in a couple of minutes, while this one launches CoppeliaSim and generates live demos
# (15-60 s each). Keeping them apart means the fast suite stays fast.
#
# What it proves, without loading the model or touching a GPU: the camera rig matches the one
# the training data was rendered with, scenes are reproducible from a seed, and the recorded
# ground-truth actions are executable to task success. Until all three hold, a low closed-loop
# success rate from a checkpoint is the harness's number rather than the model's.
#
# Environment: ttd_rollout = a clone of ttd_train with PyRep/RLBench added, so torch and the
# simulator live in one interpreter and the rollout loop is a single process. ttd_train itself
# is deliberately left untouched (it runs the training jobs).
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate ttd_rollout

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

# CoppeliaSim needs these; env_eval.rc is the single source of truth for the paths.
source /workspace/ttdu/ttd/scripts/env_eval.rc

if [ ! -d "${COPPELIASIM_ROOT:-}" ]; then
  echo "!! COPPELIASIM_ROOT=${COPPELIASIM_ROOT:-<unset>} is not a directory."
  echo "   Check /workspace/ttdu/ttd/scripts/env_eval.rc."
  exit 3
fi

# Headless rendering. CoppeliaSim needs an X display even with headless=True.
exec xvfb-run -a python -u tests/test_rollout_env.py "$@"
