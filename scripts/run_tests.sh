#!/bin/bash
# CPU-only test suite for the task-template fork. No GPU, no model load, a few minutes.
#
#   bash scripts/run_tests.sh
#
# Order matters: the assembly guard first (it is the premise every other test rests on --
# that template-driven packing reproduces the upstream training core decision-for-decision),
# then codecs, then the dataset built on them.
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate ttd_train

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

TESTS=(
  tests/test_forward_unchanged.py     # assembly reproduces upstream, per decision path
  tests/test_fusion_axis.py           # F axis: 10-segment canvas, modality dropout, train/eval parity
  tests/test_droid_gripper.py         # DROID openness = 1-gripper, not a constant
  tests/test_checkpoint_pruning.py    # collective symmetry (2 gloo ranks) + what survives
  tests/test_decode_6dof.py           # paper Sec.3.2 decoder: pose roundtrip + shipped-defect pins
  tests/test_policy_conditioning.py   # rollout policy tensors match the training conventions
  tests/test_aggregate_closedloop.py  # exact McNemar + CI; refuses unpairable / underpowered grids
  tests/test_depth_codec.py           # depth GT <-> RGB roundtrip
  tests/test_seg_codec.py             # seg GT <-> RGB roundtrip
  tests/test_normal_codec.py          # surface normal from depth: geometry + uint8 roundtrip
  tests/test_scene_seg_codec.py       # dense scene-role palette, decode, per-role IoU
  tests/test_scene_seg_dataset.py     # scene_roles on REAL episodes; pins zero `unknown` pixels
  tests/test_prompt_tags.py           # every template tag survives scrub + tokenizer
  tests/test_selfgen_dataset.py       # co-supervision, both views encoded, template mixes
  tests/test_frame_interval.py        # temporal stride reaches all five streams; interval-1 parity
  tests/test_selfgen_alignment.py     # camera tensors describe the same views/frames
  tests/test_depth_roundtrip_e2e.py   # the pixels really are the depth of those frames
  tests/test_new_episodes.py          # newly generated episodes specifically (no-op if none)
)

failed=()
for t in "${TESTS[@]}"; do
  echo "=============================================================== $t"
  # PIPESTATUS[0] is python's status; the grep only filters noisy warnings.
  python -u "$t" 2>&1 | grep -vE "FutureWarning|import pynvml"
  [ "${PIPESTATUS[0]}" -eq 0 ] || failed+=("$t")
done

echo "==============================================================="
if [ "${#failed[@]}" -eq 0 ]; then
  echo "ALL_TESTS_PASSED (${#TESTS[@]} files)"
else
  printf '!! FAILED: %s\n' "${failed[@]}"
  exit 1
fi
