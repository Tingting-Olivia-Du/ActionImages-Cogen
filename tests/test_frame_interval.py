"""`frame_interval` must set the TEMPORAL SPAN of a training window, on every stream at once.

Why this exists: `num_frames` alone does not determine the horizon. The official RLBench
release stores video already 4x-downsampled and realigns actions with `actions[::4]`
(dataset/rlbench.py:105), while rlbench_selfgen writes every stream at native 20 Hz 1:1 and
deliberately does NOT downsample (dataset/rlbench_selfgen.py:267-275). Identical
`num_frames=41` therefore means 8.0 s of motion on official data and 2.0 s on selfgen.
`frame_interval` is the knob that closes that gap, and it is only correct if it reaches
video, actions, camera params, depth and mask through ONE shared `frame_indices`.

Three groups:
  1. index arithmetic, including upstream parity at frame_interval=1 (the fix in
     helpers/io.py must be a no-op for every existing run);
  2. the short-video branch that used to ignore frame_interval entirely;
  3. end to end on real selfgen episodes: all five streams stride together.

Run: python /workspace/ttdu/ActionImages-Cogen/tests/test_frame_interval.py
"""
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.helpers.io import load_video_frames

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 512_aug, not the bare "rlbench_selfgen" symlink: that one points at the deleted 256 v2 tree
# and DANGLES, so this check has been skipping itself silently. SELFGEN_TEST_DATA overrides.
SELFGEN = os.environ.get("SELFGEN_TEST_DATA", os.path.join(REPO, "data", "rlbench_selfgen_512_aug"))
RES, NUM_FRAMES = 512, 41


def upstream_indices(total_frames, max_frames, frame_interval, start_idx):
    """The pre-fork index arithmetic, kept verbatim as the parity reference."""
    if total_frames <= max_frames:
        return list(range(total_frames)) + [total_frames - 1] * (max_frames - total_frames)
    return [min(start_idx + i * frame_interval, total_frames - 1) for i in range(max_frames)]


def forked_indices(total_frames, max_frames, frame_interval, start_idx):
    """Current arithmetic in helpers/io.py, mirrored so both branches are testable without I/O."""
    if total_frames <= max_frames:
        return [min(i * frame_interval, total_frames - 1) for i in range(max_frames)]
    return [min(start_idx + i * frame_interval, total_frames - 1) for i in range(max_frames)]


# ------------------------------------------------------- 1. parity at frame_interval=1
for total in list(range(1, 90)) + [145, 164, 200, 777]:
    for start in (0, 1, 7):
        legal_start = min(start, max(0, total - NUM_FRAMES))
        a = upstream_indices(total, NUM_FRAMES, 1, legal_start)
        b = forked_indices(total, NUM_FRAMES, 1, legal_start)
        assert a == b, f"frame_interval=1 parity broken at total={total} start={legal_start}"
print("UPSTREAM_PARITY_AT_INTERVAL_1_OK  (89 lengths x 3 starts, indices identical)")

# --------------------------------- 2. the short-video branch now respects frame_interval
short = forked_indices(35, NUM_FRAMES, 3, 0)
assert short[:5] == [0, 3, 6, 9, 12], short[:5]
assert max(short) == 34 and short[-1] == 34
assert upstream_indices(35, NUM_FRAMES, 3, 0)[:5] == [0, 1, 2, 3, 4], "reference must be stride-1"
print(f"SHORT_VIDEO_BRANCH_STRIDES_OK  total=35 interval=3 -> {short[:5]}...{short[-3:]} "
      f"(upstream gave stride 1 here)")

# The two thresholds that govern padding and augmentation:
#   no padding      iff  total >= 1 + (N-1)*F        (the window's span)
#   >1 legal start  iff  total >  N*F                (what load_video_frames' randint allows)
def n_windows(total, max_frames, interval):
    return max(0, total - max_frames * interval) + 1


for interval in (1, 2, 3, 4):
    span = 1 + (NUM_FRAMES - 1) * interval
    assert len(set(forked_indices(span, NUM_FRAMES, interval, 0))) == NUM_FRAMES, (
        f"interval={interval}: {span} frames is exactly the span, must not pad")
    # One frame short of the span: the tail must get clamped onto the final frame, i.e. the
    # window can no longer reach its nominal reach of (N-1)*interval. (Clamping is not the
    # same as duplicating -- at interval 2 exactly one index clamps and stays distinct; the
    # duplicates that show up in the padding statistics need several indices to clamp.)
    shorter = forked_indices(span - 1, NUM_FRAMES, interval, 0)
    assert shorter[-1] == span - 2, (interval, shorter[-3:])
    assert shorter[-1] < (NUM_FRAMES - 1) * interval, (
        f"interval={interval}: {span-1} frames is one short, the window must be truncated")
    need = NUM_FRAMES * interval
    assert n_windows(need, NUM_FRAMES, interval) == 1, interval
    assert n_windows(need + 1, NUM_FRAMES, interval) == 2, interval
print("SPAN_AND_WINDOW_THRESHOLDS_OK  no padding iff total >= 1+(N-1)*F; "
      f"single window iff total <= N*F (= {NUM_FRAMES*3} at interval 3)")

# validation
for bad in (0, -1, 1.5, "3"):
    try:
        load_video_frames("/nonexistent.mp4", max_frames=NUM_FRAMES, frame_interval=bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"frame_interval={bad!r} should have been rejected")
print("FRAME_INTERVAL_VALIDATION_OK  (0, -1, 1.5, '3' all rejected)")

# ------------------------------------------------- 3. end to end: all five streams stride
if not os.path.isdir(SELFGEN):
    print("SELFGEN_ABSENT_SKIPPED")
else:
    from training.dataset import RLBenchSelfgenDataset

    INTERVAL = 3
    random.seed(0)
    ds = RLBenchSelfgenDataset(base_path=SELFGEN, num_frames=NUM_FRAMES, frame_interval=INTERVAL,
                               height=RES, width=RES, template_mix="video+depth@1.0",
                               strict_getitem=True)
    checked = 0
    for i in range(6):
        s = ds[i]
        fi = list(s["frame_indices"])
        ep_dir = s["path"]
        raw = np.load(os.path.join(ep_dir, "actions.npy"))
        total = len(raw)  # every selfgen stream is this long: 20 Hz, 1:1, no downsampling
        assert len(fi) == NUM_FRAMES, len(fi)

        # The exact invariant, rather than a stride pattern: a diff of less than INTERVAL is
        # legal exactly once, at the point the tail clamps onto the final frame.
        want_idx = [min(fi[0] + k * INTERVAL, total - 1) for k in range(NUM_FRAMES)]
        assert fi == want_idx, f"{os.path.basename(ep_dir)}: {fi[:6]}... != {want_idx[:6]}..."
        assert fi[1] - fi[0] == INTERVAL, f"window did not stride at all: {fi[:5]}"

        # ...and the shared frame_indices really did reach every stream.
        assert s["action_7d"].shape[0] == NUM_FRAMES, s["action_7d"].shape
        assert s["extrinsics"].shape[0] == 2 * NUM_FRAMES, s["extrinsics"].shape
        for name, stream in s["streams"].items():
            assert stream.shape[1] == 2 * NUM_FRAMES, (name, stream.shape)

        # actions must be the strided ones, re-read independently from disk
        want_pos = raw[fi][:, :3]
        got_pos = s["action_7d"].numpy()[:, :3]
        assert np.allclose(got_pos, want_pos, atol=1e-5), (
            f"{os.path.basename(ep_dir)}: max |diff| = {np.abs(got_pos - want_pos).max():.3e}")
        checked += 1
    print(f"SELFGEN_STREAMS_STRIDE_TOGETHER_OK  interval={INTERVAL}, {checked} samples verified "
          f"against actions.npy on disk")
    print(f"  last: {os.path.basename(ep_dir)} len={total} -> window {fi[0]}..{fi[-1]} "
          f"({fi[-1] - fi[0]} native steps = {(fi[-1] - fi[0])/20:.2f} s, "
          f"{NUM_FRAMES - len(set(fi))} repeated tail frames)")

    # Negative control: at interval 1 the same episodes must NOT stride.
    random.seed(0)
    ds1 = RLBenchSelfgenDataset(base_path=SELFGEN, num_frames=NUM_FRAMES, frame_interval=1,
                                height=RES, width=RES, template_mix="video+depth@1.0",
                                strict_getitem=True)
    fi1 = list(ds1[0]["frame_indices"])
    assert fi1[1] - fi1[0] == 1, fi1[:5]
    assert fi1[-1] - fi1[0] == NUM_FRAMES - 1, fi1[-1] - fi1[0]
    print(f"INTERVAL_1_STILL_CONTIGUOUS_OK  window {fi1[0]}..{fi1[-1]} "
          f"({(fi1[-1] - fi1[0])/20:.2f} s)")

print("ALL_FRAME_INTERVAL_TESTS_PASSED")
