"""End-to-end: are the pixels in a depth sample really the depth of the frames the camera
tensors describe?

This is the single highest-value test of the depth arm. It takes `sample["streams"]["depth"]`
as the collator would hand it to the model, decodes BOTH halves back to metres through
decode_depth, and compares against depth.npz read independently from disk for the view and
frame window the sample says it used.

One assertion covers: wrong view, wrong frame window, wrong view ORDER (cond/target swapped),
wrong normalization, wrong resize interpolation, wrong channel order, and the
half-encoded-half-RGB failure that "only the target view got encoded" would produce. All of
those are silent -- they yield plausible-looking pixels and a training run that converges to
the wrong thing.

Run: python /workspace/ttdu/ActionImages-Cogen/tests/test_depth_roundtrip_e2e.py
"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset import RLBenchSelfgenDataset
from training.percep.depth_codec import MAX_VALID, MIN_VALID, decode_depth

SELFGEN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "rlbench_selfgen")
N_SAMPLES = 6
RES = 256
NUM_FRAMES = 41
# The codec itself round-trips at ~0.07% AbsRel on this data (test_depth_codec.py), so 2% is
# a generous ceiling that still fails hard on any wrong-view / wrong-window pairing.
ABSREL_MAX = 0.02


def to_uint8(video_half):
    """[C,T,H,W] in [-1,1] -> [T,H,W,3] uint8, inverting the dataset's /127.5 - 1."""
    arr = video_half.permute(1, 2, 3, 0).numpy()
    return np.clip(np.round((arr + 1.0) * 127.5), 0, 255).astype(np.uint8)


def main():
    random.seed(0)
    ds = RLBenchSelfgenDataset(base_path=SELFGEN, num_frames=NUM_FRAMES, frame_interval=1,
                               height=RES, width=RES, template_mix="video+depth@1.0",
                               strict_getitem=True)
    T = NUM_FRAMES
    worst = 0.0
    for i in range(N_SAMPLES):
        s = ds[i]
        assert "depth" in s["streams"], sorted(s["streams"])
        depth_px = s["streams"]["depth"]
        frames = s["frame_indices"]
        halves = [
            ("cond", slice(0, T), s["view_dirs"][s["view_indices"][0]]),
            ("target", slice(T, 2 * T), s["view_dirs"][s["view_indices"][1]]),
        ]
        for name, sl, view_dir in halves:
            dec = decode_depth(to_uint8(depth_px[:, sl]))            # [T,H,W] metres, NaN=invalid
            gt = np.load(os.path.join(view_dir, "depth.npz"))["depth"][frames].astype(np.float32)
            assert dec.shape == gt.shape, f"{name}: decoded {dec.shape} vs gt {gt.shape}"

            m = np.isfinite(dec) & (gt > MIN_VALID) & (gt < MAX_VALID)
            assert m.mean() > 0.5, f"{name}: only {m.mean():.1%} of pixels decoded valid"
            absrel = float(np.mean(np.abs(dec[m] - gt[m]) / gt[m]))
            worst = max(worst, absrel)
            assert absrel < ABSREL_MAX, (
                f"sample {i} {name} half (view={os.path.basename(view_dir)}): AbsRel="
                f"{absrel:.4%} >= {ABSREL_MAX:.0%}. The pixels are not the depth of the frames "
                f"the camera tensors describe -- suspect view/window/order mismatch."
            )
        print(f"sample {i}: views=({os.path.basename(halves[0][2])},"
              f"{os.path.basename(halves[1][2])}) frames[0,-1]=({frames[0]},{frames[-1]}) OK")

    print(f"DEPTH_E2E_ROUNDTRIP_OK worst AbsRel={worst:.4%} over {N_SAMPLES} samples x 2 views")

    # Negative control: the test must actually be able to fail. Swapping the two halves should
    # blow past the threshold whenever the two views genuinely differ -- if it does not, the
    # comparison above is not discriminating and the OK above means nothing.
    random.seed(0)
    s = ds[0]
    a = decode_depth(to_uint8(s["streams"]["depth"][:, 0:T]))
    gt_other = np.load(os.path.join(s["view_dirs"][s["view_indices"][1]],
                                    "depth.npz"))["depth"][s["frame_indices"]].astype(np.float32)
    m = np.isfinite(a) & (gt_other > MIN_VALID) & (gt_other < MAX_VALID)
    mismatched = float(np.mean(np.abs(a[m] - gt_other[m]) / gt_other[m]))
    assert mismatched > ABSREL_MAX, (
        f"negative control failed: cond pixels scored {mismatched:.4%} against the TARGET "
        f"view's depth, i.e. the test cannot distinguish the two views on this episode"
    )
    print(f"NEGATIVE_CONTROL_OK cross-view AbsRel={mismatched:.2%} (must exceed {ABSREL_MAX:.0%})")
    print("ALL_DEPTH_E2E_TESTS_PASSED")


if __name__ == "__main__":
    main()
