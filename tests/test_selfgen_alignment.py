"""The camera tensors must describe the same views and frames as the pixels.

This is the regression test for the misalignment class that ttd called P0-1: the perception
target used to be built from a SECOND, independent random draw of (window, view pair) while
`camera`/`extrinsics`/`intrinsics` kept the values from the first draw, so the model was
conditioned on a camera trajectory belonging to different frames than the pixels it was asked
to generate -- on every perception sample, silently, for the whole run.

ttd prevented it by monkeypatching glob/random/load_video_frames to observe what the base
class did (`_CaptureViewsAndFrames`). This fork instead has the base class RETURN the window
and view dirs it used, so the invariant is structural. This test checks the invariant end to
end: recompute the camera tensors from camera_params.json on disk, for exactly the views and
frames the sample says it used, and compare against what the sample carries.

Checked for both modalities and for BOTH views -- under `<depth><action>` both visual segments
are perception, so an error in either one is a real error.

Run: python /workspace/ttdu/ActionImages-Cogen/tests/test_selfgen_alignment.py
"""
import json
import os
import random
import sys

import numpy as np
import torch
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset import RLBenchSelfgenDataset
from training.utils import convert_intrinsics_after_center_crop_resize, get_relative_pose_batch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELFGEN = os.path.join(REPO, "data", "rlbench_selfgen")
RES, NUM_FRAMES, N_SAMPLES = 256, 41, 5


def camera_params_from_disk(view_dir, frame_indices):
    """Independently re-read the per-frame camera params, as get_camera_params would."""
    with open(os.path.join(view_dir, "camera_params.json")) as f:
        data = json.load(f)
    extr, intr = [], []
    for idx in frame_indices:
        key = str(idx) if str(idx) in data else max(data.keys(), key=lambda k: int(k))
        extr.append(np.array(data[key]["extrinsics"]))
        intr.append(np.array(data[key]["intrinsics"]))
    return np.stack(extr), np.stack(intr)


def check(sample, res):
    frames = sample["frame_indices"]
    cond_dir = sample["view_dirs"][sample["view_indices"][0]]
    tgt_dir = sample["view_dirs"][sample["view_indices"][1]]

    extr_cond, intr_cond = camera_params_from_disk(cond_dir, frames)
    extr_tgt, intr_tgt = camera_params_from_disk(tgt_dir, frames)

    # extrinsics: plain concat of [cond | target], absolute camera-to-world
    want_extr = torch.from_numpy(np.concatenate([extr_cond, extr_tgt], axis=0))
    got_extr = sample["extrinsics"]
    assert torch.allclose(got_extr.double(), want_extr.double(), atol=1e-6), (
        f"extrinsics mismatch for {os.path.basename(cond_dir)}/{os.path.basename(tgt_dir)} "
        f"@ frames[{frames[0]}..{frames[-1]}] -- max |diff| = "
        f"{(got_extr.double() - want_extr.double()).abs().max():.3e}"
    )

    # camera: relative poses w.r.t. the cond view's FIRST frame, flattened to (2T, 12)
    base = extr_cond[0]
    rel_cond = rearrange(torch.from_numpy(get_relative_pose_batch(base, extr_cond)), "t c d -> t (c d)")
    rel_tgt = rearrange(torch.from_numpy(get_relative_pose_batch(base, extr_tgt)), "t c d -> t (c d)")
    want_cam = torch.cat([rel_cond, rel_tgt], dim=0).to(torch.float32)
    assert torch.allclose(sample["camera"], want_cam, atol=1e-5), (
        f"camera (relative pose) mismatch -- max |diff| = "
        f"{(sample['camera'] - want_cam).abs().max():.3e}"
    )

    # intrinsics: rescaled for the crop+resize the video pipeline applied. The conversion is
    # per-view and needs that view's NATIVE render resolution, which we take from the depth
    # volume on disk rather than from the dataset object -- the whole point is to recompute
    # independently of the code under test.
    raw_cond = np.load(os.path.join(cond_dir, "depth.npz"))["depth"].shape[-2:]
    raw_tgt = np.load(os.path.join(tgt_dir, "depth.npz"))["depth"].shape[-2:]
    conv = convert_intrinsics_after_center_crop_resize(
        [intr_cond, intr_tgt],
        [tuple(raw_cond), tuple(raw_tgt)],
        [(res, res), (res, res)],
    )
    want_intr = np.concatenate(conv, axis=0)
    assert np.allclose(sample["intrinsics"].numpy(), want_intr, atol=1e-4), (
        f"intrinsics mismatch -- max |diff| = "
        f"{np.abs(sample['intrinsics'].numpy() - want_intr).max():.3e}"
    )
    return os.path.basename(cond_dir), os.path.basename(tgt_dir), frames[0], frames[-1]


def main():
    for mix in ("video+action@1.0", "video+depth@1.0"):
        random.seed(1)
        ds = RLBenchSelfgenDataset(base_path=SELFGEN, num_frames=NUM_FRAMES, frame_interval=1,
                                   height=RES, width=RES, template_mix=mix,
                                   strict_getitem=True)
        for i in range(N_SAMPLES):
            s = ds[i]
            c, t, f0, f1 = check(s, RES)
            assert len(s["frame_indices"]) == NUM_FRAMES
            assert s["extrinsics"].shape[0] == 2 * NUM_FRAMES
        print(f"[{mix}] ALIGNMENT_OK {N_SAMPLES} samples "
              f"(last: views={c}/{t} frames={f0}..{f1})")

    # The test must be able to fail: point it at the wrong view and confirm it complains.
    random.seed(1)
    ds = RLBenchSelfgenDataset(base_path=SELFGEN, num_frames=NUM_FRAMES, frame_interval=1,
                               height=RES, width=RES, template_mix="video+depth@1.0",
                               strict_getitem=True)
    s = ds[0]
    s["view_indices"] = [s["view_indices"][1], s["view_indices"][0]]  # swap cond/target
    try:
        check(s, RES)
    except AssertionError:
        print("NEGATIVE_CONTROL_OK swapped views are detected")
    else:
        raise AssertionError("negative control failed: swapping cond/target went unnoticed, "
                             "so this test cannot detect the P0-1 misalignment class")
    print("ALL_SELFGEN_ALIGNMENT_TESTS_PASSED")


if __name__ == "__main__":
    main()
