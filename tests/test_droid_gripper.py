"""DROID's 7th action channel must carry the real gripper signal, with the right polarity.

Regression guard for the bug where `_load_actions_npz` appended `np.ones(...)`: a constant
7th channel is invisible to an all-zeros check and to the loss, so it silently taught
"gripper never moves" on ~25% of the steps of a three-dataset run.

Also pins the POLARITY. DROID `gripper_position` is 0 = open, 1 = closed; Action-Images'
channel is OPENNESS (1 = open), the same convention RLBench's actions[:, 7] uses. Appending
the raw value would train the channel exactly backwards.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset.droid import DROIDMVDataset

N_EPISODES = 24


def main():
    ds = DROIDMVDataset(base_path="./data/droid", num_frames=41, frame_interval=4,
                        height=512, width=512)
    assert len(ds) > 0, "no DROID episodes; check data/droid symlink"

    constant_windows = 0
    for i in range(0, min(len(ds), N_EPISODES * 17), 17):
        s = ds[i]
        a7 = s["action_7d"].numpy()
        assert a7.dtype == np.float64, (
            f"action_7d must stay float64 -- it is einsum'd against float64 extrinsics in "
            f"project_point_3d_to_2d_torch_batch; got {a7.dtype}"
        )
        openness = a7[:, 6]
        assert np.isfinite(openness).all(), "non-finite openness"
        assert openness.min() >= 0.0 and openness.max() <= 1.0, (
            f"openness out of [0,1]: {openness.min()}..{openness.max()}"
        )
        # exact agreement with the file, on the frames actually sampled
        g = np.load(os.path.join(s["path"], "action.npz"))["gripper"]
        expect = 1.0 - np.clip(g[s["frame_indices"]], 0.0, 1.0)
        assert np.allclose(openness, expect, atol=1e-6), (
            f"{s['path']}: openness != 1 - gripper"
        )
        if np.allclose(openness, openness[0]):
            constant_windows += 1

    # Polarity: across whole episodes, DROID must start OPEN (openness ~1). If the sign were
    # flipped every episode would start closed, which no pick-and-place trajectory does.
    firsts = []
    for d in sorted(glob.glob("data/droid/processed/*"))[:300]:
        g = np.load(os.path.join(d, "action.npz"))["gripper"]
        firsts.append(1.0 - float(np.clip(g[0], 0.0, 1.0)))
    mean_first = float(np.mean(firsts))
    assert mean_first > 0.9, (
        f"episodes should start with the gripper OPEN (openness ~1); got mean {mean_first:.3f}. "
        f"The polarity is probably inverted."
    )

    # And the channel must actually vary somewhere, or the fix regressed to a constant.
    assert constant_windows < N_EPISODES, "every sampled window had a constant openness"

    print(f"OK: openness matches 1-gripper on sampled frames; mean first-frame openness "
          f"{mean_first:.3f}; {constant_windows} constant windows (episodes that never grasp)")


if __name__ == "__main__":
    main()
