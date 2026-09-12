"""Cross-scene replay: the LOWER bound that `--gt-replay` cannot provide.

`--gt-replay` answers "is this scene solvable at all", which is an upper bound.  It says
nothing about the opposite failure of interpretation: a task like `push_buttons` may be
solved by any plausible-looking downward sweep through the button region, in which case a
model's success rate is measuring the task's chance level, not the model.

This policy emits the ground-truth demo of a DIFFERENT scene seed of the SAME task.  It
therefore carries correct task-level motion statistics -- it is a real, human-quality
trajectory for this task -- while carrying zero information about where the objects in
*this* scene actually are.  Anything the real policy scores above this is attributable to
scene conditioning; anything at or below it is chance.

It deliberately implements the same duck-typed surface `rollout_one` uses on
`ActionImagePolicy` (`num_frames`, `run_policy`, `last_debug`, `last_frames`,
`last_confidence`) so that the ramp / interpolate / IK / streak path downstream is
byte-identical between the null and the model.  A null that took a different code path
would not be a control.
"""

from typing import List, Optional

import numpy as np


class CrossSceneNullPolicy:
    def __init__(self, donor_actions: np.ndarray, num_frames: int = 41, frame_interval: int = 1):
        self.donor = np.asarray(donor_actions, dtype=np.float64)
        self.num_frames = num_frames
        self.frame_interval = max(int(frame_interval), 1)
        self.cursor = 0
        self.last_debug: dict = {}
        self.last_frames: Optional[List[np.ndarray]] = None
        self.last_confidence: Optional[np.ndarray] = None

    def run_policy(self, rgb_views, extrinsics, intrinsics, prompt, current_pose8=None, **kw):
        """Next `num_frames` donor poses at the policy's stride; the scene inputs are ignored.

        Padding repeats the donor's final pose, which is what a policy that has finished its
        motion would emit anyway, rather than wrapping around into a spurious second attempt.
        """
        n = len(self.donor)
        idx = np.clip(self.cursor + np.arange(self.num_frames) * self.frame_interval, 0, n - 1)
        chunk = self.donor[idx].copy()
        # One replan consumes (num_frames - 1) * frame_interval + 1 native steps once the
        # chunk is interpolated back to the native rate, so the cursor advances by the same.
        self.cursor += (self.num_frames - 1) * self.frame_interval + 1
        self.last_debug = {"null": "cross_scene", "cursor": int(self.cursor), "donor_len": int(n)}
        return chunk
