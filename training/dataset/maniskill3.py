"""ManiSkill3 loader: official demonstrations replayed into the rlbench_selfgen layout.

The tree under data/maniskill3 is written by scripts/maniskill3_gen.py, which replays
OFFICIAL ManiSkill demonstration trajectories (haosulab/ManiSkill_Demonstrations on HF,
deterministic `set_state_dict` playback -- see that script's docstring for the exact
on-disk contract) through 4 randomly-placed 512x512 cameras per episode. Every file the
selfgen loader touches is written in the SAME format and the SAME conventions:

  - actions.npy         [T,8] float64, [x,y,z, qx,qy,qz,qw, openness], absolute per-frame
                        TCP pose from `agent.tcp.pose` (NOT the controller input), quat
                        scalar-last to match get_7d_action's scipy R.from_quat call.
  - camera_params.json  4x4 camera-to-world extrinsics in the RLBench camera frame
                        (x left, y up, z forward) with NEGATIVE fx/fy -- converted from
                        ManiSkill's OpenCV params and verified per episode by an
                        unproject/re-project self-check at generation time.
  - depth.npz           float16 METERS (ManiSkill renders int16 millimeters; /1000 at
                        generation). Range-checked against depth_codec (0.05, 10) there.
  - mask.npz            uint16 per-scene segmentation ids; scene_segments.json +
                        seg_targets.json follow the rlbench-scene-role-v1 protocol, with
                        the manipulated object as base_role=distractor promoted to
                        `target` by _scene_role_lut exactly like the RLBench trees.
  - meta.json           "desc" carries per-task instruction paraphrases.

Because the byte-level contract is identical, this class deliberately adds NOTHING on
top of RLBenchSelfgenDataset: the window/view-selection logic, the perception encoders,
the scene_roles startup gate and the per-template self-test all apply unchanged. It
exists as a separate class so error messages, assert_menu_supported and future
ManiSkill-specific divergence (e.g. a task filter for held-out-task splits) have a
named home, not because any behaviour differs today.

Split convention of the tree (mirrors --variations semantics):
  variation0 = training seeds; variation1 = held-out seeds of the SAME tasks (unseen
  object placements). Held-out TASKS are simply never generated into the training tree.
"""
from training.dataset.rlbench_selfgen import RLBenchSelfgenDataset


class ManiSkill3Dataset(RLBenchSelfgenDataset):
    AVAILABLE_MODALITIES = ("video", "depth", "segmentation", "normal", "action")
