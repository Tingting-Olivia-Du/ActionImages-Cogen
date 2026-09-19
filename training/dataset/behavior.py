"""BEHAVIOR-1K loader: raw challenge demos replayed into the rlbench_selfgen layout.

The tree under data/behavior is written by behavior_demos/render_selfgen.py, which
replays raw 2026-challenge demos (full serialized OmniGibson sim states) through 4
randomly-placed 512x512 external cameras. One selfgen "episode" is one MANIPULATION
CHUNK of a teleop demo (skill-annotation segments with navigation removed, split to
<=240 rendered frames), NOT a whole demo -- the R1 robot drives between manipulation
sites and those stretches carry no arm supervision. Every file follows the SAME
byte-level contract as the ManiSkill3 tree (see training/dataset/maniskill3.py):

  - actions.npy         [T,8] float64 [x,y,z, qx,qy,qz,qw, openness]: ACTIVE-arm EE
                        world pose read live from the replayed state (R1 is dual-arm;
                        the active arm per chunk comes from skill annotations, else
                        gripper-command activity; openness = normalized finger qpos).
  - camera_params.json  4x4 cam2world in the RLBench camera frame, NEGATIVE fx/fy,
                        converted from the USD/GL camera pose and verified per episode
                        by the same unproject/re-project self-check (<0.51 px).
  - depth.npz           float16 METERS from depth_linear; codec range (0.05, 10) m
                        enforced at generation (cameras resampled, else chunk dropped).
  - mask.npz            uint16 CANONICAL ids: Isaac instance ids are not stable across
                        frames/sensors, so every frame is remapped prim_path->canonical
                        id at generation. scene_segments.json roles: robot links ->
                        robot_arm/gripper (finger link names), BDDL-scope movables ->
                        distractor (promoted to target via seg_targets), scope fixed ->
                        goal, walls/floors/ceilings -> background, other scene objects
                        -> fixture (fixed) / distractor (movable).
  - meta.json           "desc" = task sentence + per-chunk skill sentences from the
                        official skill annotations ("open door the top cabinet", ...).

Split convention: variation0 = train demos (first 40 per task), variation1 = held-out
demos (last 4) of the SAME tasks; held-out TASKS and the Rs_int held-out SCENE live in
a separate data/behavior_heldout tree (all variation0). `--variations` semantics apply
unchanged.
"""
from training.dataset.rlbench_selfgen import RLBenchSelfgenDataset


class BehaviorDataset(RLBenchSelfgenDataset):
    AVAILABLE_MODALITIES = ("video", "depth", "segmentation", "normal", "action")
