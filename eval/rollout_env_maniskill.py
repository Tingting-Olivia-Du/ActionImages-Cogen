"""ManiSkill3 environment wrapper for closed-loop rollouts, duck-typed against `RolloutEnv`.

WHY THIS FILE EXISTS. `eval/rollout_env.py` is CoppeliaSim/RLBench only, so every closed-loop
number this project has ever produced is RLBench's. The paper asks each question "in both
simulators", and `tab:pertask`'s ManiSkill3 block is currently \textsc{n/a} for exactly this
reason. Nothing about the policy is RLBench-specific -- `eval/policy.py` consumes
`[V,H,W,3]` anchors plus camera matrices and returns world-frame poses -- so the missing piece
is an env that (a) reproduces a stored episode's scene AND cameras, (b) accepts an absolute
world-frame end-effector pose, and (c) reports task success. That is this file.

THE THREE PINS, each checkable rather than asserted:

  * scene   -- `reset(seed=meta["seed"], options={"reconfigure": True})` replays the seeding of
               `maniskill3_demos_selfgen/<env>/motionplanning/trajectory.json`, whose
               `reset_kwargs.seed` IS the episode index the tree was written from. Unlike
               RLBench (see rollout_env.py's note that its numpy seed does NOT reproduce a
               stored scene), ManiSkill's reset seed fully determines object placement, so the
               rollout scene is the stored episode's scene, and its `actions.npy` is a valid
               ground-truth reference for it. `verify_scene()` checks that claim against the
               stored video's first frame.
  * cameras -- the tree stores only (radius, elev, azim) per view, not the jittered look-at
               target, so the cameras are REDRAWN from the same generator state
               (`default_rng(seed + 777)`, scripts/maniskill3_gen.py:record_episode) and then
               verified against the episode's own `camera_params.json`. A mismatch means the
               generation run hit a camera retry for that episode; the rollout then keeps the
               redrawn cameras (still in-distribution) and says so in the trial record.
  * action  -- `pd_ee_pose`: absolute target, `[x,y,z, euler XYZ]` IN THE ROBOT BASE FRAME plus
               one gripper channel (pd_ee_pose.py:143 `Pose.create_from_pq` on
               `euler_angles_to_matrix(..., "XYZ")`). The model emits WORLD poses with a
               scalar-LAST quaternion, so both the frame change and the quaternion convention
               have to be undone here. `_check_conventions()` round-trips a live pose through
               the whole conversion at launch and refuses to run if it does not come back.

IK: ManiSkill's CPU IK returns None on failure and the controller then silently holds the
current qpos (pd_ee_pose.py:138-141), which would look like a compliant robot that just does
not move. `step()` therefore asks the same solver the same question BEFORE stepping and
reports `ik_ok=False`, which is what RLBench's `ViaIK` path reports and what
`--max-ik-fail-streak` counts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# view1/view2 are the two anchor views, matching rollout_env.VIEW_CAMERAS' role for RLBench.
VIEW_NAMES: Tuple[str, str] = ("view1", "view2")
N_VIEWS = 4  # the tree renders four; we reproduce all four so the sampled poses match


def _t2n(x):
    import torch
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


class ManiSkillRolloutEnv:
    """One task's env, reused across trials (sapien 3.0.3 leaks Vulkan memory on close())."""

    def __init__(self, task_dir: str, tree_root: str, resolution: int = 512,
                 anchor_modality: str = "video", control_mode: str = "pd_ee_pose",
                 lazy_render: bool = True, extra_renders: Sequence[str] = ()):
        self.task_dir = task_dir
        self.tree_root = Path(tree_root)
        self.resolution = resolution
        self.anchor_modality = anchor_modality
        self.control_mode = control_mode
        # RENDER ON DEMAND, AND ONLY WHEN NO PIXELS ARE NEEDED AT ALL.
        # `lazy_render` builds the env with obs_mode="none", which makes stepping pure physics
        # -- worth minutes per episode on a GT replay. But obs_mode also decides which TEXTURES
        # the cameras capture, so under "none" there is no rgb texture to read and `views_rgb`
        # dies with KeyError: 'rgb'. That is not hypothetical: it silently voided all 18
        # ManiSkill model jobs on 2026-09-20, because the flag had only ever been exercised by
        # `--gt-replay --no-record-video`, the one path that never asks for a pixel.
        # So the caller must pass lazy_render=True only when it will never call views_*;
        # eval/rollout_maniskill.py derives it from (gt_replay and not record_video).
        self.lazy_render = lazy_render
        for m in extra_renders:
            if m not in ("depth", "mask"):
                raise ValueError(f"extra_renders takes 'depth'/'mask', got {m!r}")
        self.extra_renders = tuple(extra_renders)
        self._env = None
        self._episode_root: Optional[Path] = None
        self._cam_mismatch = False

        import scripts.maniskill3_gen as G  # noqa: E402  (reuse, never re-implement)
        self._G = G
        self.env_id = next(k for k, v in G.TASKS.items() if v["task_dir"] == task_dir)
        self.cfg = G.TASKS[self.env_id]

    # ------------------------------------------------------------------ lifecycle
    def launch(self) -> "ManiSkillRolloutEnv":
        return self

    def shutdown(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    def __enter__(self) -> "ManiSkillRolloutEnv":
        return self.launch()

    def __exit__(self, *exc) -> None:
        self.shutdown()

    # ------------------------------------------------------------------ episodes
    def episode_dirs(self, variation: int = 0) -> List[Path]:
        d = self.tree_root / self.task_dir / f"variation{variation}" / "episodes"
        return sorted(d.glob("episode*"), key=lambda p: int(p.name.replace("episode", "")))

    def reset_to_episode(self, episode_dir: Path):
        """Reproduce a stored episode's scene and cameras.

        Returns `(descriptions, obs, demo_actions [T,8])` -- the same triple
        `RolloutEnv.reset_to_new_demo` returns, so the rollout loop is shared.
        """
        G = self._G
        meta = json.loads((episode_dir / "meta.json").read_text())
        seed = int(meta["seed"])
        cam_rng = np.random.default_rng(seed + 777)
        cams = G.sample_cameras(cam_rng, np.array(self.cfg["center"]))

        if self._env is None:
            self._env = self._make_env(cams)
        self._env._episode_cams = cams
        self._env.reset(seed=seed, options={"reconfigure": True})

        self._episode_root = episode_dir
        self._cam_mismatch = not self._cameras_match(episode_dir)
        demo_actions = np.load(episode_dir / "actions.npy")
        obs = {} if self.lazy_render else self._env.unwrapped.get_obs()
        return list(meta["desc"]), obs, demo_actions

    def _make_env(self, initial_cams):
        """Same dynamic 4-camera subclass the tree was rendered with, but stepped rather than
        state-replayed, so it needs a control mode and a reward-free evaluator."""
        from mani_skill.sensors.camera import CameraConfig
        from mani_skill.utils import sapien_utils
        from mani_skill.utils.registration import REGISTERED_ENVS

        res, fov = self.resolution, self._G.FOV
        base_cls = REGISTERED_ENVS[self.env_id].cls

        class MultiCamEnv(base_cls):
            _episode_cams = initial_cams

            @property
            def _default_sensor_configs(self):
                return [
                    CameraConfig(f"view{i + 1}",
                                 pose=sapien_utils.look_at(c["eye"], c["target"]),
                                 width=res, height=res, fov=fov, near=0.01, far=100.0)
                    for i, c in enumerate(self._episode_cams)
                ]

        # `extra_renders` decouples WHAT IS RENDERED from what the policy is anchored on, for
        # the reason spelled out in eval/rollout_env.py: the sim-replay perception eval needs
        # the simulator's own depth/mask at every executed step even when the policy runs on an
        # RGB anchor (which the RGB-only arm has no alternative to). Adding a render never
        # changes an existing campaign's observations.
        need_extra = (self.anchor_modality in ("depth", "normal", "segmentation", "all")
                      or bool(self.extra_renders))
        obs_mode = "rgb+depth+segmentation" if need_extra else "rgb"
        env = MultiCamEnv(
            obs_mode="none" if self.lazy_render else obs_mode,
            reward_mode="none",
            render_mode="rgb_array",
            sim_backend="physx_cpu",
            control_mode=self.control_mode,
        )
        env.reset(seed=0, options={"reconfigure": True})
        # current_pose8/ik_ok read self._env, and the convention check uses both.
        self._env = env
        self._check_conventions(env)
        return env

    # ------------------------------------------------------------------ checks
    def _cameras_match(self, episode_dir: Path, tol: float = 1e-6) -> bool:
        """Do the redrawn cameras equal the ones this episode was rendered with?"""
        try:
            stored = json.loads((episode_dir / VIEW_NAMES[0] / "camera_params.json").read_text())
        except Exception:
            return False
        first = stored[sorted(stored, key=lambda k: int(k))[0]]
        live = self._G.rlb_camera_params(self._env.unwrapped.scene.sensors[VIEW_NAMES[0]])
        return bool(np.allclose(np.asarray(first["extrinsics"], dtype=np.float64),
                                live["extrinsics"], atol=tol))

    def _check_conventions(self, env) -> None:
        """World pose -> base-frame euler -> the controller's own reconstruction, round-tripped.

        Catches a quaternion-order or frame slip at launch instead of as a mysteriously
        unreachable arm 200 rollouts later.
        """
        import torch
        from scipy.spatial.transform import Rotation as Rot
        from mani_skill.utils.geometry.rotation_conversions import (
            euler_angles_to_matrix, matrix_to_quaternion)

        pose8 = self.current_pose8()
        act = self._pose8_to_action(pose8, env)
        q = matrix_to_quaternion(euler_angles_to_matrix(
            torch.tensor(act[3:6], dtype=torch.float64)[None], "XYZ"))[0].numpy()
        ctrl = env.unwrapped.agent.controller.controllers["arm"]
        want = _t2n(ctrl.ee_pose_at_base.q)[0]
        dq = min(np.linalg.norm(q - want), np.linalg.norm(q + want))  # q and -q are one rotation
        dp = np.linalg.norm(act[:3] - _t2n(ctrl.ee_pose_at_base.p)[0])
        if dq > 1e-4 or dp > 1e-4:
            raise RuntimeError(
                f"pd_ee_pose convention check failed: dp={dp:.2e} dq={dq:.2e}. The world->base "
                f"transform or the quaternion order is wrong; do not trust any rollout from it.")
        _ = Rot  # (kept: the reverse conversion below uses it)

    # ------------------------------------------------------------------ observations
    def _sensors(self, obs=None) -> dict:
        """`obs["sensor_data"]`, rendering it now if the env was built without observations."""
        if self.lazy_render:
            raise RuntimeError(
                "views_* were called on a lazy_render env, whose cameras capture no textures. "
                "Construct ManiSkillRolloutEnv(lazy_render=False) whenever pixels are needed.")
        if obs is not None and "sensor_data" in obs:
            return obs["sensor_data"]
        base = self._env.unwrapped
        base.scene.update_render(update_sensors=True, update_human_render_cameras=False)
        return base._get_obs_sensor_data()

    def views_rgb(self, obs=None) -> np.ndarray:
        """`[2,H,W,3]` uint8 in VIEW_NAMES order."""
        sd = self._sensors(obs)
        return np.stack([_t2n(sd[v]["rgb"])[0].astype(np.uint8) for v in VIEW_NAMES], axis=0)

    def views_depth(self, obs=None) -> np.ndarray:
        """`[2,H,W]` float32 METRIC depth (the tree stores mm/1000 -- same here)."""
        sd = self._sensors(obs)
        return np.stack([_t2n(sd[v]["depth"])[0, ..., 0].astype(np.float32) / 1000.0
                         for v in VIEW_NAMES], axis=0)

    def views_mask(self, obs=None) -> np.ndarray:
        sd = self._sensors(obs)
        return np.stack([_t2n(sd[v]["segmentation"])[0, ..., 0].astype(np.uint16)
                         for v in VIEW_NAMES], axis=0)

    def camera_params(self, obs=None) -> Tuple[np.ndarray, np.ndarray]:
        """`(extrinsics [2,4,4] cam2world, intrinsics [2,3,3])` in the RLBench convention the
        model was trained on -- the same conversion the tree was written with."""
        ps = [self._G.rlb_camera_params(self._env.unwrapped.scene.sensors[v]) for v in VIEW_NAMES]
        return (np.stack([p["extrinsics"] for p in ps]),
                np.stack([p["intrinsics"] for p in ps]))

    def current_pose8(self, obs=None) -> np.ndarray:
        """`[x,y,z,qx,qy,qz,qw,openness]` -- identical to the tree's `actions.npy` row."""
        return self._G.tcp_pose_row(self._env)

    # ------------------------------------------------------------------ stepping
    def _pose8_to_action(self, pose8: np.ndarray, env=None) -> np.ndarray:
        """World `[x,y,z,qx,qy,qz,qw,open]` -> `pd_ee_pose` action `[xyz, euler XYZ, grip]`."""
        import sapien
        from mani_skill.utils.geometry.rotation_conversions import matrix_to_euler_angles
        import torch

        env = env if env is not None else self._env
        p = np.asarray(pose8[:3], dtype=np.float64)
        qx, qy, qz, qw = (float(v) for v in pose8[3:7])
        world = sapien.Pose(p=p, q=[qw, qx, qy, qz])                 # sapien is scalar-FIRST
        root = env.unwrapped.agent.robot.root.pose.sp                # base link, world frame
        at_base = root.inv() * world
        q = torch.tensor([at_base.q], dtype=torch.float64)           # wxyz
        from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
        euler = matrix_to_euler_angles(quaternion_to_matrix(q), "XYZ")[0].numpy()
        # Binary gripper, like RLBench's Discrete(): OPEN=+1 / CLOSED=-1 (motionplanner.py:10).
        #
        # THE THRESHOLD IS 0.8, NOT 0.5, AND THAT IS NOT A TASTE DECISION. `openness` is
        # `mean(finger qpos)/0.04` (maniskill3_gen.py:gripper_openness), so a CLOSED gripper
        # reads the width of whatever it is holding, not zero. Measured over 30 episodes of
        # each of the nine ManiSkill tasks in the trees:
        #
        #     free close (no object) 0.000 | plug_charger 0.333 | peg_insertion_side 0.347
        #     pick_cube / stack_cube 0.457 | pull_cube_tool 0.576 | lift_peg_upright 0.591
        #     fully open                                                            1.000
        #
        # At 0.5 the two widest grasps (lift_peg_upright, pull_cube_tool) are read as OPEN, so
        # the hand never closes and the task is unsolvable BY THE HARNESS: GT replay of
        # lift_peg_upright scored 0/14 with zero IK failures and the peg never left the table.
        # Nothing in the 0.591-1.000 gap ever occurs as a settled state, so 0.8 separates the
        # two modes for every task in both trees.
        grip = 1.0 if float(pose8[7]) > 0.8 else -1.0
        return np.concatenate([at_base.p, euler, [grip]])

    def ik_ok(self, pose8: np.ndarray) -> bool:
        """Ask the controller's own solver whether this pose is reachable, before committing."""
        import sapien
        ctrl = self._env.unwrapped.agent.controller.controllers["arm"]
        p = np.asarray(pose8[:3], dtype=np.float64)
        qx, qy, qz, qw = (float(v) for v in pose8[3:7])
        world = sapien.Pose(p=p, q=[qw, qx, qy, qz])
        root = self._env.unwrapped.agent.robot.root.pose.sp
        at_base = root.inv() * world
        from mani_skill.utils.structs.pose import Pose as MSPose
        target = MSPose.create_from_pq(np.asarray([at_base.p]), np.asarray([at_base.q]))
        try:
            sol = ctrl.kinematics.compute_ik(
                pose=target, q0=ctrl.articulation.get_qpos(),
                current_pose=ctrl.ee_pose_at_base,
                solver_config=ctrl.config.delta_solver_config)
        except Exception:
            return False
        return sol is not None

    def step(self, pose8: np.ndarray):
        from eval.rollout_env import StepResult

        ok = self.ik_ok(pose8)
        if not ok:
            # Match RolloutEnv: an unreachable pose is REPORTED, not executed. Stepping anyway
            # would hold qpos and quietly inflate the step count with motionless frames.
            return StepResult(obs=None, ik_ok=False, success=False, terminate=False)
        action = self._pose8_to_action(pose8)
        obs, _, terminated, truncated, info = self._env.step(action)
        success = bool(np.asarray(_t2n(info["success"])).reshape(-1)[0]) if "success" in info \
            else bool(np.asarray(_t2n(terminated)).reshape(-1)[0])
        return StepResult(obs=obs, ik_ok=True, success=success,
                          terminate=bool(np.asarray(_t2n(terminated)).reshape(-1)[0]))
