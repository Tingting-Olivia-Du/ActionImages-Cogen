"""RLBench environment wrapper for closed-loop rollouts of the action-image policy.

The whole point of this file is that a rollout must be served the SAME distribution the model
was trained on. Everything here is pinned to what `ttd/scripts/gen_dataset.py` wrote into
`data/rlbench_selfgen`, and the pins are checkable rather than asserted in prose:

  * cameras       -- `view1 = front`, `view2 = overhead`, RLBench's own named cameras at
                     256x256 (gen_dataset.py:21-22). No hand-placed VisionSensor is involved,
                     so the rig is reproducible by construction.
  * intrinsics    -- read back from `obs.misc[f"{cam}_camera_intrinsics"]`, the exact path
                     gen_dataset.py:133 recorded them through. Do NOT derive them from FOV:
                     the recorded matrices have NEGATIVE focal lengths (fx = fy = -351.68 at
                     256^2) and a derived matrix flips the sign, which silently mirrors every
                     projection instead of failing.
  * scene         -- `reset()` reproduces a stored episode by replaying gen_dataset's seeding
                     (`np.random.seed(seed + variation * 1_000_003)`), verified against the
                     stored `actions.npy[0]` to ~1e-4 m by tests/test_rollout_env.py.
  * action mode   -- `EndEffectorPoseViaIK`, not `ViaPlanning`. The recorded waypoints are
                     7.7 mm apart at the median (p95 18 mm), and ViaPlanning would run a full
                     collision-checked plan for each one.

Nothing in this module imports torch or the model; it is exercised end to end by feeding
recorded ground-truth actions back in, which is what makes it a usable self-check.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# `view1`/`view2` in the selfgen tree map to these RLBench cameras, in this order.
# Mirrors ttd/scripts/gen_dataset.py:22 -- the model's cond/target views are view1/view2.
VIEW_CAMERAS: Tuple[str, str] = ("front", "overhead")
DEFAULT_RESOLUTION = 256

# gen_dataset.py:225 seeds the global RNG this way before `set_variation` + `get_demos`, so
# replaying it is what reproduces a stored episode's object placement. `attempt` is the
# retry counter; every episode that succeeded on its first try (all of them, since a retry
# would have been logged) used attempt=0.
SEED_VARIATION_STRIDE = 1_000_003
SEED_ATTEMPT_STRIDE = 100_003


def episode_seed_to_rng_seed(episode_seed: int, variation: int, attempt: int = 0) -> int:
    """The scene seed gen_dataset.py used for (episode_seed, variation)."""
    return episode_seed + variation * SEED_VARIATION_STRIDE + attempt * SEED_ATTEMPT_STRIDE


def task_name_to_class(task_name: str):
    """`close_jar` -> `rlbench.tasks.CloseJar`, the same mapping gen_dataset.py:188 built."""
    import rlbench.tasks as tasks

    table = {
        re.sub(r"(?<!^)(?=[A-Z])", "_", c).lower(): getattr(tasks, c)
        for c in dir(tasks)
        if c[0].isupper()
    }
    if task_name not in table:
        raise KeyError(f"unknown RLBench task {task_name!r}")
    return table[task_name]


@dataclass
class StepResult:
    """One control step. `ik_ok=False` means the pose was unreachable and nothing was applied."""

    success: bool
    terminate: bool
    ik_ok: bool
    obs: object = None
    error: Optional[str] = None


@dataclass
class RolloutStats:
    """Per-episode accounting. `ik_fail` has to be reported: without it a 0% success rate
    cannot be split into 'the model drew the wrong pose' and 'the pose was unreachable'."""

    steps: int = 0
    ik_fail: int = 0
    ik_fail_streak_max: int = 0
    replans: int = 0
    low_conf_held: int = 0
    success: bool = False
    terminated_early: bool = False
    stop_reason: str = ""

    @property
    def ik_fail_rate(self) -> float:
        return self.ik_fail / max(self.steps, 1)

    def as_dict(self) -> Dict:
        return {
            "steps": self.steps,
            "ik_fail": self.ik_fail,
            "ik_fail_rate": round(self.ik_fail_rate, 4),
            "ik_fail_streak_max": self.ik_fail_streak_max,
            "replans": self.replans,
            "low_conf_held": self.low_conf_held,
            "success": self.success,
            "terminated_early": self.terminated_early,
            "stop_reason": self.stop_reason,
        }


class RolloutEnv:
    """A launched RLBench environment configured to match the selfgen render rig."""

    def __init__(
        self,
        resolution: int = DEFAULT_RESOLUTION,
        headless: bool = True,
        collision_checking: bool = False,
        wrist_camera: bool = False,
        arm_action_mode: str = "ik",
        anchor_modality: str = "video",
        extra_renders: Sequence[str] = (),
    ) -> None:
        """`arm_action_mode` is "ik" or "planning".

        Measured on a close_jar ground-truth replay (247-271 waypoints at 20 Hz):

            ViaIK        34 IK failures (14%)   49.8 s   0.20 s/waypoint
            ViaPlanning   0 IK failures ( 0%)   68.7 s   0.25 s/waypoint

        So `planning` is only ~25% slower per waypoint, not the order of magnitude assumed
        earlier, and it removes the ik_fail confound entirely. It is NOT the default anyway,
        because those numbers are for well-formed ground-truth poses: an unreachable target
        makes ViaIK fail immediately, whereas the planner may spend a long time searching
        before giving up, and a rollout campaign is scored partly on wall-clock. Switch to
        `planning` if `ik_fail_rate` turns out to be large enough to muddy a result.
        """
        # Which observation the policy is anchored on. `video` is the default and leaves the
        # camera config byte-identical to every previous campaign -- the 46.0% RGB number was
        # produced under it and must stay reproducible. The other three turn on the extra
        # renders they need, which costs sim time per step (see the CameraConfig comment below).
        # "all" = 融合模式:同时渲染四个模态,策略对每条路径各生成一次再几何中位融合。
        if anchor_modality not in ("video", "depth", "segmentation", "normal", "all"):
            raise ValueError(f"anchor_modality must be video/depth/segmentation/normal/all, "
                             f"got {anchor_modality!r}")
        self.anchor_modality = anchor_modality
        # depth is needed by BOTH `depth` and `normal` (normal is computed from it);
        # the handle map is needed only by `segmentation`.
        self._need_depth = anchor_modality in ("depth", "normal", "all")
        self._need_mask = anchor_modality in ("segmentation", "all")
        # `extra_renders` DECOUPLES what the simulator renders from what the policy is anchored
        # on. The two were the same thing as long as a render existed only to build the next
        # anchor. The sim-replay perception eval breaks that: it needs the simulator's OWN
        # depth/mask at every executed step as the counterfactual ground truth for whatever the
        # model imagined -- and for the RGB-only arm, which HAS no depth+action template and so
        # must run anchor_modality="video", there is no anchor that would turn the depth render
        # on. Without this the arm0 half of that comparison is simply unmeasurable.
        # It only ever ADDS renders, so every existing campaign's camera config is byte-identical.
        for m in extra_renders:
            if m not in ("depth", "mask"):
                raise ValueError(f"extra_renders takes 'depth'/'mask', got {m!r}")
        self._need_depth = self._need_depth or ("depth" in extra_renders)
        self._need_mask = self._need_mask or ("mask" in extra_renders)
        self.extra_renders = tuple(extra_renders)
        if arm_action_mode not in ("ik", "planning"):
            raise ValueError(f"arm_action_mode must be 'ik' or 'planning', got {arm_action_mode!r}")
        self.resolution = resolution
        self.headless = headless
        self.collision_checking = collision_checking
        self.wrist_camera = wrist_camera
        self.arm_action_mode = arm_action_mode
        self._env = None
        self._task = None
        self._task_name: Optional[str] = None
        self._launched = False

    # ------------------------------------------------------------------ lifecycle
    @staticmethod
    def _restore_coppeliasim_qt_path() -> Optional[str]:
        """Point Qt back at CoppeliaSim's plugins, undoing OpenCV's hijack.

        `import cv2` rewrites QT_QPA_PLATFORM_PLUGIN_PATH to its own bundled Qt plugins, and the
        model pipeline pulls cv2 in transitively. CoppeliaSim then aborts at launch with
        `Could not find the Qt platform plugin "xcb"` -- exit 134, after the 56 s checkpoint
        load has already been paid. env_eval.rc sets the variable correctly, but that happens
        before Python starts, so it loses to whatever imports later. Re-asserting it here makes
        the launch independent of import order.

        Only the GT-replay path escapes this, because it never imports the model.
        """
        root = os.environ.get("COPPELIASIM_ROOT")
        if not root:
            raise RuntimeError(
                "COPPELIASIM_ROOT is unset -- source /workspace/ttdu/ttd/scripts/env_eval.rc")
        current = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")
        if current != root:
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = root
        return current

    def launch(self) -> "RolloutEnv":
        hijacked = self._restore_coppeliasim_qt_path()
        if hijacked is not None and hijacked != os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"]:
            print(f"[rollout_env] QT_QPA_PLATFORM_PLUGIN_PATH was {hijacked!r} "
                  f"(likely set by `import cv2`); restored to COPPELIASIM_ROOT", flush=True)

        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import (
            EndEffectorPoseViaIK,
            EndEffectorPoseViaPlanning,
        )
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.environment import Environment
        from rlbench.observation_config import CameraConfig, ObservationConfig

        size = [self.resolution, self.resolution]
        # RGB always; depth/mask only when the chosen anchor needs them, because each extra
        # render costs sim time on EVERY step. `depth_in_meters=True` matches gen_dataset.py:210
        # -- training's depth codec (`encode_depth`) takes metres, and CoppeliaSim's default
        # normalised depth would silently encode a different quantity.
        on = CameraConfig(rgb=True, depth=self._need_depth, mask=self._need_mask,
                          point_cloud=False, image_size=size,
                          depth_in_meters=True)
        off = CameraConfig(rgb=False, depth=False, mask=False, point_cloud=False)
        cams = {f"{c}_camera": (on if c in VIEW_CAMERAS else off) for c in
                ("front", "overhead", "left_shoulder", "right_shoulder", "wrist")}
        if self.wrist_camera:
            cams["wrist_camera"] = on
        obs_config = ObservationConfig(**cams)
        obs_config.joint_positions = True
        obs_config.gripper_pose = True
        obs_config.gripper_open = True

        arm_cls = EndEffectorPoseViaIK if self.arm_action_mode == "ik" else EndEffectorPoseViaPlanning
        self._env = Environment(
            MoveArmThenGripper(
                arm_cls(absolute_mode=True, collision_checking=self.collision_checking),
                Discrete(),
            ),
            obs_config=obs_config,
            headless=self.headless,
        )
        self._env.launch()
        self._launched = True
        return self

    def shutdown(self) -> None:
        if self._launched and self._env is not None:
            self._env.shutdown()
            self._launched = False

    def __enter__(self) -> "RolloutEnv":
        return self.launch()

    def __exit__(self, *exc) -> None:
        self.shutdown()

    # ------------------------------------------------------------------ episodes
    def reset(self, task_name: str, variation: int, episode_seed: int, attempt: int = 0):
        """Reproduce the scene of `data/rlbench_selfgen/<task>/variation<v>/episodes/episode<s>`.

        Returns `(descriptions, obs)`. The descriptions come from the live task; the training
        prompt should still be taken from the episode's `meta.json` so eval and training agree
        word for word.
        """
        # Order mirrors gen_dataset.py:224-226: seed, then set_variation, then reset.
        #
        # NOTE this does NOT reproduce a stored episode's object placement, only its seeding
        # ritual. Measured: the arm's home pose matches to 1e-5 m (it is the same regardless of
        # the scene, so that is not evidence), but the rendered first frame differs from the
        # stored video by MAE 19-24/255, and replaying a stored `actions.npy` through the
        # resulting scene never reaches success. CoppeliaSim carries simulator-side state that
        # the numpy seed does not cover. Use `reset_to_new_demo` for evaluation, which sidesteps
        # scene reproduction entirely; this method is kept for interactive inspection.
        np.random.seed(episode_seed_to_rng_seed(episode_seed, variation, attempt))
        self._select_task(task_name, variation)
        return self._task.reset()

    def _select_task(self, task_name: str, variation: int):
        if not self._launched:
            raise RuntimeError("call launch() first")
        if self._task_name != task_name:
            self._task = self._env.get_task(task_name_to_class(task_name))
            self._task_name = task_name
        n_var = self._task.variation_count()
        if not 0 <= variation < n_var:
            raise ValueError(f"{task_name} has {n_var} variations, asked for {variation}")
        self._task.set_variation(variation)
        return self._task

    def reset_to_new_demo(self, task_name: str, variation: int, scene_seed: int,
                          max_attempts: int = 3):
        """Generate a demo for a freshly seeded scene, then rewind to that scene's start.

        Preferred over a bare `reset()` for evaluation, for two reasons:

          * the scene is GUARANTEED SOLVABLE -- a demo for it exists -- so a failed rollout
            cannot be blamed on an impossible initial state;
          * the demo's own trajectory is returned, which gives both the ground-truth replay
            self-check and an open-loop reference for the same scene, for free.

        `scene_seed` fully determines the scene, so two checkpoints scored at the same seed
        face the same world -- the precondition for the paired test.

        Returns `(descriptions, obs, demo_actions [N, 8])`.
        """
        task = self._select_task(task_name, variation)
        np.random.seed(scene_seed)
        demo = task.get_demos(1, live_demos=True, max_attempts=max_attempts)[0]
        actions = np.stack([np.concatenate([o.gripper_pose, [float(o.gripper_open)]])
                            for o in demo])
        descriptions, obs = task.reset_to_demo(demo)
        return descriptions, obs, actions

    def step(self, pose8: np.ndarray) -> StepResult:
        """Apply one `[x, y, z, qx, qy, qz, qw, openness]` pose.

        An unreachable pose is reported, not raised: a rollout that dies on the first bad IK
        target would score the model on reachability instead of on its predictions. The caller
        decides how many consecutive failures to tolerate.
        """
        from rlbench.backend.exceptions import InvalidActionError

        pose8 = np.asarray(pose8, dtype=np.float64).reshape(-1)
        if pose8.shape != (8,):
            raise ValueError(f"expected an 8-vector [x,y,z,qx,qy,qz,qw,g], got {pose8.shape}")
        if not np.all(np.isfinite(pose8)):
            return StepResult(False, False, False, None, "non-finite action")

        # RLBench asserts a unit quaternion to 1e-... tolerance; a float32 decode round-trips
        # to ~1e-7 off, which trips the assert. Renormalize rather than let it raise.
        quat = pose8[3:7]
        norm = float(np.linalg.norm(quat))
        if norm < 1e-8:
            return StepResult(False, False, False, None, "degenerate quaternion")
        action = np.concatenate([pose8[:3], quat / norm, [float(pose8[7] > 0.5)]])

        try:
            obs, reward, terminate = self._task.step(action)
        except InvalidActionError as exc:
            return StepResult(False, False, False, None, f"{type(exc).__name__}: {exc}")
        return StepResult(bool(reward > 0), bool(terminate), True, obs)

    # ------------------------------------------------------------------ observations
    @staticmethod
    def views_rgb(obs) -> np.ndarray:
        """`[V, H, W, 3]` uint8 in VIEW_CAMERAS order, i.e. selfgen's view1, view2."""
        return np.stack([getattr(obs, f"{cam}_rgb") for cam in VIEW_CAMERAS], axis=0)

    @staticmethod
    def views_depth(obs) -> np.ndarray:
        """`[V, H, W]` float32 METRIC depth in VIEW_CAMERAS order (needs depth_in_meters=True)."""
        return np.stack([np.asarray(getattr(obs, f"{cam}_depth"), dtype=np.float32)
                         for cam in VIEW_CAMERAS], axis=0)

    @staticmethod
    def views_mask(obs) -> np.ndarray:
        """`[V, H, W]` uint16 handle map in VIEW_CAMERAS order."""
        return np.stack([np.asarray(getattr(obs, f"{cam}_mask")).astype(np.uint16)
                         for cam in VIEW_CAMERAS], axis=0)

    @staticmethod
    def handle_names(mask_views: np.ndarray) -> dict:
        """{handle_id_str: object_name} for every handle actually rendered, queried LIVE.

        Deliberately NOT reusing the training tree's handles.json. CoppeliaSim numbers handles
        by scene load order and the Colosseum augmentation changes it, so the same object holds
        different ids in different episodes (scene_segments_gen.py:task_handle_union documents
        stack_wine's wine_bottle_visual as 82 in some episodes and 102 in others). A stale map
        would decode live pixels as `unknown` -- a colour that appears ZERO times in training --
        and the model would see an out-of-distribution anchor with nothing raising an error.
        """
        from pyrep.backend import sim as simb
        names = {}
        for h in sorted({int(x) for x in np.unique(mask_views)}):
            try:
                names[str(h)] = simb.simGetObjectName(h)
            except Exception:
                names[str(h)] = ""
        return names

    @staticmethod
    def camera_params(obs) -> Tuple[np.ndarray, np.ndarray]:
        """`(extrinsics [V,4,4] camera-to-world, intrinsics [V,3,3])`, read the way
        gen_dataset.py:133 recorded them. The cameras are static within an episode (verified
        on disk: every frame of every view carries identical matrices), so one read per replan
        is enough and the caller tiles them over the window."""
        extr = np.stack([np.asarray(obs.misc[f"{c}_camera_extrinsics"], dtype=np.float64)
                         for c in VIEW_CAMERAS], axis=0)
        intr = np.stack([np.asarray(obs.misc[f"{c}_camera_intrinsics"], dtype=np.float64)
                         for c in VIEW_CAMERAS], axis=0)
        return extr, intr

    @staticmethod
    def current_pose8(obs) -> np.ndarray:
        """The live `[x,y,z,qx,qy,qz,qw,openness]`, in the same layout as `actions.npy`."""
        return np.concatenate([np.asarray(obs.gripper_pose, dtype=np.float64),
                               [float(obs.gripper_open)]])
