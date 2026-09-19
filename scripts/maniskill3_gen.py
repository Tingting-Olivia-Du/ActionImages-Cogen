"""Generate a ManiSkill3 episode tree in the rlbench_selfgen on-disk layout.

Replays OFFICIAL ManiSkill demonstration trajectories (haosulab/ManiSkill_Demonstrations,
downloaded via `python -m mani_skill.utils.download_demo`) through deterministic
`set_state_dict` playback, rendering 4 randomly-placed 512x512 cameras per episode, and
writes episodes that RLBenchSelfgenDataset can load UNCHANGED:

    <out>/<task>/variation<V>/episodes/episode<K>/
        meta.json               {"desc": [...], "seed", "steps", "cameras", ...}
        actions.npy             [T, 8] float64  [x,y,z, qx,qy,qz,qw, openness]  (scalar-LAST
                                quaternion: get_7d_action feeds cols 3:7 straight into
                                scipy R.from_quat, which is xyzw. base.py's "[x,y,z,qw,...]"
                                comment is wrong -- follow the code, not the comment.)
        handles.json            {str(seg_id): actor/link name} for audits
        scene_segments.json     rlbench-scene-role-v1 (instances -> base_role + handles)
        seg_targets.json        rlbench-task-source-v1 (referring groups; the manipulated
                                object is base_role=distractor here and gets promoted to
                                `target` by _scene_role_lut at load time)
        view{1..4}/
            rgb/video.mp4       512x512 @ 20 fps, T frames
            camera_params.json  {str(frame): {"intrinsics": 3x3, "extrinsics": 4x4}}
            depth.npz           {"depth": [T,512,512] float16, METERS}
            mask.npz            {"mask":  [T,512,512] uint16 per-scene segmentation ids}

CAMERA CONVENTION (must match the RLBench trees, or the two sources feed the model
inconsistent camera conditioning): camera_params.json stores
  - extrinsics: 4x4 camera-to-world with the RLBench camera frame (x LEFT, y UP, z forward)
  - intrinsics: fx, fy NEGATIVE (the RLBench sign convention _focal_lengths abs()es away)
Conversion from ManiSkill's OpenCV params (verified in maniskill3_verify/report.txt:
extrinsic_cv is 3x4 world->cam OpenCV, cube-projection test PASS):
  C_rlb = inv(E_cv_4x4) @ diag(-1,-1,1,1),   K_rlb = K_cv * diag(-1,-1,1)
Both describe identical pixels; _self_check_geometry() asserts it per episode by
unprojecting rendered depth through the WRITTEN params and re-projecting.

DEPTH: ManiSkill renders int16 millimeters (shaders.py default_position_texture_transform);
written as float16 METERS. Depth 0 mm means "no geometry" (sky); cameras are sampled at
elevations that keep the sky fraction ~0 and any episode above SKY_MAX_FRAC is retried with
new cameras, so the invalid-depth sentinel path stays as rare as it is in RLBench.

SEGMENTATION: per-scene ids from segmentation_id_map. Every id that appears in ANY rendered
frame must resolve to a non-`unknown` role through scene_segments.json -- checked here, at
generation time, over all frames (the cheap startup probe in the loader is the second line
of defence, not the first).

Usage (pilot):
  python scripts/maniskill3_gen.py --env-id PickCube-v1 \
      --demo-dir /workspace/1228_tingting/maniskill3_demos --out data/maniskill3 \
      --episodes 0-4 --variation 0
"""
import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import imageio
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import mani_skill.envs  # noqa: F401
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import REGISTERED_ENVS

RES = 512
FOV = np.deg2rad(55.0)
N_VIEWS = 4
FPS = 20  # tabletop tasks run control at 20 Hz; frames are written 1:1 with control steps
# Camera-resample thresholds, evaluated on frame 0 of each view. ManiSkill tabletop scenes
# have an INFINITE ground plane and no walls (unlike RLBench's enclosed room), so a
# low-elevation camera sees ground beyond depth_codec's 10 m limit near the horizon; such
# pixels encode as the invalid sentinel -- legal, but kept rare by construction.
SKY_MAX_FRAC = 0.005   # zero-depth (no geometry at all)
FAR_MAX_FRAC = 0.02    # beyond 9.5 m (near/over the codec's 10 m limit)
NEAR_MAX_FRAC = 0.005  # closer than 6 cm (under the codec's 5 cm limit)
# Hard cap over ALL frames/views after replay: fraction of pixels outside the codec's
# strict (0.05, 10) range. Above this the episode is rejected rather than written.
OUT_OF_CODEC_MAX_FRAC = 0.03
MAX_CAMERA_RETRIES = 8

# GL->CV axis flip (OpenGL camera: x right, y up, z BACK; OpenCV: x right, y down, z fwd)
FLIP_GL_CV = np.diag([1.0, -1.0, -1.0, 1.0])
# CV->RLBench camera frame flip (RLBench: x left, y up, z forward -- the convention implied
# by RLBench's NEGATIVE fx/fy; see maniskill3_handoff follow-up notes)
FLIP_CV_RLB = np.diag([-1.0, -1.0, 1.0, 1.0])

# ---------------------------------------------------------------------------
# Per-task configuration. NO GENERIC FALLBACK for roles/instructions: a task absent
# from this table raises, because guessing roles is exactly how `unknown` pixels (or
# worse, wrong-but-plausible roles) reach training data. Extend the table task by task.
# `center`: look-at point for camera sampling (workspace center, world frame).
# `desc`: instruction paraphrases (meta.json "desc"), mirrors RLBench variation phrasing.
# `referring`: seg_targets.json groups; handles resolved from actor names at runtime.
#   The group listing the MANIPULATED object (base_role distractor below) is what promotes
#   it to `target` in the loader's scene_roles path.
# `actor_roles`: actor name -> base_role for non-robot actors. Robot links are mapped by
#   the LINK_ROLE rules below. Every actor in segmentation_id_map must be covered.
TASKS = {
    "PickCube-v1": {
        "task_dir": "pick_cube",
        "center": [0.0, 0.0, 0.05],
        "desc": [
            "pick up the red cube and move it to the goal position",
            "grasp the red cube and lift it to the target position",
            "pick the red cube up and bring it to the goal",
        ],
        "actor_roles": {
            "cube": "distractor",       # promoted to target via seg_targets at load time
            "goal_site": "goal",        # hidden at render time; mapped anyway (harmless)
            "table-workspace": "background",
            "ground": "background",
        },
        "referring": {"cube": ["cube"]},  # group name -> actor names
    },
    "StackCube-v1": {
        "task_dir": "stack_cube",
        "center": [0.0, 0.0, 0.05],
        "desc": [
            "stack the red cube on top of the green cube",
            "place the red cube onto the green cube",
            "put the red cube on the green cube",
        ],
        "actor_roles": {
            "cubeA": "distractor",
            "cubeB": "goal",
            "table-workspace": "background",
            "ground": "background",
        },
        "referring": {"red cube": ["cubeA"], "green cube": ["cubeB"]},
    },
    # Actor positions at reset (seed 0) probed 2026-09-16 to place the camera look-at
    # `center`; colors read from the task sources (push_cube.py: [12,42,160]=blue;
    # stack_pyramid.py: cubeA red, cubeB green, cubeC blue).
    "PushCube-v1": {
        "task_dir": "push_cube",
        "center": [0.08, 0.03, 0.03],
        "desc": [
            "push the blue cube onto the goal region",
            "slide the blue cube to the marked target circle",
            "push the cube until it covers the goal marker",
        ],
        "actor_roles": {
            "cube": "distractor",
            "goal_region": "goal",  # flat target disc rendered on the table
            "table-workspace": "background",
            "ground": "background",
        },
        "referring": {"cube": ["cube"], "goal region": ["goal_region"]},
    },
    "PegInsertionSide-v1": {
        "task_dir": "peg_insertion_side",
        "center": [0.0, 0.1, 0.06],
        "desc": [
            "insert the peg into the hole from the side",
            "pick up the peg and insert it sideways into the box hole",
            "put the peg horizontally into the hole of the box",
        ],
        # actor names carry a _0 suffix (per-scene build); geometry varies per seed,
        # reproduced by reset(seed, reconfigure=True) exactly as official replay does
        "actor_roles": {
            "peg_0": "distractor",
            "box_with_hole_0": "goal",
            "table-workspace": "background",
            "ground": "background",
        },
        "referring": {"peg": ["peg_0"], "box with hole": ["box_with_hole_0"]},
    },
    "PlugCharger-v1": {
        "task_dir": "plug_charger",
        "center": [0.0, 0.12, 0.06],
        "desc": [
            "plug the charger into the wall receptacle",
            "insert the charger into the socket",
            "pick up the charger and plug it into the outlet",
        ],
        "actor_roles": {
            "charger": "distractor",
            "receptacle": "goal",
            "table-workspace": "background",
            "ground": "background",
        },
        "referring": {"charger": ["charger"], "receptacle": ["receptacle"]},
    },
    "PullCubeTool-v1": {
        "task_dir": "pull_cube_tool",
        "center": [-0.05, -0.2, 0.04],
        "desc": [
            "use the l-shaped tool to pull the cube within reach",
            "grab the tool and drag the cube closer",
            "pull the cube toward the robot with the hook tool",
        ],
        "actor_roles": {
            "cube": "distractor",
            "l_shape_tool": "tool",  # first real use of the `tool` role outside RLBench
            "table-workspace": "background",
            "ground": "background",
        },
        "referring": {"cube": ["cube"], "tool": ["l_shape_tool"]},
    },
    "StackPyramid-v1": {
        "task_dir": "stack_pyramid",
        "center": [-0.04, 0.0, 0.03],
        "desc": [
            "stack the cubes into a pyramid",
            "place the green cube next to the red cube and stack the blue cube on top",
            "build a pyramid with the three cubes",
        ],
        # cubeA (red) stays put as the base; cubeB (green) and cubeC (blue) are moved,
        # so they are the promotable manipulated objects and cubeA anchors as goal.
        "actor_roles": {
            "cubeA": "goal",
            "cubeB": "distractor",
            "cubeC": "distractor",
            "table-workspace": "background",
            "ground": "background",
        },
        "referring": {
            "red cube": ["cubeA"],
            "green cube": ["cubeB"],
            "blue cube": ["cubeC"],
        },
    },
    # HELD-OUT TASKS (generate with --out data/maniskill3_heldout, NEVER into the training
    # tree): PullCube pairs with PushCube (same cube+goal_region domain, opposite motion),
    # LiftPegUpright tests reorientation, a skill family absent from training. Their demos
    # are SELF-GENERATED via mani_skill.examples.motionplanning.panda.run into
    # /workspace/1228_tingting/maniskill3_demos_selfgen (official demos for these are
    # RL-recorded and ~21 steps -- too short for a 41-frame window).
    "PullCube-v1": {
        "task_dir": "pull_cube",
        "center": [-0.1, 0.05, 0.03],
        "desc": [
            "pull the blue cube to the goal region",
            "drag the blue cube toward the robot onto the target circle",
            "pull the cube back to the marked region",
        ],
        "actor_roles": {
            "cube": "distractor",
            "goal_region": "goal",
            "table-workspace": "background",
            "ground": "background",
        },
        "referring": {"cube": ["cube"], "goal region": ["goal_region"]},
    },
    "LiftPegUpright-v1": {
        "task_dir": "lift_peg_upright",
        "center": [0.0, 0.05, 0.05],
        "desc": [
            "lift the peg and stand it upright",
            "pick up the peg and place it standing upright on the table",
            "stand the peg up vertically",
        ],
        "actor_roles": {
            "peg": "distractor",
            "table-workspace": "background",
            "ground": "background",
        },
        "referring": {"peg": ["peg"]},
    },
    # NOT yet configured, on purpose (raise rather than guess):
    # - DrawTriangle-v1 (motionplanning demos exist): 300 dot_N "ink" actors appear as the
    #   stick draws; their scene_role semantics (target? tool marks?) need a decision first.
    # - LiftPegUpright / PokeCube / PullCube / PushT / RollBall: RL-only demos named
    #   trajectory.none.<control>.physx_cuda.{h5,json}; needs a --source file-resolution
    #   tweak + one replay validated end to end before configs are added.
}

# Robot link -> role, by substring (checked in order). Mirrors the RLBench trees:
# Panda_link0..7_visual -> robot_arm; gripper/finger visuals -> gripper.
LINK_ROLE_RULES = [
    ("finger", "gripper"),
    ("hand", "gripper"),
    ("tcp", "gripper"),
    ("camera", "gripper"),  # wristcam link rides on the hand
    ("link", "robot_arm"),
    ("base", "robot_arm"),
]


def link_role(name: str) -> str:
    low = name.lower()
    for pat, role in LINK_ROLE_RULES:
        if pat in low:
            return role
    raise ValueError(f"no LINK_ROLE_RULES entry matches robot link {name!r}")


def sample_cameras(rng: np.random.Generator, center: np.ndarray):
    """4 poses on a spherical shell around `center`, RLBench-aug style.

    Elevation floor 42 deg + fov 55 keeps the top ray ~14 deg below the horizon, so the
    infinite ground plane fills the frame, zero-depth sky pixels stay ~absent and
    beyond-codec-range (>10 m) ground near the horizon stays rare (the frame-0 retry
    checks are the backstop, this is the aim). Azimuth avoids the +-30 deg cone behind
    the robot base (-x), where the arm occludes most of the workspace.
    """
    cams = []
    for _ in range(N_VIEWS):
        r = rng.uniform(0.55, 0.95)
        elev = np.deg2rad(rng.uniform(42.0, 72.0))
        azim = np.deg2rad(rng.uniform(-150.0, 150.0))
        eye = center + r * np.array(
            [np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)]
        )
        target = center + rng.normal(0.0, 0.03, 3)
        cams.append(
            {
                "eye": eye.tolist(),
                "target": target.tolist(),
                "radius": round(float(r), 4),
                "elev_deg": round(float(np.rad2deg(elev)), 1),
                "azim_deg": round(float(np.rad2deg(azim)), 1),
            }
        )
    return cams


def make_env(env_id: str, env_kwargs: dict, initial_cams: list):
    """Instantiate the task class directly (no gym wrappers -- we never step()) with a
    dynamic subclass whose sensors are ONLY our 4 randomized cameras. The default
    task cameras are dropped: they cost render time and would leak a fixed viewpoint
    into a tree whose point is viewpoint diversity. `initial_cams` is needed at the
    CLASS level because BaseEnv.__init__ runs a reconfiguring reset before the caller
    ever holds the instance; per-episode cameras are then set on the instance and
    applied by reset(options={"reconfigure": True})."""
    base_cls = REGISTERED_ENVS[env_id].cls

    class MultiCamEnv(base_cls):
        _episode_cams = initial_cams  # list of dicts from sample_cameras()

        @property
        def _default_sensor_configs(self):
            assert self._episode_cams is not None, "set _episode_cams before reset"
            return [
                CameraConfig(
                    f"view{i + 1}",
                    pose=sapien_utils.look_at(c["eye"], c["target"]),
                    width=RES,
                    height=RES,
                    fov=FOV,
                    near=0.01,
                    far=100.0,
                )
                for i, c in enumerate(self._episode_cams)
            ]

    kwargs = dict(env_kwargs)
    kwargs.pop("shader_dir", None)  # deprecated passthrough from old demo jsons
    kwargs.update(
        obs_mode="rgb+depth+segmentation",
        reward_mode="none",
        render_mode="rgb_array",
        sim_backend="physx_cpu",  # single-env deterministic state playback; render stays GPU
    )
    return MultiCamEnv(**kwargs)


def t2n(x):
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def rlb_camera_params(cam) -> dict:
    """ManiSkill OpenCV params -> RLBench-convention (cam2world 4x4, negative-f intrinsics)."""
    p = {k: t2n(v)[0] for k, v in cam.get_params().items()}
    E4 = np.eye(4)
    E4[:3, :] = p["extrinsic_cv"]
    c2w_rlb = np.linalg.inv(E4) @ FLIP_CV_RLB
    K = p["intrinsic_cv"].copy().astype(np.float64)
    K[0, 0] *= -1.0
    K[1, 1] *= -1.0
    return {"extrinsics": c2w_rlb, "intrinsics": K, "extrinsic_cv": p["extrinsic_cv"],
            "intrinsic_cv": p["intrinsic_cv"]}


def _self_check_geometry(params: dict, depth_m: np.ndarray, rng: np.random.Generator):
    """Written-params consistency: unproject rendered depth through the WRITTEN
    (RLBench-convention) params, re-project, and also re-project through the native
    OpenCV params -- all three must agree to sub-pixel. Catches any sign/axis slip in
    the conversion, per episode, before anything reaches disk."""
    K, C = params["intrinsics"], params["extrinsics"]
    Kcv, Ecv = params["intrinsic_cv"], params["extrinsic_cv"]
    vv, uu = np.where(depth_m > 0.05)
    if len(uu) < 100:
        raise RuntimeError("geometry self-check: almost no valid depth in probe frame")
    sel = rng.choice(len(uu), 100, replace=False)
    u, v, z = uu[sel].astype(np.float64), vv[sel].astype(np.float64), depth_m[vv[sel], uu[sel]].astype(np.float64)
    # unproject with RLBench-style params: x_cam = (u - cx)/fx * z, y_cam = (v - cy)/fy * z
    x_cam = (u - K[0, 2]) / K[0, 0] * z
    y_cam = (v - K[1, 2]) / K[1, 1] * z
    pts_cam = np.stack([x_cam, y_cam, z], axis=1)
    pts_w = (C[:3, :3] @ pts_cam.T + C[:3, 3:4]).T
    # re-project through the native OpenCV params
    pc = (Ecv[:3, :3] @ pts_w.T + Ecv[:3, 3:4]).T
    assert (pc[:, 2] > 0).all(), "unprojected points behind the OpenCV camera"
    u2 = Kcv[0, 0] * pc[:, 0] / pc[:, 2] + Kcv[0, 2]
    v2 = Kcv[1, 1] * pc[:, 1] / pc[:, 2] + Kcv[1, 2]
    err = np.hypot(u2 - u, v2 - v).max()
    if err > 0.51:
        raise RuntimeError(f"geometry self-check FAILED: reprojection error {err:.3f}px")
    return float(err)


def build_seg_metadata(env, cfg: dict):
    """segmentation_id_map -> (handles.json, scene_segments.json, seg_targets.json)."""
    id_map = {int(k): v for k, v in env.unwrapped.segmentation_id_map.items()}
    handles_json = {str(i): obj.name for i, obj in sorted(id_map.items())}

    name_to_ids: dict = {}
    for i, obj in id_map.items():
        name_to_ids.setdefault(obj.name, []).append(i)

    instances: dict = {"environment": {"base_role": "background", "handles": [0]}}
    robot_arm, gripper = [], []
    covered = {0}
    for i, obj in sorted(id_map.items()):
        if type(obj).__name__ == "Link":
            (robot_arm if link_role(obj.name) == "robot_arm" else gripper).append(i)
            covered.add(i)
        elif obj.name in cfg["actor_roles"]:
            role = cfg["actor_roles"][obj.name]
            if role == "background":
                instances["environment"]["handles"].append(i)
            else:
                instances.setdefault(obj.name, {"base_role": role, "handles": []})[
                    "handles"
                ].append(i)
            covered.add(i)
    uncovered = sorted(set(id_map) - covered)
    if uncovered:
        raise ValueError(
            f"actors without a role: {[(i, id_map[i].name) for i in uncovered]} -- "
            f"extend TASKS[...]['actor_roles']; refusing to guess."
        )
    instances["robot_arm"] = {"base_role": "robot_arm", "handles": robot_arm}
    instances["robot_gripper"] = {"base_role": "gripper", "handles": gripper}
    scene_segments = {"instances": instances, "protocol": "rlbench-scene-role-v1"}

    referring = {}
    for group, actor_names in cfg["referring"].items():
        handles = [i for n in actor_names for i in name_to_ids.get(n, [])]
        if not handles:
            raise ValueError(f"referring group {group!r}: no actor named {actor_names}")
        referring[group] = {"handles": handles}
    seg_targets = {
        "referring": referring,
        "referring_by_color": {},
        "protocol": "rlbench-task-source-v1",
        "task": cfg["task_dir"],
    }
    return handles_json, scene_segments, seg_targets, id_map


def gripper_openness(env) -> float:
    agent = env.unwrapped.agent
    qpos = t2n(agent.robot.get_qpos())[0]
    names = [j.name for j in agent.robot.get_active_joints()]
    finger = [i for i, n in enumerate(names) if "finger" in n.lower()]
    if not finger:
        return 1.0  # stick-type end effector: no gripper, permanently "open"
    # panda finger joints: 0 (closed) .. 0.04 (open)
    return float(np.clip(np.mean(qpos[finger]) / 0.04, 0.0, 1.0))


def tcp_pose_row(env) -> np.ndarray:
    """[x,y,z, qx,qy,qz,qw, openness] -- absolute per-frame EE pose, scalar-LAST quat."""
    pose = env.unwrapped.agent.tcp.pose
    p = t2n(pose.p)[0]
    q_wxyz = t2n(pose.q)[0]
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return np.concatenate([p, q_xyzw, [gripper_openness(env)]])


def record_episode(env_id, cfg, ep_meta, h5file, out_dir: Path, source: str, env=None):
    """Renders one episode. Returns (env, n_frames, err) -- the env is REUSED across
    episodes by the caller: constructing a fresh env per episode leaks Vulkan device
    memory in sapien 3.0.3 (close() does not return it), and 11 concurrent workers
    OOM-segfault after ~4 episodes each. One env per process + per-episode
    reset(options={"reconfigure": True}) is also what official replay_trajectory does."""
    seed = ep_meta["episode_seed"]
    traj = h5file[f"traj_{ep_meta['episode_id']}"]
    states = traj["env_states"]
    n_frames = len(next(iter(states["actors"].values()))) if "actors" in states else len(
        next(iter(states["articulations"].values()))
    )

    cam_rng = np.random.default_rng(seed + 777)
    # The out-of-codec check is CAMERA-dependent (an arm sweeping within 5 cm of a
    # low-radius camera mid-episode fails it), and cam_rng is seeded by the episode
    # seed -- so a plain per-episode failure would recur identically on every rerun
    # and the episode would be lost forever. Retry the whole render with freshly
    # drawn cameras instead (the rng state advances across attempts).
    out_msg = None
    for episode_attempt in range(3):
        for attempt in range(MAX_CAMERA_RETRIES):
            cams = sample_cameras(cam_rng, np.array(cfg["center"]))
            if env is None:
                env = make_env(env_id, ep_meta["env_kwargs"], cams)
            env._episode_cams = cams
            reset_kwargs = dict(ep_meta["reset_kwargs"])
            options = dict(reset_kwargs.get("options") or {})
            options["reconfigure"] = True  # rebuild sensors so this episode's cameras apply
            env.reset(seed=reset_kwargs.get("seed", seed), options=options)

            # probe frame 0 of every view: sky / beyond-range / too-close fractions
            env.unwrapped.set_state_dict(_state_at(states, 0))
            obs = env.unwrapped.get_obs()
            worst = {"sky": 0.0, "far": 0.0, "near": 0.0}
            for i in range(N_VIEWS):
                d = t2n(obs["sensor_data"][f"view{i+1}"]["depth"])[0, ..., 0].astype(np.float32) / 1000.0
                worst["sky"] = max(worst["sky"], float((d == 0).mean()))
                worst["far"] = max(worst["far"], float((d > 9.5).mean()))
                worst["near"] = max(worst["near"], float(((d > 0) & (d < 0.06)).mean()))
            if (worst["sky"] <= SKY_MAX_FRAC and worst["far"] <= FAR_MAX_FRAC
                    and worst["near"] <= NEAR_MAX_FRAC):
                break
            print(f"  [retry {attempt}] camera check failed {worst}, resampling")
        else:
            raise RuntimeError(
                f"episode seed {seed}: no acceptable cameras after {MAX_CAMERA_RETRIES} tries ({worst})"
            )

        handles_json, scene_segments, seg_targets, id_map = build_seg_metadata(env, cfg)
        role_of = {}
        for inst in scene_segments["instances"].values():
            for h in inst["handles"]:
                role_of[h] = inst["base_role"]

        cam_objs = {f"view{i+1}": env.unwrapped.scene.sensors[f"view{i+1}"] for i in range(N_VIEWS)}
        cam_params = {name: rlb_camera_params(c) for name, c in cam_objs.items()}

        rgb_frames = {v: [] for v in cam_objs}
        depth_frames = {v: [] for v in cam_objs}
        mask_frames = {v: [] for v in cam_objs}
        actions = []
        seen_ids = set()

        for t in range(n_frames):
            env.unwrapped.set_state_dict(_state_at(states, t))
            obs = env.unwrapped.get_obs()
            for v in cam_objs:
                data = obs["sensor_data"][v]
                rgb_frames[v].append(t2n(data["rgb"])[0])
                d_mm = t2n(data["depth"])[0, ..., 0].astype(np.int32)
                depth_frames[v].append((d_mm.astype(np.float32) / 1000.0).astype(np.float16))
                m = t2n(data["segmentation"])[0, ..., 0]
                assert m.min() >= 0, "negative segmentation id"
                mask_frames[v].append(m.astype(np.uint16))
                seen_ids.update(np.unique(m).tolist())
            actions.append(tcp_pose_row(env))

        # ---- generation-time checks, before anything is written ----
        unmapped = sorted(i for i in seen_ids if i not in role_of)
        if unmapped:
            raise RuntimeError(
                f"rendered segmentation ids with no role: "
                f"{[(i, id_map.get(i, '???').name if i in id_map else '???') for i in unmapped]}"
            )
        err = _self_check_geometry(
            cam_params["view1"], depth_frames["view1"][0].astype(np.float32), cam_rng
        )
        actions = np.stack(actions).astype(np.float64)
        qn = np.linalg.norm(actions[:, 3:7], axis=1)
        assert np.abs(qn - 1).max() < 1e-3, "non-unit quaternion in actions"
        out_frac = {}
        for v in cam_objs:
            depth_all = np.stack(depth_frames[v]).astype(np.float32)
            outside = (depth_all <= 0.05) | (depth_all >= 10.0)  # incl. zero-depth sky
            out_frac[v] = float(outside.mean())
        worst_v = max(out_frac, key=out_frac.get)
        if out_frac[worst_v] > OUT_OF_CODEC_MAX_FRAC:
            out_msg = (f"{out_frac[worst_v]*100:.2f}% of {worst_v} pixels outside "
                       f"depth_codec's (0.05, 10) m range")
            print(f"  [episode retry {episode_attempt}] {out_msg}, resampling cameras",
                  flush=True)
            continue
        break
    else:
        raise RuntimeError(
            f"episode seed {seed}: still outside depth range after 3 camera draws: "
            f"{out_msg}; refusing to write"
        )

    # ---- write ----
    out_dir.mkdir(parents=True, exist_ok=True)
    for v in cam_objs:
        vd = out_dir / v
        (vd / "rgb").mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(str(vd / "rgb" / "video.mp4"), fps=FPS,
                                codec="libx264", quality=8, macro_block_size=1) as w:
            for f in rgb_frames[v]:
                w.append_data(f)
        np.savez_compressed(vd / "depth.npz", depth=np.stack(depth_frames[v]))
        np.savez_compressed(vd / "mask.npz", mask=np.stack(mask_frames[v]))
        per_frame = {
            str(t): {
                "intrinsics": cam_params[v]["intrinsics"].tolist(),
                "extrinsics": cam_params[v]["extrinsics"].tolist(),
            }
            for t in range(n_frames)
        }
        with open(vd / "camera_params.json", "w") as f:
            json.dump(per_frame, f)

    np.save(out_dir / "actions.npy", actions)
    meta = {
        "seed": int(seed),
        "steps": int(n_frames),
        "desc": cfg["desc"],
        "env_id": env_id,
        "source": source,
        "success": bool(ep_meta.get("success", True)),
        "cameras": {v: {k: cams[i][k] for k in ("radius", "elev_deg", "azim_deg")}
                    for i, v in enumerate(sorted(cam_objs))},
        "geometry_check_px": round(err, 4),
        "depth_out_of_codec_frac": {v: round(f, 5) for v, f in out_frac.items()},
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f)
    with open(out_dir / "handles.json", "w") as f:
        json.dump(handles_json, f)
    with open(out_dir / "scene_segments.json", "w") as f:
        json.dump(scene_segments, f, indent=1)
    with open(out_dir / "seg_targets.json", "w") as f:
        json.dump(seg_targets, f, indent=1)
    return env, n_frames, err


def _state_at(states_group, t: int) -> dict:
    """h5 env_states group -> state dict for a single timestep t (still batched: [1, D])."""
    out = {}
    for kind in states_group:  # "actors" / "articulations"
        out[kind] = {name: torch.from_numpy(np.asarray(ds[t])[None]) for name, ds in states_group[kind].items()}
    return out


def parse_episode_spec(spec: str):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", required=True, choices=sorted(TASKS.keys()),
                    help="tasks absent from TASKS raise on purpose -- extend the table")
    ap.add_argument("--demo-dir", default="/workspace/1228_tingting/maniskill3_demos")
    ap.add_argument("--source", default="motionplanning",
                    help="subdir of the demo dataset (motionplanning / rl / teleop)")
    ap.add_argument("--out", default=str(REPO / "data" / "maniskill3"))
    ap.add_argument("--episodes", required=True,
                    help="episode ids in the demo file, e.g. '0-4' or '0,7,42'")
    ap.add_argument("--variation", type=int, default=0,
                    help="variation<N> dir to write into (0=train seeds, 1=held-out seeds)")
    ap.add_argument("--episode-offset", type=int, default=None,
                    help="output episode numbering start; default = first demo episode id")
    args = ap.parse_args()

    cfg = TASKS[args.env_id]
    demo_base = Path(args.demo_dir) / args.env_id / args.source
    traj_json = json.load(open(demo_base / "trajectory.json"))
    by_id = {e["episode_id"]: e for e in traj_json["episodes"]}
    env_kwargs = traj_json["env_info"]["env_kwargs"]

    ep_ids = parse_episode_spec(args.episodes)
    offset = args.episode_offset if args.episode_offset is not None else ep_ids[0]

    failures = 0
    env = None  # one env per process, reused across episodes (see record_episode docstring)
    try:
        with h5py.File(demo_base / "trajectory.h5", "r") as f:
            for k, ep_id in enumerate(ep_ids):
                ep = dict(by_id[ep_id])
                ep["env_kwargs"] = env_kwargs
                if not ep.get("success", True):
                    print(f"episode {ep_id}: not successful in demo metadata, skipping")
                    continue
                out_dir = (Path(args.out) / cfg["task_dir"] / f"variation{args.variation}"
                           / "episodes" / f"episode{offset + k}")
                if (out_dir / "meta.json").exists():
                    print(f"episode {ep_id} -> {out_dir}: exists, skipping")
                    continue
                # One bad episode (camera-retry exhaustion, a corrupt trajectory) must not
                # kill a 250-episode unattended run. meta.json is written LAST, so a failed
                # episode leaves no meta.json and a rerun retries it after the partial dir
                # is removed. The env is rebuilt after a failure: it may be mid-reset.
                try:
                    env, n, err = record_episode(args.env_id, cfg, ep, f, out_dir,
                                                 args.source, env=env)
                    print(f"episode {ep_id} -> {out_dir}: {n} frames, geom check "
                          f"{err:.3f}px", flush=True)
                except Exception:
                    failures += 1
                    import shutil
                    import traceback
                    traceback.print_exc()
                    shutil.rmtree(out_dir, ignore_errors=True)  # no partial episodes
                    if env is not None:
                        try:
                            env.close()
                        except Exception:
                            pass
                        env = None
                    print(f"episode {ep_id} -> {out_dir}: FAILED (see traceback above), "
                          f"continuing", flush=True)
    finally:
        if env is not None:
            env.close()
    if failures:
        print(f"{failures} episode(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
