"""Render LIBERO demonstrations into the rlbench_selfgen on-disk layout, with OUR cameras.

Replays the OFFICIAL LIBERO demonstrations (yifengzhu-hf/LIBERO-datasets, the raw HDF5s -- the
only copies that carry the MuJoCo `states` needed to re-render) frame by frame, from four
randomly placed 512x512 cameras per episode, and writes episodes RLBenchSelfgenDataset can load
UNCHANGED -- the same byte-level contract as scripts/maniskill3_gen.py:

    <out>/<task>/variation<V>/episodes/episode<K>/
        meta.json  actions.npy  handles.json  scene_segments.json  seg_targets.json
        view{1..4}/rgb/video.mp4  view{1..4}/camera_params.json
        view{1..4}/depth.npz (float16 METRES)  view{1..4}/mask.npz (uint16 instance ids)

WHY NOT THE OTHER LIBERO COPIES ON THIS MACHINE. The RLDS conversions (openvla's *_no_noops)
hold 256^2 RGB from LIBERO's two fixed cameras and no simulator state, so they cannot be
re-rendered from new viewpoints, and have no depth, no segmentation and no camera matrices --
three of the things this layout requires.

THE CAMERA CONTRACT, every line of which was MEASURED on this machine (2026-09-21), not assumed:

  * ORIENTATION: robosuite renders with macros.IMAGE_CONVENTION = "opengl" (origin bottom-left)
    and LIBERO does not change it, while the intrinsics/extrinsics it hands out are OpenCV
    (origin top-left). So rgb, depth AND segmentation are all flipped with [::-1] before they
    are written. Decided by projecting every object's centre into four random views across
    three suites and reading the segmentation there: 53/63 hits flipped, 12/63 as-is. (A
    single-point test on the end-effector near the image centre had pointed the other way --
    30 rows apart is not a test.)
  * DEPTH UNITS: MuJoCo's depth buffer is NORMALISED ([0.943, 0.997] raw on the first probe),
    converted with robosuite's get_real_depth_map (-> [0.19, 3.46] m, median 0.62 m).
    plan/paper_reference.md Sec. 8 records "depth median ~0.99 m, inside the codec range";
    that number was the normalised buffer, not metres.
  * CAMERA CONVENTION: get_camera_extrinsic_matrix returns an OpenCV camera-to-world pose, so
    the RLBench convention every tree here uses is  C_rlb = E_cv @ diag(-1,-1,1,1)  and
    K_rlb = K * diag(-1,-1,1)  -- the same conversion scripts/maniskill3_gen.py applies.
  * FOV: 55 deg, as in the ManiSkill tree. LIBERO's cameras default to 45 deg.
  * PLACEMENT: all four LIBERO cameras are attached to the world body (cam_bodyid == 0), so
    writing model.cam_pos / cam_quat puts them at world poses directly. They are sampled with
    EXACTLY the ManiSkill distribution (radius 0.55-0.95 m, elevation 42-72 deg, azimuth
    +-150 deg, look-at jittered N(0, 3 cm)) around the centroid of the task's movable objects.
  * Every episode passes the geometry self-check before it is written: rendered depth is
    unprojected through the WRITTEN (RLBench-convention) parameters and re-projected through
    the native OpenCV ones, and must agree to under 0.51 px.

SEGMENTATION ROLES are derived from the BDDL goal, not hand-listed per task: the first
argument of an on/in predicate is the manipulated object (base role `distractor`, promoted to
`target` by seg_targets.json's referring group, exactly as the ManiSkill tree does it), the
object owning the second argument's region is the `goal`, the subject of open/close/turnon/
turnoff is the `target` fixture, other fixtures are `fixture`, the robot and its mount are
`robot_arm`, the gripper is `gripper`, and instance 0 (table, floor, walls) is `background`.
Every rendered id must resolve to a role -- `unknown` never reaches disk.

A LIBERO DATASET QUIRK THIS TREE IS IMMUNE TO: in the raw HDF5s `states[t]` lines up with
`obs[t-1]`, not `obs[t]` (end-effector position 0.32 mm apart one step back, 8 mm at the same
index -- measured on libero_spatial task 6). Every frame AND every pose label here is computed
from the same replayed states[t], so they are aligned by construction; the HDF5's `obs` is
never read. Anyone mixing that `obs` with `states` inherits a one-step (~8 mm) misalignment.

CLEANING follows openvla's regenerate_libero_dataset: a step whose arm action is below 1e-4 and
whose gripper command did not change is a no-op and is dropped; a demonstration whose replay
never satisfies the task's own success check is dropped. Both counts are written
to meta.json, so the tree is auditable against the community's cleaned version.

Usage:
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python scripts/libero_gen.py \
      --suite libero_spatial --task-index 0 --demos 0-49 --out data/libero_spatial
"""
import argparse, json, os, re, sys, glob, zlib
from pathlib import Path

import h5py
import imageio
import numpy as np

RES = 512
FOVY = 55.0
N_VIEWS = 4
FPS = 20
CAMS = ["agentview", "frontview", "birdview", "sideview"]   # LIBERO's own, repurposed as view1..4
HELDOUT_EVERY = 10          # every 10th successful demo -> variation1 (10%, the ManiSkill ratio)
SKY_MAX_FRAC, FAR_MAX_FRAC, NEAR_MAX_FRAC = 0.005, 0.02, 0.005
OUT_OF_CODEC_MAX_FRAC = 0.03
MAX_CAMERA_RETRIES = 8
FLIP_CV_RLB = np.diag([-1.0, -1.0, 1.0, 1.0])
DEMO_ROOT = "/workspace/1228_tingting/libero_demos"


def sample_cameras(rng, center):
    """Identical to scripts/maniskill3_gen.py:sample_cameras -- same distribution, same draws."""
    cams = []
    for _ in range(N_VIEWS):
        r = rng.uniform(0.55, 0.95)
        elev = np.deg2rad(rng.uniform(42.0, 72.0))
        azim = np.deg2rad(rng.uniform(-150.0, 150.0))
        eye = center + r * np.array([np.cos(elev) * np.cos(azim),
                                     np.cos(elev) * np.sin(azim), np.sin(elev)])
        target = center + rng.normal(0.0, 0.03, 3)
        cams.append({"eye": eye.tolist(), "target": target.tolist(), "radius": round(float(r), 4),
                     "elev_deg": round(float(np.rad2deg(elev)), 1),
                     "azim_deg": round(float(np.rad2deg(azim)), 1)})
    return cams


def look_at_wxyz(eye, target):
    """MuJoCo camera frame: looks down -z, +y up, +x right. Returns the wxyz quaternion."""
    import robosuite.utils.transform_utils as TU
    f = np.asarray(target, float) - np.asarray(eye, float); f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    q = TU.mat2quat(np.stack([r, u, -f], axis=1))      # xyzw
    return np.array([q[3], q[0], q[1], q[2]])


def place_cameras(sim, cams):
    for name, c in zip(CAMS, cams):
        cid = sim.model.camera_name2id(name)
        assert sim.model.cam_bodyid[cid] == 0, f"{name} is not world-attached"
        sim.model.cam_pos[cid] = c["eye"]
        sim.model.cam_quat[cid] = look_at_wxyz(c["eye"], c["target"])
        sim.model.cam_fovy[cid] = FOVY
    sim.forward()


def rlb_camera_params(sim, name):
    import robosuite.utils.camera_utils as CU
    E_cv = CU.get_camera_extrinsic_matrix(sim, name)                 # cam2world, OpenCV
    K_cv = CU.get_camera_intrinsic_matrix(sim, name, RES, RES)
    K = K_cv.astype(np.float64).copy(); K[0, 0] *= -1.0; K[1, 1] *= -1.0
    return {"extrinsics": E_cv @ FLIP_CV_RLB, "intrinsics": K, "E_cv": E_cv, "K_cv": K_cv}


def self_check_geometry(p, depth_m, rng):
    """Unproject through the WRITTEN params, reproject through the native ones (Sec. contract)."""
    K, C, Kcv, Ecv = p["intrinsics"], p["extrinsics"], p["K_cv"], p["E_cv"]
    vv, uu = np.where((depth_m > 0.05) & (depth_m < 10.0))
    if len(uu) < 100:
        raise RuntimeError("geometry self-check: almost no valid depth")
    sel = rng.choice(len(uu), 100, replace=False)
    u, v = uu[sel].astype(float), vv[sel].astype(float); z = depth_m[vv[sel], uu[sel]].astype(float)
    pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
    pw = (C[:3, :3] @ pc.T + C[:3, 3:4]).T
    w2c = np.linalg.inv(Ecv); q = (w2c[:3, :3] @ pw.T + w2c[:3, 3:4]).T
    assert (q[:, 2] > 0).all(), "unprojected points behind the camera"
    u2 = Kcv[0, 0] * q[:, 0] / q[:, 2] + Kcv[0, 2]; v2 = Kcv[1, 1] * q[:, 1] / q[:, 2] + Kcv[1, 2]
    err = float(np.max(np.hypot(u2 - u, v2 - v)))
    if err >= 0.51:
        raise RuntimeError(f"geometry self-check failed: {err:.3f} px")
    return err


def self_check_orientation(p, mask0, env, instances):
    """Project each movable object's centre through the WRITTEN params onto the WRITTEN mask.

    The geometry self-check cannot see an upside-down image array: both of its parameter sets
    come from the same matrices. This one uses exactly what reaches disk -- the RLBench-
    convention K/C and the flipped mask -- so a wrong flip, or a robosuite whose image
    convention changed, fails here instead of silently producing mirrored training data.
    Measured on the first probe: 53/63 flipped vs 12/63 as-is, so >=50% is a wide margin.
    """
    K, C = p["intrinsics"], p["extrinsics"]
    w2c = np.linalg.inv(C)
    od = env.env.objects_dict
    hits = n = 0
    for name in od:
        if name not in instances:
            continue
        pw = env.sim.data.body_xpos[env.sim.model.body_name2id(od[name].root_body)]
        pc = w2c[:3, :3] @ pw + w2c[:3, 3]
        if pc[2] <= 0.05:
            continue
        u = K[0, 0] * pc[0] / pc[2] + K[0, 2]; v = K[1, 1] * pc[1] / pc[2] + K[1, 2]
        iu, iv = int(round(u)), int(round(v))
        if not (0 <= iu < RES and 0 <= iv < RES):
            continue
        n += 1
        hits += int(mask0[iv, iu] == instances.index(name) + 1)
    return hits, n


def gate_orientation(per_view):
    """Judge the four views TOGETHER. Per view, a correctly oriented but cluttered or occluded
    view can legitimately drop to half (measured: 4/8 on libero_10 LIVING_ROOM_SCENE2, view3,
    camera behind the arm) -- a per-view gate would reject good episodes. Pooled, correct
    orientation measured 53/63 and 25/32 while a wrong flip measured 12/63, so 0.5 on the pool
    separates them with a wide margin."""
    h = sum(a for a, _ in per_view); n = sum(b for _, b in per_view)
    if n >= 4 and h / n < 0.5:
        raise RuntimeError(f"orientation self-check failed: {h}/{n} object centres land on "
                           f"their own mask -- image flip or camera convention is wrong")


def localize_model_xml(xml):
    """Point the demo's stored MJCF at THIS machine's assets, and prove every file exists.

    Each demo stores the exact scene it was recorded in (attrs["model_file"]), with absolute
    asset paths from the author's machine: /Users/yifengz/workspace/robosuite-master/robosuite/
    models/assets/... and, for LIBERO's own objects, an old dev tree
    /Users/yifengz/workspace/libero-dev/chiliocosm/assets/... . LIBERO's own
    postprocess_model_xml rewrites the second kind only when a path component is literally
    "libero" (it is "libero-dev"/"chiliocosm" here) and only relative to the current working
    directory -- so it cannot be used as-is. Missing a file must fail loudly: MuJoCo would
    otherwise refuse the model, or worse, a renamed mesh could load a different object.
    """
    import robosuite
    from libero.libero import get_libero_path
    rs = os.path.join(os.path.dirname(robosuite.__file__), "models", "assets")
    lib = get_libero_path("assets")
    roots = (("/robosuite/models/assets/", rs), ("/chiliocosm/assets/", lib),
             ("/libero/libero/assets/", lib), ("/libero/assets/", lib))
    missing = []

    def fix(m):
        old = m.group(1)
        for key, root in roots:
            if key in old:
                new = os.path.join(root, old.split(key, 1)[1])
                if not os.path.exists(new):
                    missing.append(new)
                return f'file="{new}"'
        return m.group(0)
    out = re.sub(r'file="([^"]+)"', fix, xml)
    if missing:
        raise RuntimeError(f"{len(missing)} scene assets not found locally, e.g. {missing[:3]}")
    return out


def load_demo_scene(env, model_xml):
    """Rebuild the EXACT scene the demo was recorded in, before any state is written into it.

    This is the procedure LIBERO's and robomimic's own replay scripts follow, and it guarantees
    the joint layout the flattened state is written into is the recording's own. Measured: it
    made NO difference on 15 demos across 5 tasks in all four suites (identical success and
    identical end-effector trajectories with and without it), so env.reset() happens to build
    the same model today. It is kept because nothing guarantees that for every scene variant,
    at the cost of ~1 s per demo -- not because it fixed anything.
    """
    env.env.reset_from_xml_string(localize_model_xml(model_xml))
    env.env.sim.reset()


def render_views(env):
    """-> {view: (rgb[H,W,3] u8, depth_m[H,W] f32, seg[H,W] u16)}, ALL flipped to OpenCV rows."""
    import robosuite.utils.camera_utils as CU
    obs = env.env._get_observations(force_update=True)
    out = {}
    for i, c in enumerate(CAMS):
        rgb = obs[f"{c}_image"][::-1].copy()
        depth = CU.get_real_depth_map(env.sim, obs[f"{c}_depth"])[..., 0][::-1].astype(np.float32).copy()
        seg = obs[f"{c}_segmentation_instance"][..., 0][::-1].astype(np.uint16).copy()
        out[f"view{i + 1}"] = (rgb, depth, seg)
    return out


# ------------------------------------------------------------------ segmentation roles
ROBOT_ARM = re.compile(r"(Panda\d*|Mount\d*)$")
GRIPPER = re.compile(r"Gripper\d*$")


def owner_of(region, instances):
    """'basket_1_contain_region' -> 'basket_1' (longest instance name that prefixes it)."""
    best = None
    for n in instances:
        if region == n or region.startswith(n + "_"):
            if best is None or len(n) > len(best):
                best = n
    return best


def build_seg_metadata(env, task):
    e = env.env
    instances = list(e.model.instances_to_ids.keys())      # seg value v -> instances[v-1]
    goal = e.parsed_problem.get("goal_state", []) if isinstance(e.parsed_problem, dict) else []
    targets, goals = set(), set()
    for pred in goal:
        op, args = str(pred[0]).lower(), [str(a) for a in pred[1:]]
        if op in ("on", "in") and len(args) == 2:
            t, g = owner_of(args[0], instances), owner_of(args[1], instances)
            if t: targets.add(t)
            if g and g != t: goals.add(g)
        elif op in ("open", "close", "turnon", "turnoff") and args:
            t = owner_of(args[0], instances)
            if t: targets.add(t)
    fixtures = set(getattr(e, "fixtures_dict", {}).keys())
    role_of, instances_json, name_to_ids = {0: "background"}, {}, {}
    for i, n in enumerate(instances):
        sid = i + 1
        if GRIPPER.search(n):          role = "gripper"
        elif ROBOT_ARM.search(n):      role = "robot_arm"
        elif n in goals:               role = "goal"
        elif n in targets:             role = "distractor"   # promoted to target via referring
        elif n in fixtures:            role = "fixture"
        else:                          role = "distractor"
        role_of[sid] = role
        name_to_ids[n] = [sid]
        instances_json[n] = {"base_role": role, "handles": [sid]}
    instances_json["__background__"] = {"base_role": "background", "handles": [0]}
    scene_segments = {"instances": instances_json, "protocol": "rlbench-scene-role-v1"}
    referring = {t: {"handles": name_to_ids[t]} for t in sorted(targets)}
    if not referring:
        raise RuntimeError(f"{task.name}: BDDL goal names no manipulated object: {goal}")
    seg_targets = {"referring": referring, "referring_by_color": {},
                   "protocol": "rlbench-task-source-v1", "task": task.name}
    handles_json = {str(i + 1): n for i, n in enumerate(instances)}
    handles_json["0"] = "background"
    return handles_json, scene_segments, seg_targets, role_of


# ------------------------------------------------------------------ actions
def pose_row(env):
    """[x,y,z, qx,qy,qz,qw, openness] of the grip site -- the TCP, like ManiSkill's tcp pose."""
    o = env.env._get_observations(force_update=False)
    p = np.asarray(o["robot0_eef_pos"], float)
    q = np.asarray(o["robot0_eef_quat"], float)            # robosuite obs quats are xyzw
    g = np.asarray(o["robot0_gripper_qpos"], float)        # [+x, -x], each finger 0..0.04
    openness = float(np.clip((g[0] - g[1]) / 0.08, 0.0, 1.0))
    return np.concatenate([p, q / np.linalg.norm(q), [openness]])


def noop_mask(actions):
    """openvla regenerate_libero_dataset: arm action ~0 AND gripper command unchanged."""
    keep = np.ones(len(actions), bool)
    for t in range(len(actions)):
        prev_g = actions[t - 1, -1] if t > 0 else None
        if np.linalg.norm(actions[t, :-1]) < 1e-4 and (prev_g is None or actions[t, -1] == prev_g):
            keep[t] = False
    return keep


# ------------------------------------------------------------------ one episode
def render_episode(env, task, states, actions, seed, out_dir, desc, model_xml):
    load_demo_scene(env, model_xml)
    keep = noop_mask(actions)
    idx = np.nonzero(keep)[0]
    if len(idx) < 10:
        raise RuntimeError(f"only {len(idx)} non-no-op steps")
    # SUCCESS = REACHED DURING THE REPLAY, NOT TRUE AT THE LAST FRAME. A demonstration keeps
    # recording for a step or two after the goal is met, while the gripper releases and the
    # object settles, and the task's check can go false again in those frames. Measured on
    # libero_spatial task 6 demo 0: success on replayed frames 112-120, false on 121-122, and
    # the HDF5 itself records reward 1 -- a last-frame check dropped a good demonstration.
    succ = []
    for t in idx:
        env.set_init_state(states[t]); succ.append(bool(env.env._check_success()))
    if not any(succ):
        return None, "the replay never satisfies the task's success check"
    first_success = int(np.argmax(succ))
    handles_json, scene_segments, seg_targets, role_of = build_seg_metadata(env, task)
    od = env.env.objects_dict
    center = np.mean([env.sim.data.body_xpos[env.sim.model.body_name2id(od[n].root_body)]
                      for n in od], axis=0)
    cam_rng = np.random.default_rng(zlib.crc32(f"{task.name}|{seed}".encode()) & 0x7FFFFFFF)
    for episode_attempt in range(3):
        for _ in range(MAX_CAMERA_RETRIES):
            cams = sample_cameras(cam_rng, center)
            env.set_init_state(states[idx[0]]); place_cameras(env.sim, cams)
            v0 = render_views(env)
            worst = {"sky": 0.0, "far": 0.0, "near": 0.0}
            for _, d, _ in v0.values():
                worst["sky"] = max(worst["sky"], float((d <= 0).mean()))
                worst["far"] = max(worst["far"], float((d > 9.5).mean()))
                worst["near"] = max(worst["near"], float(((d > 0) & (d < 0.06)).mean()))
            if worst["sky"] <= SKY_MAX_FRAC and worst["far"] <= FAR_MAX_FRAC and worst["near"] <= NEAR_MAX_FRAC:
                break
        else:
            raise RuntimeError(f"no acceptable cameras after {MAX_CAMERA_RETRIES} tries ({worst})")
        params = {f"view{i + 1}": rlb_camera_params(env.sim, c) for i, c in enumerate(CAMS)}
        frames = {v: ([], [], []) for v in params}; poses = []; seen = set()
        for t in idx:
            env.set_init_state(states[t])
            for v, (rgb, d, s) in render_views(env).items():
                frames[v][0].append(rgb); frames[v][1].append(d.astype(np.float16)); frames[v][2].append(s)
                seen.update(np.unique(s).tolist())
            poses.append(pose_row(env))
        unmapped = sorted(i for i in seen if i not in role_of)
        if unmapped:
            raise RuntimeError(f"rendered segmentation ids with no role: {unmapped}")
        err = self_check_geometry(params["view1"], frames["view1"][1][0].astype(np.float32), cam_rng)
        env.set_init_state(states[idx[0]])       # object poses of frame 0, the mask checked below
        inst_names = list(env.env.model.instances_to_ids.keys())
        orient = [self_check_orientation(params[v], frames[v][2][0], env, inst_names) for v in params]
        gate_orientation(orient)
        out_frac = {v: float(((np.stack(frames[v][1]).astype(np.float32) <= 0.05) |
                              (np.stack(frames[v][1]).astype(np.float32) >= 10.0)).mean()) for v in frames}
        if max(out_frac.values()) <= OUT_OF_CODEC_MAX_FRAC:
            break
    else:
        raise RuntimeError(f"depth outside codec range after 3 camera draws: {out_frac}")

    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(idx)
    for v, p in params.items():
        vd = out_dir / v; (vd / "rgb").mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(str(vd / "rgb" / "video.mp4"), fps=FPS, codec="libx264",
                                quality=8, macro_block_size=1) as w:
            for f in frames[v][0]:
                w.append_data(f)
        np.savez_compressed(vd / "depth.npz", depth=np.stack(frames[v][1]))
        np.savez_compressed(vd / "mask.npz", mask=np.stack(frames[v][2]))
        with open(vd / "camera_params.json", "w") as f:
            json.dump({str(t): {"intrinsics": p["intrinsics"].tolist(),
                                "extrinsics": p["extrinsics"].tolist()} for t in range(n)}, f)
    poses = np.stack(poses)
    assert np.abs(np.linalg.norm(poses[:, 3:7], axis=1) - 1).max() < 1e-3
    np.save(out_dir / "actions.npy", poses)
    meta = {"seed": int(seed), "steps": int(n), "desc": [desc], "source": "libero_human_teleop",
            "suite": getattr(task, "problem_folder", ""), "task": task.name, "success": True,
            "noop_steps_dropped": int((~keep).sum()), "raw_steps": int(len(actions)),
            "first_success_frame": first_success, "success_frames": int(sum(succ)),
            "cameras": {f"view{i + 1}": {k: cams[i][k] for k in ("radius", "elev_deg", "azim_deg")}
                        for i in range(N_VIEWS)},
            "fovy_deg": FOVY, "image_convention": "opencv (flipped from robosuite opengl)",
            "geometry_check_px": round(err, 4),
            "orientation_check": {v: f"{h}/{n}" for v, (h, n) in zip(params, orient)},
            "depth_out_of_codec_frac": {v: round(f, 5) for v, f in out_frac.items()}}
    for name, obj in (("meta.json", meta), ("handles.json", handles_json),
                      ("scene_segments.json", scene_segments), ("seg_targets.json", seg_targets)):
        with open(out_dir / name, "w") as f:
            json.dump(obj, f, indent=1 if name != "meta.json" else None)
    return n, None


def parse_range(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-"); out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task-index", type=int, required=True)
    ap.add_argument("--demos", default="0-49")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    task = benchmark.get_benchmark_dict()[a.suite]().get_task(a.task_index)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    h5 = h5py.File(os.path.join(DEMO_ROOT, a.suite, f"{task.name}_demo.hdf5"), "r")
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RES, camera_widths=RES,
                             camera_names=CAMS, camera_depths=True, camera_segmentations="instance")
    env.seed(0); env.reset()
    kept = 0
    for d in parse_range(a.demos):
        key = f"data/demo_{d}"
        if key not in h5:
            continue
        var = 1 if (d % HELDOUT_EVERY == HELDOUT_EVERY - 1) else 0
        out_dir = Path(a.out) / task.name / f"variation{var}" / "episodes" / f"episode{d}"
        if (out_dir / "meta.json").exists():
            kept += 1; continue
        try:
            n, why = render_episode(env, task, h5[key]["states"][()], h5[key]["actions"][()],
                                    d, out_dir, task.language, h5[key].attrs["model_file"])
        except Exception as ex:
            print(f"  demo {d}: SKIP {type(ex).__name__}: {ex}", flush=True); continue
        if n is None:
            print(f"  demo {d}: DROP ({why})", flush=True); continue
        kept += 1
        print(f"  demo {d} -> variation{var}/episode{d}: {n} frames", flush=True)
    env.close()
    print(f"DONE {a.suite}/{task.name}: {kept} episodes", flush=True)


if __name__ == "__main__":
    main()
