"""Multiprocess RLBench self-gen dataset generator (ActionImages layout + GT depth/mask
+ per-frame cameras + Colosseum-lite background augmentation).

Layout per episode (ALL variations, matching official RLBenchMVDataset glob("variation*")):
  {out}/{task}/variation{v}/episodes/episode{seed}/    for v in range(task.variation_count())
    view{1..4}/rgb/video.mp4, view{1..4}/depth.npz, mask.npz, camera_params.json
    actions.npy [T,8] (world xyz + quat + gripper_open), meta.json
  {out}/{task}/variation{v}/variation_descriptions.pkl

Seeds: np.random.seed(seed) before each reset → recorded in meta for reproducibility.
Background aug: procedural texture pool applied to floor/walls with prob 0.8 (official
recipe uses Robot-Colosseum textures; this is a procedural approximation, logged per episode).

Usage (single worker; the launcher spawns many):
  python gen_dataset.py --tasks open_drawer,close_jar --seeds 0-9 --out .../rlbench_selfgen
"""
import os, sys, json, time, glob, pickle, argparse, traceback, zlib
import numpy as np

OUT_DEFAULT = "/workspace/ttdu/ttd/data/rlbench_selfgen"
RES = [256, 256]
VIEWS = {"view1": "front", "view2": "overhead", "view3": "left_shoulder", "view4": "right_shoulder"}
TEX_DIR = "/workspace/ttdu/ttd/data/texture_pool"


def build_texture_pool(n=48, size=256):
    """Procedural texture pool: noise / checker / stripes / gradients in varied colors."""
    os.makedirs(TEX_DIR, exist_ok=True)
    if len(glob.glob(os.path.join(TEX_DIR, "*.png"))) >= n:
        return sorted(glob.glob(os.path.join(TEX_DIR, "*.png")))
    import imageio
    rng = np.random.RandomState(1234)
    paths = []
    for i in range(n):
        kind = i % 4
        c1, c2 = rng.randint(30, 225, 3), rng.randint(30, 225, 3)
        yy, xx = np.mgrid[0:size, 0:size]
        if kind == 0:      # low-freq noise
            base = rng.rand(8, 8)
            import cv2
            m = cv2.resize(base, (size, size), interpolation=cv2.INTER_CUBIC)[..., None]
        elif kind == 1:    # checker
            k = rng.choice([16, 32, 64])
            m = (((yy // k) + (xx // k)) % 2)[..., None].astype(float)
        elif kind == 2:    # stripes
            k = rng.choice([8, 16, 32]); ang = rng.rand() * np.pi
            m = ((np.sin((xx * np.cos(ang) + yy * np.sin(ang)) / k * np.pi) > 0))[..., None].astype(float)
        else:              # gradient
            m = (xx / size * rng.rand() + yy / size * (1 - rng.rand()))[..., None]
            m = (m - m.min()) / (m.ptp() + 1e-6)
        img = (c1[None, None] * m + c2[None, None] * (1 - m)).astype(np.uint8)
        p = os.path.join(TEX_DIR, f"tex_{i:03d}.png")
        imageio.imwrite(p, img)
        paths.append(p)
    return paths


def apply_background_aug(pr, rng, tex_paths):
    """Colosseum-lite: random texture or color on floor/walls. Returns aug metadata."""
    from pyrep.objects.shape import Shape
    from pyrep.const import TextureMappingMode
    meta = {"applied": False}
    targets = []
    for name in ["Floor", "Wall1", "Wall2", "Wall3", "Wall4", "diningTable_visible"]:
        try:
            targets.append((name, Shape(name)))
        except Exception:
            pass
    if rng.rand() < 0.2 or not targets:
        for _, shape in targets:      # restore defaults (clear stale texture from prev episode)
            try:
                shape.remove_texture()
            except Exception:
                pass
        return meta
    tex_file = tex_paths[rng.randint(len(tex_paths))]
    meta.update(applied=True, texture=os.path.basename(tex_file), targets=[])
    try:
        _, texture = pr.create_texture(tex_file)
        for name, shape in targets:
            if name == "diningTable_visible" and rng.rand() < 0.5:
                continue  # keep table default half the time
            try:
                shape.set_texture(texture, TextureMappingMode.PLANE, repeat_along_u=True,
                                  repeat_along_v=True, uv_scaling=[1.0, 1.0])
                meta["targets"].append(name)
            except Exception:
                pass
    except Exception as e:
        meta["texture_error"] = str(e)
    return meta


def episode_done(ep_dir):
    return os.path.isfile(os.path.join(ep_dir, "meta.json"))


def dump_handle_names(ep_dir, demo):
    """mask id -> object name map (needed for referring-seg targets)."""
    from pyrep.backend import sim as simb
    ids = set()
    idxs = [0, len(demo) // 2, len(demo) - 1]
    for i in idxs:
        for cam in VIEWS.values():
            m = getattr(demo[i], f"{cam}_mask")
            ids.update(int(x) for x in np.unique(m))
    names = {}
    for h in sorted(ids):
        try:
            names[h] = simb.simGetObjectName(h)
        except Exception:
            names[h] = ""
    with open(os.path.join(ep_dir, "handles.json"), "w") as f:
        json.dump(names, f)


def save_episode(ep_dir, demo, aug_meta, seed, desc):
    import imageio
    os.makedirs(ep_dir, exist_ok=True)
    dump_handle_names(ep_dir, demo)
    actions = np.stack([np.concatenate([o.gripper_pose, [o.gripper_open]]) for o in demo])
    np.save(os.path.join(ep_dir, "actions.npy"), actions)
    for view, cam in VIEWS.items():
        vdir = os.path.join(ep_dir, view)
        os.makedirs(os.path.join(vdir, "rgb"), exist_ok=True)
        rgbs = [getattr(o, f"{cam}_rgb") for o in demo]
        imageio.mimwrite(os.path.join(vdir, "rgb", "video.mp4"), rgbs, fps=20, quality=8)
        np.savez_compressed(os.path.join(vdir, "depth.npz"),
                            depth=np.stack([getattr(o, f"{cam}_depth") for o in demo]).astype(np.float16))
        np.savez_compressed(os.path.join(vdir, "mask.npz"),
                            mask=np.stack([getattr(o, f"{cam}_mask") for o in demo]).astype(np.uint16))
        cam_params = {str(i): {
            "intrinsics": o.misc[f"{cam}_camera_intrinsics"].tolist(),
            "extrinsics": o.misc[f"{cam}_camera_extrinsics"].tolist(),
        } for i, o in enumerate(demo)}
        with open(os.path.join(vdir, "camera_params.json"), "w") as f:
            json.dump(cam_params, f)
    with open(os.path.join(ep_dir, "meta.json"), "w") as f:
        json.dump({"seed": seed, "steps": len(demo), "aug": aug_meta,
                   "desc": desc, "ts": time.strftime("%F %T")}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=str, required=True, help="comma-separated snake_case task names")
    ap.add_argument("--seeds", type=str, required=True, help="e.g. 0-99 or 0,3,7")
    ap.add_argument("--out", type=str, default=OUT_DEFAULT)
    ap.add_argument("--max_attempts", type=int, default=3)
    ap.add_argument("--variations", type=str, default="all",
                    help="'all' (default, unchanged behaviour) or a comma/range list of variation "
                         "indices, e.g. '0' or '0,1' or '0-2'. Use '0' to deepen variation0 only: "
                         "the train/test split trains on variation0 and holds out the rest, so "
                         "generating every variation at a large --seeds range would spend most of "
                         "the compute on held-out data.")
    # Square render resolution for all 4 views (rgb AND depth AND mask -- RLBench renders them
    # from one CameraConfig). 256 is the v2 tree; 512 is what the official ActionImages release
    # and its checkpoint were trained at. Measured 2026-08-15 on close_jar/variation0: 512 costs
    # 1.40x wall time per episode (149s vs 107s, same worker config) and ~2.6x disk -- NOT 4x,
    # because the bottleneck is motion planning + physics stepping, not rasterisation.
    # Camera intrinsics follow automatically: RLBench derives them from image_size, so 512 gives
    # cx=cy=256 and a doubled focal length with no further change here.
    ap.add_argument("--res", type=int, default=256,
                    help="square render resolution, default 256 (the v2 tree). Use 512 to match "
                         "the official ActionImages training resolution.")
    # 'procedural' is the v2/512 behaviour: background walls only, from the local texture pool.
    # 'colosseum' additionally randomises camera pose, table colour/texture and light colour,
    # reproducing what measurement shows the official release does. See colosseum_aug.py for
    # what is deliberately left out (object colour, distractors, physics) and why.
    ap.add_argument("--aug", choices=["procedural", "colosseum", "none"], default="procedural",
                    help="domain randomisation. 'procedural' = background walls only (v2 "
                         "behaviour); 'colosseum' = + camera pose / table / lights.")
    args = ap.parse_args()

    global RES
    RES = [args.res, args.res]

    def _parse_int_list(spec):
        out = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-")
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
        return out

    want_variations = None if args.variations == "all" else set(_parse_int_list(args.variations))

    if "-" in args.seeds:
        a, b = args.seeds.split("-"); seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]
    task_names = args.tasks.split(",")

    tex_paths = build_texture_pool()

    import re
    import rlbench.tasks as T
    from rlbench.environment import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.observation_config import ObservationConfig, CameraConfig

    camel = {re.sub(r"(?<!^)(?=[A-Z])", "_", c).lower(): getattr(T, c) for c in dir(T) if c[0].isupper()}

    cam_on = CameraConfig(rgb=True, depth=True, mask=True, image_size=RES, depth_in_meters=True)
    cam_off = CameraConfig(rgb=False, depth=False, mask=False, point_cloud=False)
    obs_config = ObservationConfig(front_camera=cam_on, overhead_camera=cam_on,
                                   left_shoulder_camera=cam_on, right_shoulder_camera=cam_on,
                                   wrist_camera=cam_off)
    obs_config.joint_positions = True; obs_config.gripper_pose = True; obs_config.gripper_open = True

    env = Environment(MoveArmThenGripper(JointVelocity(), Discrete()), obs_config=obs_config, headless=True)
    env.launch()
    # bind() resolves cameras/lights once, after launch: the vision sensors' ORIGINAL poses are
    # the reference the per-episode orbit is computed from, so they must be read before the
    # first randomisation moves them.
    colosseum = None
    if args.aug == "colosseum":
        from colosseum_aug import ColosseumAug
        colosseum = ColosseumAug(env._pyrep)
        colosseum.bind()
    n_ok = n_fail = n_skip = 0
    try:
        for tn in task_names:
            task = env.get_task(camel[tn])
            # Official RLBenchMVDataset globs variation* (ALL variations) — replicate that here.
            # variation_count() is task-specific (1..60); generate every variation, `seeds` demos each.
            # (Do NOT hardcode set_variation(0): that silently narrows to a single variation.)
            n_var = task.variation_count()
            # `want_variations` may name indices this task does not have (variation_count is
            # task-specific, 1..60) -- intersect rather than trust the CLI.
            var_list = [v for v in range(n_var) if want_variations is None or v in want_variations]
            print(f"[gen] {tn}: {n_var} variations, generating {var_list} x {len(seeds)} demos",
                  flush=True)
            for v in var_list:
                var_dir = os.path.join(args.out, tn, f"variation{v}")
                os.makedirs(os.path.join(var_dir, "episodes"), exist_ok=True)
                desc_p = os.path.join(var_dir, "variation_descriptions.pkl")
                for seed in seeds:
                    ep_dir = os.path.join(var_dir, "episodes", f"episode{seed}")
                    if episode_done(ep_dir):
                        n_skip += 1; continue
                    # seed the RNG per (task, variation, seed) so different variations get distinct
                    # background augs even at the same seed index.
                    # crc32, NOT hash(): Python salts str/tuple hashing per process unless
                    # PYTHONHASHSEED is pinned, so the original `hash((tn, v, seed))` made the
                    # background aug UNREPRODUCIBLE across runs -- verified 2026-08-15 by
                    # re-generating close_jar/variation0 seeds 0-3, which came back with different
                    # textures and even flipped `applied` False->True, while actions.npy and
                    # handles.json were bit-identical. crc32 is stable across processes, so a tree
                    # generated now can be regenerated later pixel-for-pixel.
                    rng = np.random.RandomState(zlib.crc32(f"{tn}|{v}|{seed}".encode()) & 0x7FFFFFFF)
                    ok = False
                    for attempt in range(args.max_attempts):
                        try:
                            np.random.seed(seed + v * 1_000_003 + attempt * 100003)
                            task.set_variation(v)
                            if colosseum is not None:
                                # Same crc32 stream as the procedural path, so the two aug modes
                                # are independently reproducible from (task, variation, seed).
                                colosseum.restore_cameras()
                                aug_meta = colosseum.randomize(
                                    np.random.default_rng(
                                        zlib.crc32(f"col|{tn}|{v}|{seed}".encode()) & 0x7FFFFFFF
                                    ),
                                    task_name=tn,
                                )
                            elif args.aug == "none":
                                aug_meta = {"applied": False}
                            else:
                                aug_meta = apply_background_aug(env._pyrep, rng, tex_paths)
                            if not os.path.isfile(desc_p):        # once per (task, variation)
                                desc, _ = task.reset()
                                with open(desc_p, "wb") as f:
                                    pickle.dump(list(desc), f)
                            with open(desc_p, "rb") as f:
                                desc = pickle.load(f)
                            demos = task.get_demos(1, live_demos=True, max_attempts=1)
                            save_episode(ep_dir, demos[0], aug_meta, seed, list(desc))
                            ok = True; break
                        except Exception:
                            traceback.print_exc()
                            time.sleep(1)
                    if ok:
                        n_ok += 1
                        print(f"[gen] {tn} var={v} seed={seed} OK ({n_ok} done)", flush=True)
                    else:
                        n_fail += 1
                        print(f"[gen] {tn} var={v} seed={seed} FAILED", flush=True)
    finally:
        env.shutdown()
    print(f"GEN_WORKER_DONE ok={n_ok} fail={n_fail} skip={n_skip}", flush=True)


if __name__ == "__main__":
    main()
