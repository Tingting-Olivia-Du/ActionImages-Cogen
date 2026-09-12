"""Ground-truth replay: the harness must solve the task from recorded actions before any
generated action is worth measuring.

This is the load-bearing self-check of EVAL_PLAN_CLOSEDLOOP.md. It costs zero GPU seconds --
no model is loaded -- yet it exercises the whole rollout path at once:

  * scene setup       -- `reset_to_new_demo` must land on the demo's own initial state;
  * camera rig        -- live intrinsics/extrinsics must match what gen_dataset wrote to disk,
                         including the negative focal lengths;
  * action mode       -- the demo's end-effector poses must be executable via IK;
  * success detection -- replaying a demo that solved the task must be scored as a success;
  * determinism       -- the same scene seed must produce the same scene, or the paired
                         McNemar test across arms is invalid.

A task that cannot clear this bar is not measurable: a 0% closed-loop rate on it would be the
harness's number, not the model's. `--min-success-rate` is therefore a gate, and the per-task
results are written to reports/closedloop/gt_replay.json so the rollout campaign can restrict
itself to tasks that passed.

Needs CoppeliaSim, so it is NOT in scripts/run_tests.sh (that suite is pure CPU, no simulator).
Run: bash scripts/run_rollout_tests.sh
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.rollout_env import VIEW_CAMERAS, RolloutEnv, episode_seed_to_rng_seed
from eval.rollout import interpolate_chunk, trial_seed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 512_aug, not the bare "rlbench_selfgen" symlink: that one points at the deleted 256 v2 tree
# and DANGLES, so this check has been skipping itself silently. SELFGEN_TEST_DATA overrides.
SELFGEN = os.environ.get("SELFGEN_TEST_DATA", os.path.join(REPO, "data", "rlbench_selfgen_512_aug"))

DEFAULT_TASKS = ["push_buttons", "open_drawer", "meat_off_grill", "close_jar"]
CAM_TOL = 1e-6
START_TOL_M = 1e-3


# How close the live rig must sit to SOME training camera on a domain-randomised tree, and how
# far out in the pose cloud it is allowed to sit. 10 cm is two orders of magnitude below the
# 0.96 m median pairwise spacing of the augmented cameras, so it still catches a genuinely
# different rig; 0.60 means the live rig must not be more eccentric than 60% of training poses.
AUG_NN_TOL_M = 0.10
AUG_MAX_ECCENTRICITY = 0.60


def _tree_is_augmented(root):
    """True if this tree carries Colosseum-style randomisation (camera pose included)."""
    for ep in sorted(glob.glob(os.path.join(root, "*", "variation0", "episodes", "episode*")))[:1]:
        try:
            with open(os.path.join(ep, "meta.json")) as f:
                return bool(json.load(f).get("aug", {}).get("applied"))
        except OSError:
            return False
    return False


def _tree_camera_positions(root, view):
    pos = []
    for ep in sorted(glob.glob(os.path.join(root, "*", "variation0", "episodes", "episode*"))):
        p = os.path.join(ep, view, "camera_params.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            cam = json.load(f)
        k = min(cam, key=lambda x: int(x))
        pos.append(np.asarray(cam[k]["extrinsics"], dtype=np.float64)[:3, 3])
    return np.stack(pos) if pos else None


def check_camera_rig(env, obs):
    """The live rig must be the one the model was trained to expect.

    On a FIXED rig that means matrix equality. On a domain-randomised tree
    (rlbench_selfgen_512_aug randomises camera radius/elevation/azimuth per episode -- measured
    position std 0.43/0.69/0.23 m, median pairwise spacing 0.96 m) matrix equality is guaranteed
    to fail and says nothing: the live rig is ONE pose and the tree is a cloud of 788.

    The meaningful question there is whether the live pose is INSIDE that cloud, because that is
    what "in distribution" means for a Pluecker-conditioned model. Measured for the current rig:
    nearest training camera 1.6 cm (view1) / 1.9 cm (view2), and 92.4% of training cameras sit
    farther from the cloud centroid than it does -- comfortably interior.

    Intrinsics are NOT randomised and are still pinned exactly, after scaling for the render
    resolution (fx scales linearly: -703.35 at 512 -> -351.68 at 256).
    """
    extr, intr = env.camera_params(obs)
    assert extr.shape == (2, 4, 4) and intr.shape == (2, 3, 3), (extr.shape, intr.shape)

    ref = None
    for task in sorted(os.listdir(SELFGEN)):
        cand = os.path.join(SELFGEN, task, "variation0", "episodes", "episode0")
        if os.path.isdir(cand):
            ref = cand
            break
    assert ref is not None, f"no episodes under {SELFGEN} to pin the rig against"
    augmented = _tree_is_augmented(SELFGEN)

    for vi, cam in enumerate(VIEW_CAMERAS):
        with open(os.path.join(ref, f"view{vi+1}", "camera_params.json")) as f:
            disk = json.load(f)
        d0 = disk[sorted(disk, key=int)[0]]
        de = np.asarray(d0["extrinsics"], dtype=np.float64)
        di = np.asarray(d0["intrinsics"], dtype=np.float64)

        if augmented:
            cloud = _tree_camera_positions(SELFGEN, f"view{vi+1}")
            assert cloud is not None, f"no camera_params under {SELFGEN} for view{vi+1}"
            d = np.linalg.norm(cloud - extr[vi][:3, 3], axis=1)
            centroid = cloud.mean(0)
            ecc = float((np.linalg.norm(cloud - centroid, axis=1)
                         < np.linalg.norm(extr[vi][:3, 3] - centroid)).mean())
            assert d.min() < AUG_NN_TOL_M, (
                f"view{vi+1} ({cam}) live camera is {d.min():.3f} m from the NEAREST of "
                f"{len(cloud)} training poses (tol {AUG_NN_TOL_M} m) -- the rollout rig is not "
                f"one this checkpoint was trained on")
            assert ecc < AUG_MAX_ECCENTRICITY, (
                f"view{vi+1} ({cam}) live camera is more eccentric than {ecc:.1%} of training "
                f"poses (max {AUG_MAX_ECCENTRICITY:.0%}) -- rollouts would extrapolate")
            print(f"    rig view{vi+1} ({cam}): nearest training pose {d.min()*100:.2f} cm, "
                  f"more eccentric than {ecc:.1%} of them  [augmented tree]")
        else:
            assert np.abs(extr[vi] - de).max() < CAM_TOL, (
                f"view{vi+1} ({cam}) extrinsics differ from the rendered rig by "
                f"{np.abs(extr[vi] - de).max():.3e} -- every rollout would be out of distribution")

        # Intrinsics: never randomised. Scale for the render resolution before comparing.
        scale = float(env.resolution) / float(d0.get("width", 512) if isinstance(d0, dict) and "width" in d0 else 512)
        di_scaled = di.copy()
        di_scaled[0, 0] *= scale
        di_scaled[1, 1] *= scale
        di_scaled[0, 2] *= scale
        di_scaled[1, 2] *= scale
        assert np.abs(intr[vi] - di_scaled).max() < 1e-3, (
            f"view{vi+1} ({cam}) intrinsics differ by {np.abs(intr[vi] - di_scaled).max():.3e} "
            f"(live {intr[vi][0,0]:.2f} vs disk {di[0,0]:.2f} scaled by {scale:.4f} for "
            f"render resolution {env.resolution})")

    # The sign a FOV-derived matrix would get wrong, silently mirroring every projection.
    assert intr[0][0, 0] < 0, f"expected negative focal length, got fx={intr[0][0,0]}"
    assert intr[0][0, 0] == intr[0][1, 1], "expected fx == fy"
    return extr, intr


def replay(env, task, variation, trial, verbose=True, stride=1, interpolate=False):
    """`stride` mimics a policy trained at that frame_interval: a frame_interval=3 checkpoint
    emits poses 3 native control steps apart, so replaying GT at stride 3 tests whether that
    SPACING is still executable before any GPU time is spent finding out."""
    seed = trial_seed(task, variation, trial)
    t0 = time.time()
    _desc, obs, actions = env.reset_to_new_demo(task, variation, seed)
    t_demo = time.time() - t0
    full_len = len(actions)
    actions = actions[::stride]
    if interpolate and stride > 1:
        # Exactly what --interpolate does to a policy chunk: subsample to the policy's stride,
        # then resample back to the native rate. If this recovers the success that raw stride-3
        # replay loses, the loss is the coarse EXECUTION spacing and not the information in the
        # waypoints -- which is the whole justification for interpolating the model's output.
        actions = interpolate_chunk(actions, stride)

    start_gap = float(np.linalg.norm(env.current_pose8(obs)[:3] - actions[0][:3]))
    assert start_gap < START_TOL_M, (
        f"{task} trial {trial}: reset_to_demo landed {start_gap:.5f} m from the demo's own "
        f"first pose -- the rewind is not returning to the recorded state")
    check_camera_rig(env, obs)

    n_ik_fail = streak = worst_streak = 0
    success = False
    tracking = []
    t0 = time.time()
    executed = 0
    for i, pose8 in enumerate(actions):
        res = env.step(pose8)
        executed = i + 1
        if not res.ik_ok:
            n_ik_fail += 1
            streak += 1
            worst_streak = max(worst_streak, streak)
            continue
        streak = 0
        obs = res.obs
        tracking.append(float(np.linalg.norm(env.current_pose8(obs)[:3] - pose8[:3])))
        if res.success:
            success = True
            break
    t_replay = time.time() - t0

    if verbose:
        print(f"    {task:18s} trial {trial:2d} stride={stride}"
              f"{'+interp' if interpolate else '       '} demo={full_len:4d} "
              f"waypoints={len(actions):4d} exec={executed:4d} "
              f"ik_fail={n_ik_fail:3d} ({n_ik_fail/max(len(actions),1)*100:5.1f}%) "
              f"track={np.median(tracking) if tracking else float('nan'):.4f}m "
              f"gen={t_demo:5.1f}s replay={t_replay:5.1f}s SUCCESS={success}", flush=True)
    return dict(task=task, variation=variation, trial=trial, scene_seed=seed, stride=stride,
                interpolated=bool(interpolate and stride > 1),
                demo_len=int(full_len), waypoints=int(len(actions)),
                executed=executed, success=success,
                ik_fail=n_ik_fail, ik_fail_rate=n_ik_fail / max(len(actions), 1),
                worst_streak=worst_streak, start_gap=start_gap,
                demo_seconds=round(t_demo, 1), replay_seconds=round(t_replay, 1),
                track_err_median=float(np.median(tracking)) if tracking else float("nan"))


def check_determinism(env, task, variation, trial):
    """The same scene seed must produce the same INITIAL SCENE, or the arms are not comparable.

    The invariant that matters for the paired test is the world the policy is dropped into,
    which is what `reset()` places from the numpy seed. The DEMO PATH is a weaker thing: RLBench
    plans it with a randomised planner whose state is not fully covered by the numpy seed, so
    the same seed can yield demos of different length (measured: open_drawer trial 1 gave 96
    and 99 steps on two calls in one process). That is tolerable -- the demo is only a reference
    trajectory and a step budget -- but a different initial scene would not be, so the check is
    on the rendered observation, not on the demo.
    """
    _d1, o1, a1 = env.reset_to_new_demo(task, variation, trial_seed(task, variation, trial))
    rgb1 = env.views_rgb(o1).astype(np.float32)
    _d2, o2, a2 = env.reset_to_new_demo(task, variation, trial_seed(task, variation, trial))
    rgb2 = env.views_rgb(o2).astype(np.float32)
    mae = float(np.abs(rgb1 - rgb2).mean())
    assert mae < 1.0, (
        f"same seed rendered a different initial scene (MAE {mae:.2f}/255) -- the two arms "
        f"would be scored on different worlds and the paired McNemar test would be invalid")
    return len(a1), len(a2), mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--variation", type=int, default=0)
    ap.add_argument("--min-success-rate", type=float, default=0.75,
                    help="gate over ALL tasks; per-task results decide which enter the campaign")
    ap.add_argument("--skip-determinism", action="store_true")
    ap.add_argument("--arm-action-mode", choices=("ik", "planning"), default="ik")
    ap.add_argument("--interpolate", action="store_true",
                    help="resample strided waypoints back to native rate before executing")
    ap.add_argument("--strides", type=int, nargs="*", default=[1],
                    help="waypoint spacing to replay at. 1 = native 20 Hz. Pass 3 to mimic a "
                         "frame_interval=3 policy's output spacing before spending GPU time.")
    args = ap.parse_args()

    print("=" * 78)
    print("GT REPLAY -- each scene's own demo through the live harness (no model, no GPU)")
    print("=" * 78)

    # Cheap contract checks first, before spending simulator time.
    assert episode_seed_to_rng_seed(0, 0) == 0
    assert episode_seed_to_rng_seed(3, 1) == 3 + 1_000_003
    assert episode_seed_to_rng_seed(3, 1, attempt=2) == 3 + 1_000_003 + 2 * 100_003
    print("SEED_FORMULA_MATCHES_GEN_DATASET_OK")

    seeds = {(t, args.variation, k) for t in args.tasks for k in range(args.trials)}
    assert len({trial_seed(*s) for s in seeds}) == len(seeds), "trial_seed collides"
    assert trial_seed("close_jar", 0, 0) == trial_seed("close_jar", 0, 0), "trial_seed unstable"
    print(f"TRIAL_SEEDS_DISTINCT_AND_STABLE_OK  ({len(seeds)} seeds)")

    results = []
    with RolloutEnv(resolution=256, headless=True, arm_action_mode=args.arm_action_mode) as env:
        print(f"ENV_LAUNCH_OK  cameras={VIEW_CAMERAS} arm={args.arm_action_mode}", flush=True)
        for task in args.tasks:
            print(f"\n  --- {task} ---", flush=True)
            for trial in range(args.trials):
                for stride in args.strides:
                    try:
                        results.append(replay(env, task, args.variation, trial, stride=stride,
                                              interpolate=args.interpolate))
                    except Exception as exc:
                        print(f"    {task} trial {trial} stride {stride}: "
                              f"ERROR {type(exc).__name__}: {exc}", flush=True)
                        results.append(dict(task=task, variation=args.variation, trial=trial,
                                            stride=stride, success=False,
                                            error=f"{type(exc).__name__}: {exc}",
                                            ik_fail_rate=float("nan"), demo_len=0))

        if not args.skip_determinism:
            n1, n2, mae = check_determinism(env, args.tasks[0], args.variation, 0)
            print(f"\nSCENE_DETERMINISM_OK  same seed -> identical initial scene "
                  f"(rendered MAE {mae:.3f}/255); reference demo {n1} vs {n2} steps "
                  f"({'identical' if n1 == n2 else 'planner varies, tolerated'})")

    print("\n" + "=" * 78)
    ok_tasks, bad_tasks = [], []
    for task in args.tasks:
        rs = [r for r in results if r["task"] == task]
        if not rs:
            continue
        sr = float(np.mean([r["success"] for r in rs]))
        ik = float(np.nanmean([r.get("ik_fail_rate", np.nan) for r in rs]))
        print(f"{task:18s} success {sr*100:5.1f}%   mean ik_fail {ik*100:5.1f}%   (n={len(rs)})")
        (ok_tasks if sr >= 0.99 else bad_tasks).append(task)

    print()
    print(f"MEASURABLE tasks (GT replay 100%): {ok_tasks}")
    if bad_tasks:
        print(f"NOT MEASURABLE (harness cannot replay GT): {bad_tasks}")
        print("  -> exclude these from the rollout campaign, or a 0% model score on them is")
        print("     the harness's number rather than the model's.")

    out = os.path.join(REPO, "reports", "closedloop")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "gt_replay.json")
    with open(path, "w") as f:
        json.dump({"results": results, "measurable_tasks": ok_tasks,
                   "unmeasurable_tasks": bad_tasks}, f, indent=2)
    print(f"wrote {path}")

    overall = float(np.mean([r["success"] for r in results]))
    assert overall >= args.min_success_rate, (
        f"GT replay reached only {overall*100:.1f}% overall (need "
        f">={args.min_success_rate*100:.0f}%). The harness cannot execute known-good actions, so "
        f"a low success rate from the MODEL would be uninterpretable. Fix before any campaign.")
    print(f"GT_REPLAY_SUCCESS_OK  {overall*100:.1f}% overall")
    print("ALL_ROLLOUT_ENV_TESTS_PASSED")


if __name__ == "__main__":
    main()
