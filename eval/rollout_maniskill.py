"""Receding-horizon closed-loop rollouts on ManiSkill3.

The RLBench sibling is `eval/rollout.py`; this file is the same loop against
`eval/rollout_env_maniskill.ManiSkillRolloutEnv`, and it imports that loop's helpers rather
than restating them, so a fix to the chunk handling cannot land in one simulator only.

WHAT DIFFERS FROM RLBENCH, and why each difference is forced:

  * `frame_interval` is 1, not 3. `training/dataset/base.py` pins the ManiSkill source to
    stride 1 (its episodes are 49-103 frames at 20 Hz; stride 3 would run off the end of most
    of them), so a 41-frame chunk is 2.05 s of motion here against 6.0 s on RLBench, and the
    step budget scales accordingly. Passing 3 here would score the checkpoint on a stride it
    never saw for this source.
  * Scenes come from the HELD-OUT TREE's episodes, not from a live demo generator: ManiSkill's
    reset seed reproduces the stored scene exactly (verified: the live TCP pose equals the
    episode's `actions.npy[0]` to 1e-4 m), which RLBench's does not. So each trial is a stored
    episode, its `actions.npy` is that scene's ground truth, and `--gt-replay` is an exact
    ceiling for the same scene rather than a fresh demo.
  * Success comes from the task's own `evaluate()`, through `info["success"]`.

`--gt-replay` needs no GPU and no checkpoint, and it is the gate this harness has to pass
before any model number from it means anything: on pull_cube episodes 0-1 it succeeds at
step 57/68 and 67/80 with zero IK failures.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import imageio
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.rollout import interpolate_chunk, ramp_from_current  # noqa: E402
from eval.rollout_env import RolloutStats  # noqa: E402
from eval.rollout_env_maniskill import ManiSkillRolloutEnv  # noqa: E402

DEFAULT_TASKS = ["pull_cube", "lift_peg_upright"]
DEFAULT_TREE = str(REPO / "data" / "maniskill3_heldout")


def rollout_one(env, policy, task: str, episode_dir: Path, trial: int, *,
                execution_horizon: int, max_steps_factor: float, gt_replay: bool,
                frame_interval: int, interpolate: bool, record_video: bool,
                video_dir: Optional[str], prompt_tag_style: str, skip_anchor_frames: int,
                max_ik_fail_streak: int) -> Dict:
    stats = RolloutStats()
    t_reset = time.time()
    descriptions, obs, demo_actions = env.reset_to_episode(episode_dir)
    reset_seconds = time.time() - t_reset

    from training.templates import parse_template, prompt_prefix
    tpl = getattr(policy, "template", "video+action") if policy is not None else "video+action"
    prompt = prompt_prefix(parse_template(tpl), style=prompt_tag_style,
                           segmentation_mode="scene_roles") + str(descriptions[0])

    effective_stride = 1 if interpolate else max(frame_interval, 1)
    policy_steps = int(np.ceil(len(demo_actions) / effective_stride))
    max_steps = max(int(max_steps_factor * policy_steps), execution_horizon)
    extr, intr = env.camera_params(obs)
    steps_per_chunk = execution_horizon if not interpolate else (execution_horizon - 1) * frame_interval + 1

    chunk: Optional[np.ndarray] = None
    streak = 0
    debug_trace: List[Dict] = []
    sim_frames: List[np.ndarray] = []
    gen_frames: List[np.ndarray] = []
    if record_video:
        sim_frames.append(env.views_rgb(obs).copy())
    t0 = time.time()

    for step in range(max_steps):
        if gt_replay:
            # HOLD THE LAST POSE instead of stopping at the end of the demo. A Cartesian
            # tracker lags its target, so the final commanded pose is reached a few steps
            # after it is issued and anything that still has to settle (a sphere rolling into
            # a bin, a cube coming to rest) has not settled when the demo runs out. Stopping
            # there measures the harness with a SHORTER budget than the model gets
            # (max_steps_factor 1.5), which is the wrong ceiling: the GT column exists to say
            # what this harness can achieve on this scene, not what it achieves in exactly
            # demo_len steps. Measured on place_sphere: 12/20 stopping at the demo's end,
            # every failure "demo exhausted" with the sphere near but outside the 5 mm
            # tolerance.
            action = demo_actions[min(step, len(demo_actions) - 1)]
            if step >= len(demo_actions):
                stats.stop_reason = "holding final pose"
        else:
            if step % steps_per_chunk == 0:
                chunk = policy.run_policy(env.views_rgb(obs), extr, intr, prompt,
                                          current_pose8=env.current_pose8(obs))
                if skip_anchor_frames > 0 and len(chunk) > skip_anchor_frames:
                    tail = chunk[skip_anchor_frames:]
                    chunk = np.concatenate(
                        [ramp_from_current(env.current_pose8(obs), tail[0], skip_anchor_frames),
                         tail], axis=0)
                chunk = chunk[:execution_horizon]
                if interpolate:
                    chunk = interpolate_chunk(chunk, frame_interval)
                stats.replans += 1
                debug_trace.append({"step": step, **getattr(policy, "last_debug", {})})
                if record_video and getattr(policy, "last_frames", None):
                    gen_frames.extend(policy.last_frames)
            action = chunk[min(step % steps_per_chunk, len(chunk) - 1)]

        res = env.step(action)
        stats.steps += 1
        if not res.ik_ok:
            stats.ik_fail += 1
            streak += 1
            stats.ik_fail_streak_max = max(stats.ik_fail_streak_max, streak)
            if streak >= max_ik_fail_streak:
                stats.terminated_early = True
                stats.stop_reason = f"{streak} consecutive IK failures"
                break
            continue
        streak = 0
        obs = res.obs
        if record_video:
            sim_frames.append(env.views_rgb(obs).copy())
        if res.success:
            stats.success = True
            stats.stop_reason = "success"
            break
    else:
        stats.stop_reason = "timeout"

    video_path = gen_path = None
    if record_video and video_dir:
        stem = f"{task}_{episode_dir.name}_t{trial}_{'SUCCESS' if stats.success else 'fail'}"
        if sim_frames:
            ex_dir = os.path.join(video_dir, "executed")
            os.makedirs(ex_dir, exist_ok=True)
            video_path = os.path.join(ex_dir, f"{stem}.mp4")
            imageio.mimwrite(video_path, [np.concatenate([f[0], f[1]], axis=1)
                                          for f in sim_frames], fps=10, quality=7)
        if gen_frames:
            gn_dir = os.path.join(video_dir, "generated")
            os.makedirs(gn_dir, exist_ok=True)
            gen_path = os.path.join(gn_dir, f"{stem}.mp4")
            imageio.mimwrite(gen_path, gen_frames, fps=15, quality=6)

    return {
        "task": task,
        "variation": int(str(episode_dir.parent.parent.name).replace("variation", "")),
        "episode": episode_dir.name,
        "sim_video": video_path,
        "generated_video": gen_path,
        "trial": trial,
        "scene_seed": int(json.loads((episode_dir / "meta.json").read_text())["seed"]),
        "prompt": prompt,
        "demo_len": int(len(demo_actions)),
        "max_steps": max_steps,
        "execution_horizon": execution_horizon,
        "frame_interval": frame_interval,
        "skip_anchor_frames": skip_anchor_frames,
        "gt_replay": gt_replay,
        "cameras_redrawn": bool(env._cam_mismatch),
        "reset_seconds": round(reset_seconds, 1),
        "rollout_seconds": round(time.time() - t0, 1),
        **stats.as_dict(),
        "policy_debug": debug_trace,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", default="ms_run")
    ap.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    ap.add_argument("--tree", default=DEFAULT_TREE)
    ap.add_argument("--variation", type=int, default=0,
                    help="0 for the held-out-TASK tree; 1 selects the held-out SEEDS of a "
                         "trained task in data/maniskill3 (tab:main Panel B's first block)")
    ap.add_argument("--num-trials", type=int, default=20)
    ap.add_argument("--max-ik-fail-streak", type=int, default=5)
    ap.add_argument("--execution-horizon", type=int, default=41)
    ap.add_argument("--num-frames", type=int, default=41)
    ap.add_argument("--prompt-tag-style", choices=("explicit", "none"), default="explicit")
    ap.add_argument("--skip-anchor-frames", type=int, default=4)
    ap.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--interpolate", action="store_true")
    ap.add_argument("--frame-interval", type=int, default=1,
                    help="1 for every ManiSkill run: the loader pins this source to stride 1")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gripper-mode", choices=("paper", "median"), default="paper")
    ap.add_argument("--axis-solver", choices=("ray", "sphere"), default="sphere")
    ap.add_argument("--max-steps-factor", type=float, default=1.5)
    ap.add_argument("--gt-replay", action="store_true")
    ap.add_argument("--out", default=str(REPO / "reports" / "closedloop_maniskill"))
    args = ap.parse_args()
    if not args.gt_replay and not args.ckpt:
        ap.error("--ckpt is required unless --gt-replay")

    policy = None
    if not args.gt_replay:
        from eval.policy import ActionImagePolicy, build_policy_pipeline
        print(f"loading {args.ckpt} ...", flush=True)
        t0 = time.time()
        pipe = build_policy_pipeline(args.ckpt, resolution=args.res)
        policy = ActionImagePolicy(pipe, num_frames=args.num_frames, resolution=args.res,
                                   cfg_scale=args.cfg, num_inference_steps=args.steps,
                                   seed=args.seed, gripper_mode=args.gripper_mode,
                                   axis_solver=args.axis_solver, anchor_modality="video")
        print(f"loaded in {time.time()-t0:.0f}s", flush=True)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"rollout_{args.tag}.json")
    results: List[Dict] = []
    errors: List[Dict] = []
    # RESUME. Without it a 10-trial look could only become a 20-trial result by throwing the
    # first 10 away. Trials already scored under THIS tag are kept and skipped -- but only if
    # the protocol is the one they were scored under; mixing two protocols in one file would
    # make every aggregate over it meaningless.
    done = set()
    if os.path.exists(out_path):
        prev = json.load(open(out_path))
        keys = ("ckpt", "tree", "variation", "cfg", "steps", "frame_interval", "prompt_tag_style",
                "skip_anchor_frames", "max_ik_fail_streak", "max_steps_factor", "seed",
                "execution_horizon", "axis_solver", "res")
        diff = {k: (prev.get("args", {}).get(k), getattr(args, k)) for k in keys
                if prev.get("args", {}).get(k) != getattr(args, k)}
        if diff:
            raise SystemExit(f"refusing to resume {out_path}: protocol differs {diff}")
        results = prev.get("results", [])
        done = {(r["task"], r["trial"]) for r in results}
        print(f"resuming {out_path}: {len(results)} trials already scored", flush=True)

    for task in args.tasks:
        # lazy_render skips rendering entirely, so it is safe ONLY when nothing will ask
        # for pixels: no policy anchor frames and no recorded video.
        with ManiSkillRolloutEnv(task, args.tree, resolution=args.res,
                                 lazy_render=(args.gt_replay and not args.record_video)) as env:
            eps = env.episode_dirs(args.variation)[: args.num_trials]
            print(f"env up; task={task} episodes={len(eps)} gt_replay={args.gt_replay}", flush=True)
            for trial, ed in enumerate(eps):
                if (task, trial) in done:
                    continue
                try:
                    r = rollout_one(
                        env, policy, task, ed, trial,
                        execution_horizon=args.execution_horizon,
                        max_steps_factor=args.max_steps_factor, gt_replay=args.gt_replay,
                        frame_interval=args.frame_interval, interpolate=args.interpolate,
                        record_video=args.record_video,
                        video_dir=os.path.join(args.out, "videos", args.tag),
                        prompt_tag_style=args.prompt_tag_style,
                        skip_anchor_frames=args.skip_anchor_frames,
                        max_ik_fail_streak=args.max_ik_fail_streak)
                except Exception as exc:
                    print(f"  {task} {ed.name}: ERROR {type(exc).__name__}: {exc}", flush=True)
                    errors.append({"task": task, "episode": ed.name, "trial": trial,
                                   "error_type": type(exc).__name__, "error": str(exc)})
                    with open(out_path, "w") as f:
                        json.dump({"args": vars(args), "errors": errors, "results": results},
                                  f, indent=2)
                    continue
                results.append(r)
                print(f"  {task:18s} trial {trial:2d} ({ed.name:10s}) success={str(r['success']):5s} "
                      f"steps={r['steps']:4d}/{r['max_steps']:4d} replans={r['replans']:2d} "
                      f"ik_fail={r['ik_fail_rate']*100:5.1f}%  {r['stop_reason']}", flush=True)
                with open(out_path, "w") as f:
                    json.dump({"args": vars(args), "errors": errors, "results": results},
                              f, indent=2)

    from eval.aggregate_closedloop import print_table, summarize
    summary = summarize(results, args.tag)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "summary": summary, "errors": errors,
                   "results": results}, f, indent=2)
    print_table(summary)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
