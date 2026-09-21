"""Receding-horizon closed-loop rollouts on RLBench.

The loop is the standard chunk cache, copied in shape from starVLA's
`examples/modelExtensions/Gemma4/eval_libero_local.py:269-285`:

    if step % execution_horizon == 0:  chunk = policy.run_policy(obs, ...)
    env.step(chunk[step % execution_horizon])

Why closed loop at all, when the paper's Table 3 is one-trial: our arms are trained on 41-frame
windows at native 20 Hz, i.e. 2.0 s of motion, while the median selfgen episode is 145 steps
(7.25 s). One generation cannot cover an episode, so the horizon has to be rebuilt by replanning.
A `frame_interval=3` arm sees 6.0 s per window and needs correspondingly fewer replans. See
EVAL_PLAN_CLOSEDLOOP.md Sec. 2.

Scenes come from `task.get_demos(live_demos=True)` + `reset_to_demo`, not from a bare `reset()`:
that guarantees the scene is SOLVABLE (a demo exists for it) and yields the ground-truth
trajectory for the same scene, so a failure cannot be blamed on an impossible initial state and
open-loop reference metrics come for free. The demo seed is fixed per trial, so both arms are
scored on exactly the same scenes -- the precondition for the paired McNemar test.
"""
from __future__ import annotations

import argparse
import hashlib
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

from eval.rollout_env import RolloutEnv, RolloutStats  # noqa: E402

# `close_jar` is deliberately absent: the simulator-backed GT replay gate gets 0% despite
# zero IK failures, so end-effector-pose + discrete-gripper replay cannot measure that task.
# Keeping it in the default model grid would silently attribute a harness floor to the model.
DEFAULT_TASKS = ["push_buttons", "open_drawer", "meat_off_grill"]


def trial_seed(task: str, variation: int, trial: int) -> int:
    """Deterministic per-(task, variation, trial) scene seed.

    Must not depend on the checkpoint, the arm, or the PROCESS: the paired test in
    EVAL_PLAN_CLOSEDLOOP.md Sec. 4.3 requires both arms to see identical scenes, and the two
    arms are scored in separate runs.

    Uses blake2b rather than the builtin `hash()`. Python salts string hashing with a random
    per-process PYTHONHASHSEED, so `hash(("push_buttons", 0, 0))` returns a different value in
    every interpreter -- measured across three processes: 628764 / 991250 / 341356. That would
    have silently given each arm its own scenes, making the paired comparison meaningless while
    every individual run still looked fine.
    """
    key = f"{task}|{variation}|{trial}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") % 1_000_000 + 1


def interpolate_chunk(chunk: np.ndarray, factor: int) -> np.ndarray:
    """Resample a pose chunk from the policy's stride back to the native control rate.

    A frame_interval=3 checkpoint emits poses 3 native steps apart (~23 mm between waypoints
    instead of ~7.7 mm). Executing those directly is legal but lossy: ground-truth replay at
    stride 3 fails trials that stride 1 solves, so the coarse spacing alone puts a ceiling on
    the achievable success rate. Interpolating restores fine-grained execution while leaving
    the model's prediction untouched -- the model supplies waypoints, the controller fills in
    between them, which is the normal division of labour.

    Position is linear; rotation is SLERP (linear interpolation of quaternions would leave the
    unit sphere and drift in angular velocity). Openness is a step function -- a gripper is
    open or closed, never 0.5 -- so it is held, not blended.
    """
    if factor <= 1:
        return chunk
    from scipy.spatial.transform import Rotation as Rot
    from scipy.spatial.transform import Slerp

    n = len(chunk)
    src = np.arange(n, dtype=float)
    dst = np.linspace(0.0, n - 1.0, (n - 1) * factor + 1)

    pos = np.stack([np.interp(dst, src, chunk[:, i]) for i in range(3)], axis=1)
    rots = Slerp(src, Rot.from_quat(chunk[:, 3:7]))(dst).as_quat()
    grip = chunk[np.clip(np.floor(dst).astype(int), 0, n - 1), 7:8]
    return np.concatenate([pos, rots, grip], axis=1)



def ramp_from_current(current_pose8: np.ndarray, target_pose8: np.ndarray, n: int) -> np.ndarray:
    """`n` intermediate poses from where the arm IS to the chunk's first trustworthy pose.

    Dropping the anchor-dominated frames leaves the first executable pose far away -- measured
    154 mm on a real close_drawer chunk once frames 0-3 were removed. Commanding that directly
    is a jump: `ViaPlanning` will path around it, but `ViaIK` cannot, and either way the arm
    lurches. Filling the dropped slots with a ramp costs the same step budget as dropping them
    (the chunk keeps its length) while turning wasted steps into useful motion.

    Rotation is SLERP; the gripper HOLDS its current state, because the model's open/close
    command belongs to the pose it was predicted for, not to the approach.
    """
    if n <= 0:
        return np.empty((0, 8), dtype=np.float64)
    from scipy.spatial.transform import Rotation as Rot
    from scipy.spatial.transform import Slerp

    cur = np.asarray(current_pose8, dtype=np.float64).reshape(8)
    tgt = np.asarray(target_pose8, dtype=np.float64).reshape(8)
    frac = np.arange(1, n + 1, dtype=float) / (n + 1)
    pos = cur[None, :3] + frac[:, None] * (tgt[:3] - cur[:3])[None, :]
    quats = Rot.from_quat(np.stack([cur[3:7], tgt[3:7]]))
    rots = Slerp([0.0, 1.0], quats)(frac).as_quat()
    grip = np.full((n, 1), cur[7])
    return np.concatenate([pos, rots, grip], axis=1)


def _anchor_views(env, obs, intr, task, seg_lut):
    """把实时观测编码成 policy 期待的 [V,H,W,3] uint8 锚定帧。

    默认 video 走原路径(`views_rgb`),所以既有的 RGB 闭环调用一个字节都没变。
    """
    m = getattr(env, "anchor_modality", "video")
    if m == "all":
        from eval import anchor_encode as AE
        d = env.views_depth(obs)
        return {"video": env.views_rgb(obs),
                "depth": AE.encode_depth_views(d),
                "normal": AE.encode_normal_views(d, intr),
                "segmentation": AE.encode_seg_views(env.views_mask(obs), seg_lut)}
    if m == "video":
        return env.views_rgb(obs)
    from eval import anchor_encode as AE
    if m == "depth":
        return AE.encode_depth_views(env.views_depth(obs))
    if m == "normal":
        return AE.encode_normal_views(env.views_depth(obs), intr)
    if m == "segmentation":
        return AE.encode_seg_views(env.views_mask(obs), seg_lut)
    raise ValueError(f"unknown anchor modality {m!r}")


def rollout_one(
    env: RolloutEnv,
    policy,
    task: str,
    variation: int,
    trial: int,
    prompt: Optional[str] = None,
    execution_horizon: int = 41,
    max_steps_factor: float = 2.0,
    max_ik_fail_streak: int = 5,
    gt_replay: bool = False,
    null_cross_scene: bool = False,
    frame_interval: int = 1,
    interpolate: bool = False,
    record_video: bool = False,
    video_dir: Optional[str] = None,
    prompt_tag_style: str = "explicit",
    skip_low_confidence: bool = False,
    skip_anchor_frames: int = 4,
    dump_dir: Optional[str] = None,
) -> Dict:
    """One episode. `gt_replay=True` runs the demo's own actions instead of the policy, which
    is the no-GPU self-check: it must succeed, or a model failure is uninterpretable."""
    if not gt_replay and not null_cross_scene:
        horizon = getattr(policy, "num_frames", execution_horizon)
        if execution_horizon > horizon:
            raise ValueError(
                f"execution_horizon={execution_horizon} exceeds the policy's chunk length "
                f"{horizon}; the loop would index past the end of the generated chunk")
    stats = RolloutStats()
    seed = trial_seed(task, variation, trial)

    donor_seed = None
    if null_cross_scene:
        # A different scene of the SAME task. Offsetting the trial index keeps it deterministic
        # and disjoint from every seed actually scored (trials are 0..num_trials-1).
        donor_seed = trial_seed(task, variation, trial + 10_000)
        _, _, donor_actions = env.reset_to_new_demo(task, variation, donor_seed)

    t0 = time.time()
    descriptions, obs, demo_actions = env.reset_to_new_demo(task, variation, seed)
    demo_seconds = time.time() - t0

    if null_cross_scene:
        from eval.null_policy import CrossSceneNullPolicy
        policy = CrossSceneNullPolicy(donor_actions, num_frames=execution_horizon,
                                      frame_interval=frame_interval if interpolate else 1)

    if prompt is None:
        # The prompt must carry the SAME tag style the checkpoint was trained with. Our arms
        # use `--prompt_tag_style explicit`, i.e. every training prompt began with
        # `<video><action> `; feeding the bare RLBench description instead is an
        # out-of-distribution text input. The official upstream checkpoint never saw those
        # tags, so it needs style="none".
        from training.templates import parse_template, prompt_prefix
        # tag 必须跟着锚定模态走:`<depth><action> ` / `<scene-seg><action> ` / `<normal><action> `。
        # 用错 tag 就是在问 checkpoint 一个它没被训过的组合(prompt scrub trap 的同类)。
        if getattr(policy, "full_anchor_fusion", False):
            # 10 段融合画布,单份 prompt(跟画布本身的 template 走),不是四路独立 prompt。
            _tpl = policy.template
            prompt = prompt_prefix(parse_template(_tpl), style=prompt_tag_style,
                                   segmentation_mode="scene_roles") + str(descriptions[0])
        elif getattr(env, "anchor_modality", "video") == "all":
            # 老的四路独立融合模式:每条路径一份 prompt,tag 各自对应自己的模态。只取四个
            # 单模态条目 -- ANCHOR_TEMPLATE 还收了 fusion/fusion_full_anchor 两个 10 段条目,
            # 混进来会问四路里多出两条它们不认的模板。
            from eval.anchor_encode import ANCHOR_TEMPLATE
            base_modalities = ("video", "depth", "segmentation", "normal")
            prompt = {m: prompt_prefix(parse_template(ANCHOR_TEMPLATE[m]), style=prompt_tag_style,
                                       segmentation_mode="scene_roles") + str(descriptions[0])
                      for m in base_modalities}
        else:
            _tpl = getattr(policy, "template", "video+action")
            prompt = prompt_prefix(parse_template(_tpl), style=prompt_tag_style,
                                   segmentation_mode="scene_roles") + str(descriptions[0])

    # The demo is recorded at the native 20 Hz control rate, but the policy emits one pose per
    # `frame_interval` native steps (it was trained on windows of that stride), so a chunk of
    # 41 poses covers 41 * frame_interval native steps. Budgeting in native units would give a
    # frame_interval=3 policy three times the intended wall-clock -- generous for a good
    # policy, but expensive for a bad one, since every wasted env.step() still runs IK settling
    # and every 41 of them triggers another 35 s generation.
    # With --interpolate the chunk is resampled back to the native rate, so one env.step() is
    # again one native step and the budget must be in native units.
    effective_stride = 1 if interpolate else max(frame_interval, 1)
    policy_steps_to_cover_demo = int(np.ceil(len(demo_actions) / effective_stride))
    max_steps = max(int(max_steps_factor * policy_steps_to_cover_demo), execution_horizon)
    extr, intr = env.camera_params(obs)
    # seg 的角色 LUT 必须【每个 episode 现场建】:CoppeliaSim 按加载顺序编 handle,
    # 离线的映射对实时场景不成立(rollout_env.handle_names 注释里有实例)。
    seg_lut = None
    if getattr(env, "anchor_modality", "video") in ("segmentation", "all"):
        from eval.anchor_encode import build_live_role_lut, unknown_fraction
        mv = env.views_mask(obs)
        lut, present, unmapped = build_live_role_lut(task, mv, env.handle_names(mv))
        unk = unknown_fraction(mv, lut)
        # 训练数据里 unknown 恒为 0(硬不变量)。实时场景里 >0 说明有 handle 没映射上,
        # 模型会看到训练中从未出现过的橙色 —— 这是 seg 这条路唯一会静默出错的地方。
        if unk > 0.001:
            raise RuntimeError(
                f"{task} trial {trial}: 实时角色图有 {unk:.2%} 的 unknown 像素"
                f"(训练数据里恒为 0)。未映射 handle: {unmapped[:8]}")
        seg_lut = lut
    # How many env.step()s one generated chunk is worth once interpolated.
    steps_per_chunk = execution_horizon if not interpolate else (execution_horizon - 1) * frame_interval + 1

    chunk: Optional[np.ndarray] = None
    streak = 0
    debug_trace: List[Dict] = []
    # What the SIMULATOR actually did, frame by frame. Distinct from the model's generated
    # video: the generated canvas contains the model's IMAGINED future RGB, which can show the
    # task being solved while the decoded actions fail in the simulator. Recording both is the
    # only way to tell "the world model is wrong" apart from "the action decode is wrong".
    sim_frames: List[np.ndarray] = []
    gen_frames: List[np.ndarray] = []   # the model's generated canvas, per replan
    # --- sim-replay perception dump -------------------------------------------------------
    # The counterfactual ground truth for "what the model imagined": at every executed step,
    # what the SIMULATOR renders after actually running the model's own decoded actions. This
    # is the only target that does not charge the model for failing to reproduce the demo.
    # Saved raw (never through the mp4 writer): the depth codec is a colour path, and H.264
    # chroma subsampling turns a metric depth map into a differently-metric depth map silently.
    # `cmd`/`ach` are the commanded and ACHIEVED end-effector poses. Their gap is the protocol's
    # own noise floor -- the imagined depth belongs to the pose the model drew, while the render
    # belongs to the pose IK actually reached -- and a perception number quoted without it is
    # not interpretable.
    dump_canvas: List[np.ndarray] = []      # [4*T, H, W, 3] uint8, one per replan
    dump_depth: List[np.ndarray] = []       # [V, H, W] float16 metric, one per executed step
    dump_mask: List[np.ndarray] = []        # [V, H, W] uint16 handle map, one per executed step
    dump_rgb: List[np.ndarray] = []         # [V, H, W, 3] uint8, one per executed step
    dump_steps: List[Dict] = []             # step -> which canvas frame produced it
    dumping = dump_dir is not None
    if record_video:
        sim_frames.append(env.views_rgb(obs).copy())
    t0 = time.time()

    for step in range(max_steps):
        if gt_replay:
            # HOLD THE LAST POSE rather than stopping when the demo runs out, so the GT
            # ceiling is measured under the SAME step budget the model gets
            # (max_steps_factor). A controller lags its target and anything still settling
            # when the last waypoint is issued has not settled yet. Measured on ManiSkill's
            # place_sphere, where the sphere has to come to rest inside a 5 mm tolerance:
            # 12/20 stopping at the demo's end, 20/20 holding -- every one of those eight
            # "failures" was the budget, not the harness.
            action = demo_actions[min(step, len(demo_actions) - 1)]
            if step >= len(demo_actions):
                stats.stop_reason = "holding final pose"
        else:
            if step % steps_per_chunk == 0:
                # The current pose is conditioning, not prediction -- see policy.run_policy.
                _av = _anchor_views(env, obs, intr, task, seg_lut)
                if getattr(policy, "full_anchor_fusion", False):
                    chunk = policy.run_policy_full_anchor(_av, extr, intr, prompt,
                                                          current_pose8=env.current_pose8(obs))
                elif isinstance(_av, dict):
                    chunk = policy.run_policy_fused(_av, extr, intr, prompt,
                                                    current_pose8=env.current_pose8(obs))
                else:
                    chunk = policy.run_policy(_av, extr, intr, prompt,
                                              current_pose8=env.current_pose8(obs))
                # Drop the chunk's leading frames: they are the RECONSTRUCTION of the pose the
                # robot is already in, not a prediction of where to go next. The action
                # segment's first latent is the anchor we supplied from `current_pose8`, and
                # the VAE's 4x temporal compression spreads that one latent over pixel frames
                # 0-1. Decoding them round-trips a known-exact pose through encode -> VAE ->
                # decode, and at the near-vertical home orientation that round trip flips the
                # approach axis: measured on close_drawer, frames 0-1 came back with the tool
                # pointing UP (+0.993 / +0.988 world-z) while ground truth and the current pose
                # both point DOWN (-0.971), a 174 deg error. IK then faithfully executes it --
                # the arm really does flip to the ceiling for two steps and snap back, which is
                # the "wiggle at the start of every rollout". Frame 2 onward tracked to 2.3 mm
                # and 0.3 deg in the same rollout, so only the anchor frames are affected.
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
                if dumping and getattr(policy, "last_frames", None):
                    dump_canvas.append(np.stack(policy.last_frames).astype(np.uint8))
                chunk_conf = getattr(policy, "last_confidence", None)
                if chunk_conf is not None:
                    chunk_conf = np.asarray(chunk_conf)[:len(chunk)]
            k = min(step % steps_per_chunk, len(chunk) - 1)
            action = chunk[k]
            # A pose whose decoded axis length is far off the encoder's 0.1 m has a median
            # rotation error of ~93 deg, and commanding it is what makes IK fail outright.
            # Holding the current pose is strictly safer than executing a coin-flip rotation.
            if (skip_low_confidence and chunk_conf is not None and not interpolate
                    and k < len(chunk_conf) and not chunk_conf[k]):
                stats.low_conf_held += 1
                action = env.current_pose8(obs).copy()
                action[7] = chunk[k][7]      # still honour the predicted gripper command

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
        if dumping:
            # `k` is the index INTO THE CHUNK, i.e. into the generated canvas's action segment,
            # so it is also the index of the perception frame the model drew for this moment.
            # Frames below `skip_anchor_frames` were overwritten by `ramp_from_current` above
            # and are NOT what the model predicted -- recorded, but flagged so the scorer drops
            # them rather than charging the ramp to the perception stream.
            dump_steps.append({
                "step": step,
                "replan": max(len(dump_canvas) - 1, 0),
                "k": int(k) if not gt_replay else -1,
                "is_ramp": bool((not gt_replay) and int(k) < skip_anchor_frames),
                "cmd": np.asarray(action, dtype=np.float32).tolist(),
                "ach": env.current_pose8(obs).astype(np.float32).tolist(),
            })
            if getattr(env, "_need_depth", False):
                dump_depth.append(env.views_depth(obs).astype(np.float16))
            if getattr(env, "_need_mask", False):
                dump_mask.append(env.views_mask(obs).astype(np.uint16))
            # LOSSLESS simulator RGB, which is the counterfactual target for the future-RGB row
            # of Tab. 2. `sim_frames` above holds the same pixels but is written out as an mp4,
            # and an LPIPS scored on H.264 output measures the codec as much as the model.
            dump_rgb.append(env.views_rgb(obs).copy())
        if res.success:
            stats.success = True
            stats.stop_reason = "success"
            break
    else:
        stats.stop_reason = "timeout"

    dump_path = None
    if dumping and dump_steps:
        os.makedirs(dump_dir, exist_ok=True)
        stem = f"{task}_v{variation}_t{trial}"
        dump_path = os.path.join(dump_dir, f"{stem}.npz")
        payload = {
            "canvas": np.stack(dump_canvas) if dump_canvas else np.zeros((0,), np.uint8),
            "cmd": np.asarray([d["cmd"] for d in dump_steps], np.float32),
            "ach": np.asarray([d["ach"] for d in dump_steps], np.float32),
            "step": np.asarray([d["step"] for d in dump_steps], np.int32),
            "replan": np.asarray([d["replan"] for d in dump_steps], np.int32),
            "k": np.asarray([d["k"] for d in dump_steps], np.int32),
            "is_ramp": np.asarray([d["is_ramp"] for d in dump_steps], bool),
        }
        if dump_depth:
            payload["sim_depth"] = np.stack(dump_depth)
        if dump_mask:
            payload["sim_mask"] = np.stack(dump_mask)
        if dump_rgb:
            payload["sim_rgb"] = np.stack(dump_rgb)
        extr, intr = env.camera_params(obs)
        payload["extrinsics"], payload["intrinsics"] = extr, intr
        # savez_compressed, not savez: the canvas is a 4-segment 512^2 uint8 stack (~129 MB per
        # replan raw) and the depth-codec colour path is smooth enough to deflate well. Measured
        # on the smoke test before committing to a campaign-sized dump.
        np.savez_compressed(dump_path, **payload)

    video_path = gen_path = None
    if record_video and video_dir:
        stem = f"{task}_v{variation}_t{trial}_{'SUCCESS' if stats.success else 'fail'}"
        # executed/ = what the SIMULATOR did, [view1 | view2], one frame per control step.
        if sim_frames:
            ex_dir = os.path.join(video_dir, "executed")
            os.makedirs(ex_dir, exist_ok=True)
            video_path = os.path.join(ex_dir, f"{stem}.mp4")
            imageio.mimwrite(video_path, [np.concatenate([f[0], f[1]], axis=1)
                                          for f in sim_frames], fps=10, quality=7)
        # generated/ = what the MODEL imagined, concatenated over replans. The canvas is
        # [v1_rgb | v1_action | v2_rgb | v2_action]; the RGB segments are a predicted future,
        # NOT an observation, which is why they can show the task solved while executed/ shows
        # it unsolved.
        if gen_frames:
            gn_dir = os.path.join(video_dir, "generated")
            os.makedirs(gn_dir, exist_ok=True)
            gen_path = os.path.join(gn_dir, f"{stem}.mp4")
            imageio.mimwrite(gen_path, gen_frames, fps=15, quality=6)

    return {
        "task": task,
        "variation": variation,
        "sim_video": video_path,
        "generated_video": gen_path,
        "percep_dump": dump_path,
        "trial": trial,
        "scene_seed": seed,
        "prompt": prompt,
        "demo_len": int(len(demo_actions)),
        "max_steps": max_steps,
        "execution_horizon": execution_horizon,
        "frame_interval": frame_interval,
        "interpolate": interpolate,
        "skip_anchor_frames": skip_anchor_frames,
        "gt_replay": gt_replay,
        "null_cross_scene": null_cross_scene,
        "donor_seed": donor_seed,
        "demo_seconds": round(demo_seconds, 1),
        "rollout_seconds": round(time.time() - t0, 1),
        **stats.as_dict(),
        "policy_debug": debug_trace,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=None, help="omit together with --gt-replay for the self-check")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    ap.add_argument("--variation", type=int, default=0)
    ap.add_argument("--max-ik-fail-streak", type=int, default=5,
                    help="abort a rollout after this many consecutive unreachable commands. "
                         "The default of 5 is a harness policy, not a property of the model: it "
                         "killed 16/20 push_buttons rollouts on variation 1 against 5/20 on "
                         "variation 0, so it can amplify a small geometric degradation into a "
                         "near-total collapse. Raise it to separate the two.")
    ap.add_argument("--num-trials", type=int, default=20,
                    help="20 matches the paper's grid (every Table 2/3 entry is a multiple of 5)")
    ap.add_argument("--execution-horizon", type=int, default=41,
                    help="41 = execute a whole chunk then replan (cheapest); 21 = execute half")
    ap.add_argument("--num-frames", type=int, default=41)
    ap.add_argument("--prompt-tag-style", choices=("explicit", "none"), default="explicit",
                    help="MUST match the checkpoint's training --prompt_tag_style. 'explicit' "
                         "for arm0/arm1/arm4 (prepends '<video><action> '); 'none' for the "
                         "upstream official checkpoint, which never saw those tags.")
    ap.add_argument("--skip-anchor-frames", type=int, default=4,
                    help="drop this many leading chunk frames. They reconstruct the pose the "
                         "robot is already in (the conditioning anchor), and that round "
                         "trip flips the approach axis at near-vertical orientations. 4 is "
                         "measured, not guessed: perturbing the anchor latent moves pixel "
                         "frames 0-3 by 59/74/100/63% of its peak and frame 4 by only 19%. "
                         "The dropped slots are refilled with a ramp from the current pose, so "
                         "the chunk keeps its length. 0 restores the old behaviour.")
    ap.add_argument("--skip-low-confidence", action="store_true",
                    help="hold position on poses whose decoded axis length is far from the "
                         "encoded 0.1 m (median rotation error ~93 deg on those) instead of "
                         "commanding them; a debug lever for the high ik_fail rate")
    ap.add_argument("--anchor-modality", default="video",
                    choices=("video", "depth", "segmentation", "normal", "all", "fusion",
                             "fusion_full_anchor"),
                    help="策略以哪种观测为锚定帧。video = 历史行为(仿真器只渲染 RGB);"
                         "depth/segmentation/normal 会打开对应的额外渲染,每步都要多花仿真"
                         "时间;fusion = 10 段融合画布,rollout 仍只观测 RGB(rgb_only),"
                         "只是 ActionImagePolicy 用完整模板问模型,RolloutEnv 端等同 video;"
                         "fusion_full_anchor = 同一 10 段画布,但每个模态都喂真锚定帧"
                         "(simulator 现场渲染 depth/mask,RolloutEnv 端等同 all)——"
                         "离线 full_anchor 上限搬到闭环,答的是「rgb_only 本身是不是瓶颈」"
                         "而不是「四路独立问再融合」(那是 all 模式)。")
    # 默认【开】。2026-09-07 改:非 RGB 锚定的三次闭环因为漏了这个 flag,300 个 rollout
    # 全部没有留下视频,而重跑要 ~33 GPU-小时。录像的边际成本很小(一次 100-rollout 的
    # campaign 约 330 MB),而缺了它就只剩一个成功率标量,失败模式无从判读。
    # 用 BooleanOptionalAction,所以 `--no-record-video` 仍然可以关掉。
    ap.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=True,
                    help="save what the SIMULATOR executed, for comparison against the model's "
                         "generated video (which shows its imagined future, not execution)")
    ap.add_argument("--interpolate", action="store_true",
                    help="resample the chunk from the policy stride back to native 20 Hz "
                         "(SLERP for rotation, held gripper). GT replay at stride 3 fails "
                         "trials that stride 1 solves, so without this the coarse spacing "
                         "itself caps the achievable success rate.")
    ap.add_argument("--frame-interval", type=int, default=1,
                    help="the checkpoint's TRAINING frame_interval (3 for arm4__seed42_fi3). "
                         "Sets how many native 20 Hz steps one predicted pose covers, which is "
                         "what the step budget is scaled by.")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--null-cross-scene", action="store_true",
                    help="LOWER bound control: replay the ground-truth demo of a DIFFERENT "
                         "scene seed of the same task, through the identical ramp / "
                         "interpolate / IK path. Carries correct task motion but zero "
                         "information about this scene, so it measures the task's chance "
                         "level. Needs no checkpoint and no GPU. See eval/null_policy.py.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gripper-mode", choices=("paper", "median"), default="paper")
    ap.add_argument("--axis-solver", choices=("ray", "sphere"), default="sphere",
                    help="'ray' = the paper's Sec. 3.2 decoder; 'sphere' = the fork deviation "
                         "that fixes its foreshortening degeneracy. Recorded in the report.")
    ap.add_argument("--arm-action-mode", choices=("ik", "planning"), default="ik",
                    help="ik fails fast on unreachable targets; planning had 0%% IK failure on "
                         "a GT replay but may search a long time on a badly predicted pose")
    ap.add_argument("--max-steps-factor", type=float, default=2.0)
    ap.add_argument("--gt-replay", action="store_true",
                    help="replay each scene's own demo instead of the policy (no GPU, no model)")
    ap.add_argument("--out", default=str(REPO / "reports" / "closedloop"))
    ap.add_argument("--dump-percep", action="store_true",
                    help="save, per trial, the model's raw generated canvas and the SIMULATOR's "
                         "own depth/mask at every executed step -- the counterfactual ground "
                         "truth for the sim-replay perception eval. Forces the extra renders on "
                         "regardless of --anchor-modality, which is what lets the RGB-only arm "
                         "(no depth+action template, so necessarily anchor=video) be scored at "
                         "all. Costs sim time on EVERY step and ~tens of MB per trial.")
    ap.add_argument("--dump-max-trials", type=int, default=0,
                    help="dump only the first N trials of each task (0 = all). The dump is for "
                         "scoring perception, which needs far fewer trials than a success rate.")
    args = ap.parse_args()

    if not args.gt_replay and not args.null_cross_scene and not args.ckpt:
        ap.error("--ckpt is required unless --gt-replay")
    if not args.tasks:
        ap.error("--tasks must contain at least one task")
    if args.num_trials < 1:
        ap.error("--num-trials must be >= 1")
    if args.num_frames < 1:
        ap.error("--num-frames must be >= 1")
    if not 1 <= args.execution_horizon <= args.num_frames:
        ap.error("--execution-horizon must be between 1 and --num-frames")
    if args.frame_interval < 1:
        ap.error("--frame-interval must be >= 1")
    if args.max_steps_factor <= 0:
        ap.error("--max-steps-factor must be > 0")

    policy = None
    if not args.gt_replay and not args.null_cross_scene:
        from eval.policy import ActionImagePolicy, build_policy_pipeline

        print(f"loading {args.ckpt} ...", flush=True)
        t0 = time.time()
        pipe = build_policy_pipeline(args.ckpt, resolution=args.res)
        policy = ActionImagePolicy(
            pipe, num_frames=args.num_frames, resolution=args.res, cfg_scale=args.cfg,
            num_inference_steps=args.steps, seed=args.seed, gripper_mode=args.gripper_mode,
            axis_solver=args.axis_solver, anchor_modality=args.anchor_modality,
        )
        print(f"loaded in {time.time()-t0:.0f}s", flush=True)

    results: List[Dict] = []
    errors: List[Dict] = []
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"rollout_{args.tag}.json")

    # RolloutEnv only knows the four RENDER-side anchors (video/depth/segmentation/normal) plus
    # the old "all" ensemble; it has no notion of the 10-segment fusion canvas. Under fusion the
    # rollout still OBSERVES nothing but RGB (rgb_only is the deployment condition -- see
    # ANCHOR_CONDITIONING["fusion"]), so the env gets "video" while ActionImagePolicy keeps
    # "fusion" to pick the full template/conditioning. Same split ActionImagePolicy.__init__
    # already does internally for its own `self.anchor_modality`. fusion_full_anchor needs the
    # SAME live depth/mask rendering the old "all" ensemble needs (real anchors for every
    # modality), so it maps to "all" at the env level -- only the POLICY-side consumption
    # differs (one 10-segment forward pass vs. four independent 4-segment queries).
    if args.anchor_modality == "fusion":
        env_anchor_modality = "video"
    elif args.anchor_modality == "fusion_full_anchor":
        env_anchor_modality = "all"
    else:
        env_anchor_modality = args.anchor_modality
    # The dump needs the simulator's own depth AND handle map at every step, whatever the
    # policy is anchored on -- see RolloutEnv.extra_renders.
    extra_renders = ("depth", "mask") if args.dump_percep else ()
    with RolloutEnv(resolution=args.res, headless=True,
                    arm_action_mode=args.arm_action_mode,
                    anchor_modality=env_anchor_modality,
                    extra_renders=extra_renders) as env:
        print(f"env up; tasks={args.tasks} trials={args.num_trials} "
              f"execution_horizon={args.execution_horizon} arm={args.arm_action_mode} "
              f"gt_replay={args.gt_replay}", flush=True)
        for task in args.tasks:
            for trial in range(args.num_trials):
                try:
                    r = rollout_one(
                        env, policy, task, args.variation, trial,
                        execution_horizon=args.execution_horizon,
                        max_steps_factor=args.max_steps_factor,
                        gt_replay=args.gt_replay,
                        null_cross_scene=args.null_cross_scene,
                        frame_interval=args.frame_interval,
                        interpolate=args.interpolate,
                        record_video=args.record_video,
                        video_dir=os.path.join(args.out, "videos", args.tag),
                        prompt_tag_style=args.prompt_tag_style,
                        skip_low_confidence=args.skip_low_confidence,
                        skip_anchor_frames=args.skip_anchor_frames,
                        max_ik_fail_streak=args.max_ik_fail_streak,
                        dump_dir=(os.path.join(args.out, "percep_dump", args.tag)
                                  if args.dump_percep and (args.dump_max_trials <= 0
                                                           or trial < args.dump_max_trials)
                                  else None),
                    )
                except Exception as exc:  # a bad demo must not kill the campaign
                    print(f"  {task} trial {trial}: ERROR {type(exc).__name__}: {exc}", flush=True)
                    # Do not let a skipped trial silently shrink the denominator. Keep errors
                    # separate from scored results so aggregate_closedloop remains backward
                    # compatible while the campaign JSON retains the missing cell and cause.
                    errors.append({
                        "task": task,
                        "variation": args.variation,
                        "trial": trial,
                        "scene_seed": trial_seed(task, args.variation, trial),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
                    with open(out_path, "w") as f:
                        json.dump({"args": vars(args), "errors": errors, "results": results},
                                  f, indent=2)
                    continue
                results.append(r)
                print(f"  {task:18s} trial {trial:2d}  success={str(r['success']):5s} "
                      f"steps={r['steps']:4d}/{r['max_steps']:4d} replans={r['replans']:2d} "
                      f"ik_fail={r['ik_fail_rate']*100:5.1f}%  {r['stop_reason']}", flush=True)
                with open(out_path, "w") as f:   # written incrementally: campaigns get killed
                    json.dump({"args": vars(args), "errors": errors, "results": results},
                              f, indent=2)

    # One summarizer for both entry points, so a table printed here and a table printed by
    # eval/aggregate_closedloop.py can never disagree.
    from eval.aggregate_closedloop import print_table, summarize

    summary = summarize(results, args.tag)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "summary": summary, "errors": errors,
                   "results": results}, f, indent=2)
    print_table(summary)
    if errors:
        print(f"WARNING: {len(errors)} requested rollout(s) errored and were not scored; "
              f"see the JSON 'errors' field.", flush=True)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
