"""The action-image model as a closed-loop policy: observation in, executable pose chunk out.

Contract (borrowed from starVLA's `model2bench_interface.py`, which is the shape their sim and
real-robot evaluators already agree on):

    policy = ActionImagePolicy(ckpt, ...)
    chunk  = policy.run_policy(rgb_views, extrinsics, intrinsics, prompt)   # [num_frames, 8]

with each row `[x, y, z, qx, qy, qz, qw, openness]` in WORLD coordinates, directly steppable by
`RolloutEnv`. Chunk-cache scheduling, gripper stickiness and any action ensembling stay on the
caller, exactly as starVLA's `policy_wrapper.py` documents -- this class is stateless.

Three conventions are load-bearing and are asserted rather than assumed, because each one fails
silently if wrong (a mirrored projection, a pose in the wrong frame, a prompt off-distribution):

  * `extrinsics` are ABSOLUTE camera-to-world 4x4, laid out `[view1_frames | view2_frames]`
    along time. That is what makes the decoded pose come out in world coordinates, which is
    what `EndEffectorPoseViaIK(frame=WORLD)` consumes. (tests/test_selfgen_alignment.py pins
    this against the training dataset.)
  * `camera` is the Pluecker conditioning input and is RELATIVE to view1's first frame -- a
    different quantity from `extrinsics`, built with `get_relative_pose_batch`.
  * `intrinsics` must be run through `convert_intrinsics_after_center_crop_resize` for the
    render-resolution -> model-resolution change, and must keep their NEGATIVE focal lengths.

The generated canvas is `[v1_rgb | v1_action | v2_rgb | v2_action]`; segments 2 and 4 are the
action images, sliced exactly as `inference.py:311-328` does.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
import torch
from einops import rearrange

from training.utils import (
    convert_intrinsics_after_center_crop_resize,
    fuse_multiview_heatmaps_to_pose_torch,
    get_relative_pose_batch,
)

# Sweep bounds for the multi-view triangulation. Fixed constants in the spirit of the official
# hardcoded `near=0.5, far=1.0` (inference.py:306-307), recalibrated to the RLBench rig: over
# 192 view-episodes the camera-centre -> end-effector distance runs 0.642 m .. 1.793 m.
# Deliberately NOT derived per episode from ground truth -- that would leak the answer into the
# decoder and make the reported position error meaningless.
DEFAULT_NEAR, DEFAULT_FAR, DEFAULT_DEPTH_SAMPLES = 0.6, 1.8, 512


class ActionImagePolicy:
    """Wraps a trained checkpoint. The model is loaded ONCE; `run_policy` is called per replan."""

    def __init__(
        self,
        pipe,
        num_frames: int = 41,
        resolution: int = 256,
        cfg_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: int = 42,
        near: float = DEFAULT_NEAR,
        far: float = DEFAULT_FAR,
        num_depth_samples: int = DEFAULT_DEPTH_SAMPLES,
        gripper_mode: str = "paper",
        render_resolution: Optional[int] = None,
        axis_solver: str = "sphere",
        anchor_modality: str = "video",
    ) -> None:
        # 锚定模态 —— arm7 的四条输入路径靠 prompt tag + 锚定帧像素选路。默认 video,
        # 这样既有的 RGB 闭环(46.0%)的调用方一个字节都不用改。
        from eval.anchor_encode import ANCHOR_TEMPLATE, ANCHOR_CONDITIONING
        # "all" = 融合模式:四条路径各生成一次再几何中位合并。此时这两个字段只是【初始值】,
        # run_policy_fused 每次调用会逐路径改写再还原,所以放 video 这条部署路径当占位。
        if anchor_modality not in set(ANCHOR_TEMPLATE) | {"all"}:
            raise ValueError(f"anchor_modality must be one of "
                             f"{sorted(set(ANCHOR_TEMPLATE) | {'all'})}, got {anchor_modality!r}")
        self.fusion = anchor_modality == "all"
        self.anchor_modality = "video" if self.fusion else anchor_modality
        self.template = ANCHOR_TEMPLATE[self.anchor_modality]
        # NOTE two unrelated things are called "fusion" here. `self.fusion` (anchor_modality
        # "all") is the OLD 4-path geometric-median ENSEMBLE. anchor_modality "fusion" is the
        # 10-segment fusion CANVAS. Both names are kept because existing reports use them.
        self._conditioning = ANCHOR_CONDITIONING.get(self.anchor_modality)
        if anchor_modality == "fusion":
            self.template = ANCHOR_TEMPLATE["fusion"]
            self._conditioning = ANCHOR_CONDITIONING["fusion"]   # rgb_only
            self.anchor_modality = "video"      # the one stream a rollout actually observes
        self.pipe = pipe
        self.num_frames = num_frames
        self.resolution = resolution
        self.render_resolution = render_resolution or resolution
        self.cfg_scale = cfg_scale
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self.near, self.far = near, far
        self.num_depth_samples = num_depth_samples
        self.gripper_mode = gripper_mode
        # "ray" = the paper's decoder; "sphere" = the fork's. Recorded per rollout so no
        # reported success rate is ambiguous about which algorithm produced it.
        self.axis_solver = axis_solver
        self.n_calls = 0
        self.last_debug: dict = {}
        self.last_frames: Optional[List[np.ndarray]] = None
        self.last_confidence: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ conditioning
    def _build_inputs(self, rgb_views: np.ndarray, extr_views: np.ndarray, intr_views: np.ndarray):
        """`anchor [V,H,W,3] uint8`, `extr [V,4,4]`, `intr [V,3,3]` -> the pipeline's tensors.

        `rgb_views` 是历史参数名;实际接受**任意锚定模态编码后**的图像。depth/segmentation/
        normal 经各自 codec 编码后同样是 [V,H,W,3] uint8(codec 就是为了让所有模态共用 RGB
        空间),所以这里的归一化与铺帧逻辑对四条路径完全相同,一行不用改。

        The cameras are static within an RLBench episode, so the single live reading is tiled
        over the window. The RGB is likewise tiled: under the i2va plan only the first latent
        frame of each segment is used as conditioning, and repeating the current observation is
        the honest filler for the rest.
        """
        V, H, W, _ = rgb_views.shape
        if V != 2:
            raise ValueError(f"expected exactly 2 views, got {V}")
        if (H, W) != (self.resolution, self.resolution):
            raise ValueError(f"expected {self.resolution}^2 frames, got {(H, W)}")
        T = self.num_frames

        # video stream: [1, 3, 2T, H, W] in [-1, 1], views concatenated along time
        px = torch.from_numpy(rgb_views.astype(np.float32) / 127.5 - 1.0)   # [V,H,W,3]
        px = px.permute(0, 3, 1, 2)                                          # [V,3,H,W]
        video = torch.stack([px[v].unsqueeze(1).repeat(1, T, 1, 1) for v in range(V)], dim=0)
        video = torch.cat([video[0], video[1]], dim=1).unsqueeze(0)          # [1,3,2T,H,W]

        # extrinsics: ABSOLUTE camera-to-world, [view1_frames | view2_frames]
        extr = np.concatenate([np.repeat(extr_views[v][None], T, axis=0) for v in range(V)], axis=0)

        # camera: RELATIVE to view1 frame 0, flattened 3x4 -> 12
        base = extr_views[0]
        rel = [rearrange(torch.from_numpy(get_relative_pose_batch(base, extr[v * T:(v + 1) * T])),
                         "t c d -> t (c d)") for v in range(V)]
        camera = torch.cat(rel, dim=0).to(torch.float32).unsqueeze(0)        # [1,2T,12]

        # intrinsics: render resolution -> model resolution (identity when they already agree)
        conv = convert_intrinsics_after_center_crop_resize(
            [np.repeat(intr_views[v][None], T, axis=0) for v in range(V)],
            [(self.render_resolution, self.render_resolution)] * V,
            [(self.resolution, self.resolution)] * V,
        )
        intr = np.concatenate(conv, axis=0)                                  # [2T,3,3]

        if intr[0, 0, 0] >= 0:
            raise ValueError(
                "intrinsics have a non-negative focal length. RLBench records fx = fy = -351.68 "
                "at 256^2; a positive value means they were derived from FOV instead of read "
                "from obs.misc, which mirrors every projection without raising.")

        return (
            video,
            camera,
            torch.from_numpy(extr).to(torch.float32).unsqueeze(0),           # [1,2T,4,4]
            torch.from_numpy(intr).to(torch.float32).unsqueeze(0),           # [1,2T,3,3]
        )

    # ------------------------------------------------------------------ the policy
    @staticmethod
    def pose8_to_action7(pose8: np.ndarray) -> np.ndarray:
        """`[x,y,z,qx,qy,qz,qw,openness]` -> the `[x,y,z,ex,ey,ez,openness]` the encoder wants.

        Euler order and quaternion convention match `RLBenchMVDataset.get_7d_action`
        (dataset/rlbench.py:113-115), which is what produced the training action images.
        """
        from scipy.spatial.transform import Rotation as Rot

        pose8 = np.asarray(pose8, dtype=np.float64).reshape(8)
        euler = Rot.from_quat(pose8[3:7]).as_euler("xyz", degrees=False)
        return np.concatenate([pose8[:3], euler, pose8[7:]])

    @torch.no_grad()
    def run_policy(
        self,
        rgb_views: np.ndarray,   # 实为 anchor_views:已按 self.anchor_modality 编码的 [V,H,W,3] uint8
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        prompt: str,
        current_pose8: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """One replan. Returns `[num_frames, 8]` world-frame poses + openness.

        `current_pose8` is the arm's CURRENT end-effector pose. It matters: both this template
        path and the official i2va path anchor every segment on its first latent frame
        (`wan_video_action_images.py:531`), so the action segment's frame 0 is CONDITIONING,
        not prediction. Official inference fills it from the dataset's true first action; in
        closed loop the equivalent is the pose the robot is actually in. Passing zeros instead
        would anchor the model on "end-effector at the world origin, identity rotation" and
        every generated chunk would start by teleporting there.
        """
        if current_pose8 is None:
            raise ValueError(
                "current_pose8 is required: the action segment's first frame is conditioning, "
                "not prediction. Pass RolloutEnv.current_pose8(obs).")
        video, camera, extr, intr = self._build_inputs(rgb_views, extrinsics, intrinsics)
        device, dtype = self.pipe.device, torch.bfloat16
        T = self.num_frames
        a7 = self.pose8_to_action7(current_pose8)
        # Only frame 0 survives the mask, but tiling the current pose is the honest filler and
        # keeps the projected action image on-manifold for the frames that are masked anyway.
        action_7d = torch.from_numpy(np.repeat(a7[None], T, axis=0)).to(dtype).unsqueeze(0).to(device)

        frames = self.pipe(
            prompt=[prompt],
            negative_prompt="",
            template=self.template,
            # Every VISUAL modality in the template needs a tensor, including ones the rollout
            # cannot observe: prepare_template_inference_latents refuses an incomplete stream set
            # (better than silently packing a shorter sequence than the prompt describes). Their
            # CONTENT is irrelevant under rgb_only -- those latents are all overwritten by noise
            # before the first denoising step. Zeros rather than a copy of the RGB, so a bug that
            # failed to mark them absent surfaces as a blank prediction instead of as a
            # plausible-looking duplicate.
            streams=self._streams_for(video.to(device=device, dtype=dtype)),
            fully_given_modalities=[],            # i2va: first frame only, nothing given
            conditioning_mode=self._conditioning,
            camera=camera.to(device=device, dtype=dtype),
            action_7d=action_7d,
            extrinsics=extr.to(device=device, dtype=dtype),
            intrinsics=intr.to(device=device, dtype=dtype),
            height=self.resolution,
            width=self.resolution,
            num_frames=T,
            cfg_scale=self.cfg_scale,
            num_inference_steps=self.num_inference_steps,
            seed=self.seed if seed is None else seed,
            tiled=False,
            tile_size=(self.resolution // 16, self.resolution // 16),
            tile_stride=(self.resolution // 32, self.resolution // 32),
            enable_usp=False,
            cfg_parallel=False,
        )
        self.n_calls += 1
        # Keep the raw canvas so the caller can save what the model IMAGINED alongside what the
        # simulator actually did, for the same rollout and the same scene. Without this the two
        # videos come from different scripts on different scenes and cannot be compared.
        self.last_frames = [np.asarray(f) for f in frames]
        return self.decode(frames, extrinsics, intrinsics)

    def _streams_for(self, anchor_video):
        """anchor pixels -> {modality: tensor} covering every visual modality in the template."""
        import torch as _t
        from training.templates import VISUAL_MODALITIES, parse_template
        streams = {self.anchor_modality: anchor_video}
        for m in parse_template(self.template):
            if m in VISUAL_MODALITIES and m not in streams:
                streams[m] = _t.zeros_like(anchor_video)
        return streams

    # ------------------------------------------------------------------ fusion
    @staticmethod
    def _geometric_median(X: np.ndarray, iters: int = 60, eps: float = 1e-9) -> np.ndarray:
        """Weiszfeld。`X [k, T, 3]` -> `[T, 3]`。

        选几何中位而不是均值:它是稳健估计量,压尾不压中位 —— 离线实测(16 episode x 41 帧)
        p90 −10.1%、p75 −3.7%,而 p50 +1.7%。闭环里一次大失误就可能终结 episode,
        所以「压尾」正是要的东西。均值会被单条坏路径拖走(离线均值反而更差)。
        """
        y = np.median(X, axis=0)
        for _ in range(iters):
            d = np.maximum(np.linalg.norm(X - y, axis=-1, keepdims=True), eps)
            w = 1.0 / d
            y = (X * w).sum(0) / w.sum(0)
        return y

    @staticmethod
    def _quat_median(Q: np.ndarray) -> np.ndarray:
        """`[k, T, 4]` -> `[T, 4]`。先对齐半球再逐分量取中位,最后重归一化。

        四元数 q 与 −q 表示同一个旋转,不对齐就会把两个相同的旋转平均成垃圾。
        以第 0 路(RGB,部署模态、离线最可靠)为参考半球。
        """
        ref = Q[0]
        aligned = np.where((Q * ref).sum(-1, keepdims=True) < 0, -Q, Q)
        m = np.median(aligned, axis=0)
        n = np.linalg.norm(m, axis=-1, keepdims=True)
        return m / np.maximum(n, 1e-9)

    def fuse_poses(self, poses: list) -> np.ndarray:
        """`[k]` 个 `[T,8]` 位姿 -> 一个 `[T,8]`。位置用几何中位,旋转用半球对齐中位,
        开合度用中位。"""
        P = np.stack(poses)                                    # [k,T,8]
        out = np.empty_like(P[0])
        out[:, :3] = self._geometric_median(P[:, :, :3])
        out[:, 3:7] = self._quat_median(P[:, :, 3:7])
        out[:, 7] = np.median(P[:, :, 7], axis=0)
        return out

    @torch.no_grad()
    def run_policy_fused(self, anchors: dict, extrinsics, intrinsics, prompts: dict,
                         current_pose8=None, seed=None) -> np.ndarray:
        """四条路径各生成一次、各解一次,再几何中位融合。

        `anchors`  {modality: [V,H,W,3] uint8}
        `prompts`  {modality: str}  —— tag 必须跟着模态走

        代价:每次 replan 4 次扩散采样(~6.8 -> ~27 分钟/rollout)。
        """
        from eval.anchor_encode import ANCHOR_TEMPLATE
        saved_m, saved_t = self.anchor_modality, self.template
        poses, frames_by_mod = [], {}
        try:
            for m, img in anchors.items():
                self.anchor_modality, self.template = m, ANCHOR_TEMPLATE[m]
                poses.append(self.run_policy(img, extrinsics, intrinsics, prompts[m],
                                             current_pose8=current_pose8, seed=seed))
                frames_by_mod[m] = self.last_frames
        finally:
            self.anchor_modality, self.template = saved_m, saved_t
        self.last_frames = frames_by_mod.get("video", self.last_frames)
        self.last_poses_by_modality = {m: p for m, p in zip(anchors, poses)}
        return self.fuse_poses(poses)

    # ------------------------------------------------------------------ decoding
    def decode(self, frames, extr_views: np.ndarray, intr_views: np.ndarray) -> np.ndarray:
        """Generated canvas -> `[T, 8]`. Split out so it is testable without a GPU."""
        arr = np.stack([np.asarray(f) for f in frames])
        # WHICH decoded frames are the action segments is a property of the template, not a
        # constant. This used to assume four equal segments and take [q:2q] and [3q:4q]; on the
        # 10-segment fusion canvas those slices are depth-view0 and normal-view0, so it would
        # have decoded a pose out of a depth map and reported it as a policy -- silently, since
        # both are just RGB. Derived from templates.inference_plan, the same function that built
        # the mask.
        from training.templates import inference_plan, parse_template
        T = self.num_frames
        plan = inference_plan(parse_template(self.template), 2, mode=self._conditioning)
        spans, off = [], 0
        for sg in plan:
            n = 1 if sg.single_frame else T
            spans.append((sg, off, off + n))
            off += n
        if off != len(arr):
            raise RuntimeError(
                f"template {self.template!r} under {self._conditioning!r} plans {off} decoded "
                f"frames but the canvas has {len(arr)}; eval and pipeline layouts disagree"
            )
        act = [(b, e) for sg, b, e in spans if sg.modality == "action"]
        if len(act) != 2:
            raise RuntimeError(f"expected one action segment per view, got {len(act)}")
        heat = torch.from_numpy(np.stack([arr[b:e] for b, e in act], axis=1)).float()
        conv = convert_intrinsics_after_center_crop_resize(
            [np.repeat(intr_views[v][None], T, axis=0) for v in range(2)],
            [(self.render_resolution, self.render_resolution)] * 2,
            [(self.resolution, self.resolution)] * 2,
        )
        ext34 = torch.from_numpy(np.stack([np.repeat(extr_views[v][None, :3, :], T, axis=0)
                                           for v in range(2)], axis=1)).float()
        int33 = torch.from_numpy(np.stack([conv[0], conv[1]], axis=1)).float()

        pose8, conf = fuse_multiview_heatmaps_to_pose_torch(
            heat, ext34, int33,
            near=self.near, far=self.far, num_depth_samples=self.num_depth_samples,
            apply_edge_smoothing=False, gripper_mode=self.gripper_mode,
            axis_solver=self.axis_solver, return_confidence=True,
        )
        pose8 = pose8.numpy().astype(np.float64)
        # Per-frame rotation trustworthiness, from the axis-length check: the encoder puts both
        # axis points exactly 0.1 m from the position point, so a decoded length far from that
        # means the depth search settled on the wrong point along the ray. Measured on
        # ground-truth renders, flagged frames had a median axis error of 93.4 deg against
        # 0.3 deg for the rest. The caller can hold position on these instead of commanding a
        # target that is probably ~180 deg wrong -- which is what makes IK fail outright.
        self.last_confidence = conf.numpy().astype(bool)

        # Instrumentation for attributing a 0% success rate (EVAL_PLAN_CLOSEDLOOP.md Sec. 4.2):
        # a low r_peak means the model stopped drawing decodable action blobs, which is a very
        # different failure from "drew them in the wrong place".
        r_peak = heat[..., 0].amax(dim=(-1, -2))          # [T, V]
        self.last_debug = {
            "r_peak_mean": float(r_peak.mean()),
            "r_peak_weak_frac": float((r_peak < 100).float().mean()),
            "openness_mean": float(np.mean(pose8[:, 7])),
            "openness_bimodal_frac": float(np.mean((pose8[:, 7] < 0.2) | (pose8[:, 7] > 0.8))),
            "low_conf_frac": float(1.0 - self.last_confidence.mean()),
        }
        return pose8


def build_policy_pipeline(ckpt_path: str, resolution: int = 256, model_id: str = "Wan-AI/Wan2.2-TI2V-5B"):
    """Load the 5B pipeline once. Kept separate so rollout.py can batch a whole grid per load
    (a checkpoint load is ~3-4 minutes; paying it per rollout would dominate the campaign).

    Same argument shape `eval/eval_perception.py:_pipeline_args` builds, so both eval paths
    construct the model identically."""
    from types import SimpleNamespace

    from inference import build_pipeline

    return build_pipeline(SimpleNamespace(
        model_id=model_id,
        ckpt_path=ckpt_path,
        height=resolution,
        width=resolution,
        use_usp=False,
        cfg_parallel=False,
        dynamic_cache_schedule=False,
        torch_compile=False,
    ))
