"""The policy's conditioning tensors must match what the training dataset emits.

`eval/policy.py` builds `video / camera / extrinsics / intrinsics` from a LIVE RLBench
observation, while training builds them from `RLBenchSelfgenDataset`. If the two disagree on a
convention, nothing raises -- the model is simply conditioned on a camera trajectory that does
not describe the pixels, and the rollout scores the bug instead of the checkpoint.

Three conventions, each verified against the dataset's own output rather than against prose:

  * `extrinsics` -- ABSOLUTE camera-to-world, `[view1_frames | view2_frames]` along time. This
    is what puts the decoded pose in world coordinates for EndEffectorPoseViaIK(frame=WORLD).
  * `camera`     -- RELATIVE to view1 frame 0, 3x4 flattened to 12. A DIFFERENT quantity from
    `extrinsics`; swapping them is the failure this test exists to catch.
  * `intrinsics` -- rescaled for render->model resolution, focal lengths still NEGATIVE.

Plus the decode side: the canvas layout `[v1_rgb | v1_action | v2_rgb | v2_action]` and a
round trip from a known 7-DoF trajectory back through `ActionImagePolicy.decode`.

Pure CPU, no simulator, no model. Run: python tests/test_policy_conditioning.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from einops import rearrange
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.policy import ActionImagePolicy
from training.utils import (
    convert_intrinsics_after_center_crop_resize,
    get_relative_pose_batch,
    project_action_5d_to_rgb_torch,
    project_actions_7d_to_5d_torch_batch,
)

RES, T = 256, 41


def look_at(eye, target, up=np.array([0.0, 0.0, 1.0])):
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    e = np.eye(4)
    e[:3, :3] = np.stack([r, u, f], axis=1)
    e[:3, 3] = eye
    return e


# A stand-in for RLBench's front/overhead rig, including the negative focal lengths it records.
CENTER = np.array([0.25, 0.0, 0.95])
EXTR = np.stack([look_at(np.array([1.35, 0.0, 1.58]), CENTER),
                 look_at(np.array([0.25, -1.25, 1.45]), CENTER)])
K = np.array([[-351.68, 0.0, 128.0], [0.0, -351.68, 128.0], [0.0, 0.0, 1.0]])
INTR = np.stack([K, K])
RGB = (np.random.RandomState(0).rand(2, RES, RES, 3) * 255).astype(np.uint8)

policy = ActionImagePolicy(pipe=None, num_frames=T, resolution=RES)

# ------------------------------------------------------------------ 1. conditioning tensors
video, camera, extr, intr = policy._build_inputs(RGB, EXTR, INTR)
assert video.shape == (1, 3, 2 * T, RES, RES), video.shape
assert camera.shape == (1, 2 * T, 12), camera.shape
assert extr.shape == (1, 2 * T, 4, 4), extr.shape
assert intr.shape == (1, 2 * T, 3, 3), intr.shape
print(f"CONDITIONING_SHAPES_OK  video{tuple(video.shape)} camera{tuple(camera.shape)} "
      f"extr{tuple(extr.shape)} intr{tuple(intr.shape)}")

# pixels: [-1, 1], view-major along time, and view1's block really is view1
assert -1.0 <= float(video.min()) and float(video.max()) <= 1.0
want_v0 = torch.from_numpy(RGB[0].astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
assert torch.allclose(video[0, :, 0], want_v0, atol=1e-6)
assert torch.allclose(video[0, :, T], torch.from_numpy(
    RGB[1].astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1), atol=1e-6)
assert torch.allclose(video[0, :, 0], video[0, :, T - 1], atol=1e-6), "view1 block must be tiled"
print("VIDEO_STREAM_LAYOUT_OK  [-1,1], view1 frames 0..T-1 then view2 frames T..2T-1")

# extrinsics are ABSOLUTE camera-to-world, tiled per view
assert np.abs(extr[0, :T].numpy() - EXTR[0]).max() < 1e-5
assert np.abs(extr[0, T:].numpy() - EXTR[1]).max() < 1e-5
print("EXTRINSICS_ARE_ABSOLUTE_C2W_OK")

# camera is RELATIVE to view1 frame 0 -- recomputed independently, the same way
# tests/test_selfgen_alignment.py checks the training sample
base = EXTR[0]
want_cam = torch.cat([
    rearrange(torch.from_numpy(get_relative_pose_batch(base, np.repeat(EXTR[v][None], T, 0))),
              "t c d -> t (c d)") for v in range(2)], dim=0).to(torch.float32)
assert torch.allclose(camera[0], want_cam, atol=1e-5), (camera[0] - want_cam).abs().max()
# and it is genuinely a DIFFERENT tensor from the flattened extrinsics
flat_extr = extr[0, :, :3, :].reshape(2 * T, 12)
assert not torch.allclose(camera[0], flat_extr, atol=1e-3), (
    "camera and extrinsics came out identical -- the relative-pose step was skipped")
# view1's own frames are the identity pose relative to themselves
eye34 = torch.eye(4)[:3].reshape(-1)
assert torch.allclose(camera[0, 0], eye34, atol=1e-5), camera[0, 0]
print("CAMERA_IS_RELATIVE_TO_VIEW1_FRAME0_OK  (and distinct from extrinsics)")

# intrinsics: rescaled, focal lengths still negative
want_intr = np.concatenate(convert_intrinsics_after_center_crop_resize(
    [np.repeat(INTR[v][None], T, 0) for v in range(2)],
    [(RES, RES)] * 2, [(RES, RES)] * 2), axis=0)
assert np.abs(intr[0].numpy() - want_intr).max() < 1e-4
assert intr[0, 0, 0, 0] < 0, intr[0, 0, 0, 0]
print(f"INTRINSICS_RESCALED_AND_SIGNED_OK  fx={intr[0,0,0,0]:.2f}")

# a positive focal length must be rejected, not silently mirrored
try:
    policy._build_inputs(RGB, EXTR, np.abs(INTR))
except ValueError as exc:
    assert "focal" in str(exc)
    print("POSITIVE_FOCAL_LENGTH_REJECTED_OK")
else:
    raise AssertionError("a FOV-derived (positive-focal) intrinsic matrix went unnoticed")

for bad, why in [(RGB[:1], "1 view"), (np.zeros((3, RES, RES, 3), np.uint8), "3 views")]:
    try:
        policy._build_inputs(bad, EXTR, INTR)
    except ValueError:
        pass
    else:
        raise AssertionError(f"{why} should have been rejected")
print("VIEW_COUNT_VALIDATED_OK")

# ------------------------------------------- 1b. the current pose is CONDITIONING, not filler
# Both this path and the official i2va path anchor every segment on its first latent frame
# (wan_video_action_images.py:531), so the action segment's frame 0 is given to the model.
# Official inference fills it from the dataset's true first action; closed loop must fill it
# from where the arm actually is. Zeros would anchor on "origin, identity rotation".
try:
    policy.run_policy(RGB, EXTR, INTR, "do something", current_pose8=None)
except ValueError as exc:
    assert "current_pose8" in str(exc), exc
    print("MISSING_CURRENT_POSE_REJECTED_OK")
else:
    raise AssertionError("run_policy accepted a missing current_pose8 -- the action anchor "
                         "would silently be the world origin")

# the pose8 -> action7 conversion must match dataset/rlbench.py:113-115 exactly
for _ in range(200):
    q = Rot.random(random_state=int(np.random.randint(1 << 30)))
    quat = q.as_quat()
    pose8_probe = np.concatenate([[0.3, -0.1, 1.0], quat, [1.0]])
    a7 = ActionImagePolicy.pose8_to_action7(pose8_probe)
    want = q.as_euler("xyz", degrees=False)
    assert np.abs(Rot.from_euler("xyz", a7[3:6]).as_matrix() - q.as_matrix()).max() < 1e-9
    assert np.abs(a7[3:6] - want).max() < 1e-9, (a7[3:6], want)
    assert a7[0] == 0.3 and a7[6] == 1.0
print("POSE8_TO_ACTION7_MATCHES_DATASET_CONVENTION_OK  (200 random rotations, xyz euler, xyzw quat)")

# ---------------------------------------------------------- 2. decode: layout + round trip
# Build a canvas in the real layout, with GT action images in segments 2 and 4, and check the
# policy recovers the trajectory that generated them.
pos = np.stack([np.linspace(0.15, 0.40, T), np.linspace(-0.15, 0.20, T),
                np.linspace(0.90, 1.05, T)], axis=1)
eul = np.stack([np.linspace(2.9, 3.1, T), np.linspace(-0.4, 0.5, T),
                np.linspace(-1.0, 1.2, T)], axis=1)
openness = (np.arange(T) < T // 2).astype(np.float64)
a7 = torch.tensor(np.concatenate([pos, eul, openness[:, None]], axis=1), dtype=torch.float32)

action_imgs = []
for v in range(2):
    e = torch.tensor(EXTR[v], dtype=torch.float32)[None, None].repeat(1, T, 1, 1)
    i = torch.tensor(INTR[v], dtype=torch.float32)[None, None].repeat(1, T, 1, 1)
    a5 = project_actions_7d_to_5d_torch_batch(a7[None], e, i)
    action_imgs.append((project_action_5d_to_rgb_torch(a5, RES, RES)[0] * 255).numpy().astype(np.uint8))

rgb_seg = np.zeros((T, RES, RES, 3), dtype=np.uint8)
canvas = np.concatenate([rgb_seg, action_imgs[0], rgb_seg, action_imgs[1]], axis=0)
assert canvas.shape[0] == 4 * T

pose8 = policy.decode([c for c in canvas], EXTR, INTR)
assert pose8.shape == (T, 8), pose8.shape

pos_err = np.linalg.norm(pose8[:, :3] - pos, axis=1)
gt_rot = Rot.from_euler("xyz", eul).as_matrix()
rel = np.matmul(np.transpose(Rot.from_quat(pose8[:, 3:7]).as_matrix(), (0, 2, 1)), gt_rot)
rot_err = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
print(f"DECODE_ROUNDTRIP_OK  pos p95={np.percentile(pos_err,95):.4f} m  "
      f"rot p95={np.percentile(rot_err,95):.2f} deg")
assert np.percentile(pos_err, 95) < 0.02, np.percentile(pos_err, 95)
assert np.percentile(rot_err, 95) < 5.0, np.percentile(rot_err, 95)
# Openness carries a small QUANTIZATION bias that the float-domain test in
# tests/test_decode_6dof.py cannot see: the encoder writes a pedestal of 0.25, generated frames
# arrive as uint8, and 0.25 * 255 = 63.75 truncates to 63, so ghat_open reads (63/255)/0.25 =
# 0.9882 rather than 1.0. It is a property of the real pipeline, not of this test. The binary
# decision has ~0.49 of margin, so it is harmless -- but ghat must never be compared to 1.0
# exactly, and a drift well beyond this floor means something else is wrong.
assert np.abs(pose8[:, 7] - openness).max() < 0.02, np.abs(pose8[:, 7] - openness).max()
assert ((pose8[:, 7] > 0.5).astype(float) == openness).all(), "open/closed decision is wrong"
print(f"DECODE_GRIPPER_OK  ghat_open={pose8[0,7]:.4f} ghat_closed={pose8[T//2,7]:.4f} "
      f"(uint8 pedestal quantization floor is 63/63.75 = 0.9882)")

# The instrumentation the campaign uses to attribute a 0% success rate.
dbg = policy.last_debug
assert dbg["r_peak_mean"] > 200, dbg
assert dbg["r_peak_weak_frac"] == 0.0, dbg
assert dbg["openness_bimodal_frac"] == 1.0, dbg
print(f"POLICY_DEBUG_INSTRUMENTATION_OK  {dbg}")

# Swapping the two action segments must change the answer -- otherwise the layout slicing is
# not actually reading the views it claims to.
swapped = np.concatenate([rgb_seg, action_imgs[1], rgb_seg, action_imgs[0]], axis=0)
pose_sw = policy.decode([c for c in swapped], EXTR, INTR)
assert np.linalg.norm(pose_sw[:, :3] - pos, axis=1).mean() > 0.05, (
    "swapping view1/view2 action segments changed nothing -- the canvas layout is not being "
    "sliced per view, so a view mix-up would go undetected")
print("CANVAS_LAYOUT_NEGATIVE_CONTROL_OK  swapped segments are detected")

# A canvas whose length is not a multiple of 4 is a real failure mode (a truncated generation).
try:
    policy.decode([c for c in canvas[:-1]], EXTR, INTR)
except RuntimeError:
    print("RAGGED_CANVAS_REJECTED_OK")
else:
    raise AssertionError("a canvas that is not 4 equal segments went unnoticed")

print("ALL_POLICY_CONDITIONING_TESTS_PASSED")
