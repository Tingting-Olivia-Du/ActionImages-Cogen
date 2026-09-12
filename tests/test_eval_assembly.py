"""CPU guard for deterministic perception-template inference assembly."""
import torch

import training.wan_video_action_images as pipeline_module
from training.wan_video_action_images import WanVideoActionImagesPipeline


def test_video_depth_perception_masks():
    pipe = object.__new__(WanVideoActionImagesPipeline)
    pipe.device = torch.device("cpu")
    pipe.torch_dtype = torch.float32
    pipe.encode_video = lambda x, **_kw: x[:, :2]
    pipe.generate_noise = lambda shape, seed, device: torch.zeros(shape, device=device)

    old_plucker = pipeline_module.get_plucker_embeddings_torch
    pipeline_module.get_plucker_embeddings_torch = lambda extr, intr, hw: torch.zeros(
        extr.shape[0], extr.shape[1], hw[0], hw[1], 6
    )
    try:
        B, V, T, H, W = 1, 2, 3, 4, 4
        streams = {
            "video": torch.ones(B, 3, V * T, H, W),
            "depth": torch.full((B, 3, V * T, H, W), 2.0),
        }
        camera = torch.zeros(B, V * T, 12)
        intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(B, V * T, 1, 1)
        action = torch.zeros(B, T, 7)
        _, latents, masks, _, seg_num, spans = pipe.prepare_template_inference_latents(
            streams, camera, action, None, intrinsics, "video+depth", ["video"], None, {}, 42
        )
    finally:
        pipeline_module.get_plucker_embeddings_torch = old_plucker

    assert seg_num == 4
    assert spans == [(0, 3), (3, 6), (6, 9), (9, 12)]
    assert tuple(latents.shape) == (B, 2, 12, H, W)
    # V0 fully given; D0 first frame only; V1 fully given; D1 first frame only.
    assert masks[:, :, 0:3].all()
    assert masks[:, :, 3].all() and not masks[:, :, 4:6].any()
    assert masks[:, :, 6:9].all()
    assert masks[:, :, 9].all() and not masks[:, :, 10:12].any()


def test_explicit_eval_mode_masks():
    """The three mode-mix choices are deterministic, not sampled during eval."""
    pipe = object.__new__(WanVideoActionImagesPipeline)
    pipe.device = torch.device("cpu")
    pipe.torch_dtype = torch.float32
    pipe.encode_video = lambda x, **_kw: x[:, :2]
    pipe.generate_noise = lambda shape, seed, device: torch.zeros(shape, device=device)

    old_plucker = pipeline_module.get_plucker_embeddings_torch
    pipeline_module.get_plucker_embeddings_torch = lambda extr, intr, hw: torch.zeros(
        extr.shape[0], extr.shape[1], hw[0], hw[1], 6
    )
    try:
        B, V, T, H, W = 1, 2, 3, 4, 4
        streams = {
            "video": torch.ones(B, 3, V * T, H, W),
            "depth": torch.full((B, 3, V * T, H, W), 2.0),
        }
        camera = torch.zeros(B, V * T, 12)
        intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(B, V * T, 1, 1)
        action = torch.zeros(B, T, 7)
        results = {}
        for mode in ("iiii", "fiii", "fifi"):
            _, _, mask, _, _, _ = pipe.prepare_template_inference_latents(
                streams, camera, action, None, intrinsics, "video+depth", [], mode, {}, 42
            )
            results[mode] = mask
    finally:
        pipeline_module.get_plucker_embeddings_torch = old_plucker

    iiii, fiii, fifi = (results[m] for m in ("iiii", "fiii", "fifi"))
    assert [bool(iiii[:, :, i].all()) for i in range(12)] == [
        True, False, False, True, False, False, True, False, False, True, False, False
    ]
    assert fiii[:, :, :3].all() and torch.equal(fiii[:, :, 3:], iiii[:, :, 3:])
    assert fifi[:, :, :3].all() and fifi[:, :, 6:9].all()
    assert torch.equal(fifi[:, :, 3:6], iiii[:, :, 3:6])
    assert torch.equal(fifi[:, :, 9:12], iiii[:, :, 9:12])


def test_single_frame_policy_spans():
    pipe = object.__new__(WanVideoActionImagesPipeline)
    pipe.device = torch.device("cpu")
    pipe.torch_dtype = torch.float32
    pipe.encode_video = lambda x, **_kw: x[:, :2]
    pipe.generate_noise = lambda shape, seed, device: torch.zeros(shape, device=device)
    old = (
        pipeline_module.get_plucker_embeddings_torch,
        pipeline_module.project_actions_7d_to_5d_torch_batch,
        pipeline_module.project_action_5d_to_rgb_torch,
    )
    pipeline_module.get_plucker_embeddings_torch = lambda extr, intr, hw: torch.zeros(
        extr.shape[0], extr.shape[1], hw[0], hw[1], 6
    )
    pipeline_module.project_actions_7d_to_5d_torch_batch = lambda a, e, i: torch.zeros(
        a.shape[0], a.shape[1], 7
    )
    pipeline_module.project_action_5d_to_rgb_torch = lambda a, h, w: torch.zeros(
        a.shape[0], a.shape[1], h, w, 3
    )
    try:
        B, V, T, H, W = 1, 2, 3, 4, 4
        streams = {"video": torch.ones(B, 3, V * T, H, W)}
        camera = torch.zeros(B, V * T, 12)
        extr = torch.eye(4).reshape(1, 1, 4, 4).repeat(B, V * T, 1, 1)
        intr = torch.eye(3).reshape(1, 1, 3, 3).repeat(B, V * T, 1, 1)
        action = torch.zeros(B, T, 7)
        _, latents, masks, _, _, spans = pipe.prepare_template_inference_latents(
            streams, camera, action, extr, intr, "video+action", [], "policy", {}, 42
        )
    finally:
        (
            pipeline_module.get_plucker_embeddings_torch,
            pipeline_module.project_actions_7d_to_5d_torch_batch,
            pipeline_module.project_action_5d_to_rgb_torch,
        ) = old
    assert spans == [(0, 1), (1, 4), (4, 5), (5, 8)]
    assert latents.shape[2] == 8
    assert masks[:, :, [0, 1, 4, 5]].all()
    assert not masks[:, :, [2, 3, 6, 7]].any()


def test_f0f0_has_no_target_anchor():
    pipe = object.__new__(WanVideoActionImagesPipeline)
    pipe.device = torch.device("cpu")
    pipe.torch_dtype = torch.float32
    pipe.encode_video = lambda x, **_kw: x[:, :2]
    pipe.generate_noise = lambda shape, seed, device: torch.zeros(shape, device=device)
    old_plucker = pipeline_module.get_plucker_embeddings_torch
    pipeline_module.get_plucker_embeddings_torch = lambda extr, intr, hw: torch.zeros(
        extr.shape[0], extr.shape[1], hw[0], hw[1], 6
    )
    try:
        B, V, T, H, W = 1, 2, 3, 4, 4
        streams = {"video": torch.ones(B, 3, V*T, H, W), "depth": torch.ones(B, 3, V*T, H, W)}
        camera = torch.zeros(B, V*T, 12)
        intr = torch.eye(3).reshape(1, 1, 3, 3).repeat(B, V*T, 1, 1)
        action = torch.zeros(B, T, 7)
        _, _, masks, _, _, spans = pipe.prepare_template_inference_latents(
            streams, camera, action, None, intr, "video+depth", [], "f0f0", {}, 42
        )
    finally:
        pipeline_module.get_plucker_embeddings_torch = old_plucker
    assert spans == [(0, 3), (3, 6), (6, 9), (9, 12)]
    assert masks[:, :, 0:3].all() and masks[:, :, 6:9].all()
    assert not masks[:, :, 3:6].any() and not masks[:, :, 9:12].any()


if __name__ == "__main__":
    test_video_depth_perception_masks()
    test_explicit_eval_mode_masks()
    test_single_frame_policy_spans()
    test_f0f0_has_no_target_anchor()
    print("EVAL_ASSEMBLY_OK")
