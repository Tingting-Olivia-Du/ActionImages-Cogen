"""CPU guard for the F axis (modality dropout) and its train/eval parity.

The F axis exists because `assemble` hands every non-fully-given segment a free first latent
frame, so the 10-segment fusion canvas would otherwise carry ten free anchors -- see
templates.FUSION_MASK_MIX_PRESETS for why that breaks both the claim and the deployment story.

The load-bearing assertion in this file is PARITY: for each named regime, the mask that
`plan_segments` builds during TRAINING must be bit-identical to the mask
`prepare_template_inference_latents` builds when asked for that regime by name. Without it a
checkpoint can be evaluated under a conditioning it never trained on, and nothing would say so
-- the same train/eval split the prompt-tag scrub trap caused.

Run:  python tests/test_fusion_axis.py
"""
import collections
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import training.wan_video_action_images as pipeline_module
from training.wan_video_action_images import WanVideoActionImagesPipeline
from training.templates import (
    FUSION_REGIMES,
    assemble,
    fusion_regime_of,
    parse_fusion_mask_mix,
    parse_template,
    plan_segments,
    segment_spans,
)

FUSION = "video+depth+segmentation+normal+action"
MODS = parse_template(FUSION)


def _latents(B=1, C=2, T_l=3, H=4, W=4):
    lat = {m: [torch.zeros(B, C, T_l, H, W) for _ in range(2)] for m in MODS}
    cam = [torch.zeros(B, T_l, C, H, W) for _ in range(2)]
    return lat, cam


def test_layout_is_ten_segments():
    plan = plan_segments(MODS, 2, rng=random.Random(0), fusion_mask_mix=parse_fusion_mask_mix("F1"))
    assert len(plan) == 10, len(plan)
    order = [(s.modality, s.view) for s in plan]
    assert order == [(m, v) for v in range(2) for m in MODS], order
    lat, cam = _latents()
    latents, cam_emb, _ = assemble(plan, lat, cam)
    assert latents.shape[2] == 30 and cam_emb.shape[1] == 30, (latents.shape, cam_emb.shape)
    print("FUSION_LAYOUT_OK  10 segments, view-major, canonical order")


def test_regime_marginals_and_role_masks():
    """The realised regime distribution matches the declared mix, and each role's mask is exact."""
    mix = parse_fusion_mask_mix("F1")
    lat, cam = _latents()
    counts = collections.Counter()
    for seed in range(4000):
        plan = plan_segments(MODS, 2, rng=random.Random(seed), fusion_mask_mix=mix)
        counts[fusion_regime_of(plan)] += 1
        _, _, masks = assemble(plan, lat, cam)
        for (a, b), seg in zip(segment_spans(plan, lat), plan):
            got = int(masks[0, 0, a:b, 0, 0].sum())
            want = 0 if seg.absent else ((b - a) if seg.fully_given else 1)
            assert got == want, (seed, seg, got, want)
    for name, p in zip(FUSION_REGIMES, mix):
        got = counts[name] / 4000
        assert abs(got - p) < 0.02, (name, got, p)
    print(f"FUSION_REGIME_MARGINALS_OK  {dict(counts)}")


def test_never_degenerate():
    """Neither all-condition (zero loss) nor all-absent (no scene anchor) may be constructible."""
    lat, cam = _latents()
    for spec in ("F0", "F1", "F2"):
        mix = parse_fusion_mask_mix(spec)
        for seed in range(600):
            plan = plan_segments(MODS, 2, rng=random.Random(seed), fusion_mask_mix=mix)
            _, _, masks = assemble(plan, lat, cam)   # asserts both directions internally
            assert masks.any() and not masks.all()
    # and the guard fires when it should: every segment absent
    from training.templates import Segment
    bad = [Segment(m, v, False, False, absent=True) for v in range(2) for m in MODS]
    try:
        assemble(bad, lat, cam)
        raise SystemExit("!! all-absent plan was accepted; the scene-anchor assert is not firing")
    except AssertionError as e:
        assert "no scene anchor" in str(e), str(e)
    print("FUSION_NEVER_DEGENERATE_OK")


def test_f0_reproduces_all_anchor():
    """F0 must be exactly the pre-axis behaviour: every segment anchored, nothing absent."""
    mix = parse_fusion_mask_mix("F0")
    for seed in range(300):
        plan = plan_segments(MODS, 2, rng=random.Random(seed), fusion_mask_mix=mix)
        assert not any(s.absent or s.fully_given or s.single_frame for s in plan), plan
    print("FUSION_F0_IS_ALL_ANCHOR_OK")


def _inference_masks(mode=None, fully_given=None, absent=None):
    """Build the eval-side canvas for the fusion template on CPU."""
    pipe = object.__new__(WanVideoActionImagesPipeline)
    pipe.device = torch.device("cpu")
    pipe.torch_dtype = torch.float32
    pipe.encode_video = lambda x, **_kw: x[:, :2, :3]      # [B,3,T,H,W] -> [B,2,3,H,W]
    pipe.generate_noise = lambda shape, seed, device: torch.zeros(shape, device=device)

    old_plucker = pipeline_module.get_plucker_embeddings_torch
    old_5d = pipeline_module.project_actions_7d_to_5d_torch_batch
    old_rgb = pipeline_module.project_action_5d_to_rgb_torch
    pipeline_module.get_plucker_embeddings_torch = lambda extr, intr, hw: torch.zeros(
        extr.shape[0], extr.shape[1], hw[0], hw[1], 6
    )
    pipeline_module.project_actions_7d_to_5d_torch_batch = lambda a, e, i: torch.zeros(
        a.shape[0], a.shape[1], 7
    )
    pipeline_module.project_action_5d_to_rgb_torch = lambda a5, h, w: torch.zeros(
        a5.shape[0], a5.shape[1], h, w, 3
    )
    try:
        B, V, T, H, W = 1, 2, 3, 4, 4
        streams = {m: torch.ones(B, 3, V * T, H, W) for m in MODS if m != "action"}
        camera = torch.zeros(B, V * T, 12)
        intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(B, V * T, 1, 1)
        extrinsics = torch.eye(4)[:3].reshape(1, 1, 3, 4).repeat(B, V * T, 1, 1)
        action = torch.zeros(B, T, 7)
        _, _, masks, _, seg_num, spans = pipe.prepare_template_inference_latents(
            streams, camera, action, extrinsics, intrinsics, FUSION,
            fully_given, mode, {}, 42, absent_modalities=absent,
        )
    finally:
        pipeline_module.get_plucker_embeddings_torch = old_plucker
        pipeline_module.project_actions_7d_to_5d_torch_batch = old_5d
        pipeline_module.project_action_5d_to_rgb_torch = old_rgb
    return masks, seg_num, spans


def test_train_eval_parity():
    """THE load-bearing test: training's regime mask == eval's same-named conditioning mask."""
    mix = parse_fusion_mask_mix("F1")
    lat, cam = _latents(T_l=3)
    # find one training seed per regime
    seeds = {}
    for seed in range(4000):
        plan = plan_segments(MODS, 2, rng=random.Random(seed), fusion_mask_mix=mix)
        seeds.setdefault(fusion_regime_of(plan), plan)
        if len(seeds) == len(FUSION_REGIMES):
            break
    missing = set(FUSION_REGIMES) - set(seeds)
    assert not missing, missing

    for regime in FUSION_REGIMES:
        train_plan = seeds[regime]
        _, _, train_masks = assemble(train_plan, lat, cam)
        if regime == "one_out":
            # `one_out` names a DRAW, not a conditioning: eval says which modality to withhold.
            withheld = sorted({s.modality for s in train_plan if s.absent})
            eval_masks, _, _ = _inference_masks(absent=withheld)
        else:
            eval_masks, _, _ = _inference_masks(mode=regime)
        assert eval_masks.shape == train_masks.shape, (regime, eval_masks.shape, train_masks.shape)
        assert torch.equal(eval_masks, train_masks.to(eval_masks.dtype)), (
            f"{regime}: eval conditioning differs from the training regime of the same name.\n"
            f"  train cond frames: {train_masks[0,0,:,0,0].int().tolist()}\n"
            f"  eval  cond frames: {eval_masks[0,0,:,0,0].int().tolist()}"
        )
    print(f"FUSION_TRAIN_EVAL_PARITY_OK  {list(FUSION_REGIMES)}")


def test_f0f0_still_means_rgb_given():
    """The historical `f0f0` name must keep producing what reports say it produced."""
    a, _, _ = _inference_masks(mode="f0f0")
    b, _, _ = _inference_masks(mode="rgb_given")
    assert torch.equal(a, b)
    print("FUSION_F0F0_ALIAS_OK")


def test_eval_rejects_contradictions():
    for kwargs, needle in [
        (dict(fully_given=["video"], absent=["video"]), "mutually exclusive"),
        (dict(absent=["lidar"]), "not in"),
        (dict(mode="one_out"), "names a DRAW"),
        (dict(mode="rgb_only", fully_given=["video"]), "not both"),
    ]:
        try:
            _inference_masks(**kwargs)
            raise SystemExit(f"!! accepted a contradictory eval request: {kwargs}")
        except ValueError as e:
            assert needle in str(e), (kwargs, str(e))
    print("FUSION_EVAL_VALIDATION_OK")


if __name__ == "__main__":
    test_layout_is_ten_segments()
    test_regime_marginals_and_role_masks()
    test_never_degenerate()
    test_f0_reproduces_all_anchor()
    test_train_eval_parity()
    test_f0f0_still_means_rgb_given()
    test_eval_rejects_contradictions()
    print("ALL_FUSION_AXIS_TESTS_PASSED")
