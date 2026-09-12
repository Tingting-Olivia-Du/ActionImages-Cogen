"""L1 regression guard: template-driven assembly must REPRODUCE the upstream training core.

HISTORY -- read this before "fixing" the test. Until 2026-08-10 this file compared the
*source text* of `forward` / `get_inputs` / the collator against the upstream commit, because
the fork's `<depth><action>` arm was a dataset-only change and the training core genuinely was
byte-identical. Its own docstring said:

    "If you are deliberately starting the 6-segment `<video><depth><action>` work, this test is
     SUPPOSED to fail -- that is the correct moment to renegotiate what the baseline means."

That moment arrived. `forward` now packs segments from `sample["streams"]` according to the
sample's task template, which is what makes the perception template (`video+depth`, RGB fully
given) expressible at all -- the substitution layout could not express it, because a depth
sample contained no RGB. So the guard changed KIND rather than being deleted: it now checks
BEHAVIOURAL equivalence, which is the strongest statement still available.

WHAT EQUIVALENCE MEANS NOW. Upstream's assembly is a pure function of a handful of decisions
(is there an action stream, is this the single-frame variant, which mode-mix branch). The
claim under test is: for template `video+action` with 2 views, given the same decisions, the
new code produces bit-identical latents, camera embedding and mask. It is NOT a claim about
RNG traces -- the action gate moved from `forward` into the dataset (so the prompt and the
layout are decided together), which necessarily consumes randomness in a different order.

The upstream code is transcribed literally below rather than imported, so this test keeps
working as a reference even if train.py is refactored again.

Run:  python /workspace/ttdu/ActionImages-Cogen/tests/test_forward_unchanged.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.templates import (  # noqa: E402
    ACTION_MASK_MIX_PRESETS,
    MODE_ALL_VISUAL_GIVEN_PROB,
    MODE_FIRST_SEGMENT_GIVEN_PROB,
    SINGLE_FRAME_VISUAL_PROB,
    assemble,
    parse_action_mask_mix,
    parse_template,
    plan_segments,
)

B, C, T_L, H, W = 1, 4, 3, 2, 2


class ScriptedRandom:
    """random.Random stand-in that returns a fixed sequence, so a decision path is a fixture."""

    def __init__(self, values):
        self._values = list(values)
        self._i = 0

    def random(self):
        if self._i >= len(self._values):
            raise AssertionError(
                f"plan_segments drew {self._i + 1} randoms, script only has {len(self._values)}"
            )
        v = self._values[self._i]
        self._i += 1
        return v

    def exhausted(self):
        return self._i == len(self._values)


def _lat():
    return torch.randn(B, C, T_L, H, W)


def _cam():
    return torch.randn(B, T_L, C, H, W)


# --------------------------------------------------------------------------------------
# Literal transcription of upstream train.py:286-346 (commit 291f71cd, the fork baseline).
# Do not "clean up": its value is that it is a copy, not a reimplementation.
# --------------------------------------------------------------------------------------
def upstream_action_branch(video_lat, action_lat, cam_lat, single_frame, p):
    video_src_latents, video_tgt_latents = video_lat
    action_src_latents, action_tgt_latents = action_lat
    camera_src_latents, camera_tgt_latents = cam_lat
    T_l = video_src_latents.shape[2]

    if single_frame:
        latents = torch.cat(
            [
                video_src_latents[:, :, [0]],
                action_src_latents,
                video_tgt_latents[:, :, [0]],
                action_tgt_latents,
            ],
            dim=2,
        )
        camera_emb = torch.cat(
            [
                camera_src_latents[:, [0]],
                camera_src_latents,
                camera_tgt_latents[:, [0]],
                camera_tgt_latents,
            ],
            dim=1,
        )
        masks = torch.zeros_like(latents, dtype=torch.bool)
        masks[:, :, 0, ...] = 1
        masks[:, :, 1, ...] = 1
        masks[:, :, T_l + 1, ...] = 1
        masks[:, :, T_l + 2, ...] = 1
    else:
        latents = torch.cat(
            [video_src_latents, action_src_latents, video_tgt_latents, action_tgt_latents], dim=2
        )
        camera_emb = torch.cat(
            [camera_src_latents, camera_src_latents, camera_tgt_latents, camera_tgt_latents], dim=1
        )
        masks = torch.zeros_like(latents, dtype=torch.bool)
        masks[:, :, 0, ...] = 1
        masks[:, :, T_l, ...] = 1
        masks[:, :, 2 * T_l, ...] = 1
        masks[:, :, 3 * T_l, ...] = 1
        if p < 0.9:  # 2 frame --> all video & action
            pass
        elif p < 0.95:  # 1st video --> 1st action + 2nd video & action
            masks[:, :, :T_l, ...] = 1
        else:  # video --> action
            masks[:, :, :T_l, ...] = 1
            masks[:, :, 2 * T_l : 3 * T_l, ...] = 1
    return latents, camera_emb, masks


def upstream_video_only(video_lat, cam_lat):
    video_src_latents, video_tgt_latents = video_lat
    camera_src_latents, camera_tgt_latents = cam_lat
    T_l = video_src_latents.shape[2]
    latents = torch.cat([video_src_latents, video_tgt_latents], dim=2)
    camera_emb = torch.cat([camera_src_latents, camera_tgt_latents], dim=1)
    masks = torch.zeros_like(latents, dtype=torch.bool)
    masks[:, :, 0, ...] = 1
    masks[:, :, T_l, ...] = 1
    return latents, camera_emb, masks


def _same(label, got, want):
    for name, g, w in zip(("latents", "camera_emb", "masks"), got, want):
        assert g.shape == w.shape, f"{label}: {name} shape {tuple(g.shape)} != {tuple(w.shape)}"
        assert torch.equal(g, w), f"{label}: {name} differs from upstream"


def test_video_action_matches_upstream():
    """Every decision path of upstream's action branch, reproduced bit-for-bit."""
    torch.manual_seed(0)
    video = [_lat(), _lat()]
    action = [_lat(), _lat()]
    cam = [_cam(), _cam()]
    lat_by_mod = {"video": video, "action": action}
    mods = parse_template("video+action")

    cases = [
        # label,                    scripted randoms,  is_rlbench, single_frame, p
        ("single-frame variant",    [0.05],            True,       True,         None),
        ("joint (p<0.90)",          [0.50, 0.50],      True,       False,        0.50),
        ("first segment given",     [0.50, 0.92],      True,       False,        0.92),
        ("v2a (p>=0.95)",           [0.50, 0.99],      True,       False,        0.99),
        # Non-rlbench: upstream evaluates `random.random() < 0.1` FIRST and the path check
        # second, so a low draw is consumed but does NOT trigger the variant.
        ("low draw, not rlbench",   [0.05, 0.50],      False,      False,        0.50),
    ]
    for label, script, is_rlbench, single_frame, p in cases:
        rng = ScriptedRandom(script)
        plan = plan_segments(mods, num_views=2, rng=rng, is_rlbench=is_rlbench)
        assert rng.exhausted(), f"{label}: unexpected number of random draws"
        got = assemble(plan, lat_by_mod, cam)
        want = upstream_action_branch(video, action, cam, single_frame, p)
        _same(label, got, want)
        print(f"  upstream-equivalent: {label}")
    print("VIDEO_ACTION_MATCHES_UPSTREAM_OK")


def test_video_only_matches_upstream():
    """Pi={video} is upstream's `else` branch -- where action dropout and zero-action data land.

    A single-modality template must draw NOTHING: "give the anchor whole" would leave the
    sequence with no predicted position, so `valid.sum()` would be 0, the loss would clamp to
    a reported 0.0, and the step would cost a full forward and backward while teaching nothing
    -- visible only as a suspiciously good loss curve.
    """
    torch.manual_seed(1)
    video = [_lat(), _lat()]
    cam = [_cam(), _cam()]
    for script in ([], ):  # no draws at all
        rng = ScriptedRandom(script)
        plan = plan_segments(parse_template("video"), num_views=2, rng=rng, is_rlbench=True)
        assert rng.exhausted()
        got = assemble(plan, {"video": video}, cam)
        _same("video-only", got, upstream_video_only(video, cam))
    print("VIDEO_ONLY_MATCHES_UPSTREAM_OK")


def test_never_fully_conditioned():
    """No reachable (template, decision) combination may leave zero prediction targets."""
    import itertools
    import random as _r

    torch.manual_seed(5)
    lat = {m: [_lat(), _lat()] for m in
           ("video", "depth", "segmentation", "normal", "action")}
    cam = [_cam(), _cam()]
    names = ["video", "video+action", "video+depth", "video+segmentation",
             "video+depth+action", "depth+action", "video+depth+segmentation+action",
             # arm7's menu. A mask mix is only safe if NO setting of it can produce a fully
             # conditioned sequence, so sweep every preset, not just the default.
             "segmentation+action", "normal+action"]
    rng = _r.Random(0)
    mixes = [None] + [parse_action_mask_mix(k) for k in sorted(ACTION_MASK_MIX_PRESETS)]
    for name, amm, _ in itertools.product(names, mixes, range(20)):
        mods = parse_template(name)
        for is_rlbench in (True, False):
            plan = plan_segments(mods, num_views=2, rng=rng, is_rlbench=is_rlbench,
                                 action_mask_mix=amm)
            _, _, masks = assemble(plan, lat, cam)   # asserts internally
            assert (~masks).any(), (name, amm)
    print("NEVER_FULLY_CONDITIONED_OK")


def test_segment_order_is_view_major():
    """[v1_video | v1_action | v2_video | v2_action] -- modality inner, view outer."""
    plan = plan_segments(parse_template("video+action"), num_views=2,
                         rng=ScriptedRandom([0.5, 0.5]), is_rlbench=True)
    assert [(s.view, s.modality) for s in plan] == [
        (0, "video"), (0, "action"), (1, "video"), (1, "action")
    ], [(s.view, s.modality) for s in plan]

    # Canonical order is video -> depth -> segmentation -> action regardless of how the
    # template was spelled, so <depth><action> and <action><depth> assemble identically.
    for spelling in ("video+depth+action", "action+depth+video", "depth+action+video"):
        plan = plan_segments(parse_template(spelling), num_views=2,
                             rng=ScriptedRandom([0.5, 0.5]), is_rlbench=True)
        assert [(s.view, s.modality) for s in plan] == [
            (0, "video"), (0, "depth"), (0, "action"),
            (1, "video"), (1, "depth"), (1, "action"),
        ], spelling
    print("SEGMENT_ORDER_OK")


def test_perception_template_gives_rgb_whole():
    """`video+depth`: the RGB segments are a full condition, the depth segments are predicted.

    This is the assertion that distinguishes PERCEPTION from generation. If the video segments
    were only anchored by their first frame, the task would be "generate an RGB video and a
    depth video that agree", not "read the depth off this RGB".
    """
    torch.manual_seed(2)
    lat = {"video": [_lat(), _lat()], "depth": [_lat(), _lat()]}
    cam = [_cam(), _cam()]
    plan = plan_segments(parse_template("video+depth"), num_views=2,
                         rng=ScriptedRandom([0.0]), is_rlbench=True)
    latents, camera_emb, masks = assemble(plan, lat, cam)

    assert latents.shape[2] == 4 * T_L, latents.shape
    assert camera_emb.shape[1] == 4 * T_L, camera_emb.shape
    given = masks[0, 0, :, 0, 0].tolist()
    assert given == (
        [True] * T_L                       # v1 rgb: whole segment given
        + [True] + [False] * (T_L - 1)     # v1 depth: first frame only -> predicted
        + [True] * T_L                     # v2 rgb
        + [True] + [False] * (T_L - 1)     # v2 depth
    ), given

    # ... and the low-probability fallback keeps some joint-generation diversity.
    plan = plan_segments(parse_template("video+depth"), num_views=2,
                         rng=ScriptedRandom([0.99]), is_rlbench=True)
    _, _, masks = assemble(plan, lat, cam)
    assert masks[0, 0, :, 0, 0].sum().item() == 4, "fallback should give first frames only"
    print("PERCEPTION_MASK_OK")


def test_cogeneration_template_is_six_segments():
    """`video+depth+action`: the cell doc 15 §2.5 proved unreachable under the old design."""
    torch.manual_seed(3)
    lat = {"video": [_lat(), _lat()], "depth": [_lat(), _lat()], "action": [_lat(), _lat()]}
    cam = [_cam(), _cam()]
    plan = plan_segments(parse_template("video+depth+action"), num_views=2,
                         rng=ScriptedRandom([0.5, 0.5]), is_rlbench=True)
    latents, camera_emb, masks = assemble(plan, lat, cam)
    assert len(plan) == 6, len(plan)
    assert latents.shape[2] == 6 * T_L, latents.shape
    assert camera_emb.shape[1] == 6 * T_L, camera_emb.shape
    # One clean first frame per segment: without it a modality cannot announce itself
    # (doc 15 §3), which is the only mechanism that tells the DiT what to draw.
    assert masks[0, 0, :, 0, 0].sum().item() == 6
    assert [i for i, v in enumerate(masks[0, 0, :, 0, 0].tolist()) if v] == [
        k * T_L for k in range(6)
    ]

    # Camera embedding is per VIEW, not per modality: a view's rgb/depth/action segments must
    # all receive that view's camera. (Upstream's `[c_src, c_src, c_tgt, c_tgt]` said the same
    # thing for 4 segments.)
    for k in range(3):
        assert torch.equal(camera_emb[:, k * T_L : (k + 1) * T_L], cam[0])
    for k in range(3, 6):
        assert torch.equal(camera_emb[:, k * T_L : (k + 1) * T_L], cam[1])
    print("COGENERATION_LAYOUT_OK")


def test_substitution_template_still_reachable():
    """`depth+action` reproduces the fork's previous layout: no RGB anywhere in the sequence.

    Kept working on purpose -- it is a legitimate arm (a depth-space world model) as long as
    it is not reported as perception.
    """
    torch.manual_seed(4)
    depth = [_lat(), _lat()]
    action = [_lat(), _lat()]
    cam = [_cam(), _cam()]
    plan = plan_segments(parse_template("depth+action"), num_views=2,
                         rng=ScriptedRandom([0.5, 0.99]), is_rlbench=True)
    got = assemble(plan, {"depth": depth, "action": action}, cam)
    # Under substitution, depth occupies the slot upstream called "video" -- so the v2a branch
    # must give the DEPTH segments whole, exactly as upstream gave the video segments.
    want = upstream_action_branch(depth, action, cam, single_frame=False, p=0.99)
    _same("depth+action v2a", got, want)
    print("SUBSTITUTION_TEMPLATE_OK")


def test_action_mask_mix_a0_is_the_upstream_default():
    """The A axis must be a no-op at its default, on EVERY decision path.

    `plan_segments` inverts the (iiii, fiii, fifi, policy) marginal back into the two
    thresholds upstream actually compares against. If that arithmetic drifts, A0 stops meaning
    "upstream" -- and nothing else would notice, because the marginal would still look right in
    aggregate while individual seeds started landing in different branches. So compare the two
    at the threshold BOUNDARIES, where an off-by-epsilon shows up as a different plan.
    """
    a0 = ACTION_MASK_MIX_PRESETS["A0"]
    # The derived thresholds must be the upstream literals to full float precision, not just
    # close: `p < t` at exactly t is the boundary these fixtures probe.
    rest = 1.0 - a0[3]
    assert a0[3] == SINGLE_FRAME_VISUAL_PROB, a0
    assert a0[0] / rest == MODE_FIRST_SEGMENT_GIVEN_PROB, a0
    assert (a0[0] + a0[1]) / rest == MODE_ALL_VISUAL_GIVEN_PROB, a0

    torch.manual_seed(11)
    lat = {m: [_lat(), _lat()] for m in ("video", "depth", "segmentation", "normal", "action")}
    cam = [_cam(), _cam()]
    # Values that straddle every boundary in both draws, including the exact thresholds.
    draws = (0.0, 0.05, 0.099999, 0.1, 0.5, 0.899999, 0.9, 0.93, 0.949999, 0.95, 0.99)
    for name in ("video+action", "depth+action", "segmentation+action", "normal+action"):
        mods = parse_template(name)
        for d0 in draws:
            for d1 in draws:
                for is_rlbench in (True, False):
                    default = assemble(plan_segments(
                        mods, 2, rng=ScriptedRandom([d0, d1]), is_rlbench=is_rlbench), lat, cam)
                    explicit = assemble(plan_segments(
                        mods, 2, rng=ScriptedRandom([d0, d1]), is_rlbench=is_rlbench,
                        action_mask_mix=a0), lat, cam)
                    _same(f"{name} A0 vs default @({d0},{d1},rlbench={is_rlbench})",
                          explicit, default)
    print("ACTION_MASK_MIX_A0_IS_UPSTREAM_OK")


def test_action_mask_mix_a1_reaches_every_branch():
    """arm7's A1 = (iiii .75, fiii 0, fifi .05, policy .20).

    Two properties that a marginal alone does not pin down: policy mode must fire on the LOW
    end of the first draw (upstream's ordering -- a mix that inverted it would train a
    different sequence from the same seed), and FIII must be UNREACHABLE at weight zero rather
    than merely rare.
    """
    a1 = parse_action_mask_mix("A1")
    mods = parse_template("depth+action")

    plan = plan_segments(mods, 2, rng=ScriptedRandom([0.19]), is_rlbench=True, action_mask_mix=a1)
    assert all(s.single_frame for s in plan if s.modality == "depth"), plan
    assert not any(s.single_frame for s in plan if s.modality == "action"), plan

    # 0.20 is the boundary: `<` means it belongs to the second draw, not to policy mode.
    plan = plan_segments(mods, 2, rng=ScriptedRandom([0.20, 0.5]), is_rlbench=True,
                         action_mask_mix=a1)
    assert not any(s.single_frame for s in plan), plan
    assert not any(s.fully_given for s in plan), plan          # IIII

    # p_fiii == 0 collapses the two thresholds, so no second-draw value can select FIII.
    for d1 in (0.0, 0.5, 0.937499, 0.9375, 0.93751, 0.99, 0.999999):
        plan = plan_segments(mods, 2, rng=ScriptedRandom([0.5, d1]), is_rlbench=True,
                             action_mask_mix=a1)
        given = [s.modality for s in plan if s.fully_given]
        assert given in ([], ["depth", "depth"]), (d1, given)  # IIII or FIFI, never FIII
    print("ACTION_MASK_MIX_A1_OK")


def test_arm7_substitution_layouts():
    """`segmentation+action` and `normal+action` lay out as `S0|A0|S1|A1` / `N0|A0|N1|A1`.

    The decoder in eval/eval_action.py slices the action segments at index 1 and 3 for all four
    of arm7's templates. That is only true while the action segment stays last within a view,
    so pin it for the two modalities arm7 introduces.
    """
    for modality, mix in (("segmentation", None), ("normal", parse_action_mask_mix("A1"))):
        plan = plan_segments(parse_template(f"{modality}+action"), num_views=2,
                             rng=ScriptedRandom([0.5, 0.5]), is_rlbench=True,
                             action_mask_mix=mix)
        assert [(s.view, s.modality) for s in plan] == [
            (0, modality), (0, "action"), (1, modality), (1, "action")
        ], (modality, plan)
    print("ARM7_SUBSTITUTION_LAYOUTS_OK")


def test_rejects_malformed_templates():
    for bad in (
        "",                # nothing
        "+",               # nothing after splitting
        "video+video",     # duplicate modality -> ambiguous segment count
        "rgb+action",      # not a modality
        "action",          # no visual anchor: no clean frame to ground the scene or the camera
    ):
        try:
            parse_template(bad)
        except ValueError:
            continue
        raise AssertionError(f"template {bad!r} should have been rejected")
    # Whitespace and separator noise are tolerated; ORDER never matters.
    assert parse_template("video+action ") == ("video", "action")
    assert parse_template("action+video") == ("video", "action")
    print("TEMPLATE_VALIDATION_OK")


def main():
    test_video_action_matches_upstream()
    test_video_only_matches_upstream()
    test_never_fully_conditioned()
    test_segment_order_is_view_major()
    test_perception_template_gives_rgb_whole()
    test_cogeneration_template_is_six_segments()
    test_substitution_template_still_reachable()
    test_action_mask_mix_a0_is_the_upstream_default()
    test_action_mask_mix_a1_reaches_every_branch()
    test_arm7_substitution_layouts()
    test_rejects_malformed_templates()
    print("ALL_FORWARD_UNCHANGED_TESTS_PASSED")


if __name__ == "__main__":
    main()
