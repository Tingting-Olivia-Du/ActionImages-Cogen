"""Scene-role codec acceptance (SEGMENTATION_SCENE_ROLES_PLAN.md §12.2).

Run: python tests/test_scene_seg_codec.py   (prints SCENE_SEG_CODEC_OK on success)

The palette-geometry tests are the load-bearing ones. The role<->colour assignment is not a
cosmetic choice: TAU=40 in seg_codec is calibrated against the palette's MINIMUM pairwise
distance, so whichever role pair sits at that minimum is the pair most likely to be confused
after VAE noise. test_tightest_pair_is_not_two_cooccurring_foreground_roles pins the swap that
keeps that pair harmless -- without it, a future "let's reorder the palette" edit silently puts
distractor and tool back at 98.3 apart and nothing else in the suite would notice.
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.percep.seg_codec import (  # noqa: E402
    ROLE_TO_LABEL,
    SCENE_ROLE_PALETTE,
    SCENE_ROLES,
    TAU,
    UNKNOWN_LABEL,
    build_role_lut,
    build_scene_prompt,
    decode_scene_roles,
    encode_scene_roles,
    scene_role_iou,
    scene_role_labels,
)

FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  ok: {msg}")


def _pairwise():
    p = SCENE_ROLE_PALETTE.astype(np.float64)
    return sorted(
        (float(np.linalg.norm(p[i] - p[j])), SCENE_ROLES[i], SCENE_ROLES[j])
        for i, j in itertools.combinations(range(len(SCENE_ROLES)), 2)
    )


def test_palette_shape():
    check(SCENE_ROLE_PALETTE.shape == (9, 3), "palette is 9x3")
    check(SCENE_ROLE_PALETTE.dtype == np.uint8, "palette is uint8")
    check(len(SCENE_ROLES) == 9, "9 role names")
    check(tuple(SCENE_ROLE_PALETTE[0]) == (0, 0, 0), "background is black")
    # Every colour distinct: two roles sharing a colour would be undecodable.
    uniq = {tuple(c) for c in SCENE_ROLE_PALETTE}
    check(len(uniq) == 9, "all 9 colours distinct")


def test_tightest_pair_is_not_two_cooccurring_foreground_roles():
    """The plan's palette swap (tool <-> unknown). See module docstring."""
    d = _pairwise()
    tightest_dist, a, b = d[0]
    check(abs(tightest_dist - 98.3) < 1.0, f"palette minimum is still 98.3 (got {tightest_dist:.1f})")
    check(
        "unknown" in (a, b),
        f"tightest pair ({a}/{b}, {tightest_dist:.1f}) must involve `unknown`, which is absent "
        f"from valid training data -- putting two real foreground roles there is the bug this "
        f"test exists to prevent",
    )
    # TAU is calibrated to stay under half the palette minimum, else the background gate and the
    # nearest-colour boundary would overlap.
    check(TAU < tightest_dist / 2, f"TAU={TAU} < min_pairwise/2 = {tightest_dist / 2:.1f}")


def test_lut_unmapped_is_unknown_not_background():
    lut = build_role_lut({5: "target", 7: "goal"})
    check(lut[5] == ROLE_TO_LABEL["target"], "mapped handle -> its role")
    check(lut[7] == ROLE_TO_LABEL["goal"], "mapped handle -> its role (2)")
    check(lut[999] == UNKNOWN_LABEL, "UNMAPPED handle -> unknown, NOT background")
    check(lut[0] == ROLE_TO_LABEL["background"], "handle 0 -> background")
    try:
        build_role_lut({5: "not_a_role"})
        check(False, "bad role name rejected")
    except ValueError:
        check(True, "bad role name rejected")


def test_clean_roundtrip_is_exact():
    """§12.2: no-VAE roundtrip per-class IoU must be exactly 1.0."""
    rng = np.random.default_rng(0)
    handles = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80], dtype=np.uint16)
    roles = dict(zip(handles.tolist(), SCENE_ROLES))
    roles[0] = "background"
    lut = build_role_lut(roles)
    handle_map = rng.choice(handles, size=(3, 32, 32)).astype(np.uint16)

    rgb = encode_scene_roles(handle_map, lut)
    check(rgb.shape == (3, 32, 32, 3) and rgb.dtype == np.uint8, "encode shape/dtype [T,H,W,3] uint8")

    gt_labels = scene_role_labels(handle_map, lut)
    dec = decode_scene_roles(rgb, present_roles=SCENE_ROLES)
    ious = scene_role_iou(dec, gt_labels, roles=SCENE_ROLES)
    worst = min(v["iou"] for v in ious.values())
    check(worst == 1.0, f"clean roundtrip per-class IoU == 1.0 for all 9 roles (worst {worst})")


def test_single_pixel_target_survives():
    """§12.2: a 1-2 px target must not be removed. Appendix A pruning would delete it."""
    lut = build_role_lut({1: "background", 2: "target"})
    handle_map = np.ones((1, 64, 64), dtype=np.uint16)
    handle_map[0, 10, 10] = 2          # one single isolated pixel
    handle_map[0, 40, 40] = 2          # and a second, far away
    rgb = encode_scene_roles(handle_map, lut)
    dec = decode_scene_roles(rgb, present_roles=["background", "target"])
    check(int(dec["target"].sum()) == 2, f"both 1-px targets survive decode (got {int(dec['target'].sum())})")


def test_subset_decode_does_not_let_absent_roles_steal_pixels():
    """§8.3: restricting to present roles is an invariant, not an optimisation."""
    lut = build_role_lut({1: "background", 2: "distractor"})
    handle_map = np.full((1, 16, 16), 2, dtype=np.uint16)
    rgb = encode_scene_roles(handle_map, lut).astype(np.float32)
    # Push distractor 60% of the way toward `unknown` -- past the midpoint, i.e. the noise level
    # a VAE roundtrip can plausibly produce on the palette's tightest pair. Derived from the
    # palette rather than hardcoded so the test follows any future recolouring.
    d = SCENE_ROLE_PALETTE[ROLE_TO_LABEL["distractor"]].astype(np.float32)
    u = SCENE_ROLE_PALETTE[ROLE_TO_LABEL["unknown"]].astype(np.float32)
    rgb = np.clip(d + 0.6 * (u - d), 0, 255).astype(np.uint8) * np.ones_like(rgb, dtype=np.uint8)

    full = decode_scene_roles(rgb, present_roles=SCENE_ROLES)
    subset = decode_scene_roles(rgb, present_roles=["background", "distractor"])
    check(
        int(subset["distractor"].sum()) == 256,
        f"subset decode keeps all 256 px as distractor (got {int(subset['distractor'].sum())})",
    )
    check(
        int(full["unknown"].sum()) > 0,
        "full-palette decode DOES lose pixels to `unknown` here -- which is exactly why the "
        "subset restriction exists",
    )


def test_background_always_competes():
    """Omitting background from present_roles must not force black pixels into a foreground role."""
    lut = build_role_lut({1: "background", 2: "target"})
    handle_map = np.ones((1, 8, 8), dtype=np.uint16)
    rgb = encode_scene_roles(handle_map, lut)
    dec = decode_scene_roles(rgb, present_roles=["target"])   # background deliberately omitted
    check("background" in dec, "background re-inserted when caller omits it")
    check(int(dec["target"].sum()) == 0, "all-black frame decodes to zero target pixels")


def test_unknown_is_visible_not_silent():
    """§7.3: an unmapped handle must show up as `unknown`, never absorbed into background."""
    lut = build_role_lut({1: "background"})       # handle 2 deliberately NOT mapped
    handle_map = np.ones((1, 8, 8), dtype=np.uint16)
    handle_map[0, :2, :] = 2
    labels = scene_role_labels(handle_map, lut)
    check(int((labels == UNKNOWN_LABEL).sum()) == 16, "unmapped handle -> 16 unknown-label px")
    dec = decode_scene_roles(encode_scene_roles(handle_map, lut), present_roles=SCENE_ROLES)
    check(int(dec["unknown"].sum()) == 16, "unknown is decodable, i.e. detectable by a coverage check")


def test_nearest_resize_produces_only_legal_labels():
    """§12.2: resize the handle map (nearest), never the encoded colour."""
    import torch
    import torch.nn.functional as F

    lut = build_role_lut({1: "background", 2: "target", 3: "robot_arm"})
    hm = np.ones((2, 64, 64), dtype=np.uint16)
    hm[:, :20, :20] = 2
    hm[:, 40:, 40:] = 3
    t = torch.from_numpy(hm.astype(np.float32))[:, None]
    small = F.interpolate(t, size=(32, 32), mode="nearest")[:, 0].numpy().astype(np.uint16)
    check(set(np.unique(small).tolist()) <= {1, 2, 3}, "nearest resize invents no new handle ids")
    rgb = encode_scene_roles(small, lut)
    legal = {tuple(c) for c in SCENE_ROLE_PALETTE}
    got = {tuple(c) for c in rgb.reshape(-1, 3)}
    check(got <= legal, "every encoded colour is a palette entry (no interpolated colours)")


def test_prompt_tag_is_distinct_from_referring():
    p = build_scene_prompt("close the red jar")
    check(p.startswith("<scene-seg>"), f"scene tag format: {p}")
    check("<seg:" not in p, "scene tag is NOT the referring tag (model must tell them apart)")
    check(p.endswith("close the red jar"), "instruction preserved verbatim")
    check("=red" not in p, "no per-object colour list in the prompt (palette is global)")


def main():
    for fn in [
        test_palette_shape,
        test_tightest_pair_is_not_two_cooccurring_foreground_roles,
        test_lut_unmapped_is_unknown_not_background,
        test_clean_roundtrip_is_exact,
        test_single_pixel_target_survives,
        test_subset_decode_does_not_let_absent_roles_steal_pixels,
        test_background_always_competes,
        test_unknown_is_visible_not_silent,
        test_nearest_resize_produces_only_legal_labels,
        test_prompt_tag_is_distinct_from_referring,
    ]:
        print(f"\n{fn.__name__}:")
        fn()

    print("\n--- palette pairwise distances (tightest 5) ---")
    for dist, a, b in _pairwise()[:5]:
        print(f"  {dist:7.1f}  {a:12s} vs {b}")

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S)")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("\nSCENE_SEG_CODEC_OK")


if __name__ == "__main__":
    main()
