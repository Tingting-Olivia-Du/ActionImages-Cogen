"""Acceptance test for seg_codec (plan/core/10-vision_banana_codec_refactor.md §6.1): exact
multi-instance roundtrip, adjacency, noise robustness (Appendix A algorithm), background-
threshold regression, and a real selfgen GT handle-map episode.

Run: python /workspace/ttdu/ActionImages-Cogen/tests/test_seg_codec.py
"""
import json
import sys

import numpy as np

from training.percep.seg_codec import (
    encode_seg_multi, decode_seg_multi, handles_to_mask, roundtrip_iou, build_seg_prompt,
    SEG_PALETTE, SEG_BG, TAU, COLOR_NAMES,
)

rng = np.random.default_rng(0)


def make_blob(shape, cy, cx, r):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2


# --- 1. exact roundtrip, 1/2/3/8 instances ---
for n_inst in (1, 2, 3, 8):
    for _ in range(5):
        centers = []
        # place non-overlapping blobs on a grid so exact-roundtrip isn't confounded by overlap
        grid = int(np.ceil(np.sqrt(n_inst)))
        cell = 256 // grid
        masks = {}
        for i in range(n_inst):
            gy, gx = divmod(i, grid)
            cy, cx = gy * cell + cell // 2, gx * cell + cell // 2
            r = min(cell // 3, 30)
            masks[f"inst{i}"] = make_blob((256, 256), cy, cx, max(r, 8))
        ious = roundtrip_iou(masks)
        for name, iou in ious.items():
            assert iou == 1.0, f"n_inst={n_inst} {name} exact roundtrip IoU={iou}"
print("EXACT_ROUNDTRIP_OK")

# 9 instances must raise
try:
    masks9 = {f"inst{i}": make_blob((256, 256), 10 + i * 20, 10 + i * 20, 8) for i in range(9)}
    encode_seg_multi(masks9)
    raise AssertionError("expected ValueError for >8 instances")
except ValueError:
    pass
print("PALETTE_OVERFLOW_RAISES_OK")

# --- 2. adjacency: two instances touching but not overlapping ---
for _ in range(10):
    m1 = np.zeros((256, 256), dtype=bool)
    m1[50:150, 50:100] = True
    m2 = np.zeros((256, 256), dtype=bool)
    m2[50:150, 100:150] = True  # shares the column-100 boundary with m1, no overlap
    assert not (m1 & m2).any()
    ious = roundtrip_iou({"a": m1, "b": m2})
    rgb, cmap = encode_seg_multi({"a": m1, "b": m2})
    rec = decode_seg_multi(rgb, cmap)
    assert not (rec["a"] & rec["b"]).any(), "adjacent instances must not overlap after decode"
    for name, iou in ious.items():
        assert iou > 0.98, f"adjacency {name} IoU={iou}"
print("ADJACENCY_OK")

# --- 3. noise robustness, Appendix A stages ---
def make_nonoverlapping_masks(rng, n_max=3):
    """Random blobs on a jittered grid so instances never overlap -- ground-truth masks for
    the seg codec's target use case (RLBench handles, each pixel <=1 instance) never overlap
    either, so an overlapping synthetic mask would test encode_seg_multi's documented "last
    writer wins" behavior instead of the decode algorithm under test."""
    n = rng.integers(1, n_max + 1)
    grid = int(np.ceil(np.sqrt(n)))
    cell = 256 // grid
    masks = {}
    for i in range(n):
        gy, gx = divmod(i, grid)
        cy = gy * cell + cell // 2 + rng.integers(-cell // 6, cell // 6 + 1)
        cx = gx * cell + cell // 2 + rng.integers(-cell // 6, cell // 6 + 1)
        r = rng.integers(cell // 6, cell // 3)
        masks[f"inst{i}"] = make_blob((256, 256), cy, cx, r)
    return masks


ious_all = []
for _ in range(20):
    masks = make_nonoverlapping_masks(rng)
    rgb, cmap = encode_seg_multi(masks)
    rgb_noisy = np.clip(rgb.astype(np.float32) + rng.normal(0, 12, rgb.shape), 0, 255).astype(np.uint8)
    rec = decode_seg_multi(rgb_noisy, cmap)
    for name, m in masks.items():
        inter = (rec[name] & m).sum()
        union = (rec[name] | m).sum()
        iou = inter / union if union > 0 else 1.0
        ious_all.append(iou)
print(f"noise IoU mean {np.mean(ious_all):.4f} min {np.min(ious_all):.4f}")
assert np.min(ious_all) > 0.98, "noise-robust roundtrip must exceed 0.98 IoU"
print("NOISE_ROBUST_OK")

# small-noise-pixel pruning: isolated 1px flip near a real blob shouldn't create a phantom instance
m = make_blob((256, 256), 128, 128, 40)
rgb, cmap = encode_seg_multi({"obj": m})
rgb2 = rgb.copy()
rgb2[10, 10] = SEG_PALETTE[0]  # isolated single foreground-colored pixel, far from the blob
rec = decode_seg_multi(rgb2, cmap)
assert not rec["obj"][10, 10], "isolated 1px noise must be erosion-pruned (Appendix A step 4)"
iou = (rec["obj"] & m).sum() / (rec["obj"] | m).sum()
assert iou > 0.99
print("ISOLATED_PIXEL_PRUNED_OK")

# --- 4. background-threshold boundary regression ---
just_bg = np.array([[[13, 13, 13]]], dtype=np.uint8)   # dist2=507 < tau^2=196? no: 3*13^2=507>196
just_fg_boundary = np.array([[[14, 14, 14]]], dtype=np.uint8)
_, cmap1 = encode_seg_multi({"obj": np.array([[True]])})
for x, label in [(np.array([[[3, 3, 3]]], dtype=np.uint8), "well within tau"),
                  (np.array([[[0, 0, 0]]], dtype=np.uint8), "exact black")]:
    rec = decode_seg_multi(x, cmap1)
    assert not rec["obj"][0, 0], f"{label} must decode background, got foreground"
print("BACKGROUND_THRESHOLD_REGRESSION_OK")

# --- 5. build_seg_prompt ---
prompt = build_seg_prompt({"item": "red", "drawer_frame": "blue"}, "put the item in the drawer")
assert prompt == "<seg: item=red, drawer_frame=blue> put the item in the drawer", prompt
print("BUILD_SEG_PROMPT_OK")

# --- 6. real selfgen episode: multi-instance task (put_item_in_drawer / reach_and_drag), using
# seg_targets.json (plan/core/12-seg-codec-redesign-bugfix-and-vb-parity.md §2) instead of the
# retired task_objects.TASK_INSTANCES substring table ---
import os

# 512_aug, not the bare "rlbench_selfgen" symlink: that one points at the deleted 256 v2 tree
# and DANGLES, so this check has been skipping itself silently. SELFGEN_TEST_DATA overrides.
_SELFGEN = os.environ.get(
    "SELFGEN_TEST_DATA",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data", "rlbench_selfgen_512_aug"))
from training.percep.seg_codec import build_referring_spec, encode_known_color, decode_known_color

candidates = [
    _SELFGEN + "/put_item_in_drawer/variation0/episodes/episode0",
    _SELFGEN + "/reach_and_drag/variation11/episodes/episode0",
]
ep = next((c for c in candidates
           if os.path.exists(os.path.join(c, "handles.json"))
           and os.path.exists(os.path.join(c, "seg_targets.json"))), None)
if ep:
    seg_targets = json.load(open(f"{ep}/seg_targets.json"))
    meta = json.load(open(f"{ep}/meta.json"))
    instruction = meta["desc"][0]
    id_groups, color_map, _dropped = build_referring_spec(seg_targets, instruction)
    assert len(id_groups) >= 1, f"no resolvable instances for {ep} instruction {instruction!r}"
    frames = np.load(f"{ep}/view1/mask.npz")
    key = sorted(frames.files)[0]
    hmap = frames[key]
    if hmap.ndim == 3:  # [T,H,W] -> use frame 0
        hmap = hmap[0]
    instance_masks = {name: handles_to_mask(hmap, ids) for name, ids in id_groups.items()}
    visible = {n: m for n, m in instance_masks.items() if m.any()}
    assert visible, f"no instance visible in frame 0 view1 for {ep}"
    if len(visible) >= 2:
        names = list(visible)
        overlap = (visible[names[0]] & visible[names[1]]).sum()
        total_fg = visible[names[0]].sum() + visible[names[1]].sum()
        assert overlap < 0.01 * total_fg, f"instances unexpectedly overlap: {overlap} px"
    rgb = encode_known_color(visible, color_map)
    dec = decode_known_color(rgb, color_map)
    for name, m in visible.items():
        inter = (dec[name] & m).sum()
        union = (dec[name] | m).sum()
        iou = inter / union if union > 0 else 1.0
        print(f"real-episode[{ep.split('/')[-4]}] instance={name} px={int(m.sum())} roundtrip IoU={iou:.4f}")
        assert iou > 0.98
    print("REAL_EPISODE_MULTI_INSTANCE_OK")
else:
    print("REAL_EPISODE_MULTI_INSTANCE_SKIPPED (no candidate episode found)")

# --- 7. known-color exact roundtrip (no Appendix A pruning -> must be lossless, not just >0.98) ---
for n_inst in (2, 4, 8):
    for _ in range(5):
        grid = int(np.ceil(np.sqrt(n_inst)))
        cell = 256 // grid
        masks = {}
        for i in range(n_inst):
            gy, gx = divmod(i, grid)
            cy, cx = gy * cell + cell // 2, gx * cell + cell // 2
            r = min(cell // 3, 30)
            masks[f"class{i}"] = make_blob((256, 256), cy, cx, max(r, 8))
        color_map = {n: COLOR_NAMES[i] for i, n in enumerate(masks)}
        rgb = encode_known_color(masks, color_map)
        dec = decode_known_color(rgb, color_map)
        for name, m in masks.items():
            inter = (dec[name] & m).sum()
            union = (dec[name] | m).sum()
            iou = inter / union if union > 0 else 1.0
            assert iou == 1.0, f"known-color n_inst={n_inst} {name} exact roundtrip IoU={iou}"
print("KNOWN_COLOR_EXACT_ROUNDTRIP_OK")

# known-color rejects a class not covered by color_map, and an out-of-palette color name
try:
    encode_known_color({"a": make_blob((64, 64), 32, 32, 10)}, {})
    raise AssertionError("expected ValueError for missing color_map entry")
except ValueError:
    pass
try:
    encode_known_color({"a": make_blob((64, 64), 32, 32, 10)}, {"a": "chartreuse"})
    raise AssertionError("expected ValueError for out-of-palette color name")
except ValueError:
    pass
print("KNOWN_COLOR_INPUT_VALIDATION_OK")

# --- 8. known-color small-target regression: sweep_to_dustpan-style tiny scattered target must
# survive roundtrip (module docstring's documented reason known-color decode has NO Appendix A
# pruning -- this asserts that design decision actually holds, not just describes it) ---
tiny = np.zeros((256, 256), dtype=bool)
for (cy, cx) in [(20, 20), (20, 235), (235, 20), (235, 235), (128, 128)]:  # 5 scattered specks,
    tiny[cy:cy + 4, cx:cx + 4] = True                                      # ~4-8px each, like "dirt"
assert 0 < tiny.sum() < 100  # well under Appendix A's THETA_SIZE=2e-4*65536~=13px per-component floor
color_map_tiny = {"dirt": "red"}
rgb_tiny = encode_known_color({"dirt": tiny}, color_map_tiny)
dec_tiny = decode_known_color(rgb_tiny, color_map_tiny)
iou_tiny = (dec_tiny["dirt"] & tiny).sum() / (dec_tiny["dirt"] | tiny).sum()
assert iou_tiny == 1.0, f"known-color must not prune small scattered targets, got IoU={iou_tiny}"
print("KNOWN_COLOR_SMALL_TARGET_SURVIVES_OK")

# --- 9. build_referring_spec color-word narrowing (plan/core/12-...md §3.3 docstring contract) ---

# 9a. no color word in instruction -> falls back to the default "referring" group untouched
seg_targets_no_color = {
    "referring": {"jar_lid": {"handles": [101]}},
    "referring_by_color": {"maroon": {"handles": [201], "replaces": "jar_lid"}},
}
groups, cmap9a, dropped9a = build_referring_spec(seg_targets_no_color, "close the jar")
assert groups == {"jar_lid": [101]}, groups
assert dropped9a == [], dropped9a
print("REFERRING_SPEC_NO_COLOR_WORD_OK")

# 9b. single color word narrows exactly the named default group, others untouched
seg_targets_1color = {
    "referring": {"push_buttons_target": {"handles": [1, 2, 3]}, "table": {"handles": [9]}},
    "referring_by_color": {
        "maroon": {"handles": [1], "replaces": "push_buttons_target"},
        "green": {"handles": [2], "replaces": "push_buttons_target"},
    },
}
groups, cmap9b, dropped9b = build_referring_spec(seg_targets_1color, "push the maroon button")
assert groups == {"push_buttons_target: maroon": [1], "table": [9]}, groups
assert dropped9b == [], dropped9b
print("REFERRING_SPEC_SINGLE_COLOR_WORD_OK")

# 9c. multiple color words narrowing the SAME default group all survive as distinct suffixed
# groups (push_buttons: "push the maroon button, then the green button, then the blue button") --
# this is the exact case the module docstring says must NOT silently collapse to the last color
seg_targets_3color = {
    "referring": {"push_buttons_target": {"handles": [1, 2, 3]}},
    "referring_by_color": {
        "maroon": {"handles": [1], "replaces": "push_buttons_target"},
        "green": {"handles": [2], "replaces": "push_buttons_target"},
        "blue": {"handles": [3], "replaces": "push_buttons_target"},
    },
}
groups, cmap9c, dropped9c = build_referring_spec(
    seg_targets_3color, "push the maroon button, then the green button, then the blue button"
)
assert set(groups) == {
    "push_buttons_target: maroon", "push_buttons_target: green", "push_buttons_target: blue",
}, groups
assert groups["push_buttons_target: maroon"] == [1]
assert groups["push_buttons_target: green"] == [2]
assert groups["push_buttons_target: blue"] == [3]
assert len(set(cmap9c.values())) == 3, f"each narrowed group must get a distinct prompt color, got {cmap9c}"
assert dropped9c == [], dropped9c
print("REFERRING_SPEC_MULTI_COLOR_SAME_GROUP_OK")

# 9d. every resolved group has zero handles -> ValueError, not a silent empty-mask target
seg_targets_empty = {"referring": {"ghost": {"handles": []}}, "referring_by_color": {}}
try:
    build_referring_spec(seg_targets_empty, "do something")
    raise AssertionError("expected ValueError when no group has any handles")
except ValueError:
    pass
print("REFERRING_SPEC_ALL_EMPTY_RAISES_OK")

# 9e. dropped_groups reports PARTIAL emptiness (plan/core/14-...md §4b: the reach_and_drag
# case -- some groups resolve, one doesn't, id_groups/color_map are unaffected by the drop but
# the caller must be able to see it happened)
seg_targets_partial = {
    "referring": {"cube": {"handles": [81]}, "stick": {"handles": [88]}, "target": {"handles": []}},
    "referring_by_color": {},
}
groups9e, cmap9e, dropped9e = build_referring_spec(seg_targets_partial, "drag the cube onto the target")
assert set(groups9e) == {"cube", "stick"}, groups9e   # unaffected: target silently absent, not an error
assert dropped9e == ["target"], dropped9e             # but now visible to the caller
print("REFERRING_SPEC_PARTIAL_DROP_REPORTED_OK")

print("ALL_SEG_CODEC_TESTS_PASSED")
