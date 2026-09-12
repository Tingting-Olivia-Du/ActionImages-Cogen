"""Perception codecs: depth / segmentation / surface-normal GT <-> RGB in [-1, 1].

All three codecs are pure numpy (+scipy for seg) with no coupling to this repo -- they encode a
per-frame annotation into the SAME pixel space the video stream already lives in, which is
what lets a perception modality occupy a visual segment of the DiT sequence with no
architecture change (see FORK_CHANGES.md and
/workspace/ttdu/ttd/plan/core/15-a4_io_current_vs_desired_action_perception_cogen.md).

Ported verbatim from ttd/src/percep/ apart from one path default in depth_codec's __main__ and
two corrections in normal_codec (metric gradients via intrinsics; no invalid sentinel) -- see its
module docstring.
Design rationale for each codec lives in its own module docstring and in
ttd/plan/core/{10,11,12}-*.md.
"""
from .depth_codec import decode_depth, encode_depth, roundtrip_absrel
from .normal_codec import (
    decode_normal,
    depth_to_normal,
    encode_normal_from_depth,
    normal_cos,
    roundtrip_cos,
)
from .seg_codec import (
    SCENE_ROLE_PALETTE,
    SCENE_ROLES,
    build_referring_spec,
    build_role_lut,
    build_scene_prompt,
    build_seg_prompt,
    decode_known_color,
    decode_scene_roles,
    encode_known_color,
    encode_scene_roles,
    handles_to_mask,
    scene_role_iou,
    scene_role_labels,
)

__all__ = [
    "encode_depth",
    "decode_depth",
    "roundtrip_absrel",
    "encode_known_color",
    "decode_known_color",
    "handles_to_mask",
    "build_referring_spec",
    "build_seg_prompt",
    # surface normal, derived from the same depth.npz (no new data). Argus Fig. 2 convention.
    "depth_to_normal",
    "encode_normal_from_depth",
    "decode_normal",
    "normal_cos",
    "roundtrip_cos",
    # scene-role protocol (SEGMENTATION_SCENE_ROLES_PLAN.md)
    "SCENE_ROLES",
    "SCENE_ROLE_PALETTE",
    "build_role_lut",
    "encode_scene_roles",
    "scene_role_labels",
    "decode_scene_roles",
    "build_scene_prompt",
    "scene_role_iou",
]
