"""Perception codecs: depth / segmentation ground truth <-> RGB in [-1, 1].

Both codecs are pure numpy (+scipy for seg) with no coupling to this repo -- they encode a
per-frame annotation into the SAME pixel space the video stream already lives in, which is
what lets a perception modality occupy a visual segment of the DiT sequence with no
architecture change (see FORK_CHANGES.md and
/workspace/ttdu/ttd/plan/core/15-a4_io_current_vs_desired_action_perception_cogen.md).

Ported verbatim from ttd/src/percep/ apart from one path default in depth_codec's __main__.
Design rationale for each codec lives in its own module docstring and in
ttd/plan/core/{10,11,12}-*.md.
"""
from .depth_codec import decode_depth, encode_depth, roundtrip_absrel
from .seg_codec import (
    build_referring_spec,
    build_seg_prompt,
    decode_known_color,
    encode_known_color,
    handles_to_mask,
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
]
