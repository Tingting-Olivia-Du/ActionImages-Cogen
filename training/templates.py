"""Task templates: which modalities occupy the DiT sequence, and which of them are given.

Two orthogonal axes decide what a training sample teaches (see
`/workspace/ttdu/ttd/plan/core/16-multitask_template_design.md` §3, and doc 15 §4.5):

  Pi   -- WHICH modalities appear as segments in the sequence. Announced by the prompt.
          This is GenCeption / Vision Banana's "prompt selects the task".
  mask -- WHICH of those segments are given (clean) versus predicted. This is
          Action-Images' own switch (i2va / a2v / v2a / video-only, paper §3.3).

"Perception" in the papers' sense is not a modality, it is a (Pi, mask) pair:
Pi = {video, depth} with the video segments FULLY given. 
do we need i2vd, (d2v), v2d

That is why the fork's earlier
substitution layout (`[v1_depth | v1_action | v2_depth | v2_action]`) could not express it --
there was no RGB anywhere in the sequence, so nothing could be "read off" it. That layout is
still available as the `depth+action` template, but it is a depth-space world model, not
perception, and must be reported as such.

A template is named by its Pi in canonical order, joined with "+":

    video+action           the official recipe (4 segments)
    video+depth            RGB given -> depth predicted (4 segments, same cost as official)
    video+segmentation     RGB given -> referring seg predicted (4 segments)
    video+depth+action     co-generation; the cell doc 15 §2.5 proved unreachable (6 segments)
    depth+action           the fork's previous substitution arm (4 segments)

Segment order is view-major, modality-minor, modality in CANONICAL_ORDER -- the same nesting
the upstream literal `[video_src, action_src, video_tgt, action_tgt]` uses. Fixing the order
means `<depth><action>` and `<action><depth>` assemble identically, so the model never has to
spend capacity learning permutation invariance (doc 15 §7.1 constraint 2).
"""
import random as _random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# Modalities that occupy a segment, in the order segments are laid out within a view.
CANONICAL_ORDER: Tuple[str, ...] = ("video", "depth", "segmentation", "action")

# Visual modalities: pixels come from disk via the dataset (`sample["streams"]`).
VISUAL_MODALITIES: Tuple[str, ...] = ("video", "depth", "segmentation")
# The action stream is different in kind: its pixels are rendered inside `forward` by
# projecting action_7d through each view's camera, so the dataset never carries them.
ACTION: str = "action"

# Atomic prompt tags. `segmentation` is the one parameterised tag -- it must carry the
# per-episode colour assignment, because "which object gets which colour" cannot be inferred
# from the instruction alone. That asymmetry is intrinsic (doc 15 §4.1).
ATOMIC_TAG: Dict[str, str] = {"video": "<video>", "depth": "<depth>", "action": "<action>"}

# --- mode-mix constants, transcribed from the upstream literals in train.py -------------
# Upstream `if random.random() < 0.1 and "rlbench" in path`: collapse the visual segments to
# their first latent frame while the action segments keep full length ("policy mode").
SINGLE_FRAME_VISUAL_PROB = 0.1
# Upstream `p = random.random()` three-way split over given/predicted patterns.
MODE_FIRST_SEGMENT_GIVEN_PROB = 0.9   # p < 0.90 -> nothing extra given (each segment's frame 0)
MODE_ALL_VISUAL_GIVEN_PROB = 0.95     # 0.90 <= p < 0.95 -> segment 0 given whole
                                      # p >= 0.95        -> every primary-visual segment given
# Templates WITHOUT an action stream are perception templates: the point of the sample is to
# read a modality off the RGB, so the RGB is given nearly always. The remaining probability
# keeps a little joint-generation diversity so the model does not conclude "RGB is always free".
PERCEPTION_VIDEO_GIVEN_PROB = 0.9


def parse_template(name: str) -> Tuple[str, ...]:
    """'depth+video' -> ('video', 'depth'), validated and canonically ordered."""
    parts = [p.strip() for p in str(name).split("+") if p.strip()]
    if not parts:
        raise ValueError(f"empty template name: {name!r}")
    unknown = [p for p in parts if p not in CANONICAL_ORDER]
    if unknown:
        raise ValueError(
            f"unknown modality {unknown} in template {name!r}; expected from {CANONICAL_ORDER}"
        )
    if len(set(parts)) != len(parts):
        raise ValueError(f"duplicate modality in template {name!r}")
    mods = tuple(m for m in CANONICAL_ORDER if m in parts)
    if not any(m in VISUAL_MODALITIES for m in mods):
        # An action-only sequence has no clean visual frame to anchor the scene, and the
        # camera embedding would describe pixels that are not in the sequence.
        raise ValueError(f"template {name!r} has no visual modality; at least one is required")
    return mods


def format_template(mods: Sequence[str]) -> str:
    """('video', 'action') -> 'video+action' (canonical order)."""
    return "+".join(m for m in CANONICAL_ORDER if m in set(mods))


def primary_visual(mods: Sequence[str]) -> str:
    """The visual modality that occupies the slot upstream called "the video segment".

    For every template containing `video` this is `video`. For the substitution template
    `depth+action` it is `depth` -- which is exactly what makes that template reproduce the
    fork's previous behaviour under the same mask rules.
    """
    for m in CANONICAL_ORDER:
        if m in VISUAL_MODALITIES and m in mods:
            return m
    raise ValueError(f"no visual modality in {mods}")


def parse_template_mix(spec: str) -> Tuple[List[Tuple[str, ...]], List[float]]:
    """'video+action@0.6,video+depth@0.4' -> ([mods, ...], cumulative probabilities)."""
    names: List[Tuple[str, ...]] = []
    weights: List[float] = []
    for item in [s.strip() for s in str(spec).split(",") if s.strip()]:
        name, _, ratio = item.partition("@")
        w = float(ratio) if ratio else 1.0
        if w < 0:
            raise ValueError(f"negative ratio in template_mix: {item!r}")
        mods = parse_template(name)
        if mods in names:
            raise ValueError(f"template {format_template(mods)!r} appears twice in {spec!r}")
        names.append(mods)
        weights.append(w)
    total = sum(weights)
    if not names or total <= 0:
        raise ValueError(f"template_mix has no positive weight: {spec!r}")
    cum, acc = [], 0.0
    for w in weights:
        acc += w / total
        cum.append(acc)
    cum[-1] = 1.0
    return names, cum


def draw_template(
    names: Sequence[Tuple[str, ...]], cum: Sequence[float], rng: Optional[_random.Random] = None
) -> Tuple[str, ...]:
    if len(names) == 1:
        return names[0]
    u = (rng or _random).random()
    for mods, c in zip(names, cum):
        if u < c:
            return mods
    return names[-1]


def seg_tag(color_map: Dict[str, str]) -> str:
    """{instance: colour} -> '<seg: item=red, drawer=green>'."""
    return "<seg: " + ", ".join(f"{n}={c}" for n, c in color_map.items()) + ">"


def prompt_prefix(
    mods: Sequence[str],
    seg_color_map: Optional[Dict[str, str]] = None,
    style: str = "explicit",
) -> str:
    """Prompt prefix for a template. `style='none'` reproduces upstream text exactly.

    `explicit` writes Pi out in full, INCLUDING `<video><action>` on the baseline template.
    That deliberately changes the text distribution of every sample relative to upstream --
    the cost, and why it is acceptable, are argued in doc 16 §3.1. Short version: every arm
    carries the same tags, so the cost is common-mode and cancels in arm-to-arm differences;
    and `style='none'` exists so the numeric regression against upstream still has a hook.
    """
    if style == "none":
        return ""
    if style != "explicit":
        raise ValueError(f"unknown prompt_tag_style {style!r}; expected 'explicit' or 'none'")
    parts = []
    for m in CANONICAL_ORDER:
        if m not in mods:
            continue
        if m == "segmentation":
            if not seg_color_map:
                raise ValueError("segmentation template needs a colour map to build its tag")
            parts.append(seg_tag(seg_color_map))
        else:
            parts.append(ATOMIC_TAG[m])
    return "".join(parts) + " "


@dataclass(frozen=True)
class Segment:
    """One contiguous stretch of the packed latent sequence."""

    modality: str
    view: int
    single_frame: bool  # collapse to latent frame 0 (upstream's 10% "policy mode" variant)
    fully_given: bool   # whole segment is a condition, not just its first frame


def plan_segments(
    mods: Sequence[str],
    num_views: int,
    rng: Optional[_random.Random] = None,
    is_rlbench: bool = True,
) -> List[Segment]:
    """Sample one (segment order, length, given/predicted) plan for this step.

    For `mods == ('video', 'action')`, `num_views == 2` this reproduces the upstream branch
    in train.py:286-340 decision-for-decision -- same probabilities, same meaning, same
    resulting mask. `tests/test_forward_unchanged.py` checks the resulting tensors against a
    literal transcription of that code.

    RNG NOTE: the draws happen in upstream's order MINUS the action gate
    (`random.random() < 0.9`), which moved into the dataset so that the prompt and the layout
    are decided together (CHANGES_TEMPLATES.md §3.2). Exact RNG *traces* therefore do not
    match upstream; equivalence is defined per-decision, not per-draw.
    """
    rng = rng or _random
    mods = tuple(m for m in CANONICAL_ORDER if m in set(mods))
    has_action = ACTION in mods
    anchor = primary_visual(mods)

    single_frame_visual = False
    given_modalities: set = set()   # every segment of these modalities is a full condition
    give_first_segment = False      # only segment 0 (view 0's anchor), upstream's 5% branch

    if has_action:
        # Upstream evaluates the random FIRST and the path check second (short-circuit order),
        # so the draw is consumed for every dataset, not only rlbench.
        collapse = rng.random() < SINGLE_FRAME_VISUAL_PROB
        if collapse and is_rlbench:
            single_frame_visual = True
        else:
            p = rng.random()
            if p < MODE_FIRST_SEGMENT_GIVEN_PROB:
                pass  # every segment contributes only its first frame
            elif p < MODE_ALL_VISUAL_GIVEN_PROB:
                give_first_segment = True
            else:
                given_modalities = {anchor}  # v2a: infer the action stream from the visuals
    elif len(mods) > 1:
        # Perception template: give the anchor visual whole so the target modality has to be
        # read off it rather than hallucinated.
        if rng.random() < PERCEPTION_VIDEO_GIVEN_PROB:
            given_modalities = {anchor}
    # A SINGLE-modality template (Pi = {video}, which is where action dropout lands) draws
    # nothing: giving its only modality whole would leave the sequence with no predicted
    # position at all, i.e. a zero-loss step that still costs a full forward and backward.
    # Upstream's video-only branch conditions on first frames only, and so does this.

    plan: List[Segment] = []
    for v in range(num_views):
        for m in mods:
            is_segment_zero = not plan  # view 0's anchor, i.e. upstream's `masks[:, :, :T_l]`
            plan.append(
                Segment(
                    modality=m,
                    view=v,
                    single_frame=single_frame_visual and m in VISUAL_MODALITIES,
                    fully_given=(m in given_modalities) or (give_first_segment and is_segment_zero),
                )
            )
    return plan


def assemble(
    plan: Sequence[Segment],
    latents_by_modality: Dict[str, Sequence[torch.Tensor]],
    cam_by_view: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """plan + per-(modality, view) latents -> (latents, camera_emb, masks).

    Shapes follow upstream exactly:
      latents_by_modality[m][v] : [B, C_l, T_l, H_l, W_l]  (packed along dim=2, the time axis)
      cam_by_view[v]            : [B, T_l, C_l, H_l, W_l]  (packed along dim=1)

    The mask offset is CUMULATIVE rather than a multiple of T_l, which is what lets segments
    have different lengths -- upstream's single-frame variant already needs that, and it is
    what upstream's hard-coded `{0, 1, T_l+1, T_l+2}` literals were spelling out by hand.
    """
    lat_segs, cam_segs = [], []
    for seg in plan:
        try:
            z = latents_by_modality[seg.modality][seg.view]
        except (KeyError, IndexError) as e:
            raise KeyError(
                f"no latents for segment {seg.modality!r} view {seg.view}; "
                f"have {sorted(latents_by_modality)}"
            ) from e
        c = cam_by_view[seg.view]
        if seg.single_frame:
            z = z[:, :, :1]
            c = c[:, :1]
        lat_segs.append(z)
        cam_segs.append(c)

    latents = torch.cat(lat_segs, dim=2)
    camera_emb = torch.cat(cam_segs, dim=1)

    masks = torch.zeros_like(latents, dtype=torch.bool)  # True: condition frames
    offset = 0
    for seg, z in zip(plan, lat_segs):
        n = z.shape[2]
        if seg.fully_given:
            masks[:, :, offset : offset + n, ...] = 1
        else:
            masks[:, :, offset, ...] = 1
        offset += n
    assert offset == latents.shape[2], f"segment offsets {offset} != sequence {latents.shape[2]}"
    # A fully-conditioned sequence produces `valid.sum() == 0`, which the loss clamps to 1 and
    # silently reports as 0.0 -- a step that costs a full forward and backward and teaches
    # nothing, visible only as a suspiciously good loss curve. Refuse to build one.
    assert not bool(masks.all()), (
        "every latent position is a condition frame; this sample has no prediction target. "
        f"plan={[(s.modality, s.view, s.fully_given) for s in plan]}"
    )
    return latents, camera_emb, masks
