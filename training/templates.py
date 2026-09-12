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
    segmentation+action    action in scene-role space   (4 segments)
    normal+action          action in surface-normal space (4 segments)

The three `<modality>+action` substitution templates are arm7's menu. They are world models in
their own visual space -- there is no RGB anywhere in the sequence -- but unlike the perception
templates they DO carry action supervision, which is the whole point of that arm: the action
stream is conditioned on four different renderings of the same scene at a fixed action budget.

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
# `normal` sits between segmentation and action: appending rather than inserting keeps every
# template that does NOT contain it laid out byte-identically (pinned by
# tests/test_forward_unchanged.py), and keeping it before `action` preserves the invariant that
# the action segment is last within a view.
CANONICAL_ORDER: Tuple[str, ...] = ("video", "depth", "segmentation", "normal", "action")

# Visual modalities: pixels come from disk via the dataset (`sample["streams"]`).
VISUAL_MODALITIES: Tuple[str, ...] = ("video", "depth", "segmentation", "normal")
# The action stream is different in kind: its pixels are rendered inside `forward` by
# projecting action_7d through each view's camera, so the dataset never carries them.
ACTION: str = "action"

# Atomic prompt tags. `segmentation` is the one parameterised tag -- it must carry the
# per-episode colour assignment, because "which object gets which colour" cannot be inferred
# from the instruction alone. That asymmetry is intrinsic (doc 15 §4.1).
ATOMIC_TAG: Dict[str, str] = {
    "video": "<video>", "depth": "<depth>", "normal": "<normal>", "action": "<action>",
}

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

# The M axis (SEGMENTATION_SCENE_ROLES_PLAN.md §10.3): how a perception template splits its
# conditioning, as (p_iiii, p_fiii, p_fifi, p_single_frame_anchor). Named presets:
#
#   M0 = (0.10, 0.00, 0.90, 0.00)  historical default; equals PERCEPTION_VIDEO_GIVEN_PROB=0.9
#   M1 = (0.45, 0.10, 0.45, 0.00)  balanced
#   M2 = (0.90, 0.05, 0.05, 0.00)  parity with the official has_action split
#
# M0 is the default so that every existing command and checkpoint keeps its exact training
# distribution; the axis only moves when asked for explicitly.
PERCEPTION_MASK_MIX_DEFAULT = (1.0 - PERCEPTION_VIDEO_GIVEN_PROB, 0.0, PERCEPTION_VIDEO_GIVEN_PROB, 0.0)
PERCEPTION_MASK_MIX_PRESETS: Dict[str, Tuple[float, float, float, float]] = {
    "M0": (0.10, 0.00, 0.90, 0.00),
    "M1": (0.45, 0.10, 0.45, 0.00),
    "M2": (0.90, 0.05, 0.05, 0.00),
}

# The A axis: the same four-way split for templates that DO contain <action>, as
# (p_iiii, p_fiii, p_fifi, p_policy). It is the exact complement of the M axis above -- every
# template hits one branch or the other, never both -- and the two are separate knobs because
# they answer different questions (M: "how is a perception task conditioned"; A: "how is an
# action task conditioned").
#
# A0 is DERIVED from the upstream literals rather than written out, so the default distribution
# and the constants it is supposed to reproduce cannot drift apart. Worked through:
#   p_policy = SINGLE_FRAME_VISUAL_PROB                                    = 0.10
#   p_iiii   = 0.90 * MODE_FIRST_SEGMENT_GIVEN_PROB                        = 0.81
#   p_fiii   = 0.90 * (MODE_ALL_VISUAL_GIVEN_PROB - MODE_FIRST_SEGMENT_..) = 0.045
#   p_fifi   = 0.90 * (1 - MODE_ALL_VISUAL_GIVEN_PROB)                     = 0.045
# `plan_segments` inverts this back to the two-draw form, so with A0 it makes literally the same
# two draws against the same thresholds as before the axis existed (see the note there).
_A0_REST = 1.0 - SINGLE_FRAME_VISUAL_PROB
ACTION_MASK_MIX_PRESETS: Dict[str, Tuple[float, float, float, float]] = {
    # Upstream / arm0 / the video+action stream of every arm before arm7.
    "A0": (
        _A0_REST * MODE_FIRST_SEGMENT_GIVEN_PROB,
        _A0_REST * (MODE_ALL_VISUAL_GIVEN_PROB - MODE_FIRST_SEGMENT_GIVEN_PROB),
        _A0_REST * (1.0 - MODE_ALL_VISUAL_GIVEN_PROB),
        SINGLE_FRAME_VISUAL_PROB,
    ),
    # arm7. Action-heavy: FIII (the cross-view mode) is the least relevant of the four to action
    # quality, so it is spent on policy mode -- which puts ALL of its loss on the action segments
    # under the same single-frame anchor conditioning the closed-loop rollout actually provides,
    # on a canvas of 24 rather than 44 latent frames (~45% shorter, ~30% of the attention). That
    # shortens the DiT only -- `forward` encodes every stream at FULL length before plan_segments
    # runs, so the VAE cost is unchanged and the wall-clock saving is small. IIII
    # still dominates because IIII is the deployment regime (eval/policy.py passes
    # fully_given_modalities=[]); FIFI stays at 5% so v2a remains readable as an offline bound.
    "A1": (0.75, 0.00, 0.05, 0.20),
    # Exactly 90/5/5 with no policy mode -- the M2 split applied to the action branch.
    "A2": (0.90, 0.05, 0.05, 0.00),
}
ACTION_MASK_MIX_DEFAULT = ACTION_MASK_MIX_PRESETS["A0"]

# --- the F axis: modality dropout on a multi-modality (fusion) canvas -------------------
#
# WHY THIS AXIS EXISTS. `assemble` hands every segment that is not `fully_given` a free first
# latent frame. On the 4-segment templates that is 4 anchors out of 44 latent frames and it is
# what upstream did. On the 10-segment fusion canvas (video+depth+segmentation+normal+action,
# 110 latent frames) it is TEN free anchors -- one per modality per view -- and that breaks the
# experiment in two separate ways:
#
#   1. It makes the claim untestable. If depth's frame 0 is always supplied, "the model
#      completes depth from the other modalities" is indistinguishable from "the model copies
#      depth's own anchor forward". The tag-swap ablation already measured which of those the
#      model reaches for: swapping the prompt tag moves metrics ~1-2%, while REMOVING the anchor
#      (the f0f0 condition) costs ~9x. The anchor is the signal.
#   2. It trains a regime deployment cannot supply. At rollout the only modality available is
#      RGB; there is no depth/segmentation/normal frame 0 to hand over. A model trained with
#      those anchors always present is being asked, at deployment, for exactly the ~9x-worse
#      condition it never saw.
#
# So the axis draws one REGIME per sample and assigns each modality one of three roles --
# GIVEN (every frame is a condition), ANCHORED (frame 0 only, the historical default), or
# ABSENT (no condition at all). Roles are assigned PER MODALITY and applied to both views: a
# per-view role would let view 0's depth anchor view 1's depth, and "absent" would be a fiction.
#
# The five regimes, in the fixed order the probability tuple uses:
#
#   0 full_anchor  every modality ANCHORED. The historical behaviour, kept so the canvas still
#                  sees the joint-generation-from-all-anchors condition it is evaluated in.
#   1 rgb_only     video ANCHORED, depth/segmentation/normal/action ABSENT. THE DEPLOYMENT
#                  REGIME: one RGB frame in, every other modality generated.
#   2 rgb_given    video fully GIVEN, the rest ABSENT. Perception + policy from a whole RGB
#                  video -- the discriminative reading, comparable to an external estimator.
#   3 one_out      one modality drawn uniformly from ALL FIVE is ABSENT, the rest ANCHORED.
#                  The mutual-completion probe. `video` and `action` are in the draw on
#                  purpose: depth->RGB and observations->action are the directions that make
#                  "mutual" mean something rather than "RGB is always the source".
#   4 policy       video ANCHORED and collapsed to one latent frame, depth/segmentation/normal
#                  ABSENT and collapsed, action full length. This is what eval/policy.py
#                  actually hands the model at replan time, on a canvas of 30 rather than 110
#                  latent frames.
#
# NOTE the prompt tag is the FULL five-modality prefix under every regime. The tag announces
# the canvas, not what was supplied. Since the ablation above shows the tag carries no
# information the model acts on, encoding absence in it would be decoration, and it would make
# the text distribution depend on a draw the evaluator has to reproduce exactly.
FUSION_REGIMES: Tuple[str, ...] = ("full_anchor", "rgb_only", "rgb_given", "one_out", "policy")
FUSION_MASK_MIX_PRESETS: Dict[str, Tuple[float, float, float, float, float]] = {
    # F1 is the Stage-1 default. Weighted so that 55% of samples (rgb_only + rgb_given) train
    # the deployment direction, 15% probe mutual completion, 10% trains the replan canvas, and
    # 30% keeps the all-anchor condition the arm is scored against. See the plan's Stage 1.
    "F1": (0.30, 0.30, 0.15, 0.15, 0.10),
    # F0 reproduces the pre-axis behaviour exactly: every segment ANCHORED, i.e. what
    # plan_segments already does for a template whose mask mix leaves given_modalities empty.
    # It exists so `fusion0-anchor` (the control that measures what the axis buys) can be
    # declared rather than obtained by omitting a flag.
    "F0": (1.00, 0.00, 0.00, 0.00, 0.00),
    # F2 drops the all-anchor regime entirely -- an ablation for "is full_anchor load-bearing,
    # or is it just diluting the deployment signal?"
    "F2": (0.00, 0.45, 0.20, 0.25, 0.10),
}


def parse_fusion_mask_mix(spec) -> Optional[Tuple[float, float, float, float, float]]:
    """The F axis: 'F1' | '0.3,0.3,0.15,0.15,0.1' | None -> per-regime probabilities.

    Returns None for an unset spec, and None is meaningful: it routes a multi-modality template
    back to the M/A axes, which is exactly the `fusion0-anchor` control. Opting in has to be
    explicit for the same reason the M and A defaults are pinned -- `video+depth+action` (arm2)
    is an existing 6-segment template with existing semantics, and silently rerouting every
    template with three or more modalities would redefine it.

    Unlike the M and A axes this is a FIVE-way split over named regimes rather than the shared
    (iiii, fiii, fifi, last) 4-tuple, so it cannot reuse `_parse_mask_mix`. The sum-to-1
    validation is kept identical, for the same reason: a silently renormalised mix would make
    two runs labelled with the same mix train on different distributions.
    """
    if spec is None or spec == "":
        return None
    if isinstance(spec, str):
        key = spec.strip()
        if key in FUSION_MASK_MIX_PRESETS:
            return FUSION_MASK_MIX_PRESETS[key]
        parts = [p for p in key.replace(" ", "").split(",") if p]
    else:
        parts = list(spec)
    if len(parts) != len(FUSION_REGIMES):
        raise ValueError(
            f"fusion_mask_mix needs {len(FUSION_REGIMES)} values "
            f"({','.join(FUSION_REGIMES)}) or a preset {sorted(FUSION_MASK_MIX_PRESETS)}; "
            f"got {spec!r}"
        )
    vals = tuple(float(p) for p in parts)
    if any(v < 0 for v in vals):
        raise ValueError(f"fusion_mask_mix has a negative probability: {vals}")
    total = sum(vals)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fusion_mask_mix must sum to 1, got {total} from {vals}")
    return vals


def _parse_mask_mix(spec, default, presets, axis: str) -> Tuple[float, float, float, float]:
    """'M2' | '0.9,0.05,0.05,0' | None -> (p_iiii, p_fiii, p_fifi, p_last).

    Validates that the four probabilities sum to 1, because a silently-renormalised mix would
    make two runs labelled with the same mix train on different distributions.
    """
    if spec is None or spec == "":
        return default
    if isinstance(spec, str):
        key = spec.strip()
        if key in presets:
            return presets[key]
        parts = [p for p in key.replace(" ", "").split(",") if p]
    else:
        parts = list(spec)
    if len(parts) != 4:
        raise ValueError(
            f"{axis} needs 4 values (iiii,fiii,fifi,single_frame) or a preset "
            f"{sorted(presets)}; got {spec!r}"
        )
    vals = tuple(float(p) for p in parts)
    if any(v < 0 for v in vals):
        raise ValueError(f"{axis} has a negative probability: {vals}")
    total = sum(vals)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{axis} must sum to 1, got {total} from {vals}")
    return vals


def parse_perception_mask_mix(spec) -> Tuple[float, float, float, float]:
    """The M axis: 'M2' | '0.9,0.05,0.05,0' | None -> (p_iiii, p_fiii, p_fifi, p_single_frame).

    Applies only to templates WITHOUT <action>. See parse_action_mask_mix for the complement.
    """
    return _parse_mask_mix(
        spec, PERCEPTION_MASK_MIX_DEFAULT, PERCEPTION_MASK_MIX_PRESETS, "perception_mask_mix"
    )


def parse_action_mask_mix(spec) -> Tuple[float, float, float, float]:
    """The A axis: 'A1' | '0.75,0,0.05,0.2' | None -> (p_iiii, p_fiii, p_fifi, p_policy).

    Applies only to templates CONTAINING <action>. The default A0 reproduces the upstream
    literals exactly, so an unset flag changes nothing anywhere.
    """
    return _parse_mask_mix(
        spec, ACTION_MASK_MIX_DEFAULT, ACTION_MASK_MIX_PRESETS, "action_mask_mix"
    )


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


def parse_per_dataset_template_mix(spec, default_mix: str, known: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """'ds=mix; ds2=mix2' -> {dataset_name: mix_string}, unlisted datasets get `default_mix`.

    Why per-dataset rather than one global `--template_mix`: the feasible template menu is a
    property of the DATA, not of the run. Only `rlbench_selfgen*` ships depth/segmentation GT,
    and bridge has no usable action. A single global mix would have to be silently filtered
    down to each tree's feasible subset, which makes the realised task proportions differ from
    the requested ones with nothing in the log saying so -- the same class of silent
    degradation `_assert_seg_available` refuses for segmentation.

    The separator is ';' between datasets and '=' between name and mix, so the existing
    'name@ratio,name@ratio' grammar keeps working unchanged on the right-hand side. Whitespace
    and newlines are ignored, so the spec can be written across several lines in a shell script.
    """
    out: Dict[str, str] = {}
    if spec is None or not str(spec).strip():
        return out
    for item in str(spec).replace("\n", " ").split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"template_mix_per_dataset entry {item!r} has no '='; expected "
                f"'dataset=template@ratio,template@ratio' entries separated by ';'"
            )
        name, _, mix = item.partition("=")
        name, mix = name.strip(), mix.strip()
        if not name or not mix:
            raise ValueError(f"template_mix_per_dataset entry {item!r} has an empty side")
        if name in out:
            raise ValueError(f"dataset {name!r} appears twice in template_mix_per_dataset")
        parse_template_mix(mix)  # validate now, so a typo fails at startup not at step 40k
        out[name] = mix
    if known is not None:
        # A key that matches no selected dataset is almost always a typo, and the failure mode
        # is invisible: the intended tree silently falls back to the global --template_mix, so
        # e.g. a misspelt `rlbench_selfgen_512_aug=` drops the entire depth stream while the
        # run still starts and every log line looks normal.
        unknown = sorted(set(out) - set(known))
        if unknown:
            raise ValueError(
                f"template_mix_per_dataset names {unknown} which are not in --dataset_name "
                f"({sorted(known)}). A menu for an unselected dataset has no effect, and the "
                f"dataset you meant would silently fall back to --template_mix."
            )
    return out


def assert_menu_supported(dataset_name: str, template_mix: str, available: Sequence[str]) -> None:
    """Refuse a template menu the dataset cannot actually serve.

    Silently dropping the impossible modality would make e.g. `bridge=video+action` train as
    plain `video` while the requested proportions say otherwise, and would put an `<action>`
    tag in a prompt whose sequence has no action segment.
    """
    have = set(available)
    for mods in parse_template_mix(template_mix)[0]:
        missing = [m for m in mods if m not in have]
        if missing:
            raise ValueError(
                f"dataset {dataset_name!r} cannot serve template "
                f"{format_template(mods)!r}: it provides {sorted(have)} but the template needs "
                f"{missing}. Fix --template_mix_per_dataset, or use a tree that has this "
                f"modality (only rlbench_selfgen* ships depth/segmentation GT)."
            )


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


SCENE_SEG_PROTOCOL = "rlbench-scene-role-v1"


def scene_seg_tag(protocol: str = SCENE_SEG_PROTOCOL) -> str:
    """-> '<scene-seg>'.

    Atomic on purpose. The palette is GLOBAL and fixed -- target is always red, goal always
    green, robot_arm always blue, in every task, episode, frame and view -- so the colour map is
    learned from the data and never needs stating. The repo's own rule (see `seg_tag`) is to
    parameterise a tag only when the mapping cannot be inferred; a constant string carries zero
    bits and cost 12 UMT5 tokens on every seg sample.

    The dropped `protocol=rlbench-scene-role-v1` also created false confidence: `scene_segments.json`
    already records the protocol, and the version was NOT bumped when the targets changed
    (2026-08-23, the NO_OBJECT_HANDLE and handle-union fixes), so two checkpoints trained on
    different targets both claimed "v1". If a second dense-segmentation protocol ever ships, give
    it its OWN atomic tag -- the way `depth` and `normal` each have one -- rather than a version
    parameter the model cannot act on.

    The tag must still differ from `<seg:`: the two protocols produce different images from the
    same instruction. It must also be scrub-invariant (no `_`, no `.`) -- see the module docstring
    of training/templates.py:scene_seg_tag's predecessor and train.py:287.

    `protocol` is kept in the signature so a caller can still reproduce the pre-2026-08-23 long
    form (`scene_seg_tag.legacy(protocol)`) when evaluating a checkpoint trained with it -- a
    checkpoint must always be asked with the tag it was trained on.
    """
    return "<scene-seg>"


def scene_seg_tag_legacy(protocol: str = SCENE_SEG_PROTOCOL) -> str:
    """The pre-2026-08-23 long form. Only for evaluating checkpoints trained before the change
    (arm1__seed42_fi3_512_aug_sr/checkpoint-2000 is the only one)."""
    return f"<scene-seg protocol={protocol}>"


def prompt_prefix(
    mods: Sequence[str],
    seg_color_map: Optional[Dict[str, str]] = None,
    style: str = "explicit",
    segmentation_mode: str = "referring",
    legacy_scene_seg_tag: bool = False,
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
            if segmentation_mode == "scene_roles":
                # A checkpoint must be asked with the tag it was TRAINED with. Everything from
                # 2026-08-23 on uses the atomic `<scene-seg>`; the one checkpoint that predates
                # the change (arm1 .../checkpoint-2000) needs the long form or it is being asked
                # a question it never saw -- the same train/eval split the scrub trap caused.
                parts.append(scene_seg_tag_legacy() if legacy_scene_seg_tag else scene_seg_tag())
            elif segmentation_mode == "referring":
                if not seg_color_map:
                    raise ValueError("segmentation template needs a colour map to build its tag")
                parts.append(seg_tag(seg_color_map))
            else:
                raise ValueError(
                    f"unknown segmentation_mode {segmentation_mode!r}; expected "
                    f"'referring' or 'scene_roles'"
                )
        else:
            parts.append(ATOMIC_TAG[m])
    return "".join(parts) + " "


@dataclass(frozen=True)
class Segment:
    """One contiguous stretch of the packed latent sequence.

    The three conditioning roles are mutually exclusive and are checked in `assemble` in the
    order (absent, fully_given, else anchor-only):

      fully_given=True   every latent frame of the segment is a clean condition   ("F")
      absent=True        NO latent frame is a condition; the whole segment is predicted
      neither            only latent frame 0 is a condition, frames 1.. are predicted ("I")

    `absent` defaults to False so that every existing construction site -- and
    `tests/test_forward_unchanged.py`, which transcribes upstream's mask code literally -- keeps
    producing byte-identical tensors. It is only ever set by the fusion branch of
    `plan_segments`, and it exists because `assemble` otherwise hands EVERY segment a free first
    frame: on a 10-segment fusion canvas that is ten free anchors, which makes cross-modal
    completion a copy task rather than a prediction task, and trains a conditioning regime
    (depth/segmentation/normal frame 0 supplied) that deployment cannot provide. See
    FUSION_MASK_MIX_PRESETS.
    """

    modality: str
    view: int
    single_frame: bool  # collapse to latent frame 0 (upstream's 10% "policy mode" variant)
    fully_given: bool   # whole segment is a condition, not just its first frame
    absent: bool = False  # no condition frames at all -- predicted from the other modalities


def plan_segments(
    mods: Sequence[str],
    num_views: int,
    rng: Optional[_random.Random] = None,
    is_rlbench: bool = True,
    perception_mask_mix: Optional[Tuple[float, float, float, float]] = None,
    action_mask_mix: Optional[Tuple[float, float, float, float]] = None,
    fusion_mask_mix: Optional[Tuple[float, float, float, float, float]] = None,
) -> List[Segment]:
    """Sample one (segment order, length, given/predicted) plan for this step.

    For `mods == ('video', 'action')`, `num_views == 2` and the DEFAULT `action_mask_mix` this
    reproduces the upstream branch in train.py:286-340 decision-for-decision -- same
    probabilities, same meaning, same resulting mask. `tests/test_forward_unchanged.py` checks
    the resulting tensors against a literal transcription of that code, and separately checks
    that passing `action_mask_mix=A0` explicitly is a no-op at every threshold boundary.

    The two mask mixes PARTITION the template space and never both apply: `action_mask_mix` for
    templates containing `action`, `perception_mask_mix` for action-free multi-modality
    templates, and neither for a single-modality template (which draws nothing at all).

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
    single_frame_anchor = False     # perception-only: collapse the ANCHOR, keep depth/seg full
    given_modalities: set = set()   # every segment of these modalities is a full condition
    absent_modalities: set = set()  # every segment of these modalities has NO condition frame
    give_first_segment = False      # only segment 0 (view 0's anchor), upstream's 5% branch

    if fusion_mask_mix is not None:
        # The F axis (see FUSION_MASK_MIX_PRESETS). It REPLACES the M and A axes rather than
        # composing with them: those two decide how much of the anchor modality is given, this
        # one decides which modalities are present at all, and stacking them would let a
        # modality be simultaneously the FIFI source and absent.
        #
        # ONE draw, against a 5-way cumulative. There is no upstream behaviour to reproduce
        # here, so the two-draw contortion the A axis keeps (see below) buys nothing and the
        # straightforward form is the honest one.
        if len(mods) < 2:
            # Nothing to redistribute: with one modality every regime degenerates to the same
            # plan, and marking it absent would leave the sequence with no scene anchor at all.
            # Fall through to the all-anchored default -- the SAME thing the M and A axes do for
            # a single-modality template (see the note at the end of this if/elif chain).
            #
            # This is reachable at RUNTIME, not just from a bad flag: the dataset drops <action>
            # with probability `action_dropout_prob`, and drops `segmentation` when an episode's
            # scene_segments.json resolves nothing. Raising here would turn that into a crash at
            # some random step tens of hours in. It cannot actually strip the fusion template
            # below three modalities, but a two-modality template plus dropout can reach one.
            return _lay_out(mods, num_views, False, False, set(), set(), False, anchor)
        u = rng.random()
        acc = 0.0
        regime = FUSION_REGIMES[-1]
        for name, p in zip(FUSION_REGIMES, fusion_mask_mix):
            acc += p
            if u < acc:
                regime = name
                break
        others = set(mods) - {anchor}
        if regime == "full_anchor":
            pass                                    # every modality keeps its first frame
        elif regime == "rgb_only":
            absent_modalities = set(others)         # one anchor frame -> everything else
        elif regime == "rgb_given":
            given_modalities = {anchor}
            absent_modalities = set(others)         # whole anchor video -> everything else
        elif regime == "one_out":
            # Uniform over ALL modalities including the anchor and action, so the probe covers
            # depth->RGB and observations->action, not just RGB->everything. `sorted` because
            # `set` iteration order is not stable across processes and this draw has to be
            # reproducible from the seed alone.
            absent_modalities = {rng.choice(sorted(mods))}
        elif regime == "policy":
            # The replan canvas eval/policy.py actually presents: a single anchor frame, no
            # other observation modality, action at full length. 30 rather than 110 latent
            # frames on the 5-modality template.
            #
            # NOT gated on `is_rlbench`, unlike the A axis's policy branch. That gate exists
            # upstream because collapsing to a single frame was an rlbench-specific trick; here
            # it would mean a run declaring F1 silently trains a DIFFERENT mix on a non-rlbench
            # tree, which is exactly the silent-renormalisation failure the sum-to-1 check in
            # parse_fusion_mask_mix refuses. The point is moot in practice -- this axis is for
            # templates carrying depth/segmentation/normal, and assert_menu_supported already
            # restricts those to rlbench_selfgen* -- so the honest mix beats the extra gate.
            single_frame_visual = True
            absent_modalities = {m for m in others if m in VISUAL_MODALITIES}
        else:  # unreachable; FUSION_REGIMES is closed
            raise ValueError(f"unknown fusion regime {regime!r}")
    elif has_action:
        # The A axis (see ACTION_MASK_MIX_PRESETS): (p_iiii, p_fiii, p_fifi, p_policy).
        #
        # THE TWO-DRAW STRUCTURE IS LOAD-BEARING and is deliberately kept even though a single
        # draw against a 4-way cumulative would express the same marginal. Upstream makes two
        # draws in this order, `tests/test_forward_unchanged.py` pins each decision path with a
        # scripted RNG, and the collapse draw is consumed for EVERY dataset because upstream
        # evaluates the random first and the path check second (short-circuit order). Flattening
        # it would keep the marginals and change which outcome a given seed produces -- exactly
        # the class of silent change that test exists to catch.
        #
        # So the marginal is inverted back into the two thresholds it came from. With the default
        # A0 that arithmetic returns SINGLE_FRAME_VISUAL_PROB / MODE_FIRST_SEGMENT_GIVEN_PROB /
        # MODE_ALL_VISUAL_GIVEN_PROB unchanged, i.e. the same two draws against the same numbers
        # as before this axis existed.
        p_iiii, p_fiii, _p_fifi, p_policy = action_mask_mix or ACTION_MASK_MIX_DEFAULT
        collapse = rng.random() < p_policy
        if collapse and is_rlbench:
            single_frame_visual = True
        else:
            # Conditional on NOT having collapsed. p_policy=1 would leave nothing to condition
            # on, so clamp rather than divide by zero -- the branch is unreachable in that case.
            rest = max(1.0 - p_policy, 1e-12)
            t_iiii = p_iiii / rest
            t_fiii = (p_iiii + p_fiii) / rest
            p = rng.random()
            if p < t_iiii:
                pass  # every segment contributes only its first frame
            elif p < t_fiii:
                give_first_segment = True
            else:
                given_modalities = {anchor}  # v2a: infer the action stream from the visuals
    elif len(mods) > 1:
        # Perception template. `perception_mask_mix` is the M axis
        # (SEGMENTATION_SCENE_ROLES_PLAN.md §10.3): (p_iiii, p_fiii, p_fifi, p_single_frame).
        #
        # The default M0 = (0.10, 0, 0.90, 0) is the historical behaviour -- give the anchor
        # visual whole so the target modality has to be READ OFF it rather than hallucinated.
        # M2 = (0.90, 0.05, 0.05, 0) instead matches the official has_action split exactly, so
        # template identity stops predicting conditioning level (§10.3: today `<seg:...>` in the
        # prompt implies "RGB will be given", a shortcut that transfers nowhere) and the
        # auxiliary task trains in the same generative IIII regime the closed-loop rollout uses.
        #
        # NOTE the single-frame branch is NOT the has_action one. There, `single_frame` applies
        # to every VISUAL modality while the action segments keep full length, so something is
        # still predicted. In a perception template EVERY segment is visual, so collapsing them
        # all leaves each segment one latent frame that is also its own condition -- masks.all()
        # and the assert in `assemble` fires. The perception variant collapses only the ANCHOR,
        # leaving depth/seg full length: "from a single RGB frame, generate the whole depth/seg
        # video", which is a distinct and meaningful mode rather than a crash.
        # BRANCH ORDER IS LOAD-BEARING, and is not the tuple order. The historical code was a
        # single `if rng.random() < PERCEPTION_VIDEO_GIVEN_PROB: given_modalities = {anchor}`,
        # i.e. a LOW draw meant FIFI. Testing p_iiii first would keep the same marginal
        # distribution while inverting which outcome a given random value produces -- so an
        # existing seed would silently train on a different mask sequence, which is exactly the
        # class of change test_forward_unchanged exists to catch. FIFI stays first.
        p_iiii, p_fiii, p_fifi, p_single = perception_mask_mix or PERCEPTION_MASK_MIX_DEFAULT
        p = rng.random()
        if p < p_fifi:
            given_modalities = {anchor}             # FIFI: every anchor segment given whole
        elif p < p_fifi + p_iiii:
            pass                                    # IIII: every segment gives only frame 0
        elif p < p_fifi + p_iiii + p_fiii:
            give_first_segment = True               # FIII: segment 0 (view 0 anchor) given whole
        else:
            single_frame_anchor = True              # single-frame perception (see above)
    # A SINGLE-modality template (Pi = {video}, which is where action dropout lands) draws
    # nothing: giving its only modality whole would leave the sequence with no predicted
    # position at all, i.e. a zero-loss step that still costs a full forward and backward.
    # Upstream's video-only branch conditions on first frames only, and so does this.

    return _lay_out(
        mods, num_views, single_frame_visual, single_frame_anchor,
        given_modalities, absent_modalities, give_first_segment, anchor,
    )


def _lay_out(
    mods, num_views, single_frame_visual, single_frame_anchor,
    given_modalities, absent_modalities, give_first_segment, anchor,
) -> List[Segment]:
    """The layout loop, shared by every branch of `plan_segments`.

    View-major, modality-minor, modality in CANONICAL_ORDER. Extracted only so the fusion
    branch's degenerate early return builds a plan by the same code as everything else -- a
    second hand-written copy of this loop is exactly how a layout drifts.
    """
    plan: List[Segment] = []
    for v in range(num_views):
        for m in mods:
            is_segment_zero = not plan  # view 0's anchor, i.e. upstream's `masks[:, :, :T_l]`
            plan.append(
                Segment(
                    modality=m,
                    view=v,
                    single_frame=(single_frame_visual and m in VISUAL_MODALITIES)
                    or (single_frame_anchor and m == anchor),
                    fully_given=(m in given_modalities) or (give_first_segment and is_segment_zero),
                    absent=m in absent_modalities,
                )
            )
    return plan


def segment_spans(
    plan: Sequence[Segment], latents_by_modality: Dict[str, Sequence[torch.Tensor]]
) -> List[Tuple[int, int]]:
    """plan -> [(start, end), ...] latent-frame spans, in the same cumulative order `assemble`
    packs them. Lets a caller attribute anything computed on the packed sequence -- a loss, an
    attention map, a decoded frame -- back to the (modality, view) that produced it.

    Kept next to `assemble` and derived the same way, so the two cannot drift: a span table that
    disagrees with the packing would mislabel every per-modality number computed from it.
    """
    spans, offset = [], 0
    for seg in plan:
        n = 1 if seg.single_frame else int(latents_by_modality[seg.modality][seg.view].shape[2])
        spans.append((offset, offset + n))
        offset += n
    return spans


def fusion_regime_of(plan: Sequence[Segment]) -> str:
    """plan -> the F-axis regime name that produced it, for logging.

    Recovered from the plan rather than threaded out of `plan_segments`, so it stays a pure
    diagnostic: no caller can accidentally start branching on a regime the plan does not
    actually have.

    One genuine ambiguity, on TWO-modality templates only: `one_out` that happens to draw the
    non-anchor modality produces exactly the `rgb_only` plan, and is reported as `rgb_only`.
    They are the same sample; only the draw that reached it differs. On the 5-modality fusion
    canvas this cannot happen, because `rgb_only` marks four modalities absent and `one_out`
    marks one.
    """
    mods = tuple(dict.fromkeys(seg.modality for seg in plan))  # already in CANONICAL_ORDER
    anchor = primary_visual(mods)
    absent = {seg.modality for seg in plan if seg.absent}
    given = {seg.modality for seg in plan if seg.fully_given}
    others = set(mods) - {anchor}
    if any(seg.single_frame for seg in plan):
        return "policy"
    if given == {anchor} and absent == others:
        return "rgb_given"
    if not absent:
        return "full_anchor"
    if absent == others:
        return "rgb_only"
    if len(absent) == 1:
        return "one_out"
    return "unknown"


def describe_plan(plan: Sequence[Segment]) -> str:
    """plan -> 'V0:F D0:- S0:- N0:- A0:- | V1:F ...', for logs and per-segment loss tables.

    Role letters match the docstring of `Segment`: F fully given, I anchored (first frame only),
    `-` absent. A trailing `1` marks a single-frame (policy-mode) segment. Views are separated
    by ' | ' so the layout reads the way the templates docstring writes it.
    """
    out, prev_view = [], None
    for seg in plan:
        if prev_view is not None and seg.view != prev_view:
            out.append("|")
        role = "-" if seg.absent else ("F" if seg.fully_given else "I")
        tag = f"{seg.modality[0].upper()}{seg.view}:{role}{'1' if seg.single_frame else ''}"
        out.append(tag)
        prev_view = seg.view
    return " ".join(out)


def inference_plan(
    modalities: Sequence[str],
    num_views: int,
    mode: Optional[str] = None,
    fully_given: Optional[Sequence[str]] = None,
    absent: Optional[Sequence[str]] = None,
) -> List[Segment]:
    """The DETERMINISTIC eval-time counterpart of `plan_segments`: named mode -> plan.

    SINGLE SOURCE OF TRUTH, and that is the whole point. Both
    `WanVideoActionImagesPipeline.prepare_template_inference_latents` and the eval scripts need
    to know which segment got which role -- the pipeline to build the mask, the scorer to slice
    the decoded frames and to label a span "generated" vs "a VAE roundtrip of ground truth".
    They used to each carry their own transcription of it (see the old `_expected_segments` in
    scripts/modality_mode_grid.py, whose docstring said "Mirrors
    prepare_template_inference_latents exactly"). Two hand-written copies of a role table drift,
    and when they do the scorer reports one modality's metric under another modality's name --
    silently, because every number still looks plausible. One function, called twice.

    `mode` is one of the four historical codes or the five F-axis regime names; alternatively
    pass explicit `fully_given` / `absent` sets (which is how `one_out` is expressed, since that
    name denotes a random DRAW, not a conditioning).
    """
    mods = tuple(m for m in CANONICAL_ORDER if m in set(modalities))
    if not mods:
        raise ValueError(f"no known modality in {modalities!r}")
    anchor = primary_visual(mods)
    given, gone = set(fully_given or ()), set(absent or ())
    single_frame_visual = False

    if mode is not None:
        m = mode.lower()
        if given or gone:
            raise ValueError("pass either mode or fully_given/absent, not both")
        if m == "one_out":
            raise ValueError(
                "mode='one_out' is ambiguous -- it names a DRAW, not a conditioning. Say which "
                "modality to withhold with absent=['depth'] instead."
            )
        others = set(mods) - {anchor}
        table = {
            "iiii":        (set(),    set()),
            "full_anchor": (set(),    set()),
            "fiii":        (set(),    set()),     # segment 0 only; positional, applied below
            "fifi":        ({anchor}, set()),
            "f0f0":        ({anchor}, others),    # predates the F axis; same sample as rgb_given
            "rgb_given":   ({anchor}, others),
            "rgb_only":    (set(),    others),
            "policy":      (set(),    {o for o in others if o in VISUAL_MODALITIES}),
        }
        if m not in table:
            raise ValueError(
                f"unknown conditioning mode {mode!r}; expected one of {sorted(table)} "
                f"(or pass fully_given/absent explicitly)"
            )
        given, gone = table[m]
        single_frame_visual = m == "policy"
        if m == "policy" and ACTION not in mods:
            raise ValueError("single-frame policy mode is defined only for templates containing action")
    else:
        m = None
        unknown = (given | gone) - set(mods)
        if unknown:
            raise ValueError(f"modalities {sorted(unknown)} are not in {format_template(mods)!r}")
        overlap = given & gone
        if overlap:
            raise ValueError(
                f"modalities {sorted(overlap)} are both fully-given and absent; the two roles "
                f"are mutually exclusive"
            )

    plan = []
    for v in range(num_views):
        for mod in mods:
            idx = len(plan)
            plan.append(
                Segment(
                    modality=mod,
                    view=v,
                    single_frame=single_frame_visual and mod in VISUAL_MODALITIES,
                    fully_given=(m == "fiii" and idx == 0) or (mod in given),
                    absent=mod in gone,
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
        if seg.absent:
            pass  # no condition frames: this modality must come from the rest of the sequence
        elif seg.fully_given:
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
    # The symmetric failure: `absent` on every segment leaves the sequence with no clean pixel
    # anywhere, so the camera embedding describes a scene the model was never shown and the
    # sample degenerates to text-to-video. Unlike masks.all() this one is NOT caught by the loss
    # (it looks like a normal, very hard step), so it has to be asserted explicitly.
    assert not bool((~masks).all()), (
        "no latent position is a condition frame; this sample has no scene anchor at all and "
        f"degenerates to text-to-video. plan={[(s.modality, s.view, s.absent) for s in plan]}"
    )
    return latents, camera_emb, masks
