"""Referring-seg <-> RGB codec (Vision Banana, arXiv:2604.20329, segmentation section).
See plan/core/12-seg-codec-redesign-bugfix-and-vb-parity.md for the derivation.

Two encode/decode families, per plan §3.1 (the paper uses ONE encoding mechanism -- text prompt
describing a color mapping, model generates an RGB image -- for all its segmentation variants;
only the prompt content and the decode algorithm differ):

  known-color (semantic / referring): every target's color is decided at ENCODE time and known
  to the decoder up front (encode_known_color/decode_known_color). This is what RLBench referring
  data actually is -- see plan §5, no task here has a genuinely unknown-instance-count scenario.

  instance (Appendix A, seed-based flood-fill over an UNKNOWN number of runtime-chosen colors):
  kept as encode_seg_multi/decode_seg_multi (legacy name) for now -- see plan §5, not needed by
  any of the 16 current RLBench tasks, so its known-color-decode shortcut (passing color_map back
  into decode) is retained as-is rather than rewritten into true Appendix A seed flood-fill. New
  code should use encode_known_color/decode_known_color; this legacy pair is not deprecated (its
  Appendix A post-processing -- background threshold, small-region pruning, erosion, TAU=40
  recalibration -- is real and still used), just not the current default for referring GT.

known-color decode has NO connected-component pruning, small-region removal, or erosion (unlike
the instance decoder's Appendix A steps 3-4): a real referring target can legitimately be a few
scattered pixels (e.g. sweep_to_dustpan's "dirt", 5 handles of 4-8px each) and running Appendix
A's THETA_SIZE=2e-4 (~13px at 256x256) pruning against it silently drops all of them -- confirmed
empirically in reports/seg_codec_survey/ (dirt IoU=0.0 on a CLEAN, noise-free roundtrip). Known-
color decode is just per-pixel nearest-color classification against the color map baked into the
prompt/spec, so it can't misfire on small legitimate targets the way instance-mode post-processing
does.

Pure functions + roundtrip acceptance (IoU > 0.98 required by spec).
All uint8 RGB in [0,255].
"""
import numpy as np
from scipy import ndimage

# 8-color palette. Pairwise sRGB distance ranges 98.3 (yellow-orange, the tightest pair) to
# 268.8 (red-cyan); background (black) is kept out of the palette (never assigned to an
# instance). NOT all pairs reach a uniform "150" separation target -- TAU below is calibrated
# against the actual minimum (98.3), not an aspirational uniform spacing.
SEG_PALETTE = np.array(
    [
        [230, 25, 75],    # red
        [60, 180, 75],    # green
        [255, 225, 25],   # yellow
        [0, 130, 200],    # blue
        [245, 130, 48],   # orange
        [145, 30, 180],   # purple
        [70, 240, 240],   # cyan
        [240, 50, 230],   # magenta
    ],
    dtype=np.uint8,
)
COLOR_NAMES = ["red", "green", "yellow", "blue", "orange", "purple", "cyan", "magenta"]
SEG_BG = np.array([0, 0, 0], dtype=np.uint8)

# TAU: recalibrated from the paper's literal value (14) to 40 (module docstring). At sigma=12
# per-channel Gaussian noise, tau=40 keeps background-gate escape probability ~0.3% (vs ~37%
# at tau=14); tau=40 is still well under half the palette's minimum pairwise distance (98.3/2
# ~= 49.1), so it doesn't blur the boundary between adjacent real colors either.
TAU = 40.0
THETA_SIZE = 2e-4       # fraction of image area (paper value, unchanged)
THETA_EROSION = 0.1     # paper value, unchanged
GAMMA_BBOX = 5.0         # bbox-growth cap for step 5 (fragment merge); not exercised by current
                         # decode_seg_multi since RLBench referred objects are single-component.


def encode_seg_multi(instance_masks: dict) -> tuple:
    """{instance_name: bool mask [...,H,W]} -> (uint8 RGB [...,H,W,3], {name: color_name}).

    Later entries overwrite earlier ones on overlapping pixels; callers should ensure the
    per-instance masks don't overlap (RLBench handle maps assign each pixel to at most one
    handle, so in practice they don't).
    """
    names = list(instance_masks)
    if len(names) > len(SEG_PALETTE):
        raise ValueError(f"too many instances ({len(names)}) for a {len(SEG_PALETTE)}-color palette")
    color_map = {n: COLOR_NAMES[i] for i, n in enumerate(names)}
    shape = next(iter(instance_masks.values())).shape
    rgb = np.zeros(shape + (3,), dtype=np.uint8)
    rgb[...] = SEG_BG
    for i, n in enumerate(names):
        rgb[instance_masks[n]] = SEG_PALETTE[i]
    return rgb, color_map


def decode_seg_multi(rgb: np.ndarray, color_map: dict) -> dict:
    """uint8 RGB [H,W,3] -> {instance_name: bool mask [H,W]}, Vision Banana Appendix A algorithm.

    Only supports single [H,W,3] frames (ndimage.label needs a fixed 2D connectivity structure);
    callers with a [T,H,W,3] video should call this per frame.
    """
    assert rgb.ndim == 3 and rgb.shape[-1] == 3, "decode_seg_multi expects a single [H,W,3] frame"
    x = rgb.astype(np.float32)
    total_area = x.shape[0] * x.shape[1]

    # 1. background threshold
    bg_mask = ((x - SEG_BG) ** 2).sum(-1) < TAU ** 2

    # 2. nearest-color grouping among colors actually used this scene (not the full palette --
    # unused reserved colors must not participate, else they can steal ambiguous pixels).
    used_names = list(color_map.keys())
    used_colors = np.array(
        [SEG_PALETTE[COLOR_NAMES.index(color_map[n])] for n in used_names], dtype=np.float32
    )
    dists = ((x[..., None, :] - used_colors) ** 2).sum(-1)   # [H,W,K]
    nearest = dists.argmin(-1)
    nearest = np.where(bg_mask, -1, nearest)

    out = {}
    for k, name in enumerate(used_names):
        comp = nearest == k
        # 16-connectivity ~= 8-connectivity (3x3 structuring element) for 2D connected components.
        labeled, n_labels = ndimage.label(comp, structure=np.ones((3, 3)))
        mask = np.zeros_like(comp)
        for lbl in range(1, n_labels + 1):
            region = labeled == lbl
            area = region.sum()
            # 3. small-region pruning
            if area < THETA_SIZE * total_area:
                continue
            # 4. boundary-artifact cleanup via erosion
            eroded = ndimage.binary_erosion(region, structure=np.ones((3, 3)))
            if eroded.sum() < THETA_EROSION * area:
                continue
            mask |= region
        out[name] = mask
    # 5. bbox-constrained fragment merge: no-op here (see module docstring); RLBench referred
    # objects are single rigid bodies so step 3-4 already produce one component per instance.
    return out


def build_seg_prompt(instance_color_map: dict, instruction: str) -> str:
    """{instance_name: color_name} + instruction -> structured "<seg: ...>" prompt prefix."""
    spec = ", ".join(f"{name}={color}" for name, color in instance_color_map.items())
    return f"<seg: {spec}> {instruction}"


def handles_to_mask(handle_map: np.ndarray, handles) -> np.ndarray:
    """RLBench uint16 handle map [..., H, W] + iterable of handles -> binary mask."""
    return np.isin(handle_map, np.asarray(list(handles)))


def roundtrip_iou(instance_masks: dict) -> dict:
    """{name: bool mask [H,W]} -> {name: IoU} after encode -> decode."""
    rgb, color_map = encode_seg_multi(instance_masks)
    rec = decode_seg_multi(rgb, color_map)
    out = {}
    for name, m in instance_masks.items():
        m = m.astype(bool)
        r = rec[name]
        inter = (r & m).sum()
        union = (r | m).sum()
        out[name] = float(inter) / float(union) if union > 0 else 1.0
    return out


# ---- known-color core: semantic + referring (plan/core/12-...-vb-parity.md §3.1-3.2) ----

def encode_known_color(class_masks: dict, color_map: dict) -> np.ndarray:
    """{name: bool mask [...,H,W]} + {name: color_name (must be in COLOR_NAMES)} -> uint8 RGB.

    Unlike encode_seg_multi, color_map is an INPUT here (caller decides colors, e.g. from
    seg_targets.json's per-episode target resolution), not an output -- known-color encoding
    doesn't need to invent a color assignment, it paints exactly the colors the caller specifies.
    """
    names = list(class_masks)
    for n in names:
        if n not in color_map:
            raise ValueError(f"class_masks has {n!r} but color_map doesn't specify its color")
        if color_map[n] not in COLOR_NAMES:
            raise ValueError(f"color {color_map[n]!r} for {n!r} not in COLOR_NAMES")
    shape = next(iter(class_masks.values())).shape
    rgb = np.zeros(shape + (3,), dtype=np.uint8)
    rgb[...] = SEG_BG
    for n in names:
        rgb[class_masks[n]] = SEG_PALETTE[COLOR_NAMES.index(color_map[n])]
    return rgb


def decode_known_color(rgb: np.ndarray, color_map: dict) -> dict:
    """uint8 RGB [...,H,W,3] -> {name: bool mask [...,H,W]}.

    Pure per-pixel nearest-color classification against color_map's colors + background -- NO
    connected-component pruning, small-region removal, or erosion (see module docstring for why:
    those are Appendix A instance-decode steps that silently drop small-but-legitimate referring
    targets like scattered dirt particles). Background is whichever color_map name (if any) maps
    to a color, plus implicit SEG_BG if no name claims black.
    """
    x = rgb.astype(np.float32)
    names = list(color_map)
    colors = np.stack(
        [SEG_BG.astype(np.float32)] + [SEG_PALETTE[COLOR_NAMES.index(color_map[n])].astype(np.float32) for n in names],
        axis=0,
    )  # [1+K, 3], index 0 = background
    dists = ((x[..., None, :] - colors) ** 2).sum(-1)   # [...,H,W,1+K]
    nearest = dists.argmin(-1)
    return {name: (nearest == (i + 1)) for i, name in enumerate(names)}


def build_referring_spec(seg_targets: dict, instruction: str) -> tuple:
    """seg_targets.json content (plan §2.1) + instruction text -> ({name: [handle_ids]}, prompt
    color_map {name: color_name}, dropped_groups). Starts from the default "referring" groups,
    then narrows any group named by referring_by_color[word]["replaces"] down to that color's
    single resolved handle, for every color word actually present in the instruction (the
    episode-random-color case: push_buttons, insert_onto_square_peg both need this -- see
    seg_targets_gen.py's _color_result, which is what writes the "replaces" key so this function
    never has to guess which default group a color word narrows).

    Multiple color words can replace the SAME default group in one instruction (push_buttons:
    "push the maroon button, then the green button, then the blue button" names 3 distinct
    targets) -- each gets its own suffixed group name ("push buttons target: maroon", etc) rather
    than overwriting the same dict key, which would silently keep only the last-processed color
    and merge the others back into a single unnarrowed blob.

    Raises ValueError if every resolved group has zero handles (nothing to segment this episode)
    -- callers should treat that as "no seg sample constructible", not build an empty-mask target
    (plan/core/10-...md §7.3: an all-empty seg target teaches the model a bogus "usually nothing
    to segment" prior).

    dropped_groups (third return value): names of groups that WERE defined in seg_targets.json
    (after color-word narrowing) but resolved to zero handles and so are silently absent from
    id_groups/color_map -- e.g. reach_and_drag's "target" group when target0 happens to be fully
    occluded in this episode (plan/core/14-seg-codec-gpu01-debug-plan.md §4b: confirmed physical
    occlusion in ~94% of reach_and_drag episodes, not a bug, but callers need visibility into it
    rather than an indistinguishable-from-complete GT). Empty list when nothing was dropped (the
    common case for tasks with no occlusion-prone parts). This is INFORMATIONAL ONLY -- it does
    not change what gets returned in id_groups/color_map, callers decide what to do with it (see
    rlbench_selfgen.py's _referred_id_groups, which accumulates per-task/per-group drop counts).
    """
    groups = dict(seg_targets.get("referring", {}))  # name -> {"handles": [...]}
    by_color = seg_targets.get("referring_by_color", {})
    low = instruction.lower()

    replaced_groups = set()
    for color_word, spec in by_color.items():
        if color_word in low and "replaces" in spec:
            target_group = spec["replaces"]
            if target_group not in replaced_groups:
                groups.pop(target_group, None)  # first color word for this group clears the default
                replaced_groups.add(target_group)
            groups[f"{target_group}: {color_word}"] = {"handles": spec["handles"]}

    id_groups = {name: spec["handles"] for name, spec in groups.items() if spec.get("handles")}
    dropped_groups = [name for name, spec in groups.items() if not spec.get("handles")]
    if not id_groups:
        raise ValueError(f"no resolvable target handles for instruction {instruction!r}")

    color_map = {name: COLOR_NAMES[i % len(COLOR_NAMES)] for i, name in enumerate(id_groups)}
    return id_groups, color_map, dropped_groups
