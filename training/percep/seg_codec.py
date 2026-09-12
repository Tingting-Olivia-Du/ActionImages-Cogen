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


# ---- scene-role core: dense semantic roles (SEGMENTATION_SCENE_ROLES_PLAN.md §6.2) ----
#
# Colour binds a cross-task FUNCTIONAL ROLE, never a simulator handle and never "the Nth
# object". Same role -> same colour in every task, episode, frame and view.
#
# The role<->colour assignment is NOT arbitrary. Measured pairwise sRGB distances over these 9
# entries put the three tightest pairs at 98.3 (distractor/tool), 109.2 (fixture/unknown) and
# 109.4 (target/tool). TAU=40 above is calibrated against the palette minimum 98.3, so whichever
# pair sits at 98.3 is the one most likely to be confused after VAE noise. `tool` and `unknown`
# are therefore SWAPPED relative to the plan's first draft: putting the tightest pair on
# distractor/tool would have assigned it to two foreground classes that genuinely co-occur
# (sweep_to_dustpan has a broom=tool next to distractors). `unknown` must never appear in valid
# training data (§7.3 hard-fails on it), so parking it on the crowded slot costs nothing.
SCENE_ROLES = (
    "background",   # 0  floor / wall / table / workspace
    "target",       # 1  the object the instruction acts on
    "goal",         # 2  container / receptacle / destination
    "robot_arm",    # 3  Panda links
    "gripper",      # 4  gripper + fingers
    "distractor",   # 5  non-target movable objects
    "tool",         # 6  broom / stick: intermediate implements
    "fixture",      # 7  drawer frame / grill / tap body / rack
    "unknown",      # 8  unmapped handle -- debug only, must be absent from training data
)
SCENE_ROLE_PALETTE = np.array(
    [
        [0, 0, 0],        # background  black
        [230, 25, 75],    # target      red
        [60, 180, 75],    # goal        green
        [0, 130, 200],    # robot_arm   blue
        [70, 240, 240],   # gripper     cyan
        [255, 225, 25],   # distractor  yellow
        [240, 50, 230],   # tool        magenta   <- swapped with unknown, see above
        [145, 30, 180],   # fixture     purple
        [245, 130, 48],   # unknown     orange    <- swapped with tool, see above
    ],
    dtype=np.uint8,
)
ROLE_TO_LABEL = {r: i for i, r in enumerate(SCENE_ROLES)}
BACKGROUND_LABEL = ROLE_TO_LABEL["background"]
UNKNOWN_LABEL = ROLE_TO_LABEL["unknown"]

# Roles that instruction resolution may assign on top of a handle's instruction-independent
# base_role (§6.3). Only these two actually conflict; gripper/robot_arm handle sets are disjoint.
INSTRUCTION_ROLES = ("target", "goal")

# RLBench renders "no object" (the sky above the room walls) as CoppeliaSim's -1, which
# rgb_handles_to_mask turns into 0xFFFFFF = 16777215; gen_dataset.py then stores the mask as
# uint16, truncating it to 0xFFFF. It is not an annotation gap -- it is empty space, i.e.
# background, and it is exactly what depth_codec already encodes as the far clip plane.
#
# Measured over 11200 frames of rlbench_selfgen_512_aug (scripts/audit_scene_roles.py): without
# this entry 5.20% of ALL pixels decode as `unknown` orange, up to 52.89% in a single view, and
# handles.json carries the "16777215": "" entry in 1005 of 1028 episodes. That violates
# SEGMENTATION_SCENE_ROLES_PLAN.md 12.1 ("unknown must be absent from training data") on almost
# every sample.
#
# Mapping it here rather than in scene_segments.json is deliberate: build_role_lut's LUT is only
# 65536 wide, so the pre-truncation 16777215 cannot be stored as a key at all -- the sentinel
# belongs to whoever knows about the uint16 cast, which is the codec. The largest real handle
# observed across the tree is ~110, so 0xFFFF cannot collide with a genuine object.
NO_OBJECT_HANDLE = 0xFFFF


def build_role_lut(handle_to_role: dict, max_handle: int = 65536) -> np.ndarray:
    """{handle_id: role_name} -> uint8 LUT[max_handle] mapping raw handle -> role label.

    Handles absent from the mapping become UNKNOWN_LABEL rather than background: an unmapped
    handle is an annotation gap and must stay visible (magenta-equivalent orange), not be
    silently absorbed into the dominant class where nobody would ever notice it.
    """
    lut = np.full(max_handle, UNKNOWN_LABEL, dtype=np.uint8)
    for handle, role in handle_to_role.items():
        if role not in ROLE_TO_LABEL:
            raise ValueError(f"unknown role {role!r} for handle {handle}; expected one of {SCENE_ROLES}")
        h = int(handle)
        if not 0 <= h < max_handle:
            raise ValueError(f"handle {h} outside [0, {max_handle})")
        lut[h] = ROLE_TO_LABEL[role]
    # Handle 0 is CoppeliaSim's "nothing"; RLBench never renders it, but if it appears it is a
    # background pixel, not an annotation gap. NO_OBJECT_HANDLE is the one that actually shows up
    # (see its definition): empty space above the walls, not a missing annotation. Both are set
    # AFTER the loop so an explicit mapping could still override them if one ever existed.
    lut[0] = BACKGROUND_LABEL
    lut[NO_OBJECT_HANDLE] = BACKGROUND_LABEL
    return lut


def encode_scene_roles(handle_map: np.ndarray, role_lut: np.ndarray) -> np.ndarray:
    """uint16 handle map [...,H,W] + LUT from build_role_lut -> uint8 RGB [...,H,W,3].

    Pure LUT indexing, no per-instance mask construction: the handle map already assigns each
    pixel to exactly one handle, so roles cannot overlap and there is no paint order to get
    wrong (which is exactly the failure mode encode_seg_multi's "later entries overwrite
    earlier ones" carries).
    """
    labels = role_lut[handle_map.astype(np.intp)]
    return SCENE_ROLE_PALETTE[labels]


def scene_role_labels(handle_map: np.ndarray, role_lut: np.ndarray) -> np.ndarray:
    """uint16 handle map -> uint8 role-label map [...,H,W] (the pre-colour intermediate)."""
    return role_lut[handle_map.astype(np.intp)]


def decode_scene_roles(rgb: np.ndarray, present_roles=None) -> dict:
    """uint8 RGB [...,H,W,3] -> {role_name: bool mask [...,H,W]}.

    `present_roles`: which roles this episode can legally contain. Restricting the nearest-colour
    search to them is NOT an optimisation -- it is the same invariant decode_known_color relies
    on ("unused reserved colors must not participate, else they can steal ambiguous pixels").
    Which roles an episode contains is derivable from scene_segments.json + the instruction, so
    it is available at both train and eval time; decoding against all 9 colours unconditionally
    would widen the misclassification surface for no reason. Defaults to every role.

    No connected-component pruning, small-region removal or erosion -- see the module docstring:
    Appendix A's THETA_SIZE pruning deletes sweep_to_dustpan's dirt (5 handles of 4-8px) outright.
    """
    if present_roles is None:
        present_roles = SCENE_ROLES
    roles = list(dict.fromkeys(present_roles))  # de-dup, keep order
    unknown = [r for r in roles if r not in ROLE_TO_LABEL]
    if unknown:
        raise ValueError(f"not scene roles: {unknown}; expected from {SCENE_ROLES}")
    if "background" not in roles:
        # Background must always compete, otherwise every black pixel is forced into some
        # foreground role and the decode is nonsense.
        roles = ["background"] + roles
    colors = np.stack([SCENE_ROLE_PALETTE[ROLE_TO_LABEL[r]].astype(np.float32) for r in roles], axis=0)
    x = rgb.astype(np.float32)
    dists = ((x[..., None, :] - colors) ** 2).sum(-1)   # [...,H,W,K]
    nearest = dists.argmin(-1)
    return {role: (nearest == i) for i, role in enumerate(roles)}


def build_scene_prompt(instruction: str, protocol: str = "rlbench-scene-role-v1") -> str:
    """instruction -> '<scene-seg> instruction'.

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

    `protocol` is accepted and ignored, so existing callers keep working.
    """
    return f"<scene-seg> {instruction}"


def scene_role_iou(pred_masks: dict, gt_labels: np.ndarray, roles=None) -> dict:
    """{role: bool mask} + uint8 GT label map -> {role: IoU}.

    Roles absent from BOTH prediction and GT get IoU 1.0 (vacuously correct); absent from GT but
    predicted gets 0.0. Callers aggregating a macro mIoU should drop the vacuous entries rather
    than let them inflate the mean -- `gt_px` is returned alongside so they can.
    """
    roles = list(roles or pred_masks)
    out = {}
    for role in roles:
        gt = gt_labels == ROLE_TO_LABEL[role]
        pd = pred_masks.get(role, np.zeros_like(gt))
        inter = int((pd & gt).sum())
        union = int((pd | gt).sum())
        out[role] = {
            "iou": (float(inter) / float(union)) if union else 1.0,
            "gt_px": int(gt.sum()),
            "pred_px": int(pd.sum()),
        }
    return out


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
