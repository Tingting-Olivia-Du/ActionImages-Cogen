"""Self-generated RLBench loader: RGB + per-frame GT depth/mask + camera, in one sample.

Ported from ttd/src/data/rlbench_selfgen.py. Subclasses the official RLBenchMVDataset so the
action / camera / view-selection contract stays byte-identical to the released stack. The
selfgen tree differs from the HF-released one in four spots, each overridden below:

  1. camera_params.json keyed by non-zero-padded frame index ("0", not "0000").
  2. task descriptions live in meta.json ("desc"), not variation_descriptions.pkl.
  3. everything is written at native 20Hz 1:1, so actions must NOT be `[::4]`-downsampled.
  4. per-view depth.npz / mask.npz + per-episode seg_targets.json supply perception GT.

WHAT THIS FORK CHANGES vs the ttd original
------------------------------------------
ttd used this dataset for a MUTUALLY EXCLUSIVE design: a sample was either an action sample
or a perception sample, and perception samples had `action_7d` zeroed so the model's
`torch.sum(action_7d**2) != 0` check routed them to a video-only branch. Action supervision
and perception supervision could therefore never occur in the same gradient step. Actions are
no longer zeroed here, which is what unlocks the `video+depth+action` cell.

An earlier revision of this fork made the perception modality REPLACE the RGB pixels in
`sample["video"]`, giving `[v1_depth | v1_action | v2_depth | v2_action]`. That kept `forward`
literally unchanged, but it also means a depth sample contains NO RGB at all -- so it is a
depth-space WORLD MODEL, not perception in the GenCeption / Vision Banana sense (RGB in ->
depth out). It is still reachable, as the `depth+action` template, and must be reported as
such rather than as a perception result.

The current design emits one tensor per modality in `sample["streams"]` plus a
`sample["template"]` naming the modality set, and `forward` packs the segments from those.
Perception is then expressible: template `video+depth`, RGB segments fully given, depth
segments predicted -- and it costs exactly what the official 4-segment recipe costs, because
depth takes the ACTION slot rather than being appended.

Invariants preserved from the previous revision:
  - BOTH views are encoded for a perception modality, never one view RGB and one view depth.
  - Perception GT is read from the SAME window and the SAME view dirs that produced the RGB,
    via the first-class provenance fields (`frame_indices` / `view_indices` / `view_dirs`).

See /workspace/ttdu/ttd/plan/core/16-multitask_template_design.md (the template design) and
15-a4_io_current_vs_desired_action_perception_cogen.md (§3 for why no output head is needed,
§6.2 for why co-supervision is the point), plus this repo's FORK_CHANGES.md.
"""
import glob
import json
import os
import random
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from training.dataset.rlbench import RLBenchMVDataset
from training.percep.depth_codec import encode_depth
from training.percep.normal_codec import encode_normal_from_depth
from training.percep.seg_codec import (
    UNKNOWN_LABEL,
    build_referring_spec,
    build_role_lut,
    encode_known_color,
    encode_scene_roles,
    handles_to_mask,
)
from training.templates import (
    ACTION,
    VISUAL_MODALITIES,
    draw_template,
    format_template,
    parse_template,
    parse_template_mix,
    primary_visual,
    prompt_prefix,
)

# Visual modalities whose pixels this dataset must load and encode itself. `video` comes from
# the parent class' RGB path; the rest go through `_encode_perception`.
PERCEPTION_MODALITIES = tuple(m for m in VISUAL_MODALITIES if m != "video")

# Above this fraction of episodes lacking seg_targets.json, a segmentation template is not
# training what its command line says and the run is refused rather than warned about. 10% is
# well above the ~1% of genuinely un-annotatable episodes in the original tree, and far below
# the 70% that a round of episode generation produced.
SEG_COVERAGE_MIN_MISSING_TO_RAISE = 0.10

# How many episodes _check_scene_roles_ready opens at startup. Each one decompresses two full
# mask volumes (~200x512x512 uint16), so this trades a few seconds of startup against how likely
# a localised annotation gap is to be caught. 4 spread across the index hits 4 different tasks;
# scripts/audit_scene_roles.py is the exhaustive check for when that is not enough.
SCENE_ROLES_PROBE_EPISODES = 4


class RLBenchSelfgenDataset(RLBenchMVDataset):
    """RLBench selfgen episodes; `template_mix` picks which task template each sample teaches."""

    # The only tree with per-frame depth/mask GT, hence the only one that can serve a
    # perception template.
    AVAILABLE_MODALITIES = ("video", "depth", "segmentation", "normal", "action")
    # getitem below applies the template itself (it must also build the perception streams);
    # letting BaseDataset do it too would prepend the prompt tags twice.
    TEMPLATE_IN_BASE = False

    def _try_load_cache(self) -> bool:
        """Always rescan self-generated data, which may grow between training runs.

        BaseDataset's cache key only describes loader parameters; it does not include the
        contents of the dataset tree. A cache made while generation was still running can
        therefore silently hide newly generated episodes. Scanning this directory is cheap
        compared with training and guarantees `self.episodes` reflects the current tree.
        """
        return False

    def _save_cache(self) -> None:
        # Do not write an index that this appendable dataset cannot safely reuse.
        return None

    def __init__(self, *args, template_mix: str = "video+action@1.0",
                 prompt_tag_style: str = "explicit", action_dropout_prob: float = 0.1,
                 strict_getitem: bool = False, variations: str = "all",
                 segmentation_mode: str = "referring",
                 legacy_scene_seg_tag: bool = False, **kwargs):
        self.strict_getitem = strict_getitem
        self.variations = variations
        self.prompt_tag_style = prompt_tag_style
        self.action_dropout_prob = float(action_dropout_prob)
        if segmentation_mode not in ("referring", "scene_roles"):
            raise ValueError(
                f"segmentation_mode must be 'referring' or 'scene_roles', got {segmentation_mode!r}"
            )
        self.segmentation_mode = segmentation_mode
        # Only for evaluating pre-2026-08-23 checkpoints; see templates.scene_seg_tag_legacy.
        self.legacy_scene_seg_tag = bool(legacy_scene_seg_tag)
        self._templates, self._template_cum = parse_template_mix(template_mix)
        # {(task, group_name): count} of seg samples where build_referring_spec silently dropped
        # a defined-but-zero-handle group (e.g. reach_and_drag's "target" when target0 is
        # occluded this episode). In-process only; a dict counter is the lowest-overhead way to
        # keep the GT-completeness gap observable instead of invisible.
        self._seg_dropped_group_counts: dict = {}

        # Skip RLBenchMVDataset.__init__ (it loads variation_descriptions.pkl, absent in
        # selfgen); go straight to BaseDataset.__init__, descriptions come from meta.json.
        from training.dataset.base import BaseDataset

        self.task_descriptions = {}
        BaseDataset.__init__(self, *args, **kwargs)

        self._apply_variation_filter()
        self._check_seg_coverage()
        self._check_scene_roles_ready()
        self._self_test()

    def _check_seg_coverage(self) -> None:
        """Refuse to silently train a seg template on episodes that have no seg ground truth.

        `seg_targets.json` is produced OFFLINE by ttd's `src/percep/seg_targets_gen.py`, not by
        the episode generator. So every episode `gen_dataset.py` adds arrives without it, and
        `_referred_id_groups` returns nothing for those -- at which point `getitem` drops
        `segmentation` from the template and the sample silently becomes a plain `video` one.

        That failure is invisible in every other instrument: loss looks normal, `_self_test`
        passes (it probes episodes until one works), and the run's own logs say the mix is
        whatever was on the command line. Measured 2026-08-10, 70% of variation0 episodes had
        no seg_targets.json (97% of the newly generated ones), which turned a nominal
        60/20/20 mix into roughly 60% action / 20% depth / 6% seg / 14% VIDEO-ONLY. Those
        video-only samples carry neither action nor perception supervision, so an arm that
        hits this is not comparable to a control arm that does not.
        """
        if not any("segmentation" in mods for mods in self._templates):
            return
        # WHICH FILE MATTERS DEPENDS ON THE PROTOCOL. Under `referring`, seg_targets.json IS the
        # signal, and its absence is what silently degrades the sample (the failure this guard was
        # written for). Under `scene_roles` the drop decision in getitem keys off `role_lut is
        # None`, i.e. off scene_segments.json; seg_targets.json is consulted only for the optional
        # instruction overlay that promotes a `distractor` to `target`, and _scene_role_lut states
        # outright that its absence is "NOT fatal". Checking seg_targets under scene_roles
        # therefore rejects trees that would evaluate perfectly well -- which is exactly what it
        # did to the never-trained-task tree, whose episodes carry scene_segments.json and no
        # seg_targets.json.
        required = ("scene_segments.json" if self.segmentation_mode == "scene_roles"
                    else "seg_targets.json")
        missing = [ep["path"] for ep in self.episodes
                   if not os.path.exists(os.path.join(ep["path"], required))]
        if not missing:
            print(f"[selfgen] seg coverage OK: {len(self.episodes)}/{len(self.episodes)} "
                  f"episodes carry {required}")
            return
        frac = len(missing) / max(len(self.episodes), 1)
        # parse_template_mix returns CUMULATIVE probabilities; recover the per-template share.
        prev = 0.0
        seg_weight = 0.0
        for mods, cum in zip(self._templates, self._template_cum):
            if "segmentation" in mods:
                seg_weight += cum - prev
            prev = cum
        msg = (
            f"{len(missing)}/{len(self.episodes)} episodes ({frac:.1%}) have no "
            f"{required}, e.g. {missing[0]}.\n"
            f"  Those samples DROP segmentation and become plain `video` samples -- silently.\n"
            f"  With seg weight {seg_weight:.0%} in template_mix, about {seg_weight * frac:.1%} "
            f"of ALL training samples become video-only (no action, no perception), which is a\n"
            f"  confound a control arm does not share.\n"
            f"  Fix: run `python -m training.percep.scene_segments_gen --root <tree> --write` "
            f"(scene_roles) or ttd's src/percep/seg_targets_gen.py (referring), or drop the "
            f"segmentation template from --template_mix."
        )
        if frac > SEG_COVERAGE_MIN_MISSING_TO_RAISE:
            raise RuntimeError("[selfgen] seg coverage too low: " + msg)
        print("[selfgen] WARNING: partial seg coverage: " + msg)

    def _check_scene_roles_ready(self) -> None:
        """Refuse to start a scene_roles run whose target would contain `unknown` pixels.

        `scene_roles` has one hard invariant (SEGMENTATION_SCENE_ROLES_PLAN.md 12.1): the
        `unknown` role -- orange, meaning "a rendered handle nobody assigned a role" -- must not
        appear in training data. It is the only role whose presence is always a metadata bug
        rather than a property of the scene.

        It went undetected once already. RLBench's "no object" sentinel (the sky above the walls)
        reached the LUT unmapped and painted 5.2% of ALL pixels orange -- up to 52.9% of a single
        view -- while every instrument stayed green: the loss is a plain MSE against whatever the
        target is, `_self_test` only checks that a template does not degrade, and the one test
        that did check pointed at a data root whose symlink had gone dangling. Two more handles
        (Panda_link0/link1, absent from some episodes' 3-frame handles.json sample) survived even
        the first fix. So the check belongs at startup, where it costs four episodes of IO and
        cannot be skipped, rather than only in a test file.

        scripts/audit_scene_roles.py is the exhaustive version; this is the cheap gate.
        """
        if self.segmentation_mode != "scene_roles":
            return
        if not any("segmentation" in mods for mods in self._templates):
            return
        missing = [ep["path"] for ep in self.episodes
                   if not os.path.exists(os.path.join(ep["path"], "scene_segments.json"))]
        if missing:
            frac = len(missing) / max(len(self.episodes), 1)
            msg = (
                f"[selfgen] {len(missing)}/{len(self.episodes)} episodes ({frac:.1%}) have no "
                f"scene_segments.json, e.g. {missing[0]}.\n"
                f"  Under --segmentation_mode scene_roles those samples DROP segmentation and "
                f"become plain `video` samples -- silently.\n"
                f"  Fix: python -m training.percep.scene_segments_gen "
                f"--root <tree> --write --verify-masks"
            )
            if frac > SEG_COVERAGE_MIN_MISSING_TO_RAISE:
                raise RuntimeError(msg)
            print("[selfgen] WARNING: " + msg)

        # Probe a spread of episodes, not the first few: episodes are grouped by task on disk and
        # the sentinel's incidence is task- and camera-dependent.
        n = len(self.episodes)
        probe = sorted({(i * n) // SCENE_ROLES_PROBE_EPISODES for i in range(SCENE_ROLES_PROBE_EPISODES)})
        worst = (0, None)
        checked = 0
        for i in probe:
            ep = self.episodes[i]["path"]
            meta = self._load_meta(ep)
            lut, _ = self._scene_role_lut(ep, meta["desc"][0])
            if lut is None:
                continue
            for view in sorted(glob.glob(os.path.join(ep, "view*")))[:2]:
                mask_path = os.path.join(view, "mask.npz")
                if not os.path.exists(mask_path):
                    continue
                labels = lut[np.load(mask_path)["mask"].astype(np.intp)]
                unk = int((labels == UNKNOWN_LABEL).sum())
                checked += 1
                if unk > worst[0]:
                    worst = (unk, view)
        if worst[0]:
            raise RuntimeError(
                f"[selfgen] scene_roles target contains {worst[0]} `unknown` (orange) pixels, "
                f"worst at {worst[1]}.\n"
                f"  An unmapped handle reached the training target. Run "
                f"`python scripts/audit_scene_roles.py --root {self.base_path}` for the full "
                f"list, then regenerate scene_segments.json.\n"
                f"  See SEGMENTATION_SCENE_ROLES_PLAN.md 12.1 -- this class must be absent."
            )
        print(f"[selfgen] scene_roles OK: zero `unknown` pixels over {checked} probed views")

    def _apply_variation_filter(self) -> None:
        """Keep only the variations this split is allowed to see.

        The train/test split is BY VARIATION: train on variation0, hold out the rest as unseen
        task configurations (different target colours, object placements, button orders, ...).
        `_load_dataset` globs `variation*` indiscriminately, so without this filter a "train"
        run silently trains on its own test set and every held-out number is meaningless. That
        failure is invisible -- loss and r_peak both look fine, they are just measured on data
        the model memorised.

        Accepted syntax:
          "all"        every variation (the default; unchanged behaviour)
          "0"          only variation0                      -> the TRAIN split
          "0,1" "0-2"  an explicit set
          "!0"         everything EXCEPT variation0         -> the TEST split
        """
        spec = (self.variations or "all").strip()
        if spec == "all":
            return
        exclude = spec.startswith("!")
        wanted = set()
        for part in spec.lstrip("!").split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-")
                wanted.update(range(int(a), int(b) + 1))
            else:
                wanted.add(int(part))
        if not wanted:
            raise ValueError(f"variations={self.variations!r} selects nothing")

        before = len(self.episodes)
        kept = []
        for ep in self.episodes:
            idx = int(ep["variation"].replace("variation", ""))
            if (idx in wanted) != exclude:
                kept.append(ep)
        self.episodes = kept
        if not self.episodes:
            raise ValueError(
                f"variations={self.variations!r} left 0 episodes out of {before} under "
                f"{self.base_path}. Check the variation indices actually present on disk."
            )
        n_tasks = len({ep["task"] for ep in self.episodes})
        print(f"[selfgen] variations={self.variations!r}: kept {len(self.episodes)}/{before} "
              f"episodes across {n_tasks} tasks")

    def _self_test(self) -> None:
        """Build one sample per template at construction time, before anything expensive.

        A broken perception path (missing depth.npz, a resolution mismatch, a codec change)
        otherwise first shows up inside the DataLoader, after the 5B model has loaded and
        wandb has started a run -- and BaseDataset.__getitem__ would retry around it. Calling
        `getitem` DIRECTLY (not `__getitem__`) means no retry wrapper and a real traceback.

        Every template in the menu is exercised, not just one: a mix that trains 90% of its
        steps on a working template and 10% on a broken one is exactly the case that would
        otherwise surface hours in.
        """
        want_shape = (3, 2 * self.num_frames, self.height, self.width)
        for mods in self._templates:
            if all(m == "video" or m == ACTION for m in mods):
                continue  # pure RGB path, byte-identical to upstream, nothing new to break
            name = format_template(mods)
            # Segmentation legitimately resolves to nothing on SOME episodes (the instruction
            # names no object this episode's seg_targets.json knows), in which case getitem
            # degrades the template. Walk a few episodes so "seg never resolves anywhere" --
            # a real, silent breakage -- is distinguished from "not on episode 0".
            sample = None
            for probe in range(min(5, len(self.episodes))):
                sample = self.getitem(probe, force_template=name)
                if sample["template"] == name:
                    break
            if sample["template"] != name:
                raise RuntimeError(
                    f"self-test: template {name!r} degraded to {sample['template']!r} on every "
                    f"one of the first {min(5, len(self.episodes))} episodes. The perception GT "
                    f"for this template is not resolvable on this data tree."
                )
            for key, tensor in sample["streams"].items():
                assert tuple(tensor.shape) == want_shape, (
                    f"self-test: {name} stream {key!r} is {tuple(tensor.shape)}, "
                    f"expected {want_shape}"
                )
                assert float(tensor.min()) >= -1.01 and float(tensor.max()) <= 1.01, (
                    f"self-test: {name} stream {key!r} outside [-1,1]: "
                    f"[{float(tensor.min()):.3f}, {float(tensor.max()):.3f}]"
                )
            print(f"[selfgen] self-test OK: template={sample['template']} "
                  f"streams={sorted(sample['streams'])} text={sample['text'][:70]!r}")

    def get_instruction(self, episode_info) -> str:
        meta = self._load_meta(episode_info["path"])
        return random.choice(meta["desc"])

    # ---- override: no [::4] downsampling ----
    # HF-released data stores video already downsampled and actions at raw 20Hz, so the
    # official loader does actions[::4] to realign. Selfgen writes everything at native
    # 20Hz 1:1 (video == depth == mask == actions == camera length), so we must NOT downsample.
    def get_8d_action(self, episode_path, frame_indices=None):
        actions = self._load_8d_actions_base(episode_path)  # [T_raw, 8], native
        if frame_indices is not None:
            actions = actions[frame_indices]
        return actions

    @lru_cache(maxsize=64)
    def _load_meta(self, episode_path: str) -> dict:
        with open(os.path.join(episode_path, "meta.json")) as f:
            return json.load(f)

    # ---- override: non-zero-padded camera keys ----
    def get_camera_params(self, camera_params_path, frame_indices=None):
        with open(camera_params_path) as f:
            camera_data = json.load(f)
        if frame_indices is None:
            frame_indices = list(range(self.num_frames))
        intr_list, extr_list = [], []
        for idx in frame_indices:
            key = str(idx)
            if key not in camera_data:
                key = max(camera_data.keys(), key=lambda k: int(k))
            intr_list.append(np.array(camera_data[key]["intrinsics"]))
            extr_list.append(np.array(camera_data[key]["extrinsics"]))
        return np.stack(extr_list), np.stack(intr_list)

    # ---- perception ground truth ----
    # maxsize is small on purpose: each entry is a full decompressed depth/mask volume
    # (157x256x256), and a perception sample now touches TWO views, so a large cache
    # multiplies worker RSS by dataloader_num_workers for very little hit-rate gain.
    @lru_cache(maxsize=4)
    def _load_depth(self, view_path: str) -> np.ndarray:
        return np.load(os.path.join(view_path, "depth.npz"))["depth"]  # [T,H,W] float16

    @lru_cache(maxsize=32)
    def _focal_lengths(self, view_path: str):
        """-> (|fx|, |fy|) in pixels at the NATIVE render resolution.

        Frame 0 only: intrinsics do not vary within an episode -- the Colosseum augmentation
        randomises camera POSE, table and lighting, never the lens (verified over 80 view dirs
        spanning all 16 tasks: zero episodes have a per-frame intrinsics change). Reading one
        frame instead of `num_frames` keeps this off the hot path.

        Magnitudes: RLBench writes fx and fy negative because its image y axis points opposite
        the pinhole convention. normal_codec takes abs() too; doing it here as well means a
        caller reading this value for anything else does not inherit the sign trap.
        """
        with open(os.path.join(view_path, "camera_params.json")) as f:
            cam = json.load(f)
        k = cam[min(cam.keys(), key=lambda x: int(x))]["intrinsics"]
        return abs(float(k[0][0])), abs(float(k[1][1]))

    @lru_cache(maxsize=4)
    def _load_mask(self, view_path: str) -> np.ndarray:
        return np.load(os.path.join(view_path, "mask.npz"))["mask"]  # [T,H,W] uint16

    @lru_cache(maxsize=64)
    def _load_seg_targets(self, episode_path: str) -> Optional[dict]:
        """episode_path/seg_targets.json, generated offline from the actual RLBench task source
        (register_graspable_objects etc). Returns None if the episode predates that file -- the
        caller must degrade to video, never fall back to substring matching over handle names
        (several tasks' target handle is an episode-random variable, and some handle names
        coincidentally contain a target substring while denoting an unrelated part). See
        ttd/plan/core/12-seg-codec-redesign-bugfix-and-vb-parity.md §1-2."""
        p = os.path.join(episode_path, "seg_targets.json")
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)

    def _referred_id_groups(self, episode_path: str, instruction: str):
        """-> ({instance_name: [handle_ids]}, {instance_name: color_name}), or ({}, {}) if this
        episode/instruction resolves nothing to segment.

        Groups that build_referring_spec reports as dropped (defined in seg_targets.json but
        resolving to zero handles this episode, e.g. an occluded reach_and_drag "target") are
        tallied so the GT-completeness gap stays observable. That tally does NOT change the
        returned groups."""
        seg_targets = self._load_seg_targets(episode_path)
        if seg_targets is None:
            return {}, {}
        try:
            id_groups, color_map, dropped_groups = build_referring_spec(seg_targets, instruction)
        except ValueError:
            return {}, {}
        if dropped_groups:
            task = seg_targets.get("task", "?")
            for name in dropped_groups:
                key = (task, name)
                self._seg_dropped_group_counts[key] = self._seg_dropped_group_counts.get(key, 0) + 1
        return id_groups, color_map

    @lru_cache(maxsize=64)
    def _load_scene_segments(self, episode_path: str) -> Optional[dict]:
        """episode_path/scene_segments.json -> {"instances": {name: {handles, base_role}}}.

        Deliberately holds NO target/goal resolution. Those depend on the instruction
        (push_buttons and put_groceries_in_cupboard both pick their target from the text), and
        seg_targets.json + build_referring_spec is already the single source of truth for that
        (SEGMENTATION_SCENE_ROLES_PLAN.md §7.1). Storing a second copy here is how the two
        would drift apart while every acceptance check kept passing.
        """
        p = os.path.join(episode_path, "scene_segments.json")
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)

    def _scene_role_lut(self, episode_path: str, instruction: str):
        """-> (uint8 LUT[65536] handle->role label, sorted list of roles present), or (None, None).

        Composition order IS the role priority (§6.3): every handle first takes its
        instruction-independent `base_role`, then the instruction's referred groups overwrite
        theirs with `target`. Only that one level actually conflicts -- gripper and robot_arm
        handle sets are disjoint, so their relative order never matters.
        """
        scene = self._load_scene_segments(episode_path)
        if not scene:
            return None, None
        handle_to_role = {}
        for inst in scene.get("instances", {}).values():
            role = inst.get("base_role")
            if role is None:
                continue
            for h in inst.get("handles", []):
                handle_to_role[int(h)] = role
        if not handle_to_role:
            return None, None

        # Instruction-dependent overlay. A missing/unresolvable seg_targets.json is NOT fatal
        # here (unlike the referring protocol, where it is the entire signal): the scene still
        # has robot, fixtures and distractors to supervise.
        #
        # seg_targets.json lists every object the instruction REFERS TO, which is not the same
        # as the object being manipulated: 9 of the 16 tasks name two or three (put_item_in_drawer
        # names the item AND the drawer; sweep_to_dustpan names broom, dirt AND dustpan). Marking
        # all of them `target` would repaint the receptacle and the implement red and destroy
        # exactly the goal/tool distinction this protocol exists to express.
        #
        # So the two sources are combined by what each actually knows. seg_targets knows "the
        # instruction mentions this object" (and resolves the episode-random colour cases);
        # scene_segments knows "this object is a receptacle / implement / fixture". A referred
        # object is promoted to `target` only where its base role says it is a manipulable
        # object; a referred goal stays a goal. That is §6.3's `target > goal > tool > fixture`
        # chain read correctly -- the chain disambiguates an object that qualifies for several
        # roles, it does not licence overwriting a more specific role with a less specific one.
        PROMOTABLE = {"distractor"}
        id_groups, _ = self._referred_id_groups(episode_path, instruction)
        for handles in id_groups.values():
            for h in handles:
                if handle_to_role.get(int(h)) in PROMOTABLE:
                    handle_to_role[int(h)] = "target"

        lut = build_role_lut(handle_to_role)
        present = sorted(set(handle_to_role.values()) | {"background"})
        return lut, present

    def _to_model_res(self, gt: np.ndarray) -> np.ndarray:
        """[T,H,W] annotation at native render resolution -> [T,height,width], NEAREST.

        Resize the GT *before* encoding, never the encoded colour. Bilinear across a depth
        discontinuity invents a depth that exists nowhere in the scene; bilinear across the
        RGB-cube path invents a colour that path_decode happily projects onto the wrong
        segment. Both produce plausible-looking pixels that are quietly wrong.
        """
        if gt.shape[-2:] == (self.height, self.width):
            return gt
        t = torch.from_numpy(gt.astype(np.float32))[:, None]  # [T,1,h,w]
        t = F.interpolate(t, size=(self.height, self.width), mode="nearest")
        return t[:, 0].numpy()

    def _encode_perception(self, modality, view_path, frame_indices, ctx) -> torch.Tensor:
        """-> [C, T, H, W] float in [-1,1], matching what frame_process produces for RGB.

        The square assertion is load-bearing rather than defensive: frame_process applies
        CenterCropToAspect + Resize to RGB but the perception path only resizes, so on a
        non-square target the two halves of `video` would differ in shape and torch.cat would
        raise -- inside a retry loop, as endless log spam rather than a failure.
        """
        assert self.height == self.width, (
            f"perception modalities assume a square target (CenterCropToAspect is a no-op only "
            f"then); got {self.height}x{self.width}"
        )
        if modality == "depth":
            depth = self._to_model_res(self._load_depth(view_path)[frame_indices].astype(np.float32))
            rgb = encode_depth(depth)  # [T,H,W,3] uint8
        elif modality == "normal":
            raw = self._load_depth(view_path)[frame_indices].astype(np.float32)
            fx, fy = self._focal_lengths(view_path)
            # _to_model_res subsamples the depth grid, so one pixel spans more of the scene and
            # the focal length IN PIXELS shrinks by exactly that factor. Without this rescale a
            # 256 arm's normals are tilted ~2x relative to a 512 arm's from identical geometry --
            # and nothing downstream would flag it, because the result is still a unit field.
            sx, sy = self.width / raw.shape[-1], self.height / raw.shape[-2]
            depth = self._to_model_res(raw)
            rgb = encode_normal_from_depth(depth, fx * sx, fy * sy)  # [T,H,W,3] uint8
        elif modality == "segmentation":
            mask_map = self._to_model_res(self._load_mask(view_path)[frame_indices]).astype(np.uint16)
            if "role_lut" in ctx:
                # Scene roles: pure LUT indexing. No per-instance masks and therefore no paint
                # order -- the handle map already assigns each pixel to exactly one handle.
                rgb = encode_scene_roles(mask_map, ctx["role_lut"])  # [T,H,W,3] uint8
            else:
                instance_masks = {
                    name: handles_to_mask(mask_map, ids) for name, ids in ctx["id_groups"].items()
                }
                rgb = encode_known_color(instance_masks, ctx["color_map"])  # [T,H,W,3] uint8
        else:
            raise ValueError(f"not a perception modality: {modality!r}")
        assert rgb.shape == (len(frame_indices), self.height, self.width, 3), (
            f"{modality} encode produced {rgb.shape}, expected "
            f"{(len(frame_indices), self.height, self.width, 3)}"
        )
        t = torch.from_numpy(rgb).float() / 127.5 - 1.0
        return rearrange(t, "t h w c -> c t h w")

    def getitem(self, index, force_template: Optional[str] = None):
        sample = super().getitem(index)  # official skeleton; carries window/view provenance
        rgb = sample["video"]

        if force_template is not None:
            mods = parse_template(force_template)
        else:
            mods = draw_template(self._templates, self._template_cum)
            # Upstream's action gate (`random.random() < 0.9` at train.py:286) lives here now:
            # deciding it in the dataset is what keeps the prompt and the segment layout
            # consistent. Deciding it in forward -- as upstream does -- means the text can
            # promise an action stream that the assembled sequence does not contain.
            if ACTION in mods and random.random() < self.action_dropout_prob:
                mods = tuple(m for m in mods if m != ACTION)

        color_map = None
        ctx: dict = {}
        # ORDERING IS LOAD-BEARING. Both seg protocols resolve their object handles by matching
        # words of the INSTRUCTION against seg_targets.json. `sample["text"]` is still the bare
        # instruction at this point; the prompt prefix is prepended at the END of this method.
        # Reversing the two would feed `<video><seg: ...> pick up the lid` into the matcher,
        # resolve nothing, and silently degrade the sample to video-only -- with a normal-looking
        # loss curve. Assert rather than trust, because the failure is invisible downstream.
        assert not sample["text"].startswith("<"), (
            f"seg context must be resolved from the raw instruction, but sample['text'] is "
            f"already prefixed: {sample['text'][:60]!r}. The prompt_prefix() call at the end of "
            f"getitem must stay AFTER this block."
        )
        if "segmentation" in mods:
            if self.segmentation_mode == "scene_roles":
                role_lut, present_roles = self._scene_role_lut(sample["path"], sample["text"])
                if role_lut is None:
                    # No scene_segments.json for this episode (or it resolves nothing). Same
                    # policy as referring below: drop segmentation rather than emit a degenerate
                    # target. An all-background role map would teach "the scene is empty".
                    mods = tuple(m for m in mods if m != "segmentation")
                    if not any(m in VISUAL_MODALITIES for m in mods):
                        mods = ("video",) + mods
                else:
                    ctx = {"role_lut": role_lut, "present_roles": present_roles}
            else:
                id_groups, color_map = self._referred_id_groups(sample["path"], sample["text"])
                if not id_groups:
                    # Nothing in this episode's seg_targets.json resolves for this instruction.
                    # Drop segmentation rather than emit an all-empty target, which would teach a
                    # bogus "usually nothing to segment" prior. Fall back to RGB if that would
                    # leave the sequence with no visual anchor at all.
                    mods = tuple(m for m in mods if m != "segmentation")
                    color_map = None
                    if not any(m in VISUAL_MODALITIES for m in mods):
                        mods = ("video",) + mods
                else:
                    ctx = {"id_groups": id_groups, "color_map": color_map}

        perception = [m for m in mods if m in PERCEPTION_MODALITIES]
        streams = {}
        if "video" in mods:
            streams["video"] = rgb

        if perception:
            view_dirs, view_indices = sample["view_dirs"], sample["view_indices"]
            frame_indices = sample["frame_indices"]
            if not view_dirs or len(view_indices) != 2 or max(view_indices) >= len(view_dirs):
                # Refuse to guess. Re-deriving the window/views here would be a fresh random
                # draw describing different pixels than the camera tensors already in `sample`
                # -- the exact misalignment ttd's P0-1 fix existed to prevent.
                raise RuntimeError(
                    f"missing view provenance for {sample['path']}: view_dirs={view_dirs} "
                    f"view_indices={view_indices}. RLBenchMVDataset.get_all_views must record "
                    f"_last_view_dirs (see FORK_CHANGES.md)."
                )
            for modality in perception:
                per_view = [
                    self._encode_perception(modality, view_dirs[vi], frame_indices, ctx)
                    for vi in view_indices
                ]
                # BOTH views, in the same [cond | target] order super().getitem() used for
                # `video`, so the camera/extrinsics/intrinsics tensors keep describing these
                # exact pixels -- and so an RGB segment and a depth segment of the same view
                # are the same scene from the same camera at the same instants.
                streams[modality] = torch.cat(per_view, dim=1)  # C, 2T, H, W

        sample["streams"] = streams
        sample["template"] = format_template(mods)
        # `video` stays populated as the shape reference forward derives B/T/H/W from, and as
        # the back-compat alias for anything that still reads a single visual tensor. For the
        # substitution templates (no `video` in Pi) that is the perception tensor, which is
        # exactly the previous revision's behaviour.
        sample["video"] = streams[primary_visual(mods)]
        sample["visual_modality"] = primary_visual(mods)
        # Applied LAST, and exactly once: everything above that reads `sample["text"]` needs the
        # bare instruction (see the assert near the top of this method).
        assert not sample["text"].startswith("<"), (
            f"prompt prefix applied twice: {sample['text'][:60]!r}"
        )
        sample["text"] = (
            prompt_prefix(mods, color_map, self.prompt_tag_style, self.segmentation_mode,
                          legacy_scene_seg_tag=self.legacy_scene_seg_tag)
            + sample["text"]
        )
        # NOTE: action_7d / action_8d are deliberately left INTACT. Zeroing them is what made
        # perception and action mutually exclusive in ttd; keeping them is what makes
        # video+depth+action co-supervision happen at all.
        return sample
