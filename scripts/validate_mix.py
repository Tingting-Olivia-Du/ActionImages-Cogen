"""Check the three-dataset + depth mix before committing a long run.

Verifies, in order:
  1. capability gating -- a menu a tree cannot serve is a startup error, not a silent downgrade
  2. per-dataset samples -- shapes, prompt tags present exactly once, streams match the template
  3. CombDataset joint distribution of (dataset, template) against what was requested
  4. mask-mode marginals from plan_segments for both the action and the perception branch

Run:
    python scripts/validate_mix.py                       # the planned config
    python scripts/validate_mix.py --comb_samples 600    # tighter distribution estimate
"""
import argparse
import collections
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset import CombDataset
from training.templates import (
    ACTION,
    VISUAL_MODALITIES,
    assert_menu_supported,
    parse_action_mask_mix,
    parse_per_dataset_template_mix,
    parse_perception_mask_mix,
    parse_template_mix,
    plan_segments,
)

DEFAULT_DS = "rlbench_selfgen_512_aug@0.62,droid@0.28,bridge@0.10"
DEFAULT_MENU = (
    "rlbench_selfgen_512_aug=video+action@0.67,video+depth@0.33;"
    "droid=video+action@1.0;"
    "bridge=video@1.0"
)
DEFAULT_PMM = "0.81,0.045,0.045,0.10"
# Atomic tags whose prompt must contain AT MOST ONE occurrence. The parameterised segmentation
# tags are checked separately below: `<seg: item=red, drawer=green>` carries a colour list, so a
# fixed-string count would not detect a duplicate, and `<scene_seg protocol=...>` must never
# appear alongside `<seg:` -- the two protocols produce different images from the same
# instruction and the model has no other way to tell them apart.
TAGS = ("<video>", "<action>", "<depth>", "<normal>")
# No trailing space: the tag is the atomic `<scene-seg>` since 2026-08-23. Matching on
# "<scene-seg" alone also catches the legacy `<scene-seg protocol=...>` form, which is what we
# want -- both are segmentation tags and neither may appear twice or alongside `<seg:`.
SEG_TAG_PREFIXES = ("<seg:", "<scene-seg")


def mask_mode(plan):
    """Classify one sampled plan as iiii / fiii / fifi / policy-single.

    The anchor is DERIVED from the plan, never assumed to be `video`. A hardcoded anchor makes
    this function silently wrong for the substitution templates (`depth+action` and friends,
    arm7's whole menu): the generator below would filter to zero segments, `all([])` is True,
    and every single plan would be reported as `fifi`. A preflight that cannot distinguish the
    mask modes is worse than none -- it says PASS while describing a distribution that does not
    exist.
    """
    anchor = next(s.modality for s in plan if s.modality in VISUAL_MODALITIES)
    anchor_full = all(s.fully_given for s in plan if s.modality == anchor)
    if anchor_full:
        return "fifi"
    if plan[0].fully_given:
        return "fiii"
    if any(s.single_frame for s in plan):
        return "policy/single"
    return "iiii"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="./data")
    ap.add_argument("--dataset_name", default=DEFAULT_DS)
    ap.add_argument("--template_mix_per_dataset", default=DEFAULT_MENU)
    ap.add_argument("--perception_mask_mix", default=DEFAULT_PMM)
    ap.add_argument("--action_mask_mix", default=None,
                    help="A0/A1/A2 or 'iiii,fiii,fifi,policy'. Default None = A0 = upstream.")
    ap.add_argument("--action_dropout_prob", type=float, default=0.1)
    ap.add_argument("--num_frames", type=int, default=41)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--per_dataset", type=int, default=6)
    ap.add_argument("--comb_samples", type=int, default=300)
    # train_mix.sh passes --variations 0, which cuts selfgen from 1028 episodes to 788. Without
    # it this script validates a dataset the run never sees.
    ap.add_argument("--variations", default="0")
    # Must match the run being validated: the two protocols emit different prompt tags and
    # different pixels, so validating `referring` tells you nothing about a `scene_roles` arm.
    ap.add_argument("--segmentation_mode", default="referring",
                    choices=("referring", "scene_roles"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    problems = 0
    menus = parse_per_dataset_template_mix(args.template_mix_per_dataset, "video+action@1.0")
    requested = {}
    for spec in args.dataset_name.split(","):
        name, _, ratio = spec.strip().partition("@")
        requested[name.strip()] = float(ratio)
    total_r = sum(requested.values())
    requested = {k: v / total_r for k, v in requested.items()}

    print("=" * 78)
    print("1. capability gating")
    print("=" * 78)
    from training.dataset.base import BaseDataset
    from training.dataset.bridge import BridgeMVDataset
    from training.dataset.droid import DROIDMVDataset
    from training.dataset.rlbench import RLBenchMVDataset
    from training.dataset.rlbench_selfgen import RLBenchSelfgenDataset

    caps = {
        "bridge": BridgeMVDataset.AVAILABLE_MODALITIES,
        "droid": DROIDMVDataset.AVAILABLE_MODALITIES,
        "rlbench": RLBenchMVDataset.AVAILABLE_MODALITIES,
        "rlbench_selfgen_512_aug": RLBenchSelfgenDataset.AVAILABLE_MODALITIES,
    }
    for name, av in caps.items():
        print(f"  {name:26s} provides {list(av)}")
    for name, mix in menus.items():
        assert_menu_supported(name, mix, caps.get(name, BaseDataset.AVAILABLE_MODALITIES))
        print(f"  OK  {name} = {mix}")
    # negative control: the gate must actually fire
    try:
        assert_menu_supported("bridge", "video+action@1.0", caps["bridge"])
        print("  !! capability gate did NOT fire for bridge=video+action")
        problems += 1
    except ValueError:
        print("  OK  gate fires for the impossible menu (bridge=video+action)")

    print()
    print("=" * 78)
    print(f"2. CombDataset: {args.dataset_name}")
    print("=" * 78)
    comb = CombDataset(
        dataset_path=args.data_path,
        dataset_specs=args.dataset_name,
        num_frames=args.num_frames,
        frame_interval=1,
        height=args.height,
        width=args.width,
        action_dropout_prob=args.action_dropout_prob,
        template_mix_per_dataset=args.template_mix_per_dataset,
        variations=args.variations,
        segmentation_mode=args.segmentation_mode,
    )
    print(f"  CombDataset length = {len(comb)}")

    loader = torch.utils.data.DataLoader(
        comb, batch_size=1, shuffle=True, num_workers=args.workers, drop_last=True
    )
    seen = collections.Counter()
    joint = collections.Counter()
    T = args.num_frames
    for i, b in enumerate(loader):
        if i >= args.comb_samples:
            break
        p = b["path"][0]
        src = ("bridge" if "/bridge/" in p else "droid" if "/droid/" in p
               else "rlbench_selfgen_512_aug" if "selfgen" in p else "rlbench")
        tmpl = b["template"][0]
        seen[src] += 1
        joint[(src, tmpl)] += 1
        if tuple(b["video"].shape) != (1, 3, 2 * T, args.height, args.width):
            print(f"  !! {src}/{tmpl}: video {tuple(b['video'].shape)}")
            problems += 1
        text = b["text"][0]
        for tag in TAGS:
            if text.count(tag) > 1:
                print(f"  !! duplicated {tag} in prompt: {text[:70]!r}")
                problems += 1
        n_seg_tags = sum(text.count(pfx) for pfx in SEG_TAG_PREFIXES)
        if n_seg_tags > 1:
            print(f"  !! {n_seg_tags} segmentation tags in one prompt: {text[:90]!r}")
            problems += 1
        if "segmentation" in tmpl and n_seg_tags != 1:
            print(f"  !! template {tmpl} but {n_seg_tags} segmentation tags: {text[:90]!r}")
            problems += 1
        for m in ("depth", "segmentation", "normal"):
            if m in tmpl and m not in b["streams"]:
                print(f"  !! template {tmpl} but no {m} stream")
                problems += 1
        if ACTION in tmpl and float(torch.sum(b["action_7d"].double() ** 2)) == 0:
            print(f"  !! {src}: template promises <action> but action_7d is all zeros")
            problems += 1

    tot = sum(seen.values())
    print(f"\n  drew {tot} batches")
    print(f"  {'dataset':28s} {'got':>6s} {'requested':>10s}")
    for name in requested:
        print(f"  {name:28s} {seen[name]/tot:6.3f} {requested[name]:10.3f}")
    print(f"\n  {'(dataset, template)':52s} {'share':>7s}")
    for (src, tmpl), n in sorted(joint.items(), key=lambda kv: -kv[1]):
        print(f"  {src + ' / ' + tmpl:52s} {n/tot:7.3f}")
    by_tmpl = collections.Counter()
    for (_src, tmpl), n in joint.items():
        by_tmpl[tmpl] += n
    print(f"\n  global task shares:")
    for tmpl, n in sorted(by_tmpl.items(), key=lambda kv: -kv[1]):
        print(f"    {tmpl:24s} {n/tot:6.3f}")

    print()
    print("=" * 78)
    print("3. mask-mode marginals (40k draws each)")
    print("=" * 78)
    pmm = parse_perception_mask_mix(args.perception_mask_mix)
    amm = parse_action_mask_mix(args.action_mask_mix)
    print(f"  perception_mask_mix -> {tuple(round(v, 4) for v in pmm)}")
    print(f"  action_mask_mix     -> {tuple(round(v, 4) for v in amm)}")
    # Every template actually reachable from either axis. The substitution family is listed
    # explicitly because it is arm7's whole menu and it is the family the old hardcoded anchor
    # got wrong -- if these four rows do not agree with each other, the anchor logic is broken
    # again, since by construction they differ only in which visual modality they carry.
    probes = [
        (("video", ACTION), {"action_mask_mix": amm}),
        (("depth", ACTION), {"action_mask_mix": amm}),
        (("segmentation", ACTION), {"action_mask_mix": amm}),
        (("normal", ACTION), {"action_mask_mix": amm}),
        (("video", "depth"), {"perception_mask_mix": pmm}),
    ]
    for mods, kw in probes:
        rng = random.Random(0)
        c = collections.Counter()
        for _ in range(40000):
            c[mask_mode(plan_segments(mods, 2, rng=rng, is_rlbench=True, **kw))] += 1
        n = sum(c.values())
        name = "+".join(mods)
        print(f"  {name:22s} " + "  ".join(f"{k}={c[k]/n:.3f}" for k in
                                           ("iiii", "fiii", "fifi", "policy/single")))

    print()
    print("RESULT:", "PASS" if problems == 0 else f"FAIL ({problems} problems)")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
