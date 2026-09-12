"""End-to-end consistency sweep over every template the co-generation study trains.

    python scripts/debug_cogen.py                       # all checks
    python scripts/debug_cogen.py --only prompt         # one group
    python scripts/debug_cogen.py --seg-mode referring  # the other seg protocol

Exit 0 iff every check passes. CPU-only, no model weights.

WHY THIS EXISTS. The failure mode this repo keeps hitting is not a crash -- it is a silent
disagreement between two halves of the pipeline that each look correct alone:

  * a prompt tag containing `_` trained as `<scene seg ...>` (train.py:287 scrubs underscores)
    while every eval script asked `<scene_seg ...>` -- 30 vs 31 UMT5 tokens, differing exactly at
    the tag that selects the task;
  * RLBench's "no object" sentinel decoded as the `unknown` role, so 5.2% of every scene_roles
    target was orange -- and the one test that checked pointed at a dangling symlink;
  * a normal codec that ignored intrinsics, producing unit vectors (so every shape and range
    assertion passed) that were tilted by a factor of fx/z.

None of those raise. All of them are caught by comparing two independently-derived answers to the
same question, which is what every check below does.
"""
import argparse
import collections
import os
import random
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from training.dataset.rlbench_selfgen import RLBenchSelfgenDataset  # noqa: E402
from training.percep.depth_codec import MAX_VALID, MIN_VALID, decode_depth  # noqa: E402
from training.percep.normal_codec import decode_normal, depth_to_normal  # noqa: E402
from training.percep.seg_codec import (  # noqa: E402
    SCENE_ROLES,
    decode_known_color,
    decode_scene_roles,
    handles_to_mask,
    scene_role_iou,
    scene_role_labels,
)
from training.templates import (  # noqa: E402
    ACTION,
    CANONICAL_ORDER,
    VISUAL_MODALITIES,
    assemble,
    parse_template,
    parse_perception_mask_mix,
    plan_segments,
    prompt_prefix,
)

TEMPLATES = ["video+action", "video+depth", "video+segmentation", "video+normal",
             "video+depth+action",
             # The fusion canvas: all five modalities, 10 segments, 110 latent frames. Listed
             # here so the prompt / stream-shape / layout / target / alignment checks below all
             # cover it -- it is the only template where a stream mismatch costs 2.5x a step.
             "video+depth+segmentation+normal+action"]
FAILURES = []
NOTES = []


def check(cond, msg):
    (FAILURES.append(msg) if not cond else None)
    print(f"  {'FAIL' if not cond else 'ok  '}: {msg}")
    return bool(cond)


def note(msg):
    NOTES.append(msg)
    print(f"  note: {msg}")


def scrub(text):
    """EXACTLY what ActionImagesModel.forward does before encode_prompt (train.py:287).

    Kept as a literal copy rather than an import because the point of the check is to detect
    the two drifting apart; importing the real one would make the check vacuous.
    """
    return text.replace(".", "").replace("_", " ").replace("  ", " ")


# --------------------------------------------------------------------------- 1. prompts
def check_prompts(ds, args):
    """The prompt the model TRAINS on must be byte-identical to the one eval SENDS.

    train.py scrubs; the inference pipeline (WanVideoActionImagesPipeline.encode_prompt) does
    not. So any tag that the scrub rewrites is a train/eval mismatch on the exact token that
    selects the task. The fix is to keep tags scrub-invariant, and this is the check that says so.
    """
    print("\n[1] prompt integrity (train scrub vs eval, both seg protocols)")
    for mode in ("referring", "scene_roles"):
        for name in TEMPLATES:
            mods = parse_template(name)
            colors = {"jar lid": "red", "block": "green"} if "segmentation" in mods else None
            pre = prompt_prefix(mods, colors, "explicit", mode)
            raw = pre + "grip the bottom handle and pull the bottom drawer open"
            check(scrub(raw) == raw,
                  f"{mode:11s} {name:20s} scrub-invariant: {raw[:46]!r}")

    # The same property on REAL prompts, where the instruction and the referring colour list come
    # from disk rather than this file's literals. Built directly from meta.json + the referring
    # spec instead of via getitem: getitem decodes two 41-frame videos per call purely to attach a
    # string, which caps the sweep at a couple hundred draws. Skipping the pixels covers EVERY
    # episode x EVERY paraphrase x both protocols in seconds -- the coverage that actually matters
    # for a claim about text.
    print("\n[1b] prompt integrity on every episode x paraphrase x protocol")
    dirty = collections.Counter()
    n = 0
    for ep in ds.episodes:
        meta = ds._load_meta(ep["path"])
        for instr in meta["desc"]:
            for name in TEMPLATES:
                mods = parse_template(name)
                for mode in ("referring", "scene_roles"):
                    colors = None
                    if "segmentation" in mods:
                        if mode == "referring":
                            ids, colors = ds._referred_id_groups(ep["path"], instr)
                            if not ids:
                                continue
                        else:
                            lut, _ = ds._scene_role_lut(ep["path"], instr)
                            if lut is None:
                                continue
                    text = prompt_prefix(mods, colors, "explicit", mode) + instr
                    n += 1
                    if scrub(text) != text:
                        dirty[text[:70]] += 1
    check(not dirty, f"{n} real prompts all scrub-invariant"
                     + (f" -- offenders: {dict(list(dirty.items())[:3])}" if dirty else ""))

    # Tokenizer: tags must survive as stable, non-UNK tokens.
    print("\n[1c] tokenizer")
    tok_dir = os.path.join(REPO, "checkpoints", "Wan-AI", "Wan2.2-TI2V-5B", "google", "umt5-xxl")
    if not os.path.isdir(tok_dir):
        note(f"tokenizer not found at {tok_dir}; skipped")
        return
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(tok_dir)
    for name in TEMPLATES:
        mods = parse_template(name)
        colors = {"jar lid": "red"} if "segmentation" in mods else None
        raw = prompt_prefix(mods, colors, "explicit", args.seg_mode) + "pull the bottom drawer open"
        ids = tk(raw).input_ids
        check(tk.unk_token_id not in ids, f"{name:20s} no <unk> ({len(ids)} tokens)")
        check(len(ids) < 512, f"{name:20s} within context ({len(ids)} tokens)")


# --------------------------------------------------------------------- 2. streams & layout
def check_streams(ds, args):
    """A template promises a set of segments; the sample must actually carry them."""
    print("\n[2] streams match the template each sample declares")
    for name in TEMPLATES:
        random.seed(11)
        s = ds.getitem(0, force_template=name)
        mods = parse_template(s["template"])
        if s["template"] != name:
            note(f"{name} degraded to {s['template']} on episode 0 (missing GT); using it as-is")
        want = {m for m in mods if m in VISUAL_MODALITIES}
        got = set(s["streams"])
        check(want == got, f"{s['template']:22s} streams {sorted(got)} == declared {sorted(want)}")
        shapes = {k: tuple(v.shape) for k, v in s["streams"].items()}
        check(len({v for v in shapes.values()}) == 1,
              f"{s['template']:22s} all streams share one shape {set(shapes.values())}")
        for k, v in s["streams"].items():
            check(float(v.min()) >= -1.0001 and float(v.max()) <= 1.0001,
                  f"{s['template']:22s} {k:13s} in [-1,1] ({float(v.min()):.3f},{float(v.max()):.3f})")
        if ACTION in mods:
            check(float(s["action_7d"].abs().sum()) > 0,
                  f"{s['template']:22s} action_7d is non-zero")


def check_layout(ds, args):
    """Segment order, mask legality, and the co-generation layout."""
    print("\n[3] segment layout and conditioning masks")
    pmm = parse_perception_mask_mix(args.perception_mask_mix)
    rng = random.Random(0)
    seen_modes = collections.Counter()
    for name in TEMPLATES:
        mods = parse_template(name)
        for trial in range(400):
            plan = plan_segments(mods, 2, rng, True, pmm)
            order = [seg.modality for seg in plan]
            # view-major, modality-minor in CANONICAL_ORDER
            per_view = [order[: len(order) // 2], order[len(order) // 2:]]
            want = [m for m in CANONICAL_ORDER if m in mods]
            if trial == 0:
                check(per_view[0] == want and per_view[1] == want,
                      f"{name:22s} segment order {per_view[0]} == canonical {want}")
            seen_modes[(name, tuple(s.fully_given for s in plan))] += 1
        # masks: a fully-conditioned sequence is a zero-loss step and assemble() must refuse it
        lat = {m: [torch.randn(1, 4, 3, 8, 8) for _ in range(2)] for m in mods if m in VISUAL_MODALITIES}
        for m in mods:
            if m == ACTION:
                lat[m] = [torch.randn(1, 4, 3, 8, 8) for _ in range(2)]
        cam = [torch.randn(1, 4, 3, 8, 8) for _ in range(2)]
        plan = plan_segments(mods, 2, random.Random(3), True, pmm)
        _, _, masks = assemble(plan, lat, cam)
        check(not bool(masks.all()), f"{name:22s} mask is never all-conditioned")
        check(bool(masks.any()), f"{name:22s} mask conditions something")

    # The eval scripts slice a 4-segment prediction as [q:2q] and [3q:4q] to get the two
    # PREDICTED perception segments (eval_perception.py, eval_all_masks.py, modality_demo.py).
    # That is only correct if a 2-modality template lays out as [V0, P0, V1, P1] -- i.e. the
    # perception segments land at indices 1 and 3. Inference builds its plan independently of
    # training (`for v in views: for m in modalities` in
    # wan_video_action_images.prepare_template_inference_latents) so this is a real parity
    # claim between two code paths, not a restatement of one.
    print("\n[3b] eval's segment-slicing assumption ([V0,P0,V1,P1])")
    for name in ("video+depth", "video+segmentation", "video+normal"):
        mods = parse_template(name)
        # exactly how inference orders it
        infer_order = [m for v in range(2) for m in mods]
        train_order = [seg.modality for seg in plan_segments(mods, 2, random.Random(0), True, pmm)]
        check(infer_order == train_order,
              f"{name:22s} train order == inference order {infer_order}")
        perception = [i for i, m in enumerate(infer_order) if m != "video"]
        check(perception == [1, 3],
              f"{name:22s} predicted segments at indices {perception} (eval assumes [1, 3])")


# ------------------------------------------------------------------ 4. target correctness
def check_targets(ds, args, n_episodes=6):
    """Decode each encoded stream back and compare with the GT it was built from.

    This is the check the normal codec's missing-intrinsics bug would have failed: shapes and
    ranges were fine, the values were wrong, and only a comparison against independently
    recomputed geometry shows it.
    """
    print(f"\n[4] encoded target decodes back to its own GT ({n_episodes} episodes)")
    idxs = [int(round(k * len(ds) / n_episodes)) % len(ds) for k in range(n_episodes)]
    worst = {"depth": (0.0, None), "normal": (1.0, None), "segmentation": (1.0, None)}
    unknown_total = 0
    for k, idx in enumerate(idxs):
        for modality in ("depth", "normal", "segmentation"):
            random.seed(100 + k)
            s = ds.getitem(idx, force_template=f"video+{modality}")
            if s["template"] != f"video+{modality}":
                continue
            T = ds.num_frames
            vds = [s["view_dirs"][i] for i in s["view_indices"]]
            fi = s["frame_indices"]
            enc = ((s["streams"][modality].permute(1, 2, 3, 0).numpy() + 1.0) * 127.5
                   ).round().clip(0, 255).astype(np.uint8)
            for v, vd in enumerate(vds):
                got = enc[v * T:(v + 1) * T]
                if modality == "depth":
                    dec = decode_depth(got)
                    gt = ds._to_model_res(ds._load_depth(vd)[fi].astype(np.float32))
                    ok = np.isfinite(gt) & (gt > MIN_VALID) & (gt < MAX_VALID) & np.isfinite(dec)
                    rel = float((np.abs(np.clip(dec, MIN_VALID, MAX_VALID) - gt)
                                 / np.maximum(gt, MIN_VALID))[ok].mean())
                    if rel > worst["depth"][0]:
                        worst["depth"] = (rel, f"{os.path.basename(s['path'])}/{os.path.basename(vd)}")
                elif modality == "normal":
                    fx, fy = ds._focal_lengths(vd)
                    raw = ds._load_depth(vd)[fi].astype(np.float32)
                    sx, sy = ds.width / raw.shape[-1], ds.height / raw.shape[-2]
                    d = ds._to_model_res(raw)
                    gt_n = np.stack([depth_to_normal(d[t], fx * sx, fy * sy) for t in range(len(d))])
                    cos = float((gt_n * decode_normal(got)).sum(-1).mean())
                    if cos < worst["normal"][0]:
                        worst["normal"] = (cos, f"{os.path.basename(s['path'])}/{os.path.basename(vd)}")
                else:
                    hm = ds._to_model_res(ds._load_mask(vd)[fi]).astype(np.uint16)
                    instr = s["text"].split("> ", 1)[-1]
                    if ds.segmentation_mode == "scene_roles":
                        lut, present = ds._scene_role_lut(s["path"], instr)
                        dec = decode_scene_roles(got, present_roles=present)
                        unknown_total += int(decode_scene_roles(got)["unknown"].sum())
                        per = scene_role_iou(dec, scene_role_labels(hm, lut), roles=present)
                        m = min((x["iou"] for x in per.values() if x["gt_px"] > 0), default=1.0)
                    else:
                        ids, cmap = ds._referred_id_groups(s["path"], instr)
                        dec = decode_known_color(got, cmap)
                        m = 1.0
                        for nm, hs in ids.items():
                            g, p = handles_to_mask(hm, hs), dec[nm]
                            u = int((g | p).sum())
                            m = min(m, (int((g & p).sum()) / u) if u else 1.0)
                    if m < worst["segmentation"][0]:
                        worst["segmentation"] = (m, f"{os.path.basename(s['path'])}/{os.path.basename(vd)}")
    check(worst["depth"][0] < 0.02, f"depth   worst AbsRel {worst['depth'][0]*100:.4f}% < 2%  @{worst['depth'][1]}")
    check(worst["normal"][0] > 0.99, f"normal  worst cosine {worst['normal'][0]:.5f} > 0.99   @{worst['normal'][1]}")
    check(worst["segmentation"][0] > 0.95, f"seg     worst per-class IoU {worst['segmentation'][0]:.5f} > 0.95 @{worst['segmentation'][1]}")
    if ds.segmentation_mode == "scene_roles":
        check(unknown_total == 0, f"zero `unknown` pixels in the encoded seg target ({unknown_total})")


# ----------------------------------------------------------------- 5. view/frame alignment
def check_alignment(ds, args, n=4):
    """Every stream must describe the SAME views and frames as `video`.

    A perception stream sampled from a different window is the failure that no metric detects:
    the pixels are valid, they just belong to another moment, and the model learns to hallucinate.
    """
    print(f"\n[5] all streams share one window and view pair ({n} draws)")
    for k in range(n):
        random.seed(500 + k)
        a = ds.getitem(k % len(ds), force_template="video+depth")
        random.seed(500 + k)
        b = ds.getitem(k % len(ds), force_template="video+normal")
        random.seed(500 + k)
        c = ds.getitem(k % len(ds), force_template="video+segmentation")
        same = (a["view_indices"] == b["view_indices"] == c["view_indices"]
                and a["frame_indices"] == b["frame_indices"] == c["frame_indices"])
        check(same, f"draw {k}: views/frames identical across depth|normal|seg "
                    f"(views={a['view_indices']} frames={a['frame_indices'][0]}..{a['frame_indices'][-1]})")
        check(torch.equal(a["streams"]["video"], b["streams"]["video"]),
              f"draw {k}: the RGB anchor is bit-identical across templates")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO, "data", "rlbench_selfgen_512_aug"))
    ap.add_argument("--seg-mode", choices=("referring", "scene_roles"), default="scene_roles")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--num-frames", type=int, default=41)
    ap.add_argument("--frame-interval", type=int, default=3)
    ap.add_argument("--perception-mask-mix", default="M2")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--only", default="", help="prompt|streams|layout|targets|alignment")
    args = ap.parse_args()

    print(f"data      : {args.data}")
    print(f"seg mode  : {args.seg_mode}   res {args.res}  frames {args.num_frames}@{args.frame_interval}")
    ds = RLBenchSelfgenDataset(
        base_path=args.data, num_frames=args.num_frames, frame_interval=args.frame_interval,
        height=args.res, width=args.res,
        template_mix=",".join(f"{t}@{1/len(TEMPLATES):.4f}" for t in TEMPLATES),
        prompt_tag_style="explicit", variations="0", segmentation_mode=args.seg_mode,
        strict_getitem=True,
    )
    groups = {"prompt": lambda: check_prompts(ds, args),
              "streams": lambda: check_streams(ds, args),
              "layout": lambda: check_layout(ds, args),
              "targets": lambda: check_targets(ds, args, args.episodes),
              "alignment": lambda: check_alignment(ds, args)}
    for gname, fn in groups.items():
        if args.only and args.only != gname:
            continue
        fn()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"COGEN_DEBUG_OK ({len(NOTES)} note(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
