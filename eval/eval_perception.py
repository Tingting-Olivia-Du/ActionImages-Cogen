"""Evaluate a Cogen checkpoint on one prompt-controlled RLBench perception episode.

The sequence and masks match training exactly:
  video+depth:        V0 | D0 | V1 | D1
  video+segmentation: V0 | S0 | V1 | S1
RGB/video segments are fully given; perception segments keep their first frame and predict the
rest.  Dataset provenance supplies the exact views and temporal window used for both pixels and
camera tensors.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from diffsynth import save_video
from inference import build_pipeline
from training.dataset import RLBenchSelfgenDataset
from training.percep.depth_codec import MAX_VALID, MIN_VALID, decode_depth
from training.percep.normal_codec import decode_normal, depth_to_normal
from training.percep.seg_codec import (
    decode_known_color,
    decode_scene_roles,
    handles_to_mask,
    scene_role_labels,
    scene_role_iou,
)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task", choices=("depth", "seg", "normal"), required=True)
    # MUST match the checkpoint's training value. `referring` and `scene_roles` emit different
    # prompt tags and different pixels from the same episode, so evaluating a scene_roles arm
    # under `referring` measures a task it was never trained on and reports it as a bad score.
    p.add_argument("--segmentation_mode", choices=("referring", "scene_roles"), default="referring")
    p.add_argument("--legacy_scene_seg_tag", action="store_true",
                   help="emit the pre-2026-08-23 '<scene-seg protocol=...>' tag. Required for "
                        "checkpoints trained before the tag was shortened to '<scene-seg>' -- "
                        "asking a checkpoint with a tag it never saw measures nothing.")
    p.add_argument("--episode", default="open_drawer/variation0/episodes/episode0")
    # 512_aug: the bare "rlbench_selfgen" symlink points at the deleted v2 tree and dangles.
    p.add_argument("--data", default=str(REPO / "data" / "rlbench_selfgen_512_aug"))
    p.add_argument("--output", default=str(REPO / "comparisons_perception"))
    p.add_argument("--tag", default="arm1_step2500")
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--num_frames", type=int, default=41)
    p.add_argument("--frame_interval", type=int, default=1,
                   help="MUST match the checkpoint's training --frame_interval. 1 for "
                        "arm0/arm1, 3 for arm4__seed42_fi3. Mismatching it makes the "
                        "conditioning out of distribution.")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--conditioning",
        choices=("fifi", "iiii"),
        default="fifi",
        help="fifi gives both RGB segments fully; iiii gives only the first frame of every segment",
    )
    return p.parse_args()


def _pipeline_args(args):
    return SimpleNamespace(
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        ckpt_path=args.ckpt,
        height=args.res,
        width=args.res,
        use_usp=False,
        cfg_parallel=False,
        dynamic_cache_schedule=False,
        torch_compile=False,
    )


def _find_episode(ds, path):
    wanted = os.path.normpath(path)
    for i, episode in enumerate(ds.episodes):
        if os.path.normpath(episode["path"]) == wanted:
            return i
    raise RuntimeError(f"episode not found: {path}")


def _strip_prefix(text):
    # Explicit prompts are `<video><depth> instruction` or `<video><seg: ...> instruction`.
    return text.split("> ", 1)[1] if "> " in text else text


def _to_u8(stream):
    return np.clip(np.round((stream.permute(1, 2, 3, 0).numpy() + 1.0) * 127.5), 0, 255).astype(np.uint8)


def _save_contact_sheet(grid, path, indices=(0, 10, 20, 30, 40)):
    chosen = [grid[min(i, len(grid) - 1)] for i in indices]
    h, w = chosen[0].shape[:2]
    header = 28
    canvas = Image.new("RGB", (w * len(chosen), h + header), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    for col, (idx, frame) in enumerate(zip(indices, chosen)):
        canvas.paste(Image.fromarray(frame), (col * w, header))
        draw.text((col * w + 8, 4), f"frame {idx}", font=font, fill="white")
    canvas.save(path)


def _guard_seg_mode(args):
    """Refuse an eval whose seg protocol disagrees with the checkpoint's own directory name.

    train_arm.sh stamps `_sr` into the output path for `--segmentation_mode scene_roles`,
    precisely because the two protocols are different supervision signals sharing one modality
    name. `--segmentation_mode` here defaults to `referring` for back-compatibility, so
    evaluating a scene_roles checkpoint and forgetting the flag silently measures a task the
    model was never trained on and reports it as a bad score -- the same class of silent
    train/eval disagreement as the prompt-scrub trap (see training/templates.py:scene_seg_tag).

    Only fires on the seg task, and only when the path convention is unambiguous.
    """
    if args.task != "seg":
        return
    is_sr_ckpt = "_sr" in os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt)))) \
        or "_sr/" in os.path.abspath(args.ckpt)
    if is_sr_ckpt and args.segmentation_mode != "scene_roles":
        raise SystemExit(
            f"checkpoint path says scene_roles (`_sr`) but --segmentation_mode={args.segmentation_mode}.\n"
            f"  {args.ckpt}\n"
            f"  The two protocols emit different prompt tags AND different target pixels; this "
            f"eval would score the model on a task it never trained on.\n"
            f"  Pass --segmentation_mode scene_roles (or --segmentation_mode {args.segmentation_mode} "
            f"with a matching checkpoint)."
        )
    if not is_sr_ckpt and args.segmentation_mode == "scene_roles":
        raise SystemExit(
            f"--segmentation_mode scene_roles but the checkpoint path carries no `_sr` marker:\n"
            f"  {args.ckpt}\n"
            f"  If this really is a scene_roles run, rename its output dir to end in `_sr` so the "
            f"convention keeps working, or set ALLOW_SEG_MODE_MISMATCH=1."
        )


def main():
    args = _parse_args()
    if not os.environ.get("ALLOW_SEG_MODE_MISMATCH"):
        _guard_seg_mode(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    modality = {"depth": "depth", "seg": "segmentation", "normal": "normal"}[args.task]
    template = f"video+{modality}"
    ds = RLBenchSelfgenDataset(
        base_path=args.data,
        num_frames=args.num_frames,
        # Must match the checkpoint's TRAINING value, not the default. A window spans
        # num_frames * frame_interval native 20 Hz steps, so evaluating an arm trained at 3
        # with the default 1 conditions it on 2.0 s of motion where it learned 6.0 s -- an
        # out-of-distribution input whose failures cannot be attributed to the model.
        frame_interval=args.frame_interval,
        height=args.res,
        width=args.res,
        template_mix=f"{template}@1.0",
        prompt_tag_style="explicit",
        strict_getitem=True,
        variations="all",
        segmentation_mode=args.segmentation_mode,
        legacy_scene_seg_tag=args.legacy_scene_seg_tag,
    )
    episode_path = os.path.join(args.data, args.episode)
    idx = _find_episode(ds, episode_path)
    # Reset after dataset self-tests so the evaluated sample is stable across task modes.
    random.seed(args.seed)
    sample = ds.getitem(idx, force_template=template)
    if sample["template"] != template:
        raise RuntimeError(f"requested {template}, dataset returned {sample['template']}")

    print(
        f"[sample] prompt={sample['text']!r} views={sample['view_indices']} "
        f"frames={sample['frame_indices'][0]}..{sample['frame_indices'][-1]} path={sample['path']}"
    )
    streams_cpu = sample["streams"]
    T = args.num_frames
    view_dirs = [sample["view_dirs"][i] for i in sample["view_indices"]]
    frame_indices = sample["frame_indices"]

    raw_text = _strip_prefix(sample["text"])
    id_groups = color_map = role_lut = present_roles = None
    if args.task == "seg":
        if args.segmentation_mode == "scene_roles":
            role_lut, present_roles = ds._scene_role_lut(sample["path"], raw_text)
            if role_lut is None:
                raise RuntimeError(f"no scene_segments.json for {sample['path']}")
        else:
            id_groups, color_map = ds._referred_id_groups(sample["path"], raw_text)
            if not id_groups:
                raise RuntimeError(
                    f"no referring segmentation targets for {sample['path']} / {raw_text!r}")

    pipe = build_pipeline(_pipeline_args(args))
    device, dtype = pipe.device, torch.bfloat16
    model_streams = {k: v.unsqueeze(0).to(device=device, dtype=dtype) for k, v in streams_cpu.items()}
    t0 = time.time()
    fully_given_modalities = ["video"] if args.conditioning == "fifi" else []
    pred = pipe(
        prompt=[sample["text"]],
        negative_prompt="",
        template=template,
        streams=model_streams,
        fully_given_modalities=fully_given_modalities,
        camera=sample["camera"].unsqueeze(0).to(device=device, dtype=dtype),
        action_7d=sample["action_7d"].unsqueeze(0).to(device=device, dtype=dtype),
        extrinsics=sample["extrinsics"].unsqueeze(0).to(device=device, dtype=dtype),
        intrinsics=sample["intrinsics"].unsqueeze(0).to(device=device, dtype=dtype),
        height=args.res,
        width=args.res,
        num_frames=T,
        cfg_scale=args.cfg,
        num_inference_steps=args.steps,
        seed=args.seed,
        tiled=False,
        tile_size=(args.res // 16, args.res // 16),
        tile_stride=(args.res // 32, args.res // 32),
        enable_usp=False,
        cfg_parallel=False,
    )
    gen_seconds = time.time() - t0
    frames = np.stack([np.asarray(f) for f in pred])
    if len(frames) % 4:
        raise RuntimeError(f"expected four equal decoded segments, got {frames.shape}")
    q = len(frames) // 4
    pred_views = [frames[q : 2 * q], frames[3 * q : 4 * q]]
    gt_encoded = _to_u8(streams_cpu[modality])
    gt_encoded_views = [gt_encoded[:T], gt_encoded[T : 2 * T]]
    rgb = _to_u8(streams_cpu["video"])
    rgb_views = [rgb[:T], rgb[T : 2 * T]]

    report = {
        "tag": args.tag,
        "task": args.task,
        "template": template,
        "checkpoint": args.ckpt,
        "episode": sample["path"],
        "prompt": sample["text"],
        "view_indices": sample["view_indices"],
        "view_dirs": view_dirs,
        "frame_indices": frame_indices,
        "resolution": args.res,
        "num_frames": T,
        "frame_interval": args.frame_interval,
        "window_span_seconds": round((T - 1) * args.frame_interval / 20.0, 2),
        "steps": args.steps,
        "cfg": args.cfg,
        "seed": args.seed,
        "conditioning": args.conditioning,
        "segmentation_mode": args.segmentation_mode if args.task == "seg" else None,
        "generation_seconds": round(gen_seconds, 2),
        "views": [],
    }

    if args.task == "depth":
        for v, (pred_rgb, view_dir) in enumerate(zip(pred_views, view_dirs)):
            dec = decode_depth(pred_rgb.astype(np.uint8))
            gt = ds._to_model_res(ds._load_depth(view_dir)[frame_indices].astype(np.float32))
            gt_valid = np.isfinite(gt) & (gt > MIN_VALID) & (gt < MAX_VALID)
            valid = gt_valid & np.isfinite(dec)
            rel = np.abs(np.clip(dec, MIN_VALID, MAX_VALID) - gt) / np.maximum(gt, MIN_VALID)
            absrel = float(rel[valid].mean()) if valid.any() else None
            # Invalid generated pixels receive unit relative error; report valid_frac alongside it.
            penalized = np.where(valid, rel, 1.0)
            report["views"].append(
                {
                    "view": v,
                    "valid_frac": round(float(valid.sum() / max(gt_valid.sum(), 1)), 6),
                    "absrel_valid": None if absrel is None else round(absrel, 6),
                    "absrel_penalized": round(float(penalized[gt_valid].mean()), 6),
                }
            )
    elif args.task == "normal":
        # Angular agreement, not IoU: the normal field is continuous, and a per-pixel cosine is
        # the metric the literature (Argus Tab. 2 reports mean angular error on NYUv2) uses.
        # `flat_frac` is reported alongside because the table, floor and far-clip sky are a large
        # majority of the pixels and are trivially reconstructable -- a high mean cosine that is
        # entirely flat regions is not evidence the model learned geometry.
        for v, (pred_rgb, view_dir) in enumerate(zip(pred_views, view_dirs)):
            fx, fy = ds._focal_lengths(view_dir)
            raw = ds._load_depth(view_dir)[frame_indices].astype(np.float32)
            sx, sy = args.res / raw.shape[-1], args.res / raw.shape[-2]
            depth = ds._to_model_res(raw)
            gt_n = np.stack([depth_to_normal(depth[t], fx * sx, fy * sy) for t in range(len(depth))])
            pred_n = decode_normal(pred_rgb.astype(np.uint8))
            per_px = (gt_n * pred_n).sum(-1)
            dom = gt_n.reshape(-1, 3).mean(0)
            dom = dom / max(float(np.linalg.norm(dom)), 1e-6)
            flat = (1.0 - (gt_n * dom).sum(-1)) < 0.05
            report["views"].append(
                {
                    "view": v,
                    "mean_cos": round(float(per_px.mean()), 6),
                    "mean_angular_deg": round(float(np.degrees(np.arccos(
                        np.clip(per_px, -1, 1))).mean()), 4),
                    "mean_cos_flat": round(float(per_px[flat].mean()), 6) if flat.any() else None,
                    "mean_cos_detailed": round(float(per_px[~flat].mean()), 6) if (~flat).any() else None,
                    "flat_frac": round(float(flat.mean()), 6),
                }
            )
    elif args.segmentation_mode == "scene_roles":
        for v, (pred_rgb, view_dir) in enumerate(zip(pred_views, view_dirs)):
            handle_map = ds._to_model_res(ds._load_mask(view_dir)[frame_indices]).astype(np.uint16)
            gt_labels = scene_role_labels(handle_map, role_lut)
            # present_roles is REQUIRED, not an optimisation: decoding against all nine colours
            # lets a role this episode cannot contain steal ambiguous pixels (decode_scene_roles'
            # docstring). Which roles an episode has is known at eval time exactly as at train.
            decoded = decode_scene_roles(pred_rgb.astype(np.uint8), present_roles=present_roles)
            per_role = scene_role_iou(decoded, gt_labels, roles=present_roles)
            # Macro mIoU over roles that are actually PRESENT in the GT. scene_role_iou scores an
            # absent-and-unpredicted role 1.0 (vacuously correct); averaging those in would let a
            # task with few roles outscore a crowded one for doing less.
            scored = [m["iou"] for m in per_role.values() if m["gt_px"] > 0]
            report["views"].append(
                {
                    "view": v,
                    "roles": per_role,
                    "macro_miou": round(float(np.mean(scored)), 6) if scored else None,
                    "n_roles_scored": len(scored),
                    "unknown_pred_px": int(decode_scene_roles(
                        pred_rgb.astype(np.uint8))["unknown"].sum()),
                }
            )
    else:
        for v, (pred_rgb, view_dir) in enumerate(zip(pred_views, view_dirs)):
            decoded = decode_known_color(pred_rgb.astype(np.uint8), color_map)
            handle_map = ds._to_model_res(ds._load_mask(view_dir)[frame_indices]).astype(np.uint16)
            per_instance = {}
            for name, handles in id_groups.items():
                gt = handles_to_mask(handle_map, handles)
                pd = decoded[name]
                inter = int((pd & gt).sum())
                union = int((pd | gt).sum())
                per_instance[name] = {
                    "iou": round(inter / union if union else (1.0 if not pd.any() else 0.0), 6),
                    "gt_px": int(gt.sum()),
                    "pred_px": int(pd.sum()),
                }
            report["views"].append({"view": v, "instances": per_instance})

    suffix = "" if args.conditioning == "fifi" else f"_{args.conditioning}"
    out_dir = Path(args.output) / f"{args.tag}_{args.task}_{args.res}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [np.concatenate([rgb_views[v], gt_encoded_views[v], pred_views[v]], axis=2) for v in range(2)]
    grid = np.concatenate(rows, axis=1)
    save_video([Image.fromarray(f) for f in grid], str(out_dir / "rgb_gt_prediction.mp4"), fps=8, quality=8)
    _save_contact_sheet(grid, out_dir / "rgb_gt_prediction_contact_sheet.png")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print("PERCEPTION_EVAL_OK", json.dumps(report))


if __name__ == "__main__":
    main()
