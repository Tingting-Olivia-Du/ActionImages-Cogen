"""Evaluate one checkpoint on every training-defined task/mask protocol.

The model is loaded once.  Valid cells are:
  action: IIII, FIII, FIFI, policy (single-frame visual segments)
  depth:  IIII, FIII, FIFI
  seg:    IIII, FIII, FIFI
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
from training.percep.seg_codec import decode_known_color, handles_to_mask
from training.utils import (
    fuse_multiview_heatmaps_to_7d_point_torch,
    project_action_5d_to_rgb_torch,
    project_actions_7d_to_5d_torch_batch,
)

import training as _training_package
if REPO not in Path(_training_package.__file__).resolve().parents:
    raise RuntimeError(
        f"wrong training package: {_training_package.__file__}; expected it below {REPO}. "
        f"Run with PYTHONPATH={REPO}."
    )


TASK_MOD = {"action": "action", "depth": "depth", "seg": "segmentation"}
TASK_MODES = {
    "action": ("iiii", "fiii", "fifi", "policy"),
    "depth": ("iiii", "fiii", "fifi"),
    "seg": ("iiii", "fiii", "fifi"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--res", type=int, required=True)
    p.add_argument("--output", default=str(REPO / "comparisons_all_masks"))
    p.add_argument("--episode", default="open_drawer/variation0/episodes/episode0")
    p.add_argument("--data", default=str(REPO / "data" / "rlbench_selfgen_512_aug"),
                   help="512_aug; the bare rlbench_selfgen symlink points at the deleted v2 tree")
    p.add_argument("--num_frames", type=int, default=41)
    p.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help="MUST match the checkpoint's training frame interval (3 for arm4__seed42_fi3)",
    )
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt-tag-style", choices=("explicit", "none"), default="explicit")
    p.add_argument("--tasks", nargs="+", choices=tuple(TASK_MOD), default=list(TASK_MOD))
    p.add_argument("--modes", nargs="+", choices=("iiii", "fiii", "fifi", "f0f0", "policy"), default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def find_episode(ds, path):
    wanted = os.path.normpath(path)
    for i, ep in enumerate(ds.episodes):
        if os.path.normpath(ep["path"]) == wanted:
            return i
    raise RuntimeError(f"episode not found: {path}")


def load_sample(args, task):
    modality = TASK_MOD[task]
    template = f"video+{modality}"
    ds = RLBenchSelfgenDataset(
        base_path=args.data,
        num_frames=args.num_frames,
        frame_interval=args.frame_interval,
        height=args.res,
        width=args.res,
        template_mix=f"{template}@1.0",
        prompt_tag_style=args.prompt_tag_style,
        strict_getitem=True,
        variations="all",
    )
    idx = find_episode(ds, os.path.join(args.data, args.episode))
    random.seed(args.seed)
    sample = ds.getitem(idx, force_template=template)
    if sample["template"] != template:
        raise RuntimeError(f"requested {template}, got {sample['template']}")
    return ds, sample, template


def to_u8(stream):
    x = stream.permute(1, 2, 3, 0).float().cpu().numpy()
    return np.clip(np.round((x + 1) * 127.5), 0, 255).astype(np.uint8)


def segment_predictions(frames, mode, T):
    lengths = [1, T, 1, T] if mode == "policy" else [T, T, T, T]
    if sum(lengths) != len(frames):
        raise RuntimeError(f"decoded {len(frames)} frames, expected segments {lengths}")
    out, off = [], 0
    for n in lengths:
        out.append(frames[off : off + n])
        off += n
    return out


def save_sheet(grid, path, indices=(0, 10, 20, 30, 40)):
    chosen = [(i, grid[min(i, len(grid) - 1)]) for i in indices]
    h, w = chosen[0][1].shape[:2]
    canvas = Image.new("RGB", (w * len(chosen), h + 28), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    for col, (idx, frame) in enumerate(chosen):
        canvas.paste(Image.fromarray(frame), (col * w, 28))
        draw.text((col * w + 8, 4), f"frame {idx}", font=font, fill="white")
    canvas.save(path)


def label_frames(frames, label):
    """Add a permanent header without covering generated pixels."""
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    labeled = []
    for frame in frames:
        # 32 keeps both 256+32 and 512+32 divisible by common video macroblocks.
        image = Image.new("RGB", (frame.shape[1], frame.shape[0] + 32), (18, 18, 18))
        image.paste(Image.fromarray(frame), (0, 32))
        ImageDraw.Draw(image).text((7, 7), label, font=font, fill="white")
        labeled.append(np.asarray(image))
    return np.stack(labeled)


def rgb_segment_label(mode, view):
    if mode == "iiii":
        return f"S{view * 2} RGB V{view}: PRED-I"
    if mode == "fiii" and view == 0:
        return "S0 RGB V0: COND-F"
    if mode == "fiii":
        return "S2 RGB V1: PRED-I"
    if mode == "fifi":
        return f"S{view * 2} RGB V{view}: COND-F"
    if mode == "f0f0":
        return f"S{view * 2} RGB V{view}: COND-F"
    return f"S{view * 2} RGB V{view}: COND-1F"


def save_all_segment_visuals(out, task, mode, rgb_views, gt_views, segs, T):
    """Save exact raw segments plus individually and jointly labelled videos."""
    target = task.upper()
    segment_dir = out / "segments"
    segment_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        out / "generated_segments.npz",
        segment0_rgb_view0=segs[0],
        segment1_target_view0=segs[1],
        segment2_rgb_view1=segs[2],
        segment3_target_view1=segs[3],
    )
    manifest = {"mode": mode, "legend": {"I": "first latent given; remaining latents generated", "F": "full segment given", "0": "no target latent given; whole segment generated", "1F": "single-frame policy segment"}, "segments": []}

    pred_rgb_labeled = []
    pred_target_labeled = []
    for view in range(2):
        rgb_label = rgb_segment_label(mode, view)
        target_label = f"S{view * 2 + 1} {target} V{view}: PRED-{'0' if mode == 'f0f0' else 'I'}"
        manifest["segments"].extend([
            {"index": view * 2, "view": view, "modality": "rgb", "label": rgb_label,
             "actual_frames": len(segs[view * 2]), "status": "condition" if "COND" in rgb_label else "prediction"},
            {"index": view * 2 + 1, "view": view, "modality": task, "label": target_label,
             "actual_frames": len(segs[view * 2 + 1]), "status": "prediction"},
        ])
        gt_rgb = label_frames(rgb_views[view], f"GT RGB V{view} (reference)")
        gt_target = label_frames(gt_views[view], f"GT {target} V{view} (reference)")
        pred_rgb = label_frames(segs[view * 2], rgb_label)
        pred_target = label_frames(segs[view * 2 + 1], target_label)
        pred_rgb_labeled.append(pred_rgb)
        pred_target_labeled.append(pred_target)

        save_video([Image.fromarray(x) for x in gt_rgb], str(segment_dir / f"gt_rgb_view{view}.mp4"), fps=8, quality=8)
        save_video(
            [Image.fromarray(x) for x in pred_rgb],
            str(segment_dir / f"segment{view * 2}_{'cond' if 'COND' in rgb_label else 'pred'}_rgb_view{view}.mp4"),
            fps=8,
            quality=8,
        )
        save_video([Image.fromarray(x) for x in gt_target], str(segment_dir / f"gt_{task}_view{view}.mp4"), fps=8, quality=8)
        save_video(
            [Image.fromarray(x) for x in pred_target],
            str(segment_dir / f"segment{view * 2 + 1}_pred_{task}_view{view}.mp4"),
            fps=8,
            quality=8,
        )

    with open(out / "segment_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Policy RGB really has one frame.  Repeat only in the combined viewer so columns align;
    # the independent segment MP4 and NPZ above retain its exact one-frame length.
    viewer_rgb = [np.repeat(x, T, axis=0) if len(x) == 1 else x for x in pred_rgb_labeled]
    rows = []
    for view in range(2):
        rows.append(
            np.concatenate(
                [
                    label_frames(rgb_views[view], f"GT RGB V{view}"),
                    viewer_rgb[view],
                    label_frames(gt_views[view], f"GT {target} V{view}"),
                    pred_target_labeled[view],
                ],
                axis=2,
            )
        )
    return np.concatenate(rows, axis=1)


def action_ground_truth(sample, res, device):
    action = sample["action_7d"].unsqueeze(0).to(device=device, dtype=torch.float32)
    V, T = 2, action.shape[1]
    rep = action.repeat(1, V, 1)
    extr = sample["extrinsics"].unsqueeze(0).to(device=device, dtype=torch.float32)
    intr = sample["intrinsics"].unsqueeze(0).to(device=device, dtype=torch.float32)
    a5 = project_actions_7d_to_5d_torch_batch(rep, extr, intr)
    rgb = project_action_5d_to_rgb_torch(a5, res, res)[0]
    return (rgb.reshape(V, T, res, res, 3) * 255).round().byte().cpu().numpy()


def gt_tool_x_axis(action):
    a = action[:, 3:6].float()
    cx, cy, cz = torch.cos(a[:, 0]), torch.cos(a[:, 1]), torch.cos(a[:, 2])
    sx, sy, sz = torch.sin(a[:, 0]), torch.sin(a[:, 1]), torch.sin(a[:, 2])
    # Rz @ Ry @ Rx @ xhat; Rx leaves xhat unchanged.
    return torch.stack([cz * cy, sz * cy, -sy], dim=-1)


def action_metrics(pred_views, gt_views, sample, device):
    pred = np.stack(pred_views, axis=1)  # T,V,H,W,3
    gt = np.stack(gt_views, axis=1)
    r_peak = pred[..., 0].reshape(pred.shape[0], pred.shape[1], -1).max(-1).min(-1)
    strong = r_peak >= 100
    extr = sample["extrinsics"].reshape(2, -1, 4, 4).permute(1, 0, 2, 3)[..., :3, :]
    intr = sample["intrinsics"].reshape(2, -1, 3, 3).permute(1, 0, 2, 3)
    decoded = fuse_multiview_heatmaps_to_7d_point_torch(
        torch.from_numpy(pred).to(device=device, dtype=torch.float32),
        extr.to(device=device, dtype=torch.float32),
        intr.to(device=device, dtype=torch.float32),
        near=0.6,
        far=1.8,
        num_depth_samples=512,
        apply_edge_smoothing=False,
    ).cpu()
    target = sample["action_7d"].float().cpu()
    pos_err = torch.linalg.vector_norm(decoded[:, :3] - target[:, :3], dim=-1).numpy()
    axis_gt = gt_tool_x_axis(target)
    cosine = (decoded[:, 3:6] * axis_gt).sum(-1).clamp(-1, 1)
    axis_err = torch.rad2deg(torch.acos(cosine)).numpy()
    pred_open = np.median(pred[..., 2], axis=(1, 2, 3)) > 32
    gt_open = target[:, 6].numpy() > 0.5
    return {
        "r_peak_mean": float(r_peak.mean()),
        "r_peak_weak_frac": float((~strong).mean()),
        "heatmap_mse": float(np.mean(((pred.astype(np.float32) - gt) / 255.0) ** 2)),
        "pos_err_mean_m": float(pos_err.mean()),
        "pos_err_strong_median_m": float(np.median(pos_err[strong])) if strong.any() else None,
        "axis_err_strong_median_deg": float(np.median(axis_err[strong])) if strong.any() else None,
        "gripper_acc": float((pred_open == gt_open).mean()),
        "strong_frames": int(strong.sum()),
    }


def perception_metrics(task, pred_views, ds, sample):
    view_dirs = [sample["view_dirs"][i] for i in sample["view_indices"]]
    ids = colors = None
    if task == "seg":
        raw = sample["text"].split("> ", 1)[1] if "> " in sample["text"] else sample["text"]
        ids, colors = ds._referred_id_groups(sample["path"], raw)
    views = []
    for pred, view_dir in zip(pred_views, view_dirs):
        if task == "depth":
            dec = decode_depth(pred)
            gt = ds._to_model_res(ds._load_depth(view_dir)[sample["frame_indices"]].astype(np.float32))
            gt_valid = np.isfinite(gt) & (gt > MIN_VALID) & (gt < MAX_VALID)
            valid = gt_valid & np.isfinite(dec)
            rel = np.abs(np.clip(dec, MIN_VALID, MAX_VALID) - gt) / np.maximum(gt, MIN_VALID)
            views.append({
                "valid_frac": float(valid.sum() / max(gt_valid.sum(), 1)),
                "absrel_valid": float(rel[valid].mean()) if valid.any() else None,
                "absrel_penalized": float(np.where(valid, rel, 1.0)[gt_valid].mean()),
            })
        else:
            decoded = decode_known_color(pred, colors)
            handles = ds._to_model_res(ds._load_mask(view_dir)[sample["frame_indices"]]).astype(np.uint16)
            inst = {}
            for name, hs in ids.items():
                gt = handles_to_mask(handles, hs)
                pd = decoded[name]
                union = int((gt | pd).sum())
                inst[name] = {
                    "iou": float((gt & pd).sum() / union) if union else float(not pd.any()),
                    "gt_px": int(gt.sum()), "pred_px": int(pd.sum()),
                }
            views.append({"instances": inst})
    return {"views": views}


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    samples = {task: load_sample(args, task) for task in args.tasks}
    pipe = build_pipeline(SimpleNamespace(
        model_id="Wan-AI/Wan2.2-TI2V-5B", ckpt_path=args.ckpt,
        height=args.res, width=args.res, use_usp=False, cfg_parallel=False,
        dynamic_cache_schedule=False, torch_compile=False,
    ))
    device, dtype = pipe.device, torch.bfloat16
    root = Path(args.output) / args.tag
    root.mkdir(parents=True, exist_ok=True)
    summary = []

    for task in args.tasks:
        ds, sample, template = samples[task]
        streams = {k: v.unsqueeze(0).to(device=device, dtype=dtype) for k, v in sample["streams"].items()}
        rgb = to_u8(sample["streams"]["video"])
        rgb_views = [rgb[:args.num_frames], rgb[args.num_frames:]]
        if task == "action":
            gt_views = list(action_ground_truth(sample, args.res, device))
        else:
            gt = to_u8(sample["streams"][TASK_MOD[task]])
            gt_views = [gt[:args.num_frames], gt[args.num_frames:]]

        modes = list(TASK_MODES[task]) if args.modes is None else [
            m for m in args.modes if not (m == "policy" and task != "action")
        ]
        for mode in modes:
            out = root / task / mode
            out.mkdir(parents=True, exist_ok=True)
            metrics_path = out / "metrics.json"
            if args.resume and metrics_path.exists():
                with open(metrics_path) as f:
                    report = json.load(f)
                summary.append(report)
                print("EVAL_CELL_CACHED", args.tag, task, mode, flush=True)
                continue
            t0 = time.time()
            pred_flat = pipe(
                prompt=[sample["text"]], negative_prompt="", template=template, streams=streams,
                fully_given_modalities=[], conditioning_mode=mode,
                camera=sample["camera"].unsqueeze(0).to(device=device, dtype=dtype),
                action_7d=sample["action_7d"].unsqueeze(0).to(device=device, dtype=dtype),
                extrinsics=sample["extrinsics"].unsqueeze(0).to(device=device, dtype=dtype),
                intrinsics=sample["intrinsics"].unsqueeze(0).to(device=device, dtype=dtype),
                height=args.res, width=args.res, num_frames=args.num_frames,
                cfg_scale=args.cfg, num_inference_steps=args.steps, seed=args.seed,
                tiled=False, tile_size=(args.res // 16, args.res // 16),
                tile_stride=(args.res // 32, args.res // 32), enable_usp=False, cfg_parallel=False,
            )
            frames = np.stack([np.asarray(x) for x in pred_flat])
            segs = segment_predictions(frames, mode, args.num_frames)
            pred_views = [segs[1], segs[3]]
            report = {
                "tag": args.tag, "checkpoint": args.ckpt, "task": task, "template": template,
                "mode": mode, "resolution": args.res, "episode": sample["path"],
                "prompt": sample["text"], "view_indices": sample["view_indices"],
                "prompt_tag_style": args.prompt_tag_style,
                "frame_interval": args.frame_interval,
                "window_span_seconds": round(
                    (args.num_frames - 1) * args.frame_interval / 20.0, 2
                ),
                "frame_indices": sample["frame_indices"], "steps": args.steps,
                "cfg": args.cfg, "seed": args.seed, "generation_seconds": time.time() - t0,
                "training_support": not (
                    args.tag == "official_step125750" and task in {"depth", "seg"}
                    or args.tag.startswith("arm0_") and task in {"depth", "seg"}
                ),
            }
            if task == "action":
                report["metrics"] = action_metrics(pred_views, gt_views, sample, device)
            else:
                report["metrics"] = perception_metrics(task, pred_views, ds, sample)
            grid = save_all_segment_visuals(out, task, mode, rgb_views, gt_views, segs, args.num_frames)
            save_video([Image.fromarray(x) for x in grid], str(out / "all_segments_labeled.mp4"), fps=8, quality=8)
            save_sheet(grid, out / "rgb_gt_prediction_contact_sheet.png")
            with open(metrics_path, "w") as f:
                json.dump(report, f, indent=2)
            summary.append(report)
            with open(root / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            print("EVAL_CELL_OK", args.tag, task, mode, json.dumps(report["metrics"]), flush=True)

    print("EVAL_CHECKPOINT_OK", args.tag, len(summary), flush=True)


if __name__ == "__main__":
    main()
