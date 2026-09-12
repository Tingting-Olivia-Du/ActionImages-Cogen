"""One checkpoint, one episode, one instruction -- four modalities, selected only by prompt tag.

    python scripts/modality_demo.py --ckpt outputs/arm6__seed42_fi3_512_aug_sr/checkpoint-10000/step10000.ckpt \
        --tag arm6_step10000 --modalities action,depth,segmentation,normal

This is the deliverable of the co-generation direction, so it is deliberately the narrowest
possible demonstration of it: the episode, the window, the views, the camera tensors, the seed and
the instruction text are all held fixed, and the ONLY thing that varies between the panels is the
prompt prefix (`<video><action>` / `<video><depth>` / `<video><scene-seg ...>` / `<video><normal>`).
If the panels differ, the difference is attributable to the tag and nothing else.

Runs under conditioning_mode="iiii" -- every segment keeps only its first latent frame. That is
the regime a closed-loop rollout uses and the one PERCEPTION_MASK_MIX=M2 trains 90% of perception
samples in, so the picture shows what the model does at deployment rather than a discriminative
RGB->X mapping it is never asked for. Pass --conditioning fifi for the (easier) upper bound.

Writes reports/modality_demo/<tag>/: one PNG contact sheet, one side-by-side MP4, per-modality
MP4s, and metrics.json.
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

from diffsynth import save_video  # noqa: E402
from inference import build_pipeline  # noqa: E402
from training.dataset import RLBenchSelfgenDataset  # noqa: E402
from training.percep.depth_codec import MAX_VALID, MIN_VALID, decode_depth  # noqa: E402
from training.percep.normal_codec import decode_normal, depth_to_normal  # noqa: E402
from training.percep.seg_codec import (  # noqa: E402
    decode_scene_roles,
    scene_role_iou,
    scene_role_labels,
)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--modalities", default="action,depth,segmentation,normal",
                   help="comma list; each becomes one `video+<m>` panel")
    p.add_argument("--episode", default="open_drawer/variation0/episodes/episode0")
    p.add_argument("--data", default=str(REPO / "data" / "rlbench_selfgen_512_aug"))
    p.add_argument("--output", default=str(REPO / "reports" / "modality_demo"))
    p.add_argument("--segmentation_mode", choices=("referring", "scene_roles"), default="scene_roles")
    p.add_argument("--legacy_scene_seg_tag", action="store_true",
                   help="pre-2026-08-23 '<scene-seg protocol=...>' tag; needed for checkpoints "
                        "trained before the tag was shortened")
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--num_frames", type=int, default=41)
    p.add_argument("--frame_interval", type=int, default=3,
                   help="MUST match the checkpoint's training value or the conditioning is OOD")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    # iiii  -- every segment keeps its first latent frame. That frame is the target modality
    #          itself, so the model can identify the task from the ANCHOR and never read the
    #          prompt. Good for "can it continue this modality", useless for "does the tag work".
    # f0f0  -- video segments fully given, perception segments get NO anchor at all. The tag is
    #          then the ONLY thing that says which modality to produce. This is the mode that
    #          actually tests prompt-controlled generation.
    # fifi  -- video fully given, perception keeps its first frame. The easy upper bound.
    p.add_argument("--conditioning", choices=("iiii", "f0f0", "fifi"), default="iiii")
    p.add_argument("--frames_shown", default="0,10,20,30,40")
    return p.parse_args()


def _pipeline_args(a):
    return SimpleNamespace(model_id="Wan-AI/Wan2.2-TI2V-5B", ckpt_path=a.ckpt, height=a.res,
                           width=a.res, use_usp=False, cfg_parallel=False,
                           dynamic_cache_schedule=False, torch_compile=False)


def _u8(stream):
    return np.clip(np.round((stream.permute(1, 2, 3, 0).numpy() + 1.0) * 127.5), 0, 255).astype(np.uint8)


def _find_episode(ds, path):
    wanted = os.path.normpath(path)
    for i, ep in enumerate(ds.episodes):
        if os.path.normpath(ep["path"]) == wanted:
            return i
    raise RuntimeError(f"episode not found: {path}")


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate(pipe, ds, idx, modality, args):
    """-> (prompt, [pred view0, pred view1], [gt view0, gt view1], [rgb view0, rgb view1], sample)."""
    template = f"video+{modality}"
    # Re-seed before EVERY draw: getitem picks the window, the views and one of three instruction
    # paraphrases at random. Without this the panels would differ by episode content as well as by
    # tag, and the comparison this script exists to make would be meaningless.
    _seed_all(args.seed)
    sample = ds.getitem(idx, force_template=template)
    if sample["template"] != template:
        raise RuntimeError(f"requested {template}, dataset returned {sample['template']!r} "
                           f"(the modality was dropped -- missing GT for this episode?)")
    T = args.num_frames
    device = pipe.device
    dtype = torch.bfloat16
    streams = {k: v.unsqueeze(0).to(device=device, dtype=dtype) for k, v in sample["streams"].items()}
    _seed_all(args.seed)
    # `fully_given_modalities` and `conditioning_mode` are mutually exclusive (the pipeline
    # raises if both are set), and only the latter can express f0f0's anchor-free segments.
    cond_kw = ({"conditioning_mode": args.conditioning} if args.conditioning == "f0f0"
               else {"fully_given_modalities": ["video"] if args.conditioning == "fifi" else []})
    pred = pipe(
        prompt=[sample["text"]], negative_prompt="", template=template, streams=streams,
        **cond_kw,
        camera=sample["camera"].unsqueeze(0).to(device=device, dtype=dtype),
        action_7d=sample["action_7d"].unsqueeze(0).to(device=device, dtype=dtype),
        extrinsics=sample["extrinsics"].unsqueeze(0).to(device=device, dtype=dtype),
        intrinsics=sample["intrinsics"].unsqueeze(0).to(device=device, dtype=dtype),
        height=args.res, width=args.res, num_frames=T, cfg_scale=args.cfg,
        num_inference_steps=args.steps, seed=args.seed, tiled=False,
        tile_size=(args.res // 16, args.res // 16), tile_stride=(args.res // 32, args.res // 32),
        enable_usp=False, cfg_parallel=False,
    )
    frames = np.stack([np.asarray(f) for f in pred])
    if len(frames) % 4:
        raise RuntimeError(f"expected four equal decoded segments, got {frames.shape}")
    q = len(frames) // 4
    # Layout is [V0, P0, V1, P1] (pinned by scripts/debug_cogen.py [3b]). Segments 1 and 3 are
    # the perception/action stream; segments 0 and 2 are the model's own RGB prediction, which
    # under iiii conditioning is GENERATED from the first frame alone, not copied. Both matter:
    # the perception panel says whether the prompt selected the right modality, the RGB panel
    # says whether the world model underneath is still intact.
    pred_views = [frames[q:2 * q], frames[3 * q:4 * q]]
    pred_rgb_views = [frames[0:q], frames[2 * q:3 * q]]
    rgb = _u8(sample["streams"]["video"])
    rgb_views = [rgb[:T], rgb[T:2 * T]]
    if modality == "action":
        # The action stream has no GT on disk: its pixels are rendered inside forward() by
        # projecting action_7d through each view's camera (train.py). Nothing to compare against
        # here, so the panel is qualitative -- which is what this figure is for.
        gt_views = [None, None]
    else:
        gt = _u8(sample["streams"][modality])
        gt_views = [gt[:T], gt[T:2 * T]]
    return sample, pred_views, gt_views, rgb_views, pred_rgb_views


def score(modality, ds, sample, pred_views, args):
    """A cheap per-modality number, so the sheet says how good each panel is, not just how it looks."""
    view_dirs = [sample["view_dirs"][i] for i in sample["view_indices"]]
    fi = sample["frame_indices"]
    out = []
    for v, (pred, vd) in enumerate(zip(pred_views, view_dirs)):
        p8 = pred.astype(np.uint8)
        if modality == "depth":
            dec = decode_depth(p8)
            gt = ds._to_model_res(ds._load_depth(vd)[fi].astype(np.float32))
            ok = np.isfinite(gt) & (gt > MIN_VALID) & (gt < MAX_VALID) & np.isfinite(dec)
            rel = np.abs(np.clip(dec, MIN_VALID, MAX_VALID) - gt) / np.maximum(gt, MIN_VALID)
            out.append({"view": v, "metric": "absrel",
                        "value": round(float(rel[ok].mean()), 5) if ok.any() else None,
                        "valid_frac": round(float(ok.mean()), 5)})
        elif modality == "normal":
            fx, fy = ds._focal_lengths(vd)
            raw = ds._load_depth(vd)[fi].astype(np.float32)
            sx, sy = args.res / raw.shape[-1], args.res / raw.shape[-2]
            d = ds._to_model_res(raw)
            gt_n = np.stack([depth_to_normal(d[t], fx * sx, fy * sy) for t in range(len(d))])
            cos = float((gt_n * decode_normal(p8)).sum(-1).mean())
            out.append({"view": v, "metric": "mean_cos", "value": round(cos, 5),
                        "mean_angular_deg": round(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))), 3)})
        elif modality == "segmentation" and args.segmentation_mode == "scene_roles":
            raw_text = sample["text"].split("> ", 1)[-1]
            lut, present = ds._scene_role_lut(sample["path"], raw_text)
            hm = ds._to_model_res(ds._load_mask(vd)[fi]).astype(np.uint16)
            per_role = scene_role_iou(decode_scene_roles(p8, present_roles=present),
                                      scene_role_labels(hm, lut), roles=present)
            scored = [m["iou"] for m in per_role.values() if m["gt_px"] > 0]
            out.append({"view": v, "metric": "macro_miou",
                        "value": round(float(np.mean(scored)), 5) if scored else None,
                        "roles": per_role})
        else:
            out.append({"view": v, "metric": None, "value": None})
    return out


def contact_sheet(panels, rgb_views, indices, path, header_text, pred_rgbs=None):
    """Rows = [RGB] + one per modality (pred over GT); columns = frames. View 0 only."""
    h, w = rgb_views[0][0].shape[:2]
    scale = max(1, w // 256)
    cell = w // scale
    rows = [("RGB ground truth", [rgb_views[0][i] for i in indices], None)]
    for name, pred, gt, _ in panels:
        if pred_rgbs and name in pred_rgbs:
            rows.append((f"<{name}> RGB pred", [pred_rgbs[name][0][i] for i in indices], None))
        rows.append((f"<{name}> pred", [pred[0][i] for i in indices], None))
        if gt[0] is not None:
            rows.append((f"<{name}> GT", [gt[0][i] for i in indices], None))
    lbl, top = 210, 100
    canvas = Image.new("RGB", (lbl + cell * len(indices), top + cell * len(rows)), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    f_big = ImageFont.truetype(FONT, 22)
    f_sm = ImageFont.truetype(FONT, 15)
    for i, line in enumerate(header_text.split("\n")[:3]):
        draw.text((10, 8 + 21 * i), line, font=f_big if i == 0 else f_sm, fill="white")
    for c, idx in enumerate(indices):
        draw.text((lbl + c * cell + 6, top - 22), f"t={idx}", font=f_sm, fill=(170, 170, 175))
    for r, (name, frames, _) in enumerate(rows):
        draw.text((10, top + r * cell + cell // 2 - 9), name, font=f_sm,
                  fill=(120, 220, 140) if "pred" in name else (170, 170, 175))
        for c, fr in enumerate(frames):
            canvas.paste(Image.fromarray(fr).resize((cell, cell), Image.NEAREST),
                         (lbl + c * cell, top + r * cell))
    canvas.save(path)
    return path


def main():
    args = _parse_args()
    mods = [m.strip() for m in args.modalities.split(",") if m.strip()]
    indices = [int(i) for i in args.frames_shown.split(",")]

    _seed_all(args.seed)
    ds = RLBenchSelfgenDataset(
        base_path=args.data, num_frames=args.num_frames, frame_interval=args.frame_interval,
        height=args.res, width=args.res,
        template_mix=",".join(f"video+{m}@{1.0 / len(mods):.4f}" for m in mods),
        prompt_tag_style="explicit", strict_getitem=True, variations="all",
        segmentation_mode=args.segmentation_mode,
        legacy_scene_seg_tag=args.legacy_scene_seg_tag,
    )
    idx = _find_episode(ds, os.path.join(args.data, args.episode))
    pipe = build_pipeline(_pipeline_args(args))

    out_dir = Path(args.output) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    panels, metrics, prompts = [], {}, {}
    pred_rgbs, rgb_psnr = {}, {}
    rgb_views = None
    for m in mods:
        t0 = time.time()
        sample, pred, gt, rgb_views, pred_rgb = generate(pipe, ds, idx, m, args)
        panels.append((m, pred, gt, sample))
        pred_rgbs[m] = pred_rgb
        save_video([Image.fromarray(f) for f in pred_rgb[0]],
                   str(out_dir / f"{m}__rgb-pred_view0.mp4"), fps=8, quality=7)
        # RGB fidelity of the generated video segment, per modality. If this collapses only for
        # one modality, the auxiliary stream is interfering with the world model rather than
        # merely being hard to produce.
        gtv = rgb_views[0].astype(np.float64)
        mse = float(((pred_rgb[0].astype(np.float64) - gtv) ** 2).mean())
        rgb_psnr[m] = round(10.0 * np.log10(255.0 ** 2 / max(mse, 1e-9)), 3)
        prompts[m] = sample["text"]
        metrics[m] = score(m, ds, sample, pred, args)
        save_video([Image.fromarray(f) for f in pred[0]], str(out_dir / f"{m}_view0.mp4"), fps=8, quality=7)
        print(f"  {m:14s} {time.time() - t0:6.1f}s  prompt={sample['text'][:80]!r}")

    ep = panels[0][3]
    header = (f"{args.tag}  --  one checkpoint, one episode, one instruction; only the prompt tag differs\n"
              f"{args.episode}   conditioning={args.conditioning}   res={args.res}  "
              f"frame_interval={args.frame_interval}  seed={args.seed}\n"
              f'"{ep["text"].split("> ", 1)[-1]}"')
    sheet = contact_sheet(panels, rgb_views, indices, str(out_dir / "contact_sheet.png"),
                          header, pred_rgbs=pred_rgbs)

    # Every panel is drawn from the SAME window and views; assert it rather than trust it, because
    # a silent re-draw is exactly the failure that would make this figure a lie.
    ref = (panels[0][3]["path"], list(panels[0][3]["view_indices"]), list(panels[0][3]["frame_indices"]))
    for m, _, _, s in panels[1:]:
        got = (s["path"], list(s["view_indices"]), list(s["frame_indices"]))
        if got != ref:
            raise RuntimeError(f"panel {m} drew a DIFFERENT sample than {mods[0]}: {got} != {ref}")

    # Lossless dump of everything the contact sheet draws. The per-modality MP4s are h264 at
    # quality=7, which is fine for looking at but shows ringing on the seg palette's hard colour
    # boundaries and on the action blobs -- neither belongs in a paper figure. Figure builders
    # should read this, not the videos.
    arrays = {"rgb_gt": rgb_views[0]}
    for m, pred, gt, _ in panels:
        arrays[f"{m}__pred"] = pred[0]
        arrays[f"{m}__rgb_pred"] = pred_rgbs[m][0]
        if gt[0] is not None:
            arrays[f"{m}__gt"] = gt[0]
    np.savez_compressed(out_dir / "frames_view0.npz", **arrays)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"tag": args.tag, "ckpt": args.ckpt, "episode": ep["path"],
                   "instruction": ep["text"].split("> ", 1)[-1],
                   "conditioning": args.conditioning, "segmentation_mode": args.segmentation_mode,
                   "res": args.res, "frame_interval": args.frame_interval, "seed": args.seed,
                   "view_indices": ep["view_indices"], "frame_indices": ep["frame_indices"],
                   "prompts": prompts, "metrics": metrics,
                   "generated_rgb_psnr_db": rgb_psnr}, f, indent=1)
    print(f"\nwrote {sheet}")
    print(f"wrote {out_dir / 'metrics.json'}")
    for m in mods:
        vals = [v.get("value") for v in metrics[m]]
        print(f"  {m:14s} {metrics[m][0]['metric']}: {vals}   generated-RGB PSNR {rgb_psnr.get(m)} dB")


if __name__ == "__main__":
    main()
