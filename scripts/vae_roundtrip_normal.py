"""VETO GATE for the surface-normal stream: how much of it survives a VAE roundtrip?

    python scripts/vae_roundtrip_normal.py --episodes 12 --res 512

Same question as scripts/vae_roundtrip_seg.py, same veto logic, different metric. Encode the
ground-truth normal video with the Wan VAE, decode it straight back, and measure the angular
agreement with the GT. No model is involved, so the result is an UPPER BOUND on anything a
trained model can produce through the same decoder.

Why normal needs its own gate rather than inheriting depth's. Both streams are derived from the
same depth.npz, but they occupy the RGB cube completely differently: depth_codec walks a
1-dimensional Hamilton path through saturated cube vertices, while a normal field is a smooth,
low-saturation 2-sphere embedding clustered near (128,128,255). A codec whose entire signal lives
in small perturbations around one point is exactly the kind the VAE's 8x spatial / 4x temporal
compression can flatten, and it would flatten it into something that still LOOKS like a plausible
normal map. Mean cosine catches that; PSNR on the RGB would not.

GATE: mean cosine >= 0.95. Below that the stream is not worth a 44-hour arm and the fix is the
encoding (or the resolution), not more training.

The RGB PSNR control is mandatory and inherited from the seg script for the reason documented
there: a failed VAE load reports catastrophic numbers for everything, which reads exactly like
the veto this script exists to detect. Never let a load failure look like a finding.
"""
import argparse
import collections
import json
import os
import random
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from training.dataset.rlbench_selfgen import RLBenchSelfgenDataset  # noqa: E402
from training.percep.normal_codec import decode_normal, normal_cos  # noqa: E402

# Reuse rather than re-implement: these three are the load-bearing parts of the seg gate and any
# divergence between the two scripts would make their numbers incomparable.
from scripts.vae_roundtrip_seg import load_vae, rgb_control, roundtrip  # noqa: E402

GATE_COS = 0.95
GATE_PSNR = 15.0  # below this the VAE itself is broken; see rgb_control's docstring


def _u8(stream_c2thw):
    return ((stream_c2thw.permute(1, 2, 3, 0).numpy() + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO, "data", "rlbench_selfgen_512_aug"))
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--num-frames", type=int, default=41)
    ap.add_argument("--frame-interval", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(REPO, "reports", "vae_roundtrip_normal.txt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    vae = load_vae(device, dtype)

    ds = RLBenchSelfgenDataset(
        base_path=args.data, num_frames=args.num_frames, frame_interval=args.frame_interval,
        height=args.res, width=args.res, template_mix="video+normal@1.0",
        variations="0", strict_getitem=True,
    )

    # Evenly spaced, not a fixed stride: the tree is sorted by task, so a small stride samples one
    # task repeatedly and a "median across tasks" would be a median of one task.
    n = len(ds)
    indices = [int(round(k * n / args.episodes)) % n for k in range(args.episodes)]

    cos_all, psnrs = [], []
    by_task = collections.defaultdict(list)
    # Flat surfaces dominate the pixel count (table, walls, floor), and they are also the easiest
    # to reconstruct. Splitting by local normal variation keeps the aggregate from hiding a total
    # loss of the detail that actually matters -- object edges and the gripper.
    by_detail = collections.defaultdict(list)

    for k, idx in enumerate(indices):
        random.seed(k)
        s = ds.getitem(idx, force_template="video+normal")
        task = s["path"].replace(args.data, "").strip(os.sep).split(os.sep)[0]

        gt_rgb = _u8(s["streams"]["normal"])
        rec_rgb, latshape = roundtrip(vae, s["streams"]["normal"], device, dtype)
        gt_n, rec_n = decode_normal(gt_rgb), decode_normal(rec_rgb)

        per_px = (gt_n * rec_n).sum(-1)                       # [2T,H,W] cosine per pixel
        cos = float(per_px.mean())
        cos_all.append(cos)
        by_task[task].append(cos)

        # "detail" = how far this pixel's GT normal is from the frame's dominant normal.
        dom = gt_n.reshape(-1, 3).mean(0)
        dom = dom / max(float(np.linalg.norm(dom)), 1e-6)
        dev = 1.0 - (gt_n * dom).sum(-1)
        flat = dev < 0.05
        by_detail["flat (dev<0.05)"].append(float(per_px[flat].mean()) if flat.any() else np.nan)
        by_detail["detailed"].append(float(per_px[~flat].mean()) if (~flat).any() else np.nan)

        psnrs.append(rgb_control(vae, s["streams"]["video"], device, dtype))
        print(f"  [{k + 1}/{len(indices)}] {task:32s} cos={cos:.4f}  latent={tuple(latshape)}")

    lines = []
    lines.append(f"VAE roundtrip -- surface normal stream   ({args.data}, res {args.res}, "
                 f"{args.num_frames}f @ interval {args.frame_interval}, {len(indices)} episodes)")
    lines.append("")
    lines.append(f"mean cosine  : {np.mean(cos_all):.4f}")
    lines.append(f"median       : {np.median(cos_all):.4f}")
    lines.append(f"min / max    : {np.min(cos_all):.4f} / {np.max(cos_all):.4f}")
    lines.append(f"mean angular : {np.degrees(np.arccos(np.clip(np.mean(cos_all), -1, 1))):.2f} deg")
    lines.append("")
    lines.append("by region:")
    for key, vals in by_detail.items():
        v = np.asarray(vals, dtype=float)
        lines.append(f"  {key:20s} mean cos {np.nanmean(v):.4f}")
    lines.append("")
    lines.append("by task:")
    for task in sorted(by_task):
        lines.append(f"  {task:32s} {np.mean(by_task[task]):.4f}")
    lines.append("")
    lines.append(f"RGB PSNR control: {np.mean(psnrs):.2f} dB  (gate {GATE_PSNR} dB)")
    lines.append("")

    mean_cos = float(np.mean(cos_all))
    mean_psnr = float(np.mean(psnrs))
    if mean_psnr < GATE_PSNR:
        lines.append(f"ABORT: RGB control {mean_psnr:.2f} dB < {GATE_PSNR}. The VAE is not loaded "
                     f"correctly -- every number above is meaningless. Do NOT read this as a "
                     f"verdict on the normal codec.")
        verdict = 2
    elif mean_cos >= GATE_COS:
        lines.append(f"PASS: mean cosine {mean_cos:.4f} >= {GATE_COS}. The VAE is not the "
                     f"bottleneck for <normal>; the stream is worth training.")
        verdict = 0
    else:
        lines.append(f"VETO: mean cosine {mean_cos:.4f} < {GATE_COS}. The compression destroys the "
                     f"normal field before any model sees it. Fix the encoding or the resolution "
                     f"before spending an arm on it.")
        verdict = 1

    text = "\n".join(lines)
    print("\n" + text)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text + "\n")
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump({"mean_cos": mean_cos, "median_cos": float(np.median(cos_all)),
                   "rgb_psnr": mean_psnr, "gate_cos": GATE_COS, "verdict": verdict,
                   "per_task": {t: float(np.mean(v)) for t, v in by_task.items()}}, f, indent=1)
    print(f"\nwrote {args.out}")
    return verdict


if __name__ == "__main__":
    sys.exit(main())
