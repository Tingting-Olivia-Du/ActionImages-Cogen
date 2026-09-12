"""VETO GATE for the depth stream: how much of it survives a VAE roundtrip?

    python scripts/vae_roundtrip_depth.py --episodes 12 --res 512

The third of the three gates, alongside scripts/vae_roundtrip_seg.py and
scripts/vae_roundtrip_normal.py, and it reuses their loader/roundtrip/control helpers so the
three numbers are comparable. Encode the GT depth video (already codec-encoded to RGB by
depth_codec) with the Wan VAE, decode it straight back, and measure how far the decoded metric
depth has moved. No model is involved, so the result is an UPPER BOUND on anything a trained
model can produce through the same decoder.

WHY THE REFERENCE IS THE DECODED GT STREAM, NOT THE RAW depth.npz. Comparing against raw depth
would fold codec quantisation (measured <2% AbsRel by depth_codec's own acceptance test) into a
number whose column heading says "VAE roundtrip", and would make this gate incomparable with the
normal gate, which compares decode(GT stream) against decode(roundtrip stream) for exactly this
reason. Quantisation cancels; what is left is the VAE.

TWO METRICS, because the RGB-cube codec has two distinct failure modes:
  absrel      -- the depth moved. Smooth degradation, what the reader expects.
  off_path    -- the pixel left the Hamilton path far enough that decode_depth calls it INVALID.
                 This one is a cliff, not a slope: an off-path pixel yields no depth at all
                 rather than a slightly wrong one, and an aggregate AbsRel computed only over
                 surviving pixels would hide it completely. Report both or neither.

GATE: absrel <= 0.05 on pixels valid in both, AND off_path <= 0.02. Below either, the stream is
not worth an arm and the fix is the encoding (or the resolution), not more training.

The RGB PSNR control is mandatory and inherited for the reason documented in the seg script: a
failed VAE load reports catastrophic numbers for everything, which reads exactly like the veto
this script exists to detect. Never let a load failure look like a finding.
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
from training.percep.depth_codec import decode_depth  # noqa: E402
from scripts.vae_roundtrip_seg import load_vae, rgb_control, roundtrip  # noqa: E402

GATE_ABSREL = 0.05
GATE_OFFPATH = 0.02
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
    ap.add_argument("--out", default=os.path.join(REPO, "reports", "vae_roundtrip_depth.txt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    vae = load_vae(device, dtype)

    ds = RLBenchSelfgenDataset(
        base_path=args.data, num_frames=args.num_frames, frame_interval=args.frame_interval,
        height=args.res, width=args.res, template_mix="video+depth@1.0",
        variations="0", strict_getitem=True,
    )

    # Evenly spaced, not a fixed stride: the tree is sorted by task, so a small stride samples one
    # task repeatedly and a "median across tasks" would be a median of one task.
    n = len(ds)
    indices = [int(round(k * n / args.episodes)) % n for k in range(args.episodes)]

    absrels, offpaths, psnrs = [], [], []
    by_task = collections.defaultdict(list)
    # Near-field pixels are what manipulation actually needs, and they are also where the Barron
    # transform spends most of its dynamic range. An aggregate dominated by the far wall would
    # hide a loss of exactly the range that matters.
    by_range = collections.defaultdict(list)

    for k, idx in enumerate(indices):
        random.seed(k)
        s = ds.getitem(idx, force_template="video+depth")
        task = s["path"].replace(args.data, "").strip(os.sep).split(os.sep)[0]

        gt_rgb = _u8(s["streams"]["depth"])
        rec_rgb, latshape = roundtrip(vae, s["streams"]["depth"], device, dtype)
        gt_d, rec_d = decode_depth(gt_rgb), decode_depth(rec_rgb)

        gt_ok = np.isfinite(gt_d)
        both = gt_ok & np.isfinite(rec_d)
        # off_path is measured over pixels the GT stream decodes fine, so it isolates damage the
        # VAE did rather than counting the sky the codec already calls invalid.
        off = float((gt_ok & ~np.isfinite(rec_d)).sum() / max(int(gt_ok.sum()), 1))
        rel = np.abs(rec_d[both] - gt_d[both]) / np.clip(gt_d[both], 1e-6, None)
        absrel = float(rel.mean())

        absrels.append(absrel)
        offpaths.append(off)
        by_task[task].append(absrel)

        near = both & (gt_d < 1.5)
        far = both & (gt_d >= 1.5)
        by_range["near (<1.5 m)"].append(
            float((np.abs(rec_d[near] - gt_d[near]) / gt_d[near]).mean()) if near.any() else np.nan)
        by_range["far (>=1.5 m)"].append(
            float((np.abs(rec_d[far] - gt_d[far]) / gt_d[far]).mean()) if far.any() else np.nan)

        psnrs.append(rgb_control(vae, s["streams"]["video"], device, dtype))
        print(f"  [{k + 1}/{len(indices)}] {task:32s} AbsRel={absrel * 100:.3f}%  "
              f"off_path={off * 100:.3f}%  latent={tuple(latshape)}")

    lines = []
    lines.append(f"VAE roundtrip -- depth stream   ({args.data}, res {args.res}, "
                 f"{args.num_frames}f @ interval {args.frame_interval}, {len(indices)} episodes)")
    lines.append("")
    lines.append(f"mean AbsRel  : {np.mean(absrels) * 100:.3f}%")
    lines.append(f"median       : {np.median(absrels) * 100:.3f}%")
    lines.append(f"min / max    : {np.min(absrels) * 100:.3f}% / {np.max(absrels) * 100:.3f}%")
    lines.append(f"off-path frac: {np.mean(offpaths) * 100:.3f}%  "
                 f"(GT-valid pixels the roundtrip pushed off the cube path)")
    lines.append("")
    lines.append("by range:")
    for key, vals in by_range.items():
        lines.append(f"  {key:20s} AbsRel {np.nanmean(np.asarray(vals, float)) * 100:.3f}%")
    lines.append("")
    lines.append("by task:")
    for task in sorted(by_task):
        lines.append(f"  {task:32s} {np.mean(by_task[task]) * 100:.3f}%")
    lines.append("")
    lines.append(f"RGB PSNR control: {np.mean(psnrs):.2f} dB  (gate {GATE_PSNR} dB)")
    lines.append("")

    mean_absrel, mean_off, mean_psnr = float(np.mean(absrels)), float(np.mean(offpaths)), float(np.mean(psnrs))
    if mean_psnr < GATE_PSNR:
        lines.append(f"ABORT: RGB control {mean_psnr:.2f} dB < {GATE_PSNR}. The VAE is not loaded "
                     f"correctly -- every number above is meaningless. Do NOT read this as a "
                     f"verdict on the depth codec.")
        verdict = 2
    elif mean_absrel <= GATE_ABSREL and mean_off <= GATE_OFFPATH:
        lines.append(f"PASS: AbsRel {mean_absrel * 100:.3f}% <= {GATE_ABSREL * 100:g}% and off-path "
                     f"{mean_off * 100:.3f}% <= {GATE_OFFPATH * 100:g}%. The VAE is not the "
                     f"bottleneck for <depth>; the stream is worth training.")
        verdict = 0
    else:
        lines.append(f"FAIL: AbsRel {mean_absrel * 100:.3f}% (gate {GATE_ABSREL * 100:g}%), off-path "
                     f"{mean_off * 100:.3f}% (gate {GATE_OFFPATH * 100:g}%). Fix the encoding or the "
                     f"resolution before spending an arm on this stream.")
        verdict = 1

    text = "\n".join(lines)
    print("\n" + text)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text + "\n")
    print(f"\nwrote {args.out}")
    raise SystemExit(verdict)


if __name__ == "__main__":
    main()
