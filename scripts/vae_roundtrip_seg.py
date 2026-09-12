"""P0.2 -- per-class VAE roundtrip for segmentation GT (SEGMENTATION_SCENE_ROLES_PLAN.md §3.2).

    python scripts/vae_roundtrip_seg.py --episodes 12 --res 256

THE HARD VETO GATE. It asks one question, with NO model involved:

    if we encode the ground-truth segmentation video with the VAE and decode it straight back,
    how much of each class survives?

That is an upper bound on anything a trained model can achieve, because the model has to hit a
latent that the same decoder then turns into pixels. If the current target-only protocol's
`target` already dies here (median IoU < 0.5), the signal is being destroyed by the 8x spatial /
4x temporal compression, not by a shortage of supervised pixels -- and scene_roles cannot fix it,
because it does not change how many pixels a target occupies. In that case the fix is resolution,
and Phase B/C should be paused (plan §3.2).

Reports both protocols side by side so the comparison is like-for-like on the same episodes:
  * referring   -- today's encoding (target coloured, everything else black)
  * scene_roles -- the proposed dense role map
"""
import argparse
import collections
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from training.dataset.rlbench_selfgen import RLBenchSelfgenDataset  # noqa: E402
from training.percep.seg_codec import (  # noqa: E402
    ROLE_TO_LABEL,
    SCENE_ROLES,
    decode_known_color,
    decode_scene_roles,
)


def load_vae(device, dtype):
    """Load ONLY the VAE. The DiT is 12.8 GB and irrelevant here -- this diagnostic must be
    cheap enough that there is no excuse to skip it before committing to the annotation work.

    load_state_dict is STRICT and the converter is mandatory. An earlier revision of this script
    used `strict=False` inside a bare try/except, which silently produced a randomly-initialised
    VAE -- and a random VAE reports near-zero IoU for EVERY class, including the 85%-of-pixels
    background, which reads exactly like the veto this script exists to detect. Never make a
    load failure look like a finding.
    """
    from diffsynth.models.wan_video_vae import WanVideoVAE38

    path = os.path.join(REPO, "checkpoints", "Wan-AI", "Wan2.2-TI2V-5B", "Wan2.2_VAE.pth")
    if not os.path.exists(path):
        import glob
        hits = glob.glob(os.path.join(REPO, "checkpoints", "**", "*VAE*.pth"), recursive=True)
        if not hits:
            raise FileNotFoundError(f"no VAE checkpoint under {REPO}/checkpoints")
        path = hits[0]
    print(f"VAE: {path}")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    converted = WanVideoVAE38.state_dict_converter().from_civitai(sd)
    if isinstance(converted, tuple):
        converted = converted[0]
    vae = WanVideoVAE38()
    vae.load_state_dict(converted, strict=True)     # strict: a partial load is a bug, not a warning
    return vae.to(device=device, dtype=dtype).eval()


def rgb_control(vae, rgb_c2thw, device, dtype):
    """Roundtrip the RGB stream and report PSNR. This is the control that separates
    'the VAE destroys small segments' from 'the VAE is not loaded'. A working Wan VAE
    reconstructs natural video at well over 20 dB; a random one sits near 5 dB."""
    u8, _ = roundtrip(vae, rgb_c2thw, device, dtype)
    gt = ((rgb_c2thw.permute(1, 2, 3, 0).numpy() + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)
    mse = float(((u8.astype(np.float64) - gt.astype(np.float64)) ** 2).mean())
    return 10.0 * np.log10(255.0 ** 2 / max(mse, 1e-9))


@torch.no_grad()
def _rt_one(vae, x_cthw, device, dtype):
    x = x_cthw.unsqueeze(0).to(device=device, dtype=dtype)   # [1,C,T,H,W]
    lat = vae.encode(x, device=device)
    if isinstance(lat, (list, tuple)):
        lat = lat[0]
    if lat.dim() == 4:
        lat = lat.unsqueeze(0)
    rec = vae.decode(lat, device=device)
    if isinstance(rec, (list, tuple)):
        rec = rec[0]
    if rec.dim() == 4:
        rec = rec.unsqueeze(0)
    return rec[0].float().clamp(-1, 1), lat.shape               # [C,T,H,W]


def roundtrip(vae, stream_c2thw, device, dtype):
    """[C, 2T, H, W] in [-1,1] -> uint8 RGB [2T,H,W,3] after VAE encode+decode.

    Each VIEW is encoded separately, exactly as train.py does
    (`encode_video(video[:, :, v*T:(v+1)*T])`). That is not a cosmetic detail: the Wan VAE
    compresses time 4x with T_lat = 1 + (T-1)/4, so it only round-trips cleanly when
    T = 1 (mod 4). num_frames=41 satisfies that per view; the concatenated 82-frame stream does
    not, and feeding it whole silently returns 81 frames -- which shows up as a shape mismatch
    here but would show up as a quietly misaligned comparison in a less careful script.
    """
    T = stream_c2thw.shape[1] // 2
    outs, latshape = [], None
    for v in range(2):
        rec, latshape = _rt_one(vae, stream_c2thw[:, v * T:(v + 1) * T], device, dtype)
        outs.append(rec)
    rec = torch.cat(outs, dim=1)                                # [C,2T,H,W]
    u8 = ((rec.permute(1, 2, 3, 0).cpu().numpy() + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)
    return u8, latshape


SIZE_BUCKETS = ("<0.25%", "0.25-1%", "1-5%", ">5%")


def _bucket(frac):
    if frac < 0.0025:
        return "<0.25%"
    if frac < 0.01:
        return "0.25-1%"
    if frac < 0.05:
        return "1-5%"
    return ">5%"


def iou(pred, gt):
    inter = int((pred & gt).sum())
    union = int((pred | gt).sum())
    return (float(inter) / float(union)) if union else None    # None = class absent both sides


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--num-frames", type=int, default=41)
    ap.add_argument("--frame-interval", type=int, default=3)
    ap.add_argument("--stride", type=int, default=97, help="index stride, to spread across tasks")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    vae = load_vae(device, dtype)

    # 512_aug, not "rlbench_selfgen": that symlink points at the deleted v2 tree and dangles.
    common = dict(base_path=os.path.join(REPO, "data", "rlbench_selfgen_512_aug"),
                  num_frames=args.num_frames, frame_interval=args.frame_interval,
                  height=args.res, width=args.res,
                  template_mix="video+segmentation@1.0", variations="0", strict_getitem=True)
    ds_ref = RLBenchSelfgenDataset(segmentation_mode="referring", **common)
    ds_rol = RLBenchSelfgenDataset(segmentation_mode="scene_roles", **common)

    ref_ious = []                                  # referring: the target instances
    role_ious = collections.defaultdict(list)      # scene_roles: per role
    rgb_psnrs = []                                 # sanity control, see rgb_control()
    by_task = collections.defaultdict(list)        # referring target IoU, per task
    by_size = collections.defaultdict(list)        # referring target IoU, per GT-area bucket
    tasks = []

    # Evenly spaced indices, not a fixed stride: the tree is sorted by task, so a small stride
    # samples one task over and over and the "median across tasks" would be a median of one task.
    n = len(ds_ref)
    indices = [int(round(k * n / args.episodes)) % n for k in range(args.episodes)]

    import random
    for k, idx in enumerate(indices):
        random.seed(k)
        s_ref = ds_ref.getitem(idx, force_template="video+segmentation")
        random.seed(k)
        s_rol = ds_rol.getitem(idx, force_template="video+segmentation")
        task = s_ref["path"].split("episodes/")[0].strip("/").split("/")[-3]
        tasks.append(task)

        # ---- referring ----
        id_groups, color_map = ds_ref._referred_id_groups(s_ref["path"], s_ref["text"].split("> ")[-1])
        u8, latshape = roundtrip(vae, s_ref["streams"]["segmentation"], device, dtype)
        gt8 = ((s_ref["streams"]["segmentation"].permute(1, 2, 3, 0).numpy() + 1.0) * 127.5
               ).round().astype(np.uint8)
        if color_map:
            dec_p = decode_known_color(u8, color_map)
            dec_g = decode_known_color(gt8, color_map)
            for name in color_map:
                v = iou(dec_p[name], dec_g[name])
                if v is not None:
                    ref_ious.append(v)
                    # The plan's worry is specifically about SMALL targets (median 1% of pixels,
                    # some tasks 0.2%). An aggregate median hides them, so bucket by GT area --
                    # this is plan §11.7's area-bucketed recall applied to the roundtrip bound.
                    frac = float(dec_g[name].mean())
                    by_task[task].append(v)
                    by_size[_bucket(frac)].append(v)

        # ---- scene_roles ----
        lut, present = ds_rol._scene_role_lut(s_rol["path"], s_rol["text"].split("> ")[-1])
        u8r, _ = roundtrip(vae, s_rol["streams"]["segmentation"], device, dtype)
        gt8r = ((s_rol["streams"]["segmentation"].permute(1, 2, 3, 0).numpy() + 1.0) * 127.5
                ).round().astype(np.uint8)
        dec_p = decode_scene_roles(u8r, present_roles=present)
        dec_g = decode_scene_roles(gt8r, present_roles=present)
        for role in present:
            if not dec_g[role].any():
                continue
            v = iou(dec_p[role], dec_g[role])
            if v is not None:
                role_ious[role].append(v)

        p = rgb_control(vae, s_ref["streams"]["video"], device, dtype)
        rgb_psnrs.append(p)
        print(f"  [{k + 1}/{args.episodes}] {task:32s} latent={tuple(latshape)}  rgb_psnr={p:.1f}dB")

    def stats(v):
        a = np.array(v, dtype=float)
        return (f"n={len(a):3d}  median={np.median(a):.4f}  mean={np.mean(a):.4f}  "
                f"p10={np.percentile(a, 10):.4f}  min={a.min():.4f}")

    print("\n" + "=" * 78)
    print(f"VAE per-class roundtrip  (res={args.res}, {args.num_frames} frames, "
          f"{len(set(tasks))} tasks, {args.episodes} episodes)")
    print("=" * 78)

    print("\n--- referring protocol (today) ---")
    ref_med = None
    if ref_ious:
        ref_med = float(np.median(ref_ious))
        print(f"  target instances  {stats(ref_ious)}")
    else:
        print("  no target instances resolved")

    print("\n--- scene_roles protocol (proposed) ---")
    for role in SCENE_ROLES:
        if role in role_ious and role_ious[role]:
            print(f"  {role:12s}      {stats(role_ious[role])}")

    print("\n--- referring `target` by GT area (the small-target question) ---")
    for b in SIZE_BUCKETS:
        if by_size.get(b):
            print(f"  {b:10s}        {stats(by_size[b])}")

    print("\n--- referring `target` by task ---")
    for t in sorted(by_task):
        a = np.array(by_task[t], dtype=float)
        print(f"  {t:32s} n={len(a):3d}  median={np.median(a):.4f}  min={a.min():.4f}")

    print("\n--- sanity control: RGB roundtrip ---")
    rgb_med = float(np.median(rgb_psnrs))
    print(f"  RGB PSNR          median={rgb_med:.2f} dB  min={min(rgb_psnrs):.2f} dB")

    print("\n" + "=" * 78)
    print("VERDICT (plan §3.2)")
    print("=" * 78)
    if rgb_med < 15.0:
        print(f"  ABORT: RGB roundtrip is only {rgb_med:.1f} dB. A working Wan VAE is well above")
        print("  20 dB, so the VAE is not loaded correctly and NONE of the IoUs above mean")
        print("  anything. Fix the checkpoint load before reading any verdict.")
        return 3
    if ref_med is None:
        print("  INCONCLUSIVE: no referring targets resolved.")
        return 2
    print(f"  referring `target` roundtrip IoU median = {ref_med:.4f}")
    if ref_med < 0.5:
        print("  --> BELOW 0.5: the VAE is destroying the target signal. Dense role labels do")
        print("      NOT change how many pixels a target occupies, so they cannot fix this.")
        print("      PAUSE Phase B/C; the problem is resolution/compression (plan §3.2).")
        return 1
    if ref_med > 0.8:
        print("  --> ABOVE 0.8: the VAE is not the bottleneck. Plan proceeds.")
    else:
        print("  --> BETWEEN 0.5 and 0.8: not a veto, but the VAE is taking a real bite.")
        print("      Report it alongside any seg result; it caps what any model can reach.")
    tgt = role_ious.get("target", [])
    if tgt:
        print(f"  scene_roles `target` median = {np.median(tgt):.4f} "
              f"(same pixels, different neighbours)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
