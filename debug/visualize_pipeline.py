"""Stage-by-stage visual trace of the <depth><action> pipeline, GT on disk -> DiT latents.

Every assertion in tests/ checks a number. This checks the PICTURES, because several failure
modes in this pipeline produce numbers that look fine:

  - a depth encoding that survives decode() perfectly but is destroyed by the VAE;
  - an action image rendered against the wrong view's camera (still a valid-looking heatmap);
  - a segment order that is self-consistent but not what the pretrained prior expects;
  - a condition mask that marks the wrong frames (loss still decreases, just on the wrong thing).

Stages (each writes a PNG into debug/out/):

  S1  GT depth / seg on disk, as metres / handle ids
  S2  codec encode -> the RGB the model actually sees
  S3  the dataset tensor itself, BOTH halves, next to the RGB the same window would give
  S4  codec-only roundtrip error (no VAE) -- the known-good floor
  S5  VAE roundtrip: encode -> VAE.encode -> VAE.decode -> decode          <-- the open risk
  S6  natural-RGB VAE roundtrip on the same clip -- the control for S5
  S7  the action-image stream, exactly as ActionImagesModel.forward builds it
  S8  the assembled 4-segment sequence + condition mask
  S9  per-segment latent statistics

S5 is the one that has never been measured. plan/core/9-depth-seg-codec-vs-genception.md
§8.2 flags it as required-before-adoption: the codec's colour path is a thin 1-D curve
through RGB space, i.e. plausibly out of distribution for a VAE trained on natural video.
If the VAE cannot preserve it, the depth arm cannot work no matter how correct the data
plumbing is -- and every test in tests/ would still pass.

Usage:
    cd /workspace/ttdu/ActionImages-Cogen && export PYTHONPATH=$PWD
    CUDA_VISIBLE_DEVICES=<free gpu> python debug/visualize_pipeline.py
    python debug/visualize_pipeline.py --no-vae     # skip S5/S6/S9 (no GPU needed)
"""
import argparse
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from training.dataset import RLBenchSelfgenDataset
from training.percep.depth_codec import MAX_VALID, MIN_VALID, decode_depth, encode_depth
from training.percep.seg_codec import decode_known_color, encode_known_color, handles_to_mask
from training.utils import (
    project_action_5d_to_rgb_torch,
    project_actions_7d_to_5d_torch_batch,
)

OUT = os.path.join(REPO, "debug", "out")
RES, NUM_FRAMES = 256, 41
FRAME_PICKS = 4  # how many frames of the clip to show per stage


def savefig(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, REPO)}")
    return path


def to_uint8(chw_tensor):
    """[C,T,H,W] in [-1,1] -> [T,H,W,3] uint8 (inverse of the dataset's /127.5 - 1)."""
    arr = chw_tensor.permute(1, 2, 3, 0).float().numpy()
    return np.clip(np.round((arr + 1.0) * 127.5), 0, 255).astype(np.uint8)


def picks(T, n=FRAME_PICKS):
    return list(np.linspace(0, T - 1, n).astype(int))


# --------------------------------------------------------------------------------------
def stage1_2_3(ds, index, report):
    """GT on disk -> codec encode -> the dataset tensor, for both views."""
    random.seed(7)
    rgb_sample = ds.getitem(index, force_template="video+action")
    random.seed(7)
    dep_sample = ds.getitem(index, force_template="depth+action")
    assert rgb_sample["view_indices"] == dep_sample["view_indices"]
    assert rgb_sample["frame_indices"] == dep_sample["frame_indices"]

    frames = dep_sample["frame_indices"]
    views = [dep_sample["view_dirs"][i] for i in dep_sample["view_indices"]]
    T = NUM_FRAMES
    fi = picks(T)
    report["episode"] = dep_sample["path"]
    report["views"] = [os.path.basename(v) for v in views]
    report["frame_window"] = [int(frames[0]), int(frames[-1])]
    report["text_depth"] = dep_sample["text"]
    report["text_video"] = rgb_sample["text"]

    # S1: raw GT depth in metres, straight off disk
    fig, axes = plt.subplots(2, len(fi), figsize=(3.1 * len(fi), 6.4))
    for r, view in enumerate(views):
        gt = np.load(os.path.join(view, "depth.npz"))["depth"][frames].astype(np.float32)
        for c, f in enumerate(fi):
            ax = axes[r, c]
            im = ax.imshow(gt[f], cmap="turbo")
            ax.set_title(f"{os.path.basename(view)}  t={f}\n"
                         f"{np.nanmin(gt[f]):.2f}–{np.nanmax(gt[f]):.2f} m", fontsize=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("S1  GT depth on disk (metres). Row = view; the two rows are the two "
                 "segments a <depth><action> sample fills.", fontsize=11)
    savefig(fig, "S1_gt_depth.png")

    # S2/S3: encode -> RGB, and the dataset tensor, side by side with natural RGB
    dep_u8 = to_uint8(dep_sample["video"])   # [2T,H,W,3]
    rgb_u8 = to_uint8(rgb_sample["video"])
    fig, axes = plt.subplots(3, len(fi) * 2, figsize=(2.5 * len(fi) * 2, 7.6))
    for half, (name, off) in enumerate([("cond", 0), ("target", T)]):
        view = views[half]
        gt = np.load(os.path.join(view, "depth.npz"))["depth"][frames].astype(np.float32)
        direct = encode_depth(gt)  # what encode_depth alone produces
        for c, f in enumerate(fi):
            col = half * len(fi) + c
            axes[0, col].imshow(rgb_u8[off + f]);   axes[0, col].axis("off")
            axes[1, col].imshow(direct[f]);          axes[1, col].axis("off")
            axes[2, col].imshow(dep_u8[off + f]);    axes[2, col].axis("off")
            axes[0, col].set_title(f"{name} {os.path.basename(view)} t={f}", fontsize=8)
    axes[0, 0].set_ylabel("natural RGB", fontsize=9)
    for r, lab in enumerate(["S3 natural RGB\n(video arm)", "S2 encode_depth(GT)",
                             "S3 dataset tensor\n(depth arm)"]):
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(lab, fontsize=9)
    fig.suptitle("S2/S3  Row2 vs Row3 must match exactly (the tensor IS the codec output). "
                 "Row3 must differ from Row1 in BOTH halves -- if either half looks like "
                 "Row1, only one view was encoded.", fontsize=11)
    savefig(fig, "S2_S3_encode_and_tensor.png")

    # the tensor must equal a direct encode, bit for bit modulo the uint8 round trip
    mismatch = {}
    for half, off in [("cond", 0), ("target", T)]:
        view = views[0 if half == "cond" else 1]
        gt = np.load(os.path.join(view, "depth.npz"))["depth"][frames].astype(np.float32)
        direct = encode_depth(gt)
        d = np.abs(direct.astype(int) - dep_u8[off:off + T].astype(int)).max()
        mismatch[half] = int(d)
    report["S3_tensor_vs_direct_encode_max_abs_diff"] = mismatch
    print(f"  S3 tensor vs direct encode, max |diff| per half: {mismatch} (expect 0/0)")
    return dep_sample, rgb_sample, views, frames


# --------------------------------------------------------------------------------------
def stage4_codec_only(views, frames, report):
    """decode(encode(GT)) vs GT -- the no-VAE floor."""
    fi = picks(NUM_FRAMES)
    fig, axes = plt.subplots(len(views), len(fi), figsize=(3.1 * len(fi), 3.2 * len(views)))
    stats = {}
    for r, view in enumerate(views):
        gt = np.load(os.path.join(view, "depth.npz"))["depth"][frames].astype(np.float32)
        dec = decode_depth(encode_depth(gt))
        m = np.isfinite(dec) & (gt > MIN_VALID) & (gt < MAX_VALID)
        absrel = np.abs(dec - gt) / np.where(gt > 0, gt, np.nan)
        stats[os.path.basename(view)] = {
            "absrel_mean": float(np.mean(absrel[m])),
            "valid_frac": float(m.mean()),
        }
        for c, f in enumerate(fi):
            ax = axes[r, c]
            im = ax.imshow(absrel[f] * 100, cmap="magma", vmin=0, vmax=2.0)
            ax.set_title(f"{os.path.basename(view)} t={f}", fontsize=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, label="AbsRel %")
    fig.suptitle("S4  Codec-only roundtrip error (no VAE). This is the floor: any larger "
                 "error downstream is the VAE's, not the codec's.", fontsize=11)
    savefig(fig, "S4_codec_only_error.png")
    report["S4_codec_only"] = stats
    print(f"  S4 codec-only: {stats}")


# --------------------------------------------------------------------------------------
def load_vae(device):
    """Load ONLY the VAE (no DiT, no T5) -- ~250MB instead of ~32GB."""
    from training.models import ModelConfig
    from training.wan_video_action_images import WanVideoActionImagesPipeline

    # from_pretrained dereferences tokenizer_config unconditionally, so it must be passed even
    # though nothing here encodes text. It only resolves a path -- no weights are loaded.
    pipe = WanVideoActionImagesPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[ModelConfig(local_model_path="checkpoints",
                                   model_id="Wan-AI/Wan2.2-TI2V-5B",
                                   origin_file_pattern="Wan*_VAE.pth",
                                   offload_device=None, skip_download=True)],
        tokenizer_config=ModelConfig(local_model_path="checkpoints",
                                     model_id="Wan-AI/Wan2.2-TI2V-5B",
                                     origin_file_pattern="google/*", skip_download=True),
        redirect_common_files=False,
    )
    return pipe


def vae_roundtrip(pipe, seg_cthw, device):
    """ONE segment [C,T,H,W] in [-1,1] -> VAE encode -> decode, plus its latent.

    Per SEGMENT, not per sample: `ActionImagesModel.forward` calls encode_video separately on
    `video[:, :, :T]` and `video[:, :, T:]` (train.py:236-237), so the two views never share a
    temporal context inside the VAE. Encoding the concatenated [C,2T,H,W] tensor instead would
    both mix them and hit Wan's T_latent = (T-1)/4 + 1 rule at a non-integral point -- 82
    frames round-trips back as 81, which is how this was caught.
    """
    x = seg_cthw.unsqueeze(0).to(device=device, dtype=pipe.torch_dtype)  # [1,C,T,H,W]
    with torch.no_grad():
        lat = pipe.encode_video(x)
        rec = pipe.decode_video(lat.to(device=device, dtype=pipe.torch_dtype))
    rec = rec[0].float().cpu().clamp(-1, 1)
    assert rec.shape == seg_cthw.shape, (
        f"VAE roundtrip changed shape {tuple(seg_cthw.shape)} -> {tuple(rec.shape)}; "
        f"T must satisfy T_latent = (T-1)/4 + 1 exactly"
    )
    return rec, lat[0].float().cpu()


def stage5_6_vae(pipe, dep_sample, rgb_sample, views, frames, device, report):
    """The open question: does the depth encoding survive the frozen video VAE?"""
    T = NUM_FRAMES
    fi = picks(T)
    results = {"per_half": {}, "pixel_l1": {}}

    # Round-trip each SEGMENT independently, exactly as forward does.
    rec = {}
    for tag, sample in (("depth", dep_sample), ("video", rgb_sample)):
        halves = []
        for off in (0, T):
            r, lat = vae_roundtrip(pipe, sample["video"][:, off:off + T], device)
            halves.append(r)
            results.setdefault(f"{tag}_latent_shape", list(lat.shape))
        rec[tag] = torch.cat(halves, dim=1)
        results["pixel_l1"][tag] = float((rec[tag] - sample["video"]).abs().mean())

    dep_rec_u8, dep_pre_u8 = to_uint8(rec["depth"]), to_uint8(dep_sample["video"])
    fig, axes = plt.subplots(4, len(fi), figsize=(3.1 * len(fi), 12.4))
    for half, (name, off) in enumerate([("cond", 0), ("target", T)]):
        view = views[half]
        gt = np.load(os.path.join(view, "depth.npz"))["depth"][frames].astype(np.float32)
        dec_pre = decode_depth(dep_pre_u8[off:off + T])
        dec_post = decode_depth(dep_rec_u8[off:off + T])
        gt_ok = (gt > MIN_VALID) & (gt < MAX_VALID)
        m_pre = np.isfinite(dec_pre) & gt_ok
        m_post = np.isfinite(dec_post) & gt_ok
        results["per_half"][name] = {
            "view": os.path.basename(view),
            "absrel_pre_vae": float(np.mean(np.abs(dec_pre[m_pre] - gt[m_pre]) / gt[m_pre])),
            "absrel_post_vae": (float(np.mean(np.abs(dec_post[m_post] - gt[m_post]) / gt[m_post]))
                                if m_post.any() else float("nan")),
            "valid_frac_pre_vae": float(m_pre.mean()),
            "valid_frac_post_vae": float(m_post.mean()),
            "gt_valid_frac": float(gt_ok.mean()),
        }
        if half == 0:  # plot the cond half in detail
            for c, f in enumerate(fi):
                axes[0, c].imshow(dep_pre_u8[off + f]); axes[0, c].axis("off")
                axes[0, c].set_title(f"t={f}", fontsize=8)
                axes[1, c].imshow(dep_rec_u8[off + f]); axes[1, c].axis("off")
                d = np.abs(dep_rec_u8[off + f].astype(int)
                           - dep_pre_u8[off + f].astype(int)).mean(-1)
                im = axes[2, c].imshow(d, cmap="magma", vmin=0, vmax=40)
                axes[2, c].axis("off")
                plt.colorbar(im, ax=axes[2, c], fraction=0.046, label="|dRGB|")
                err = np.abs(dec_post[f] - gt[f]) / np.where(gt[f] > 0, gt[f], np.nan)
                im = axes[3, c].imshow(err * 100, cmap="magma", vmin=0, vmax=25)
                axes[3, c].axis("off")
                plt.colorbar(im, ax=axes[3, c], fraction=0.046, label="AbsRel %")
    for r, lab in enumerate(["depth RGB\n(model input)", "after VAE\nenc+dec",
                             "|dRGB|", "decoded depth\nAbsRel"]):
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(lab, fontsize=9)
    ph = results["per_half"]["cond"]
    fig.suptitle(f"S5  Does the depth encoding survive the frozen video VAE?\n"
                 f"cond half: AbsRel {ph['absrel_pre_vae']*100:.3f}% -> "
                 f"{ph['absrel_post_vae']*100:.2f}%, decodable pixels "
                 f"{ph['valid_frac_pre_vae']*100:.1f}% -> {ph['valid_frac_post_vae']*100:.1f}%. "
                 f"White in row 4 = pushed off the colour path (NaN).", fontsize=11)
    savefig(fig, "S5_vae_roundtrip_depth.png")

    # S6: the control -- how much does the same VAE hurt NATURAL rgb on the same clip?
    fig, axes = plt.subplots(2, len(fi), figsize=(3.1 * len(fi), 6.4))
    rgb_rec_u8 = to_uint8(rec["video"])
    for c, f in enumerate(fi):
        axes[0, c].imshow(to_uint8(rgb_sample["video"])[f]); axes[0, c].axis("off")
        axes[0, c].set_title(f"t={f}", fontsize=8)
        axes[1, c].imshow(rgb_rec_u8[f]); axes[1, c].axis("off")
    for r, lab in enumerate(["natural RGB", "after VAE"]):
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(lab, fontsize=9)
    fig.suptitle(f"S6  Control: the same VAE on natural RGB, same clip. pixel L1 = "
                 f"{results['pixel_l1']['video']:.4f} (rgb) vs "
                 f"{results['pixel_l1']['depth']:.4f} (depth)", fontsize=11)
    savefig(fig, "S6_vae_roundtrip_rgb_control.png")

    report["S5_S6_vae"] = results
    print(f"  S5/S6 VAE: {json.dumps(results, indent=2)}")


# --------------------------------------------------------------------------------------
def stage7_8_action_and_layout(dep_sample, device, report):
    """The action stream and the assembled sequence, built exactly as forward does."""
    T = NUM_FRAMES
    a7 = dep_sample["action_7d"].unsqueeze(0)
    extr = dep_sample["extrinsics"].unsqueeze(0)
    intr = dep_sample["intrinsics"].unsqueeze(0)
    num_views = dep_sample["video"].shape[1] // T

    # verbatim from ActionImagesModel.forward (train.py:257-262)
    a7r = a7.repeat(1, num_views, 1)
    a5 = project_actions_7d_to_5d_torch_batch(a7r, extr, intr)
    av = project_action_5d_to_rgb_torch(a5, RES, RES)      # [B,2T,H,W,3] in [0,1]
    av = av.permute(0, 4, 1, 2, 3)
    av = av * 2 - 1
    action_u8 = to_uint8(av[0])

    fi = picks(T)
    fig, axes = plt.subplots(2, len(fi), figsize=(3.1 * len(fi), 6.4))
    for half, (name, off) in enumerate([("view1", 0), ("view2", T)]):
        for c, f in enumerate(fi):
            axes[half, c].imshow(action_u8[off + f]); axes[half, c].axis("off")
            axes[half, c].set_title(f"{name} t={f}", fontsize=8)
    for r, lab in enumerate(["action image\nsegment 2", "action image\nsegment 4"]):
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(lab, fontsize=9)
    fig.suptitle("S7  The action stream: the SAME 3D trajectory reprojected through each "
                 "view's own camera. The two rows must differ (different viewpoints) but "
                 "trace the same motion.", fontsize=11)
    savefig(fig, "S7_action_images.png")
    report["S7_action_nonzero_frac"] = float((action_u8 > 8).mean())

    # S8: the assembled sequence + condition mask, in DiT segment order
    dep_u8 = to_uint8(dep_sample["video"])
    T_l = (T - 1) // 4 + 1
    seg_names = ["1: v1_depth", "2: v1_action", "3: v2_depth", "4: v2_action"]
    seg_px = [dep_u8[:T], action_u8[:T], dep_u8[T:], action_u8[T:]]
    fig, axes = plt.subplots(4, 5, figsize=(15, 12.2))
    for r, (nm, px) in enumerate(zip(seg_names, seg_px)):
        show = [0] + picks(T, 4)[1:]
        for c, f in enumerate(show):
            axes[r, c].imshow(px[f]); axes[r, c].axis("off")
            cond = (c == 0)
            axes[r, c].set_title(f"t={f}" + ("  [CONDITION: clean]" if cond else ""),
                                 fontsize=8, color=("tab:green" if cond else "black"))
            if cond:
                for s in ("top", "bottom", "left", "right"):
                    axes[r, c].axis("on")
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
                for sp in axes[r, c].spines.values():
                    sp.set_edgecolor("tab:green"); sp.set_linewidth(3)
        mask = np.zeros(T_l); mask[0] = 1
        axes[r, 4].imshow(mask[None, :], cmap="Greens", vmin=0, vmax=1, aspect="auto")
        axes[r, 4].set_title(f"latent mask ({T_l} frames)", fontsize=8)
        axes[r, 4].set_yticks([]); axes[r, 4].set_xticks(range(T_l))
        axes[r, 4].set_xticklabels(range(T_l), fontsize=6)
        axes[r, 0].set_ylabel(nm, fontsize=10)
    fig.suptitle("S8  The assembled sequence the DiT sees: 4 segments concatenated along "
                 "TIME, one forward pass.\nGreen = the clean first latent frame of each "
                 "segment -- that is how the model knows what modality the segment is.",
                 fontsize=11)
    savefig(fig, "S8_assembled_sequence.png")
    report["S8_segments"] = seg_names
    report["S8_T_latent_per_segment"] = int(T_l)


def stage9_latents(pipe, dep_sample, rgb_sample, device, report):
    """Per-segment latent statistics: is the depth segment in the same numeric regime as RGB?"""
    T = NUM_FRAMES
    stats, flat = {}, {}
    for tag, sample in (("depth", dep_sample), ("video", rgb_sample)):
        segs = []
        for off in (0, T):  # per segment, as forward does
            x = sample["video"][:, off:off + T].unsqueeze(0).to(device=device, dtype=pipe.torch_dtype)
            with torch.no_grad():
                segs.append(pipe.encode_video(x)[0].float().cpu())
        stats[tag] = {
            "shape_per_segment": list(segs[0].shape),
            "seg1_mean": float(segs[0].mean()), "seg1_std": float(segs[0].std()),
            "seg3_mean": float(segs[1].mean()), "seg3_std": float(segs[1].std()),
            "abs_max": float(max(s.abs().max() for s in segs)),
        }
        flat[tag] = torch.cat([s.flatten() for s in segs]).numpy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for tag in ("depth", "video"):
        ax.hist(flat[tag], bins=160, alpha=0.55, density=True,
                label=f"{tag} (mean={flat[tag].mean():.2f}, std={flat[tag].std():.2f})")
    ax.set_xlabel("latent value"); ax.set_ylabel("density"); ax.legend()
    ax.set_title("S9  Latent distribution: depth vs natural RGB through the same VAE.\n"
                 "A large shift means the depth segment sits off the manifold the DiT was "
                 "pretrained on -- warm-starting would then be fighting the prior.")
    savefig(fig, "S9_latent_distribution.png")
    report["S9_latents"] = stats
    print(f"  S9 latents: {json.dumps(stats, indent=2)}")


def stage10_segment_scales(pipe, dep_sample, rgb_sample, device, report):
    """Latent scale of ALL FOUR segments, for both arms.

    S9 shows depth latents are ~2x wider than RGB latents. Whether that matters depends on
    what the model already copes with: the official <video><action> layout is ALSO
    heterogeneous, because an action image (near-black with a few gaussian blobs) is nothing
    like natural video. If the action segments already sit far from the RGB segments, then
    depth's offset is within the range the pretrained prior handles. If instead all official
    segments cluster tightly and only depth is an outlier, the depth arm starts off-manifold
    and the two arms experience different effective SNR at the same diffusion timestep --
    which would be a confound in the comparison, not just a slower start.
    """
    T = NUM_FRAMES
    a7 = dep_sample["action_7d"].unsqueeze(0)
    a5 = project_actions_7d_to_5d_torch_batch(
        a7.repeat(1, 2, 1), dep_sample["extrinsics"].unsqueeze(0),
        dep_sample["intrinsics"].unsqueeze(0))
    av = project_action_5d_to_rgb_torch(a5, RES, RES).permute(0, 4, 1, 2, 3) * 2 - 1

    def enc(seg):
        x = seg.unsqueeze(0).to(device=device, dtype=pipe.torch_dtype)
        with torch.no_grad():
            return pipe.encode_video(x)[0].float().cpu()

    segs = {
        "v1_rgb": enc(rgb_sample["video"][:, :T]),
        "v2_rgb": enc(rgb_sample["video"][:, T:]),
        "v1_depth": enc(dep_sample["video"][:, :T]),
        "v2_depth": enc(dep_sample["video"][:, T:]),
        "v1_action": enc(av[0][:, :T]),
        "v2_action": enc(av[0][:, T:]),
    }
    stats = {k: {"mean": float(v.mean()), "std": float(v.std()),
                 "p99_abs": float(v.abs().flatten().kthvalue(int(0.99 * v.numel()))[0])}
             for k, v in segs.items()}

    fig, ax = plt.subplots(figsize=(9, 4.6))
    order = ["v1_rgb", "v2_rgb", "v1_depth", "v2_depth", "v1_action", "v2_action"]
    colors = ["tab:orange", "tab:orange", "tab:blue", "tab:blue", "tab:gray", "tab:gray"]
    ax.bar(range(len(order)), [stats[k]["std"] for k in order], color=colors, alpha=0.85)
    for i, k in enumerate(order):
        ax.text(i, stats[k]["std"] + 0.04, f"{stats[k]['std']:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=20)
    ax.set_ylabel("latent std")
    ax.set_title("S10  Per-segment latent scale.\n"
                 "orange = official video arm's visual segments, grey = action segments "
                 "(present in BOTH arms), blue = depth arm's visual segments.")
    savefig(fig, "S10_segment_latent_scales.png")
    report["S10_segment_scales"] = stats
    print(f"  S10 segment scales: "
          f"{ {k: round(v['std'], 3) for k, v in stats.items()} }")


# ======================================================================================
# Template-era stages. S1-S10 predate task templates and still describe the pieces
# (codec, action rendering, latent scale). What they cannot show is the thing the template
# design actually introduced: WHICH modalities are in the sequence and WHICH of them are
# given. Those are the two axes the whole study turns on, and until S11 nothing drew them.
# ======================================================================================
TEMPLATE_GALLERY = [
    ("video+action", "official recipe -- the control arm"),
    ("video+depth", "PERCEPTION: RGB given -> depth predicted"),
    ("video+segmentation", "PERCEPTION: RGB given -> referring seg predicted"),
    ("video+depth+action", "CO-GENERATION: 6 segments, the previously unreachable cell"),
    ("depth+action", "WORLD MODEL: no RGB anywhere -- NOT perception"),
]


def _sample_for(ds, index, template, seed=7):
    random.seed(seed)
    return ds.getitem(index, force_template=template)


def _action_pixels(sample):
    """The action stream exactly as ActionImagesModel.forward renders it. [2T,H,W,3] uint8."""
    T = NUM_FRAMES
    a7 = sample["action_7d"].unsqueeze(0)
    num_views = sample["video"].shape[1] // T
    a5 = project_actions_7d_to_5d_torch_batch(
        a7.repeat(1, num_views, 1),
        sample["extrinsics"].unsqueeze(0),
        sample["intrinsics"].unsqueeze(0),
    )
    av = project_action_5d_to_rgb_torch(a5, RES, RES).permute(0, 4, 1, 2, 3) * 2 - 1
    return to_uint8(av[0])


def stage11_template_gallery(ds, index, report):
    """One row per segment, for every template: what the DiT sees and what it must predict.

    The green/red strip on the right is the condition mask straight out of plan_segments +
    assemble -- not a redrawing of it. That matters: a mask that is drawn by hand next to a
    layout built by code can agree with the picture and disagree with training.
    """
    from training.templates import assemble, parse_template, plan_segments

    T = NUM_FRAMES
    T_l = (T - 1) // 4 + 1
    gallery = {}

    for template, blurb in TEMPLATE_GALLERY:
        sample = _sample_for(ds, index, template)
        if sample["template"] != template:
            print(f"  !! {template} degraded to {sample['template']} on index {index}; skipping")
            continue
        mods = parse_template(sample["template"])
        px = {k: to_uint8(v) for k, v in sample["streams"].items()}
        if "action" in mods:
            px["action"] = _action_pixels(sample)

        # Build the REAL plan/mask with a dummy latent set of the right shape.
        dummy = {m: [torch.zeros(1, 4, T_l, 2, 2) for _ in range(2)] for m in mods}
        cam = [torch.zeros(1, T_l, 4, 2, 2) for _ in range(2)]
        rng = random.Random(0)
        plan = plan_segments(mods, num_views=2, rng=rng, is_rlbench=True)
        _, _, masks = assemble(plan, dummy, cam)
        given = masks[0, 0, :, 0, 0].tolist()

        n_seg = len(plan)
        show = picks(T, 4)
        fig, axes = plt.subplots(n_seg, 5, figsize=(15, 2.9 * n_seg + 1.2))
        if n_seg == 1:
            axes = axes[None, :]
        off = 0
        for r, seg in enumerate(plan):
            half = slice(seg.view * T, (seg.view + 1) * T)
            frames_px = px[seg.modality][half]
            seg_len = 1 if seg.single_frame else T_l
            seg_given = given[off:off + seg_len]
            off += seg_len
            for c, f in enumerate(show):
                axes[r, c].imshow(frames_px[f]); axes[r, c].axis("off")
                # latent frame index this pixel frame lands in (VAE compresses 4x in time)
                lf = 0 if f == 0 else min(seg_len - 1, (f - 1) // 4 + 1)
                is_given = bool(seg_given[lf])
                axes[r, c].set_title(
                    f"t={f}  " + ("GIVEN" if is_given else "predict"),
                    fontsize=8, color=("tab:green" if is_given else "tab:red"))
            strip = np.array(seg_given, dtype=float)[None, :]
            axes[r, 4].imshow(strip, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
            axes[r, 4].set_title(f"mask, {seg_len} latent frames", fontsize=8)
            axes[r, 4].set_yticks([]); axes[r, 4].set_xticks(range(seg_len))
            axes[r, 4].set_xticklabels(range(seg_len), fontsize=6)
            axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
            axes[r, 0].set_ylabel(f"seg {r}\nview{seg.view + 1} {seg.modality}", fontsize=9)

        n_given = int(sum(given))
        fig.suptitle(
            f"S11  template = {template}\n{blurb}\n"
            f"prompt: {sample['text'][:110]}\n"
            f"{n_seg} segments, {len(given)} latent frames, {n_given} given / "
            f"{len(given) - n_given} predicted",
            fontsize=11)
        savefig(fig, f"S11_template_{template.replace('+', '_')}.png")
        gallery[template] = {
            "segments": [f"v{s.view + 1}_{s.modality}" for s in plan],
            "latent_frames": len(given),
            "given": n_given,
            "predicted": len(given) - n_given,
            "prompt": sample["text"],
            "streams": sorted(sample["streams"]),
        }
    report["S11_templates"] = gallery
    return gallery


def stage12_seg_chain(ds, index, report):
    """The segmentation arm end to end: GT handle map -> known-colour RGB -> decode -> IoU.

    Never visualised before. FORK_CHANGES §5 records that the seg branch was written and unit
    tested but never run, so every failure mode here is unobserved: a colour map that does not
    match the prompt, a target that is empty from one viewpoint, a palette colour the decoder
    rounds to its neighbour.
    """
    from training.percep.seg_codec import build_referring_spec

    sample = _sample_for(ds, index, "video+segmentation")
    if "segmentation" not in sample["streams"]:
        print(f"  !! index {index} resolves no seg target; skipping S12")
        return None
    frames = sample["frame_indices"]
    views = [sample["view_dirs"][i] for i in sample["view_indices"]]
    T, fi = NUM_FRAMES, picks(NUM_FRAMES)

    instruction = sample["text"].split("> ", 1)[-1]
    seg_targets = json.load(open(os.path.join(sample["path"], "seg_targets.json")))
    id_groups, color_map, _ = build_referring_spec(seg_targets, instruction)

    seg_u8 = to_uint8(sample["streams"]["segmentation"])
    rgb_u8 = to_uint8(sample["streams"]["video"])

    ious, empty = {}, 0
    fig, axes = plt.subplots(4, len(fi) * 2, figsize=(2.5 * len(fi) * 2, 10.4))
    for half, (name, offset) in enumerate([("cond", 0), ("target", T)]):
        view = views[half]
        mask_map = np.load(os.path.join(view, "mask.npz"))["mask"][frames]
        dec = decode_known_color(seg_u8[offset:offset + T], color_map)
        for c, f in enumerate(fi):
            col = half * len(fi) + c
            axes[0, col].imshow(rgb_u8[offset + f]); axes[0, col].axis("off")
            axes[0, col].set_title(f"{name} {os.path.basename(view)} t={f}", fontsize=8)
            axes[1, col].imshow(mask_map[f], cmap="tab20"); axes[1, col].axis("off")
            axes[2, col].imshow(seg_u8[offset + f]); axes[2, col].axis("off")
            overlay = rgb_u8[offset + f].astype(np.float32).copy()
            paint = seg_u8[offset + f].sum(-1) > 30
            overlay[paint] = 0.35 * overlay[paint] + 0.65 * seg_u8[offset + f][paint]
            axes[3, col].imshow(overlay.astype(np.uint8)); axes[3, col].axis("off")
        for inst, ids in id_groups.items():
            gt = handles_to_mask(mask_map, ids)
            pred = dec[inst]
            if gt.sum() == 0:
                empty += 1
                continue
            inter = float((gt & pred).sum()); union = float((gt | pred).sum())
            ious[f"{name}/{inst}"] = inter / union if union else 1.0
    for r, lab in enumerate(["natural RGB", "GT handle map", "known-colour encode",
                             "encode over RGB"]):
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(lab, fontsize=9)
    worst = min(ious.values()) if ious else float("nan")
    fig.suptitle(
        f"S12  segmentation chain. colour map = {color_map}\n"
        f"prompt: {sample['text'][:110]}\n"
        f"roundtrip IoU worst={worst:.4f} over {len(ious)} instance-views "
        f"({empty} not visible from that viewpoint -> skipped)", fontsize=11)
    savefig(fig, "S12_seg_chain.png")
    report["S12_seg"] = {"color_map": color_map, "iou": ious, "worst_iou": worst,
                         "not_visible": empty, "prompt": sample["text"]}
    print(f"  S12 seg worst IoU={worst:.4f} over {len(ious)} instance-views, {empty} not visible")
    return sample


def stage13_mode_mix(report):
    """The four given/predicted patterns a video+action sample can draw, side by side.

    Same template, same pixels -- only the mask differs. This is Action-Images' own task
    switch (paper §3.3: i2va / a2v / v2a / video-only), and it is orthogonal to the prompt.
    Drawing it makes clear why the given/predict axis is NOT part of the tag language.
    """
    from training.templates import assemble, parse_template, plan_segments

    T_l = (NUM_FRAMES - 1) // 4 + 1
    mods = parse_template("video+action")
    dummy = {m: [torch.zeros(1, 4, T_l, 2, 2) for _ in range(2)] for m in mods}
    cam = [torch.zeros(1, T_l, 4, 2, 2) for _ in range(2)]

    class Scripted:
        def __init__(self, vals): self.v = list(vals); self.i = 0
        def random(self):
            x = self.v[self.i]; self.i += 1; return x

    cases = [
        ("joint (p<0.90, ~81% of steps)\ni2va: first frames -> everything", [0.5, 0.5]),
        ("first segment given (0.90<=p<0.95)\nview1 video known", [0.5, 0.92]),
        ("v2a (p>=0.95)\nboth videos known -> infer action", [0.5, 0.99]),
        ("single-frame variant (10%, rlbench only)\npolicy mode: video collapsed to 1 frame",
         [0.05]),
    ]
    fig, axes = plt.subplots(len(cases), 1, figsize=(13, 2.5 * len(cases)))
    out = {}
    for ax, (label, script) in zip(axes, cases):
        plan = plan_segments(mods, num_views=2, rng=Scripted(script), is_rlbench=True)
        _, _, masks = assemble(plan, dummy, cam)
        given = np.array(masks[0, 0, :, 0, 0].tolist(), dtype=float)
        ax.imshow(given[None, :], cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks([])
        ax.set_xticks(range(len(given)))
        ax.set_xticklabels(range(len(given)), fontsize=6)
        bound = 0
        for seg in plan:
            n = 1 if seg.single_frame else T_l
            ax.axvline(bound - 0.5, color="black", lw=2)
            ax.text(bound + n / 2 - 0.5, -0.62, f"v{seg.view + 1}_{seg.modality}",
                    ha="center", fontsize=8)
            bound += n
        ax.set_title(f"{label}   ->  {int(given.sum())}/{len(given)} latent frames given",
                     fontsize=9, loc="left")
        out[label.split("\n")[0]] = {"given": int(given.sum()), "total": int(len(given))}
    fig.suptitle("S13  video+action: the SAME prompt, four different masks.\n"
                 "green = given (kept clean), red = denoised from noise. The prompt picks the "
                 "modalities; the mask picks the task.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, "S13_mode_mix_masks.png")
    report["S13_mode_mix"] = out


def stage14_seg_vae(pipe, seg_sample, device, report):
    """Does the VAE preserve the seg palette? The open risk for the seg arm, same as S5's.

    The palette is eight saturated colours on pure black -- far from natural video statistics.
    If the VAE rounds them together, decode_known_color assigns pixels to the wrong instance
    and the seg arm cannot work regardless of how correct the data plumbing is. Every test in
    tests/ would still pass, because they all stop at the codec.
    """
    from training.percep.seg_codec import build_referring_spec

    T = NUM_FRAMES
    instruction = seg_sample["text"].split("> ", 1)[-1]
    seg_targets = json.load(open(os.path.join(seg_sample["path"], "seg_targets.json")))
    id_groups, color_map, _ = build_referring_spec(seg_targets, instruction)

    rows, stats = [], {}
    for name, offset, vi in (("cond", 0, 0), ("target", T, 1)):
        seg = seg_sample["streams"]["segmentation"][:, offset:offset + T]
        rec, _ = vae_roundtrip(pipe, seg, device)
        pre_u8, post_u8 = to_uint8(seg), to_uint8(rec)
        view = seg_sample["view_dirs"][seg_sample["view_indices"][vi]]
        mask_map = np.load(os.path.join(view, "mask.npz"))["mask"][seg_sample["frame_indices"]]
        dec_pre = decode_known_color(pre_u8, color_map)
        dec_post = decode_known_color(post_u8, color_map)
        for inst, ids in id_groups.items():
            gt = handles_to_mask(mask_map, ids)
            if gt.sum() == 0:
                continue
            for tag, dec in (("codec_only", dec_pre), ("after_vae", dec_post)):
                inter = float((gt & dec[inst]).sum()); union = float((gt | dec[inst]).sum())
                stats[f"{name}/{inst}/{tag}"] = inter / union if union else 1.0
        rows.append((name, pre_u8, post_u8))

    fi = picks(T)
    fig, axes = plt.subplots(3, len(fi) * 2, figsize=(2.5 * len(fi) * 2, 7.8))
    for half, (name, pre_u8, post_u8) in enumerate(rows):
        for c, f in enumerate(fi):
            col = half * len(fi) + c
            axes[0, col].imshow(pre_u8[f]); axes[0, col].axis("off")
            axes[0, col].set_title(f"{name} t={f}", fontsize=8)
            axes[1, col].imshow(post_u8[f]); axes[1, col].axis("off")
            diff = np.abs(pre_u8[f].astype(int) - post_u8[f].astype(int)).sum(-1)
            im = axes[2, col].imshow(diff, cmap="magma", vmin=0, vmax=120)
            axes[2, col].axis("off")
    for r, lab in enumerate(["encode (codec only)", "after VAE roundtrip", "|difference|"]):
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(lab, fontsize=9)
    pre_ious = [v for k, v in stats.items() if k.endswith("codec_only")]
    post_ious = [v for k, v in stats.items() if k.endswith("after_vae")]
    fig.suptitle(
        f"S14  seg through the VAE -- the risk S5 measured for depth, never measured for seg.\n"
        f"IoU codec-only min={min(pre_ious) if pre_ious else float('nan'):.4f}  ->  "
        f"after VAE min={min(post_ious) if post_ious else float('nan'):.4f}", fontsize=11)
    savefig(fig, "S14_seg_vae_roundtrip.png")
    report["S14_seg_vae"] = stats
    print(f"  S14 seg IoU codec-only min={min(pre_ious) if pre_ious else float('nan'):.4f}, "
          f"after VAE min={min(post_ious) if post_ious else float('nan'):.4f}")


def stage15_cost(gallery, report):
    """Sequence length and attention cost per template -- the numbers doc 16 §3 claims."""
    if not gallery:
        return
    H_l = RES // 16          # Wan2.2 VAE upsampling_factor
    tok_per_lat = (H_l // 2) ** 2   # DiT patch_size (1,2,2)
    names = list(gallery)
    toks = [gallery[n]["latent_frames"] * tok_per_lat for n in names]
    base = toks[names.index("video+action")] if "video+action" in names else toks[0]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    colors = ["tab:gray" if n == "video+action" else
              ("tab:red" if len(n.split("+")) > 2 else "tab:blue") for n in names]
    ax.bar(range(len(names)), toks, color=colors, alpha=0.85)
    for i, (n, t) in enumerate(zip(names, toks)):
        ax.text(i, t + 60, f"{t}\n{t / base:.2f}x seq\n{(t / base) ** 2:.2f}x attn",
                ha="center", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, fontsize=9)
    ax.set_ylabel(f"DiT tokens ({RES}^2, {NUM_FRAMES} frames, 2 views)")
    ax.set_ylim(0, max(toks) * 1.28)
    ax.set_title("S15  Cost per template. The perception templates are FREE relative to the\n"
                 "official recipe -- depth takes the action slot rather than being appended.")
    savefig(fig, "S15_template_cost.png")
    report["S15_cost"] = {n: {"tokens": t, "rel_seq": round(t / base, 3),
                              "rel_attn": round((t / base) ** 2, 3)}
                          for n, t in zip(names, toks)}
    print(f"  S15 tokens: { {n: t for n, t in zip(names, toks)} }")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--no-vae", action="store_true", help="skip GPU stages S5/S6/S9/S14")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    report = {"index": args.index}
    ds = RLBenchSelfgenDataset(base_path=os.path.join(REPO, "data", "rlbench_selfgen"),
                               num_frames=NUM_FRAMES, frame_interval=1, height=RES, width=RES,
                               template_mix="depth+action@1.0", strict_getitem=True)

    print("S1/S2/S3  GT -> codec -> dataset tensor")
    dep, rgb, views, frames = stage1_2_3(ds, args.index, report)
    print("S4  codec-only roundtrip")
    stage4_codec_only(views, frames, report)
    print("S7/S8  action stream + assembled sequence")
    stage7_8_action_and_layout(dep, "cpu", report)

    # Template-era stages. These are the ones that describe the CURRENT design.
    print("S11  per-template layout + condition mask")
    gallery = stage11_template_gallery(ds, args.index, report)
    print("S12  segmentation chain (GT -> known-colour -> decode -> IoU)")
    seg_sample = stage12_seg_chain(ds, args.index, report)
    print("S13  mode-mix masks for video+action")
    stage13_mode_mix(report)
    print("S15  per-template sequence length / attention cost")
    stage15_cost(gallery, report)

    if not args.no_vae:
        device = "cuda"
        print("S5/S6/S9  loading VAE ...")
        pipe = load_vae(device)
        stage5_6_vae(pipe, dep, rgb, views, frames, device, report)
        stage9_latents(pipe, dep, rgb, device, report)
        stage10_segment_scales(pipe, dep, rgb, device, report)
        if seg_sample is not None:
            print("S14  seg through the VAE")
            stage14_seg_vae(pipe, seg_sample, device, report)

    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport -> {os.path.relpath(os.path.join(OUT, 'report.json'), REPO)}")


if __name__ == "__main__":
    main()
