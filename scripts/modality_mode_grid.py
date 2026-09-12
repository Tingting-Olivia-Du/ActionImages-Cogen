"""Every modality x every conditioning mode, from one checkpoint, in one model load.

    python scripts/modality_mode_grid.py --ckpt .../step1500.ckpt --tag arm6_1500 --gpu 6

Answers two questions the single-mode demo cannot:

  1. does each TAG still select its own modality, and
  2. does each CONDITIONING MODE the training mix actually produces still work at inference.

The modes are not decoration. `plan_segments` draws them during training with fixed
probabilities, so a mode that is trained but never checked is a blind spot:

    templates containing <action>   iiii 0.81 | fiii 0.045 | fifi 0.045 | policy 0.10
    perception templates (M2)       iiii 0.90 | fiii 0.05  | fifi 0.05

  iiii    every segment keeps its first latent frame. The rollout regime.
  fiii    segment 0 given whole, the rest keep their first frame.
  fifi    the anchor (video) given whole; the target keeps its first frame. The easy bound.
  policy  visual segments collapse to ONE latent frame while action stays full length --
          rlbench-only, 10% of action templates. It is the reason this script cannot slice the
          returned frames into equal quarters: segment lengths differ within one sequence.
  f0f0    NOT a training mode. Video fully given, target gets NO anchor, so the prompt is the
          only thing naming the modality. Included as the tag-control probe.

SEGMENT SLICING. The pipeline concatenates all decoded segments and returns a flat array; it
does not return the spans. They are recomputed here from the same rule
`prepare_template_inference_latents` uses (view-major, modality-minor in CANONICAL_ORDER,
`single_frame` only under policy), and the total is asserted against the array actually
returned -- if the two ever disagree, every metric below would be silently measuring the wrong
pixels.
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
    decode_scene_roles, scene_role_iou, scene_role_labels,
)
from training.templates import CANONICAL_ORDER, VISUAL_MODALITIES, inference_plan, parse_template  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# (template, modes). `policy` needs an action segment; `video` alone has no target to withhold,
# so f0f0 is meaningless for it.
# f0f0 is deliberately absent. It is not a training mode, and measured on
# arm6/checkpoint-1500 all four modalities collapse to the same noisy RGB under it
# (depth absrel 0.087 -> 0.816, seg mIoU 0.386 -> 0.059 with `unknown` pixels up 1300x,
# normal cos 0.896 -> -0.291). The conclusion -- the anchor frame, not the tag, is what
# currently selects the modality -- is established; re-measuring it every run buys nothing.
GRID = [
    ("video+action",       ["iiii", "fiii", "fifi", "policy"]),
    ("video+depth",        ["iiii", "fiii", "fifi"]),
    ("video+segmentation", ["iiii", "fiii", "fifi"]),
    ("video+normal",       ["iiii", "fiii", "fifi"]),
    # `video` alone: no fifi. video IS the anchor, so fifi gives every segment in full and the
    # sequence has nothing left to predict -- assemble() rightly refuses it
    # ("every latent position is a condition frame"). Not a bug, an impossible request.
    ("video",              ["iiii", "fiii"]),
    # --- arm7 的顶替族。与上面的 video+X 是【不同的任务】,不要混着读: ---
    # video+X   : RGB 给定 -> X 预测        = 感知(可与 DA2/Marigold 等外部基线比)
    # X+action  : 一帧 X   -> 未来 X + action = X 空间的世界模型(外部基线没有对应物)
    #
    # 而且 FIFI 在这一族里【给定】anchor 段而不是生成它(anchor = 该族唯一的视觉模态),
    # 所以 fifi 下 X 段的指标量到的是 VAE 往返地板(实测 depth 0.36% AbsRel),不是生成质量。
    # 保留 fifi 是因为它对 ACTION 段仍然有意义 —— 那是 v2a(完整观测 -> 动作)的上界读数。
    ("depth+action",         ["iiii", "fifi"]),
    ("segmentation+action",  ["iiii", "fifi"]),
    ("normal+action",        ["iiii", "fifi"]),
    # --- the 10-segment fusion canvas (V0 D0 S0 N0 A0 | V1 D1 S1 N1 A1) ---------------------
    # FOUR DIFFERENT QUESTIONS, not one metric at four settings:
    #   full_anchor : every modality gets its own first frame -> joint 5-way video prediction.
    #                 The only row comparable to arm6/arm7's `iiii` numbers.
    #   rgb_only    : ONE RGB frame, nothing else -> every other modality generated from scratch.
    #                 THE DEPLOYMENT REGIME, and the row this arm exists to win.
    #   rgb_given   : whole RGB video -> the other four. The only row comparable to an external
    #                 depth/normal estimator, which also sees full RGB.
    #   policy      : the replan canvas eval/policy.py actually presents (30 latent frames).
    # `one_out` is absent because it names a random DRAW, not a conditioning.
    ("video+depth+segmentation+normal+action",
                             ["full_anchor", "rgb_only", "rgb_given", "policy"]),
]


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--gpu", default="auto",   # physical index; overrides CUDA_VISIBLE_DEVICES
                   help="'auto' picks the emptiest card with enough free memory and waits if "
                        "none has it; an explicit index skips the check")
    p.add_argument("--need_mib", type=int, default=30000,
                   help="free memory required before loading. The 5B DiT + T5 + VAE settle at "
                        "~28 GB at 512; loading blind is how three probe runs were lost to OOM "
                        "PART-WAY THROUGH model load, after the episode list was already fixed.")
    p.add_argument("--wait_min", type=int, default=0,
                   help="minutes to keep waiting for a card with --need_mib free. 0 = fail fast.")
    p.add_argument("--claim_gpu", default="",
                   help="index of a card to TAKE OVER. CUDA is initialised on it first (a context "
                        "costs ~430 MiB and succeeds even while the card is full), then --kill_first "
                        "runs, then the memory is claimed in a tight retry loop. Skips pick_gpu.")
    p.add_argument("--kill_first", default="",
                   help="pgrep -f pattern to SIGKILL once CUDA is warm, immediately before "
                        "claiming. This is how the claim wins: torch is already loaded, so the "
                        "gap between the card being freed and being taken is milliseconds "
                        "instead of the ~10 s a cold process needs.")
    p.add_argument("--claim_timeout_s", type=int, default=180)
    p.add_argument("--episode", default="open_drawer/variation0/episodes/episode0",
                   help="ONE episode, or a comma list. A single episode gives no confidence "
                        "interval, and this repository has a precedent (ARM7_RESULTS.md) of a "
                        "headline ratio reversing once it was properly powered -- so prefer a "
                        "list. Episodes are the OUTERMOST loop and metrics.json is rewritten "
                        "after every one, so an interrupted run still yields a usable n rather "
                        "than nothing. Shorthand: 'tasks:N' = episode0 of the first N tasks.")
    p.add_argument("--data", default=str(REPO / "data" / "rlbench_selfgen_512_aug"))
    p.add_argument("--out", default=str(REPO / "reports" / "modality_mode_grid"))
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--num_frames", type=int, default=41)
    p.add_argument("--frame_interval", type=int, default=3)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--segmentation_mode", default="scene_roles", choices=("referring", "scene_roles"))
    p.add_argument("--legacy_scene_seg_tag", action="store_true")
    p.add_argument("--only", default="", help="comma list of templates to restrict to")
    p.add_argument("--withhold", default="",
                   help="comma list of modalities to run ONE-OUT on, e.g. "
                        "'depth,normal,segmentation,video,action'. For each, the modality is "
                        "marked ABSENT (no condition frame at all) while every other segment "
                        "keeps its anchor -- the F axis's `one_out` regime. It is a separate "
                        "flag rather than a GRID mode because `one_out` names a random DRAW; "
                        "only the caller knows which modality it wants withheld. Rows appear as "
                        "'<template>|out:<modality>'. Combined with the grid's `full_anchor` "
                        "(nothing withheld) and `rgb_only` (everything but video withheld), this "
                        "gives the mutual-completion reading: does a modality get better as more "
                        "of the others are present?")
    p.add_argument("--modes", default="",
                   help="comma list of conditioning modes, overriding GRID's per-template list. "
                        "The reason this exists: f0f0 is absent from GRID by default (not a "
                        "training mode, and re-measuring it every run buys nothing), but the "
                        "tag-swap ablation needs it AT THE SAME CHECKPOINT to state how much "
                        "the anchor is worth relative to the tag. Comparing f0f0 at step 1500 "
                        "against a tag swap at step 9000 measures two different models.")
    return p.parse_args()


def seg_plan(template, num_views, mode, absent=None):
    """-> [(modality, view, single_frame, fully_given, absent)], in packing order.

    DERIVED from templates.inference_plan -- the same function the pipeline builds its mask from
    -- rather than transcribed. The previous version was a hand-written mirror whose docstring
    said "Mirrors prepare_template_inference_latents exactly"; it went stale the moment the F
    axis added a third role (`absent`), and a stale mirror here does not crash, it mislabels
    which decoded span belongs to which modality.

    `fully_given` still matters for READING the numbers: a segment handed over whole is a VAE
    roundtrip of ground truth, so its PSNR is a floor (~30.7 dB here) and must never be reported
    as generation quality. `absent` is the opposite extreme -- nothing given at all, so the span
    is fully generated, which is the deployment-relevant reading.
    """
    plan = inference_plan(parse_template(template), num_views, mode=mode, absent=absent)
    return [(sg.modality, sg.view, sg.single_frame, sg.fully_given, sg.absent) for sg in plan]


def spans_for(plan, T):
    """Decoded pixel spans per segment. A single_frame segment is ONE latent frame, which the
    4x-temporal VAE decodes to one pixel frame; a full segment is T."""
    out, off = [], 0
    for _, _, single, _given, _absent in plan:
        n = 1 if single else T
        out.append((off, off + n))
        off += n
    return out, off


def free_mib():
    """[(index, free MiB)] straight from nvidia-smi. memory.used, never utilization: a tenant
    holding 45 GB at 0% util still owns the card."""
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    rows = []
    for line in out.strip().splitlines():
        i, used, total = (int(x) for x in line.split(","))
        rows.append((i, total - used))
    return rows


def claim_gpu(a):
    """Warm CUDA, kill the incumbent, then take the memory before anyone else can.

    Two handoffs were lost to the same race: the caller confirmed a card was free, launched a
    cold python process, and by the time torch had imported (~10 s) an outside tenant had taken
    39 GB. The check was never the problem -- the ~10 s of process startup between the check and
    the allocation was.

    The fix is to pay that cost BEFORE the card is free. A CUDA context on a full card costs
    ~430 MiB and is granted anyway (observed: our own doomed run held exactly that while a
    tenant held 39 GB). So initialise here, kill the trainer, and retry the big allocation on a
    0.2 s cadence -- the window shrinks from ten seconds to milliseconds.
    """
    import subprocess
    import torch
    torch.cuda.init()
    torch.zeros(1, device="cuda")          # force a real context now, not at first use
    print(f"[grid] CUDA warm on the target card ({torch.cuda.memory_reserved()//(1024*1024)} MiB ctx)",
          flush=True)

    if a.kill_first:
        pids = subprocess.run(["pgrep", "-f", a.kill_first], capture_output=True, text=True).stdout.split()
        me = str(os.getpid())
        pids = [p for p in pids if p != me]
        print(f"[grid] killing incumbent [{a.kill_first}]: {pids}", flush=True)
        for p in pids:
            subprocess.run(["kill", "-9", p], capture_output=True)

    want = int(a.need_mib * 0.90) * 1024 * 1024
    deadline = time.time() + a.claim_timeout_s
    while True:
        try:
            block = torch.empty(want // 2, dtype=torch.float16, device="cuda")
            del block                      # stays in PyTorch's pool, not returned to the driver
            print(f"[grid] claimed {torch.cuda.memory_reserved()//(1024*1024)} MiB", flush=True)
            return
        except RuntimeError:
            if time.time() > deadline:
                sys.exit(f"[grid] could not claim {a.need_mib} MiB within {a.claim_timeout_s}s")
            time.sleep(0.2)


def reserve_vram(need_mib, margin=0.90):
    """Claim the card NOW, before the ~40 s of imports and weight loading.

    Checking that a card is free and then loading a 12.8 GB checkpoint into it are separated by
    long enough for someone else to take it -- measured: a handoff verified GPUs 1 and 2 free at
    18:08:34, launched immediately, and both runs died at `pipe.to(device)` because a tenant had
    claimed 39 GB in between. A gate alone cannot close a window it opens.

    So allocate the memory first, then release the tensor. PyTorch's caching allocator does NOT
    hand freed blocks back to the driver (only `empty_cache()` does), so the pool stays ours and
    the model load draws from it. From the driver's point of view the card is occupied from the
    first second.
    """
    import torch
    if not torch.cuda.is_available():
        return
    want = int(need_mib * margin) * 1024 * 1024
    try:
        block = torch.empty(want // 2, dtype=torch.float16, device="cuda")
    except RuntimeError as exc:
        sys.exit(f"[grid] could not reserve {want // (1024*1024)} MiB on the chosen card: {exc}")
    del block                       # freed to PyTorch's pool, NOT back to the driver
    got = torch.cuda.memory_reserved() // (1024 * 1024)
    print(f"[grid] reserved {got} MiB up front (survives the model load)", flush=True)


def pick_gpu(a):
    """The emptiest card with `need_mib` free, waiting up to `wait_min` for one to appear."""
    deadline = time.time() + a.wait_min * 60
    while True:
        cand = sorted(free_mib(), key=lambda r: -r[1])
        if cand and cand[0][1] >= a.need_mib:
            return str(cand[0][0])
        msg = "  ".join(f"gpu{i}:{f}" for i, f in sorted(cand))
        if time.time() >= deadline:
            sys.exit(f"[grid] no GPU has {a.need_mib} MiB free (need ~28 GB for the 5B model at "
                     f"512). Free now: {msg}. Re-run with --wait_min N to queue.")
        print(f"[grid] waiting for {a.need_mib} MiB free ({msg})", flush=True)
        time.sleep(120)


def _seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _u8(stream):
    return np.clip(np.round((stream.permute(1, 2, 3, 0).numpy() + 1.0) * 127.5), 0, 255).astype(np.uint8)


def score(modality, ds, sample, pred, res):
    """pred: [T,H,W,3] uint8 for ONE view's target segment."""
    vd, fi = sample["_view_dir"], sample["frame_indices"]
    if modality == "depth":
        dec = decode_depth(pred)
        gt = ds._to_model_res(ds._load_depth(vd)[fi].astype(np.float32))[: len(pred)]
        ok = np.isfinite(gt) & (gt > MIN_VALID) & (gt < MAX_VALID) & np.isfinite(dec)
        rel = np.abs(np.clip(dec, MIN_VALID, MAX_VALID) - gt) / np.maximum(gt, MIN_VALID)
        # Per-frame as well as pooled. Under IIII the model generates the frame it then
        # interprets, so a divergence between its predicted future and the real one is
        # charged to this metric with no correspondence step. Frame 0 is the given anchor,
        # so if the pooled error is drift rather than bad geometry it has to grow with t --
        # and stay flat under FIFI, where the real frame is supplied at every segment.
        per_frame = [round(float(rel[t][ok[t]].mean()), 5) if ok[t].any() else None
                     for t in range(len(rel))]
        # Fraction of pixels the codec could decode at all. A depth image is a colour path;
        # pixels whose colour leaves that path decode to NaN and are dropped from the mean.
        # A frame where the model wandered off the path therefore scores on a shrinking and
        # unrepresentative subset, which looks like a sudden jump in AbsRel rather than a
        # gradual degradation -- so the coverage has to be recorded next to the error.
        gt_ok = np.isfinite(gt) & (gt > MIN_VALID) & (gt < MAX_VALID)
        cov = [round(float(ok[t].sum() / max(gt_ok[t].sum(), 1)), 5) for t in range(len(rel))]
        return {"metric": "absrel",
                "value": round(float(rel[ok].mean()), 5) if ok.any() else None,
                "absrel_per_frame": per_frame,
                "decode_coverage_per_frame": cov,
                "decode_coverage": round(float(np.mean(cov)), 5)}
    if modality == "normal":
        fx, fy = ds._focal_lengths(vd)
        raw = ds._load_depth(vd)[fi].astype(np.float32)
        sx, sy = res / raw.shape[-1], res / raw.shape[-2]
        d = ds._to_model_res(raw)[: len(pred)]
        gt_n = np.stack([depth_to_normal(d[t], fx * sx, fy * sy) for t in range(len(d))])
        return {"metric": "mean_cos", "value": round(float((gt_n * decode_normal(pred)).sum(-1).mean()), 5)}
    if modality == "segmentation":
        instr = sample["text"].split("> ", 1)[-1]
        lut, present = ds._scene_role_lut(sample["path"], instr)
        hm = ds._to_model_res(ds._load_mask(vd)[fi]).astype(np.uint16)[: len(pred)]
        per = scene_role_iou(decode_scene_roles(pred, present_roles=present),
                             scene_role_labels(hm, lut), roles=present)
        scored = [m["iou"] for m in per.values() if m["gt_px"] > 0]
        # Per-role IoU alongside the macro average. Macro-mIoU is NOT comparable across tasks
        # with different role counts -- close_box has two task objects against a dozen in the
        # trained tasks, so fewer roles each covering more pixels inflates the mean. Keeping the
        # per-role numbers lets a reader restrict the comparison to roles both sides actually
        # have (robot_arm, gripper, background, ...), which is comparable.
        return {"metric": "macro_miou",
                "value": round(float(np.mean(scored)), 5) if scored else None,
                "n_roles": len(scored),
                "per_role_iou": {r: round(float(m["iou"]), 5)
                                 for r, m in per.items() if m["gt_px"] > 0},
                "unknown_px": int(decode_scene_roles(pred)["unknown"].sum())}
    return {"metric": None, "value": None}


def score_action(pred_v0, pred_v1, sample, T):
    """Decode the two action heatmaps back to a 6-DoF pose and compare with the GT trajectory.

    REPLACES r_peak as the headline number. r_peak is `heat[...,0].amax()`, the red-channel
    peak, and ANY rgb image saturates it: measured on arm6/checkpoint-1500 the f0f0 run scored
    r_peak 255.0 -- the best of all five modes -- while the picture was plainly noisy RGB and
    not an action image at all. It was the contact sheet that caught that, not the metric.
    r_peak is kept alongside, because "stopped drawing blobs" and "drew them in the wrong
    place" are genuinely different failures, but it can no longer be read as correctness.

    `sparsity` is the cheap discriminator r_peak lacks: an action image is mostly black with a
    few Gaussian blobs, an RGB frame is not.
    """
    from training.utils import fuse_multiview_heatmaps_to_pose_torch
    from scipy.spatial.transform import Rotation as Rot

    n = min(len(pred_v0), len(pred_v1), T)
    heat = torch.from_numpy(np.stack([pred_v0[:n], pred_v1[:n]], axis=1)).float()   # [n,V,H,W,3]
    ex, inr = sample["extrinsics"], sample["intrinsics"]
    e34 = torch.stack([ex[:T][:n, :3, :], ex[T:][:n, :3, :]], dim=1).float()
    i33 = torch.stack([inr[:T][:n], inr[T:][:n]], dim=1).float()
    pos, rot = fuse_multiview_heatmaps_to_pose_torch(
        heat, e34, i33, near=0.6, far=1.8, num_depth_samples=512,
        apply_edge_smoothing=False, return_matrix=True, axis_solver="sphere",
        constrain_axis_depth=False)
    gt7 = sample["action_7d"].numpy()[:n]
    # `pos` is a POSE8 row [x,y,z,qx,qy,qz,qw,openness], not a 3-vector -- eval/policy.py names
    # it `_p` and only ever uses the companion rotation matrix, so the shape is easy to assume
    # wrong. Take the translation explicitly.
    pos8 = pos.numpy()
    pos_err = np.linalg.norm(pos8[:, :3] - gt7[:, :3], axis=-1)
    gt_R = Rot.from_euler("xyz", gt7[:, 3:6]).as_matrix()
    rel = np.matmul(np.transpose(rot.numpy(), (0, 2, 1)), gt_R)
    rot_err = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    # Gripper openness, same derivation as eval/eval_action.py:154-167 -- pose8's 8th column is
    # the openness recovered from the blue-channel pedestal, thresholded at 0.5 and compared with
    # action_7d's 7th column. Added here because tab:wam_preservation wants gripper accuracy and
    # this is the only action scorer the batch harness calls; eval_action.py computes it but runs
    # a different, single-episode path.
    pred_open = (pos8[:, 7] > 0.5).astype(float)
    gripper_acc = float((pred_open == gt7[:, 6]).mean())

    r = pred_v0[..., 0].reshape(len(pred_v0), -1).max(axis=1)
    dark = float((pred_v0.max(axis=-1) < 30).mean())
    return {"metric": "pos_err_m", "value": round(float(np.median(pos_err)), 4),
            "gripper_acc": round(gripper_acc, 4),
            "pos_err_mean_m": round(float(pos_err.mean()), 4),
            "rot_err_med_deg": round(float(np.median(rot_err)), 2),
            "rot_err_over90_frac": round(float((rot_err > 90).mean()), 4),
            "r_peak": round(float(r.mean()), 2),
            "sparsity_dark_frac": round(dark, 4)}


def main():
    a = _args()
    # The gate runs for an EXPLICIT index too. It did not, and that cost a whole handoff: the
    # caller had verified GPUs 1 and 2 were free, passed `--gpu 1`, and the check was skipped --
    # so the run went straight to a card a tenant had taken in the meantime.
    if a.claim_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.claim_gpu)
        print(f"[grid] taking over GPU {a.claim_gpu}", flush=True)
        claim_gpu(a)
    else:
        # `--gpu` used to be accepted and then silently ignored -- main only ever consulted
        # --claim_gpu or pick_gpu. A run launched as `CUDA_VISIBLE_DEVICES=6 ... --gpu 0`
        # printed "using GPU 2" (pick_gpu reads nvidia-smi, which enumerates PHYSICAL cards
        # regardless of the mask) while actually landing on 6, because torch had already
        # initialised under the inherited mask. Two different indices in play and neither
        # matched the log. Honour the flag, and say which numbering the printed index is in.
        if a.gpu != "auto":
            gpu, how = str(a.gpu), "explicit --gpu"
        else:
            gpu, how = pick_gpu(a), "picked by free memory"
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            print(f"[grid] NOTE: inherited CUDA_VISIBLE_DEVICES="
                  f"{os.environ['CUDA_VISIBLE_DEVICES']!r}; overriding it with {gpu!r} "
                  f"(physical index). Pass only one of the two.", flush=True)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        print(f"[grid] using physical GPU {gpu} ({how})", flush=True)
        reserve_vram(a.need_mib)
    only = {t.strip() for t in a.only.split(",") if t.strip()}
    grid = [(t, ms) for t, ms in GRID if not only or t in only]

    out_dir = Path(a.out) / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    templates = sorted({t for t, _ in grid})
    _seed_all(a.seed)
    ds = RLBenchSelfgenDataset(
        base_path=a.data, num_frames=a.num_frames, frame_interval=a.frame_interval,
        height=a.res, width=a.res,
        template_mix=",".join(f"{t}@{1/len(templates):.4f}" for t in templates),
        prompt_tag_style="explicit", strict_getitem=True, variations="all",
        segmentation_mode=a.segmentation_mode, legacy_scene_seg_tag=a.legacy_scene_seg_tag,
    )
    if a.episode.startswith("tasks:"):
        n_task = int(a.episode.split(":", 1)[1])
        tasks = sorted({e["path"].split(os.sep)[-4] for e in ds.episodes})[:n_task]
        ep_list = [f"{t}/variation0/episodes/episode0" for t in tasks]
    else:
        ep_list = [e.strip() for e in a.episode.split(",") if e.strip()]
    ep_idx = []
    for ep in ep_list:
        want = os.path.normpath(os.path.join(a.data, ep))
        i = next((i for i, e in enumerate(ds.episodes) if os.path.normpath(e["path"]) == want), None)
        if i is None:
            print(f"  SKIP {ep}: not in the dataset after the variation filter")
            continue
        ep_idx.append((ep, i))
    if not ep_idx:
        raise SystemExit(f"none of {ep_list} resolved against {a.data}")
    print(f"[grid] {len(ep_idx)} episode(s): {[e for e, _ in ep_idx]}", flush=True)

    pipe = build_pipeline(SimpleNamespace(
        model_id="Wan-AI/Wan2.2-TI2V-5B", ckpt_path=a.ckpt, height=a.res, width=a.res,
        use_usp=False, cfg_parallel=False, dynamic_cache_schedule=False, torch_compile=False))
    device, dtype = pipe.device, torch.bfloat16
    T = a.num_frames
    results, sheets = {}, {}

    if a.modes:
        want = [m.strip() for m in a.modes.split(",") if m.strip()]
        # `policy` needs an action segment and `video` has no target to withhold: keep GRID's
        # own filtering rather than forcing a mode onto a template that cannot express it.
        grid = [(t, [m for m in want if m in ms or m == "f0f0"]) for t, ms in grid]
        grid = [(t, ms) for t, ms in grid if ms and t != "video"]
    # Each entry is (mode, absent_set). A grid mode carries no explicit absent set; a
    # --withhold entry carries exactly one and no mode.
    withhold = [w.strip() for w in a.withhold.split(",") if w.strip()]
    # Episodes OUTERMOST: metrics.json is merged after each (template, mode), so a run
    # killed when a tenant takes the card still leaves every completed episode scored.
    for _ep, idx in ep_idx:
        print(f'[grid] episode {_ep}', flush=True)
        for template, modes in grid:
            mods = parse_template(template)
            target = next((m for m in mods if m != "video"), None)
            bad = [w for w in withhold if w not in mods]
            if bad:
                raise SystemExit(f"--withhold {bad} not in template {template!r} ({list(mods)})")
            jobs = [(m, None) for m in modes] + [(None, [w]) for w in withhold]
            for mode, absent in jobs:
                key = f"{_ep}|{template}|" + (mode if mode else f"out:{absent[0]}")
                t0 = time.time()
                _seed_all(a.seed)
                s = ds.getitem(idx, force_template=template)
                if s["template"] != template:
                    print(f"  SKIP {key}: dataset degraded to {s['template']!r}")
                    continue
                s["_view_dir"] = [s["view_dirs"][i] for i in s["view_indices"]][0]
                streams = {k: v.unsqueeze(0).to(device=device, dtype=dtype) for k, v in s["streams"].items()}
                _seed_all(a.seed)
                frames = np.stack([np.asarray(f) for f in pipe(
                    prompt=[s["text"]], negative_prompt="", template=template, streams=streams,
                    conditioning_mode=mode, absent_modalities=absent,
                    camera=s["camera"].unsqueeze(0).to(device=device, dtype=dtype),
                    action_7d=s["action_7d"].unsqueeze(0).to(device=device, dtype=dtype),
                    extrinsics=s["extrinsics"].unsqueeze(0).to(device=device, dtype=dtype),
                    intrinsics=s["intrinsics"].unsqueeze(0).to(device=device, dtype=dtype),
                    height=a.res, width=a.res, num_frames=T, cfg_scale=a.cfg,
                    num_inference_steps=a.steps, seed=a.seed, tiled=False,
                    tile_size=(a.res // 16, a.res // 16), tile_stride=(a.res // 32, a.res // 32),
                    enable_usp=False, cfg_parallel=False)])

                plan = seg_plan(template, 2, mode, absent)
                spans, total = spans_for(plan, T)
                assert total == len(frames), (
                    f"{key}: recomputed spans total {total} frames but the pipeline returned "
                    f"{len(frames)} -- the slicing rule and the pipeline disagree")

                per_seg_by_mod = set()
                per_seg, rec = {}, {"template": template, "mode": mode, "prompt": s["text"],
                                    "seconds": round(time.time() - t0, 1), "segments": []}
                for (m, v, single, given, absent), (b, e) in zip(plan, spans):
                    clip = frames[b:e]
                    name = f"{template.replace('+','-')}__{mode}__{m}_view{v}"
                    save_video([Image.fromarray(f) for f in clip], str(out_dir / f"{name}.mp4"),
                               fps=8, quality=7)
                    per_seg[(m, v)] = clip
                    per_seg_by_mod.add(m)
                    entry = {"modality": m, "view": v, "frames": int(len(clip)),
                             "single_frame": single, "fully_given": given, "absent": absent}
                    if v == 0:
                        if m not in ("video", "action"):
                            entry.update(score(m, ds, s, clip, a.res))
                        if m == "video":
                            gt = _u8(s["streams"]["video"])[:T][: len(clip)]
                            mse = float(((clip.astype(np.float64) - gt.astype(np.float64)) ** 2).mean())
                            entry.update({"metric": "psnr_db",
                                          "value": round(10 * np.log10(255.0 ** 2 / max(mse, 1e-9)), 3)})
                    rec["segments"].append(entry)
                if "action" in per_seg_by_mod:
                    e0 = next(e for e in rec["segments"] if e["modality"] == "action" and e["view"] == 0)
                    e0.update(score_action(per_seg[("action", 0)], per_seg[("action", 1)], s, T))
                results[key] = rec
                    # Only the FIRST episode gets a contact sheet. With several episodes the grid
                # image becomes unreadable and very large, and the sheet is a qualitative aid --
                # the numbers are what multiple episodes are for.
                if _ep == ep_idx[0][0]:
                    sheets[key] = (per_seg, s)
                head = [f"{e['modality']}v{e['view']}:{e.get('metric')}={e.get('value')}"
                        + ("(given)" if e.get("fully_given") else "(absent)" if e.get("absent") else "")
                        for e in rec["segments"] if e.get("metric")]
                print(f"  {key:34s} {rec['seconds']:6.1f}s  {' '.join(head)}", flush=True)

                # MERGE, never overwrite. Two runs against the same checkpoint with different
                # --only sets share a tag, and a plain dump silently deletes the other one's work:
                # backfilling `video+action,video+normal` onto tag arm6_2500 destroyed the depth and
                # segmentation results a previous run had put there, and the loss was only visible
                # later as holes in a chart.
                merged = {}
                mpath = out_dir / "metrics.json"
                if mpath.exists():
                    try:
                        merged = json.load(open(mpath)).get("results", {})
                    except (OSError, ValueError):
                        merged = {}
                merged.update(results)
                with open(mpath, "w") as f:
                    json.dump({"tag": a.tag, "ckpt": a.ckpt, "episode": a.episode,
                               "res": a.res, "frame_interval": a.frame_interval, "seed": a.seed,
                               "segmentation_mode": a.segmentation_mode, "results": merged}, f, indent=1)

    build_sheet(sheets, out_dir, a)

    # ---- across-episode aggregate -------------------------------------------------------
    # A single episode gives a point with no spread, which is how a 1.31x ratio in
    # ARM7_RESULTS.md survived long enough to be quoted before a powered re-test reversed it.
    # Reported per (template, mode, modality): n, median, and a bootstrap 95% CI of the mean.
    import collections as _c, random as _r
    agg = _c.defaultdict(list)
    for key, rec in results.items():
        _, tmpl, mode = key.split("|", 2)
        for e in rec["segments"]:
            if e.get("metric") is None or e.get("view") != 0:
                continue
            agg[(tmpl, mode, e["modality"], e["metric"])].append(e["value"])

    def _ci(v, n_boot=10000, seed=0):
        if len(v) < 2:
            return (None, None)
        rng = _r.Random(seed)
        means = sorted(sum(rng.choices(v, k=len(v))) / len(v) for _ in range(n_boot))
        return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])

    print(f"\n=== across-episode aggregate (n episodes = {len(ep_idx)}) ===")
    rows = {}
    for (tmpl, mode, mod, metric), v in sorted(agg.items()):
        v = [x for x in v if isinstance(x, (int, float))]
        if not v:
            continue
        med = sorted(v)[len(v) // 2]
        lo, hi = _ci(v)
        rows[f"{tmpl}|{mode}|{mod}|{metric}"] = {
            "n": len(v), "median": round(med, 5), "mean": round(sum(v) / len(v), 5),
            "ci95_lo": None if lo is None else round(lo, 5),
            "ci95_hi": None if hi is None else round(hi, 5), "values": [round(x, 5) for x in v]}
        ci = "" if lo is None else f"  CI95 [{lo:.4f}, {hi:.4f}]"
        print(f"  {mode:14s} {mod:13s} {metric:12s} n={len(v):2d} med={med:.4f}{ci}")
    mpath = out_dir / "metrics.json"
    if mpath.exists():
        try:
            blob = json.load(open(mpath))
        except (OSError, ValueError):
            blob = {}
        blob["aggregate"] = rows
        blob["episodes"] = [e for e, _ in ep_idx]
        with open(mpath, "w") as f:
            json.dump(blob, f, indent=1)
    print(f"\nwrote {out_dir}")
    return 0


def build_sheet(sheets, out_dir, a):
    """One row per (template, mode); columns are frames of the TARGET segment, view 0."""
    if not sheets:
        return
    rows = []
    for key, (per_seg, s) in sheets.items():
        template, mode = key.split("|")
        tgt = next((m for m in parse_template(template) if m != "video"), "video")
        clip = per_seg.get((tgt, 0))
        if clip is None:
            continue
        rows.append((f"{template}  [{mode}]", clip))
    if not rows:
        return
    cell, lbl, top = 180, 250, 74
    ncol = 5
    canvas = Image.new("RGB", (lbl + cell * ncol, top + cell * len(rows)), (18, 18, 20))
    d = ImageDraw.Draw(canvas)
    fb = ImageFont.truetype(FONT, 20); fs = ImageFont.truetype(FONT, 14)
    d.text((10, 10), f"{a.tag}  --  every modality x every conditioning mode", font=fb, fill="white")
    d.text((10, 36), f"{a.episode}   res={a.res}  fi={a.frame_interval}  seed={a.seed}  "
                     f"target segment, view 0", font=fs, fill=(160, 160, 168))
    for r, (name, clip) in enumerate(rows):
        d.text((10, top + r * cell + cell // 2 - 8), name, font=fs, fill=(150, 210, 165))
        picks = np.linspace(0, len(clip) - 1, ncol).astype(int)
        for c, i in enumerate(picks):
            canvas.paste(Image.fromarray(clip[i]).resize((cell, cell), Image.NEAREST),
                         (lbl + c * cell, top + r * cell))
    canvas.save(out_dir / "grid.png")


if __name__ == "__main__":
    sys.exit(main())
