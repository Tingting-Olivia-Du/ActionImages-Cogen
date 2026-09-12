"""Open-loop action evaluation: generate action images, decode them, score against GT.

Produces both halves of "what does the action head actually do":

  * VIDEO   -- the raw generated canvas, plus the official `combine_action_video` overlay
               (`[rgb | action | 0.5 blend]`), so the action blobs can be seen on the arm.
  * NUMBERS -- the decoded 6-DoF trajectory against ground truth, using the paper's full
               decoder (`fuse_multiview_heatmaps_to_pose_torch`, Sec. 3.2).

Metrics, and why each one is here (EVAL_PLAN_CLOSEDLOOP.md Sec. 4.2):

  r_peak_mean            per-frame max of the red channel. THE primary health signal: the
                         action is a Gaussian blob, so a low r_peak means the model stopped
                         drawing anything decodable -- a collapse the loss curve does not show.
  r_peak_weak_frac       fraction of frames below 100, so a few strong frames cannot hide a
                         mostly-dead sequence.
  pos_err_strong_median  position error over frames where BOTH views have r_peak >= 100.
                         Unconditional pos_err lies: once the blob is gone the argmax is noise
                         and the error "stabilises" instead of exploding.
  rot_err_strong_median  geodesic rotation error on the same frames. Free, and it is what
                         closed-loop execution actually needs.
  gripper_acc            from the blue-channel pedestal, paper Eq. (7).

`--frame_interval` MUST match the checkpoint's training value or the conditioning is out of
distribution (arm4__seed42_fi3 was trained at 3).

`--template` picks WHICH OBSERVATION MODALITY the action is decoded from. The four supported
templates all lay out as `X0 | A0 | X1 | A1`, so the decoder below is identical for all of them;
only the anchor pixels differ. This is arm7's whole point -- it trains `video+action`,
`depth+action`, `segmentation+action` and `normal+action` in one checkpoint, and the question
"can it act from a depth observation?" is exactly `--template depth+action`. Asking a checkpoint
for a template it never trained on is off-distribution and the numbers mean nothing: arm0-arm6
only ever saw `video+action`.

Reference point: the paper's Table 4 reports 3DErr = 12.2 mm for the official model on
in-domain RLBench; the codec round-trip floor measured on GT renders is ~4 mm.
"""
from __future__ import annotations

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
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from diffsynth import save_video  # noqa: E402
from inference import build_pipeline  # noqa: E402
from training.dataset import RLBenchSelfgenDataset  # noqa: E402
from training.templates import parse_template  # noqa: E402
from training.helpers.io import combine_action_video  # noqa: E402
from training.utils import fuse_multiview_heatmaps_to_pose_torch  # noqa: E402

NEAR, FAR, DEPTH_SAMPLES = 0.6, 1.8, 512
R_PEAK_STRONG = 100.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", nargs="*", default=[
        "open_drawer/variation0/episodes/episode0",
        "push_buttons/variation0/episodes/episode0",
        "meat_off_grill/variation0/episodes/episode0",
    ])
    p.add_argument("--data", default=str(REPO / "data" / "rlbench_selfgen_512_aug"),
                   help="512_aug; the bare rlbench_selfgen symlink points at the deleted v2 tree")
    p.add_argument("--output", default=str(REPO / "reports" / "closedloop" / "openloop_action"))
    p.add_argument("--tag", default="arm4_step5000")
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--num_frames", type=int, default=41)
    p.add_argument("--frame_interval", type=int, default=1,
                   help="MUST match the checkpoint's training value (3 for arm4__seed42_fi3)")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--template", default="video+action",
                   choices=("video+action", "depth+action", "segmentation+action",
                            "normal+action"),
                   help="Which observation modality anchors the action. Must be one the "
                        "checkpoint was TRAINED on -- arm7 has all four, every earlier arm has "
                        "only video+action.")
    p.add_argument("--segmentation_mode", default="scene_roles",
                   choices=("referring", "scene_roles"),
                   help="Only used by --template segmentation+action. Must match the "
                        "checkpoint's training protocol: the two emit different prompt tags AND "
                        "different pixels, so the wrong one asks a question the model never saw.")
    p.add_argument("--protocol", choices=("i2va", "a2v"), default="i2va",
                   help="i2va = predict the action (the real test); a2v = action fully given, "
                        "which measures the encode/VAE round-trip FLOOR and is independent of "
                        "the checkpoint")
    p.add_argument("--gripper-mode", choices=("paper", "median"), default="paper")
    p.add_argument("--axis-solver", choices=("ray", "sphere"), default="sphere",
                   help="'ray' is the PAPER's method (Sec. 3.2); 'sphere' is a fork deviation "
                        "that avoids its foreshortening degeneracy. Use 'ray' for anything "
                        "compared against the paper's Table 4.")
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()
    # The [X0|A0|X1|A1] slice in evaluate() reads the action segments at index 1 and 3, which is
    # only right for a TWO-modality template. `choices` already guarantees that, so this is a
    # guard against someone widening it -- checked here rather than in evaluate() because the
    # failure would otherwise land after a full 50-step generation.
    if len(parse_template(args.template)) != 2:
        p.error(f"--template {args.template!r} is not a 2-modality template; the [X0|A0|X1|A1] "
                f"segment slice in evaluate() would read the wrong pixels.")
    return args


def find_episode(ds, path):
    wanted = os.path.normpath(path)
    for i, ep in enumerate(ds.episodes):
        if os.path.normpath(ep["path"]) == wanted:
            return i
    raise RuntimeError(f"episode not found: {path}")


def evaluate(pipe, ds, idx, args, out_dir):
    random.seed(args.seed)
    sample = ds.getitem(idx, force_template=args.template)
    # getitem DEGRADES rather than fails when a template's ground truth does not resolve on this
    # episode (segmentation with no scene_segments.json falls back to video+action). Silently
    # scoring a video+action sample and labelling it `depth+action` is the one failure this
    # script must not have, so check what actually came back.
    if sample["template"] != args.template:
        raise RuntimeError(
            f"requested {args.template!r} but the dataset served {sample['template']!r} for "
            f"{sample['path']} -- its ground truth for that modality does not resolve.")
    T = args.num_frames
    device, dtype = pipe.device, torch.bfloat16

    gt_a7 = sample["action_7d"].numpy().astype(np.float64)      # [T, 7] xyz + euler + openness
    extr = sample["extrinsics"]                                  # [2T, 4, 4] absolute c2w
    intr = sample["intrinsics"]                                  # [2T, 3, 3]

    t0 = time.time()
    frames = pipe(
        prompt=[sample["text"]],
        negative_prompt="",
        template=args.template,
        streams={k: v.unsqueeze(0).to(device=device, dtype=dtype)
                 for k, v in sample["streams"].items()},
        fully_given_modalities=["action"] if args.protocol == "a2v" else [],
        camera=sample["camera"].unsqueeze(0).to(device=device, dtype=dtype),
        action_7d=sample["action_7d"].unsqueeze(0).to(device=device, dtype=dtype),
        extrinsics=extr.unsqueeze(0).to(device=device, dtype=dtype),
        intrinsics=intr.unsqueeze(0).to(device=device, dtype=dtype),
        height=args.res, width=args.res, num_frames=T,
        cfg_scale=args.cfg, num_inference_steps=args.steps, seed=args.seed,
        tiled=False, tile_size=(args.res // 16, args.res // 16),
        tile_stride=(args.res // 32, args.res // 32),
        enable_usp=False, cfg_parallel=False,
    )
    gen_seconds = time.time() - t0

    arr = np.stack([np.asarray(f) for f in frames])
    if len(arr) % 4:
        raise RuntimeError(f"expected 4 equal segments, got {arr.shape}")
    q = len(arr) // 4
    # [X0 | A0 | X1 | A1] -- same slices as inference.py:311-313. Correct for all four supported
    # templates because each is (one visual modality, action) and segment order is view-major
    # with action last within a view, so the action segments are always index 1 and 3. A THREE-
    # modality template (video+depth+action) is six segments and these slices would silently
    # score the wrong pixels, so refuse rather than trust the choices= list to stay in sync.
    heat = torch.from_numpy(np.stack([arr[q:2 * q], arr[3 * q:4 * q]], axis=1)).float()

    ext34 = torch.stack([extr[:T, :3, :], extr[T:, :3, :]], dim=1).float()   # [T, 2, 3, 4]
    int33 = torch.stack([intr[:T], intr[T:]], dim=1).float()                 # [T, 2, 3, 3]
    pose8 = fuse_multiview_heatmaps_to_pose_torch(
        heat, ext34, int33, near=NEAR, far=FAR, num_depth_samples=DEPTH_SAMPLES,
        apply_edge_smoothing=False, gripper_mode=args.gripper_mode,
        axis_solver=args.axis_solver,
    ).numpy().astype(np.float64)

    # --- metrics ---------------------------------------------------------------------
    r_peak = heat[..., 0].amax(dim=(-1, -2)).numpy()          # [T, V]
    strong = (r_peak >= R_PEAK_STRONG).all(axis=1)            # both views decodable

    pos_err = np.linalg.norm(pose8[:, :3] - gt_a7[:, :3], axis=1)
    gt_rot = Rot.from_euler("xyz", gt_a7[:, 3:6]).as_matrix()
    rel = np.matmul(np.transpose(Rot.from_quat(pose8[:, 3:7]).as_matrix(), (0, 2, 1)), gt_rot)
    rot_err = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))

    gt_open = gt_a7[:, 6]
    pred_open = (pose8[:, 7] > 0.5).astype(float)

    def med(x, mask):
        return float(np.median(x[mask])) if mask.any() else float("nan")

    metrics = {
        "r_peak_mean": float(r_peak.mean()),
        "r_peak_weak_frac": float((r_peak < R_PEAK_STRONG).mean()),
        "strong_frac": float(strong.mean()),
        "pos_err_strong_median_m": med(pos_err, strong),
        "pos_err_mean_m": float(pos_err.mean()),
        "rot_err_strong_median_deg": med(rot_err, strong),
        "gripper_acc": float((pred_open == gt_open).mean()),
        "gripper_ghat_mean": float(pose8[:, 7].mean()),
        "gt_open_frac": float(gt_open.mean()),
        "generation_seconds": round(gen_seconds, 1),
    }

    # --- video -----------------------------------------------------------------------
    stem = (f"{args.tag}_{sample['path'].strip('/').replace('/', '_')[-58:]}"
            f"_{args.template.replace('+', '-')}_{args.protocol}")
    if not args.no_video:
        os.makedirs(out_dir, exist_ok=True)
        save_video(list(frames), os.path.join(out_dir, f"{stem}_raw.mp4"), fps=15, quality=6)
        # The official overlay: [anchor | action | 0.5 blend], i.e. action blobs on the arm.
        # Under --template depth+action the left panel is the DEPTH render, not RGB -- which is
        # the point: it shows what the model was actually looking at when it drew the action.
        overlay = combine_action_video(frames)
        save_video([Image.fromarray(np.asarray(f)) for f in overlay],
                   os.path.join(out_dir, f"{stem}_overlay.mp4"), fps=15, quality=6)

    return {
        "episode": sample["path"],
        "prompt": sample["text"],
        "template": args.template,
        "protocol": args.protocol,
        "axis_solver": args.axis_solver,
        "frame_interval": args.frame_interval,
        "window_span_seconds": round((T - 1) * args.frame_interval / 20.0, 2),
        "view_indices": sample["view_indices"],
        "frame_indices": [int(i) for i in sample["frame_indices"]],
        **metrics,
        "per_frame": {
            "r_peak": r_peak.round(1).tolist(),
            "pos_err_m": pos_err.round(4).tolist(),
            "rot_err_deg": rot_err.round(2).tolist(),
            "ghat": pose8[:, 7].round(4).tolist(),
            # 解出来的【位姿本身】,不只是误差标量。跨模态融合必须在位姿层面做:
            # pos_err 是误差【向量的模长】,模长之间相关不代表向量相关 —— 四路可能朝同一
            # 方向偏(融合无效)也可能朝不同方向偏(融合有效),只看模长分辨不出来。
            # 存下来才能离线试各种融合规则而不用重跑扩散采样。
            "pose8": pose8.round(5).tolist(),
            "gt_action7": gt_a7.round(5).tolist(),
        },
    }


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ds = RLBenchSelfgenDataset(
        base_path=args.data, num_frames=args.num_frames, frame_interval=args.frame_interval,
        height=args.res, width=args.res, template_mix=f"{args.template}@1.0",
        prompt_tag_style="explicit", strict_getitem=True, variations="all",
        segmentation_mode=args.segmentation_mode,
    )
    pipe = build_pipeline(SimpleNamespace(
        model_id="Wan-AI/Wan2.2-TI2V-5B", ckpt_path=args.ckpt,
        height=args.res, width=args.res, use_usp=False, cfg_parallel=False,
        dynamic_cache_schedule=False, torch_compile=False,
    ))

    out_dir = os.path.join(args.output, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    reports = []
    for ep in args.episodes:
        idx = find_episode(ds, os.path.join(args.data, ep))
        r = evaluate(pipe, ds, idx, args, out_dir)
        reports.append(r)
        print(f"[{args.template}] [{ep}] r_peak={r['r_peak_mean']:6.1f} weak={r['r_peak_weak_frac']*100:5.1f}% "
              f"pos={r['pos_err_strong_median_m']:.4f}m rot={r['rot_err_strong_median_deg']:6.2f}deg "
              f"grip_acc={r['gripper_acc']*100:5.1f}% ({r['generation_seconds']:.0f}s)", flush=True)

    agg = {
        "tag": args.tag, "checkpoint": args.ckpt, "template": args.template,
        "protocol": args.protocol,
        "axis_solver": args.axis_solver,
        "frame_interval": args.frame_interval, "cfg": args.cfg, "steps": args.steps,
        "n_episodes": len(reports),
        "r_peak_mean": round(float(np.mean([r["r_peak_mean"] for r in reports])), 1),
        "r_peak_weak_frac": round(float(np.mean([r["r_peak_weak_frac"] for r in reports])), 4),
        "pos_err_strong_median_m": round(float(np.nanmedian(
            [r["pos_err_strong_median_m"] for r in reports])), 4),
        "rot_err_strong_median_deg": round(float(np.nanmedian(
            [r["rot_err_strong_median_deg"] for r in reports])), 2),
        "gripper_acc": round(float(np.mean([r["gripper_acc"] for r in reports])), 4),
    }
    # The template goes in the filename: without it a depth+action run overwrites the
    # video+action run's report in the same --tag directory, and the four readings this script
    # exists to compare would clobber each other.
    path = os.path.join(out_dir, f"report_{args.template.replace('+', '-')}_{args.protocol}.json")
    with open(path, "w") as f:
        json.dump({"summary": agg, "episodes": reports}, f, indent=2)
    print("\n" + json.dumps(agg, indent=2))
    print(f"wrote {path}  (videos in {out_dir})")


if __name__ == "__main__":
    main()
