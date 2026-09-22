"""Causal, role-aware interpretability for XGenAct action generation.

The experiment deletes a task region from the *conditioning RGB anchor* and compares it with
an equally large deletion from background.  Clean and intervened generations use the same
episode window, cameras, prompt, and diffusion seed.  The useful quantity is therefore not a
pretty saliency map but a causal contrast:

    selectivity(group) = action_shift(delete_group) - action_shift(control_group)

Run Arm 0 and Arm 7 in separate processes/GPU masks, then compare their JSON files with
``scripts/compare_mechanism_roles.py``.  Example:

    CUDA_VISIBLE_DEVICES=6 python -u scripts/mechanism_causal_roles.py \
      --ckpt outputs/arm0_joint2src_seed42_fi3_6k/checkpoint-4000/step4000.ckpt \
      --tag arm0_4k --task close_microwave --target-instances microwave_door

    CUDA_VISIBLE_DEVICES=7 python -u scripts/mechanism_causal_roles.py \
      --ckpt outputs/arm7_joint2src_seed42_fi3_6k/checkpoint-4000/step4000.ckpt \
      --tag arm7_4k --task close_microwave --target-instances microwave_door

The script deliberately reads instance handles from ``scene_segments.json`` instead of relying
on the dense role stream.  The unseen RLBench tree has no ``seg_targets.json``; consequently its
manipulated object may retain the base role ``distractor``.  Calling that label ``target`` would
make the interpretation wrong even if the picture looked plausible.
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
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rot

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import scripts.modality_mode_grid as G  # noqa: E402
from inference import build_pipeline  # noqa: E402
from training.dataset import RLBenchSelfgenDataset  # noqa: E402
from training.templates import parse_template, prompt_prefix  # noqa: E402
from training.utils import fuse_multiview_heatmaps_to_pose_torch  # noqa: E402


DEFAULT_CONDITIONS = ("clean", "delete_target", "control_target")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--task", default="close_microwave")
    p.add_argument("--variation", type=int, default=0)
    p.add_argument("--episodes", default="0,1,2",
                   help="comma-separated episode numbers; each gets a deterministic window")
    p.add_argument("--data", default=str(REPO / "data" / "rlbench_unseen_tasks_512_clean"))
    p.add_argument("--out", default=str(REPO / "reports" / "mechanism_roles"))
    p.add_argument("--target-instances", default="microwave_door",
                   help="comma-separated scene_segments instance names defining the manipulated object")
    p.add_argument("--goal-instances", default="",
                   help="optional explicit goal instances; otherwise base_role=goal is used")
    p.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS),
                   help="clean, delete_<group>, control_<group>, or keep_structure; groups are "
                        "target,goal,fixture,robot,task,structure")
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--num-frames", type=int, default=41)
    p.add_argument("--frame-interval", type=int, default=3)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42,
                   help="shared diffusion seed; sample windows use seed+episode")
    p.add_argument("--inpaint-radius", type=float, default=5.0)
    p.add_argument("--dilate-px", type=int, default=3)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def _csv(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _uint8_video(stream: torch.Tensor) -> np.ndarray:
    """Dataset stream [3,2T,H,W] in [-1,1] -> [2T,H,W,3] uint8."""
    return np.rint((stream.permute(1, 2, 3, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def _normalised_video(frames: np.ndarray) -> torch.Tensor:
    """[2T,H,W,3] uint8 -> dataset stream [3,2T,H,W] in [-1,1]."""
    return torch.from_numpy(frames.astype(np.float32) / 127.5 - 1.0).permute(3, 0, 1, 2).contiguous()


def _resize_labels_nearest(labels: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    if labels.shape == size:
        return labels
    x = torch.from_numpy(labels.astype(np.float32))[None, None]
    return F.interpolate(x, size=size, mode="nearest")[0, 0].numpy().astype(labels.dtype)


def _scene_groups(sample: Mapping, resolution: int, target_instances: Sequence[str],
                  goal_instances: Sequence[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Return boolean masks [V,H,W] for task groups at the sampled anchor frame.

    Masks are derived from the exact view/frame provenance chosen by the dataset.  This avoids
    a common interpretability bug: loading frame zero while the evaluated window starts later,
    or loading view1 while the model happened to draw views 2 and 4.
    """
    ep = Path(sample["path"])
    spec = json.load(open(ep / "scene_segments.json"))
    instances = spec.get("instances", {})
    missing = sorted(set(target_instances) - set(instances))
    if missing:
        raise ValueError(f"target instances {missing} absent from {ep / 'scene_segments.json'}; "
                         f"available={sorted(instances)}")
    if goal_instances:
        missing = sorted(set(goal_instances) - set(instances))
        if missing:
            raise ValueError(f"goal instances {missing} absent from scene spec; available={sorted(instances)}")

    def handles(names: Iterable[str] = (), roles: Iterable[str] = ()) -> set[int]:
        names, roles = set(names), set(roles)
        out: set[int] = set()
        for name, rec in instances.items():
            if name in names or rec.get("base_role") in roles:
                out.update(int(h) for h in rec.get("handles", []))
        return out

    handle_sets = {
        "target": handles(target_instances),
        "goal": handles(goal_instances) if goal_instances else handles(roles=("goal",)),
        "fixture": handles(roles=("fixture",)),
        "robot": handles(roles=("robot_arm", "gripper")),
        "background": handles(roles=("background",)),
    }
    anchor_idx = int(sample["frame_indices"][0])
    masks: Dict[str, List[np.ndarray]] = {k: [] for k in handle_sets}
    chosen_dirs = [sample["view_dirs"][i] for i in sample["view_indices"]]
    for view_dir in chosen_dirs:
        label = np.load(Path(view_dir) / "mask.npz")["mask"][anchor_idx]
        label = _resize_labels_nearest(label, (resolution, resolution))
        for group, ids in handle_sets.items():
            masks[group].append(np.isin(label, np.asarray(sorted(ids), dtype=label.dtype)))

    out = {k: np.stack(v) for k, v in masks.items()}
    out["task"] = out["target"] | out["goal"]
    out["structure"] = out["task"] | out["fixture"] | out["robot"]
    known = out["structure"] | out["background"]
    out["background"] |= ~known  # empty/no-object pixels are valid background controls
    meta = {
        "target_instances": list(target_instances),
        "goal_instances": list(goal_instances),
        "handle_sets": {k: sorted(v) for k, v in handle_sets.items()},
        "anchor_source_frame": anchor_idx,
        "view_dirs": [str(v) for v in chosen_dirs],
    }
    return out, meta


def dilate_masks(masks: np.ndarray, pixels: int) -> np.ndarray:
    """Dilate each [H,W] mask without letting one view leak into another."""
    masks = np.asarray(masks, dtype=bool)
    if pixels <= 0:
        return masks.copy()
    kernel = np.ones((2 * pixels + 1, 2 * pixels + 1), np.uint8)
    return np.stack([cv2.dilate(m.astype(np.uint8), kernel, iterations=1).astype(bool) for m in masks])


def matched_background_masks(source: np.ndarray, background: np.ndarray) -> np.ndarray:
    """A deterministic contiguous background control with exactly the source area per view.

    The centre is the point deepest inside background.  Selecting the nearest background
    pixels around it yields a compact blob instead of salt-and-pepper corruption.  Exact area
    matching makes deletion magnitude an invalid explanation for target-vs-control differences.
    """
    source, background = np.asarray(source, bool), np.asarray(background, bool)
    controls = []
    for src, bg in zip(source, background):
        n = int(src.sum())
        if n == 0:
            controls.append(np.zeros_like(src))
            continue
        ys, xs = np.nonzero(bg)
        if len(ys) < n:
            raise ValueError(f"background has {len(ys)} pixels but an area-matched control needs {n}")
        dist = cv2.distanceTransform(bg.astype(np.uint8), cv2.DIST_L2, 5)
        cy, cx = np.unravel_index(int(dist.argmax()), dist.shape)
        order = np.argsort((ys - cy) ** 2 + (xs - cx) ** 2, kind="stable")[:n]
        control = np.zeros_like(src)
        control[ys[order], xs[order]] = True
        controls.append(control)
    return np.stack(controls)


def intervene_rgb(anchor_frames: np.ndarray, masks: Mapping[str, np.ndarray], condition: str,
                  inpaint_radius: float = 5.0, dilate_px: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Apply an intervention to [V,H,W,3] anchors; return pixels and the applied mask."""
    anchors = np.asarray(anchor_frames, dtype=np.uint8)
    if condition == "clean":
        return anchors.copy(), np.zeros(anchors.shape[:3], dtype=bool)

    if condition == "keep_structure":
        keep = dilate_masks(masks["structure"], dilate_px)
        out = np.stack([cv2.GaussianBlur(im, (0, 0), sigmaX=12, sigmaY=12) for im in anchors])
        out[keep] = anchors[keep]
        return out, ~keep

    prefix, sep, group = condition.partition("_")
    if not sep or prefix not in {"delete", "control"} or group not in masks:
        raise ValueError(f"unknown condition {condition!r}; expected clean, keep_structure, "
                         f"delete_<group>, or control_<group>; groups={sorted(masks)}")
    region = dilate_masks(masks[group], dilate_px)
    if prefix == "control":
        # Match the *applied*, dilated deletion area, not the raw annotation area.
        region = matched_background_masks(region, masks["background"] & ~region)
    out = []
    for im, region_v in zip(anchors, region):
        if region_v.any():
            out.append(cv2.inpaint(im, region_v.astype(np.uint8) * 255,
                                   float(inpaint_radius), cv2.INPAINT_TELEA))
        else:
            out.append(im.copy())
    return np.stack(out), region


def decode_action(frames: np.ndarray, sample: Mapping, template: str, T: int) -> Dict[str, object]:
    plan = G.seg_plan(template, 2, "iiii")
    spans, total = G.spans_for(plan, T)
    if total != len(frames):
        raise RuntimeError(f"decoded canvas has {len(frames)} frames, plan expects {total}")
    clips = {v: frames[b:e] for (m, v, *_), (b, e) in zip(plan, spans) if m == "action"}
    n = min(len(clips[0]), len(clips[1]), T)
    heat = torch.from_numpy(np.stack([clips[0][:n], clips[1][:n]], axis=1)).float()
    ex, intr = sample["extrinsics"], sample["intrinsics"]
    e34 = torch.stack([ex[:T][:n, :3, :], ex[T:][:n, :3, :]], dim=1).float()
    i33 = torch.stack([intr[:T][:n], intr[T:][:n]], dim=1).float()
    pose, rot = fuse_multiview_heatmaps_to_pose_torch(
        heat, e34, i33, near=0.6, far=1.8, num_depth_samples=512,
        apply_edge_smoothing=False, return_matrix=True, axis_solver="sphere",
        constrain_axis_depth=False)
    pose8 = pose.numpy().astype(np.float64)
    gt7 = sample["action_7d"].numpy()[:n].astype(np.float64)
    pos_err = np.linalg.norm(pose8[:, :3] - gt7[:, :3], axis=-1)
    gt_R = Rot.from_euler("xyz", gt7[:, 3:6]).as_matrix()
    rel = np.matmul(np.transpose(rot.numpy(), (0, 2, 1)), gt_R)
    rot_err = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    pred_open = pose8[:, 7] > 0.5
    return {
        "pose8": pose8,
        "gt7": gt7,
        "pos_err": pos_err,
        "rot_err": rot_err,
        "metrics": {
            "pos_err_median_m": float(np.median(pos_err)),
            "pos_err_mean_m": float(np.mean(pos_err)),
            "rot_err_median_deg": float(np.median(rot_err)),
            "gripper_acc": float(np.mean(pred_open == (gt7[:, 6] > 0.5))),
            "r_peak_mean": float(heat[..., 0].amax(dim=(-1, -2)).mean()),
        },
    }


def _jsonable_action(decoded: Mapping[str, object]) -> Dict[str, object]:
    return {
        **decoded["metrics"],
        "pose8": np.asarray(decoded["pose8"]).round(6).tolist(),
        "gt_action7": np.asarray(decoded["gt7"]).round(6).tolist(),
        "pos_err_per_frame_m": np.asarray(decoded["pos_err"]).round(6).tolist(),
        "rot_err_per_frame_deg": np.asarray(decoded["rot_err"]).round(4).tolist(),
    }


def _add_clean_contrasts(records: Dict[str, Dict[str, object]]) -> None:
    if "clean" not in records:
        return
    clean = records["clean"]
    clean_pose = np.asarray(clean["pose8"], dtype=np.float64)
    clean_err = np.asarray(clean["pos_err_per_frame_m"], dtype=np.float64)
    for condition, rec in records.items():
        pose = np.asarray(rec["pose8"], dtype=np.float64)
        err = np.asarray(rec["pos_err_per_frame_m"], dtype=np.float64)
        n = min(len(pose), len(clean_pose))
        shift = np.linalg.norm(pose[:n, :3] - clean_pose[:n, :3], axis=-1)
        rec["trajectory_shift_median_m"] = round(float(np.median(shift)), 6)
        rec["trajectory_shift_mean_m"] = round(float(np.mean(shift)), 6)
        rec["delta_pos_error_mean_m"] = round(float(np.mean(err[:n] - clean_err[:n])), 6)
        rec["delta_pos_error_median_m"] = round(
            float(np.median(err[:n]) - np.median(clean_err[:n])), 6)


def _plot_episode(path: Path, anchors: np.ndarray, masks: Mapping[str, np.ndarray],
                  edited: Mapping[str, np.ndarray], records: Mapping[str, Mapping[str, object]]) -> None:
    conditions = list(records)
    fig = plt.figure(figsize=(4.1 * max(len(conditions), 2), 7.5))
    gs = fig.add_gridspec(2, max(len(conditions), 2), height_ratios=(1, 1.05))
    target = masks["target"][0]
    for j, condition in enumerate(conditions):
        ax = fig.add_subplot(gs[0, j])
        im = edited[condition][0].copy()
        if condition == "clean":
            overlay = im.copy()
            overlay[target] = (255, 45, 45)
            im = cv2.addWeighted(im, 0.68, overlay, 0.32, 0)
        ax.imshow(im)
        rec = records[condition]
        ax.set_title(f"{condition}\nerr={rec['pos_err_median_m']:.3f} m")
        ax.axis("off")
    ax3 = fig.add_subplot(gs[1, :], projection="3d")
    first = next(iter(records.values()))
    gt = np.asarray(first["gt_action7"])
    ax3.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="black", lw=2.5, label="ground truth")
    colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))
    for color, condition in zip(colors, conditions):
        pose = np.asarray(records[condition]["pose8"])
        ax3.plot(pose[:, 0], pose[:, 1], pose[:, 2], color=color, lw=1.8,
                 label=f"{condition} (shift {records[condition].get('trajectory_shift_median_m', 0):.3f}m)")
    ax3.set_xlabel("world x [m]")
    ax3.set_ylabel("world y [m]")
    ax3.set_zlabel("world z [m]")
    ax3.legend(loc="best", fontsize=8)
    fig.suptitle("Role-conditioned causal intervention (red overlay = manipulated object)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    a = _args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit("Set CUDA_VISIBLE_DEVICES before Python starts (use physical GPU 6 or 7).")
    episodes = [int(x) for x in _csv(a.episodes)]
    conditions = _csv(a.conditions)
    if "clean" not in conditions:
        conditions.insert(0, "clean")
    targets, goals = _csv(a.target_instances), _csv(a.goal_instances)
    if not targets:
        raise SystemExit("--target-instances must name at least one scene_segments instance")

    out_dir = Path(a.out) / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{a.task}_causal_roles.json"
    report: Dict[str, object] = {
        "tag": a.tag, "checkpoint": str(Path(a.ckpt).resolve()), "task": a.task,
        "data": str(Path(a.data).resolve()), "variation": a.variation,
        "target_instances": targets, "goal_instances": goals,
        "conditions": conditions, "seed": a.seed, "steps": a.steps, "cfg": a.cfg,
        "frame_interval": a.frame_interval, "resolution": a.res, "episodes": {},
    }
    if report_path.exists() and not a.no_resume:
        old = json.load(open(report_path))
        # Reports written by the first pilot predate the top-level instance fields.  Recover
        # them from per-episode provenance so those valid cells remain resumable without making
        # a different target definition look compatible.
        old_eps = old.get("episodes", {})
        first_old_ep = next(iter(old_eps.values()), {})
        old_mask_meta = first_old_ep.get("mask_meta", {})
        old_targets = old.get("target_instances", old_mask_meta.get("target_instances"))
        old_goals = old.get("goal_instances", old_mask_meta.get("goal_instances"))
        same = (
            all(old.get(k) == report.get(k) for k in
                ("checkpoint", "task", "variation", "conditions", "seed", "steps", "cfg",
                 "frame_interval", "resolution"))
            and old_targets == targets
            and old_goals == goals
        )
        if same:
            report = old
            report["target_instances"], report["goal_instances"] = targets, goals
            print(f"[mechanism] resuming {report_path}", flush=True)

    ds = RLBenchSelfgenDataset(
        base_path=a.data, num_frames=a.num_frames, frame_interval=a.frame_interval,
        height=a.res, width=a.res, template_mix="video+action@1.0",
        prompt_tag_style="explicit", strict_getitem=True, variations="all",
        segmentation_mode="scene_roles")
    ep_paths = {
        ep: os.path.normpath(os.path.join(a.data, a.task, f"variation{a.variation}",
                                          "episodes", f"episode{ep}")) for ep in episodes
    }
    idx_by_ep = {}
    for ep, wanted in ep_paths.items():
        idx = next((i for i, rec in enumerate(ds.episodes)
                    if os.path.normpath(rec["path"]) == wanted), None)
        if idx is None:
            raise SystemExit(f"episode not found in dataset index: {wanted}")
        idx_by_ep[ep] = idx

    print(f"[mechanism] visible GPU={os.environ['CUDA_VISIBLE_DEVICES']} model={a.tag}", flush=True)
    pipe = build_pipeline(SimpleNamespace(
        model_id="Wan-AI/Wan2.2-TI2V-5B", ckpt_path=a.ckpt, height=a.res, width=a.res,
        use_usp=False, cfg_parallel=False, dynamic_cache_schedule=False, torch_compile=False))
    device, dtype = pipe.device, torch.bfloat16
    template = "video+action"

    for ep in episodes:
        ep_key = f"episode{ep}"
        _seed_all(a.seed + ep)
        sample = ds.getitem(idx_by_ep[ep], force_template=template)
        if sample["template"] != template:
            raise RuntimeError(f"dataset served {sample['template']!r}, expected {template!r}")
        frames = _uint8_video(sample["streams"]["video"])
        T = a.num_frames
        anchors = np.stack([frames[0], frames[T]])
        masks, mask_meta = _scene_groups(sample, a.res, targets, goals)
        prompt = prompt_prefix(parse_template(template), style="explicit") + sample["text"].split("> ", 1)[-1]
        expected_prefix = "<video><action> "
        if not prompt.startswith(expected_prefix):
            raise AssertionError(f"unexpected action prompt {prompt!r}")

        ep_report = report["episodes"].setdefault(ep_key, {})
        ep_report.update({
            "path": sample["path"], "prompt": prompt,
            "view_indices": [int(x) for x in sample["view_indices"]],
            "frame_indices": [int(x) for x in sample["frame_indices"]],
            "mask_meta": mask_meta,
            "raw_area_fraction": {k: [round(float(x.mean()), 7) for x in v]
                                  for k, v in masks.items()},
        })
        records: Dict[str, Dict[str, object]] = ep_report.setdefault("results", {})
        edited_for_plot: Dict[str, np.ndarray] = {}

        for condition in conditions:
            edited_anchor, applied = intervene_rgb(
                anchors, masks, condition, inpaint_radius=a.inpaint_radius, dilate_px=a.dilate_px)
            edited_for_plot[condition] = edited_anchor
            if condition in records and not a.no_resume:
                print(f"  SKIP {ep_key}/{condition} already complete", flush=True)
                continue
            edited_video = frames.copy()
            edited_video[0], edited_video[T] = edited_anchor[0], edited_anchor[1]
            stream = _normalised_video(edited_video).unsqueeze(0).to(device=device, dtype=dtype)
            t0 = time.time()
            _seed_all(a.seed)
            generated = np.stack([np.asarray(x) for x in pipe(
                prompt=[prompt], negative_prompt="", template=template, streams={"video": stream},
                conditioning_mode="iiii",
                camera=sample["camera"].unsqueeze(0).to(device=device, dtype=dtype),
                action_7d=sample["action_7d"].unsqueeze(0).to(device=device, dtype=dtype),
                extrinsics=sample["extrinsics"].unsqueeze(0).to(device=device, dtype=dtype),
                intrinsics=sample["intrinsics"].unsqueeze(0).to(device=device, dtype=dtype),
                height=a.res, width=a.res, num_frames=T, cfg_scale=a.cfg,
                num_inference_steps=a.steps, seed=a.seed, tiled=False,
                tile_size=(a.res // 16, a.res // 16), tile_stride=(a.res // 32, a.res // 32),
                enable_usp=False, cfg_parallel=False)])
            decoded = decode_action(generated, sample, template, T)
            records[condition] = {
                "condition": condition,
                "applied_area_fraction": [round(float(x.mean()), 7) for x in applied],
                "seconds": round(time.time() - t0, 1),
                **_jsonable_action(decoded),
            }
            _add_clean_contrasts(records)
            with open(report_path, "w") as f:
                json.dump(report, f, indent=1)
            r = records[condition]
            print(f"  {ep_key}/{condition:18s} err={r['pos_err_median_m']:.4f}m "
                  f"shift={r.get('trajectory_shift_median_m', 0):.4f}m "
                  f"area={100*np.mean(r['applied_area_fraction']):.2f}% "
                  f"({r['seconds']:.0f}s)", flush=True)

        _add_clean_contrasts(records)
        if not a.no_figures and all(c in records for c in conditions):
            _plot_episode(out_dir / f"{a.task}_{ep_key}.png", anchors, masks,
                          edited_for_plot, records)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=1)

    # Compact per-condition macro summary, while preserving every paired per-frame value above.
    summary = {}
    for condition in conditions:
        rows = [ep["results"][condition] for ep in report["episodes"].values()
                if condition in ep.get("results", {})]
        if not rows:
            continue
        keys = ("pos_err_median_m", "pos_err_mean_m", "trajectory_shift_median_m",
                "trajectory_shift_mean_m", "delta_pos_error_mean_m")
        summary[condition] = {k: round(float(np.median([r[k] for r in rows])), 6)
                              for k in keys if k in rows[0]}
        summary[condition]["n"] = len(rows)
    report["summary_median_over_episodes"] = summary
    with open(report_path, "w") as f:
        json.dump(report, f, indent=1)
    print(f"[mechanism] wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
