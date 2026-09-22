"""Compare paired Arm 7 / Arm 0 causal-role reports and make paper-ready figures.

Besides the aggregate bar plot, this writes an ``*_rgb_explainer.png`` panel.  That panel
puts the exact RGB interventions above Arm 0 and Arm 7 trajectories projected into the same
camera, so the causal test can be understood without reading a 3-D trajectory plot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm0", required=True)
    p.add_argument("--arm7", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _ci(values, n_boot, seed):
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        return [None, None, None]
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, (n_boot, len(x)), replace=True).mean(1)
    return [float(x.mean()), *np.percentile(boots, [2.5, 97.5]).tolist()]


def _read_video_frame(path: Path, index: int) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("RGB explainer requires OpenCV (run in the ttd_eval env)") from exc
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {index} from {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _dilate(mask: np.ndarray, pixels: int = 3) -> np.ndarray:
    import cv2
    kernel = np.ones((2 * pixels + 1, 2 * pixels + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _matched_background(source: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Same deterministic equal-area control used by mechanism_causal_roles.py."""
    import cv2
    ys, xs = np.nonzero(background)
    n = int(source.sum())
    if len(ys) < n:
        raise RuntimeError("not enough background pixels for the matched control")
    dist = cv2.distanceTransform(background.astype(np.uint8), cv2.DIST_L2, 5)
    cy, cx = np.unravel_index(int(dist.argmax()), dist.shape)
    order = np.argsort((ys - cy) ** 2 + (xs - cx) ** 2, kind="stable")[:n]
    control = np.zeros_like(source)
    control[ys[order], xs[order]] = True
    return control


def _inpaint(rgb: np.ndarray, region: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.inpaint(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                       region.astype(np.uint8) * 255, 5.0,
                       cv2.INPAINT_TELEA)[:, :, ::-1]


def _rgb_inputs(episode: dict):
    """Reconstruct the exact first-view images used by the causal intervention."""
    meta = episode["mask_meta"]
    view_dir = Path(meta["view_dirs"][0])
    frame = int(meta["anchor_source_frame"])
    rgb = _read_video_frame(view_dir / "rgb" / "video.mp4", frame)
    labels = np.load(view_dir / "mask.npz")["mask"][frame]
    if labels.shape != rgb.shape[:2]:
        import cv2
        labels = cv2.resize(labels, rgb.shape[1::-1], interpolation=cv2.INTER_NEAREST)

    handles = meta["handle_sets"]
    role = {name: np.isin(labels, np.asarray(ids, dtype=labels.dtype))
            for name, ids in handles.items()}
    known = role["target"] | role["goal"] | role["fixture"] | role["robot"] | role["background"]
    role["background"] |= ~known
    target = _dilate(role["target"])
    control = _matched_background(target, role["background"] & ~target)

    with open(view_dir / "camera_params.json") as f:
        camera = json.load(f)[str(frame)]
    return {
        "rgb": rgb,
        "delete_target": _inpaint(rgb, target),
        "control_target": _inpaint(rgb, control),
        "target_raw": role["target"],
        "target": target,
        "control": control,
        "camera": camera,
    }


def _project(points, camera):
    """Project world XYZ through the recorded camera-to-world matrix."""
    xyz = np.asarray(points, dtype=np.float64)[:, :3]
    extr = np.asarray(camera["extrinsics"], dtype=np.float64)
    intr = np.asarray(camera["intrinsics"], dtype=np.float64)
    camera_xyz = (extr[:3, :3].T @ (xyz - extr[:3, 3]).T).T
    image_h = (intr @ camera_xyz.T).T
    return image_h[:, :2] / image_h[:, 2:3]


def _trajectory(ax, xy, color, label, linestyle="-", zorder=5):
    line, = ax.plot(xy[:, 0], xy[:, 1], color=color, lw=2.5, ls=linestyle,
                    label=label, zorder=zorder)
    line.set_path_effects([pe.Stroke(linewidth=4.5, foreground="black", alpha=.60), pe.Normal()])
    ax.scatter(xy[0, 0], xy[0, 1], s=28, marker="o", c=color, edgecolor="black",
               linewidth=.8, zorder=zorder + 1)
    ax.scatter(xy[-1, 0], xy[-1, 1], s=75, marker="*", c=color, edgecolor="black",
               linewidth=.8, zorder=zorder + 1)


def _make_rgb_explainer(reports, episode_name: str, out_path: Path):
    """Make a 3x3 input/Arm0/Arm7 comparison for one paired episode."""
    episodes = {name: report["episodes"][episode_name] for name, report in reports.items()}
    e0, e7 = episodes["Arm 0"], episodes["Arm 7"]
    if (e0["mask_meta"]["view_dirs"] != e7["mask_meta"]["view_dirs"]
            or e0["mask_meta"]["anchor_source_frame"] != e7["mask_meta"]["anchor_source_frame"]):
        raise RuntimeError("RGB explainer requires paired reports with the same episode/views")
    visual = _rgb_inputs(e0)
    camera = visual["camera"]
    conditions = ("clean", "delete_target", "control_target")
    images = (visual["rgb"], visual["delete_target"], visual["control_target"])
    input_titles = ("A. Original RGB", "B. Microwave door removed", "C. Equal-area background removed")
    masks = (visual["target_raw"], visual["target"], visual["control"])
    mask_colors = ("#00ffff", "#ff9f1c", "#ffe66d")
    area = 100 * visual["target"].mean()

    fig, axes = plt.subplots(3, 3, figsize=(13.2, 13.0))
    for col, (image, title, mask, mask_color) in enumerate(zip(images, input_titles, masks, mask_colors)):
        ax = axes[0, col]
        ax.imshow(image)
        ax.contour(mask.astype(float), levels=[.5], colors=[mask_color], linewidths=2.0)
        ax.set_title(title, fontsize=13, weight="bold")
        if col == 0:
            ax.text(.02, .03, "cyan outline = target door", transform=ax.transAxes,
                    color="white", fontsize=10, bbox=dict(facecolor="black", alpha=.68, pad=4))
        else:
            ax.text(.02, .03, f"changed pixels = {area:.2f}%", transform=ax.transAxes,
                    color="white", fontsize=10, bbox=dict(facecolor="black", alpha=.68, pad=4))
        ax.set_axis_off()

    gt = _project(e0["results"]["clean"]["gt_action7"], camera)
    for row, model in enumerate(("Arm 0", "Arm 7"), start=1):
        episode = episodes[model]
        clean = _project(episode["results"]["clean"]["pose8"], camera)
        for col, (condition, image) in enumerate(zip(conditions, images)):
            ax = axes[row, col]
            ax.imshow(image)
            _trajectory(ax, gt, "white", "Ground truth", linestyle="--", zorder=4)
            _trajectory(ax, clean, "#168aad", "Clean prediction", zorder=5)
            if condition != "clean":
                changed = _project(episode["results"][condition]["pose8"], camera)
                _trajectory(ax, changed, "#ff7f0e", "After intervention", zorder=6)
                shift_cm = 100 * episode["results"][condition]["trajectory_shift_median_m"]
                ax.text(.02, .03, f"median trajectory shift: {shift_cm:.1f} cm",
                        transform=ax.transAxes, color="white", fontsize=10,
                        bbox=dict(facecolor="black", alpha=.72, pad=4))
            else:
                ax.text(.02, .03, "circle = start   star = end", transform=ax.transAxes,
                        color="white", fontsize=10,
                        bbox=dict(facecolor="black", alpha=.72, pad=4))
            ax.set_xlim(0, image.shape[1]); ax.set_ylim(image.shape[0], 0); ax.set_axis_off()
        axes[row, 0].text(-.08, .5, f"{model}\n" + ("RGB baseline" if model == "Arm 0" else "XGenAct"),
                          transform=axes[row, 0].transAxes, rotation=90, va="center", ha="center",
                          fontsize=14, weight="bold")

    arm0_specific = 100 * (e0["results"]["delete_target"]["trajectory_shift_median_m"]
                           - e0["results"]["control_target"]["trajectory_shift_median_m"])
    arm7_specific = 100 * (e7["results"]["delete_target"]["trajectory_shift_median_m"]
                           - e7["results"]["control_target"]["trajectory_shift_median_m"])
    handles = [
        plt.Line2D([0], [0], color="white", lw=2.5, ls="--",
                   path_effects=[pe.Stroke(linewidth=4.5, foreground="black"), pe.Normal()],
                   label="Ground truth"),
        plt.Line2D([0], [0], color="#168aad", lw=3, label="Clean prediction"),
        plt.Line2D([0], [0], color="#ff7f0e", lw=3, label="Prediction after intervention"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11)
    fig.suptitle(
        f"close_microwave / {episode_name}: exact RGB inputs and image-plane action trajectories\n"
        f"Door-specific response = door shift - equal-area background shift:  "
        f"Arm 0 {arm0_specific:.1f} cm   |   Arm 7 {arm7_specific:.1f} cm",
        fontsize=15, weight="bold", y=.985)
    fig.subplots_adjust(left=.055, right=.99, top=.92, bottom=.065, wspace=.035, hspace=.12)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    a = _args()
    reports = {"Arm 0": json.load(open(a.arm0)), "Arm 7": json.load(open(a.arm7))}
    common_eps = sorted(set(reports["Arm 0"]["episodes"]) & set(reports["Arm 7"]["episodes"]))
    common_conditions = [c for c in reports["Arm 7"]["conditions"]
                         if c in reports["Arm 0"]["conditions"] and c != "clean"]
    if not common_eps or not common_conditions:
        raise SystemExit("reports have no common completed episodes/conditions")

    metrics = ("trajectory_shift_median_m", "delta_pos_error_mean_m")
    result = {
        "arm0": str(Path(a.arm0).resolve()), "arm7": str(Path(a.arm7).resolve()),
        "paired_episodes": common_eps, "conditions": common_conditions, "metrics": {},
    }
    for metric in metrics:
        result["metrics"][metric] = {}
        for model, report in reports.items():
            result["metrics"][metric][model] = {}
            for condition in common_conditions:
                vals = [report["episodes"][ep]["results"][condition][metric] for ep in common_eps]
                result["metrics"][metric][model][condition] = {
                    "values": vals, "mean_ci95": _ci(vals, a.bootstrap, a.seed)}

    # Difference-in-differences is the central claim: target deletion over matched background,
    # then Arm 7 over Arm 0.  Keeping the raw paired values in JSON prevents a bar plot from
    # hiding heterogeneous or sign-flipped episodes.
    pairs = []
    for condition in common_conditions:
        if not condition.startswith("delete_"):
            continue
        control = "control_" + condition[len("delete_"):]
        if control not in common_conditions:
            continue
        for metric in metrics:
            selectivity = {}
            for model, report in reports.items():
                vals = [report["episodes"][ep]["results"][condition][metric]
                        - report["episodes"][ep]["results"][control][metric] for ep in common_eps]
                selectivity[model] = vals
            did = (np.asarray(selectivity["Arm 7"]) - np.asarray(selectivity["Arm 0"])).tolist()
            pairs.append({
                "metric": metric, "deletion": condition, "control": control,
                "selectivity": {m: {"values": v, "mean_ci95": _ci(v, a.bootstrap, a.seed)}
                                for m, v in selectivity.items()},
                "arm7_minus_arm0_selectivity": {
                    "values": did, "mean_ci95": _ci(did, a.bootstrap, a.seed)},
            })
    result["causal_selectivity"] = pairs

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(result, f, indent=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    width = 0.36
    x = np.arange(len(common_conditions))
    for ax, metric, title in zip(
            axes, metrics, ("Causal trajectory response", "Action-error increase")):
        for offset, (model, color) in zip((-width / 2, width / 2),
                                          (("Arm 0", "#777777"), ("Arm 7", "#d23282"))):
            means, lows, highs = [], [], []
            for condition in common_conditions:
                mean, lo, hi = result["metrics"][metric][model][condition]["mean_ci95"]
                means.append(mean); lows.append(mean - lo); highs.append(hi - mean)
            ax.bar(x + offset, means, width, label=model, color=color,
                   yerr=np.stack([lows, highs]), capsize=3)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xticks(x, [c.replace("_", "\n") for c in common_conditions])
        ax.set_ylabel("metres")
        ax.set_title(title)
    axes[0].legend(frameon=False)
    fig.suptitle(f"Role-aware intervention, paired n={len(common_eps)}")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    rgb_out = out.with_name(out.stem + "_rgb_explainer.png")
    _make_rgb_explainer(reports, common_eps[0], rgb_out)
    print(f"wrote {out.with_suffix('.json')}, {out.with_suffix('.png')}, and {rgb_out}")


if __name__ == "__main__":
    main()
