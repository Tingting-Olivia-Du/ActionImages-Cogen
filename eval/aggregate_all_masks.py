"""Aggregate a complete evaluation root into CSV, markdown tables, and metric heatmaps."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "comparisons_all_masks"
ORDER = ["official_step125750"] + [f"arm0_step{x}" for x in (1000, 1500, 2000, 2500)] + [
    f"arm1_step{x}" for x in (1000, 1500, 2000, 2500)
]
MODES = {"action": ["iiii", "fiii", "fifi", "policy"], "depth": ["iiii", "fiii", "fifi"], "seg": ["iiii", "fiii", "fifi"]}


def mean_depth(m):
    return float(np.mean([v["absrel_penalized"] for v in m["views"]]))


def mean_seg(m):
    vals = [x["iou"] for v in m["views"] for x in v["instances"].values()]
    return float(np.mean(vals))


def read_all():
    reports = {}
    for tag in ORDER:
        for task, modes in MODES.items():
            for mode in modes:
                path = ROOT / tag / task / mode / "metrics.json"
                if not path.exists():
                    raise FileNotFoundError(path)
                with open(path) as f:
                    reports[tag, task, mode] = json.load(f)
    return reports


def rows(reports):
    out = []
    for (tag, task, mode), r in reports.items():
        base = {
            "checkpoint": tag, "resolution": r["resolution"], "task": task, "mode": mode,
            "training_support": r["training_support"], "generation_seconds": r["generation_seconds"],
            "r_peak_mean": "", "r_peak_weak_frac": "", "heatmap_mse": "",
            "pos_err_mean_m": "", "pos_err_strong_median_m": "",
            "axis_err_strong_median_deg": "", "gripper_acc": "",
            "depth_absrel_penalized_mean": "", "depth_valid_frac_mean": "", "seg_iou_mean": "",
        }
        m = r["metrics"]
        if task == "action":
            for k in ("r_peak_mean", "r_peak_weak_frac", "heatmap_mse", "pos_err_mean_m",
                      "pos_err_strong_median_m", "axis_err_strong_median_deg", "gripper_acc"):
                base[k] = m[k]
        elif task == "depth":
            base["depth_absrel_penalized_mean"] = mean_depth(m)
            base["depth_valid_frac_mean"] = float(np.mean([v["valid_frac"] for v in m["views"]]))
        else:
            base["seg_iou_mean"] = mean_seg(m)
        out.append(base)
    return out


def table(reports, task, metric):
    modes = MODES[task]
    lines = ["| checkpoint | res | " + " | ".join(m.upper() for m in modes) + " |",
             "|---|---:|" + "|".join(["---:"] * len(modes)) + "|"]
    for tag in ORDER:
        vals = []
        for mode in modes:
            r = reports[tag, task, mode]
            if task == "action": value = r["metrics"][metric]
            elif task == "depth": value = mean_depth(r["metrics"])
            else: value = mean_seg(r["metrics"])
            vals.append(f"{value:.4f}")
        lines.append(f"| {tag} | {reports[tag, task, modes[0]]['resolution']} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def heatmaps(reports):
    panels = [
        ("action", "Action position error (m, lower is better)", lambda r: r["metrics"]["pos_err_strong_median_m"], "magma_r"),
        ("depth", "Depth penalized AbsRel (lower is better)", lambda r: mean_depth(r["metrics"]), "magma_r"),
        ("seg", "Segmentation IoU (higher is better)", lambda r: mean_seg(r["metrics"]), "viridis"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    for ax, (task, title, fn, cmap) in zip(axes, panels):
        modes = MODES[task]
        data = np.array([[fn(reports[tag, task, mode]) for mode in modes] for tag in ORDER])
        im = ax.imshow(data, aspect="auto", cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(modes)), [m.upper() for m in modes])
        ax.set_yticks(range(len(ORDER)), ORDER, fontsize=8)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if data[i,j] > (data.min()+data.max())/2 else "black")
        fig.colorbar(im, ax=ax, shrink=.7)
    fig.savefig(ROOT / "metric_heatmaps.png", dpi=180)
    plt.close(fig)


def main():
    global ROOT
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(ROOT))
    ROOT = Path(p.parse_args().root)
    reports = read_all()
    flat = rows(reports)
    with open(ROOT / "all_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader(); w.writerows(flat)
    heatmaps(reports)
    readme = f"""# All-checkpoint / all-mask evaluation

Completed **90/90** deterministic generation cells on one fixed seen RLBench episode
`open_drawer/variation0/episodes/episode0`, frames 7..47, views 0 and 2, seed 42, CFG 7.5,
50 diffusion steps. Official uses 512×512; arm0/arm1 use 256×256.

Mask definitions: `IIII` gives only the first latent of every segment; `FIII` gives the entire
first segment; `FIFI` gives both RGB segments; `policy` collapses both RGB segments to one latent
and is defined only for action templates. Official and arm0 depth/seg rows are zero-shot negative
controls (`training_support=false`), not trained capabilities.

## Action: strong-frame median position error (m; lower is better)

{table(reports, 'action', 'pos_err_strong_median_m')}

## Depth: mean penalized AbsRel across two views (lower is better)

{table(reports, 'depth', '')}

## Segmentation: mean IoU across two views (higher is better)

{table(reports, 'seg', '')}

## Files

- `all_metrics.csv`: all scalar metrics, one row per cell.
- `metric_heatmaps.png`: compact checkpoint × mask comparison.
- `<checkpoint>/<task>/<mode>/metrics.json`: full per-view metrics and provenance.
- `<checkpoint>/<task>/<mode>/all_segments_labeled.mp4`: labelled GT RGB | model RGB | GT target | model target.
- `<checkpoint>/<task>/<mode>/segments/*.mp4`: eight separately labelled GT/model segment videos.
- `<checkpoint>/<task>/<mode>/generated_segments.npz`: exact raw frames for all four model segments.
- `<checkpoint>/<task>/<mode>/segment_manifest.json`: segment index, modality, view, status and actual length.
- `<checkpoint>/<task>/<mode>/rgb_gt_prediction_contact_sheet.png`: five-frame labelled contact sheet.
- `logs/`: complete run logs; `launcher_results.json` records all nine successful processes.

These are single-episode probes, not a multi-episode benchmark or RLBench success rate. Resolution
is intentionally different by request, so official-vs-fork comparisons include a resolution/domain
effect; within arm0/arm1 all settings are matched.
"""
    (ROOT / "README.md").write_text(readme)
    print(f"AGGREGATE_OK rows={len(flat)} root={ROOT}")


if __name__ == "__main__":
    main()
