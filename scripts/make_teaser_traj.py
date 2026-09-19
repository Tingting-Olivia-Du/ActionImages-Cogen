#!/usr/bin/env python
"""Trajectory-overlay panel for Figure 1: the four observation spaces, decoded, against GT.

WHAT IT PLOTS. `eval/eval_action.py` already stores the decoded pose per frame, not just the
error scalar -- `episodes[i].per_frame.pose8` is [T,8] (xyz, quat xyzw, gripper) and
`gt_action7` is [T,7]. The error magnitude alone cannot show whether four paths agree: four
trajectories can be equally wrong in four different directions. So this figure draws the paths
themselves, all four decoded by the same multi-view fusion against one ground truth.

INPUT. One report per template, written by eval_action.py as

    <output>/<tag>/report_<template-with-dashes>_<protocol>.json

so the four files for a checkpoint are report_video-action_i2va.json,
report_depth-action_i2va.json, report_segmentation-action_i2va.json,
report_normal-action_i2va.json. Pass the directory that holds them.

EPISODE CHOICE. Only episodes present in all four reports can be overlaid. Among those the
default is the one whose *median* position error across the four spaces is itself the median
of the candidates -- a typical case rather than the best one. `--episode` overrides by
substring, `--rank 0` picks the best if you want a qualitative panel and say so in the caption.

COLOURS. Fixed per observation space and shared with Figs/fig_teaser.tex. They are checked for
colour-vision separation (OKLab dE >= 15 normal, >= 8 under deuteranopia and protanopia); if
you change one, re-check the whole set rather than picking by eye. Ground truth is black and
dashed: it is a reference, not a fifth category.

USAGE
    python scripts/make_teaser_traj.py \
        --reports reports/closedloop/openloop_action/arm7_6k \
        --out ../6a89bc13108d28a2448f410a/Figs/teaser/traj_overlay.pdf

    # layout for an appendix panel: x, y, z against time, which reads more precisely than 3D
    python scripts/make_teaser_traj.py --reports <dir> --layout xyz --out <path>

    # preview the layout before any evaluation has run; the output is stamped SYNTHETIC and
    # written to a different filename so it cannot be mistaken for a result
    python scripts/make_teaser_traj.py --selftest --out /tmp/preview.pdf
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

# observation space -> (template as eval_action.py spells it, label, colour)
SPACES = [
    ("video+action",        "RGB",       "#3060C0"),
    ("depth+action",        "depth",     "#12907A"),
    ("segmentation+action", "segmentation", "#C8641A"),
    ("normal+action",       "normal",    "#C22E86"),
]
GT_COLOR = "#1A1A1A"

INK = "#333333"
MUTED = "#8A8A8A"


def report_path(d: Path, template: str, protocol: str) -> Path:
    return d / f"report_{template.replace('+', '-')}_{protocol}.json"


def load_reports(d: Path, protocol: str) -> dict:
    """template -> {episode_path: (pose8 [T,8], gt7 [T,7])}."""
    out = {}
    for template, _, _ in SPACES:
        p = report_path(d, template, protocol)
        if not p.exists():
            raise SystemExit(
                f"missing {p}\n"
                f"run:  python eval/eval_action.py --template {template} "
                f"--frame_interval 3 --res 512 --segmentation_mode scene_roles "
                f"--protocol {protocol} --output {d.parent} --tag {d.name}"
            )
        blob = json.load(open(p))
        eps = {}
        for ep in blob.get("episodes", []):
            pf = ep.get("per_frame", {})
            if "pose8" not in pf or "gt_action7" not in pf:
                continue
            eps[ep["episode"]] = (np.asarray(pf["pose8"], float),
                                  np.asarray(pf["gt_action7"], float))
        out[template] = eps
    return out


def pick_episode(reports: dict, want: str | None, rank: int | None) -> str:
    shared = set.intersection(*(set(v) for v in reports.values()))
    if not shared:
        raise SystemExit("no episode appears in all four reports; they must be run on one split")
    if want:
        hits = sorted(e for e in shared if want in e)
        if not hits:
            raise SystemExit(f"--episode {want!r} matches none of {len(shared)} shared episodes")
        return hits[0]

    def score(ep):  # median over spaces of the median per-frame position error
        per_space = []
        for template, _, _ in SPACES:
            pose8, gt7 = reports[template][ep]
            n = min(len(pose8), len(gt7))
            per_space.append(np.median(np.linalg.norm(pose8[:n, :3] - gt7[:n, :3], axis=1)))
        return float(np.median(per_space))

    ordered = sorted(shared, key=score)
    idx = rank if rank is not None else len(ordered) // 2
    return ordered[max(0, min(idx, len(ordered) - 1))]


def synthetic() -> tuple[dict, str]:
    """Plausible-looking curves for a layout preview. Never used for a result."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 41)
    gt = np.stack([0.25 + 0.30 * t, -0.10 + 0.22 * np.sin(np.pi * t), 0.85 + 0.18 * t ** 1.7], 1)
    reports, gt7 = {}, np.concatenate([gt, np.zeros((len(t), 4))], 1)
    for k, (template, _, _) in enumerate(SPACES):
        drift = (0.008 + 0.010 * k) * np.cumsum(rng.normal(size=(len(t), 3)), 0) / np.sqrt(len(t))
        pose8 = np.concatenate([gt + drift, np.zeros((len(t), 5))], 1)
        reports[template] = {"SYNTHETIC": (pose8, gt7)}
    return reports, "SYNTHETIC"


def style_axes_3d(ax):
    ax.set_xlabel("x (m)", labelpad=0, fontsize=7, color=INK)
    ax.set_ylabel("y (m)", labelpad=0, fontsize=7, color=INK)
    ax.set_zlabel("z (m)", labelpad=-1, fontsize=7, color=INK)
    ax.tick_params(labelsize=6, colors=MUTED, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(plt.MaxNLocator(4))
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor("white")
        pane.pane.set_edgecolor("#DDDDDD")
        pane._axinfo["grid"].update(color="#EAEAEA", linewidth=0.5)


def equalise_3d(ax, pts):
    lo, hi = pts.min(0), pts.max(0)
    c, r = (lo + hi) / 2, max(hi - lo).item() / 2 or 0.1
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def draw_3d(reports, episode, args):
    fig = plt.figure(figsize=(args.width, args.height))
    ax = fig.add_subplot(111, projection="3d")
    allpts = []
    gt = None
    for template, label, color in SPACES:
        pose8, gt7 = reports[template][episode]
        n = min(len(pose8), len(gt7))
        xyz, gt = pose8[:n, :3], gt7[:n, :3]
        allpts.append(xyz)
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, lw=1.6,
                solid_capstyle="round", label=label, zorder=3)
    allpts.append(gt)
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color=GT_COLOR, lw=1.4, ls=(0, (3, 2)),
            label="ground truth", zorder=4)
    ax.scatter(*gt[0], s=14, color=GT_COLOR, zorder=5)

    equalise_3d(ax, np.concatenate(allpts, 0))
    style_axes_3d(ax)
    ax.view_init(elev=args.elev, azim=args.azim)
    # the tight bbox is unreliable for 3d axes and clips the x label; reserve the room here
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.12, top=0.90)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, fontsize=6,
              frameon=False, handlelength=1.3, columnspacing=1.0, handletextpad=0.5)
    return fig


def draw_xyz(reports, episode, args):
    fig, axes = plt.subplots(3, 1, figsize=(args.width, args.height), sharex=True)
    for row, name in enumerate("xyz"):
        ax = axes[row]
        for template, label, color in SPACES:
            pose8, gt7 = reports[template][episode]
            n = min(len(pose8), len(gt7))
            ax.plot(np.arange(n), pose8[:n, row], color=color, lw=1.4,
                    label=label if row == 0 else None)
        ax.plot(np.arange(n), gt7[:n, row], color=GT_COLOR, lw=1.2, ls=(0, (3, 2)),
                label="ground truth" if row == 0 else None)
        ax.set_ylabel(f"{name} (m)", fontsize=7, color=INK)
        ax.tick_params(labelsize=6, colors=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#CCCCCC")
        ax.grid(axis="y", color="#EEEEEE", lw=0.5)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("frame", fontsize=7, color=INK)
    axes[0].legend(ncol=5, fontsize=6, frameon=False, loc="lower left",
                   bbox_to_anchor=(0, 1.02), handlelength=1.4, columnspacing=1.0)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports", type=Path, help="directory holding the four report_*.json files")
    ap.add_argument("--protocol", default="i2va", help="protocol suffix of the report filenames")
    ap.add_argument("--episode", default=None, help="substring of the episode path to plot")
    ap.add_argument("--rank", type=int, default=None,
                    help="0 = best of the shared episodes; default is the median one")
    ap.add_argument("--layout", choices=["3d", "xyz"], default="3d")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--width", type=float, default=2.6, help="inches")
    ap.add_argument("--height", type=float, default=2.3, help="inches")
    ap.add_argument("--elev", type=float, default=22.0)
    ap.add_argument("--azim", type=float, default=-58.0)
    ap.add_argument("--selftest", action="store_true",
                    help="draw synthetic curves to preview the layout; output is stamped")
    args = ap.parse_args()

    if args.selftest:
        reports, episode = synthetic()
    elif args.reports:
        reports = load_reports(args.reports, args.protocol)
        episode = pick_episode(reports, args.episode, args.rank)
    else:
        raise SystemExit("pass --reports <dir>, or --selftest to preview the layout")

    plt.rcParams.update({"font.family": "sans-serif", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = (draw_3d if args.layout == "3d" else draw_xyz)(reports, episode, args)

    if args.selftest:
        fig.text(0.5, 0.5, "SYNTHETIC", fontsize=22, color="#C22E86", alpha=0.25,
                 ha="center", va="center", rotation=24, zorder=10)
        args.out = args.out.with_name(args.out.stem + "_SELFTEST" + args.out.suffix)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tight = "tight" if args.layout == "xyz" else None
    fig.savefig(args.out, bbox_inches=tight, pad_inches=0.05, transparent=True)
    # an SVG next to it, for slides
    fig.savefig(args.out.with_suffix(".svg"), bbox_inches=tight, pad_inches=0.05,
                transparent=True)
    plt.close(fig)
    print(f"episode: {episode}")
    print(f"wrote {args.out}  and  {args.out.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
