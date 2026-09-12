#!/usr/bin/env python3
"""Visual, executable comparison of ActionImages/train.py and this fork's train.py.

This intentionally uses tiny labelled latent tensors instead of loading Wan.  The property
under test is sequence construction (segment order, length, camera repetition and clean/noisy
mask), which is independent of VAE/DiT weights.  The upstream functions below are literal
translations of ActionImages/train.py:256-316; the fork side calls the production
training.templates implementation used by ActionImages-Cogen/train.py.

Run from the fork root:
    PYTHONPATH=. python debug/compare_train_versions.py
Outputs:
    debug/out/train_version_comparison.png
    debug/out/train_stream_slicing.png
    debug/out/train_version_comparison.json
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from training.templates import assemble, parse_template, plan_segments  # noqa: E402

OUT = os.path.join(REPO, "debug", "out")
T_L = 4
COLORS = {
    "video": "#4C78A8",
    "depth": "#F2CF5B",
    "segmentation": "#B279A2",
    "action": "#F58518",
}


class ScriptedRandom:
    def __init__(self, values: Sequence[float]):
        self.values = list(values)
        self.i = 0

    def random(self) -> float:
        value = self.values[self.i]
        self.i += 1
        return value


@dataclass(frozen=True)
class DrawSegment:
    modality: str
    view: int
    length: int


def latent(value: int) -> torch.Tensor:
    return torch.full((1, 1, T_L, 1, 1), float(value))


def camera(value: int) -> torch.Tensor:
    return torch.full((1, T_L, 1, 1, 1), float(value))


def fixtures():
    values = {"video": 10, "depth": 30, "segmentation": 50, "action": 70}
    latents = {m: [latent(n), latent(n + 10)] for m, n in values.items()}
    cameras = [camera(100), camera(200)]
    return latents, cameras


def upstream_action(video, action, cams, *, single_frame: bool, p: float = 0.5):
    """Literal behaviour of ActionImages/train.py's action-present branch."""
    v0, v1 = video
    a0, a1 = action
    c0, c1 = cams
    if single_frame:
        z = torch.cat([v0[:, :, [0]], a0, v1[:, :, [0]], a1], dim=2)
        cam = torch.cat([c0[:, [0]], c0, c1[:, [0]], c1], dim=1)
        mask = torch.zeros_like(z, dtype=torch.bool)
        for i in (0, 1, T_L + 1, T_L + 2):
            mask[:, :, i] = True
        segs = [DrawSegment("video", 0, 1), DrawSegment("action", 0, T_L),
                DrawSegment("video", 1, 1), DrawSegment("action", 1, T_L)]
    else:
        z = torch.cat([v0, a0, v1, a1], dim=2)
        cam = torch.cat([c0, c0, c1, c1], dim=1)
        mask = torch.zeros_like(z, dtype=torch.bool)
        for i in (0, T_L, 2 * T_L, 3 * T_L):
            mask[:, :, i] = True
        if 0.90 <= p < 0.95:
            mask[:, :, :T_L] = True
        elif p >= 0.95:
            mask[:, :, :T_L] = True
            mask[:, :, 2 * T_L:3 * T_L] = True
        segs = [DrawSegment("video", 0, T_L), DrawSegment("action", 0, T_L),
                DrawSegment("video", 1, T_L), DrawSegment("action", 1, T_L)]
    return (z, cam, mask), segs


def upstream_video_only(video, cams):
    z = torch.cat(video, dim=2)
    cam = torch.cat(cams, dim=1)
    mask = torch.zeros_like(z, dtype=torch.bool)
    mask[:, :, 0] = True
    mask[:, :, T_L] = True
    segs = [DrawSegment("video", 0, T_L), DrawSegment("video", 1, T_L)]
    return (z, cam, mask), segs


def fork_case(template: str, randoms: Sequence[float]):
    all_latents, cams = fixtures()
    mods = parse_template(template)
    plan = plan_segments(mods, 2, rng=ScriptedRandom(randoms), is_rlbench=True)
    used = {m: all_latents[m] for m in mods}
    tensors = assemble(plan, used, cams)
    segs = [DrawSegment(s.modality, s.view, 1 if s.single_frame else T_L) for s in plan]
    return tensors, segs


def same_tensors(a, b) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b))


def draw_layout(ax, title: str, tensors, segments: Sequence[DrawSegment], note: str = ""):
    ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    if tensors is None:
        ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#EEEEEE", edgecolor="#999999"))
        ax.text(0.5, 0.5, note, ha="center", va="center", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        return

    z, cam, mask = tensors
    given = mask[0, 0, :, 0, 0].tolist()
    offset = 0
    for seg in segments:
        for j in range(seg.length):
            is_given = bool(given[offset + j])
            rect = Rectangle(
                (offset + j, 0), 1, 1,
                facecolor=COLORS[seg.modality],
                alpha=1.0 if is_given else 0.35,
                edgecolor="#159447" if is_given else "#C83E4D",
                linewidth=2.5 if is_given else 1.0,
                hatch=None if is_given else "//",
            )
            ax.add_patch(rect)
            ax.text(offset + j + 0.5, 0.5, "G" if is_given else "P",
                    ha="center", va="center", fontsize=7)
        ax.text(offset + seg.length / 2, 1.10, f"v{seg.view + 1} {seg.modality}",
                ha="center", va="bottom", fontsize=7)
        ax.axvline(offset, color="black", linewidth=1.2)
        offset += seg.length
    ax.axvline(offset, color="black", linewidth=1.2)
    ax.text(0, -0.28, f"sequence={z.shape[2]} latent frames; camera={cam.shape[1]}; "
            f"given={sum(given)}, predicted={len(given)-sum(given)} {note}", fontsize=7)
    ax.set_xlim(0, offset)
    ax.set_ylim(-0.42, 1.48)
    ax.set_yticks([])
    ax.set_xticks(range(offset + 1))
    ax.tick_params(axis="x", labelsize=6)


def layout_comparison(report):
    all_latents, cams = fixtures()
    cases = []

    old, old_segs = upstream_action(all_latents["video"], all_latents["action"], cams,
                                    single_frame=False, p=0.5)
    new, new_segs = fork_case("video+action", [0.5, 0.5])
    cases.append(("A. baseline joint generation", old, old_segs, new, new_segs, True))

    old, old_segs = upstream_action(all_latents["video"], all_latents["action"], cams,
                                    single_frame=False, p=0.99)
    new, new_segs = fork_case("video+action", [0.5, 0.99])
    cases.append(("B. video-to-action mask", old, old_segs, new, new_segs, True))

    old, old_segs = upstream_video_only(all_latents["video"], cams)
    new, new_segs = fork_case("video", [])
    cases.append(("C. video only", old, old_segs, new, new_segs, True))

    new, new_segs = fork_case("video+depth", [0.0])
    cases.append(("D. RGB-to-depth perception", None, [], new, new_segs, False))

    new, new_segs = fork_case("video+depth+action", [0.5, 0.5])
    cases.append(("E. six-segment co-generation", None, [], new, new_segs, False))

    old, old_segs = upstream_action(all_latents["video"], all_latents["action"], cams,
                                    single_frame=True)
    new, new_segs = fork_case("video+action", [0.05])
    cases.append(("F. RLBench single-frame policy", old, old_segs, new, new_segs, True))

    fig, axes = plt.subplots(len(cases), 2, figsize=(16, 2.25 * len(cases)))
    for row, (name, old, old_segs, new, new_segs, should_match) in enumerate(cases):
        equal = old is not None and same_tensors(old, new)
        if should_match:
            assert equal, f"{name}: fork no longer reproduces upstream"
        old_note = "not expressible: upstream has only video/action slots" if old is None else ""
        draw_layout(axes[row, 0], f"{name} — original", old, old_segs, old_note)
        draw_layout(axes[row, 1], f"{name} — Cogen", new, new_segs,
                    "(bit-identical)" if equal else "(new capability)")
        report[name] = {
            "original_available": old is not None,
            "bit_identical": equal if old is not None else None,
            "original_length": old[0].shape[2] if old is not None else None,
            "cogen_length": new[0].shape[2],
            "cogen_segments": [f"v{s.view + 1}:{s.modality}:{s.length}" for s in new_segs],
        }

    fig.suptitle(
        "Original train.py vs ActionImages-Cogen/train.py\n"
        "color = modality; G/green border = clean condition; P/red hatch = prediction target",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path = os.path.join(OUT, "train_version_comparison.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def stream_slicing(report):
    """Visualise the exact v*T:(v+1)*T slicing in the code quoted by the user."""
    t = 4
    mods = ["video", "depth", "segmentation"]
    streams = {}
    for mi, mod in enumerate(mods):
        # Values make both modality and view/frame identity visible.
        streams[mod] = torch.tensor(
            [mi * 100 + 0, mi * 100 + 1, mi * 100 + 2, mi * 100 + 3,
             mi * 100 + 10, mi * 100 + 11, mi * 100 + 12, mi * 100 + 13]
        )

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    for ax, title, visible in (
        (axes[0], "Original: only the `video` tensor", ["video"]),
        (axes[1], "Cogen: one stream per requested visual modality", mods),
    ):
        ax.set_title(title, fontsize=11, fontweight="bold")
        for row, mod in enumerate(visible):
            vals = streams[mod]
            for view in range(2):
                sl = vals[view * t:(view + 1) * t]
                for frame, value in enumerate(sl.tolist()):
                    x = view * (t + 1) + frame
                    ax.add_patch(Rectangle((x, row), 1, 0.75, facecolor=COLORS[mod], alpha=0.75,
                                           edgecolor="black"))
                    ax.text(x + 0.5, row + 0.38, str(value), ha="center", va="center", fontsize=8)
                ax.text(view * (t + 1) + t / 2, row + 0.82, f"view {view}: [{view*t}:{(view+1)*t}]",
                        ha="center", fontsize=7)
            ax.text(-0.2, row + 0.38, mod, ha="right", va="center", fontsize=9)
        ax.set_xlim(-1.5, 2 * (t + 1) - 1)
        ax.set_ylim(-0.25, len(visible) + 0.35)
        ax.axis("off")
    fig.suptitle("Per-view VAE inputs produced by pixels[:, :, v*T:(v+1)*T, ...]",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = os.path.join(OUT, "train_stream_slicing.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    report["stream_slicing"] = {
        "T": t,
        "views": 2,
        "original_encoded_modalities": ["video"],
        "cogen_encoded_modalities": mods,
        "slices": ["0:4", "4:8"],
    }
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {}
    layout_path = layout_comparison(report)
    slicing_path = stream_slicing(report)
    report_path = os.path.join(OUT, "train_version_comparison.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("PASS: all four shared cases are bit-identical")
    print(f"layout  -> {layout_path}")
    print(f"slicing -> {slicing_path}")
    print(f"report  -> {report_path}")


if __name__ == "__main__":
    main()
