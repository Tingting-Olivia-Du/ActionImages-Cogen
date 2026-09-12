#!/usr/bin/env python
"""Per-task contact sheets of the dense scene-role segmentation target.

Run: python scripts/scene_roles_contact_sheet.py [--root data/rlbench_selfgen_512_aug]
Writes reports/scene_roles_contact_sheets/{task}.png plus _summary.png.

SEGMENTATION_SCENE_ROLES_PLAN.md 14 Phase A asks for a per-task human confirmation of the role
assignment, and SCENE_ROLES_IMPLEMENTATION_REPORT.md 3.2 records that it was never done. Every
automated check the plan defines is a check of INTERNAL consistency -- that the palette decodes,
that no pixel is `unknown`, that the target overlay does not eat the goal. None of them can see
that a task's TASK_ROLES table names the wrong object, because the table is also what the test
compares against. Only a person looking at the picture can catch that, so the picture has to
exist.

Reads through RLBenchSelfgenDataset._scene_role_lut rather than base_role alone: the
instruction->target promotion lives there, and without it nothing is ever red.

WHAT TO CHECK, per sheet:
  1. the sky above the walls is BLACK (background), not orange. Orange = a handle nobody mapped;
     that was a real bug (see scripts/audit_scene_roles.py) and this is the visual version of it.
  2. exactly one object is RED and it is the one the printed instruction names.
  3. green `goal` is the destination, not the manipulandum. put_item_in_drawer's drawers are
     green; open_drawer's identical drawers are yellow `distractor`. Same geometry, opposite
     role -- this pair is the sharpest test of whether the tables are right.
  4. sweep_to_dustpan: broom is MAGENTA `tool`, dirt is YELLOW and has not vanished (the dirt is
     5 handles of 4-8 px; the codec deliberately does no small-region pruning).
"""
import argparse
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from training.dataset import RLBenchSelfgenDataset  # noqa: E402
from training.percep.seg_codec import (  # noqa: E402
    SCENE_ROLE_PALETTE,
    SCENE_ROLES,
    encode_scene_roles,
    scene_role_labels,
)

OUTDIR = os.path.join(REPO, "reports", "scene_roles_contact_sheets")
N_FRAMES = 4


def _legend_handles(present=None):
    return [
        mpatches.Patch(color=SCENE_ROLE_PALETTE[i] / 255.0, label=r)
        for i, r in enumerate(SCENE_ROLES)
        if present is None or r in present
    ]


def _read_rgb(view_path, frame_indices):
    import imageio.v3 as iio

    frames = iio.imread(os.path.join(view_path, "rgb", "video.mp4"), plugin="pyav")
    return frames[frame_indices]


def sheet_for_task(ds, task, episode_path, views=("view1", "view2")):
    """-> (figure, {role: pixel fraction}, unknown_px, instruction)."""
    # Fixed seed: get_instruction picks at random among the 3 paraphrases, and the whole point of
    # the sheet is that the printed instruction is the one the red pixels were resolved from.
    random.seed(0)
    info = {"path": episode_path}
    instruction = ds.get_instruction(info) if hasattr(ds, "get_instruction") else ""
    lut, present = ds._scene_role_lut(episode_path, instruction)
    if lut is None:
        return None, None, None, instruction

    mask0 = ds._load_mask(os.path.join(episode_path, views[0]))
    T = mask0.shape[0]
    idx = list(np.linspace(0, T - 1, N_FRAMES).astype(int))

    nrows = 2 * len(views)
    fig, axes = plt.subplots(nrows, N_FRAMES, figsize=(3.0 * N_FRAMES, 3.0 * nrows))
    axes = np.atleast_2d(axes)

    counts = np.zeros(len(SCENE_ROLES), dtype=np.int64)
    for vi, view in enumerate(views):
        vp = os.path.join(episode_path, view)
        mask = ds._load_mask(vp)[idx]
        rgb = _read_rgb(vp, idx)
        seg = encode_scene_roles(mask.astype(np.uint16), lut)
        labels = scene_role_labels(mask.astype(np.uint16), lut)
        counts += np.bincount(labels.ravel(), minlength=len(SCENE_ROLES))
        for c, f in enumerate(idx):
            axes[2 * vi, c].imshow(rgb[c])
            axes[2 * vi, c].set_title(f"{view} t={f}", fontsize=9)
            axes[2 * vi + 1, c].imshow(seg[c])
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    for vi, view in enumerate(views):
        axes[2 * vi, 0].set_ylabel("RGB", fontsize=10)
        axes[2 * vi + 1, 0].set_ylabel("scene_roles", fontsize=10)

    tot = counts.sum()
    frac = {r: counts[i] / tot for i, r in enumerate(SCENE_ROLES)}
    unknown_px = int(counts[SCENE_ROLES.index("unknown")])
    share = "  ".join(f"{r}={frac[r] * 100:.2f}%" for r in SCENE_ROLES if counts[SCENE_ROLES.index(r)])
    warn = "" if unknown_px == 0 else f"    !! {unknown_px} UNKNOWN px -- an unmapped handle"
    fig.suptitle(
        f"{task}   {os.path.basename(episode_path)}\n"
        f'instruction: "{instruction}"\n{share}{warn}',
        fontsize=11,
    )
    fig.legend(handles=_legend_handles(present), loc="lower center", ncol=9, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    return fig, frac, unknown_px, instruction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "data", "rlbench_selfgen_512_aug"))
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--tasks", default="", help="comma list; default all")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    ds = RLBenchSelfgenDataset(
        base_path=args.root, num_frames=5, frame_interval=3, height=args.res, width=args.res,
        template_mix="video+segmentation@1.0", variations="0", segmentation_mode="scene_roles",
    )
    # self.episodes entries are {"path", "task", ...} dicts (rlbench_selfgen.py:220).
    by_task = {}
    for ep in ds.episodes:
        by_task.setdefault(ep["task"], ep["path"])

    wanted = [t.strip() for t in args.tasks.split(",") if t.strip()] or sorted(by_task)
    summary, bad = [], []
    for task in wanted:
        ep = by_task.get(task)
        if ep is None:
            print(f"  {task}: no variation0 episode, skipped")
            continue
        fig, frac, unknown_px, instr = sheet_for_task(ds, task, ep)
        if fig is None:
            print(f"  {task}: no scene_segments.json, skipped")
            continue
        out = os.path.join(OUTDIR, f"{task}.png")
        fig.savefig(out, dpi=90, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        summary.append((task, frac, unknown_px))
        if unknown_px:
            bad.append(task)
        print(f"  wrote {os.path.relpath(out, REPO)}  unknown={unknown_px}  \"{instr[:50]}\"")

    # Summary: role composition per task, so a missing role is visible without opening 16 files.
    fig, ax = plt.subplots(figsize=(13, 0.42 * len(summary) + 2.4))
    tasks = [s[0] for s in summary]
    left = np.zeros(len(summary))
    for i, role in enumerate(SCENE_ROLES):
        vals = np.array([s[1][role] for s in summary])
        ax.barh(tasks, vals, left=left, color=SCENE_ROLE_PALETTE[i] / 255.0,
                edgecolor="white", linewidth=0.4, label=role)
        left += vals
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("pixel share")
    ax.set_title(
        "scene_roles composition per task (variation0, view1+view2, 4 frames)\n"
        "any orange = an unmapped handle reached the training target",
        fontsize=11,
    )
    ax.legend(ncol=9, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "_summary.png")
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(out, REPO)}")

    if bad:
        print(f"\nUNKNOWN pixels present in: {bad}")
        return 1
    print(f"\n{len(summary)} sheets, zero unknown pixels. Now LOOK at them (see module docstring).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
