#!/usr/bin/env python
"""Dump the frame slots Figure 1 expects, straight from the loader.

WHAT THESE ARE. Ground-truth training streams, not model output: the loader's own codec images
for one episode, which is what the model is trained to produce. They stand in for generated
frames while the runs are still training, so the figure's layout can be read and discussed.
The caption must say so until real generations replace them -- an unlabelled placeholder in a
teaser is indistinguishable from a claimed result.

Frames come from `ds.getitem(idx, force_template=...)["streams"][key]`, shape [3, 2T, H, W] in
[-1,1], the same tensors training consumes. The second half of the time axis is the target view;
we take that one, since the first is the conditioning view.

USAGE (needs the training env)
    /opt/conda/envs/ttd_train/bin/python scripts/make_teaser_frames.py \
        --out ../6a89bc13108d28a2448f410a/Figs/teaser --task close_jar
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = "/workspace/1228_tingting/ActionImages-Cogen"
sys.path.insert(0, REPO)
os.chdir(REPO)

from training.dataset.rlbench_selfgen import RLBenchSelfgenDataset  # noqa: E402

ARM7_MIX = ("video+action@0.4,depth+action@0.2,"
            "segmentation+action@0.2,normal+action@0.2")
# stream key -> (template that carries it, filename stem used by Figs/fig_teaser.tex)
STREAMS = [
    ("video",        "video+action",        "video"),
    ("depth",        "depth+action",        "depth"),
    ("segmentation", "segmentation+action", "seg"),
    ("normal",       "normal+action",       "normal"),
]


def to_png(t, frame, size):
    """[3,2T,H,W] in [-1,1] -> PIL image of one frame from the target-view half."""
    arr = ((t[:, frame].permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr).resize((size, size), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/rlbench_selfgen_512_aug_wide")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", default=None, help="substring of the episode path to pick")
    ap.add_argument("--index", type=int, default=0, help="used when --task is not given")
    ap.add_argument("--frames", default="0,13,26,40", help="anchor,t0,t1,t2 within the window")
    ap.add_argument("--size", type=int, default=192, help="output edge in pixels")
    ap.add_argument("--frame_interval", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7,
                    help="reseeded before every getitem so all four rows share view and window")
    a = ap.parse_args()

    ds = RLBenchSelfgenDataset(
        base_path=a.data, num_frames=41, frame_interval=a.frame_interval,
        height=512, width=512, template_mix=ARM7_MIX,
        segmentation_mode="scene_roles", variations="0", strict_getitem=True,
    )

    idx = a.index
    if a.task:
        hits = [i for i in range(len(ds)) if a.task in str(ds.episodes[i])]
        if not hits:
            raise SystemExit(f"--task {a.task!r} matched no episode")
        idx = hits[len(hits) // 2]

    want = [int(x) for x in a.frames.split(",")]
    a.out.mkdir(parents=True, exist_ok=True)

    for key, template, stem in STREAMS:
        # base.py:193 samples the two views with the global `random`, and the window with it
        # too, so an unseeded second call returns a DIFFERENT camera. Reseed per call or the
        # four rows of the figure are four different shots and the claim it makes is false.
        random.seed(a.seed)
        s = ds.getitem(idx, force_template=template)
        t = s["streams"][key]
        T = t.shape[1] // 2
        names = ["anchor", "t0", "t1", "t2"]
        for name, f in zip(names, want):
            to_png(t, T + min(f, T - 1), a.size).save(a.out / f"{stem}_{name}.png")
        # NOTE the action stream is not in `streams`: it is rendered in-loop from the pose,
        # not loaded from disk, so the figure keeps its schematic action blocks.

    print(f"episode: {ds.episodes[idx]}")
    print(f"wrote {len(STREAMS) * 4} frames to {a.out}")


if __name__ == "__main__":
    main()
