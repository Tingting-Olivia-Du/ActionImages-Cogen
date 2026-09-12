#!/usr/bin/env python
"""Whole-tree audit of the dense scene-role segmentation target.

Run: python scripts/audit_scene_roles.py [--root data/rlbench_selfgen_512_aug]

WHY THIS EXISTS. `scene_roles` is the training target for `video+segmentation` under
`--segmentation_mode scene_roles`, and its one hard invariant (SEGMENTATION_SCENE_ROLES_PLAN.md
12.1) is that the `unknown` role -- orange, meaning "a handle nobody annotated" -- never appears
in training data. That invariant was being violated on essentially every sample:
`handles.json` carries RLBench's "no object" sentinel as `"16777215": ""`, `resolve_roles`
dropped it as unmapped, `gen_dataset.py` truncated it to 65535 in the uint16 mask, and
`build_role_lut` therefore painted the sky orange -- 5.20% of ALL pixels, up to 52.89% of a
single view. `tests/test_scene_seg_dataset.py` could not catch it because its data root pointed
at a symlink that no longer resolves.

So the check belongs somewhere that reads the ACTUAL tree end to end rather than a handful of
samples, and that returns an exit code a launcher can gate on. That is this script.

It reads mask.npz directly instead of going through RLBenchSelfgenDataset because the dataset
resizes to model resolution and samples `num_frames` per episode -- both of which can hide a
small unmapped region. Every frame at native resolution, or the audit is not an audit.

Exit 0 iff zero `unknown` pixels tree-wide.
"""
import argparse
import collections
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.percep.seg_codec import (  # noqa: E402
    SCENE_ROLES,
    UNKNOWN_LABEL,
    build_role_lut,
)

from training.percep.scene_segments_gen import TASK_ROLES  # noqa: E402

# Facts about this tree that look like coverage bugs and are not. Both come from
# gen_dataset.py:dump_handle_names sampling only frames [0, T//2, T-1]: an object that is
# occluded in all three never reaches handles.json, so it never reaches scene_segments.json.
# That costs a role its colour on those episodes; it does NOT leave pixels unmapped, because the
# object is not rendered in those frames either. Recorded so the report distinguishes them from a
# real gap rather than the reader having to remember.
KNOWN_SPARSE_ROLES = {
    # scene_segments_gen.py:112 -- reach_and_drag's drag destination is occluded ~94% of the time.
    ("reach_and_drag", "goal"): "target0 occluded in most episodes (documented upstream)",
}


def _episode_roles(ep):
    """-> (task, episode, {role: pixel count}, [(view, unknown_px, total_px, [handles])])."""
    task = ep.split(os.sep)[-4]
    with open(os.path.join(ep, "scene_segments.json")) as f:
        scene = json.load(f)
    handle_to_role = {}
    for inst in scene.get("instances", {}).values():
        role = inst.get("base_role")
        if role is None:
            continue
        for h in inst.get("handles", []):
            handle_to_role[int(h)] = role
    if not handle_to_role:
        return task, ep, {}, [("<no instances>", -1, 0, [])]

    # base_role only, no instruction overlay: this audits the LUT and the handle map, and the
    # target promotion is a relabel of already-mapped handles that cannot create unknown pixels.
    # `target` therefore reads 0 here by construction -- that is expected, not a finding.
    lut = build_role_lut(handle_to_role)

    counts = collections.Counter()
    offenders = []
    for vd in sorted(glob.glob(os.path.join(ep, "view*"))):
        mp = os.path.join(vd, "mask.npz")
        if not os.path.exists(mp):
            continue
        mask = np.load(mp)["mask"]
        labels = lut[mask.astype(np.intp)]
        c = np.bincount(labels.ravel(), minlength=len(SCENE_ROLES))
        for i, r in enumerate(SCENE_ROLES):
            counts[r] += int(c[i])
        unk = int(c[UNKNOWN_LABEL])
        if unk:
            bad_handles = sorted(int(h) for h in np.unique(mask[labels == UNKNOWN_LABEL]))
            offenders.append((os.path.basename(vd), unk, int(labels.size), bad_handles[:8]))
    return task, ep, dict(counts), offenders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/rlbench_selfgen_512_aug")
    ap.add_argument("--views", default="all", help="'all' or a comma list like 'view1,view2'")
    ap.add_argument("--limit", type=int, default=0, help="cap episodes (debugging this script)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    eps = sorted(glob.glob(os.path.join(args.root, "*", "variation*", "episodes", "episode*")))
    if not eps:
        sys.exit(f"no episodes under {args.root}")
    missing = [e for e in eps if not os.path.exists(os.path.join(e, "scene_segments.json"))]
    if args.limit:
        eps = eps[:: max(1, len(eps) // args.limit)][: args.limit]
    print(f"root      : {args.root}")
    print(f"episodes  : {len(eps)}" + (f"  (sampled from {len(eps)})" if args.limit else ""))
    print(f"missing scene_segments.json: {len(missing)}")

    totals = collections.Counter()
    offenders = []
    per_task_roles = collections.defaultdict(collections.Counter)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, (task, ep, counts, offs) in enumerate(pool.map(_episode_roles, eps, chunksize=4), 1):
            totals.update(counts)
            for r, v in counts.items():
                if v:
                    per_task_roles[task][r] += v
            for view, unk, size, handles in offs:
                offenders.append((task, os.path.basename(ep), view, unk, size, handles))
            if n % 100 == 0:
                print(f"  ... {n}/{len(eps)}", flush=True)

    tot = sum(totals.values())
    print(f"\npixels scanned: {tot / 1e6:.1f}M")
    print("global role distribution:")
    for r in SCENE_ROLES:
        print(f"  {r:12s} {totals[r] / tot * 100:9.4f}%   {totals[r]}")

    # Only report a role as missing if the task's own TASK_ROLES table DECLARES it. Listing every
    # role a task lacks is noise: close_jar has no `tool` because closing a jar uses no implement,
    # and saying so on 15 of 16 tasks buries the one line that means something.
    print("\nroles declared by TASK_ROLES but with zero pixels tree-wide:")
    any_missing = False
    for task in sorted(per_task_roles):
        declared = set(TASK_ROLES.get(task, {}).values()) | {"background", "robot_arm", "gripper"}
        absent = [r for r in SCENE_ROLES
                  if r in declared and r != "target" and not per_task_roles[task][r]]
        if not absent:
            continue
        any_missing = True
        note = "".join(
            f"   [known: {KNOWN_SPARSE_ROLES[(task, r)]}]" for r in absent if (task, r) in KNOWN_SPARSE_ROLES
        )
        print(f"  {task:32s} {','.join(absent)}{note}")
    if not any_missing:
        print("  (none)")

    print(f"\n(episode, view) with `unknown` pixels: {len(offenders)}")
    for task, ep, view, unk, size, handles in sorted(offenders, key=lambda o: -o[3])[:30]:
        print(f"  {task:30s} {ep:12s} {view:6s} {unk:10d} ({unk / size * 100:6.3f}%)  handles={handles}")

    if totals[SCENE_ROLES[UNKNOWN_LABEL]] or missing:
        print("\nSCENE_ROLES_AUDIT_FAILED")
        return 1
    print("\nSCENE_ROLES_AUDIT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
