"""Per-task segmentation pixel-occupancy statistics (SEGMENTATION_SCENE_ROLES_PLAN.md §4).

    python scripts/seg_pixel_stats.py --out reports/seg_pixel_stats.csv

Answers "how sparse is the current supervision, and how much denser does scene_roles make it",
per task, from the real handle maps. The plan's loss-weight design (§8.4) reads off this table,
so it is a committed script rather than an ad-hoc sample: the first draft's numbers came from
16 tasks x 3 episodes x 3 frames and understated the cross-task spread by a lot.

Reports two things per task that must not be collapsed into one number:
  * the MEDIAN across episodes -- the typical case;
  * the SPREAD across tasks -- measured at 20x (reach_and_drag has 1% foreground objects,
    put_groceries_in_cupboard has 33%). Any globally-fixed loss weight misfits one end, which is
    why §8.4 normalises per sample instead.
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


from training.percep.scene_segments_gen import resolve_roles  # noqa: E402


def episode_stats(ep: str, task: str, views, max_frames: int):
    with open(os.path.join(ep, "handles.json")) as f:
        handles = json.load(f)
    instances, _ = resolve_roles(task, handles)
    role_of = {h: inst["base_role"] for inst in instances.values() for h in inst["handles"]}

    tgt = []
    sp = os.path.join(ep, "seg_targets.json")
    if os.path.exists(sp):
        with open(sp) as f:
            for g in json.load(f).get("referring", {}).values():
                tgt += list(g.get("handles", []))

    frac = collections.Counter()
    total = 0
    for v in views:
        mp = os.path.join(ep, f"view{v}", "mask.npz")
        if not os.path.exists(mp):
            continue
        m = np.load(mp)["mask"]
        if max_frames and m.shape[0] > max_frames:
            m = m[:: max(1, m.shape[0] // max_frames)]
        total += m.size
        ids, counts = np.unique(m, return_counts=True)
        for i, c in zip(ids.tolist(), counts.tolist()):
            frac[role_of.get(i, "unknown")] += c
            if i in tgt:
                frac["referred_target"] += c
    if not total:
        return None
    return {k: v / total for k, v in frac.items()}, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "data", "rlbench_selfgen_512_aug"),
                    help="512_aug; the bare rlbench_selfgen symlink points at the deleted v2 tree")
    ap.add_argument("--out", default=os.path.join(REPO, "reports", "seg_pixel_stats.csv"))
    ap.add_argument("--views", default="1,2")
    ap.add_argument("--episodes-per-task", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=40, help="0 = every frame")
    args = ap.parse_args()

    views = [int(v) for v in args.views.split(",")]
    ROLES = ["referred_target", "target", "goal", "tool", "fixture", "distractor",
             "robot_arm", "gripper", "background", "unknown"]

    per_task = collections.defaultdict(lambda: collections.defaultdict(list))
    tasks = sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
    for task in tasks:
        eps = sorted(glob.glob(os.path.join(args.root, task, "variation*", "episodes", "episode*")))
        for ep in eps[: args.episodes_per_task]:
            r = episode_stats(ep, task, views, args.max_frames)
            if r is None:
                continue
            fracs, _ = r
            for role in ROLES:
                per_task[task][role].append(fracs.get(role, 0.0))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    hdr = ["task", "n_episodes"] + [f"{r}_median_pct" for r in ROLES]
    lines = [",".join(hdr)]
    print(f"{'task':32s} " + " ".join(f"{r[:9]:>9s}" for r in ROLES))
    spread = {}
    for task in tasks:
        if task not in per_task:
            continue
        n = len(per_task[task][ROLES[0]])
        meds = [float(np.median(per_task[task][r])) for r in ROLES]
        spread[task] = meds
        lines.append(",".join([task, str(n)] + [f"{m * 100:.4f}" for m in meds]))
        print(f"{task:32s} " + " ".join(f"{m * 100:8.2f}%" for m in meds))

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")

    fg = {t: 1.0 - m[ROLES.index("background")] for t, m in spread.items()}
    tg = {t: m[ROLES.index("referred_target")] for t, m in spread.items()}
    print(f"\nwrote {args.out}")
    print(f"\nreferred_target (today's supervision): median {np.median(list(tg.values())) * 100:.2f}%, "
          f"min {min(tg.values()) * 100:.2f}% ({min(tg, key=tg.get)}), "
          f"max {max(tg.values()) * 100:.2f}% ({max(tg, key=tg.get)})")
    print(f"non-background (scene_roles supervision): median {np.median(list(fg.values())) * 100:.2f}%, "
          f"min {min(fg.values()) * 100:.2f}% ({min(fg, key=fg.get)}), "
          f"max {max(fg.values()) * 100:.2f}% ({max(fg, key=fg.get)})")
    print(f"cross-task spread of non-background: {max(fg.values()) / max(min(fg.values()), 1e-9):.1f}x "
          f"-- a single global loss weight misfits one end (plan §8.4)")


if __name__ == "__main__":
    main()
