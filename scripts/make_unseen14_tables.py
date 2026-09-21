#!/usr/bin/env python
"""Closed-loop tables for the 14-task RLBench tree and the ManiSkill3 blocks.

Supersedes make_pertask_table.py, which is hardwired to the old `arm6_alltasks` tags, the
6-task tree and one simulator. This reads whatever has landed so far, so it is safe to run
mid-campaign: incomplete cells print as x/n with n < 20 and are flagged.

THREE THINGS IT DOES THAT A MEAN OVER TASKS DOES NOT:

  * separates the two zeros. GT-replay runs each scene's own demonstration through the
    identical controller, so a task near zero there is a limit of the actuation harness. Those
    tasks are listed but excluded from the gated aggregate.
  * pairs the models. `trial_seed()` depends only on (task, variation, trial), so arm7 and
    arm0 face identical scenes; the comparison is an exact McNemar on the discordant pairs,
    not two independent proportions.
  * reports Wilson intervals, which do not go negative at the low rates most of these tasks
    sit at.

    python scripts/make_unseen14_tables.py            # readable
    python scripts/make_unseen14_tables.py --latex    # tab:pertask rows
"""
import argparse, collections, glob, json, os, sys
from math import comb
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RLB = os.path.join(REPO, "reports", "closedloop_unseen14")
MS = os.path.join(REPO, "reports", "closedloop_maniskill")
# Whichever checkpoints actually have results, in a fixed order so two runs of this
# script never disagree about column order. Auto-detected rather than hardcoded: the
# campaign's model set changed twice (6k -> 4k, official trimmed to the untested tasks)
# and a stale list silently drops a whole column.
PREFERRED = ["official", "arm0_6k", "arm7_6k", "arm0_4k", "arm7_4k"]


def _present_models():
    import re
    seen = set()
    for d, pat in ((RLB, r"rollout_u14_(.+?)_([a-z0-9_]+)\.json"),
                   (MS, r"rollout_(?:ms|msv1)_(.+?)_([a-z0-9_]+)\.json")):
        for f in glob.glob(os.path.join(d, "rollout_*.json")):
            for m in PREFERRED:
                if f"_{m}_" in os.path.basename(f):
                    seen.add(m)
    return [m for m in PREFERRED if m in seen] or PREFERRED


MODELS = _present_models()
FLOOR_MAX = 0.25   # GT-replay at or below this is a harness floor, not a model result


def load(path):
    """-> {(task, trial): success}, plus the task set, or {} if the file is absent."""
    if not os.path.exists(path):
        return {}
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return {}
    return {(r["task"], r["trial"]): bool(r.get("success")) for r in d.get("results", [])}


def block(name, tag_of, gt_tag_of, tasks):
    """One table block: per-task cells for every model plus the GT gate."""
    gt = {}
    for t in tasks:
        gt.update(load(gt_tag_of(t)))
    cells = {m: {} for m in MODELS}
    for m in MODELS:
        for t in tasks:
            cells[m].update(load(tag_of(m, t)))
    return {"name": name, "tasks": tasks, "gt": gt, "cells": cells}


def rate(d, task=None):
    v = [s for (t, _), s in d.items() if task is None or t == task]
    return sum(v), len(v)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p, den = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(a, b):
    """Two-sided exact McNemar over the pairs present in BOTH dicts."""
    keys = sorted(set(a) & set(b))
    n01 = sum(1 for k in keys if not a[k] and b[k])
    n10 = sum(1 for k in keys if a[k] and not b[k])
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0, len(keys)
    lo = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(lo + 1)) / 2 ** n)
    return n01, n10, p, len(keys)


def report(b, latex=False):
    print(f"\n===== {b['name']} =====")
    gated, floors = [], []
    for t in b["tasks"]:
        gk, gn = rate(b["gt"], t)
        (floors if gn and gk / gn <= FLOOR_MAX else gated).append(t)
    hdr = f"{'task':28s} {'GT':>8s}" + "".join(f" {m:>10s}" for m in MODELS)
    print(hdr)
    for t in b["tasks"]:
        gk, gn = rate(b["gt"], t)
        row = f"{t:28s} {f'{gk}/{gn}' if gn else '-':>8s}"
        for m in MODELS:
            k, n = rate(b["cells"][m], t)
            row += f" {f'{k}/{n}' if n else '-':>10s}"
        if t in floors:
            row += "   <- harness floor"
        print(row)
    print("-" * len(hdr))
    for label, tasks in (("aggregate (all)", b["tasks"]), ("aggregate (gated)", gated)):
        row = f"{label:28s}"
        gk = gn = 0
        for t in tasks:
            k, n = rate(b["gt"], t); gk += k; gn += n
        row += f" {f'{gk}/{gn}' if gn else '-':>8s}"
        for m in MODELS:
            k = n = 0
            for t in tasks:
                kk, nn = rate(b["cells"][m], t); k += kk; n += nn
            row += f" {f'{k}/{n}' if n else '-':>10s}"
        print(row)
        if tasks:
            for m in MODELS:
                k = n = 0
                for t in tasks:
                    kk, nn = rate(b["cells"][m], t); k += kk; n += nn
                if n:
                    lo, hi = wilson(k, n)
                    print(f"    {m:12s} {100*k/n:5.1f}%  95% CI [{100*lo:.1f}, {100*hi:.1f}]  n={n}")
    if floors:
        print(f"  harness floors excluded from the gated aggregate: {', '.join(floors)}")

    a7, a0 = b["cells"]["arm7_6k"], b["cells"]["arm0_6k"]
    gatedset = set(gated)
    a7g = {k: v for k, v in a7.items() if k[0] in gatedset}
    a0g = {k: v for k, v in a0.items() if k[0] in gatedset}
    if a7g and a0g:
        n01, n10, p, npair = mcnemar_exact(a0g, a7g)
        print(f"  McNemar arm7 vs arm0 (gated, paired scenes): arm7-only wins {n01}, "
              f"arm0-only wins {n10}, exact two-sided p={p:.4f}, {npair} pairs")


def export(blocks, out_dir):
    """Write a machine-readable CSV and a human-readable Markdown snapshot.

    One row per (block, task, model) cell plus the GT gate, so a cell can never be read
    without the ceiling it has to be read against.
    """
    import csv, datetime
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    csv_path = os.path.join(out_dir, f"closedloop_{stamp}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block", "task", "gt_success", "gt_n", "harness_floor",
                    "model", "success", "n", "rate_pct"])
        for b in blocks:
            gated = [t for t in b["tasks"]
                     if not (rate(b["gt"], t)[1] and rate(b["gt"], t)[0] / rate(b["gt"], t)[1] <= FLOOR_MAX)]
            for t in b["tasks"]:
                gk, gn = rate(b["gt"], t)
                for m in MODELS:
                    k, n = rate(b["cells"][m], t)
                    if not n:
                        continue
                    w.writerow([b["name"], t, gk, gn, t not in gated, m, k, n,
                                round(100 * k / n, 1)])
    md_path = os.path.join(out_dir, f"closedloop_{stamp}.md")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for b in blocks:
            report(b)
    with open(md_path, "w") as f:
        f.write(f"# Closed-loop results — snapshot {stamp}\n\n")
        f.write("Generated by `scripts/make_unseen14_tables.py --export`. Cells are\n"
                "successes/trials; a campaign still running shows n < 20. **GT** is the\n"
                "ground-truth-replay ceiling: each scene's own demonstration through the same\n"
                "controller, so a task near zero there is a limit of the actuation harness and\n"
                "is excluded from the gated aggregate.\n\n```\n")
        f.write(buf.getvalue())
        f.write("```\n")
    print(f"\nwrote {csv_path}\n      {md_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--export", metavar="DIR", default=None,
                    help="also write a timestamped CSV + Markdown snapshot into DIR")
    a = ap.parse_args()

    rlb_tasks = ["close_box", "close_drawer", "close_microwave", "take_lid_off_saucepan",
                 "toilet_seat_down", "basketball_in_hoop", "meat_on_grill", "light_bulb_out",
                 "take_item_out_of_drawer", "take_money_out_safe", "put_rubbish_in_bin",
                 "stack_cups", "open_window", "wipe_desk"]
    # The corrected GT protocol (hold the final pose through the same budget the model gets)
    # is `gtreplay2`; fall back to the first sweep where it has not landed yet.
    def gt_rlb(t):
        p2 = f"{RLB}/rollout_u14_gtreplay2_{t}.json"
        return p2 if os.path.exists(p2) else f"{RLB}/rollout_u14_gtreplay_{t}.json"

    blocks = [
        block("RLBench, 14 never-trained tasks (clean domain)",
              lambda m, t: f"{RLB}/rollout_u14_{m}_{t}.json", gt_rlb, rlb_tasks),
        block("ManiSkill3, held-out TASKS",
              lambda m, t: f"{MS}/rollout_ms_{m}_{t}.json",
              lambda t: f"{MS}/rollout_msgt2_{t}_v0.json",
              ["pull_cube", "place_sphere", "lift_peg_upright"]),
        block("ManiSkill3, trained tasks / held-out seeds (variation1)",
              lambda m, t: f"{MS}/rollout_msv1_{m}_{t}.json",
              lambda t: f"{MS}/rollout_msgt2_{t}_v1.json",
              ["pick_cube", "stack_cube", "pull_cube_tool", "push_cube", "stack_pyramid",
               "peg_insertion_side", "plug_charger"]),
    ]
    for b in blocks:
        report(b, a.latex)
    if a.export:
        export(blocks, a.export)


if __name__ == "__main__":
    main()
