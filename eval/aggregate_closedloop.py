"""Turn rollout JSONs into the success-rate table and the paired comparison between arms.

Two arms are compared with an EXACT McNemar test on the paired outcomes, not with a two-sample
test on the two success rates. The pairing is what buys the power: both arms are scored on the
same `scene_seed`s, so the discordant pairs (one arm solved it, the other did not) carry all
the information, and the many scenes where both succeed or both fail contribute nothing but
noise to an unpaired test. With 20 trials x 4 tasks the discordant count is small, which is
exactly the regime where the exact binomial form matters and the chi-square approximation does
not hold.

Also reports, because a bare success rate hides the two ways a rollout campaign lies:
  * `ik_fail_rate`  -- unreachable targets, i.e. the harness's failure rather than the model's;
  * `timeout_frac`  -- ran out of steps rather than doing the wrong thing;
  * `r_peak_mean`   -- whether the model was still drawing decodable action blobs at all.

Usage:
    python eval/aggregate_closedloop.py reports/closedloop/rollout_*.json
    python eval/aggregate_closedloop.py --compare arm0.json arm1.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def load(path: str) -> Tuple[Dict, List[Dict]]:
    """Load one rollout JSON, or merge a comma-separated list of shards.

    A 100-rollout campaign is sharded across GPUs by task, so one arm arrives as several
    files. Merging here (rather than by hand) keeps the (task, variation, trial) keys and the
    scene_seeds intact, which is what the paired test depends on.
    """
    paths = [p for p in path.split(",") if p]
    args: Dict = {}
    results: List[Dict] = []
    seen = set()
    for p in paths:
        with open(p) as f:
            blob = json.load(f)
        args = args or blob.get("args", {})
        for r in blob["results"]:
            key = (r["task"], r["variation"], r["trial"])
            if key in seen:
                raise SystemExit(
                    f"duplicate cell {key} across shards -- the shards overlap, so merging "
                    f"them would double-count. Check the --tasks split.")
            seen.add(key)
            results.append(r)
    if len(paths) > 1:
        print(f"merged {len(paths)} shards -> {len(results)} rollouts")
    return args, results


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value.

    `b` = pairs where A succeeded and B failed, `c` = the reverse. Under H0 each discordant
    pair is a fair coin, so the p-value is the two-sided binomial tail. Returns 1.0 when there
    are no discordant pairs (nothing to distinguish the arms with).
    """
    n = b + c
    if n == 0:
        return 1.0
    from scipy.stats import binomtest

    return float(binomtest(b, n, 0.5, alternative="two-sided").pvalue)


def bootstrap_ci(paired: List[Tuple[bool, bool]], n_boot: int = 10000,
                 seed: int = 0) -> Tuple[float, float]:
    """95% percentile CI for (rate_A - rate_B) in percentage points, resampling SCENES."""
    rng = np.random.default_rng(seed)
    a = np.array([p[0] for p in paired], dtype=float)
    b = np.array([p[1] for p in paired], dtype=float)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = (a[idx].mean(axis=1) - b[idx].mean(axis=1)) * 100
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def summarize(results: List[Dict], label: str = "") -> Dict:
    by_task: Dict[str, List[Dict]] = {}
    for r in results:
        by_task.setdefault(r["task"], []).append(r)

    rows = []
    for task, rs in sorted(by_task.items()):
        peaks = [d.get("r_peak_mean") for r in rs for d in r.get("policy_debug", [])
                 if d.get("r_peak_mean") is not None]
        rows.append({
            "task": task,
            "n": len(rs),
            "success_rate": round(float(np.mean([r["success"] for r in rs])) * 100, 1),
            "ik_fail_rate": round(float(np.mean([r.get("ik_fail_rate", 0) for r in rs])) * 100, 2),
            "timeout_frac": round(float(np.mean([r.get("stop_reason") == "timeout" for r in rs])), 3),
            "mean_replans": round(float(np.mean([r.get("replans", 0) for r in rs])), 2),
            "mean_steps": round(float(np.mean([r.get("steps", 0) for r in rs])), 1),
            "r_peak_mean": round(float(np.mean(peaks)), 1) if peaks else None,
        })
    overall = (round(float(np.mean([r["success"] for r in results])) * 100, 1)
               if results else None)
    return {"label": label, "n": len(results), "success_rate": overall, "per_task": rows}


def print_table(summary: Dict) -> None:
    overall = (f"{summary['success_rate']}%"
               if summary["success_rate"] is not None else "n/a")
    print(f"\n=== {summary['label'] or 'summary'} "
          f"(n={summary['n']}, overall {overall}) ===")
    hdr = (f"{'task':20s} {'n':>3s} {'succ%':>6s} {'ik_fail%':>9s} {'timeout':>8s} "
           f"{'replans':>8s} {'steps':>7s} {'r_peak':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in summary["per_task"]:
        rp = f"{r['r_peak_mean']:.1f}" if r["r_peak_mean"] is not None else "-"
        print(f"{r['task']:20s} {r['n']:3d} {r['success_rate']:6.1f} {r['ik_fail_rate']:9.2f} "
              f"{r['timeout_frac']:8.3f} {r['mean_replans']:8.2f} {r['mean_steps']:7.1f} {rp:>7s}")


def compare(path_a: str, path_b: str, out: Optional[str] = None) -> Dict:
    args_a, res_a = load(path_a)
    args_b, res_b = load(path_b)
    label_a = args_a.get("tag") or os.path.basename(path_a)
    label_b = args_b.get("tag") or os.path.basename(path_b)

    key = lambda r: (r["task"], r["variation"], r["trial"])  # noqa: E731
    map_a = {key(r): r for r in res_a}
    map_b = {key(r): r for r in res_b}
    shared = sorted(set(map_a) & set(map_b))
    if not shared:
        raise SystemExit("no (task, variation, trial) cells in common -- nothing to pair")

    # The pairing is only valid if both arms actually saw the same world.
    mismatched = [k for k in shared
                  if map_a[k].get("scene_seed") != map_b[k].get("scene_seed")]
    if mismatched:
        raise SystemExit(
            f"{len(mismatched)} paired cells have different scene_seed (e.g. {mismatched[0]}). "
            f"The arms were scored on different scenes, so a paired test is invalid.")

    print_table(summarize(res_a, label_a))
    print_table(summarize(res_b, label_b))

    print(f"\n=== paired comparison on {len(shared)} shared scenes ===")
    print(f"{'task':20s} {'n':>3s} {label_a[:10]:>11s} {label_b[:10]:>11s} "
          f"{'b':>3s} {'c':>3s} {'diff pp':>8s} {'p':>8s}  95% CI")
    per_task = {}
    for task in sorted({k[0] for k in shared}):
        cells = [k for k in shared if k[0] == task]
        paired = [(bool(map_a[k]["success"]), bool(map_b[k]["success"])) for k in cells]
        b = sum(1 for x, y in paired if x and not y)
        c = sum(1 for x, y in paired if y and not x)
        ra = np.mean([p[0] for p in paired]) * 100
        rb = np.mean([p[1] for p in paired]) * 100
        p = mcnemar_exact(b, c)
        lo, hi = bootstrap_ci(paired)
        per_task[task] = {"n": len(paired), "rate_a": ra, "rate_b": rb, "b": b, "c": c,
                          "p_value": p, "ci95": [lo, hi]}
        print(f"{task:20s} {len(paired):3d} {ra:10.1f}% {rb:10.1f}% {b:3d} {c:3d} "
              f"{ra-rb:+8.1f} {p:8.4f}  [{lo:+.1f}, {hi:+.1f}]")

    paired_all = [(bool(map_a[k]["success"]), bool(map_b[k]["success"])) for k in shared]
    b = sum(1 for x, y in paired_all if x and not y)
    c = sum(1 for x, y in paired_all if y and not x)
    ra = np.mean([p[0] for p in paired_all]) * 100
    rb = np.mean([p[1] for p in paired_all]) * 100
    p = mcnemar_exact(b, c)
    lo, hi = bootstrap_ci(paired_all)
    print("-" * 88)
    print(f"{'POOLED':20s} {len(paired_all):3d} {ra:10.1f}% {rb:10.1f}% {b:3d} {c:3d} "
          f"{ra-rb:+8.1f} {p:8.4f}  [{lo:+.1f}, {hi:+.1f}]")

    print(f"\ndiscordant pairs: {b + c} of {len(paired_all)}")
    if b + c == 0:
        verdict = ("NO DISCRIMINATION -- the two arms succeeded and failed on exactly the same "
                   "scenes. This grid cannot separate them at any sample size; report it as "
                   "such rather than as 'no difference'.")
    elif b + c < 6:
        verdict = (f"UNDERPOWERED -- only {b+c} discordant pairs. Even a real effect could not "
                   f"reach p<0.05 here (the smallest attainable two-sided p with {b+c} "
                   f"discordant pairs is {mcnemar_exact(b+c, 0):.3f}). Add trials before "
                   f"concluding anything.")
    elif p < 0.05:
        better = label_a if b > c else label_b
        verdict = f"DIFFERENT (p={p:.4f}) -- {better} solved more of the discordant scenes."
    else:
        verdict = (f"NO SIGNIFICANT DIFFERENCE (p={p:.4f}); the 95% CI on the difference is "
                   f"[{lo:+.1f}, {hi:+.1f}] pp, so effects outside that range are excluded.")
    print(f"VERDICT: {verdict}")

    blob = {"a": label_a, "b": label_b, "n_shared": len(shared), "per_task": per_task,
            "pooled": {"rate_a": ra, "rate_b": rb, "b": b, "c": c, "p_value": p,
                       "ci95": [lo, hi]}, "verdict": verdict}
    if out:
        with open(out, "w") as f:
            json.dump(blob, f, indent=2)
        print(f"wrote {out}")
    return blob


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--compare", action="store_true",
                    help="treat the two paths as arms to compare pairwise")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.compare:
        if len(args.paths) != 2:
            ap.error("--compare takes exactly two files")
        compare(args.paths[0], args.paths[1], args.out)
        return

    for path in args.paths:
        a, res = load(path)
        print_table(summarize(res, a.get("tag") or os.path.basename(path)))


if __name__ == "__main__":
    main()
