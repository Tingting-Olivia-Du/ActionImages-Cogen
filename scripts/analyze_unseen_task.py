"""Recompute the never-trained-task tier and the unified-vs-specialist control at whatever
sample size is currently on disk.

Two numbers come out of this, and they answer different questions:

  tier-3 means      what tab:generalization's "unseen task" column reports for the unified model
  paired diffs      whether a single-modality specialist holds up equally, which is the control
                    for "is cross-task robustness something unification buys" (it is not, at
                    n=16; this rechecks at n=48)

Paired, because both models are scored on the identical episodes -- comparing overlapping CIs
would throw away most of the power.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.heldout_batch_eval import bootstrap_ci

SPEC = {"depth": "depthspec_2000_unseen_task",
        "segmentation": "segspec_2000_unseen_task",
        "normal": "normalspec_2000_unseen_task"}
UNI = "arm6_10000_unseen_task"


def cells(tag, mod, split="unseen_task"):
    f = f"reports/heldout_batch_eval/{tag}/per_episode.json"
    if not os.path.exists(f):
        return {}
    d = json.load(open(f))["results"]
    return {k.split("|")[0]: v for k, v in d.items()
            if v["modality"] == mod and v["split"] == split and v.get("value") is not None}


def main():
    from scipy.stats import wilcoxon
    rng = np.random.default_rng(42)

    print("══ tier-3(从没训练过的任务)unified 的绝对值 ══")
    for mod in SPEC:
        c = cells(UNI, mod)
        v = [r["value"] for r in c.values()]
        tasks = sorted({e.split("/")[0] for e in c})
        if not v:
            print(f"  {mod:14s} 无数据"); continue
        m, lo, hi = bootstrap_ci(v, 10000, 42)
        print(f"  {mod:14s} {m:.4f} [{lo:.4f},{hi:.4f}]  n={len(v)}  任务数={len(tasks)}")

    print("\n══ 控制实验:unified vs 单模态 specialist(配对)══")
    for mod, spec in SPEC.items():
        A, B = cells(UNI, mod), cells(spec, mod)
        eps = sorted(set(A) & set(B))
        if len(eps) < 4:
            print(f"  {mod:14s} 共同 episode 仅 {len(eps)}"); continue
        a = np.array([A[e]["value"] for e in eps]); b = np.array([B[e]["value"] for e in eps])
        d = a - b
        boot = rng.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5]); p = wilcoxon(a, b).pvalue
        verdict = "CI 跨 0,无可检测差异" if lo < 0 < hi else "★ 有差异"
        print(f"  {mod:14s} n={len(eps):2d}  unified={a.mean():.4f}  spec={b.mean():.4f}  "
              f"差 {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3f}  {verdict}")

    # per-role IoU, only for roles both sides have -- the comparable version of seg
    ur = cells(UNI, "segmentation")
    have = [r for r in ur.values() if r.get("per_role_iou")]
    if have:
        print("\n══ seg per-role IoU(跨任务可比的那部分)══")
        allroles = {}
        for r in have:
            for role, iou in r["per_role_iou"].items():
                allroles.setdefault(role, []).append(iou)
        for role, v in sorted(allroles.items(), key=lambda x: -len(x[1])):
            print(f"  {role:14s} {np.mean(v):.4f}  (出现在 {len(v)}/{len(have)} 个 episode)")
    else:
        print("\n(per_role_iou 尚未记录 —— 需要用新版 score() 重算 seg 单元)")


if __name__ == "__main__":
    main()
