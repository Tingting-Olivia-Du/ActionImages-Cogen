"""Read the tag-swap / f0f0 / repeat runs and print the matrix with a noise floor.

The whole point is the comparison of three effect sizes measured on ONE checkpoint:
  tag effect      = spread across a row (anchor fixed, only the text changes)
  anchor effect   = diagonal vs f0f0 (text fixed and correct, anchor removed)
  noise floor     = the same cell run twice (nothing changes)
A tag effect that is not comfortably larger than the noise floor is not an effect.
"""
import json, os, sys

ROOT = "reports/tag_swap"
GRID = "reports/modality_mode_grid"
BETTER = {"absrel": "lower", "macro_miou": "higher", "mean_cos": "higher"}
TAGS = ["depth", "segmentation", "normal", "action"]


def load(tag):
    p = os.path.join(ROOT, tag, "metrics.json")
    return json.load(open(p))["results"] if os.path.exists(p) else {}


def f0f0(tag):
    p = os.path.join(GRID, tag, "metrics.json")
    if not os.path.exists(p):
        return {}
    out = {}
    for k, rec in json.load(open(p))["results"].items():
        tpl, mode = k.split("|")
        if mode != "f0f0":
            continue
        e = next((s for s in rec["segments"]
                  if s["modality"] != "video" and s["view"] == 0), None)
        if e and e.get("value") is not None:
            out[e["modality"]] = e["value"]
    return out


def main():
    main_r = load(sys.argv[1] if len(sys.argv) > 1 else "arm6_9000")
    rep_r = load("arm6_9000_repeat")
    zero = f0f0("arm6_9000_f0f0")
    if not main_r:
        sys.exit("no tag-swap results yet")

    targets = []
    for k in main_r:
        t = main_r[k]["target"]
        if t not in targets:
            targets.append(t)

    print(f"{'target':14s} {'metric':11s} " + " ".join(f"{t:>10s}" for t in TAGS)
          + f" {'跨度':>8s} {'对角名次':>9s}")
    print("-" * 96)
    for t in targets:
        row = {main_r[k]["tag"]: main_r[k]["value"] for k in main_r if main_r[k]["target"] == t}
        met = next(main_r[k]["metric"] for k in main_r if main_r[k]["target"] == t)
        vals = [row.get(g) for g in TAGS]
        ok = [v for v in vals if v is not None]
        if not ok:
            continue
        spread = (max(ok) - min(ok)) / abs(sum(ok) / len(ok))
        ordered = sorted(ok, reverse=(BETTER[met] == "higher"))
        rank = ordered.index(row[t]) + 1 if t in row else None
        cells = " ".join(f"{v:10.5f}" if v is not None else f"{'—':>10s}" for v in vals)
        print(f"{t:14s} {met:11s} {cells} {spread:7.2%} {f'{rank}/{len(ok)}':>9s}")

    print()
    if zero:
        print("同一 checkpoint 上的三种效应")
        print(f"  {'target':14s} {'对角(有anchor)':>16s} {'f0f0(无anchor)':>16s} {'anchor 效应':>13s} {'tag 效应':>11s}")
        for t in targets:
            row = {main_r[k]["tag"]: main_r[k]["value"] for k in main_r if main_r[k]["target"] == t}
            met = next(main_r[k]["metric"] for k in main_r if main_r[k]["target"] == t)
            if t not in row or t not in zero:
                continue
            d, z = row[t], zero[t]
            ok = [v for v in row.values() if v is not None]
            tag_eff = (max(ok) - min(ok)) / abs(d)
            anc_eff = abs(z - d) / abs(d)
            print(f"  {t:14s} {d:16.5f} {z:16.5f} {anc_eff:12.1%} {tag_eff:10.2%}"
                  + f"   → anchor 是 tag 的 {anc_eff/tag_eff:,.0f} 倍" if tag_eff else "")
    else:
        print("f0f0 补测尚未完成 —— anchor 效应无法在同一 checkpoint 上给出")

    print()
    if rep_r:
        print("确定性噪声底（同一格跑两次）")
        n_same = n_tot = 0
        worst = 0.0
        for k, r in rep_r.items():
            if k not in main_r:
                continue
            a, b = main_r[k]["value"], r["value"]
            n_tot += 1
            if a == b:
                n_same += 1
            else:
                worst = max(worst, abs(a - b) / abs(a))
            print(f"  {k:26s} {a:10.5f} vs {b:10.5f}   {'逐位相同' if a==b else f'差 {abs(a-b)/abs(a):.3%}'}")
        print(f"  → {n_same}/{n_tot} 逐位相同；最大相对差 {worst:.4%}")
        print(f"  → 噪声底 {worst:.4%}，因此上表 0.3–1.7% 的跨度"
              + ("是真实的文本扰动效应（但方向与正确性无关）" if worst < 0.003 else "无法与噪声区分"))
    else:
        print("重复实验尚未完成 —— 噪声底未知")


if __name__ == "__main__":
    main()
