#!/usr/bin/env python
"""Per-task closed-loop table.

The aggregate hides the actual shape of the result: most tasks are zero and a couple carry
the mean. It also has to separate two different zeros -- a task the model cannot do, and a
task the actuation harness cannot express. GT-replay executes each demonstration's own
actions through the identical controller, so a task scoring ~0 there is a harness floor and
is excluded from the aggregate rather than charged to the model.
"""
import json, os, collections, sys
import numpy as np

CL = "reports/closedloop"

def load(tag):
    p = f"{CL}/rollout_{tag}.json"
    if not os.path.exists(p): return {}
    d = json.load(open(p))
    o = collections.defaultdict(lambda: [0, 0])
    for r in d.get("results", []):
        o[r["task"]][1] += 1
        if r.get("success"): o[r["task"]][0] += 1
    return o

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h), min(1, c+h))

def main():
    g0, g1 = load("gt_gate_all_v0"), load("gt_gate_all_v1")
    m0, m1 = load("arm6_alltasks_v0"), load("arm6_alltasks_v1")
    tasks = sorted(set(g0) | set(g1))
    FLOOR = [t for t in tasks if g0.get(t, [0,1])[0] <= 1 and g1.get(t, [0,1])[0] <= 1]
    body, agg0, agg1 = [], [0,0], [0,0]
    for t in tasks:
        floor = t in FLOOR
        cells = []
        for d in (g0, g1, m0, m1):
            k, n = d.get(t, [0, 0])
            cells.append(f"{k}/{n}" if n else "---")
        if not floor:
            agg0[0] += m0.get(t,[0,0])[0]; agg0[1] += m0.get(t,[0,0])[1]
            agg1[0] += m1.get(t,[0,0])[0]; agg1[1] += m1.get(t,[0,0])[1]
        name = t.replace("_", r"\_")
        mark = r"$^{\dagger}$" if floor else ""
        body.append(f"\\texttt{{{name}}}{mark} & {cells[0]} & {cells[1]} & "
                    f"\\textbf{{{cells[2]}}} & \\textbf{{{cells[3]}}} \\\\")
    lo0, hi0 = wilson(*agg0); lo1, hi1 = wilson(*agg1)
    print(r"""\begin{table}[t]
\caption{Closed-loop control, per task. Every trained task with more than one variation is
included; the three-task default of the evaluation harness happens to contain both tasks the
model can do, so an aggregate over it is not representative.
\textbf{GT-replay} runs each demonstration's own actions through the identical
end-effector-pose controller and is the ceiling this harness imposes, not a model result.
$^{\dagger}$ marks a task whose ground-truth replay also fails: the controller cannot express
it, so its zero is excluded from the aggregate rather than charged to the model.
$n{=}10$ trials per cell. Individual cells carry run-to-run noise: re-running an
identical configuration flips $7\%$ of rollouts, because the sampling-based motion planner
is not seeded by us. A cell is therefore $\pm 0.8$ successes, so $2/10$ against $1/10$ is
not a difference; the aggregate is $\pm 2.5$ points.}
\label{tab:pertask}
\centering
\small
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{GT-replay ceiling} & \multicolumn{2}{c}{Unified model} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Task & var.\ 0 & var.\ 1 & var.\ 0 & var.\ 1 \\
\midrule""")
    print("\n".join(body))
    print(r"\midrule")
    print(f"\\textbf{{Aggregate}} (excl.\\ $\\dagger$) & --- & --- & "
          f"\\textbf{{{agg0[0]}/{agg0[1]}}} & \\textbf{{{agg1[0]}/{agg1[1]}}} \\\\")
    print(f"\\quad success rate & --- & --- & {100*agg0[0]/max(agg0[1],1):.1f}\\% "
          f"[{100*lo0:.1f}, {100*hi0:.1f}] & {100*agg1[0]/max(agg1[1],1):.1f}\\% "
          f"[{100*lo1:.1f}, {100*hi1:.1f}] \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")
    nz0 = sum(1 for t in tasks if t not in FLOOR and m0.get(t,[0,0])[0] > 0)
    tot = len([t for t in tasks if t not in FLOOR])
    print(f"\n% prose: {nz0}/{tot} measurable tasks are non-zero at variation 0; "
          f"harness floor: {', '.join(FLOOR) or 'none'}", file=sys.stderr)

if __name__ == "__main__":
    main()
