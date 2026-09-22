"""Table of the LIBERO held-out validation (scripts/run_libero_val.sh) and the step it selects.

    python scripts/libero_val_table.py

Rows are (arm, step), columns are suites; each cell is the median-over-strong-frames position error
(cm) averaged over that suite's 5 episodes, then rotation error (deg) and gripper accuracy. The
mean column averages the four suites with equal weight.

STEP SELECTION RULE (fixed in advance, so it is not tuned on closed-loop results): one step for
ALL THREE arms -- the step with the lowest position error averaged over arms and suites, among
steps where every arm has all four suites. A tie within 0.5 cm goes to the earlier step. Pass
that step to run_single_source_eval.sh as LIB_STEP. Closed-loop success is never used to choose it.
"""
import glob, json, os, collections
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports", "libero_val")
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
cell = collections.defaultdict(dict)                     # tag -> suite -> summary
for p in glob.glob(f"{R}/*/*/report_video-action_i2va.json"):
    su, tag = p.split(os.sep)[-3], p.split(os.sep)[-2]
    s = json.load(open(p))["summary"]
    cell[tag][su] = (100 * s["pos_err_strong_median_m"], s["rot_err_strong_median_deg"], s["gripper_acc"])

def key(t):
    a, _, st = t.partition("_")
    return (a != "official", a, int(st) if st.isdigit() else 0)

print(f"{'':14s} " + " ".join(f"{s[7:]:>20s}" for s in SUITES) + f" {'mean pos cm':>12s}")
for tag in sorted(cell, key=key):
    row = [cell[tag].get(s) for s in SUITES]
    txt = " ".join(f"{r[0]:6.1f}cm {r[1]:5.0f}° {100*r[2]:4.0f}%" if r else f"{'-':>20s}" for r in row)
    m = np.mean([r[0] for r in row]) if all(row) else float("nan")
    print(f"{tag:14s} {txt} {m:12.1f}")

steps = collections.defaultdict(dict)
for tag, d in cell.items():
    a, _, st = tag.partition("_")
    if a != "official" and st.isdigit() and all(s in d for s in SUITES):
        steps[int(st)][a] = np.mean([d[s][0] for s in SUITES])
full = {st: np.mean(list(v.values())) for st, v in steps.items() if set(v) >= {"arm1", "arm2", "arm3"}}
if full:
    best = min(full.values())
    pick = min(st for st, v in full.items() if v <= best + 0.5)
    print("\nsteps with all 3 arms: " + ", ".join(f"{st}: {v:.1f} cm" for st, v in sorted(full.items())))
    print(f"SELECTED LIB_STEP={pick}")
else:
    print("\nno step has all three arms x four suites yet -- nothing selected")
