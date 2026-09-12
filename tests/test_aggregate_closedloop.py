"""The paired statistics must be right, because they are what the conclusion rests on.

A closed-loop campaign produces a handful of discordant pairs, which is the regime where the
usual chi-square McNemar approximation is invalid and where "no significant difference" is
routinely confused with "not enough data to tell". These tests pin:

  * the exact binomial McNemar against hand-computable cases;
  * that the pairing is refused when the two arms did not see the same scenes;
  * that an underpowered grid is REPORTED as underpowered rather than as a null result;
  * that a grid with zero discordant pairs is reported as having no discriminating power.

Pure CPU. Run: python tests/test_aggregate_closedloop.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.aggregate_closedloop import bootstrap_ci, compare, mcnemar_exact, summarize
from eval.rollout import trial_seed

# ------------------------------------------------- scene seeds must survive a new interpreter
# The two arms of a paired comparison are scored in SEPARATE runs, so a seed that is only
# stable within one process silently gives each arm its own scenes. The builtin hash() does
# exactly that: Python salts string hashing per process, so `abs(hash(("push_buttons",0,0)))`
# returned 628764 / 991250 / 341356 on three consecutive interpreters. An in-process
# determinism check cannot see this -- it has to cross a process boundary.
import subprocess  # noqa: E402

_prog = (
    "import sys; sys.path.insert(0, %r);"
    "from eval.rollout import trial_seed;"
    "print([trial_seed('push_buttons', 0, t) for t in range(4)])"
    % os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_runs = [subprocess.run([sys.executable, "-c", _prog], capture_output=True, text=True,
                        env={**os.environ, "PYTHONHASHSEED": str(salt)}).stdout.strip()
         for salt in ("0", "1", "12345")]
assert _runs[0] and all(r == _runs[0] for r in _runs), (
    f"trial_seed changes across processes / PYTHONHASHSEED values: {_runs}. Each arm would be "
    f"scored on different scenes and the paired McNemar test would be invalid.")
assert str(trial_seed("push_buttons", 0, 0)) in _runs[0], (_runs[0], trial_seed("push_buttons", 0, 0))
assert len({trial_seed("t", 0, i) for i in range(200)}) == 200, "trial_seed collides over trials"
print(f"TRIAL_SEED_STABLE_ACROSS_PROCESSES_OK  {_runs[0]} under 3 PYTHONHASHSEED values")

# ------------------------------------------------------------------ exact McNemar
assert mcnemar_exact(0, 0) == 1.0, "no discordant pairs -> nothing to test"
assert mcnemar_exact(5, 5) == 1.0, "perfectly balanced -> p = 1"
# All discordant pairs favour one arm: two-sided p = 2 * 0.5^n
for n in (1, 2, 3, 4, 5, 6, 8, 10):
    expect = min(1.0, 2 * 0.5 ** n)
    got = mcnemar_exact(n, 0)
    assert abs(got - expect) < 1e-12, (n, got, expect)
assert mcnemar_exact(4, 0) == mcnemar_exact(0, 4), "must be symmetric in b/c"
print("MCNEMAR_EXACT_MATCHES_BINOMIAL_OK  p(n,0) = 2*0.5^n for n=1..10")

# The smallest attainable p sets the power floor: with 5 discordant pairs even a perfect
# split cannot reach 0.05, which is exactly why the underpowered branch exists.
assert mcnemar_exact(5, 0) > 0.05, mcnemar_exact(5, 0)
assert mcnemar_exact(6, 0) < 0.05, mcnemar_exact(6, 0)
print(f"POWER_FLOOR_OK  5 discordant pairs -> min p={mcnemar_exact(5,0):.4f} (>0.05); "
      f"6 -> {mcnemar_exact(6,0):.4f}")

# ------------------------------------------------------------------ bootstrap CI
paired_same = [(True, True)] * 10 + [(False, False)] * 10
lo, hi = bootstrap_ci(paired_same, n_boot=2000, seed=0)
assert lo == hi == 0.0, (lo, hi)
paired_a_better = [(True, False)] * 20
lo, hi = bootstrap_ci(paired_a_better, n_boot=2000, seed=0)
assert lo == hi == 100.0, (lo, hi)
print("BOOTSTRAP_CI_DEGENERATE_CASES_OK  identical->[0,0], A-always->[100,100]")

# The CI and the p-value must tell the same story. A 6-vs-2 split over 20 scenes is a 20 pp
# gap that is NOT significant (p=0.289), so its CI must contain zero -- if the CI excluded
# zero while McNemar said "not significant", one of the two would be lying.
for b_n, c_n, both in [(6, 2, 12), (14, 0, 6), (1, 1, 18)]:
    paired = [(True, False)] * b_n + [(False, True)] * c_n + [(True, True)] * both
    lo, hi = bootstrap_ci(paired, n_boot=20000, seed=0)
    p = mcnemar_exact(b_n, c_n)
    observed = (b_n - c_n) / len(paired) * 100
    assert lo <= observed <= hi, (b_n, c_n, lo, observed, hi)
    excludes_zero = lo > 0 or hi < 0
    assert excludes_zero == (p < 0.05), (
        f"b={b_n} c={c_n}: McNemar p={p:.4f} but CI [{lo:+.1f},{hi:+.1f}] "
        f"{'excludes' if excludes_zero else 'includes'} zero -- they disagree")
    print(f"  b={b_n:2d} c={c_n:2d}: diff {observed:+5.1f} pp  p={p:.4f}  "
          f"CI [{lo:+.1f},{hi:+.1f}]  {'significant' if p < 0.05 else 'not significant'}")
print("BOOTSTRAP_CI_AGREES_WITH_MCNEMAR_OK")


# ------------------------------------------------------------------ end to end
def write(path, tag, outcomes, seed_offset=0):
    """outcomes: {task: [bool, ...]} -> a rollout JSON in the shape eval/rollout.py writes."""
    results = []
    for task, succ in outcomes.items():
        for trial, s in enumerate(succ):
            results.append({
                "task": task, "variation": 0, "trial": trial,
                "scene_seed": 1000 + trial + seed_offset,
                "success": bool(s), "ik_fail_rate": 0.0, "stop_reason": "success" if s else "timeout",
                "replans": 4, "steps": 120, "policy_debug": [{"step": 0, "r_peak_mean": 210.0}],
            })
    with open(path, "w") as f:
        json.dump({"args": {"tag": tag}, "results": results}, f)


tmp = tempfile.mkdtemp()
a_path = os.path.join(tmp, "arm0.json")
b_path = os.path.join(tmp, "arm1.json")

# summarize() basics
write(a_path, "arm0", {"push_buttons": [True] * 5 + [False] * 5})
_args, res = json.load(open(a_path))["args"], json.load(open(a_path))["results"]
s = summarize(res, "arm0")
assert s["success_rate"] == 50.0, s
assert s["per_task"][0]["r_peak_mean"] == 210.0, s
print("SUMMARIZE_OK  50% success, r_peak surfaced")

# A campaign whose requested cells all errored must stay valid JSON and report n/a, not NaN.
empty = summarize([], "all_errors")
assert empty == {"label": "all_errors", "n": 0, "success_rate": None, "per_task": []}, empty
print("EMPTY_SUMMARY_IS_NA_OK")

# a clearly different pair of arms
write(a_path, "arm0", {"push_buttons": [True] * 10, "open_drawer": [True] * 10})
write(b_path, "arm1", {"push_buttons": [False] * 10, "open_drawer": [False] * 10})
blob = compare(a_path, b_path)
assert blob["pooled"]["b"] == 20 and blob["pooled"]["c"] == 0, blob["pooled"]
assert blob["pooled"]["p_value"] < 1e-5, blob["pooled"]
assert "DIFFERENT" in blob["verdict"], blob["verdict"]
print("COMPARE_DETECTS_REAL_DIFFERENCE_OK")

# identical arms -> no discriminating power, and it must SAY that
write(a_path, "arm0", {"push_buttons": [True, False] * 5})
write(b_path, "arm1", {"push_buttons": [True, False] * 5})
blob = compare(a_path, b_path)
assert blob["pooled"]["b"] == blob["pooled"]["c"] == 0, blob["pooled"]
assert "NO DISCRIMINATION" in blob["verdict"], blob["verdict"]
print("ZERO_DISCORDANT_REPORTED_AS_NO_POWER_OK")

# a 3-vs-0 split is a real-looking gap that CANNOT reach significance -- must say underpowered,
# not "no difference"
write(a_path, "arm0", {"push_buttons": [True] * 3 + [True] * 7})
write(b_path, "arm1", {"push_buttons": [False] * 3 + [True] * 7})
blob = compare(a_path, b_path)
assert blob["pooled"]["b"] == 3 and blob["pooled"]["c"] == 0, blob["pooled"]
assert "UNDERPOWERED" in blob["verdict"], blob["verdict"]
print("UNDERPOWERED_GRID_REPORTED_AS_SUCH_OK  (3 discordant pairs, not called a null result)")

# mismatched scenes must be refused, not silently paired
write(a_path, "arm0", {"push_buttons": [True] * 5})
write(b_path, "arm1", {"push_buttons": [False] * 5}, seed_offset=777)
try:
    compare(a_path, b_path)
except SystemExit as exc:
    assert "different scene_seed" in str(exc), exc
    print("MISMATCHED_SCENES_REFUSED_OK")
else:
    raise AssertionError("arms scored on different scenes were paired anyway -- the McNemar "
                         "test would be comparing different worlds")

# no overlapping cells at all
write(a_path, "arm0", {"push_buttons": [True] * 5})
write(b_path, "arm1", {"open_drawer": [True] * 5})
try:
    compare(a_path, b_path)
except SystemExit as exc:
    assert "nothing to pair" in str(exc), exc
    print("NO_SHARED_CELLS_REFUSED_OK")
else:
    raise AssertionError("disjoint grids were compared anyway")

print("ALL_AGGREGATE_CLOSEDLOOP_TESTS_PASSED")
