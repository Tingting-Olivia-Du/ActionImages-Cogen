#!/bin/bash
# Prove the installed code is the shipped code, and that it WORKS on this machine.
#
#   bash verify.sh /path/to/ActionImages-Cogen [--scorer-only] [--dump <gt_replay.npz>]
#
# 1. sha256 of every installed file against the manifest -- byte identity, not "looks right".
# 2. imports + signatures the pipeline depends on, INCLUDING the repo modules it does not ship
#    (scripts/modality_mode_grid.py, training/percep/*). A drifted copy of one of those is the
#    realistic failure: it would not crash, it would mis-slice the canvas.
# 3. the ORACLE test: synthetic canvases built from the simulator's own frames must score as
#    perfect (LPIPS 0, mIoU 1, AbsRel ~ codec, cos ~ 1) with copy-anchor strictly worse. It runs
#    the real scorer end to end, so it catches an off-by-one between executed step and canvas
#    frame that hashes cannot. Needs a GT-replay dump; without --dump it makes one (~6 min,
#    CPU + xvfb + CoppeliaSim, no GPU, no checkpoint).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${1:?usage: verify.sh <repo root> [--scorer-only] [--dump f.npz]}"; shift
MODE=all; DUMP=""
while [ $# -gt 0 ]; do case "$1" in --scorer-only) MODE=scorer;; --dump) DUMP="$2"; shift;; esac; shift; done
FAIL=0
echo "== 1. hashes"
while read -r h rel; do
  case "$rel" in eval/rollout*.py) [ "$MODE" = scorer ] && continue ;; esac
  got="$(sha256sum "$DST/$rel" 2>/dev/null | cut -d' ' -f1)"
  if [ "$got" = "$h" ]; then echo "  OK   $rel"; else echo "  FAIL $rel"; FAIL=1; fi
done < "$HERE/MANIFEST.sha256"
cd "$DST"
echo "== 2. imports"
python - "$MODE" <<'PY' || FAIL=1
import sys, inspect
sys.path.insert(0, ".")
from scripts.modality_mode_grid import seg_plan, spans_for
p = seg_plan("depth+action", 2, "iiii")
assert len(p[0]) == 5, f"seg_plan returns {len(p[0])}-tuples; this scorer needs the 5-tuple (modality, view, single_frame, fully_given, absent) version"
assert [m for m, *_ in p] == ["depth", "action", "depth", "action"], p
import scripts.score_sim_replay as S
for f in ("score_one", "pick_winners", "prep_seg", "_lut_and_present"):
    assert hasattr(S, f), f
if sys.argv[1] != "scorer":
    import eval.rollout_env as E
    assert "extra_renders" in inspect.signature(E.RolloutEnv.__init__).parameters
    src = open("eval/rollout.py").read()
    for key in ("--dump-percep", "sim_rgb", "seg_lut", "seg_present"):
        assert key in src, f"eval/rollout.py lacks {key}"
print("  OK   imports / signatures")
PY
echo "== 3. cross-machine reproduction (shipped fixtures, no rollout, no checkpoint)"
# (a) ORACLE: synthetic perfect-world-model canvases built from a GT-replay dump made on the
#     origin machine. Must PASS the perfection checks AND match the origin's numbers to 1e-4.
python tests/test_score_sim_replay_synthetic.py "$HERE/fixtures/oracle_gt_close_microwave.npz" \
  --expect "$HERE/fixtures/oracle_expected.json" 2>&1 | grep -E "direct|PASS|FAIL|!=" || FAIL=1
python tests/test_score_sim_replay_synthetic.py "$HERE/fixtures/oracle_gt_close_microwave.npz" \
  --expect "$HERE/fixtures/oracle_expected.json" > /dev/null 2>&1 || FAIL=1
# (b) REAL MODEL: one arm7 depth+action trial. The same bytes must score identically here.
python - "$HERE/fixtures" <<'PY2' || FAIL=1
import json, sys
sys.path.insert(0, ".")
from scripts.score_sim_replay import score_one
fx = sys.argv[1]
exp = json.load(open(f"{fx}/model_expected.json"))["depth:direct"]
got = score_one(f"{fx}/model/percep_arm7_depth_close_microwave/close_microwave_v0_t1.npz",
                "depth", ["direct"])["depth:direct"]
bad = [k for k, v in exp.items() if abs(got[k] - v) > 1e-4]
assert not bad, f"real-model fixture differs from origin on {bad}: got {[got[k] for k in bad]} want {[exp[k] for k in bad]}"
print(f"  OK   real-model fixture reproduces origin: dyn {got['dyn_value']}/{got['dyn_copy_anchor']}")
PY2
# (c) THIS machine's rollout: only in full mode, since --scorer-only keeps the local rollout.py.
if [ "$MODE" != scorer ]; then
  if [ -z "$DUMP" ]; then
    OUTD="$(mktemp -d)"
    echo "  making a GT-replay dump with the INSTALLED rollout.py in $OUTD (~6 min, CPU) ..."
    xvfb-run -a python eval/rollout.py --gt-replay --tasks close_microwave --variation 0 \
      --num-trials 1 --tag gt --res 512 --frame-interval 3 --arm-action-mode planning \
      --max-steps-factor 1.5 --dump-percep --out "$OUTD" > "$OUTD/log" 2>&1 \
      || { echo "  FAIL gt-replay (see $OUTD/log)"; FAIL=1; }
    DUMP="$OUTD/percep_dump/gt/close_microwave_v0_t0.npz"
  fi
  python - "$DUMP" <<'PY3' || FAIL=1
import sys, numpy as np
d = np.load(sys.argv[1])
miss = {"sim_depth", "sim_mask", "sim_rgb", "cmd", "ach", "seg_lut", "seg_present"} - set(d.files)
assert not miss, f"dump lacks {sorted(miss)} -- the installed rollout.py is not writing them"
g = np.median(np.linalg.norm(d["cmd"][:, :3] - d["ach"][:, :3], axis=-1))
assert g < 1e-3, f"controller floor {g:.4f} m >= 1 mm: the simulator is not executing what it is told"
print(f"  OK   local rollout dump keys; controller floor {g*1000:.2f} mm")
PY3
  python tests/test_score_sim_replay_synthetic.py "$DUMP" 2>&1 | grep -E "PASS|FAIL" || FAIL=1
fi
echo; [ "$FAIL" = 0 ] && echo "VERIFIED" || { echo "NOT VERIFIED -- do not start scoring"; exit 1; }
