#!/usr/bin/env python
"""Oracle test for scripts/score_sim_replay.py, with no model and no GPU rollout.

Takes a real GT-replay dump (simulator depth / mask / lossless RGB / live role LUT) and builds
SYNTHETIC canvases whose "imagined" frames ARE the simulator's frames at the matching step --
i.e. a perfect world model. Every direct row must then score as perfect up to its codec:

    rgb:direct    LPIPS    ~ 0            (identity)
    seg:direct    mIoU     = 1            (scene-role palette is lossless)
    depth:direct  AbsRel   ~ codec error  (a few 1e-3)
    normal:direct cos      ~ 1            (codec quantisation only)

and copy-anchor must be strictly WORSE than the perfect prediction on the dynamic region,
because the anchor frame by definition shows the scene before it moved. A row that fails
either check is mis-indexing frames (the classic off-by-one between executed step and canvas
frame) or scoring the wrong segment -- both of which produce plausible-looking numbers
otherwise, which is why the check exists.

    python tests/test_score_sim_replay_synthetic.py <gt_replay_dump.npz> [--experts]

`--experts` also runs the ceiling and cascade rows (needs a GPU with ~4 GB free). For the
oracle canvas the cascade reads frames identical to the simulator's, so cascade must equal
ceiling on the frames both score -- a direct check that the two rows see the same pixels.
"""
import os, sys, tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from training.percep.depth_codec import encode_depth
from training.percep.normal_codec import encode_normal_from_depth
from training.percep.seg_codec import SCENE_ROLE_PALETTE
from scripts.modality_mode_grid import seg_plan, spans_for
from scripts import score_sim_replay as S

T = 41


def synth(src, anchor, out_dir):
    d = dict(np.load(src))
    n = min(len(d["step"]), T)
    for key in ("cmd", "ach", "step", "sim_depth", "sim_mask", "sim_rgb"):
        d[key] = d[key][:n]
    # Pretend one chunk covered these n executed steps, frame k <-> step k, with the same
    # ramp rule as the real rollout (skip_anchor_frames=4).
    d["k"] = np.arange(n, dtype=np.int32)
    d["replan"] = np.zeros(n, np.int32)
    d["is_ramp"] = d["k"] < 4
    lut, _present = S._lut_and_present(np.load(src))   # whichever key the writer used
    intr = d["intrinsics"]
    plan = seg_plan(S.ANCHOR_TEMPLATE[anchor], 2, "iiii")
    spans, total = spans_for(plan, T)
    canvas = np.zeros((1, total, 512, 512, 3), np.uint8)
    for (m, v, _s, _g, _a), (b, e) in zip(plan, spans):
        if m == "action":
            continue
        for k in range(n):
            if m == "video":
                fr = d["sim_rgb"][k, v]
            elif m == "depth":
                fr = encode_depth(d["sim_depth"][k, v].astype(np.float32))
            elif m == "normal":
                fr = encode_normal_from_depth(d["sim_depth"][k, v].astype(np.float32)[None],
                                              abs(float(intr[v][0, 0])), abs(float(intr[v][1, 1])))[0]
            else:
                fr = SCENE_ROLE_PALETTE[lut[d["sim_mask"][k, v].astype(np.intp)]]
            canvas[0, b + k] = fr
    d["canvas"] = canvas
    tag = f"percep_armX_{anchor}_close_microwave"
    os.makedirs(os.path.join(out_dir, tag), exist_ok=True)
    p = os.path.join(out_dir, tag, "close_microwave_v0_t0.npz")
    np.savez(p, **d)
    return p


def main():
    src = sys.argv[1]
    experts = "--experts" in sys.argv
    # --expect FILE: exact numbers produced on the ORIGIN machine from the same fixture. Same
    # bytes in must give the same numbers out, to 1e-4; that is the cross-machine reproduction
    # check for the scorer and the codecs, independent of any rollout.
    expect_vals = None
    if "--expect" in sys.argv:
        import json
        expect_vals = json.load(open(sys.argv[sys.argv.index("--expect") + 1]))
    got_vals = {}
    tmp = tempfile.mkdtemp(prefix="simreplay_oracle_")
    fails = []
    expect = {  # row -> (predicate on value, description)
        "rgb:direct":    (lambda x: x < 0.01,  "LPIPS < 0.01"),
        "seg:direct":    (lambda x: x > 0.999, "mIoU > 0.999"),
        "depth:direct":  (lambda x: x < 0.02,  "AbsRel < 0.02 (codec)"),
        "normal:direct": (lambda x: x > 0.95,  "cos > 0.95 (codec)"),
    }
    for anchor in ("video", "segmentation", "depth", "normal"):
        p = synth(src, anchor, tmp)
        res = S.score_one(p, anchor, ["direct"])
        res.pop("_skipped", None)
        row = S.DIRECT_ROW[anchor]
        r = res.get(row)
        if r is None:
            fails.append(f"{row}: produced no result"); continue
        ok_v, desc = expect[row]
        lower = S.LOWER_IS_BETTER[r["modality"]]
        floor_worse = (r["dyn_copy_anchor"] > r["dyn_value"]) if lower else (r["dyn_copy_anchor"] < r["dyn_value"])
        print(f"{row:14s} full={r['value']:.5f}  dyn={r['dyn_value']:.5f}  dyn_floor={r['dyn_copy_anchor']:.5f}"
              f"  dyn_frac={r['dyn_frac']:.3f}  n={r['n_frames']}")
        got_vals[row] = {k: r[k] for k in ("value", "copy_anchor", "dyn_value", "dyn_copy_anchor", "n_frames")}
        if expect_vals and row in expect_vals:
            for kk, ev in expect_vals[row].items():
                if ev is not None and abs(got_vals[row][kk] - ev) > 1e-4:
                    fails.append(f"{row}.{kk}: {got_vals[row][kk]} != origin {ev} -> scorer/codec differs from the origin machine")
        if not ok_v(r["value"]):
            fails.append(f"{row}: value {r['value']} fails {desc}")
        if not floor_worse:
            fails.append(f"{row}: copy-anchor ({r['dyn_copy_anchor']}) is not worse than a perfect "
                         f"prediction ({r['dyn_value']}) on the dynamic region -> frames mis-indexed")
    if experts:
        p = synth(src, "video", tmp)
        ceil = S.score_one(p, "video", ["ceiling"], ceiling_stride=1)
        ceil.pop("_skipped", None)
        for row, r in sorted(ceil.items()):
            print(f"{row:34s} full={r['value']}  dyn={r['dyn_value']}")
        winners = S.pick_winners({row: {"modality": r["modality"], "expert": r["expert"],
                                        "value": r["value"], "dyn_value": r["dyn_value"]}
                                  for row, r in ceil.items()})
        print("winners:", winners)
        casc = S.score_one(p, "video", ["cascade"], winners=winners)
        casc.pop("_skipped", None)
        for mod in ("depth", "normal", "seg"):
            if mod not in winners:
                continue
            cr = f"{mod}:cascade:{winners[mod]}"
            ce = (f"normal:ceiling:{winners['normal']}:{winners['_normal_convention']}"
                  if mod == "normal" else f"{mod}:ceiling:{winners[mod]}")
            a, b = casc[cr]["value"], ceil[ce]["value"]
            print(f"{cr:30s} {a}   vs ceiling {b}")
            if abs(a - b) > 0.02 * max(abs(b), 1e-3):
                fails.append(f"{cr}={a} differs from {ce}={b} on an oracle canvas -> the cascade "
                             f"and ceiling rows are not reading the same pixels")
    if "--write-expect" in sys.argv:
        import json
        json.dump(got_vals, open(sys.argv[sys.argv.index("--write-expect") + 1], "w"), indent=1)
    print("\n" + ("PASS" if not fails else "FAIL:\n  " + "\n  ".join(fails)))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
