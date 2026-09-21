#!/usr/bin/env python
"""Score a sim-replay perception dump: the model's imagined perception against the SIMULATOR's
own render of the future the model's OWN actions actually produced.

THE POINT. Every other perception number in this repo scores a generated frame t against the
DEMONSTRATION's frame t, so it charges the model twice: once for bad geometry and once for not
reproducing a trajectory it was never required to reproduce. Under IIII the second term
dominates and is not separable. Here the target is instead

    D_sim(s0, a_hat_{1:t})   -- render the simulator AFTER executing the model's own actions

which is the counterfactual ground truth for what the model committed to. Trajectory divergence
does not enter: the simulator went wherever the model sent it.

WHAT IT DOES AND DOES NOT MEASURE. This is accuracy -- real metric geometry, real occlusion,
real object poses, arbitrated by a referee the model cannot influence except through its
actions -- but it is accuracy CONDITIONAL ON THE MODEL'S OWN TRAJECTORY. A model that plans to
do nothing, and correctly renders a static scene, scores well. So it is meaningless without the
two guards this script always reports alongside:

  copy_anchor   the same metric with the anchor frame copied to every step. A frozen
                imagination cannot beat it; if the model does not beat it, nothing else here
                means anything.
  exec_gap_m    commanded vs ACHIEVED end-effector position. The imagined depth belongs to the
                pose the model DREW, the render belongs to the pose the controller REACHED.
                That gap is the protocol's own noise floor. Measured 0.3 mm median under
                `planning` on a GT replay, i.e. ~3 orders below the depth differences at stake,
                but it is a property of the run and has to be re-read per campaign.

FRAMES THAT ARE DROPPED, and why each is not a choice:
  is_ramp    rollout.py's `skip_anchor_frames` OVERWRITES the first frames of every chunk with
             `ramp_from_current`. Those executed poses are not what the model drew, so charging
             the resulting render to the perception stream would measure the ramp.
  k == 0     the anchor itself, which is ground truth handed over.
"""
import argparse, glob, json, os, sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from training.percep.depth_codec import MAX_VALID, MIN_VALID, decode_depth
from training.percep.normal_codec import decode_normal, depth_to_normal
from training.percep.seg_codec import decode_scene_roles
from scripts.modality_mode_grid import seg_plan, spans_for

ANCHOR_TEMPLATE = {"video": "video+action", "depth": "depth+action",
                   "segmentation": "segmentation+action", "normal": "normal+action"}

# --- the cascade, scored against the SAME counterfactual ---------------------------------
# An RGB-anchored dump carries no perception segment: the model imagined future RGB, not future
# depth. The matched route is the CASCADE -- read depth off those generated frames with a
# feed-forward specialist -- and it has to face the identical target, D_sim(s0, a_hat), because
# both routes drove the same simulator with the same decoded actions. Scoring the cascade
# against the demo while scoring co-generation against the counterfactual would hand one of them
# a different ground truth and make the comparison meaningless.
#
# A METRIC specialist, so neither side needs a fitted scale: our codec emits metres and so does
# this, and the two routes then compete on identical terms with zero free parameters. That
# removes the alignment argument rather than settling it, which matters because a free scale
# fitted against the ground truth is exactly the asymmetry the paper leans on.
#
# Depth Anything V2 Metric Indoor, not UniDepth V2 -- purely an availability call. UniDepth is
# the probe suite's metric counterpart, but it ships its own package and `unidepth` is not
# installed in this container (only its HF weights are cached), whereas DA V2 Metric is natively
# a transformers depth-estimation pipeline. Both are `METRIC` in scripts/probe_depth_baselines.py.
#
# STANDING CAVEAT, the same one that probe carries: this model is ZERO-SHOT on RLBench renders
# while ours is fine-tuned on exactly this distribution. A cascade row built on it bounds how
# hard the task is; it does not rank the two designs on its own.
# CANDIDATES. Several are run and the STRONGEST becomes the counterpart, because picking a
# weak one would flatter us -- the same discipline scripts/probe_normal_specialists.py already
# applies to Lotus vs Marigold. The selection criterion is fixed in advance and is measured on
# the EXPERT-CEILING row (expert on REAL frames), never on the cascade row: selecting on the
# number you are about to report is selecting on your own result. See PERCEP_SIMREPLAY.md §6.
#
# Only METRIC models are listed. A scale-ambiguous model needs a fitted scale, and until the
# anchor-frame affine fit is implemented it would be scored on a handicap it never signed up
# for. `metric` in probe_depth_baselines.py is {depthpro, da2metric, unidepth, metric3d}.
DEPTH_EXPERTS = {
    "da2metric": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    "depthpro":  "apple/DepthPro-hf",
    "unidepth":  "lpiccinelli/unidepth-v2-vitl14",   # needs the `unidepth` package
}
SPECIALIST_NAME = os.environ.get("SPECIALIST", "da2metric")
_SPECIALIST = {}


def _specialist(dev=None):
    """Metric depth expert -> callable(rgb_u8[, K]) -> depth in metres, 0 free parameters."""
    key = SPECIALIST_NAME
    if key not in _SPECIALIST:
        dev = int(os.environ.get("SPECIALIST_GPU", "0")) if dev is None else dev
        import torch
        if key not in DEPTH_EXPERTS:
            raise ValueError(f"unknown specialist {key!r}; have {sorted(DEPTH_EXPERTS)}")
        if key == "unidepth":
            # Ships its own package rather than a transformers config. Absent here as of
            # 2026-09-20 (weights cached, module missing), which is the only reason da2metric
            # is the default -- it is an availability call, not a quality one.
            from unidepth.models import UniDepthV2
            m = UniDepthV2.from_pretrained(DEPTH_EXPERTS[key]).to(f"cuda:{dev}").eval()
            def infer(rgb_u8, K=None):
                x = torch.from_numpy(np.ascontiguousarray(rgb_u8)).permute(2, 0, 1).to(f"cuda:{dev}")
                Kt = None if K is None else torch.from_numpy(np.asarray(K, np.float32)).to(f"cuda:{dev}")
                with torch.no_grad():
                    return m.infer(x, Kt)["depth"].squeeze().float().cpu().numpy()
        else:
            from PIL import Image
            from transformers import pipeline
            pipe = pipeline("depth-estimation", model=DEPTH_EXPERTS[key], device=dev)
            def infer(rgb_u8, K=None):
                d = pipe(Image.fromarray(np.ascontiguousarray(rgb_u8)))["predicted_depth"]
                d = d.float().cpu().numpy() if torch.is_tensor(d) else np.asarray(d, np.float32)
                return np.squeeze(d)
        _SPECIALIST[key] = infer
    return _SPECIALIST[key]


def absrel(pred, gt):
    """Relative error over pixels BOTH sides could express. `coverage` is returned with it
    because the depth codec is a colour path: a frame where the model wandered off that path
    decodes to NaN and would otherwise be scored on a shrinking, unrepresentative subset."""
    ok = np.isfinite(gt) & (gt > MIN_VALID) & (gt < MAX_VALID) & np.isfinite(pred)
    gt_ok = np.isfinite(gt) & (gt > MIN_VALID) & (gt < MAX_VALID)
    if not ok.any():
        return None, 0.0
    rel = np.abs(np.clip(pred, MIN_VALID, MAX_VALID) - gt) / np.maximum(gt, MIN_VALID)
    return float(rel[ok].mean()), float(ok.sum() / max(gt_ok.sum(), 1))


def score_one(npz_path, anchor, num_frames=41, res=512):
    d = np.load(npz_path)
    if d["canvas"].size == 0:
        return None
    canvas = d["canvas"]                      # [n_replan, 4*T, H, W, 3] uint8
    plan, _ = seg_plan(ANCHOR_TEMPLATE[anchor], 2, "iiii"), None
    spans, total = spans_for(plan, num_frames)
    if canvas.shape[1] != total:
        raise RuntimeError(f"{npz_path}: canvas has {canvas.shape[1]} frames, plan wants {total}")
    # Which spans hold the ANCHOR modality (the perception stream), per view.
    percep_span = {v: sp for (m, v, _sf, _g, _a), sp in zip(plan, spans) if m == anchor}
    if len(percep_span) != 2:
        raise RuntimeError(f"{npz_path}: expected 2 {anchor!r} segments, got {sorted(percep_span)}")

    sim_depth = d["sim_depth"].astype(np.float32) if "sim_depth" in d else None
    sim_mask = d["sim_mask"] if "sim_mask" in d else None
    keep = (~d["is_ramp"]) & (d["k"] > 0)
    exec_gap = np.linalg.norm(d["cmd"][:, :3] - d["ach"][:, :3], axis=-1)
    # Per-replan SIMULATOR-side reference frame, for the dynamic region. The first recorded step
    # of a chunk is the closest thing the dump holds to the observation the model was anchored
    # on, and the dynamic region is defined against it: pixels the episode actually changes.
    # WHY THIS IS NOT OPTIONAL. The camera is fixed within an episode, so a full-frame average
    # is dominated by background that never moves -- and copy-anchor is exactly right there, by
    # construction. Measured on the first trial scored (arm7 depth, toilet_seat_down): full
    # frame gave AbsRel 0.0630 against a copy-anchor floor of 0.0644, i.e. the model sat ON the
    # floor. That is a statement about how little of the frame moves, not about the model.
    ref_row = {}
    for i in np.flatnonzero(d["k"] >= 0):
        rep = int(d["replan"][i])
        if rep not in ref_row:
            ref_row[rep] = i
    rows, floor_rows, cov_rows = [], [], []
    dyn_rows, dyn_floor_rows, dyn_frac = [], [], []
    # The anchor frame is identical for every k of a replan, so its specialist pass is computed
    # once per (replan, view) rather than once per scored frame -- the cascade floor otherwise
    # doubles the eval's cost for a constant.
    anch_cache = {}
    per_k = defaultdict(list)
    gaps = []
    for i in np.flatnonzero(keep):
        rep, k = int(d["replan"][i]), int(d["k"][i])
        if rep >= len(canvas):
            continue
        gaps.append(float(exec_gap[i]))
        for v in (0, 1):
            b, e = percep_span[v]
            if k >= e - b:
                continue
            pred_rgb = canvas[rep, b + k]
            anchor_rgb = canvas[rep, b]          # frame 0 = the given anchor -> copy-anchor floor
            if anchor == "video":
                # Cascade: the specialist reads the frames the model IMAGINED. The floor is the
                # same specialist on the model's ANCHOR frame -- "assume the RGB does not change,
                # then perceive it" -- which is the cascade's own copy-anchor, not ours.
                gt = sim_depth[i, v]
                pred_d = _specialist()(pred_rgb)
                if (rep, v) not in anch_cache:
                    anch_cache[(rep, v)] = _specialist()(anchor_rgb)
                anch_d = anch_cache[(rep, v)]
                val, cov = absrel(pred_d, gt)
                fl, _ = absrel(anch_d, gt)
            elif anchor == "depth":
                gt = sim_depth[i, v]
                val, cov = absrel(decode_depth(pred_rgb[None])[0], gt)
                fl, _ = absrel(decode_depth(anchor_rgb[None])[0], gt)
            elif anchor == "normal":
                # Sim normals are the analytic operator on the sim's OWN depth, the same
                # operator training encoded with, so the target and the prediction live in one
                # convention by construction rather than by a fitted sign.
                fx, fy = float(d["intrinsics"][v][0, 0]), float(d["intrinsics"][v][1, 1])
                gt_n = depth_to_normal(sim_depth[i, v], abs(fx), abs(fy))
                val = float((gt_n * decode_normal(pred_rgb[None])[0]).sum(-1).mean())
                fl = float((gt_n * decode_normal(anchor_rgb[None])[0]).sum(-1).mean())
                cov = 1.0
            else:
                continue
            if val is None:
                continue
            rows.append(val); floor_rows.append(fl); cov_rows.append(cov)
            per_k[k].append(val)
            # Dynamic region: pixels whose COUNTERFACTUAL ground truth moved at least 1 cm
            # relative to this chunk's reference render. 1 cm is the codec's own resolution
            # scale, not a tuned threshold, and `dyn_frac` is reported so the reader can see
            # how much of the frame each number actually covers.
            ri = ref_row.get(rep)
            if ri is not None and sim_depth is not None:
                gt_t, gt_0 = sim_depth[i, v], sim_depth[ri, v]
                moved = np.isfinite(gt_t) & np.isfinite(gt_0) & (np.abs(gt_t - gt_0) > 0.01)
                if moved.sum() > 100:
                    if anchor == "video":
                        dv, _ = absrel(pred_d[moved], gt_t[moved])
                        df, _ = absrel(anch_d[moved], gt_t[moved])
                    elif anchor == "depth":
                        dv, _ = absrel(decode_depth(pred_rgb[None])[0][moved], gt_t[moved])
                        df, _ = absrel(decode_depth(anchor_rgb[None])[0][moved], gt_t[moved])
                    else:
                        dv = float((gt_n[moved] * decode_normal(pred_rgb[None])[0][moved]).sum(-1).mean())
                        df = float((gt_n[moved] * decode_normal(anchor_rgb[None])[0][moved]).sum(-1).mean())
                    if dv is not None and df is not None:
                        dyn_rows.append(dv); dyn_floor_rows.append(df)
                        dyn_frac.append(float(moved.mean()))
    if not rows:
        return None
    metric = "absrel" if anchor in ("depth", "video") else "mean_cos"
    # Horizon bins. The exec gap GROWS with k (measured 0.5 mm for k<10, 118 mm median for
    # k>=20 on the first trial), so a pooled number mixes a regime where the simulator faithfully
    # executed what the model drew with one where it did not. Binning is how that stays visible.
    bins = {}
    for lo, hi in ((1, 10), (10, 20), (20, 41)):
        vs = [x for k, xs in per_k.items() if lo <= k < hi for x in xs]
        gs = [float(exec_gap[i]) for i in np.flatnonzero(keep)
              if lo <= int(d["k"][i]) < hi]
        if vs:
            bins[f"k{lo}_{hi}"] = {"value": round(float(np.mean(vs)), 5), "n": len(vs),
                                   "exec_gap_m_median": round(float(np.median(gs)), 5) if gs else None}
    return {"anchor": anchor, "metric": metric, "n_frames": len(rows),
            "value": round(float(np.mean(rows)), 5),
            "copy_anchor": round(float(np.mean(floor_rows)), 5),
            "dyn_value": round(float(np.mean(dyn_rows)), 5) if dyn_rows else None,
            "dyn_copy_anchor": round(float(np.mean(dyn_floor_rows)), 5) if dyn_floor_rows else None,
            "dyn_frac": round(float(np.mean(dyn_frac)), 5) if dyn_frac else None,
            "n_dyn_frames": len(dyn_rows),
            "by_horizon": bins,
            "decode_coverage": round(float(np.mean(cov_rows)), 5),
            "exec_gap_m_median": round(float(np.median(gaps)), 5) if gaps else None,
            "exec_gap_m_p90": round(float(np.percentile(gaps, 90)), 5) if gaps else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_root", default=os.path.join(REPO, "reports", "percep_simreplay",
                                                        "percep_dump"))
    ap.add_argument("--out", default=os.path.join(REPO, "reports", "percep_simreplay",
                                                  "scores.json"))
    ap.add_argument("--specialist", default=None, choices=sorted(DEPTH_EXPERTS),
                    help="which metric depth expert the cascade rows use. Run ALL of them on "
                         "the expert-ceiling row first and take the strongest; see "
                         "PERCEP_SIMREPLAY.md 6. The chosen name is recorded in the output.")
    ap.add_argument("--gpu", type=int, default=None,
                    help="GPU for the cascade specialist. The rollout campaign occupies 0-3 "
                         "with ~19 GB headroom each, so a 1.3 GB depth model coexists, but "
                         "pinning it is safer than racing an allocator.")
    a = ap.parse_args()
    if a.gpu is not None:
        os.environ["SPECIALIST_GPU"] = str(a.gpu)
    if a.specialist is not None:
        global SPECIALIST_NAME
        SPECIALIST_NAME = a.specialist
    out = defaultdict(list)
    for tagdir in sorted(glob.glob(os.path.join(a.dump_root, "*"))):
        tag = os.path.basename(tagdir)
        anchor = tag.split("_")[2] if len(tag.split("_")) > 2 else "video"
        for f in sorted(glob.glob(os.path.join(tagdir, "*.npz"))):
            try:
                r = score_one(f, anchor)
            except Exception as exc:
                print(f"  {os.path.basename(f)}: ERROR {type(exc).__name__}: {exc}", flush=True)
                continue
            if r is None:
                continue
            r["trial_file"] = os.path.basename(f)
            out[tag].append(r)
            print(f"  {tag}/{r['trial_file']:42s} {r.get('metric')}={r.get('value')} "
                  f"floor={r.get('copy_anchor')} n={r.get('n_frames')}", flush=True)
    agg = {}
    for tag, rs in out.items():
        vals = [r["value"] for r in rs if r.get("value") is not None]
        fls = [r["copy_anchor"] for r in rs if r.get("copy_anchor") is not None]
        if not vals:
            continue
        dvs = [r["dyn_value"] for r in rs if r.get("dyn_value") is not None]
        dfs = [r["dyn_copy_anchor"] for r in rs if r.get("dyn_copy_anchor") is not None]
        dfr = [r["dyn_frac"] for r in rs if r.get("dyn_frac") is not None]
        gap = [r["exec_gap_m_median"] for r in rs if r.get("exec_gap_m_median") is not None]
        agg[tag] = {"n_trials": len(vals), "metric": rs[0]["metric"],
                    "value": round(float(np.mean(vals)), 5),
                    "copy_anchor": round(float(np.mean(fls)), 5),
                    "dyn_value": round(float(np.mean(dvs)), 5) if dvs else None,
                    "dyn_copy_anchor": round(float(np.mean(dfs)), 5) if dfs else None,
                    "dyn_frac": round(float(np.mean(dfr)), 5) if dfr else None,
                    "exec_gap_m_median": round(float(np.mean(gap)), 5) if gap else None,
                    "n_frames": int(sum(r["n_frames"] for r in rs))}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"specialist": SPECIALIST_NAME, "specialist_repo": DEPTH_EXPERTS[SPECIALIST_NAME],
               "per_trial": out, "aggregate": agg}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    for tag, v in sorted(agg.items()):
        dv = "   --  " if v["dyn_value"] is None else f"{v['dyn_value']:7.4f}"
        df = "   --  " if v["dyn_copy_anchor"] is None else f"{v['dyn_copy_anchor']:7.4f}"
        fr = "  -- " if v["dyn_frac"] is None else f"{100*v['dyn_frac']:4.1f}%"
        print(f"{tag:42s} {v['metric']:8s} full {v['value']:7.4f}/{v['copy_anchor']:7.4f}"
              f"   dyn {dv}/{df} ({fr})   gap {v['exec_gap_m_median']}   n={v['n_trials']}")


if __name__ == "__main__":
    main()
