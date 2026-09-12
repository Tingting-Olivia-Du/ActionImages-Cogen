"""Batch quantitative eval of one checkpoint's four output streams over many held-out episodes.

    python scripts/heldout_batch_eval.py --ckpt .../step10000.ckpt --tag arm6_10000_heldout --gpu 0

WHY THIS EXISTS. Every quantitative number in the current draft comes from n=1 (tab:versatility)
or n=4 (tab:crosstask) held-IN episodes. EVAL_PLAN.md designed a proper batch protocol back on
2026-08-11 -- 16 episodes, 8 seen (variation0, trained on) + 8 unseen (variation1, held out),
paired for significance -- but never got written: at the time the inference pipeline could not
assemble arbitrary template/stream combinations (EVAL_PLAN.md's "B2" blocker). That path has
since been built and debugged by scripts/modality_mode_grid.py and scripts/tag_swap_ablation.py
(`pipe(..., template=, streams=, conditioning_mode=)`), so this script is pure orchestration: it
reuses modality_mode_grid's GPU-claiming, dataset, pipeline, span-slicing and scoring functions
verbatim and loops them over EVAL_PLAN.md's exact 16-episode list (verified to exist in the
current data/rlbench_selfgen_512_aug tree -- the tree EVAL_PLAN.md itself targeted a different,
now-deleted 256^2 tree) instead of one.

MODE. Defaults to `iiii`, the deployment regime (rollout uses it, and 90% of perception training
draws land there under the shipped M2 mixture). `--mode fifi` gives the RGB in full so only the
auxiliary stream is predicted; that is the discriminative setting and the only one comparable
with an external depth or normal estimator, which is handed a real frame rather than asked to
invent one. This is not the full mode grid modality_mode_grid.py sweeps for one episode; it is
one mode swept over many episodes, which is the axis tab:versatility and tab:crosstask missed.

SEGMENTATION DEGENERATE-GT GUARD. EVAL_PLAN.md worried that some episodes' GT mask is essentially
empty (e.g. an occluded jar lid under the old referring-mask protocol) and would report a
meaningless IoU=0. Under scene_roles this is handled already, one level down: scripts.
modality_mode_grid.score() calls scene_role_iou() which returns gt_px per role and drops any role
with gt_px==0 before averaging (seg_codec.py:359-378). If EVERY role in an episode is vacuous,
score() returns value=None, and this script's aggregator skips None rather than counting it as a
zero -- so no separate pre-scan step is needed, just correct handling of the None case.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import scripts.modality_mode_grid as G  # noqa: E402  (reuse, never re-implement)
from inference import build_pipeline  # noqa: E402
from training.dataset import RLBenchSelfgenDataset  # noqa: E402

# EVAL_PLAN.md Sec. 3.3, verbatim task list. Verified present (both variation0 and variation1,
# episode0) in data/rlbench_selfgen_512_aug on 2026-08-29.
TASKS = [
    "close_jar", "insert_onto_square_peg", "light_bulb_in", "meat_off_grill",
    "open_drawer", "push_buttons", "reach_and_drag", "put_item_in_drawer",
]
SEEN_EPISODES = [f"{t}/variation0/episodes/episode0" for t in TASKS]
UNSEEN_EPISODES = [f"{t}/variation1/episodes/episode0" for t in TASKS]

# modality name -> template string. All four are announced in one categorical draw so a single
# RLBenchSelfgenDataset can serve every force_template call (mirrors modality_mode_grid.py:373).
MODALITY_TEMPLATE = {
    "depth": "video+depth",
    "segmentation": "video+segmentation",
    "normal": "video+normal",
    "action": "video+action",
    # arm7's substitution family: the action stream anchored on a NON-RGB observation. Added as
    # NEW keys rather than by changing the four above, because every previous run's per_episode
    # keys ("<ep>|action", ...) must keep meaning exactly what they meant.
    #
    # These read the SAME quantity as "action" -- a decoded 6-DoF trajectory scored against the
    # same GT -- through a different observation space, so they are directly comparable to the
    # "action" row. That comparison is the whole point of arm7.
    "action_from_depth": "depth+action",
    "action_from_seg": "segmentation+action",
    "action_from_normal": "normal+action",
}
# Scored by G.score_action rather than G.score. Derived from the table so a new entry cannot be
# added without landing in the right branch.
ACTION_MODALITIES = frozenset(
    m for m, t in MODALITY_TEMPLATE.items() if t.split("+")[-1] == "action")
MODE_DEFAULT = "iiii"


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--gpu", default="auto")
    p.add_argument("--need_mib", type=int, default=30000)
    p.add_argument("--wait_min", type=int, default=0)
    p.add_argument("--data", default=str(REPO / "data" / "rlbench_selfgen_512_aug"))
    p.add_argument("--out", default=str(REPO / "reports" / "heldout_batch_eval"))
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--num_frames", type=int, default=41)
    p.add_argument("--frame_interval", type=int, default=3)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--segmentation_mode", default="scene_roles", choices=("referring", "scene_roles"))
    p.add_argument("--modalities", default="", help="comma list restricting MODALITY_TEMPLATE keys")
    p.add_argument("--episodes", default="",
                    help="comma list of relative episode paths, overriding the default 16 "
                         "(smoke-test with a single episode before committing to the full run)")
    p.add_argument("--n_boot", type=int, default=10000, help="bootstrap resamples for the 95% CI")
    p.add_argument("--mode", default=MODE_DEFAULT, choices=("iiii", "fiii", "fifi"),
                   help="conditioning plan. `iiii` is the deployment regime and the default. "
                        "`fifi` gives the RGB in full so only the auxiliary stream is predicted "
                        "-- that is the DISCRIMINATIVE setting, and the only one comparable with "
                        "an external depth or normal estimator, which is handed a real frame "
                        "rather than asked to invent one.")
    p.add_argument("--prompt_tag_style", default="explicit", choices=("explicit", "none"),
                   help="'explicit' prefixes <video><depth> etc, which is what our fine-tuned "
                        "arms were trained on. The RELEASED checkpoint never saw modality tags, "
                        "so scoring it with them measures prompt-distribution shift rather than "
                        "capability -- pass 'none' for it, exactly as the closed-loop harness "
                        "does (scripts/run_official_closedloop_postfix.sh).")
    p.add_argument("--split_label", default="",
                   help="force the seen/unseen tier for every episode (e.g. 'unseen_task' for "
                        "the never-trained-task tree, whose episodes are variation0 and would "
                        "otherwise be misfiled as training data)")
    p.add_argument("--no_resume", action="store_true",
                   help="recompute every cell even if this tag already has scored results")
    return p.parse_args()


def _episode_split(ep_paths, forced=""):
    """Tag each requested episode with the generalization tier it belongs to.

    By default the tier is read off the variation: variation0 was trained on ("seen"), anything
    else was not ("unseen"). That inference is WRONG for the never-trained tasks, whose episodes
    are variation0 of a task absent from the training tree -- the strongest held-out tier there
    is, but indistinguishable from training data by path alone. `--split_label unseen_task`
    overrides it for those runs rather than letting them be silently filed as "seen".
    """
    if forced:
        return {ep: forced for ep in ep_paths}
    return {ep: ("seen" if "/variation0/" in ep else "unseen") for ep in ep_paths}


_LPIPS_MODEL = None


def _lpips_model(device):
    """Loaded once, lazily -- not every invocation of this script needs it, and the AlexNet
    trunk (~230MB) is a one-time download on a fresh machine."""
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        _LPIPS_MODEL = lpips.LPIPS(net="alex").to(device).eval()
    return _LPIPS_MODEL


def score_video(pred, gt, device):
    """pred, gt: [T,H,W,3] uint8. No RGB metric (PSNR/SSIM/LPIPS) existed anywhere in this repo
    before this function -- tab:headline and tab:offline_preservation in the paper draft both
    need it (LPIPS/SSIM), and the existing given-RGB PSNR in tab:versatility is the frozen-VAE
    round-trip ceiling, not a measure of predicted-future quality.
    """
    n = min(len(pred), len(gt))
    pred, gt = pred[:n].astype(np.float64), gt[:n].astype(np.float64)
    mse = float(((pred - gt) ** 2).mean())
    psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-9))

    ssim_vals = [ssim(gt[t], pred[t], channel_axis=-1, data_range=255) for t in range(n)]

    with torch.no_grad():
        p = torch.from_numpy(pred / 127.5 - 1.0).permute(0, 3, 1, 2).float().to(device)
        g = torch.from_numpy(gt / 127.5 - 1.0).permute(0, 3, 1, 2).float().to(device)
        lp = _lpips_model(device)(p, g).mean().item()

    return {"metric": "lpips", "value": round(lp, 5),
            "psnr_db": round(float(psnr), 3), "ssim": round(float(np.mean(ssim_vals)), 5)}


def bootstrap_ci(values, n_boot, seed, alpha=0.05):
    """Percentile bootstrap on the mean. `values` already has None entries dropped by the caller."""
    vals = np.asarray(values, dtype=np.float64)
    if len(vals) == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(vals.mean()), float(lo), float(hi)


def main():
    a = _args()
    # SETTING CUDA_VISIBLE_DEVICES HERE IS TOO LATE IF CUDA IS ALREADY INITIALISED.
    # Importing this module pulls in torch and scripts.modality_mode_grid, and by the time we
    # get here `torch.cuda.is_initialized()` is already True -- after which the environment
    # variable is ignored and every run silently lands on physical GPU 0 no matter what --gpu
    # says. Two concurrent evals therefore raced onto the same card: the first reserved 26.9 GB
    # and the second died with "could not reserve 27000 MiB" while its log claimed "using GPU 7"
    # (2026-09-05). Running on the wrong card silently is worse than not starting, so refuse.
    #
    # The fix is to mask the card BEFORE python starts:
    #     CUDA_VISIBLE_DEVICES=7 python -u scripts/heldout_batch_eval.py ... --gpu 0
    # Inside that mask the only visible card is index 0, which is the physical card you chose.
    preset = os.environ.get("CUDA_VISIBLE_DEVICES")
    if torch.cuda.is_initialized() and not preset:
        sys.exit(
            "[heldout] CUDA is already initialised, so setting CUDA_VISIBLE_DEVICES from inside\n"
            "          this process would be ignored and the run would land on physical GPU 0\n"
            "          regardless of --gpu. Mask the card before python starts instead:\n"
            f"            CUDA_VISIBLE_DEVICES=<physical> {' '.join(sys.argv[:1])} ... --gpu 0")
    if preset:
        # Already masked by the caller: index 0 is the only card we can see. Honour it and do
        # not re-point the variable, which would index into the masked view a second time.
        print(f"[heldout] CUDA_VISIBLE_DEVICES={preset} was set by the caller; using it",
              flush=True)
    elif a.gpu == "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = G.pick_gpu(a)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    print(f"[heldout] using GPU {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
    torch.cuda.init(); torch.zeros(1, device="cuda")
    G.reserve_vram(a.need_mib)

    modalities = [m.strip() for m in a.modalities.split(",") if m.strip()] or list(MODALITY_TEMPLATE)
    templates = sorted({MODALITY_TEMPLATE[m] for m in modalities})
    episodes = [e.strip() for e in a.episodes.split(",") if e.strip()] or (SEEN_EPISODES + UNSEEN_EPISODES)
    split_of = _episode_split(episodes, a.split_label)

    ds = RLBenchSelfgenDataset(
        base_path=a.data, num_frames=a.num_frames, frame_interval=a.frame_interval,
        height=a.res, width=a.res,
        template_mix=",".join(f"{t}@{1/len(templates):.4f}" for t in templates),
        prompt_tag_style=a.prompt_tag_style, strict_getitem=True, variations="all",
        segmentation_mode=a.segmentation_mode)

    ep_idx = {}
    missing = []
    for ep in episodes:
        want = os.path.normpath(os.path.join(a.data, ep))
        idx = next((i for i, e in enumerate(ds.episodes) if os.path.normpath(e["path"]) == want), None)
        if idx is None:
            missing.append(ep)
        else:
            ep_idx[ep] = idx
    if missing:
        sys.exit(f"[heldout] episode(s) not found in dataset index: {missing}")

    pipe = build_pipeline(SimpleNamespace(
        model_id="Wan-AI/Wan2.2-TI2V-5B", ckpt_path=a.ckpt, height=a.res, width=a.res,
        use_usp=False, cfg_parallel=False, dynamic_cache_schedule=False, torch_compile=False))
    device, dtype, T = pipe.device, torch.bfloat16, a.num_frames

    out_dir = Path(a.out) / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # RESUME. Load whatever this tag already scored and skip those cells. Two reasons this is
    # worth the twelve lines: a 16-episode x 4-modality sweep is ~3h and an interruption used to
    # cost all of it, and -- the case that motivated it -- when a scoring bug nulls one modality,
    # only that modality needs recomputing rather than the whole sweep. A cell counts as done
    # only if it carries a non-None `value`, so nulled cells are retried automatically.
    per_episode = {}  # "episode|modality" -> record
    resume_path = out_dir / "per_episode.json"
    if not a.no_resume and resume_path.exists():
        try:
            per_episode = json.load(open(resume_path)).get("results", {})
            done = sum(1 for r in per_episode.values()
                       if r.get("modality") != "video" and r.get("value") is not None)
            print(f"[heldout] resuming from {resume_path}: {done} scored cells already present",
                  flush=True)
        except (OSError, ValueError) as exc:
            print(f"[heldout] could not read {resume_path} ({exc}); starting fresh", flush=True)
            per_episode = {}

    for ep in episodes:
        idx = ep_idx[ep]
        for modality in modalities:
            template = MODALITY_TEMPLATE[modality]
            key = f"{ep}|{modality}"
            prev = per_episode.get(key)
            if prev is not None and prev.get("value") is not None:
                print(f"  SKIP {key:52s} already scored "
                      f"{prev.get('metric')}={prev.get('value')}", flush=True)
                continue
            t0 = time.time()
            G._seed_all(a.seed)
            s = ds.getitem(idx, force_template=template)
            if s["template"] != template:
                print(f"  SKIP {key}: dataset degraded to {s['template']!r}", flush=True)
                continue
            s["_view_dir"] = [s["view_dirs"][i] for i in s["view_indices"]][0]
            streams = {k: v.unsqueeze(0).to(device=device, dtype=dtype) for k, v in s["streams"].items()}
            G._seed_all(a.seed)
            frames = np.stack([np.asarray(f) for f in pipe(
                prompt=[s["text"]], negative_prompt="", template=template, streams=streams,
                conditioning_mode=a.mode,
                camera=s["camera"].unsqueeze(0).to(device=device, dtype=dtype),
                action_7d=s["action_7d"].unsqueeze(0).to(device=device, dtype=dtype),
                extrinsics=s["extrinsics"].unsqueeze(0).to(device=device, dtype=dtype),
                intrinsics=s["intrinsics"].unsqueeze(0).to(device=device, dtype=dtype),
                height=a.res, width=a.res, num_frames=T, cfg_scale=a.cfg,
                num_inference_steps=a.steps, seed=a.seed, tiled=False,
                tile_size=(a.res // 16, a.res // 16), tile_stride=(a.res // 32, a.res // 32),
                enable_usp=False, cfg_parallel=False)])

            plan = G.seg_plan(template, 2, a.mode)
            spans, total = G.spans_for(plan, T)
            assert total == len(frames), f"{key}: spans {total} != returned {len(frames)}"

            rec = {"episode": ep, "split": split_of[ep], "modality": modality,
                   "seconds": round(time.time() - t0, 1)}
            if modality in ACTION_MODALITIES:
                clips = {v: frames[b:e] for (m, v, _sf, _g), (b, e) in zip(plan, spans) if m == "action"}
                # G.score_action already returns {"metric": "pos_err_m", "value": <median>, ...}.
                # Do NOT re-point `value` here: "pos_err_m" is the metric NAME, never a key of the
                # returned dict, so an `if "pos_err_m" in rec` guard silently evaluates False and
                # overwrites the correct median with None. That bug nulled every action cell of
                # the first arm6 run.
                rec.update(G.score_action(clips[0], clips[1], s, T))
            else:
                (b, e) = next(sp for (m, v, _sf, _g), sp in zip(plan, spans) if m == modality and v == 0)
                rec.update(G.score(modality, ds, s, frames[b:e], a.res))
            per_episode[key] = rec
            print(f"  {key:55s} {rec.get('metric')}={rec.get('value')}  ({rec['seconds']}s)", flush=True)

            # RGB quality (PSNR/SSIM/LPIPS) of the SAME generation's video segment, view 0 --
            # every template's first segment is "video" (CANONICAL_ORDER), so this is free: no
            # extra generation call, just another read of frames already produced above. Recorded
            # once per (episode, template-context) rather than once per episode, because whether
            # RGB quality is independent of the companion modality is itself worth being able to
            # check, not assumed.
            # The ANCHOR segment's generation quality, free from the same frames. This used to
            # hardcode "video" on the claim that "every template's first segment is video" --
            # true for the video+X family, FALSE for arm7's substitution templates, where the
            # generator is empty and next() raises StopIteration. Take the anchor from the plan.
            anchor = next(m for (m, _v, _sf, _g) in plan if m in G.VISUAL_MODALITIES)
            (vb, ve) = next(sp for (m, v, _sf, _g), sp in zip(plan, spans)
                            if m == anchor and v == 0)
            if anchor == "video":
                gt_video = G._u8(s["streams"]["video"])[:T][: ve - vb]
                vrec = {"episode": ep, "split": split_of[ep], "modality": "video", "via": modality}
                vrec.update(score_video(frames[vb:ve], gt_video, device))
                per_episode[f"{ep}|video__via_{modality}"] = vrec
            else:
                # e.g. the depth video generated INSIDE `depth+action`. Previously unmeasurable:
                # arm6's depth AbsRel came from `video+depth` (RGB given -> depth predicted),
                # a different template AND a different conditioning, so it never transferred here.
                arec = {"episode": ep, "split": split_of[ep], "modality": anchor, "via": modality}
                arec.update(G.score(anchor, ds, s, frames[vb:ve], a.res))
                per_episode[f"{ep}|{anchor}__via_{modality}"] = arec

            with open(out_dir / "per_episode.json", "w") as f:
                json.dump({"tag": a.tag, "ckpt": a.ckpt, "mode": a.mode, "seed": a.seed,
                           "segmentation_mode": a.segmentation_mode, "mode": a.mode,
                           "results": per_episode}, f, indent=1)

    def _agg(recs, field):
        vals = [r[field] for r in recs if r.get(field) is not None]
        n_dropped = len(recs) - len(vals)
        mean, lo, hi = bootstrap_ci(vals, a.n_boot, a.seed)
        return {"n": len(vals), "n_dropped_none": n_dropped,
                "mean": round(mean, 5) if mean is not None else None,
                "ci95_lo": round(lo, 5) if lo is not None else None,
                "ci95_hi": round(hi, 5) if hi is not None else None}

    # Aggregate over every modality PRESENT in per_episode.json, not just the ones this
    # invocation was asked to generate. With resume enabled a partial re-run (say
    # `--modalities action` to repair one broken modality) would otherwise rewrite summary.json
    # with only that modality and silently drop the other three, even though their records are
    # still sitting in per_episode.json untouched.
    present_mods = [m for m in list(MODALITY_TEMPLATE) + ["video"]
                    if any(r["modality"] == m for r in per_episode.values())]
    splits = sorted({r["split"] for r in per_episode.values()})
    summary = {}
    for modality in [m for m in present_mods if m != "video"]:
        for split in splits:
            recs = [r for r in per_episode.values() if r["modality"] == modality and r["split"] == split]
            entry = _agg(recs, "value")
            entry["metric"] = next((r["metric"] for r in recs if r.get("metric")), None)
            # Action carries more than its headline metric; tab:wam_preservation wants rotation
            # error alongside position error, and score_action has been recording it all along.
            if modality in ACTION_MODALITIES:
                for extra in ("pos_err_mean_m", "rot_err_med_deg", "rot_err_over90_frac",
                              "gripper_acc", "r_peak"):
                    if any(extra in r for r in recs):
                        entry[extra] = _agg(recs, extra)
            summary[f"{modality}|{split}"] = entry

    # RGB quality is pooled across every template context ("via" depth/segmentation/normal/action)
    # -- IIII always predicts future RGB regardless of the companion stream, so whether that
    # quality is actually independent of the companion modality is itself checked, not assumed,
    # by ALSO reporting a per-`via` breakdown alongside the pooled number.
    for split in splits:
        vrecs = [r for r in per_episode.values() if r["modality"] == "video" and r["split"] == split]
        entry = _agg(vrecs, "value")
        entry["metric"] = "lpips"
        entry["psnr_db"] = _agg(vrecs, "psnr_db")
        entry["ssim"] = _agg(vrecs, "ssim")
        entry["by_via"] = {
            via: _agg([r for r in vrecs if r["via"] == via], "value")
            for via in sorted({r["via"] for r in vrecs})
        }
        summary[f"video|{split}"] = entry

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"tag": a.tag, "ckpt": a.ckpt, "mode": a.mode, "n_boot": a.n_boot,
                   "episodes": episodes, "summary": summary}, f, indent=1)

    print("\n=== held-out batch summary (mean, 95% bootstrap CI) ===")
    for modality in present_mods:
        for split in splits:
            r = summary.get(f"{modality}|{split}")
            # A split can be present as a key but hold no scored episodes -- the unseen-task
            # tree has no variation 1, so its "unseen" bucket is empty and n is None. This is
            # only the summary printout; per_episode.json and summary.json are already on
            # disk by now, so crashing here used to look exactly like a failed run.
            if r is None or r.get("n") is None:
                continue
            print(f"  {modality:14s} {split:7s} n={r['n']:<3d} (dropped {r['n_dropped_none']}) "
                  f"{r['metric']:>12s} = {r['mean']}  [{r['ci95_lo']}, {r['ci95_hi']}]")
            if modality == "video":
                via_str = ", ".join(f"{k}: {v['mean']}" for k, v in r["by_via"].items())
                print(f"    psnr_db = {r['psnr_db']['mean']}, ssim = {r['ssim']['mean']}, "
                      f"by_via = {{{via_str}}}")
    print(f"\nwrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
