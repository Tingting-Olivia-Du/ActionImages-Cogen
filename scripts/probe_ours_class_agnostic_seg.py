"""Our own class-agnostic best-overlap IoU, so the SAM comparison is symmetric.

scripts/probe_sam_vs_ours.py measured SAM under a label-free protocol: for every ground-truth
role region, the best IoU achieved by ANY proposal. That number (0.619) cannot be set against our
macro-mIoU (0.461), because macro-mIoU additionally requires every region to be given the right
role name, and because SAM emits ~13x more masks than there are regions -- best-overlap rewards
over-segmentation. Scoring ourselves under the identical protocol is the only way to say anything
about relative boundary quality.

The GT regions, the frame indices and the best_overlap function are imported from the SAM probe
rather than reimplemented, so the two runs cannot silently diverge.

BOTH CONDITIONING MODES ARE REPORTED. SAM is handed a real RGB frame, so FIFI (RGB given, only
the mask predicted) is the like-for-like setting. IIII, where the RGB is generated too, is the
harder regime the rest of the paper reports; it is included so the two numbers in the paper stay
connected, not because it is comparable with SAM.
"""
import os, sys, time
import numpy as np, torch
from types import SimpleNamespace
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)

import scripts.modality_mode_grid as G
from scripts.probe_sam_vs_ours import EPS, FRAMES, best_overlap, DATA_TREE, SPLIT_TAG
from inference import build_pipeline
from training.dataset import RLBenchSelfgenDataset
from training.percep.seg_codec import decode_scene_roles, scene_role_labels, ROLE_TO_LABEL



# --- durable output ---------------------------------------------------------
# ~4h of generation, so each mode's result is written as soon as it finishes and
# an already-finished mode is skipped on a restart.
import json as _json
_EXT = os.path.join(REPO, "reports", "external")
MODEL_TAG = os.environ.get("MODEL_TAG", "ours")
def _p(mode, n): return os.path.join(_EXT, f"{MODEL_TAG}_classagnostic_{mode}{SPLIT_TAG}_n{n}.json")
def _save(mode, n, payload):
    os.makedirs(_EXT, exist_ok=True)
    with open(_p(mode, n), "w") as f: _json.dump(dict(payload, n_episodes=n), f, indent=1)
    print(f"[saved] {_p(mode, n)}", flush=True)

def main():
    modes = (sys.argv[1] if len(sys.argv) > 1 else "fifi,iiii").split(",")
    n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    G.reserve_vram(27000)
    ds = RLBenchSelfgenDataset(base_path=DATA_TREE,
        num_frames=41, frame_interval=3, height=512, width=512,
        template_mix="video+segmentation@1.0", prompt_tag_style="explicit",
        strict_getitem=True, variations="all", segmentation_mode="scene_roles")
    pipe = build_pipeline(SimpleNamespace(model_id="Wan-AI/Wan2.2-TI2V-5B",
        ckpt_path=os.environ.get("CKPT", "outputs/.grid_pin/step10000.ckpt"), height=512, width=512,
        use_usp=False, cfg_parallel=False, dynamic_cache_schedule=False, torch_compile=False))
    dev, dt, T = pipe.device, torch.bfloat16, 41

    for mode in modes:
        if os.path.exists(_p(mode, n_ep)):
            print(f'[skip] {mode} n={n_ep} already done'); continue
        scores, ours_counts, gt_counts = [], [], []
        for ep in EPS[:n_ep]:
            want = os.path.normpath(os.path.join(DATA_TREE, ep))
            idx = next(i for i, e in enumerate(ds.episodes) if os.path.normpath(e["path"]) == want)
            G._seed_all(42)
            s = ds.getitem(idx, force_template="video+segmentation")
            if s["template"] != "video+segmentation":
                continue
            vd = [s["view_dirs"][i] for i in s["view_indices"]][0]; fi = s["frame_indices"]
            instr = s["text"].split("> ", 1)[-1]
            lut, present = ds._scene_role_lut(s["path"], instr)
            hm = ds._to_model_res(ds._load_mask(vd)[fi]).astype(np.uint16)
            streams = {k: v.unsqueeze(0).to(device=dev, dtype=dt) for k, v in s["streams"].items()}
            G._seed_all(42)
            t0 = time.time()
            frames = np.stack([np.asarray(f) for f in pipe(
                prompt=[s["text"]], negative_prompt="", template="video+segmentation",
                streams=streams, conditioning_mode=mode,
                camera=s["camera"].unsqueeze(0).to(device=dev, dtype=dt),
                action_7d=s["action_7d"].unsqueeze(0).to(device=dev, dtype=dt),
                extrinsics=s["extrinsics"].unsqueeze(0).to(device=dev, dtype=dt),
                intrinsics=s["intrinsics"].unsqueeze(0).to(device=dev, dtype=dt),
                height=512, width=512, num_frames=T, cfg_scale=7.5, num_inference_steps=50,
                seed=42, tiled=False, tile_size=(32, 32), tile_stride=(16, 16),
                enable_usp=False, cfg_parallel=False)])
            plan = G.seg_plan("video+segmentation", 2, mode)
            spans, total = G.spans_for(plan, T)
            assert total == len(frames)
            b, e = next(sp for (m, v, _s, _g), sp in zip(plan, spans)
                        if m == "segmentation" and v == 0)
            pred = frames[b:e]
            ep_s = []
            for t in FRAMES:
                if t >= len(pred): continue
                gt_lab = scene_role_labels(hm[t][None], lut)[0]
                regions = [gt_lab == ROLE_TO_LABEL[r] for r in present if r != "background"]
                dec = decode_scene_roles(pred[t][None], present_roles=present)
                masks = [dec[r][0] for r in dec if r not in ("background", "unknown")
                         and dec[r][0].sum() > 0]
                sc = best_overlap(regions, masks)
                if sc:
                    ep_s += sc; ours_counts.append(len(masks))
                    gt_counts.append(len([r for r in regions if r.sum() >= 200]))
            if ep_s:
                scores += ep_s
                print(f"  [{mode}] {ep.split('/')[0]:24s} best-overlap={np.mean(ep_s):.4f} "
                      f"({len(ep_s)} regions, {time.time()-t0:.0f}s)", flush=True)
        print(f"\n=== 我们 ({mode}), {len(scores)} 个 GT 区域 ===")
        print(f"  class-agnostic best-overlap IoU = {np.mean(scores):.4f}")
        print(f"  我们平均每帧 {np.mean(ours_counts):.1f} 个 mask,GT {np.mean(gt_counts):.1f} 个区域"
              f"  (过分割 {np.mean(ours_counts)/max(np.mean(gt_counts),1):.1f}x)")
        _save(mode, n_ep, {'model': f'ours-{mode}', 'best_overlap_iou': float(np.mean(scores)),
                           'n_gt_regions_scored': len(scores),
                           'ours_masks_per_frame': float(np.mean(ours_counts)),
                           'gt_regions_per_frame': float(np.mean(gt_counts))})

if __name__ == "__main__":
    main()
