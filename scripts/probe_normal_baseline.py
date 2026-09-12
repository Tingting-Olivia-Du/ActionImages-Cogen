#!/usr/bin/env python
"""Surface-normal counterpart for the main table.

The table's normal row had no counterpart, which would read as "nobody else does this"
when the truth is "we never ran one". Normals from a depth estimator are the standard
baseline and need no new checkpoint: run Depth Anything V2, align it to the ground truth
the same way the depth row does, then apply the identical depth_to_normal operator the
ground-truth normals are built with. Any gap is then the depth model's geometry, not a
difference in how normals are defined.
"""
import os, sys, random
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import spearmanr
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from training.dataset import RLBenchSelfgenDataset
from training.percep.depth_codec import MIN_VALID, MAX_VALID
from training.percep.normal_codec import depth_to_normal
from scripts.probe_depth_baselines import DATA_TREE, EPS, SPLIT_TAG, disparity_align

def _seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def main():
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    out = os.path.join(REPO, "reports", "external", f"normal_da2{SPLIT_TAG}_n{n_ep}.json")
    if os.path.exists(out):
        print(f"[skip] {out} exists"); print(open(out).read()); return
    from transformers import pipeline
    from PIL import Image
    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", device=0)
    ds = RLBenchSelfgenDataset(base_path=DATA_TREE, num_frames=41, frame_interval=3,
        height=512, width=512, template_mix="video+depth@1.0", prompt_tag_style="explicit",
        strict_getitem=True, variations="all", segmentation_mode="scene_roles")
    scores = []
    for ep in EPS[:n_ep]:
        want = os.path.normpath(os.path.join(DATA_TREE, ep))
        idx = next(i for i, e in enumerate(ds.episodes) if os.path.normpath(e["path"]) == want)
        _seed_all(42)
        s = ds.getitem(idx, force_template="video+depth")
        vd = [s["view_dirs"][i] for i in s["view_indices"]][0]; fi = s["frame_indices"]
        fx, fy = ds._focal_lengths(vd)
        raw = ds._load_depth(vd)[fi].astype(np.float32)
        sx, sy = 512 / raw.shape[-1], 512 / raw.shape[-2]
        gt = ds._to_model_res(raw)
        rgb = ((s["streams"]["video"].permute(1,2,3,0).numpy()+1)*127.5)[:len(gt)].astype(np.uint8)
        sub = list(range(0, len(gt), 4))
        d = np.stack([np.squeeze(np.asarray(pipe(Image.fromarray(f))["predicted_depth"], np.float32))
                      for f in rgb[sub]])
        if d.shape[-2:] != gt.shape[-2:]:
            d = F.interpolate(torch.from_numpy(d)[:,None], size=gt.shape[-2:],
                              mode="bilinear", align_corners=False)[:,0].numpy()
        g = gt[sub]
        d = disparity_align(d, g)                       # DA-V2 emits disparity (rho<0, verified)
        pn = np.stack([depth_to_normal(d[t], fx*sx, fy*sy) for t in range(len(d))])
        gn = np.stack([depth_to_normal(g[t], fx*sx, fy*sy) for t in range(len(g))])
        c = float((pn*gn).sum(-1).mean()); scores.append(c)
        print(f"  {ep.split('/')[0]:24s} cos={c:.4f}", flush=True)
    import json
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"model": "Depth Anything V2-L -> depth_to_normal",
               "mean_cos": float(np.mean(scores)),
               "median_cos": float(np.median(scores)),
               "per_episode": [float(x) for x in scores], "n_episodes": n_ep},
              open(out, "w"), indent=1)
    print(f"\nnormals from DA-V2 depth, {len(scores)} episodes: mean cos = {np.mean(scores):.4f}")
    print(f"[saved] {out}")

if __name__ == "__main__":
    main()
