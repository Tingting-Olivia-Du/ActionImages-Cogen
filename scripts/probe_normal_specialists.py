#!/usr/bin/env python
"""Dedicated surface-normal counterparts: Lotus-G and Marigold-Normals.

Normals derived from a depth estimator are a weak stand-in; these are models trained to
predict normals directly, which is what the main table's counterpart column should hold.
Both are run, and the stronger one becomes the counterpart -- picking the weaker would
flatter us.

Coordinate convention is CALIBRATED, not assumed. Our ground-truth normals come from
depth_to_normal, whose frame is x-right, y-DOWN, z-AWAY from the camera (a fronto-parallel
surface gives [0,0,1]). Normal-estimation benchmarks generally use x-right, y-UP,
z-TOWARDS the camera. Getting this wrong silently flips the cosine, so all four plausible
sign conventions are scored and every one is written to the json; the reported number uses
whichever agrees best with ground truth. That criterion is fixed before seeing which model
wins and is identical for both models, so it selects a convention, not a result.

Caveat that belongs in the paper: both models are trained on real photographs and RLBench
frames are synthetic renders, so this is a domain-shifted number for them -- the same
caveat that applies to the depth counterparts.
"""
import os, sys, json, random
import numpy as np, torch, torch.nn.functional as F
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from training.dataset import RLBenchSelfgenDataset
from training.percep.normal_codec import depth_to_normal
from scripts.probe_depth_baselines import DATA_TREE, EPS, SPLIT_TAG

CONVENTIONS = {                       # applied to the model's [-1,1] xyz output
    "identity":      np.array([ 1,  1,  1], np.float32),
    "flip_y":        np.array([ 1, -1,  1], np.float32),
    "flip_z":        np.array([ 1,  1, -1], np.float32),
    "flip_yz":       np.array([ 1, -1, -1], np.float32),
}

def _seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def build(which, dev):
    if which == "lotus":
        sys.path.insert(0, "/workspace/ttdu/third_party_lotus")
        from pipeline import LotusGPipeline
        pipe = LotusGPipeline.from_pretrained("jingheya/lotus-normal-g-v1-1",
                                              torch_dtype=torch.float16).to(dev)
        pipe.set_progress_bar_config(disable=True)
        # infer.py: one step, timestep 999, task_emb=[sin,cos] of [1,0], output in [0,1]
        te = torch.tensor([1, 0]).float().unsqueeze(0).to(dev)
        te = torch.cat([torch.sin(te), torch.cos(te)], dim=-1)
        def run(rgb_u8):                                  # [H,W,3] uint8 -> [H,W,3] in [-1,1]
            x = torch.tensor(rgb_u8.astype(np.float32)).permute(2,0,1)[None].to(dev)/127.5 - 1.0
            with torch.no_grad(), torch.autocast(dev):
                out = pipe(rgb_in=x, prompt="", num_inference_steps=1, timesteps=[999],
                           task_emb=te, output_type="np",
                           generator=torch.Generator(device=dev).manual_seed(42)).images[0]
            return out * 2.0 - 1.0
        return run
    from diffusers import MarigoldNormalsPipeline
    from PIL import Image
    pipe = MarigoldNormalsPipeline.from_pretrained("prs-eth/marigold-normals-v1-1",
                                                   torch_dtype=torch.float16).to(dev)
    pipe.set_progress_bar_config(disable=True)
    def run(rgb_u8):                                   # prediction is already unit xyz in [-1,1]
        with torch.no_grad():
            out = pipe(Image.fromarray(rgb_u8),
                       generator=torch.Generator(device=dev).manual_seed(42))
        p = out.prediction
        p = p.detach().cpu().numpy() if torch.is_tensor(p) else np.asarray(p)
        p = np.squeeze(p)
        if p.ndim == 3 and p.shape[0] == 3:             # [3,H,W] -> [H,W,3]
            p = np.transpose(p, (1, 2, 0))
        return p.astype(np.float32)
    return run

def main():
    which = sys.argv[1]; n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    out_p = os.path.join(REPO, "reports", "external", f"normal_{which}{SPLIT_TAG}_n{n_ep}.json")
    if os.path.exists(out_p):
        print(f"[skip] {out_p} exists"); print(open(out_p).read()); return
    dev = "cuda"
    run = build(which, dev)
    ds = RLBenchSelfgenDataset(base_path=DATA_TREE, num_frames=41, frame_interval=3,
        height=512, width=512, template_mix="video+depth@1.0", prompt_tag_style="explicit",
        strict_getitem=True, variations="all", segmentation_mode="scene_roles")
    per_conv = {k: [] for k in CONVENTIONS}
    for ep in EPS[:n_ep]:
        want = os.path.normpath(os.path.join(DATA_TREE, ep))
        idx = next(i for i, e in enumerate(ds.episodes) if os.path.normpath(e["path"]) == want)
        _seed_all(42)
        s = ds.getitem(idx, force_template="video+depth")
        vd = [s["view_dirs"][i] for i in s["view_indices"]][0]; fi = s["frame_indices"]
        fx, fy = ds._focal_lengths(vd)
        raw = ds._load_depth(vd)[fi].astype(np.float32)
        sx, sy = 512 / raw.shape[-1], 512 / raw.shape[-2]
        gt_d = ds._to_model_res(raw)
        rgb = ((s["streams"]["video"].permute(1,2,3,0).numpy()+1)*127.5)[:len(gt_d)].astype(np.uint8)
        sub = list(range(0, len(gt_d), 4))
        acc = {k: [] for k in CONVENTIONS}
        for t in sub:
            pn = run(rgb[t])
            if pn.shape[:2] != gt_d.shape[-2:]:
                pn = F.interpolate(torch.from_numpy(pn).permute(2,0,1)[None],
                                   size=gt_d.shape[-2:], mode="bilinear",
                                   align_corners=False)[0].permute(1,2,0).numpy()
            pn = pn / np.clip(np.linalg.norm(pn, axis=-1, keepdims=True), 1e-6, None)
            gn = depth_to_normal(gt_d[t], fx*sx, fy*sy)
            for k, sign in CONVENTIONS.items():
                acc[k].append(float(((pn*sign) * gn).sum(-1).mean()))
        for k in CONVENTIONS: per_conv[k].append(float(np.mean(acc[k])))
        best = max(CONVENTIONS, key=lambda k: np.mean(per_conv[k]))
        print(f"  {ep.split('/')[0]:24s} " +
              "  ".join(f"{k}={np.mean(acc[k]):+.4f}" for k in CONVENTIONS), flush=True)
    means = {k: float(np.mean(v)) for k, v in per_conv.items()}
    chosen = max(means, key=means.get)
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    json.dump({"model": {"lotus": "Lotus-G normal v1-1",
                         "marigold": "Marigold-Normals v1-1"}[which],
               "convention_chosen": chosen,
               "mean_cos_all_conventions": means,
               "mean_cos": means[chosen],
               "median_cos": float(np.median(per_conv[chosen])),
               "per_episode": per_conv[chosen], "n_episodes": n_ep}, open(out_p, "w"), indent=1)
    print(f"\n{which}: convention={chosen}  mean cos={means[chosen]:.4f}")
    print(f"  all conventions: { {k: round(v,4) for k,v in means.items()} }")
    print(f"[saved] {out_p}")

if __name__ == "__main__":
    main()
