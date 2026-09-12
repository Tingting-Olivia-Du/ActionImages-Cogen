#!/usr/bin/env python
"""Open-vocabulary counterpart for NAMED scene-role segmentation.

The main table currently reports n/a here on the grounds that SAM emits unnamed regions.
That is true of SAM and false of the field: open-vocabulary segmenters take the role name
as text and return a mask for it, which is exactly our task. Leaving n/a would claim no one
can do this when we simply had not run one.

CLIPSeg is the closest fit -- it is text-conditioned dense segmentation, so each scene role
becomes a prompt and the argmax over roles yields a label map scored with the identical
macro-mIoU our own segmentation uses. The roles present in an episode are given to it, the
same set our decoder is restricted to, so neither side has to guess the vocabulary.

Caveat for the paper: CLIPSeg is trained on real photographs and predicts per-frame from a
single image, while our model emits the whole clip; and role names like "fixture" or
"distractor" are ours, not natural language it was trained on. It is a floor for what an
off-the-shelf open-vocabulary model achieves here, not a ceiling for the approach.
"""
import os, sys, json, random
import numpy as np, torch, torch.nn.functional as F
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from training.dataset import RLBenchSelfgenDataset
from training.percep.seg_codec import scene_role_labels, ROLE_TO_LABEL
from scripts.probe_sam_vs_ours import EPS, FRAMES, DATA_TREE, SPLIT_TAG

# Role -> the phrase CLIPSeg is actually asked for. Our internal names are not English.
ROLE_PROMPT = {
    "target":     "the object the robot is manipulating",
    "goal":       "the container or place where the object should go",
    "distractor": "another similar object that is not the target",
    "fixture":    "the table or furniture in the scene",
    "robot_arm":  "the robot arm",
    "gripper":    "the robot gripper fingers",
    "background": "the background wall and floor",
}

def _seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def main():
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    out_p = os.path.join(REPO, "reports", "external", f"namedseg_clipseg{SPLIT_TAG}_n{n_ep}.json")
    if os.path.exists(out_p):
        print(f"[skip] {out_p} exists"); print(open(out_p).read()); return
    dev = "cuda"
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    net = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(dev).eval()

    ds = RLBenchSelfgenDataset(base_path=DATA_TREE, num_frames=41, frame_interval=3,
        height=512, width=512, template_mix="video+segmentation@1.0",
        prompt_tag_style="explicit", strict_getitem=True, variations="all",
        segmentation_mode="scene_roles")
    per_ep = []
    for ep in EPS[:n_ep]:
        want = os.path.normpath(os.path.join(DATA_TREE, ep))
        idx = next(i for i, e in enumerate(ds.episodes) if os.path.normpath(e["path"]) == want)
        _seed_all(42)
        s = ds.getitem(idx, force_template="video+segmentation")
        vd = [s["view_dirs"][i] for i in s["view_indices"]][0]; fi = s["frame_indices"]
        instr = s["text"].split("> ", 1)[-1]
        lut, present = ds._scene_role_lut(s["path"], instr)
        hm = ds._to_model_res(ds._load_mask(vd)[fi]).astype(np.uint16)
        rgb = ((s["streams"]["video"].permute(1,2,3,0).numpy()+1)*127.5).astype(np.uint8)
        roles = [r for r in present]
        prompts = [ROLE_PROMPT.get(r, r.replace("_", " ")) for r in roles]
        ious = []
        for t in FRAMES:
            if t >= len(hm): continue
            from PIL import Image
            inp = proc(text=prompts, images=[Image.fromarray(rgb[t])]*len(prompts),
                       padding=True, return_tensors="pt").to(dev)
            with torch.no_grad():
                logits = net(**inp).logits                       # [R,352,352]
            if logits.ndim == 2: logits = logits[None]
            up = F.interpolate(logits[None].float(), size=hm.shape[-2:],
                               mode="bilinear", align_corners=False)[0]
            pred = np.array(roles)[up.argmax(0).cpu().numpy()]    # per-pixel role name
            gt_lab = scene_role_labels(hm[t][None], lut)[0]
            for r in roles:
                g = (gt_lab == ROLE_TO_LABEL[r])
                if g.sum() == 0: continue
                p = (pred == r)
                u = (g | p).sum()
                ious.append(float((g & p).sum() / u) if u else 0.0)
        if ious:
            per_ep.append(float(np.mean(ious)))
            print(f"  {ep.split('/')[0]:24s} macro-mIoU={per_ep[-1]:.4f} ({len(ious)} roles)", flush=True)
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    json.dump({"model": "CLIPSeg rd64-refined (open-vocabulary)",
               "macro_miou": float(np.mean(per_ep)),
               "median": float(np.median(per_ep)),
               "per_episode": per_ep, "n_episodes": n_ep}, open(out_p, "w"), indent=1)
    print(f"\nCLIPSeg named scene-role macro-mIoU = {np.mean(per_ep):.4f}  (n={len(per_ep)})")
    print(f"[saved] {out_p}")

if __name__ == "__main__":
    main()
