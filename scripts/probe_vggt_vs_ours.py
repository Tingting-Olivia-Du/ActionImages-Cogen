"""Pilot: how does our depth compare with a dedicated 3D foundation model on the SAME episodes?

WHY THIS IS NOT A DROP-IN BASELINE, and what the script does about it.

1. SCALE. VGGT predicts depth up to an unknown scale. Measured earlier on DROID
   (MULTIDATASET_DEPTH_PLAN.md 2.0) no constant converts it to metres: the ideal per-sample
   scale has 46% CV. Our codec, by contrast, emits metric depth directly. Comparing raw AbsRel
   would therefore score VGGT on a handicap it never signed up for, so we median-align VGGT to
   the ground truth per frame -- the standard protocol for scale-ambiguous depth models -- and
   report our model both ways. That our model needs no alignment is itself a result, not
   something to hide by aligning both.

2. CONDITIONING. VGGT sees a real RGB frame and predicts its depth. Under IIII our model
   GENERATES the future RGB and its depth together, which is a strictly harder problem. The
   comparable setting is FIFI, where the RGB is given and only depth is predicted. We report
   FIFI for the like-for-like number and IIII for the deployment regime, and never present the
   IIII number as if it were a depth-estimation result.
"""
import json, os, sys, random
import numpy as np, torch
from scipy.stats import spearmanr
import torch.nn.functional as F
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
from training.dataset import RLBenchSelfgenDataset
from training.percep.depth_codec import MIN_VALID, MAX_VALID

TASKS = ["close_jar","insert_onto_square_peg","light_bulb_in","meat_off_grill",
         "open_drawer","push_buttons","reach_and_drag","put_item_in_drawer"]
# The SAME held-out episodes Table 1 uses, so the external rows and our rows share a sample size
# and a set of scenes. N_EP_PER_TASK is read from the environment so the external baselines can
# be scaled in step with the main table rather than drifting to a different n.
# The tree and the episode list are overridable so the same probe can be pointed at the
# seen split (variation0) and at the evaluation-only unseen-task tree, which is what the
# merged master table needs to stop the External column from having a single populated cell.
# EPS_FILE wins over the generated list; DATA_TREE selects which tree to read.
DATA_TREE = os.environ.get("DATA_TREE", f"{REPO}/data/rlbench_selfgen_512_aug")
_VAR = os.environ.get("EVAL_VARIATION", "1")
_N = int(os.environ.get("N_EP_PER_TASK", "5"))
_EPS_FILE = os.environ.get("EPS_FILE", "")
if _EPS_FILE:
    EPS = [e for e in open(_EPS_FILE).read().strip().split(",") if e]
else:
    EPS = [f"{t}/variation{_VAR}/episodes/episode{i}" for t in TASKS for i in range(_N)]
SPLIT_TAG = os.environ.get("SPLIT_TAG", "")   # appended to the output json name

def absrel(pred, gt):
    ok = np.isfinite(gt) & (gt > MIN_VALID) & (gt < MAX_VALID) & np.isfinite(pred)
    if not ok.any(): return None
    return float((np.abs(np.clip(pred, MIN_VALID, MAX_VALID) - gt) / np.maximum(gt, MIN_VALID))[ok].mean())


# --- durable output + idempotence -------------------------------------------
# The probes used to print only to stdout, so a result lived in one log file and
# re-running one cost a full GPU pass. Two queue workers can now race for the
# same probe safely: whoever writes the json first wins, the other exits.
import json as _json
_EXT = os.path.join(REPO, "reports", "external")
def _done_path(name, n):
    return os.path.join(_EXT, f"{name}{SPLIT_TAG}_n{n}.json")
def _already_done(name, n):
    p = _done_path(name, n)
    if os.path.exists(p):
        print(f"[skip] {p} already exists -- another worker did this one")
        print(open(p).read())
        return True
    return False
def _save(name, n, payload):
    os.makedirs(_EXT, exist_ok=True)
    p = _done_path(name, n)
    payload = dict(payload); payload["n_episodes"] = n
    with open(p, "w") as f:
        _json.dump(payload, f, indent=1)
    print(f"[saved] {p}")


def _seed_all(s=42):
    """Same seed discipline as heldout_batch_eval / modality_mode_grid.

    getitem draws the camera view and the temporal window at random and records them
    as provenance, so an unseeded probe scores a DIFFERENT view and window than our
    model was scored on -- the comparison stops being paired. Two unseeded runs of
    this same probe differed by up to 0.12 AbsRel per episode.
    """
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def main():
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    if _already_done('vggt', n_ep): return
    from vggt.models.vggt import VGGT
    dev = "cuda"
    model = VGGT.from_pretrained("facebook/VGGT-1B",
                                 cache_dir="/workspace/ttdu/ActionImages/checkpoints").to(dev).eval()
    ds = RLBenchSelfgenDataset(base_path=DATA_TREE,
        num_frames=41, frame_interval=3, height=512, width=512,
        template_mix="video+depth@1.0", prompt_tag_style="explicit",
        strict_getitem=True, variations="all", segmentation_mode="scene_roles")
    raw, aligned, rhos = [], [], []
    for ep in EPS[:n_ep]:
        want = os.path.normpath(os.path.join(DATA_TREE, ep))
        idx = next(i for i,e in enumerate(ds.episodes) if os.path.normpath(e["path"])==want)
        _seed_all(42)
        s = ds.getitem(idx, force_template="video+depth")
        vd = [s["view_dirs"][i] for i in s["view_indices"]][0]
        fi = s["frame_indices"]
        gt = ds._to_model_res(ds._load_depth(vd)[fi].astype(np.float32))          # [T,H,W] metric
        rgb = ((s["streams"]["video"].permute(1,2,3,0).numpy()+1)*127.5)[:len(gt)] # [T,H,W,3] 0-255
        # VGGT's patch embedding requires a multiple of 14, so run at 518 and bring the
        # predicted depth back to 512 before scoring against the ground truth.
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            im = torch.from_numpy(rgb/255.).permute(0,3,1,2).float().to(dev)
            im = F.interpolate(im, size=(518,518), mode="bilinear", align_corners=False)
            out = model(im[None])
        d = out["depth"].squeeze().float()
        if d.ndim == 4: d = d[..., 0]
        d = F.interpolate(d[:,None], size=gt.shape[-2:], mode="bilinear",
                          align_corners=False)[:,0].cpu().numpy()
        # per-frame median alignment: the standard fix for a scale-ambiguous predictor.
        # The sign of rho(pred, GT) is recorded rather than assumed -- a disparity model
        # anti-correlates with metric depth, and median-scaling one would be meaningless.
        al = np.stack([d[t] * (np.median(gt[t][gt[t]>MIN_VALID]) / max(np.median(d[t]),1e-6))
                       for t in range(len(gt))])
        ok = np.isfinite(d) & np.isfinite(gt)
        rho = float(spearmanr(d[ok].ravel()[:200000], gt[ok].ravel()[:200000]).statistic)
        r, a = absrel(d, gt), absrel(al, gt)
        raw.append(r); aligned.append(a); rhos.append(rho)
        print(f"  {ep.split('/')[0]:24s} raw={r:.4f}  median-aligned={a:.4f}  "
              f"rho(pred,gt)={rho:+.3f}", flush=True)
    mean_rho=float(np.mean(rhos))
    print(f"\nVGGT-1B on GT RGB, {len(raw)} episodes:")
    print(f"  mean rho(pred, GT depth) = {mean_rho:+.3f}  ->  "
          f"{'depth (median-scale alignment is correct)' if mean_rho>0 else 'DISPARITY -- alignment is WRONG'}")
    print(f"  raw AbsRel            {np.mean(raw):.4f}   (未对齐 —— VGGT 不输出米制)")
    print(f"  median-aligned AbsRel {np.mean(aligned):.4f}   ← 可比的那个数")
    print(f"\n我们的模型(同一批 held-out episode,输出直接就是米制,无需对齐):")
    print(f"  IIII(同时生成 RGB 与 depth)  0.154")
    print(f"  FIFI(给定 RGB,只预测 depth)  见下一步")

    _save('vggt', n_ep, {'model':'VGGT-1B','absrel_raw':float(np.mean(raw)),
                         'absrel_median_aligned':float(np.mean(aligned)),
                         'per_episode_aligned':[float(x) for x in aligned],
                         'mean_rho_pred_gt':mean_rho,
                         'output_space':'depth' if mean_rho>0 else 'disparity'})

if __name__ == "__main__":
    main()
