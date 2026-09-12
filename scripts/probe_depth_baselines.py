"""External monocular-depth baselines on the same held-out episodes.

MODELS. Depth Anything V3 (depth-anything/DA3-Large) and UniDepth V2
(lpiccinelli/unidepth-v2-vitl14). They differ in a way that matters here:

  UniDepth V2 predicts METRIC depth given intrinsics, so it can be scored directly, exactly as
  our codec is. It is the only baseline that competes on our own terms.
  Depth Anything V3 predicts relative/affine-invariant depth, so it needs alignment, like VGGT.

We therefore report both raw and median-aligned AbsRel for every model, and let the alignment
column carry the point: needing a free scale parameter fitted against the ground truth is itself
a capability difference, not a nuisance to be normalised away.

CAVEAT, stated once and applying to all of them: every external model here is ZERO-SHOT on
RLBench renders while ours is fine-tuned on this exact distribution. These numbers bound how hard
the depth task is; they do not rank the models.
"""
import os, sys, random
from scipy.stats import spearmanr
import numpy as np, torch, torch.nn.functional as F
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

def median_align(d, gt):
    """Per-frame median scaling, for a model that predicts depth up to an unknown scale."""
    return np.stack([d[t] * (np.median(gt[t][gt[t] > MIN_VALID]) / max(np.median(d[t]), 1e-6))
                     for t in range(len(gt))])


def disparity_align(disp, gt):
    """Affine-invariant alignment IN DISPARITY SPACE, then invert to depth.

    Depth Anything's relative models emit inverse depth, not depth: larger means nearer. Median
    scaling a disparity map against a depth map is meaningless -- it was giving AbsRel above 7 --
    so the standard protocol is used instead: least-squares fit of scale and shift from the
    prediction to 1/GT, then invert. This is the alignment those papers evaluate under, and it is
    strictly more generous than the median scaling applied to the metric-space models.
    """
    out = []
    for t in range(len(gt)):
        g = gt[t]; m = (g > MIN_VALID) & (g < MAX_VALID) & np.isfinite(g)
        if m.sum() < 100:
            out.append(np.full_like(g, np.nan)); continue
        x = disp[t][m].astype(np.float64); y = (1.0 / g[m]).astype(np.float64)
        A = np.stack([x, np.ones_like(x)], 1)
        sc, sh = np.linalg.lstsq(A, y, rcond=None)[0]
        d_hat = sc * disp[t] + sh
        out.append(1.0 / np.clip(d_hat, 1e-6, None))
    return np.stack(out)


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
    which = sys.argv[1]; n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if _already_done(which, n_ep): return
    dev = "cuda"
    if which == "da3":
        # DA3 ships its own package (its HF repo is not a transformers config). It takes a LIST of
        # frames and returns a prediction object, and unlike DA V2 it produces depth rather than
        # disparity, so it is aligned by median like the other metric-space models.
        from depth_anything_3.api import DepthAnything3
        m = DepthAnything3.from_pretrained("depth-anything/DA3-Large").to(dev).eval()
        def infer(rgb):
            with torch.no_grad():
                pr = m.inference([f for f in rgb])
            d = getattr(pr, "depth", None)
            if d is None: d = pr["depth"]
            d = d.float().cpu().numpy() if torch.is_tensor(d) else np.asarray(d, np.float32)
            return np.squeeze(d)
    elif which in ("depthpro", "da2metric"):
        # Metric predictors: they emit metres directly, so they face the same zero-free-parameter
        # scoring our codec does. These are the counterparts that can actually beat us on the
        # asymmetry we lean on, which is why they belong in the sweep.
        from transformers import pipeline
        from PIL import Image
        repo = ("apple/DepthPro-hf" if which == "depthpro"
                else "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
        pipe = pipeline("depth-estimation", model=repo, device=0)
        def infer(rgb):
            outs = []
            for f in rgb:
                d = pipe(Image.fromarray(f))["predicted_depth"]
                d = d.float().cpu().numpy() if torch.is_tensor(d) else np.asarray(d, np.float32)
                outs.append(np.squeeze(d))
            return np.stack(outs)
    elif which in ("lotusdepth", "marigolddepth"):
        from PIL import Image
        if which == "lotusdepth":
            sys.path.insert(0, "/workspace/ttdu/third_party_lotus")
            from pipeline import LotusGPipeline
            pl = LotusGPipeline.from_pretrained("jingheya/lotus-depth-g-v2-0-disparity",
                                                torch_dtype=torch.float16).to(dev)
            pl.set_progress_bar_config(disable=True)
            te = torch.tensor([1, 0]).float().unsqueeze(0).to(dev)
            te = torch.cat([torch.sin(te), torch.cos(te)], dim=-1)
            def infer(rgb):
                outs = []
                for f in rgb:
                    x = torch.tensor(f.astype(np.float32)).permute(2,0,1)[None].to(dev)/127.5 - 1.0
                    with torch.no_grad(), torch.autocast(dev):
                        o = pl(rgb_in=x, prompt="", num_inference_steps=1, timesteps=[999],
                               task_emb=te, output_type="np",
                               generator=torch.Generator(device=dev).manual_seed(42)).images[0]
                    outs.append(np.asarray(o, np.float32).mean(-1))
                return np.stack(outs)
        else:
            from diffusers import MarigoldDepthPipeline
            pl = MarigoldDepthPipeline.from_pretrained("prs-eth/marigold-depth-v1-1",
                                                       torch_dtype=torch.float16).to(dev)
            pl.set_progress_bar_config(disable=True)
            def infer(rgb):
                outs = []
                for f in rgb:
                    with torch.no_grad():
                        o = pl(Image.fromarray(f),
                               generator=torch.Generator(device=dev).manual_seed(42)).prediction
                    o = o.detach().cpu().numpy() if torch.is_tensor(o) else np.asarray(o)
                    outs.append(np.squeeze(o).astype(np.float32))
                return np.stack(outs)
    elif which == "da2":
        # DA3's HF repo ships a bare {model_name, config} json, not a transformers config, so it
        # needs the official depth-anything-3 package. DA V2 is natively supported and is the
        # same family, so it is the baseline that runs without a source install.
        from transformers import pipeline
        pipe = pipeline("depth-estimation",
                        model="depth-anything/Depth-Anything-V2-Large-hf", device=0)
        from PIL import Image
        def infer(rgb):                      # [T,H,W,3] uint8 -> [T,H,W]
            outs = []
            for f in rgb:
                d = pipe(Image.fromarray(f))["predicted_depth"]
                d = d.float().cpu().numpy() if torch.is_tensor(d) else np.asarray(d, np.float32)
                outs.append(np.squeeze(d))
            return np.stack(outs)
    else:
        from unidepth.models import UniDepthV2
        model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to(dev).eval()
        def infer(rgb, K=None):
            outs = []
            for f in rgb:
                x = torch.from_numpy(f).permute(2,0,1).to(dev)
                with torch.no_grad():
                    p = model.infer(x, K)
                outs.append(p["depth"].squeeze().float().cpu().numpy())
            return np.stack(outs)

    ds = RLBenchSelfgenDataset(base_path=DATA_TREE,
        num_frames=41, frame_interval=3, height=512, width=512,
        template_mix="video+depth@1.0", prompt_tag_style="explicit",
        strict_getitem=True, variations="all", segmentation_mode="scene_roles")
    raw, ali = [], []
    rhos, ali_med, ali_disp = [], [], []
    for ep in EPS[:n_ep]:
        want = os.path.normpath(os.path.join(DATA_TREE, ep))
        idx = next(i for i,e in enumerate(ds.episodes) if os.path.normpath(e["path"])==want)
        _seed_all(42)
        s = ds.getitem(idx, force_template="video+depth")
        vd = [s["view_dirs"][i] for i in s["view_indices"]][0]; fi = s["frame_indices"]
        gt = ds._to_model_res(ds._load_depth(vd)[fi].astype(np.float32))
        rgb = ((s["streams"]["video"].permute(1,2,3,0).numpy()+1)*127.5)[:len(gt)].astype(np.uint8)
        sub = list(range(0, len(gt), 4))                 # every 4th frame: these run per-image
        d = infer(rgb[sub])
        if d.shape[-2:] != gt.shape[-2:]:
            d = F.interpolate(torch.from_numpy(d)[:,None], size=gt.shape[-2:],
                              mode="bilinear", align_corners=False)[:,0].numpy()
        g = gt[sub]
        # Which output space a model uses is decided from the data, not from its docs.
        # Assuming DA V2 emitted depth once produced AbsRel 7.5 and a conclusion that
        # happened to flatter us; a disparity model anti-correlates with metric depth,
        # so the sign of that correlation is what settles it.
        rho = float(spearmanr(d[np.isfinite(d) & np.isfinite(g)].ravel()[:200000],
                              g[np.isfinite(d) & np.isfinite(g)].ravel()[:200000]).statistic)
        a_med, a_disp = absrel(median_align(d, g), g), absrel(disparity_align(d, g), g)
        r = absrel(d, g)
        rhos.append(rho); ali_med.append(a_med); ali_disp.append(a_disp)
        raw.append(r); ali.append(a_disp if rho < 0 else a_med)
        print(f"  {ep.split('/')[0]:24s} raw={r:.4f}  median={a_med:.4f}  "
              f"disparity={a_disp:.4f}  rho(pred,gt)={rho:+.3f}", flush=True)
    mean_rho = float(np.mean(rhos))
    space = "disparity (inverse depth)" if mean_rho < 0 else "depth"
    # A model that claims metric output is scored the way it claims: no alignment, no free
    # parameters, the same footing our codec is on. Granting it a per-frame scale would hand
    # it a fitted parameter we do not get, which flatters it on exactly the axis this
    # comparison is about. All three numbers are stored either way, so the appendix can show
    # what its geometry is worth once scale is forgiven.
    METRIC = {"depthpro", "da2metric", "unidepth", "metric3d"}
    if which in METRIC:
        chosen = "none (metric, 0 free parameters)"
    else:
        chosen = "disparity-affine" if mean_rho < 0 else "median-scale"
    print(f"\n{which.upper()} on GT RGB, {len(raw)} episodes:")
    print(f"  mean rho(pred, GT depth) = {mean_rho:+.3f}  ->  output space is {space}")
    print(f"  raw AbsRel                 {np.mean(raw):.4f}")
    print(f"  median-scale aligned       {np.mean(ali_med):.4f}")
    print(f"  disparity-affine aligned   {np.mean(ali_disp):.4f}")
    reported = float(np.mean(raw)) if which in METRIC else float(np.mean(ali))
    print(f"  >> reported ({chosen}):    {reported:.4f}")

    _save(which, n_ep, {'model':which.upper(),'absrel_raw':float(np.mean(raw)),
                        'absrel_aligned':reported,
                        'model_class':'metric' if which in METRIC else ('disparity' if mean_rho<0 else 'scale-ambiguous'),
                        'free_parameters_per_frame': 0 if which in METRIC else (2 if mean_rho<0 else 1),
                        'alignment':chosen,
                        'output_space':space,
                        'mean_rho_pred_gt':mean_rho,
                        'absrel_median_scale':float(np.mean(ali_med)),
                        'absrel_disparity_affine':float(np.mean(ali_disp)),
                        'per_episode_aligned':[float(x) for x in ali]})

if __name__ == "__main__":
    main()
