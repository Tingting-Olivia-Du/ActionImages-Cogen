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
  exec_gap_m    commanded vs ACHIEVED end-effector position. The imagined frame belongs to the
                pose the model DREW, the render belongs to the pose the controller REACHED.
                That gap is the protocol's own noise floor.

ROWS. One dump can feed several rows of Tables/tab_main.tex (Table 2). A row is named
`<modality>:<route>[:<expert>]`:

  depth:direct  normal:direct  seg:direct  rgb:direct      the arm's own native output
  depth:cascade:<e>  normal:cascade:<e>  seg:cascade:<e>   a frozen expert reads the arm's
                                                           IMAGINED RGB (video-anchored dumps)
  depth:ceiling:<e>  normal:ceiling:<e>  seg:ceiling:<e>   the same expert reads the SIMULATOR's
                                                           lossless RGB. Non-deployable upper
                                                           bound on the cascade; also the ONLY
                                                           row an expert may be selected on.

`--rows direct` (default) scores whatever the anchor natively produced. `--rows ceiling` sweeps
every candidate expert and writes ceiling.json, which names the winner per modality.
`--rows cascade` then reads ceiling.json and uses ONLY those winners -- choosing an expert by
the cascade row would be selecting on the number being reported. See PERCEP_SIMREPLAY.md sec. 6.

FRAMES THAT ARE DROPPED, and why each is not a choice:
  is_ramp    rollout.py's `skip_anchor_frames` OVERWRITES the first frames of every chunk with
             `ramp_from_current`. Those executed poses are not what the model drew.
  k == 0     the anchor itself, which is ground truth handed over.
"""
import argparse, fnmatch, glob, json, os, sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from training.percep.depth_codec import MAX_VALID, MIN_VALID, decode_depth
from training.percep.normal_codec import decode_normal, depth_to_normal
from training.percep.seg_codec import (
    ROLE_TO_LABEL, SCENE_ROLES, UNKNOWN_LABEL, decode_scene_roles)
from scripts.modality_mode_grid import seg_plan, spans_for

ANCHOR_TEMPLATE = {"video": "video+action", "depth": "depth+action",
                   "segmentation": "segmentation+action", "normal": "normal+action"}
# anchor modality of a dump -> the direct row it feeds
DIRECT_ROW = {"depth": "depth:direct", "normal": "normal:direct",
              "segmentation": "seg:direct", "video": "rgb:direct"}
METRIC = {"depth": "absrel", "normal": "mean_cos", "seg": "miou_merged", "rgb": "lpips"}
LOWER_IS_BETTER = {"depth": True, "normal": False, "seg": False, "rgb": True}

# Dynamic region: pixels whose COUNTERFACTUAL depth moved >1 cm against the chunk's reference
# render. Every modality uses this same region -- including segmentation and RGB -- so a row's
# `dyn` numbers all cover the same pixels. 1 cm is the codec's resolution scale, not a tuned
# threshold; `dyn_frac` is reported so the reader sees how much of the frame it is.
DYN_THRESH_M = 0.01
MIN_DYN_PX = 100


# ============================================================ experts =========================
# Candidates per modality. Several are run on the CEILING row and the strongest becomes the
# counterpart, because picking a weak one would flatter us (the discipline
# probe_normal_specialists.py already states). Depth lists METRIC models only: a
# scale-ambiguous model needs a fitted scale, and until the anchor-frame affine fit exists it
# would be scored on a handicap it never signed up for.
DEPTH_EXPERTS = {
    "da2metric": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    "depthpro":  "apple/DepthPro-hf",
    "unidepth":  "lpiccinelli/unidepth-v2-vitl14",   # NOT on PyPI: pip install git+https://github.com/lpiccinelli-eth/UniDepth
}
NORMAL_EXPERTS = {
    "marigold": "prs-eth/marigold-normals-v1-1",
    "lotus":    "jingheya/lotus-normal-g-v1-1",       # needs the Lotus repo: $LOTUS_REPO
}
# Named scene-role segmentation needs TEXT -> dense mask. Three families, following the
# baselines Vision Banana (arXiv:2604.20329, Tables 2-3) compares against:
#   clipseg      text-conditioned dense segmentation, one pass per role prompt
#   samclip      SAM proposes class-agnostic masks, CLIP names each one ("SAM + CLIP")
#   groundedsam  Grounding DINO finds boxes from the role text, SAM turns boxes into masks
# SAM 3 (Vision Banana's main segmentation baseline) is not here: its weights are gated
# (facebook/sam3, manual approval) and it needs the `sam3` package or transformers >= 5.
SEG_EXPERTS = {
    "clipseg":     "CIDAS/clipseg-rd64-refined",
    "samclip":     "facebook/sam-vit-huge + openai/clip-vit-large-patch14",
    "groundedsam": "IDEA-Research/grounding-dino-base + facebook/sam-vit-huge",
}
# Grounding DINO wants short noun phrases, not the sentences CLIPSeg/CLIP get. `target` and
# `distractor` share one phrase: they are merged at scoring time anyway (see prep_seg), and the
# detector has no way to know which object the instruction refers to. `background` is never
# queried -- it is whatever no detection claims.
GDINO_PHRASE = {"target": "object", "distractor": "object", "goal": "container",
                "fixture": "furniture", "robot_arm": "robot arm", "gripper": "robot gripper",
                "tool": "tool"}
EXPERTS = {"depth": DEPTH_EXPERTS, "normal": NORMAL_EXPERTS, "seg": SEG_EXPERTS}

# Normal experts emit their own axis convention. Ours (depth_to_normal) is x-right, y-DOWN,
# z-AWAY; benchmarks are usually y-up, z-toward. All four are scored on the CEILING row and the
# best is recorded in ceiling.json; the cascade then applies that one. Selecting it on real
# frames selects a convention, not a result -- same rule as probe_normal_specialists.py.
CONVENTIONS = {
    "identity": np.array([1, 1, 1], np.float32),
    "flip_y":   np.array([1, -1, 1], np.float32),
    "flip_z":   np.array([1, 1, -1], np.float32),
    "flip_yz":  np.array([1, -1, -1], np.float32),
}

# Role -> the phrase CLIPSeg is asked for; kept identical to probe_named_seg_baseline.py so the
# appendix probe and this table use the same vocabulary.
ROLE_PROMPT = {
    "target":     "the object the robot is manipulating",
    "goal":       "the container or place where the object should go",
    "distractor": "another similar object that is not the target",
    "fixture":    "the table or furniture in the scene",
    "robot_arm":  "the robot arm",
    "gripper":    "the robot gripper fingers",
    "background": "the background wall and floor",
}

_CACHE = {}


def _dev():
    return int(os.environ.get("SPECIALIST_GPU", "0"))


def _resize(x, hw):
    """[H,W] or [H,W,C] float -> hw, bilinear. Experts do not all return the input size."""
    if x.shape[:2] == tuple(hw):
        return x
    import torch, torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    t = t[None, None] if t.ndim == 2 else t.permute(2, 0, 1)[None]
    t = F.interpolate(t, size=tuple(hw), mode="bilinear", align_corners=False)[0]
    return (t[0] if x.ndim == 2 else t.permute(1, 2, 0)).numpy()


def depth_expert(name):
    """-> callable(rgb_u8 [H,W,3]) -> metric depth [H,W] in metres, 0 free parameters."""
    key = ("depth", name)
    if key in _CACHE:
        return _CACHE[key]
    import torch
    dev = _dev()
    if name == "unidepth":
        from unidepth.models import UniDepthV2
        m = UniDepthV2.from_pretrained(DEPTH_EXPERTS[name]).to(f"cuda:{dev}").eval()
        def infer(rgb_u8):
            x = torch.from_numpy(np.ascontiguousarray(rgb_u8)).permute(2, 0, 1).to(f"cuda:{dev}")
            with torch.no_grad():
                d = m.infer(x)["depth"].squeeze().float().cpu().numpy()
            return _resize(d, rgb_u8.shape[:2])
    else:
        from PIL import Image
        from transformers import pipeline
        pipe = pipeline("depth-estimation", model=DEPTH_EXPERTS[name], device=dev)
        def infer(rgb_u8):
            d = pipe(Image.fromarray(np.ascontiguousarray(rgb_u8)))["predicted_depth"]
            d = d.float().cpu().numpy() if torch.is_tensor(d) else np.asarray(d, np.float32)
            return _resize(np.squeeze(d).astype(np.float32), rgb_u8.shape[:2])
    _CACHE[key] = infer
    return infer


def normal_expert(name):
    """-> callable(rgb_u8) -> unit normals [H,W,3] in the EXPERT's own convention."""
    key = ("normal", name)
    if key in _CACHE:
        return _CACHE[key]
    import torch
    dev = f"cuda:{_dev()}"
    if name == "lotus":
        repo = os.environ.get("LOTUS_REPO", "/workspace/ttdu/third_party_lotus")
        sys.path.insert(0, repo)
        from pipeline import LotusGPipeline
        pipe = LotusGPipeline.from_pretrained(NORMAL_EXPERTS[name], torch_dtype=torch.float16).to(dev)
        pipe.set_progress_bar_config(disable=True)
        te = torch.tensor([1, 0]).float().unsqueeze(0).to(dev)
        te = torch.cat([torch.sin(te), torch.cos(te)], dim=-1)
        def infer(rgb_u8):
            x = torch.tensor(rgb_u8.astype(np.float32)).permute(2, 0, 1)[None].to(dev) / 127.5 - 1.0
            with torch.no_grad(), torch.autocast("cuda"):
                out = pipe(rgb_in=x, prompt="", num_inference_steps=1, timesteps=[999],
                           task_emb=te, output_type="np",
                           generator=torch.Generator(device=dev).manual_seed(42)).images[0]
            n = _resize(out * 2.0 - 1.0, rgb_u8.shape[:2])
            return n / np.clip(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6, None)
    else:
        from diffusers import MarigoldNormalsPipeline
        from PIL import Image
        pipe = MarigoldNormalsPipeline.from_pretrained(NORMAL_EXPERTS[name],
                                                       torch_dtype=torch.float16).to(dev)
        pipe.set_progress_bar_config(disable=True)
        def infer(rgb_u8):
            with torch.no_grad():
                p = pipe(Image.fromarray(rgb_u8),
                         generator=torch.Generator(device=dev).manual_seed(42)).prediction
            p = p.detach().cpu().numpy() if torch.is_tensor(p) else np.asarray(p)
            p = np.squeeze(p)
            if p.ndim == 3 and p.shape[0] == 3:
                p = np.transpose(p, (1, 2, 0))
            n = _resize(p.astype(np.float32), rgb_u8.shape[:2])
            return n / np.clip(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6, None)
    _CACHE[key] = infer
    return infer


def seg_expert(name):
    """-> callable(rgb_u8, present_roles) -> role-label map [H,W] uint8 (argmax over roles)."""
    key = ("seg", name)
    if key in _CACHE:
        return _CACHE[key]
    import torch, torch.nn.functional as F
    from PIL import Image
    dev = f"cuda:{_dev()}"
    bg = ROLE_TO_LABEL["background"]
    if name == "samclip":
        from transformers import CLIPModel, CLIPProcessor, pipeline
        gen = pipeline("mask-generation", model="facebook/sam-vit-huge", device=_dev())
        clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
        cproc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        def infer(rgb_u8, present):
            roles = list(present)
            texts = [f"a photo of {ROLE_PROMPT.get(r, r.replace('_', ' '))}" for r in roles]
            # 64 points per batch, not 256: at 256 the SAM-H mask decoder alone asked for >14 GB
            # and OOM'd on a card shared with other jobs. Same masks, only slower.
            masks = gen(Image.fromarray(rgb_u8), points_per_batch=64)["masks"]
            lab = np.full(rgb_u8.shape[:2], bg, np.uint8)
            if not len(masks):
                return lab
            # Each mask is shown to CLIP on its own: crop to its box, grey out everything
            # outside it, so the label is about THAT region rather than its surroundings.
            crops, keep = [], []
            for m in masks:
                m = np.asarray(m, bool)
                ys, xs = np.nonzero(m)
                if len(ys) < 16:
                    continue
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                c = rgb_u8[y0:y1, x0:x1].copy()
                c[~m[y0:y1, x0:x1]] = 127
                crops.append(Image.fromarray(c)); keep.append(m)
            if not keep:
                return lab
            with torch.no_grad():
                inp = cproc(text=texts, images=crops, return_tensors="pt", padding=True).to(dev)
                logits = clip(**inp).logits_per_image            # [n_masks, n_roles]
            role_idx = logits.argmax(-1).cpu().numpy()
            # Paint large masks first so smaller, more specific ones win overlaps -- the
            # usual SAM convention, since its proposals nest (a gripper inside the arm).
            for j in np.argsort([-m.sum() for m in keep]):
                lab[keep[j]] = ROLE_TO_LABEL[roles[role_idx[j]]]
            return lab
        _CACHE[key] = infer
        return infer
    if name == "groundedsam":
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                                  SamModel, SamProcessor)
        gid = "IDEA-Research/grounding-dino-base"
        gproc = AutoProcessor.from_pretrained(gid)
        gdino = AutoModelForZeroShotObjectDetection.from_pretrained(gid).to(dev).eval()
        sproc = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        sam = SamModel.from_pretrained("facebook/sam-vit-huge").to(dev).eval()
        def infer(rgb_u8, present):
            phrase_to_role = {}
            for r in present:
                if r in GDINO_PHRASE:
                    # `object` serves target and distractor; they are merged downstream.
                    phrase_to_role.setdefault(GDINO_PHRASE[r], "distractor" if r == "target" else r)
            lab = np.full(rgb_u8.shape[:2], bg, np.uint8)
            if not phrase_to_role:
                return lab
            img = Image.fromarray(rgb_u8)
            text = ". ".join(phrase_to_role) + "."
            with torch.no_grad():
                inp = gproc(images=img, text=text, return_tensors="pt").to(dev)
                out = gdino(**inp)
            res = gproc.post_process_grounded_object_detection(
                out, inp.input_ids, threshold=0.3, text_threshold=0.25,
                target_sizes=[rgb_u8.shape[:2]])[0]
            names = res.get("text_labels", res.get("labels"))
            dets = []
            for box, score, nm in zip(res["boxes"].tolist(), res["scores"].tolist(), names):
                nm = str(nm).strip()
                role = phrase_to_role.get(nm)
                if role is None:     # a partial phrase ("robot") that fits several roles: drop it
                    hits = [r for p, r in phrase_to_role.items() if nm and nm in p]
                    role = hits[0] if len(set(hits)) == 1 else None
                if role is not None:
                    dets.append((score, box, role))
            if not dets:
                return lab
            dets.sort(key=lambda t: t[0])                         # highest score painted last
            with torch.no_grad():
                sin = sproc(img, input_boxes=[[d[1] for d in dets]], return_tensors="pt").to(dev)
                so = sam(**sin, multimask_output=False)
            ms = sproc.image_processor.post_process_masks(
                so.pred_masks.cpu(), sin["original_sizes"].cpu(), sin["reshaped_input_sizes"].cpu())[0]
            for j, (_, _, role) in enumerate(dets):
                lab[ms[j, 0].numpy().astype(bool)] = ROLE_TO_LABEL[role]
            return lab
        _CACHE[key] = infer
        return infer
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    proc = CLIPSegProcessor.from_pretrained(SEG_EXPERTS[name])
    net = CLIPSegForImageSegmentation.from_pretrained(SEG_EXPERTS[name]).to(dev).eval()
    def infer(rgb_u8, present):
        roles = list(present)
        prompts = [ROLE_PROMPT.get(r, r.replace("_", " ")) for r in roles]
        inp = proc(text=prompts, images=[Image.fromarray(rgb_u8)] * len(prompts),
                   padding=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            logits = net(**inp).logits
        if logits.ndim == 2:
            logits = logits[None]
        up = F.interpolate(logits[None].float(), size=rgb_u8.shape[:2], mode="bilinear",
                           align_corners=False)[0]
        idx = up.argmax(0).cpu().numpy()
        lut = np.array([ROLE_TO_LABEL[r] for r in roles], np.uint8)
        return lut[idx]
    _CACHE[key] = infer
    return infer


def _lpips():
    if "lpips" not in _CACHE:
        import lpips
        _CACHE["lpips"] = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(f"cuda:{_dev()}").eval()
    return _CACHE["lpips"]


# ============================================================ metrics =========================
# Each modality is `prep(pred, gt) -> (payload, ok)` then `reduce(payload, ok, region)`, so the
# full-frame and dynamic-region numbers come from ONE pass and cannot drift apart.

def prep_depth(pred, gt):
    ok = np.isfinite(gt) & (gt > MIN_VALID) & (gt < MAX_VALID) & np.isfinite(pred)
    rel = np.abs(np.clip(np.nan_to_num(pred, nan=MIN_VALID), MIN_VALID, MAX_VALID) - gt) \
        / np.maximum(gt, MIN_VALID)
    gt_ok = np.isfinite(gt) & (gt > MIN_VALID) & (gt < MAX_VALID)
    return (rel, gt_ok), ok


def prep_normal(pred, gt_n):
    ok = np.isfinite(gt_n).all(-1) & np.isfinite(pred).all(-1)
    return ((np.nan_to_num(pred) * np.nan_to_num(gt_n)).sum(-1), None), ok


def prep_rgb(pred, gt_rgb):
    import torch
    net = _lpips()
    dev = next(net.parameters()).device
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float().to(dev) / 127.5 - 1.0
    with torch.no_grad():
        m = net(t(pred), t(gt_rgb))[0, 0].float().cpu().numpy()
    return (_resize(m, gt_rgb.shape[:2]), None), np.ones(gt_rgb.shape[:2], bool)


# `target` is the one scene role that depends on the INSTRUCTION rather than on the scene:
# training promotes the referred `distractor` objects to `target` from a per-episode
# seg_targets.json (RLBenchSelfgenDataset._scene_role_lut, PROMOTABLE = {"distractor"}). The
# live LUT has no such file -- nor does the unseen-task tree -- so the simulator GT can never
# contain `target`, while the model is trained to paint it. Scoring the two classes separately
# would charge every correct target prediction as an error. Merging target INTO distractor, in
# prediction and GT alike, removes exactly that one instruction-dependent distinction and
# nothing else (target is by construction a promoted distractor). Reported as mIoU_merged.
_TARGET, _DISTRACTOR = ROLE_TO_LABEL["target"], ROLE_TO_LABEL["distractor"]


def _merge_target(lab):
    lab = lab.copy()
    lab[lab == _TARGET] = _DISTRACTOR
    return lab


def prep_seg(pred_lab, gt_lab):
    # `unknown` GT pixels are an annotation gap (unmapped handle), not a scene role: excluded.
    return (_merge_target(pred_lab), _merge_target(gt_lab)), gt_lab != UNKNOWN_LABEL


def reduce(mod, payload, ok, region):
    """-> (value, coverage) over ok & region; value None if too few pixels."""
    m = ok if region is None else (ok & region)
    if mod == "depth":
        rel, gt_ok = payload
        if not m.any():
            return None, 0.0
        base = gt_ok if region is None else (gt_ok & region)
        return float(rel[m].mean()), float(m.sum() / max(base.sum(), 1))
    if mod in ("normal", "rgb"):
        val, _ = payload
        return (float(val[m].mean()), 1.0) if m.any() else (None, 0.0)
    pred_lab, gt_lab = payload            # seg: macro IoU over roles PRESENT in the GT region
    ious = []
    for lab in np.unique(gt_lab[m]):
        g = (gt_lab == lab) & m
        p = (pred_lab == lab) & m
        u = (g | p).sum()
        if g.sum():
            ious.append(float((g & p).sum() / u))
    return (float(np.mean(ious)), 1.0) if ious else (None, 0.0)


# ============================================================ dump access =====================

def _lut_and_present(d):
    """The live scene-role LUT, under whichever key the writing machine used.

    eval/rollout.py writes `seg_lut` + `seg_present`. The second machine patched its own copy
    before that existed, so older dumps may carry a differently named key; accept any 65536-
    long uint8 array with 'lut' in its name. `present` falls back to the roles that actually
    occur in the labelled simulator masks.
    """
    lut = None
    for k in ("seg_lut", "role_lut", "live_role_lut", "lut"):
        if k in d.files:
            lut = d[k]; break
    if lut is None:
        for k in d.files:
            if "lut" in k.lower() and d[k].ndim == 1 and d[k].size == 65536:
                lut = d[k]; break
    if lut is None or "sim_mask" not in d.files:
        return None, None
    present = None
    for k in ("seg_present", "present_roles", "present", "seg_present_roles"):
        if k in d.files:
            present = [str(x) for x in d[k].tolist()]; break
    if present is None:
        labs = np.unique(lut[d["sim_mask"].astype(np.intp)])
        present = [SCENE_ROLES[int(l)] for l in labs if int(l) != UNKNOWN_LABEL]
    if "background" not in present:
        present = ["background"] + present
    return lut.astype(np.uint8), present


def _labels_from_decode(masks, hw):
    lab = np.full(hw, UNKNOWN_LABEL, np.uint8)
    for role, m in masks.items():
        lab[m] = ROLE_TO_LABEL[role]
    return lab


# ============================================================ scoring =======================

def score_one(npz_path, anchor, rows, winners=None, ceiling_stride=4, num_frames=41):
    """-> {row: result dict} for every requested row this dump can support."""
    d = np.load(npz_path)
    if d["canvas"].size == 0:
        return {}
    canvas = d["canvas"]
    plan = seg_plan(ANCHOR_TEMPLATE[anchor], 2, "iiii")
    spans, total = spans_for(plan, num_frames)
    if canvas.shape[1] != total:
        raise RuntimeError(f"{npz_path}: canvas has {canvas.shape[1]} frames, plan wants {total}")
    span = {v: sp for (m, v, _sf, _g, _a), sp in zip(plan, spans) if m == anchor}
    if len(span) != 2:
        raise RuntimeError(f"{npz_path}: expected 2 {anchor!r} segments, got {sorted(span)}")

    has = set(d.files)
    sim_depth = d["sim_depth"].astype(np.float32) if "sim_depth" in has else None
    sim_rgb = d["sim_rgb"] if "sim_rgb" in has else None
    lut, present = _lut_and_present(d)
    intr = d["intrinsics"]
    keep = (~d["is_ramp"]) & (d["k"] > 0)
    exec_gap = np.linalg.norm(d["cmd"][:, :3] - d["ach"][:, :3], axis=-1)

    # ---- which rows this dump can feed ------------------------------------------------------
    todo = []                                   # (row, mod, kind, expert)
    if "direct" in rows:
        todo.append((DIRECT_ROW[anchor], DIRECT_ROW[anchor].split(":")[0], "direct", None))
    if anchor == "video" and "cascade" in rows:
        for mod, e in (winners or {}).items():
            if mod.startswith("_"):          # bookkeeping (scores, normal convention), not experts
                continue
            todo.append((f"{mod}:cascade:{e}", mod, "cascade", e))
    if "ceiling" in rows and sim_rgb is not None:
        for mod, cands in EXPERTS.items():
            for e in cands:
                if mod == "normal":
                    for c in CONVENTIONS:
                        todo.append((f"normal:ceiling:{e}:{c}", "normal", "ceiling", (e, c)))
                else:
                    todo.append((f"{mod}:ceiling:{e}", mod, "ceiling", e))
    skipped = {}
    ok_todo = []
    for row in todo:
        _, mod, kind, e = row
        need = []
        if mod in ("depth", "normal") and sim_depth is None: need.append("sim_depth")
        if mod == "seg" and lut is None: need.append("seg_lut+sim_mask")
        if (mod == "rgb" or kind == "ceiling") and sim_rgb is None: need.append("sim_rgb")
        if need:
            skipped[row[0]] = "dump lacks " + ",".join(need)
        else:
            ok_todo.append(row)
    todo = ok_todo

    ref_row = {}
    for i in np.flatnonzero(d["k"] >= 0):
        ref_row.setdefault(int(d["replan"][i]), i)

    acc = {r[0]: defaultdict(list) for r in todo}
    fail = {}
    gtn_cache, anch_cache, raw_cache = {}, {}, {}
    nfail = defaultdict(int)
    scored_idx = [int(i) for i in np.flatnonzero(keep) if int(d["replan"][i]) < len(canvas)]
    for n_i, i in enumerate(scored_idx):
        rep, k = int(d["replan"][i]), int(d["k"][i])
        for v in (0, 1):
            b, e_ = span[v]
            if k >= e_ - b:
                continue
            pred_frame, anchor_frame = canvas[rep, b + k], canvas[rep, b]
            ri = ref_row.get(rep)
            moved = None
            if sim_depth is not None and ri is not None:
                g1, g0 = sim_depth[i, v], sim_depth[ri, v]
                moved = np.isfinite(g1) & np.isfinite(g0) & (np.abs(g1 - g0) > DYN_THRESH_M)
                if moved.sum() <= MIN_DYN_PX:
                    moved = None

            def gt_for(mod):
                if mod == "depth":
                    return sim_depth[i, v]
                if mod == "normal":
                    if (i, v) not in gtn_cache:
                        fx, fy = abs(float(intr[v][0, 0])), abs(float(intr[v][1, 1]))
                        gtn_cache[(i, v)] = depth_to_normal(sim_depth[i, v], fx, fy)
                    return gtn_cache[(i, v)]
                if mod == "seg":
                    return lut[d["sim_mask"][i, v].astype(np.intp)]
                return sim_rgb[i, v]

            for row, mod, kind, ex in todo:
                if kind == "ceiling" and n_i % ceiling_stride:
                    continue
                try:
                    gt = gt_for(mod)
                    # --- prediction and its copy-anchor floor -------------------------------
                    if kind == "direct":
                        f = {"depth": lambda x: decode_depth(x[None])[0],
                             "normal": lambda x: decode_normal(x[None])[0],
                             # Decode with `target` ALLOWED even though the live GT cannot
                             # contain it: restricting the nearest-colour search to the live
                             # roles would snap the model's red target pixels to the nearest
                             # remaining colour -- purple, i.e. `fixture` -- and a correct
                             # prediction would be scored as the wrong class twice over.
                             "seg": lambda x: _labels_from_decode(
                                 decode_scene_roles(x, present_roles=sorted(set(present) | {"target"})),
                                 x.shape[:2]),
                             "rgb": lambda x: x}[mod]
                        pred, flr = f(pred_frame), f(anchor_frame)
                    else:
                        src = sim_rgb[i, v] if kind == "ceiling" else pred_frame
                        if mod == "depth":
                            fn = depth_expert(ex)
                        elif mod == "normal":
                            name, conv = ex if kind == "ceiling" else (ex, winners["_normal_convention"])
                            sign = CONVENTIONS[conv]
                            base = normal_expert(name)
                            # The ceiling scores all four conventions from ONE expert pass: the
                            # raw prediction is cached per frame (keyed by the frame's buffer,
                            # which is stable for the life of this npz) and only re-signed.
                            def fn(x, base=base, sign=sign, name=name):
                                ck = (name, x.__array_interface__["data"][0])
                                if ck not in raw_cache:
                                    raw_cache[ck] = base(x)
                                return raw_cache[ck] * sign
                        else:
                            base = seg_expert(ex)
                            fn = lambda x, base=base: base(x, present)
                        pred = fn(src)
                        flr = None
                        if kind == "cascade":
                            ck = (row, rep, v)
                            if ck not in anch_cache:
                                anch_cache[ck] = fn(anchor_frame)
                            flr = anch_cache[ck]
                    prep = {"depth": prep_depth, "normal": prep_normal,
                            "rgb": prep_rgb, "seg": prep_seg}[mod]
                    pp, ok = prep(pred, gt)
                    val, cov = reduce(mod, pp, ok, None)
                    if val is None:
                        continue
                    a = acc[row]
                    a["value"].append(val); a["cov"].append(cov); a["k"].append(k)
                    fp = fok = None
                    if flr is not None:
                        fp, fok = prep(flr, gt)
                        a["floor"].append(reduce(mod, fp, fok, None)[0])
                    if moved is not None:
                        dv, _ = reduce(mod, pp, ok, moved)
                        if dv is not None:
                            a["dyn"].append(dv); a["dyn_frac"].append(float(moved.mean()))
                            if fp is not None:
                                a["dyn_floor"].append(reduce(mod, fp, fok, moved)[0])
                except Exception as exc:
                    fail[row] = f"{type(exc).__name__}: {exc}"
                    nfail[row] += 1
                    if "out of memory" in str(exc).lower():
                        import torch
                        torch.cuda.empty_cache()

    gaps = [float(exec_gap[i]) for i in scored_idx]
    out = {}
    for row, mod, kind, ex in todo:
        a = acc[row]
        if row in fail and not a["value"]:
            skipped[row] = fail[row]
            continue
        if not a["value"]:
            continue
        mean = lambda xs: round(float(np.mean([x for x in xs if x is not None])), 5) \
            if any(x is not None for x in xs) else None
        bins = {}
        for lo, hi in ((1, 10), (10, 20), (20, 41)):
            vs = [x for x, kk in zip(a["value"], a["k"]) if lo <= kk < hi]
            gs = [float(exec_gap[i]) for i in scored_idx if lo <= int(d["k"][i]) < hi]
            if vs:
                bins[f"k{lo}_{hi}"] = {"value": round(float(np.mean(vs)), 5), "n": len(vs),
                                       "exec_gap_m_median": round(float(np.median(gs)), 5) if gs else None}
        out[row] = {"row": row, "modality": mod, "route": kind, "expert": ex if kind != "ceiling" or mod != "normal" else list(ex),
                    "metric": METRIC[mod], "anchor": anchor, "n_frames": len(a["value"]),
                    "value": mean(a["value"]), "copy_anchor": mean(a["floor"]),
                    "dyn_value": mean(a["dyn"]), "dyn_copy_anchor": mean(a["dyn_floor"]),
                    "dyn_frac": mean(a["dyn_frac"]), "n_dyn_frames": len(a["dyn"]),
                    "by_horizon": bins, "decode_coverage": mean(a["cov"]),
                    "exec_gap_m_median": round(float(np.median(gaps)), 5) if gaps else None,
                    "exec_gap_m_p90": round(float(np.percentile(gaps, 90)), 5) if gaps else None,
                    "seg_unknown_frac": float(d["seg_unknown_frac"]) if "seg_unknown_frac" in has else None,
                    # A row where SOME frames failed (OOM, a missing package mid-run) is NOT a
                    # result on the same frames as its neighbours. Recorded so it can never be
                    # read as complete; the Table-2 view marks it with '!'.
                    "n_failed_frames": nfail[row],
                    "failure": fail.get(row) if nfail[row] else None}
    if skipped:
        out["_skipped"] = skipped
    return out


def parse_tag(tag):
    """percep_<arm>_<anchor>_<task...> -> (arm, anchor, task)."""
    p = tag.split("_")
    return p[1], p[2], "_".join(p[3:])


def pick_winners(ceil_agg):
    """Fixed-in-advance rule: per modality, the candidate with the best ceiling score on the
    DYNAMIC region (the region Table 2 reports), falling back to full frame. For normals the
    convention is chosen jointly with the expert, both on real frames only."""
    best = {}
    for key, v in ceil_agg.items():
        mod = v["modality"]
        score = v["dyn_value"] if v["dyn_value"] is not None else v["value"]
        if score is None:
            continue
        better = (score < best[mod][0]) if (mod in best and LOWER_IS_BETTER[mod]) else \
                 (score > best[mod][0]) if mod in best else True
        if better:
            best[mod] = (score, v["expert"], key)
    winners = {}
    for mod, (score, ex, key) in best.items():
        if mod == "normal":
            winners["normal"], winners["_normal_convention"] = ex[0], ex[1]
        else:
            winners[mod] = ex
        winners[f"_{mod}_ceiling_score"] = score
    return winners


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dump_root", default=os.path.join(REPO, "reports", "percep_simreplay", "percep_dump"))
    ap.add_argument("--out", default=None, help="default: <dump_root>/../scores_<rows>.json")
    ap.add_argument("--rows", default="direct",
                    help="comma list of direct,cascade,ceiling. Run `ceiling` BEFORE `cascade`.")
    ap.add_argument("--tags", default="*", help="fnmatch filter on dump tags, e.g. 'percep_arm7_*'")
    ap.add_argument("--ceiling_tags", default="*_video_*",
                    help="which dumps the ceiling sweep reads. Video-anchored dumps by default: "
                         "every machine has them (the future-RGB row), and any dump with "
                         "sim_rgb shows the same scenes, so reading all of them only repeats work.")
    ap.add_argument("--ceiling_stride", type=int, default=4,
                    help="score every Nth executed step on the ceiling row (experts are slow and "
                         "the ceiling does not depend on the model).")
    ap.add_argument("--ceiling_json", default=None,
                    help="winners file written by --rows ceiling; read by --rows cascade. "
                         "Default: <dump_root>/../ceiling.json")
    ap.add_argument("--experts", default=None,
                    help="restrict ceiling candidates, e.g. 'depth=da2metric+depthpro,normal=marigold'")
    ap.add_argument("--cascade", default=None,
                    help="OVERRIDE the ceiling winners for the cascade, e.g. "
                         "'depth=depthpro,normal=marigold:flip_yz,seg=clipseg'. For appendix "
                         "sweeps only -- the main table uses the ceiling winners.")
    ap.add_argument("--gpu", type=int, default=None)
    a = ap.parse_args()
    if a.gpu is not None:
        os.environ["SPECIALIST_GPU"] = str(a.gpu)
    rows = [r.strip() for r in a.rows.split(",") if r.strip()]
    base = os.path.dirname(os.path.abspath(a.dump_root))
    out_path = a.out or os.path.join(base, f"scores_{'_'.join(rows)}.json")
    ceil_path = a.ceiling_json or os.path.join(base, "ceiling.json")

    if a.experts:
        for part in a.experts.split(","):
            mod, names = part.split("=")
            keep = set(names.split("+"))
            for n in list(EXPERTS[mod]):
                if n not in keep:
                    EXPERTS[mod].pop(n)

    # ceiling AND cascade in one call: the ceiling must finish (and write ceiling.json) before
    # the cascade may read its winners, so run it as its own pass first. Two passes rather than
    # one interleaved loop keeps the rule visible -- the cascade cannot see anything but the
    # file the ceiling pass wrote.
    if "ceiling" in rows and "cascade" in rows and not a.cascade:
        import subprocess
        argv = [x for x in sys.argv[1:] if not x.startswith("--rows=")]
        if "--rows" in argv:
            i = argv.index("--rows")
            del argv[i:i + 2]
        argv += ["--rows", "ceiling"]
        print("== pass 1/2: ceiling (writes the expert winners) ==", flush=True)
        subprocess.run([sys.executable, os.path.abspath(__file__)] + argv
                       + ["--ceiling_json", ceil_path], check=True)
        rows = [r for r in rows if r != "ceiling"]
        out_path = a.out or os.path.join(base, f"scores_{'_'.join(rows)}.json")
        print(f"== pass 2/2: {','.join(rows)} ==", flush=True)

    winners = None
    if "cascade" in rows:
        if a.cascade:
            winners = {}
            for part in a.cascade.split(","):
                mod, spec = part.split("=")
                if mod == "normal":
                    name, conv = (spec.split(":") + ["identity"])[:2]
                    winners["normal"], winners["_normal_convention"] = name, conv
                else:
                    winners[mod] = spec
        elif os.path.exists(ceil_path):
            winners = json.load(open(ceil_path))["winners"]
        else:
            sys.exit(f"--rows cascade needs {ceil_path}: run `--rows ceiling` first. The expert "
                     f"is chosen on the ceiling row, never on the cascade row (PERCEP_SIMREPLAY.md 6).")
        print(f"cascade experts: { {k: v for k, v in winners.items() if not k.startswith('_') or k == '_normal_convention'} }")

    per_trial = defaultdict(list)
    skipped_all = defaultdict(set)
    for tagdir in sorted(glob.glob(os.path.join(a.dump_root, "*"))):
        tag = os.path.basename(tagdir)
        if not fnmatch.fnmatch(tag, a.tags):
            continue
        arm, anchor, task = parse_tag(tag)
        tag_rows = [r for r in rows if r != "ceiling" or fnmatch.fnmatch(tag, a.ceiling_tags)]
        if not tag_rows:
            continue
        for f in sorted(glob.glob(os.path.join(tagdir, "*.npz"))):
            try:
                res = score_one(f, anchor, tag_rows, winners, a.ceiling_stride)
            except Exception as exc:
                print(f"  {tag}/{os.path.basename(f)}: ERROR {type(exc).__name__}: {exc}", flush=True)
                continue
            for why_row, why in res.pop("_skipped", {}).items():
                skipped_all[(tag, why_row)].add(why)
            for row, r in res.items():
                r.update(trial_file=os.path.basename(f), arm=arm, task=task, tag=tag)
                per_trial[f"{tag}|{row}"].append(r)
                print(f"  {tag}/{r['trial_file']:28s} {row:34s} {r['metric']}={r['value']} "
                      f"floor={r['copy_anchor']} dyn={r['dyn_value']}/{r['dyn_copy_anchor']} "
                      f"n={r['n_frames']}" + (f"  !! {r['n_failed_frames']} FRAMES FAILED" if r.get("n_failed_frames") else ""),
                      flush=True)

    agg = {}
    for key, rs in per_trial.items():
        m = lambda f: (round(float(np.mean([r[f] for r in rs if r.get(f) is not None])), 5)
                       if any(r.get(f) is not None for r in rs) else None)
        r0 = rs[0]
        agg[key] = {"arm": r0["arm"], "task": r0["task"], "row": r0["row"],
                    "modality": r0["modality"], "route": r0["route"], "expert": r0["expert"],
                    "metric": r0["metric"], "n_trials": len(rs),
                    "value": m("value"), "copy_anchor": m("copy_anchor"),
                    "dyn_value": m("dyn_value"), "dyn_copy_anchor": m("dyn_copy_anchor"),
                    "dyn_frac": m("dyn_frac"), "decode_coverage": m("decode_coverage"),
                    "exec_gap_m_median": m("exec_gap_m_median"),
                    "n_frames": int(sum(r["n_frames"] for r in rs)),
                    "n_failed_frames": int(sum(r.get("n_failed_frames", 0) for r in rs))}

    result = {"rows": rows, "per_trial": per_trial, "aggregate": agg,
              "skipped": {f"{t}|{r}": sorted(w) for (t, r), w in skipped_all.items()}}
    if "cascade" in rows:
        result["cascade_experts"] = winners
    if "ceiling" in rows:
        # Pool the ceiling over every dump and task it was run on: it is a property of the
        # expert on this domain, not of any arm.
        pooled = defaultdict(list)
        for key, v in agg.items():
            if v["route"] == "ceiling":
                pooled[v["row"]].append(v)
        ceil_agg = {}
        for row, vs in pooled.items():
            pick = lambda f: (round(float(np.mean([x[f] for x in vs if x[f] is not None])), 5)
                              if any(x[f] is not None for x in vs) else None)
            ceil_agg[row] = {"modality": vs[0]["modality"], "expert": vs[0]["expert"],
                             "value": pick("value"), "dyn_value": pick("dyn_value"),
                             "n_tasks": len(vs)}
        winners_c = pick_winners(ceil_agg)
        result["ceiling_pooled"] = ceil_agg
        result["winners"] = winners_c
        json.dump({"winners": winners_c, "ceiling_pooled": ceil_agg,
                   "rule": "per modality, best pooled ceiling score on the dynamic region "
                           "(full frame if no dynamic pixels); normal convention chosen jointly"},
                  open(ceil_path, "w"), indent=1)
        print(f"\nwrote {ceil_path}\nwinners: {winners_c}")
    json.dump(result, open(out_path, "w"), indent=1)
    print(f"wrote {out_path}")
    partial = {k: v for k, v in agg.items() if v["n_failed_frames"]}
    if partial:
        print("\n!! PARTIAL rows -- some frames FAILED and were not scored. Do not report these:")
        for k, v in sorted(partial.items()):
            print(f"  {k:60s} failed {v['n_failed_frames']} frames, scored {v['n_frames']}")
    if skipped_all:
        print("\nSKIPPED rows (dump could not support them):")
        for (t, r), w in sorted(skipped_all.items()):
            print(f"  {t:44s} {r:30s} {'; '.join(sorted(w))}")

    # ---- Table-2 view: dyn / dyn_floor per task, then the RLBench macro average -------------
    tasks = ["close_microwave", "toilet_seat_down", "close_box", "meat_on_grill"]
    lines = defaultdict(dict)
    # Ceiling rows: only the WINNERS appear in Table 2, one cell per task, averaged over
    # whichever arms' dumps were read (the ceiling does not depend on the arm).
    if "ceiling" in rows:
        win_rows = set()
        for mod in ("depth", "seg"):
            if mod in winners_c:
                win_rows.add(f"{mod}:ceiling:{winners_c[mod]}")
        if "normal" in winners_c:
            win_rows.add(f"normal:ceiling:{winners_c['normal']}:{winners_c['_normal_convention']}")
        per_task = defaultdict(list)
        for v in agg.values():
            if v["route"] == "ceiling" and v["row"] in win_rows:
                per_task[(v["modality"], v["row"], v["task"])].append(v)
        for (mod, row, task), vs in per_task.items():
            dvs = [x["dyn_value"] for x in vs if x["dyn_value"] is not None]
            val = float(np.mean(dvs)) if dvs else None
            lines[(mod, row, "ceil")][task] = ("--" if val is None else f"{val:.3f}", val, None,
                                               sum(x["n_trials"] for x in vs))
    for key, v in agg.items():
        if v["route"] == "ceiling":
            continue
        cell = "--" if v["dyn_value"] is None else (
            f"{v['dyn_value']:.3f}" + ("" if v["dyn_copy_anchor"] is None else f"/{v['dyn_copy_anchor']:.3f}"))
        if v["n_failed_frames"]:
            cell += "!"
        lines[(v["modality"], v["row"], v["arm"])][v["task"]] = (cell, v["dyn_value"], v["dyn_copy_anchor"], v["n_trials"])
    if lines:
        print("\nTable 2 view (dynamic region: prediction/copy-anchor floor; n=trials):")
        print(f"{'row':30s} {'arm':5s} " + " ".join(f"{t[:16]:>18s}" for t in tasks) + f" {'RL avg':>14s}")
        for (mod, row, arm), cells in sorted(lines.items()):
            txt = [f"{cells[t][0]}(n{cells[t][3]})" if t in cells else "--" for t in tasks]
            vals = [cells[t][1] for t in tasks if t in cells and cells[t][1] is not None]
            fls = [cells[t][2] for t in tasks if t in cells and cells[t][2] is not None]
            avg = "--" if len(vals) < len(tasks) else (
                f"{np.mean(vals):.3f}" + (f"/{np.mean(fls):.3f}" if len(fls) == len(tasks) else ""))
            print(f"{row:30s} {arm:5s} " + " ".join(f"{x:>18s}" for x in txt) + f" {avg:>14s}")


if __name__ == "__main__":
    main()
