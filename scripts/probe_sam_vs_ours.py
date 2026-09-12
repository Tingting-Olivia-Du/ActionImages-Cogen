"""Class-agnostic segmentation quality: SAM versus our scene-role decoder.

THE COMPARISON HAS TO IGNORE LABELS, because the two models do not produce the same kind of
output. SAM emits unnamed regions; we emit a role map (target / goal / distractor / fixture /
robot_arm / gripper / background). There is no vocabulary in common, so mIoU over roles is
undefined for SAM. What IS comparable is whether the BOUNDARIES are right.

METRIC. For every ground-truth role region in a frame, take the best-overlapping predicted mask
and record that IoU; average over regions. This asks "was this object carved out correctly by
somebody", independent of what it was called. Both models are scored the same way: for us, the
candidate set is the per-role masks our decoder produces; for SAM, the automatic mask proposals.

WHY THE MASK COUNT IS REPORTED ALONGSIDE. Best-overlap IoU rewards over-segmentation -- a model
emitting 200 proposals will cover every region by luck. SAM emits many; our decoder emits at most
one region per role. The count is what keeps the reader honest about that asymmetry, and it is
why this is reported as a boundary-quality probe rather than as a head-to-head score.
"""
import json, os, sys, random
import numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
from training.dataset import RLBenchSelfgenDataset
from training.percep.seg_codec import scene_role_labels, ROLE_TO_LABEL

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
FRAMES = [0, 20, 40]          # 3 frames per episode: SAM's proposal generator is slow


def best_overlap(gt_regions, pred_masks):
    """mean over GT regions of the best IoU achieved by ANY predicted mask."""
    out = []
    for g in gt_regions:
        if g.sum() < 200:                    # ignore slivers; a few px of a role is not a region
            continue
        best = 0.0
        for p in pred_masks:
            inter = np.logical_and(g, p).sum()
            if inter == 0:
                continue
            best = max(best, inter / np.logical_or(g, p).sum())
        out.append(best)
    return out



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
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    if _already_done('sam', n_ep): return
    from transformers import pipeline
    gen = pipeline("mask-generation", model="facebook/sam-vit-huge", device=0,
                   points_per_batch=64)
    ds = RLBenchSelfgenDataset(base_path=DATA_TREE,
        num_frames=41, frame_interval=3, height=512, width=512,
        template_mix="video+segmentation@1.0", prompt_tag_style="explicit",
        strict_getitem=True, variations="all", segmentation_mode="scene_roles")
    from PIL import Image
    sam_scores, sam_counts, n_regions = [], [], []
    for ep in EPS[:n_ep]:
        want = os.path.normpath(os.path.join(DATA_TREE, ep))
        idx = next(i for i,e in enumerate(ds.episodes) if os.path.normpath(e["path"])==want)
        _seed_all(42)
        s = ds.getitem(idx, force_template="video+segmentation")
        vd = [s["view_dirs"][i] for i in s["view_indices"]][0]
        fi = s["frame_indices"]
        instr = s["text"].split("> ", 1)[-1]
        lut, present = ds._scene_role_lut(s["path"], instr)
        hm = ds._to_model_res(ds._load_mask(vd)[fi]).astype(np.uint16)
        rgb = ((s["streams"]["video"].permute(1,2,3,0).numpy()+1)*127.5).astype(np.uint8)
        ep_s = []
        for t in FRAMES:
            gt_lab = scene_role_labels(hm[t][None], lut)[0]
            regions = [gt_lab == ROLE_TO_LABEL[r] for r in present if r != "background"]
            out = gen(Image.fromarray(rgb[t]), points_per_crop=16)
            masks = [np.asarray(m) for m in out["masks"]]
            sc = best_overlap(regions, masks)
            if sc:
                ep_s += sc; sam_counts.append(len(masks))
                n_regions.append(len([r for r in regions if r.sum() >= 200]))
        if ep_s:
            sam_scores += ep_s
            print(f"  {ep.split('/')[0]:24s} best-overlap IoU={np.mean(ep_s):.4f} "
                  f"({len(ep_s)} regions, SAM 平均 {np.mean(sam_counts[-3:]):.0f} 个 mask)", flush=True)
    _save('sam', n_ep, {'model':'SAM ViT-H','best_overlap_iou':float(np.mean(sam_scores)),
                        'n_gt_regions_scored':len(sam_scores),
                        'sam_masks_per_frame':float(np.mean(sam_counts)),
                        'gt_regions_per_frame':float(np.mean(n_regions))})
    print(f"\nSAM (ViT-H, 自动 mask 生成), {len(sam_scores)} 个 GT 区域:")
    print(f"  class-agnostic best-overlap IoU = {np.mean(sam_scores):.4f}")
    print(f"  SAM 平均每帧产生 {np.mean(sam_counts):.0f} 个 mask,GT 平均 {np.mean(n_regions):.1f} 个区域")
    print(f"  ⚠️ best-overlap 会奖励过分割 —— SAM 的 mask 数是 GT 区域数的 "
          f"{np.mean(sam_counts)/max(np.mean(n_regions),1):.0f} 倍")

if __name__ == "__main__":
    main()
