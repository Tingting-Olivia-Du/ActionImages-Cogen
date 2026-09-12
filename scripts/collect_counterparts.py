#!/usr/bin/env python
"""Collect every counterpart result into one appendix table + the main table's best-of.

Filenames encode the split, so parse it rather than globbing loosely: a plain *_n40.json is
the held-out variation-1 run, *_seen_* is variation 0, *_task_* is the unseen-task tree.
Reading those as one pool is how two different splits looked like an irreproducible probe.
"""
import json, glob, os, re, sys
import numpy as np

EXT = "reports/external"
SPLIT = {"seen": "seen", "task": "unseen task", "var1": "unseen var.", "": "unseen var."}

def load_all():
    rows = []
    for f in sorted(glob.glob(f"{EXT}/*.json")):
        b = os.path.basename(f)[:-5]
        m = re.match(r"^(?P<name>.+?)(?:_(?P<split>seen|task|var1))?_n(?P<n>\d+)$", b)
        if not m: continue
        d = json.load(open(f))
        rows.append(dict(file=b, name=m["name"], split=SPLIT[m["split"] or ""],
                         n=int(m["n"]), d=d))
    # The sweep re-ran a few probes under an explicit _var1 tag that already existed
    # untagged. Identical inputs and seed, so the pair is a reproducibility check, not two
    # measurements -- keep the richer record (it carries model_class) and drop the other.
    best = {}
    for r in rows:
        k = (r["name"], r["split"], r["n"])
        if k not in best or len(r["d"]) > len(best[k]["d"]):
            best[k] = r
    return list(best.values())

# Model identity -> the display name and citation used in the appendix table. Naming a model
# in a results table without a reference is the kind of thing a reviewer flags immediately,
# and the probe filenames are internal shorthand rather than published names.
CITE = {
    "da2":            (r"Depth Anything V2-L \citep{yang2024depthanythingv2}",),
    "da2metric":      (r"Depth Anything V2-L (metric) \citep{yang2024depthanythingv2}",),
    "da3":            (r"Depth Anything 3-L \citep{lin2025depthanything3}",),
    "vggt":           (r"VGGT-1B \citep{wang2025vggt}",),
    "depthpro":       (r"Depth Pro \citep{bochkovskii2024depthpro}",),
    "lotusdepth":     (r"Lotus-G depth v2-0 \citep{he2025lotus}",),
    "marigolddepth":  (r"Marigold depth v1-1 \citep{ke2024marigold}",),
    "normal_lotus":   (r"Lotus-G normal v1-1 \citep{he2025lotus}",),
    "normal_marigold":(r"Marigold-Normals v1-1 \citep{ke2024marigold}",),
    "normal_da2":     (r"normals from Depth Anything V2 \citep{yang2024depthanythingv2}",),
    "sam":            (r"SAM ViT-H \citep{kirillov2023sam}",),
    "namedseg_clipseg":(r"CLIPSeg \citep{luddecke2022clipseg}",),
    "ours_classagnostic_fifi": ("U-CoGen (FIFI)",),
    "ours_classagnostic_iiii": ("U-CoGen (IIII)",),
    "segspec_classagnostic_iiii": ("Segmentation specialist",),
}


def disp(name):
    return CITE.get(name, (name.replace("_", chr(92) + "_"),))[0]


def val(r):
    d = r["d"]
    # VGGT's probe predates the shared key name and writes absrel_median_aligned; dropping it
    # silently removed a counterpart from the "best of" comparison.
    if "absrel_median_aligned" in d and "absrel_aligned" not in d:
        return "absrel_aligned", d["absrel_median_aligned"]
    for k in ("absrel_aligned", "mean_cos", "best_overlap_iou", "macro_miou"):
        if k in d: return k, d[k]
    return None, None

def main():
    rows = load_all()
    print(f"{'file':34s} {'split':13s} {'n':>3s} {'metric':18s} {'value':>8s}  extra")
    for r in sorted(rows, key=lambda r: (r["split"], r["name"])):
        k, v = val(r)
        if v is None: continue
        d = r["d"]
        extra = ""
        if k == "absrel_aligned":
            extra = (f"class={d.get('model_class','?')} free={d.get('free_parameters_per_frame','?')} "
                     f"raw={d.get('absrel_raw',float('nan')):.4f} "
                     f"medsc={d.get('absrel_median_scale',float('nan')):.4f} "
                     f"dispaff={d.get('absrel_disparity_affine',float('nan')):.4f}")
        elif k == "mean_cos":
            extra = f"conv={d.get('convention_chosen','?')}"
        elif k == "best_overlap_iou":
            mpf = d.get('sam_masks_per_frame') or d.get('ours_masks_per_frame')
            extra = f"masks/frame={mpf:.1f}" if mpf else ""
        print(f"{r['file']:34s} {r['split']:13s} {r['n']:3d} {k:18s} {v:8.4f}  {extra}")

    print("\n=== BEST COUNTERPART per (metric, split) ===")
    groups = {}
    for r in rows:
        k, v = val(r)
        if v is None or r["name"].startswith("ours"): continue
        better_low = k in ("absrel_aligned",)
        groups.setdefault((k, r["split"]), []).append((v, r["name"], r["n"], better_low))
    for (k, sp), lst in sorted(groups.items()):
        low = lst[0][3]
        best = min(lst, key=lambda t: t[0]) if low else max(lst, key=lambda t: t[0])
        others = ", ".join(f"{n}={v:.4f}" for v, n, _, _ in sorted(lst, key=lambda t: t[0], reverse=not low))
        print(f"  {k:18s} {sp:13s} -> {best[1]} = {best[0]:.4f}   (all: {others})")

if __name__ == "__main__" and "--latex" not in sys.argv:
    main()


def rgb_rows():
    """RGB quality for every model that generates video, all three metrics.

    LPIPS is what the main table reports. PSNR and SSIM are computed by the same pass and are
    included here because reporting only the perceptual metric invites the question of whether
    it was the flattering one. It was not -- the three metrics rank the models identically.
    """
    import collections
    R = "reports/heldout_batch_eval"
    want = [("init_125750", "ActionImages (released)", "held-out"),
            ("actionspec_4000_matched", "Action specialist", "held-out"),
            ("arm6_10000", "U-CoGen (IIII)", "held-out"),
            ("init_125750_unseen_task", "ActionImages (released)", "unseen task"),
            ("actionspec_4000_unseen_task", "Action specialist", "unseen task"),
            ("arm6_10000_unseen_task", "U-CoGen (IIII)", "unseen task")]
    out = []
    for tag, lbl, tier in want:
        f = f"{R}/{tag}/per_episode.json"
        if not os.path.exists(f): continue
        r = json.load(open(f))["results"]
        key = "/variation1/" if tier == "held-out" else None
        acc = collections.defaultdict(list)
        for k, v in r.items():
            if k.rsplit("|", 1)[1] != "video__via_action": continue
            if key and key not in k: continue
            for m in ("value", "psnr_db", "ssim"):
                if v.get(m) is not None: acc[m].append(v[m])
        if acc:
            out.append((lbl, tier, float(np.mean(acc["value"])),
                        float(np.mean(acc["psnr_db"])), float(np.mean(acc["ssim"])),
                        len(acc["value"])))
    return out


def latex_appendix():
    """Full counterpart table for the appendix: every model, every tier, every alignment."""
    rows = load_all()
    depth = [r for r in rows if val(r)[0] == "absrel_aligned"]
    norm  = [r for r in rows if val(r)[0] == "mean_cos"]
    seg   = [r for r in rows if val(r)[0] in ("best_overlap_iou", "macro_miou")]
    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"\caption{Every counterpart run, on every tier. The main table reports the best "
             r"per row; this is the pool it was chosen from. \textbf{Free} is the number of "
             r"parameters fitted against the ground truth per frame before scoring: metric "
             r"models are scored as they claim, with none. Alignment is decided from the sign "
             r"of each model's correlation with ground-truth depth rather than assumed, and "
             r"all three alignments are listed so the choice is auditable. All models are "
             r"trained on real photographs while these frames are synthetic renders.}")
    L.append(r"\label{app:external}")
    L.append(r"\centering\small")
    L.append(r"\begin{tabular}{llcrrrr}")
    L.append(r"\toprule")
    L.append(r"Model & Split & Free & Reported & raw & med.-scale & disp.-affine \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{7}{l}{\emph{Depth} (AbsRel $\downarrow$)} \\")
    for r in sorted(depth, key=lambda r: (r["split"], val(r)[1])):
        d = r["d"]; v = val(r)[1]
        g = lambda k: (f"{d[k]:.4f}" if isinstance(d.get(k), (int, float)) else "---")
        L.append(f"{disp(r['name'])} & {r['split']} & "
                 f"{d.get('free_parameters_per_frame','---')} & \\textbf{{{v:.4f}}} & "
                 f"{g('absrel_raw')} & {g('absrel_median_scale')} & {g('absrel_disparity_affine')} \\\\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{7}{l}{\emph{Surface normal} (mean cosine $\uparrow$)} \\")
    for r in sorted(norm, key=lambda r: (r["split"], -val(r)[1])):
        d = r["d"]
        L.append(f"{disp(r['name'])} & {r['split']} & --- & "
                 f"\\textbf{{{val(r)[1]:.4f}}} & \\multicolumn{{3}}{{l}}{{convention: "
                 f"{d.get('convention_chosen','---')}}} \\\\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{7}{l}{\emph{Segmentation} (bo-IoU / mIoU $\uparrow$)} \\")
    for r in sorted(seg, key=lambda r: (r["split"], -val(r)[1])):
        d = r["d"]
        mpf = d.get("sam_masks_per_frame") or d.get("ours_masks_per_frame")
        note = f"{mpf:.0f} masks/frame" if mpf else ""
        L.append(f"{disp(r['name'])} & {r['split']} & --- & "
                 f"\\textbf{{{val(r)[1]:.4f}}} & \\multicolumn{{3}}{{l}}{{{note}}} \\\\")
    rgb = rgb_rows()
    if rgb:
        L.append(r"\midrule")
        L.append(r"\multicolumn{7}{l}{\emph{Future RGB} --- LPIPS $\downarrow$ / PSNR (dB) $\uparrow$ / SSIM $\uparrow$} \\")
        for lbl, tier, lp, ps, ss, n in rgb:
            L.append(f"{lbl} & {tier} & --- & \\textbf{{{lp:.4f}}} & {ps:.2f} & {ss:.4f} & $n{{=}}{n}$ \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


if __name__ == "__main__" and "--latex" in sys.argv:
    print(latex_appendix())
