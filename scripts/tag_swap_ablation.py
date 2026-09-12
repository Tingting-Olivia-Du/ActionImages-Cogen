"""Does the prompt TAG select the modality, or does the ANCHOR FRAME?

    python scripts/tag_swap_ablation.py --ckpt .../step9000.ckpt --tag arm6_9000 --gpu 5

THE QUESTION THIS EXISTS TO SETTLE. The research claim of this arm is "one checkpoint, switch
the prompt tag, get a different modality". `modality_mode_grid`'s f0f0 probe already showed the
model collapses when the anchor is REMOVED (depth absrel 0.087 -> 0.816, seg mIoU 0.386 -> 0.059,
normal cos 0.896 -> -0.291), which says the anchor is sufficient. It does not say whether the tag
contributes anything at all, because in f0f0 both the anchor AND the usual conditioning are gone.

This isolates the tag. Everything the pipeline sees stays fixed at the TARGET modality -- the
segment layout (`template=`), the streams, the anchor frame -- and ONLY the text changes:

    template=video+depth, streams=depth, anchor=depth frame 0
      prompt `<video><depth> ...`      <- agreement
      prompt `<video><normal> ...`     <- tag lies, anchor tells the truth
      prompt `<video><scene-seg> ...`
      prompt `<video><action> ...`

then the output is decoded and scored AS THE TARGET. Two outcomes, both informative:

  * scores flat across tags  -> the tag is decorative; the anchor carries the modality, and the
    "switch the tag" framing does not survive contact with the evidence.
  * scores degrade off-diagonal -> the tag does real work, and f0f0's collapse is about losing
    the anchor's spatial grounding, not about the tag being ignored.

MODE. `iiii` only: it is what the closed-loop rollout uses and 0.90 of perception training draws.
Adding modes multiplies runtime without changing which of the two outcomes above obtains.
"""
import argparse, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import scripts.modality_mode_grid as G          # noqa: E402  (reuse, never re-implement)
from training.templates import prompt_prefix, parse_template   # noqa: E402
from training.dataset import RLBenchSelfgenDataset             # noqa: E402
from inference import build_pipeline                           # noqa: E402

TARGETS = ["depth", "segmentation", "normal"]
TAGS    = ["depth", "segmentation", "normal", "action"]
MODE    = "iiii"


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--gpu", default="auto")
    p.add_argument("--need_mib", type=int, default=30000)
    p.add_argument("--episode", default="open_drawer/variation0/episodes/episode0")
    p.add_argument("--data", default=str(REPO / "data" / "rlbench_selfgen_512_aug"))
    p.add_argument("--out", default=str(REPO / "reports" / "tag_swap"))
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--num_frames", type=int, default=41)
    p.add_argument("--frame_interval", type=int, default=3)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--segmentation_mode", default="scene_roles")
    p.add_argument("--targets", default="", help="comma list restricting TARGETS")
    p.add_argument("--tags", default="", help="comma list restricting TAGS")
    return p.parse_args()


def main():
    a = _args()
    if a.gpu == "auto":
        a.gpu = str(G.pick_gpu(a))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    print(f"[swap] using GPU {a.gpu}", flush=True)
    torch.cuda.init(); torch.zeros(1, device="cuda")
    G.reserve_vram(a.need_mib)

    ds = RLBenchSelfgenDataset(
        base_path=a.data, num_frames=a.num_frames, frame_interval=a.frame_interval,
        height=a.res, width=a.res, template_mix="video+action@1.0",
        prompt_tag_style="explicit", strict_getitem=True, variations="all",
        segmentation_mode=a.segmentation_mode)
    idx = next(i for i, e in enumerate(ds.episodes) if e["path"].endswith(a.episode))

    pipe = build_pipeline(SimpleNamespace(
        model_id="Wan-AI/Wan2.2-TI2V-5B", ckpt_path=a.ckpt, height=a.res, width=a.res,
        use_usp=False, cfg_parallel=False, dynamic_cache_schedule=False, torch_compile=False))
    device, dtype, T = pipe.device, torch.bfloat16, a.num_frames

    targets = [t.strip() for t in a.targets.split(",") if t.strip()] or TARGETS
    tags = [t.strip() for t in a.tags.split(",") if t.strip()] or TAGS
    results = {}
    for target in targets:
        template = f"video+{target}"
        G._seed_all(a.seed)
        s = ds.getitem(idx, force_template=template)
        if s["template"] != template:
            print(f"  SKIP {template}: dataset degraded to {s['template']!r}", flush=True)
            continue
        s["_view_dir"] = [s["view_dirs"][i] for i in s["view_indices"]][0]
        streams = {k: v.unsqueeze(0).to(device=device, dtype=dtype) for k, v in s["streams"].items()}
        # The dataset already wrote the honest prompt; strip it back to the bare instruction so
        # every tag below is prepended to IDENTICAL text. Splitting on "> " is exactly how
        # modality_mode_grid:271 recovers the instruction.
        instr = s["text"].split("> ", 1)[-1]

        for tag_mod in tags:
            prefix = prompt_prefix(parse_template(f"video+{tag_mod}"), style="explicit",
                                   segmentation_mode=a.segmentation_mode)
            # NO separator: prompt_prefix already ends in a space, exactly as the dataset
            # concatenates it (rlbench_selfgen.py:681). An extra space would make the string
            # `<video><depth>  open the drawer`, which train.py:287 scrubs ("  " -> " ") but the
            # inference path does NOT -- the same train/eval divergence that made <scene_seg>
            # silently condition on a different string for 2000 steps.
            prompt = f"{prefix}{instr}"
            if tag_mod == target:
                assert prompt == s["text"], (
                    f"the agreement cell must reproduce the dataset's own prompt byte for byte, "
                    f"or the whole row is measured against a different string:\n"
                    f"  built   {prompt!r}\n  dataset {s['text']!r}")
            key = f"{target}|{tag_mod}"
            t0 = time.time()
            G._seed_all(a.seed)
            frames = np.stack([np.asarray(f) for f in pipe(
                prompt=[prompt], negative_prompt="", template=template, streams=streams,
                conditioning_mode=MODE,
                camera=s["camera"].unsqueeze(0).to(device=device, dtype=dtype),
                action_7d=s["action_7d"].unsqueeze(0).to(device=device, dtype=dtype),
                extrinsics=s["extrinsics"].unsqueeze(0).to(device=device, dtype=dtype),
                intrinsics=s["intrinsics"].unsqueeze(0).to(device=device, dtype=dtype),
                height=a.res, width=a.res, num_frames=T, cfg_scale=a.cfg,
                num_inference_steps=a.steps, seed=a.seed, tiled=False,
                tile_size=(a.res // 16, a.res // 16), tile_stride=(a.res // 32, a.res // 32),
                enable_usp=False, cfg_parallel=False)])
            # Same span recomputation + assertion as the grid: the pipeline returns a flat array.
            plan = G.seg_plan(template, 2, MODE)
            spans, total = G.spans_for(plan, T)
            assert total == len(frames), f"{key}: spans {total} != returned {len(frames)}"
            # `plan` carries (modality, view, single_frame, fully_given); `spans` the matching
            # (begin, end) pixel offsets. They are zipped, not self-describing -- same pairing
            # modality_mode_grid:402 uses.
            (b, e) = next(sp for (m, v, _sf, _g), sp in zip(plan, spans)
                          if m == target and v == 0)
            pred = frames[b:e]
            sc = G.score(target, ds, s, pred, a.res)
            results[key] = {"target": target, "tag": tag_mod, "prompt": prompt,
                            "metric": sc["metric"], "value": sc["value"],
                            "seconds": round(time.time() - t0, 1)}
            print(f"  {key:28s} {sc['metric']:11s} {sc['value']}   ({results[key]['seconds']}s)",
                  flush=True)
            out = Path(a.out) / a.tag
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / f"{target}__tag_{tag_mod}.npy", pred[:: max(len(pred) // 8, 1)])
            with open(out / "metrics.json", "w") as f:
                json.dump({"tag": a.tag, "ckpt": a.ckpt, "episode": a.episode, "mode": MODE,
                           "seed": a.seed, "segmentation_mode": a.segmentation_mode,
                           "results": results}, f, indent=1)

    print("\n=== 对角线 = tag 与 anchor 一致；离对角 = tag 说谎 ===")
    for target in TARGETS:
        row = [results.get(f"{target}|{t}") for t in TAGS]
        if not any(row):
            continue
        m = next(r["metric"] for r in row if r)
        print(f"  {target:14s} [{m:10s}] " +
              "  ".join(f"{t}={r['value']}" if r else f"{t}=—" for t, r in zip(TAGS, row)))
    print(f"\nwrote {Path(a.out) / a.tag / 'metrics.json'}")


if __name__ == "__main__":
    main()
