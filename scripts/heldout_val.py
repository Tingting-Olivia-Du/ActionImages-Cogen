"""Held-out validation loss on the variations training never sees.

    python scripts/heldout_val.py --ckpt .../step6500.ckpt --gpu 5
    python scripts/heldout_val.py --all-checkpoints --gpu 5      # every ckpt still on disk

WHY. arm6 trains on variation0 only and holds out variation1/2, but nothing ever evaluates
them: there is no eval_strategy, no eval_dataset, and the only in-training signal is the training
loss. At step 6500 that loss has stopped improving and is drifting UP (+0.0019/1000 steps over
4000-6500) while the model is on its 16.5th pass over 788 episodes, heading for 25.4 -- exactly
the regime train_arm.sh's own header records arm0 collapsing in (r_peak 98.5 -> 47.0 -> 3.7 while
"the loss curve looked fine"). Without a held-out number there is no way to tell overfitting from
noise.

TWO DESIGN POINTS, both load-bearing:

1. IT REUSES ActionImagesModel.forward. A re-implementation of the loss would be a second copy of
   the scrub, the Pluecker embedding, the action rendering, plan_segments and assemble -- and the
   whole history of this project is silent divergence between two copies of one thing. The number
   is comparable to the training loss because it is produced by the same code.

2. THE SAMPLE SET IS FROZEN ACROSS CHECKPOINTS. Training loss has a per-step std of ~0.017 while
   the trend being chased is ~0.002/1000 steps -- most of that spread is the random diffusion
   timestep, not the model. So every checkpoint is scored on the SAME (episode, template, plan,
   timestep, noise) tuples: `torch.manual_seed`/`random.seed` are reset per sample from a fixed
   base, so checkpoint-to-checkpoint differences are model differences and nothing else. An
   unfrozen validation set would need thousands of samples to see the same signal.
"""
import argparse, atexit, glob, json, os, random, shutil, sys, time
from types import SimpleNamespace

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="")
    p.add_argument("--all-checkpoints", action="store_true",
                   help="score every checkpoint under --out-dir's arm, oldest first")
    p.add_argument("--arm-dir", default="outputs/arm6__seed42_fi3_512_aug_sr")
    p.add_argument("--data", default="data/rlbench_selfgen_512_aug")
    p.add_argument("--variations", default="!0", help="'!0' = everything except the train split")
    p.add_argument("--template-mix",
                   default="video+action@0.4,video+depth@0.2,video+segmentation@0.2,video+normal@0.2")
    p.add_argument("--segmentation-mode", default="scene_roles")
    p.add_argument("--perception-mask-mix", default="M2")
    p.add_argument("--n", type=int, default=160, help="frozen sample count")
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--num-frames", type=int, default=41)
    p.add_argument("--frame-interval", type=int, default=3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--gpu", default="auto")
    p.add_argument("--need-mib", type=int, default=30000)
    p.add_argument("--out", default="reports/heldout_val.json")
    p.add_argument("--pin-dir", default="outputs/.heldout_pin",
                   help="hardlink staging dir; must share a filesystem with --arm-dir")
    return p.parse_args()


def free_mib():
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return [(int(a), int(c) - int(b))
            for a, b, c in (l.split(",") for l in out.strip().splitlines())]


def pin_checkpoints(ckpts, pin_dir):
    """Hardlink each .ckpt so the trainer's SAVE_TOP_K rotation cannot delete it mid-run.

    The first attempt at this script died 24 minutes in: it enumerated [4500..6500] at launch,
    spent 22 min loading the DiT/T5/VAE, and by then `checkpoint-6500` had been written and the
    rotation had unlinked `step4500.ckpt`. A hardlink keeps the inode alive at zero disk cost --
    the trainer's `rm` just drops a refcount -- and it is only safe because the pin dir sits on
    the same filesystem as outputs/. If os.link is unavailable (different fs), fall back to
    checking existence right before each load and skipping what has vanished.
    """
    os.makedirs(pin_dir, exist_ok=True)
    atexit.register(lambda: shutil.rmtree(pin_dir, ignore_errors=True))
    out = []
    for step, path in ckpts:
        link = os.path.join(pin_dir, f"step{step}.ckpt")
        try:
            if not os.path.exists(link):
                os.link(path, link)
            out.append((step, link))
        except OSError as e:
            print(f"[val] pin failed for {path} ({e}); will re-check at load time", flush=True)
            out.append((step, path))
    return out


def stratified_picks(ds, n, rng):
    """Sample n episodes without replacement, proportionally per task.

    The first version drew `rng.randrange(len(ds))` n times WITH replacement: 160 draws from the
    240 held-out episodes touched only 111 of them (46%), scoring 8 episodes 3+ times while 129
    were never seen. Paired checkpoint comparison still worked -- the set is frozen either way --
    but the mean represented under half the split. Per-task proportions also drifted 3x
    (close_jar 13.1% vs meat_off_grill 4.4%) because the tasks hold 10 or 20 episodes each.
    """
    by_task = {}
    for i, e in enumerate(ds.episodes):
        by_task.setdefault(e["path"].split(os.sep)[-4], []).append(i)
    if n >= len(ds):
        return sorted(i for v in by_task.values() for i in v)
    # Largest-remainder apportionment so the quotas sum to exactly n.
    quota, rem = {}, []
    for t, idxs in sorted(by_task.items()):
        exact = n * len(idxs) / len(ds)
        quota[t] = min(int(exact), len(idxs))
        rem.append((exact - int(exact), t))
    for _, t in sorted(rem, reverse=True):
        if sum(quota.values()) >= n:
            break
        if quota[t] < len(by_task[t]):
            quota[t] += 1
    picks = []
    for t, idxs in sorted(by_task.items()):
        picks.extend(rng.sample(idxs, quota[t]))
    rng.shuffle(picks)
    return picks


def main():
    a = _args()
    if a.gpu == "auto":
        cand = sorted(free_mib(), key=lambda r: -r[1])
        if not cand or cand[0][1] < a.need_mib:
            sys.exit(f"[val] no GPU with {a.need_mib} MiB free: {cand}")
        a.gpu = str(cand[0][0])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    print(f"[val] GPU {a.gpu}", flush=True)

    import torch as T
    # Claim the card before the slow import/load, same reason as modality_mode_grid.
    T.cuda.init(); T.zeros(1, device="cuda")
    blk = T.empty(int(a.need_mib * 0.9) * 1024 * 1024 // 2, dtype=T.float16, device="cuda"); del blk

    from train import ActionImagesModel
    from training.dataset import RLBenchSelfgenDataset

    ckpts = []
    if a.all_checkpoints:
        for d in sorted(glob.glob(os.path.join(a.arm_dir, "checkpoint-*")),
                        key=lambda p: int(p.rsplit("-", 1)[1])):
            s = int(d.rsplit("-", 1)[1])
            f = os.path.join(d, f"step{s}.ckpt")
            if os.path.exists(f):
                ckpts.append((s, f))
    else:
        s = int(os.path.basename(a.ckpt).replace("step", "").replace(".ckpt", ""))
        ckpts = [(s, a.ckpt)]
    ckpts = pin_checkpoints(ckpts, a.pin_dir)
    print(f"[val] {len(ckpts)} checkpoint(s): {[s for s, _ in ckpts]}", flush=True)

    ds = RLBenchSelfgenDataset(
        base_path=a.data, num_frames=a.num_frames, frame_interval=a.frame_interval,
        height=a.res, width=a.res, template_mix=a.template_mix,
        prompt_tag_style="explicit", variations=a.variations,
        segmentation_mode=a.segmentation_mode, strict_getitem=True,
    )
    print(f"[val] held-out split '{a.variations}': {len(ds)} episodes", flush=True)

    # ---- freeze the sample set ONCE, before any model exists ----------------------------
    rng = random.Random(a.seed)
    picks = stratified_picks(ds, a.n, rng)
    samples = []
    for i, idx in enumerate(picks):
        random.seed(a.seed + i)                    # window, views, template, paraphrase
        torch.manual_seed(a.seed + i)
        s = ds.getitem(idx)
        samples.append(s)
    from collections import Counter
    print(f"[val] frozen set: {len(set(picks))}/{len(ds)} episodes, "
          f"{len(Counter(s['path'].split(os.sep)[-4] for s in samples))} tasks", flush=True)
    print(f"[val] templates: {dict(Counter(s['template'] for s in samples))}", flush=True)

    results = {}
    if os.path.exists(a.out):
        try:
            results = json.load(open(a.out)).get("results", {})
        except (OSError, ValueError):
            results = {}

    model = None
    for step, path in ckpts:
        t0 = time.time()
        if not os.path.exists(path):
            print(f"[val] step {step}: {path} is gone (rotation won the race); skipping",
                  flush=True)
            continue
        if model is None:
            model = ActionImagesModel(
                model_id="Wan-AI/Wan2.2-TI2V-5B", use_gradient_checkpointing=False,
                resume_ckpt_path=path, resolution=(a.res, a.res), full_param=True,
                perception_mask_mix=a.perception_mask_mix,
            ).to("cuda")
            # In training DeepSpeed's `bf16: true` does this cast; standalone nothing does, and
            # the failure is a dtype mismatch deep inside the DiT's time embedding. Matches what
            # inference.build_pipeline does for the same reason.
            model.pipe.to(dtype=torch.bfloat16)
            model.eval()
        else:
            # Reload weights only -- the VAE and text encoder are identical across checkpoints,
            # and reconstructing the pipeline per checkpoint would dominate the runtime.
            sd = torch.load(path, map_location="cpu")
            model.pipe.dit.load_state_dict(sd, strict=True)
            del sd
        per_template, losses = {}, []
        with torch.no_grad():
            for i, s in enumerate(samples):
                # Same tuple for every checkpoint: the plan draw inside forward and the
                # timestep/noise draw all come from these two seeds.
                random.seed(a.seed * 7919 + i)
                torch.manual_seed(a.seed * 7919 + i)
                batch = {
                    "text": [s["text"]], "video": s["video"].unsqueeze(0),
                    "camera": s["camera"].unsqueeze(0), "action_7d": s["action_7d"].unsqueeze(0),
                    "extrinsics": s["extrinsics"].unsqueeze(0),
                    "intrinsics": s["intrinsics"].unsqueeze(0),
                    "template": [s["template"]], "path": [s["path"]],
                    "streams": {k: v.unsqueeze(0) for k, v in s["streams"].items()},
                }
                out = model(**batch)
                L = float(out["loss"])
                losses.append(L)
                per_template.setdefault(s["template"], []).append(L)
                if (i + 1) % 40 == 0:
                    print(f"    [{step}] {i+1}/{len(samples)}  running mean {np.mean(losses):.5f}",
                          flush=True)
        rec = {"n": len(losses), "mean": round(float(np.mean(losses)), 6),
               "median": round(float(np.median(losses)), 6),
               "sem": round(float(np.std(losses) / max(len(losses) ** 0.5, 1)), 6),
               "per_template": {k: {"n": len(v), "mean": round(float(np.mean(v)), 6)}
                                for k, v in sorted(per_template.items())},
               "seconds": round(time.time() - t0, 1)}
        results[str(step)] = rec
        print(f"  step {step:5d}  held-out {rec['mean']:.5f} +- {rec['sem']:.5f}  "
              + "  ".join(f"{k}:{v['mean']:.4f}" for k, v in rec["per_template"].items())
              + f"   ({rec['seconds']}s)", flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"arm_dir": a.arm_dir, "variations": a.variations, "n": a.n,
                       "seed": a.seed, "res": a.res, "frame_interval": a.frame_interval,
                       "template_mix": a.template_mix,
                       "perception_mask_mix": a.perception_mask_mix, "results": results}, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
