"""Extract exact action-query -> visual-anchor attention rows from selected DiT blocks.

This is an illustration/probe, not standalone causal evidence.  Pair its target-enrichment score
with ``mechanism_causal_roles.py`` and ask whether attention enrichment predicts deletion
sensitivity across episodes.

The implementation never materializes the full SxS attention matrix.  It samples action-query
rows, reconstructs their exact softmax against every key from post-RoPE Q/K, and stores only the
mean key distribution.  At 512px this is tens of MB rather than many GB per head/layer.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from inference import build_pipeline  # noqa: E402
from scripts.mechanism_causal_roles import (  # noqa: E402
    _scene_groups, _seed_all, decode_action,
)
from training.dataset import RLBenchSelfgenDataset  # noqa: E402
from training.templates import parse_template, prompt_prefix  # noqa: E402
from training.utils import project_actions_7d_to_5d_torch_batch  # noqa: E402


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--task", default="close_microwave")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--variation", type=int, default=0)
    p.add_argument("--data", default=str(REPO / "data" / "rlbench_unseen_tasks_512_clean"))
    p.add_argument("--target-instances", default="microwave_door")
    p.add_argument("--goal-instances", default="")
    p.add_argument("--out", default=str(REPO / "reports" / "mechanism_attention"))
    p.add_argument("--blocks", default="0,7,15,22,29")
    p.add_argument("--denoise-steps", default="0,25,49",
                   help="zero-based sampler steps; must be less than --steps")
    p.add_argument("--query-samples", type=int, default=64)
    p.add_argument("--query-mode", choices=("trajectory", "random"), default="trajectory",
                   help="trajectory samples rows around GT-projected action points (same rows "
                        "for both models); random is a diagnostic over the mostly blank canvas")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--num-frames", type=int, default=41)
    p.add_argument("--frame-interval", type=int, default=3)
    return p.parse_args()


def _csv_int(value: str) -> List[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def _csv(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


class AttentionTransportRecorder:
    def __init__(self, pipe, blocks: Sequence[int], denoise_steps: Sequence[int],
                 num_inference_steps: int, cfg_scale: float, f: int, h: int, w: int,
                 action_frames: Sequence[int], query_samples: int, seed: int,
                 query_indices: Sequence[int] | None = None):
        self.pipe = pipe
        self.blocks = list(blocks)
        self.f, self.h, self.w = int(f), int(h), int(w)
        self.tokens_per_frame = self.h * self.w
        self.cfg_branches = 2 if cfg_scale != 1.0 else 1
        self.capture_calls = {self.cfg_branches * int(s): int(s) for s in denoise_steps}
        if any(s < 0 or s >= num_inference_steps for s in denoise_steps):
            raise ValueError(f"denoise steps {denoise_steps} outside [0,{num_inference_steps})")
        rng = np.random.default_rng(seed)
        if query_indices is None:
            candidates = np.concatenate([
                np.arange(t * self.tokens_per_frame, (t + 1) * self.tokens_per_frame)
                for t in action_frames
            ])
        else:
            candidates = np.unique(np.asarray(query_indices, dtype=np.int64))
        n = min(int(query_samples), len(candidates))
        self.query_indices = np.sort(rng.choice(candidates, n, replace=False))
        self.records: List[Dict[str, object]] = []
        self._original = {}
        self._calls = {b: 0 for b in self.blocks}

    def install(self):
        expected = self.f * self.h * self.w
        for block_idx in self.blocks:
            module = self.pipe.dit.blocks[block_idx].self_attn.attn
            original = module.forward
            self._original[block_idx] = original
            recorder = self

            def wrapped(this, q, k, v, *, _idx=block_idx, _orig=original):
                call = recorder._calls[_idx]
                recorder._calls[_idx] += 1
                if call in recorder.capture_calls:
                    if q.shape[1] != expected:
                        raise RuntimeError(
                            f"attention sequence {q.shape[1]} != expected f*h*w={expected}; "
                            "template/latent grid mapping is stale")
                    recorder._capture(_idx, recorder.capture_calls[call], q, k)
                return _orig(q, k, v)

            module.forward = types.MethodType(wrapped, module)
        return self

    def remove(self):
        for block_idx, original in self._original.items():
            self.pipe.dit.blocks[block_idx].self_attn.attn.forward = original
        self._original.clear()

    @torch.no_grad()
    def _capture(self, block: int, denoise_step: int, q: torch.Tensor, k: torch.Tensor):
        heads = self.pipe.dit.blocks[block].self_attn.num_heads
        head_dim = q.shape[-1] // heads
        qh = q[:, torch.as_tensor(self.query_indices, device=q.device)].reshape(
            q.shape[0], len(self.query_indices), heads, head_dim)
        kh = k.reshape(k.shape[0], k.shape[1], heads, head_dim)
        # fp32 softmax is important here: attention probabilities, unlike the model output, are
        # the reported measurement and should not acquire bf16 underflow merely to save ~70 MB.
        score = torch.einsum("bqhd,bshd->bhqs", qh.float(), kh.float()) / math.sqrt(head_dim)
        key_mass = score.softmax(dim=-1).mean(dim=(0, 1, 2))
        key_mass = key_mass.reshape(self.f, self.h, self.w).cpu().numpy()
        self.records.append({"block": block, "denoise_step": denoise_step, "key_mass": key_mass})


def trajectory_query_indices(sample: Mapping, T_l: int, h: int, w: int,
                             image_size: int, radius: int = 1) -> np.ndarray:
    """Action-token rows near the GT-projected trajectory, shared exactly across models.

    A random query over an action image mostly selects black canvas, so its attention is not an
    interpretable statement about the generated pose.  This selector uses ground truth only to
    choose which rows to inspect; it is never provided to the denoiser and cannot improve its
    action.  The causal experiment remains free of this post-hoc selection.
    """
    T = int(sample["action_7d"].shape[0])
    action = sample["action_7d"].unsqueeze(0).repeat(1, 2, 1)
    action_5d = project_actions_7d_to_5d_torch_batch(
        action, sample["extrinsics"].unsqueeze(0), sample["intrinsics"].unsqueeze(0)
    )[0].reshape(2, T, 7)
    indices = []
    # One causal VAE frame at t=0, then one latent frame per four pixel frames.
    for view in range(2):
        action_segment_start = (2 * view + 1) * T_l
        for latent_t in range(1, T_l):  # frame zero is a clean action anchor, not a prediction
            pixel_t = min(4 * latent_t, T - 1)
            # Gripper centre and the two orientation endpoints all describe the predicted pose.
            for point in range(3):
                x_px = float(action_5d[view, pixel_t, 2 * point])
                y_px = float(action_5d[view, pixel_t, 2 * point + 1])
                if not (np.isfinite(x_px) and np.isfinite(y_px)):
                    continue
                gx = int(np.floor(x_px * w / image_size))
                gy = int(np.floor(y_px * h / image_size))
                if not (0 <= gx < w and 0 <= gy < h):
                    continue
                frame = action_segment_start + latent_t
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        yy, xx = gy + dy, gx + dx
                        if 0 <= yy < h and 0 <= xx < w:
                            indices.append(frame * h * w + yy * w + xx)
    if not indices:
        raise RuntimeError("no projected action points fell inside the token grid")
    return np.unique(np.asarray(indices, dtype=np.int64))


def _target_metrics(key_mass: np.ndarray, target_masks: np.ndarray, anchor_frames: Sequence[int]):
    out = []
    for view, (mask, frame) in enumerate(zip(target_masks, anchor_frames)):
        occupancy = F.adaptive_avg_pool2d(
            torch.from_numpy(mask.astype(np.float32))[None, None], key_mass.shape[-2:]
        )[0, 0].numpy()
        attn = key_mass[frame]
        area = float(occupancy.mean())
        attended = float((attn * occupancy).sum() / max(attn.sum(), 1e-20))
        out.append({
            "view": view, "anchor_frame": int(frame), "target_area_fraction": area,
            "attention_target_fraction": attended,
            "target_enrichment": attended / max(area, 1e-20),
            "anchor_attention_mass": float(attn.sum()),
        })
    return out


def _plot(records: Sequence[Mapping], target_masks: np.ndarray, anchor_frames: Sequence[int], path: Path):
    n = len(records)
    fig, axes = plt.subplots(n, 2, figsize=(7.2, 3.15 * n), squeeze=False)
    for row, rec in enumerate(records):
        maps = rec["key_mass"]
        for view, frame in enumerate(anchor_frames):
            ax = axes[row, view]
            heat = maps[frame]
            ax.imshow(heat, cmap="magma")
            small_mask = F.interpolate(
                torch.from_numpy(target_masks[view].astype(np.float32))[None, None],
                size=heat.shape, mode="nearest")[0, 0].numpy()
            ax.contour(small_mask, levels=[0.5], colors=["cyan"], linewidths=1.0)
            metric = rec["target_metrics"][view]
            ax.set_title(f"block {rec['block']}, step {rec['denoise_step']}, view {view}\n"
                         f"target enrichment={metric['target_enrichment']:.2f}x")
            ax.axis("off")
    fig.suptitle("Action-query attention to RGB anchor (cyan = manipulated object)")
    fig.tight_layout()
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)


def main():
    a = _args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit("Set CUDA_VISIBLE_DEVICES before Python starts.")
    blocks, denoise_steps = _csv_int(a.blocks), _csv_int(a.denoise_steps)
    template = "video+action"
    _seed_all(a.seed + a.episode)
    ds = RLBenchSelfgenDataset(
        base_path=a.data, num_frames=a.num_frames, frame_interval=a.frame_interval,
        height=a.res, width=a.res, template_mix=f"{template}@1", prompt_tag_style="explicit",
        strict_getitem=True, variations="all", segmentation_mode="scene_roles")
    wanted = os.path.normpath(os.path.join(
        a.data, a.task, f"variation{a.variation}", "episodes", f"episode{a.episode}"))
    idx = next((i for i, e in enumerate(ds.episodes) if os.path.normpath(e["path"]) == wanted), None)
    if idx is None:
        raise SystemExit(f"episode not found: {wanted}")
    _seed_all(a.seed + a.episode)
    sample = ds.getitem(idx, force_template=template)
    masks, mask_meta = _scene_groups(
        sample, a.res, _csv(a.target_instances), _csv(a.goal_instances))
    instruction = sample["text"].split("> ", 1)[-1]
    prompt = prompt_prefix(parse_template(template), style="explicit") + instruction

    pipe = build_pipeline(SimpleNamespace(
        model_id="Wan-AI/Wan2.2-TI2V-5B", ckpt_path=a.ckpt, height=a.res, width=a.res,
        use_usp=False, cfg_parallel=False, dynamic_cache_schedule=False, torch_compile=False))
    T_l = 1 + (a.num_frames - 1) // 4
    n_segments = 4
    f = n_segments * T_l
    h = (a.res // pipe.vae.upsampling_factor) // pipe.dit.patch_size[1]
    w = (a.res // pipe.vae.upsampling_factor) // pipe.dit.patch_size[2]
    # [video0 | action0 | video1 | action1].  Exclude each action segment's clean frame zero.
    action_frames = list(range(T_l + 1, 2 * T_l)) + list(range(3 * T_l + 1, 4 * T_l))
    anchor_frames = (0, 2 * T_l)
    query_indices = None
    if a.query_mode == "trajectory":
        query_indices = trajectory_query_indices(sample, T_l, h, w, a.res)
    recorder = AttentionTransportRecorder(
        pipe, blocks, denoise_steps, a.steps, a.cfg, f, h, w, action_frames,
        a.query_samples, a.seed, query_indices=query_indices).install()
    device, dtype = pipe.device, torch.bfloat16
    try:
        _seed_all(a.seed)
        generated = np.stack([np.asarray(x) for x in pipe(
            prompt=[prompt], negative_prompt="", template=template,
            streams={"video": sample["streams"]["video"].unsqueeze(0).to(device=device, dtype=dtype)},
            conditioning_mode="iiii",
            camera=sample["camera"].unsqueeze(0).to(device=device, dtype=dtype),
            action_7d=sample["action_7d"].unsqueeze(0).to(device=device, dtype=dtype),
            extrinsics=sample["extrinsics"].unsqueeze(0).to(device=device, dtype=dtype),
            intrinsics=sample["intrinsics"].unsqueeze(0).to(device=device, dtype=dtype),
            height=a.res, width=a.res, num_frames=a.num_frames, cfg_scale=a.cfg,
            num_inference_steps=a.steps, seed=a.seed, tiled=False,
            tile_size=(a.res // 16, a.res // 16), tile_stride=(a.res // 32, a.res // 32),
            enable_usp=False, cfg_parallel=False)])
    finally:
        recorder.remove()
    if len(recorder.records) != len(blocks) * len(denoise_steps):
        raise RuntimeError(f"captured {len(recorder.records)} maps, expected "
                           f"{len(blocks) * len(denoise_steps)}")

    decoded = decode_action(generated, sample, template, a.num_frames)
    out_dir = Path(a.out) / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    serial = []
    arrays = {}
    segment_names = ("video_v0", "action_v0", "video_v1", "action_v1")
    for i, rec in enumerate(sorted(recorder.records, key=lambda x: (x["denoise_step"], x["block"]))):
        mass = rec["key_mass"]
        target_metrics = _target_metrics(mass, masks["target"], anchor_frames)
        segment_mass = {name: float(mass[j * T_l:(j + 1) * T_l].sum())
                        for j, name in enumerate(segment_names)}
        serial.append({"block": rec["block"], "denoise_step": rec["denoise_step"],
                       "target_metrics": target_metrics, "segment_attention_mass": segment_mass})
        rec["target_metrics"] = target_metrics
        arrays[f"b{rec['block']}_s{rec['denoise_step']}"] = mass.astype(np.float32)
    np.savez_compressed(out_dir / "attention_maps.npz", **arrays)
    report = {
        "tag": a.tag, "checkpoint": str(Path(a.ckpt).resolve()), "task": a.task,
        "episode": a.episode, "path": sample["path"], "prompt": prompt,
        "view_indices": [int(x) for x in sample["view_indices"]],
        "frame_indices": [int(x) for x in sample["frame_indices"]], "mask_meta": mask_meta,
        "blocks": blocks, "denoise_steps": denoise_steps, "steps": a.steps, "cfg": a.cfg,
        "query_mode": a.query_mode, "query_samples": len(recorder.query_indices),
        "query_indices": recorder.query_indices.tolist(),
        "latent_grid": [f, h, w], "attention": serial,
        "action_metrics": decoded["metrics"],
    }
    with open(out_dir / "attention_transport.json", "w") as fp:
        json.dump(report, fp, indent=1)
    ordered = sorted(recorder.records, key=lambda x: (x["denoise_step"], x["block"]))
    _plot(ordered, masks["target"], anchor_frames, out_dir / "attention_transport.png")
    print(f"wrote {out_dir / 'attention_transport.json'}")


if __name__ == "__main__":
    main()
