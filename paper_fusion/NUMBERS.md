# NUMBERS — every quantity in the manuscript, with where it came from

Rule for this directory: **no number appears in the .tex without a row here.** A number whose
row says MEASURED may be stated as fact; one whose row says PENDING must appear in the .tex as
an explicit placeholder (`\TODO{...}`), never as a plausible-looking value. The previous
manuscript used `[XX]` for the same purpose (see `../paper/README.md`).

## Architecture / cost — MEASURED

| Quantity | Value | Source |
|---|---|---|
| Base model | Wan2.2-TI2V-5B, dim 3072, 30 layers, 24 heads, patch (1,2,2) | state-dict hash branch, `training/models/wan_video_dit.py` |
| VAE | `WanVideoVAE38`, z_dim 48, spatial 16x, temporal 4x | `diffsynth/models/wan_video_vae.py:1354-1381` |
| Latent per segment | `[B,48,11,32,32]` at 512^2, 41 frames | derived; printed by the smoke run |
| Tokens per segment | 11 x 16 x 16 = 2,816 | patchify (1,2,2) |
| 4-segment canvas | f=44, 11,264 tokens | arm0-arm7u |
| **10-segment canvas** | **f=110, 28,160 tokens** | this work |
| policy-mode canvas | f=30 | measured, `describe_plan` |
| 4-seg throughput | 20.5 s/it, 2x RTX6000Ada | `logs_arm7u.txt` |
| **10-seg throughput** | **36-37 s/it, 3-4x RTX6000Ada** | `logs_fusion0.txt`, `logs_fusion0_wide.txt` |
| 10-seg peak VRAM, 4 ranks | **38.8 GB / 48 GB** (2400 x 1s samples, none >43 GB) | `scratchpad/fusion0_mem.txt` |
| 10-seg peak VRAM, 3 ranks | 44.2 GB observed (1 of 150 x 4s samples) | coarser sampling; see caveat in text |
| Fixed per-step overhead | ~16.5 s (45%), = 14 VAE encodes + CPU Adam | solved from the policy/full step-time pair |
| SDPA backend at this shape | FLASH_ATTENTION (54.9 ms vs 88.1 ms mem-efficient) | `aten._fused_sdp_choice` on `[1,24,28160,128]` bf16 |

## Mask axis — MEASURED

| Quantity | Value | Source |
|---|---|---|
| F1 regime mix | (0.30, 0.30, 0.15, 0.15, 0.10) | `FUSION_MASK_MIX_PRESETS` |
| Realised mix over 4000 draws | 0.307 / 0.316 / 0.139 / 0.138 / 0.100 | `tests/test_fusion_axis.py` |
| Condition frames per regime | 10 / 2 / 22 / 8 / 4 (of 110, 110, 110, 110, 30) | measured via `assemble` |
| Action share of predicted positions | **0.263** overall (0.20 anchored regimes, 0.77 policy) vs 0.500 for a 4-segment arm | counted over 2000 draws |
| Policy-mode step-time signature | 7-8% of steps at 20-21 s vs 37 s | `logs_fusion0.txt`, independent confirmation the axis fires |

## Data — MEASURED

| Quantity | Value | Source |
|---|---|---|
| Base tree | 16 tasks, 788 variation-0 episodes, 240 held-out, 35 GB | `rlbench_selfgen_512_aug` |
| **Scaled tree** | **16 tasks, 3,960 variation-0 episodes, 240 held-out, 137 GB** | `rlbench_selfgen_512_aug_wide` |
| Codec fidelity, scaled tree | depth AbsRel med 7e-4; normal cos 1.0000; unknown px 0 | 99 episodes / 396 (episode,view) |
| Same statistics on base tree | depth 7e-4; normal 1.0000; unknown 0 | same script, control |
| `Roof` incident | 3 of 4,200 episodes; handle-union amplified to 540 | Sec. on data scaling |

## Results — fusion0 @ step 2000, BASE tree, n=1 episode — MEASURED but UNDERPOWERED

Single episode (`open_drawer/variation0/episode0`), single seed, 40% of training. Reported in
the text only as a direction indicator, never as a headline.

| regime | video PSNR | depth AbsRel | depth decode_coverage | normal cos | seg unknown px |
|---|---|---|---|---|---|
| full_anchor | 15.384 | 0.1212 | 0.99951 | 0.83155 | 35119 |
| rgb_only | 14.828 | 0.39497 | 0.40939 | -0.28561 | 4218 |
| rgb_given | 30.705 (given) | 1.20215 | 0.04342 | -0.34457 | 10371 |
| policy | 32.295 | 0.84418 | 0.99918 | 0.71803 | 0 |

External reference points, from `ARM7_RESULTS.md` at step 10000 (different arm, 4-segment):
arm7 depth AbsRel 0.1315, normal cos 0.8805; depth specialist 0.2267, normal specialist 0.3699.

## Training-time diagnostics, scaled-tree arm, 5,000 steps — MEASURED, and well powered

Extracted from the run's own W&B transaction log (`wandb/run-20260909_034125-11gts31i/*.wandb`,
parsed via `wandb.sdk.internal.datastore`; keys arrive under `nested_key`, not `key`). These need
no GPU and are averaged over thousands of steps, so they carry far more power than the n=1
offline grid above. **They reverse the step-2000 single-episode read.**

Realised F1 schedule over 5,001 steps — note the denominator: `segloss_pos/0` is logged only when
segment 0 has a predicted position, which excludes `rgb_given` (video fully given) and `policy`
(video collapsed to its anchor), i.e. 25.5% of steps. Using it as the denominator inflates every
share by 1/0.745.

| regime | realised | declared |
|---|---|---|
| full_anchor | 0.299 | 0.30 |
| rgb_only | 0.301 | 0.30 |
| rgb_given | 0.157 | 0.15 |
| one_out | 0.145 | 0.15 |
| policy | 0.098 | 0.10 |

**`absent / anchored` loss ratio, full-run means** (n = 2,000–3,000 steps per cell):

| modality | anchored | absent | ratio |
|---|---|---|---|
| video | 1.7383 | 2.5859 | 1.488 |
| depth | 0.7206 | 1.0141 | 1.407 |
| segmentation | 0.7646 | 0.8984 | 1.175 |
| normal | 1.2789 | 2.2867 | 1.788 |
| **action** | 0.2978 | 0.2816 | **0.945** |

Action's ratio is **below 1**: it is no harder to produce without its own anchor, which is direct
evidence that the co-present observations carry what action needs.

**Trend, first 1,000 steps → last 1,000** (falling = learning to generate without an anchor):

| modality | early | late | Δ |
|---|---|---|---|
| depth | 1.776 | 1.266 | **−0.510** |
| action | 1.374 | 0.860 | **−0.513** |
| segmentation | 1.390 | 1.097 | −0.293 |
| video | 1.635 | 1.409 | −0.226 |
| **normal** | 1.705 | **1.855** | **+0.150** |

Four of five improve. **`normal` is the exception and gets worse** — notable because it is
analytically a function of depth, so the model is evidently not learning that constraint.

**Rotary-position extrapolation: ruled out.** Same-modality loss, view 0 (positions 0–43) → view 1
(55–109): video −0.0228, depth +0.0165, segmentation +0.0052, normal +0.0062, action +0.0052 —
i.e. 0.3–1.9%, one of them negative. Systematic position degradation would be uniform and large.

**Peak memory, allocator high-water:** mean 36.29 GB, max 36.64 GB over 5,000 steps at four ranks.
(`nvidia-smi` read 38.8 GB; the difference is CUDA context.)

**Learning rate `5e-7` is conservative, not too high** (from the trainer log, 500 records):

| phase | lr | grad_norm med | p90 | max | loss med |
|---|---|---|---|---|---|
| 0–10% (warmup) | 5e-9 → | 0.716 | 2.640 | 6.567 | 0.0801 |
| 10–20% | 2.55e-7 | 0.339 | 0.731 | 0.866 | 0.0577 |
| 20–50% | 5e-7 | 0.295 | 0.522 | 3.085 | 0.0483 |
| 50–80% | 5e-7 | 0.288 | 0.611 | 1.811 | 0.0445 |
| 80–100% | 5e-7 | 0.269 | 0.504 | 0.821 | 0.0423 |

grad_norm falls monotonically; only 5.0% of steps touch the 1.0 clip and those concentrate in
warmup; loss still descending at the end. Under-trained, not unstable.

## PENDING — must stay a placeholder in the .tex until measured

| Quantity | Status |
|---|---|
| Scaled-tree arm (5,000 steps, `TRAIN_EXIT=0`) per-modality grid | eval queued; all 8 GPUs held by outside tenants |
| **Mutual-completion matrix** (one-out per modality) | queued, stage 1b of `scripts/wait_and_eval_all.sh` |
| **Closed-loop action success rate** | queued, stage 2 |
| F0 all-anchor control arm (`fusion0-anchor`) | not trained |
| Held-out (variation 1/2) numbers for any fusion arm | not run |
| Multi-episode / multi-seed CIs for any fusion number | not run |
| ~~RoPE-extrapolation read~~ | **DONE — extracted, extrapolation ruled out (see above)** |
