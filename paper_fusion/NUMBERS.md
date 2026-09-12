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

## PENDING — must stay a placeholder in the .tex until measured

| Quantity | Status |
|---|---|
| Scaled-tree arm (5,000 steps, `TRAIN_EXIT=0`) per-modality grid | eval queued; all 8 GPUs held by outside tenants |
| **Mutual-completion matrix** (one-out per modality) | queued, stage 1b of `scripts/wait_and_eval_all.sh` |
| **Closed-loop action success rate** | queued, stage 2 |
| F0 all-anchor control arm (`fusion0-anchor`) | not trained |
| Held-out (variation 1/2) numbers for any fusion arm | not run |
| Multi-episode / multi-seed CIs for any fusion number | not run |
| RoPE-extrapolation read from `segloss_pos/<k>` | logged to wandb; not yet extracted |
