# Handoff: running the fusion-arm evaluation on another machine

Evaluates one checkpoint of the **full-modality canvas** arm (ten segments,
`video+depth+segmentation+normal+action`, both views) on two generalisation tiers plus
closed-loop action. Everything here has been run on the origin host except the GPU stages, which
are blocked there by other tenants — that is the only reason this is being moved.

**Read §1 and §5 before starting.** §1 stops you uploading 137 GB you do not need; §5 lists the
flags whose defaults silently measure the wrong thing.

---

## 0. What question this answers

The arm puts all five modalities in ONE latent canvas and trains a schedule that sometimes
withholds a modality entirely (no clean frame at all), so the model must produce it from the
others. The evaluation asks:

1. **Per-modality quality under four conditioning regimes** (`full_anchor`, `rgb_only`,
   `rgb_given`, `policy`) — in particular `rgb_only`, which is the deployment condition: one RGB
   frame per view and nothing else.
2. **The mutual-completion matrix** — withhold each modality in turn and score it. Quality rising
   with the number of co-present modalities means they supervise each other; **flat means they
   merely share a decoder**, and that outcome is pre-registered as a reason to stop this
   direction rather than scale it.
3. **Closed-loop action success** through the same ten-segment canvas under `rgb_only`.

Two tiers, both generalisation — variation-0 (trained on) is deliberately excluded:

| Tier | Data | Meaning |
|---|---|---|
| **T2** | wide tree, `variation1` | unseen layout, task **was** trained on |
| **T3** | unseen-task tree | 6 tasks **never** trained on |

Consequence to keep in mind: with no variation-0 tier there is **no in-distribution baseline**, so
T2/T3 are absolute quality, not "degraded by X% versus seen". What survives is T2-vs-T3 (layout
generalisation vs task generalisation) and the matrix, which is a within-episode contrast and
never needed a seen baseline.

---

## 1. Artefacts — upload ~13 GB, not 150 GB

| Artefact | Size | Where it comes from |
|---|---|---|
| This repository | small | your GitHub |
| **Arm checkpoint** `step5000.ckpt` | **12 GB** | your HF repo |
| **Eval data subset** (14 episodes) | **~420 MB** | your HF repo |
| Wan2.2-TI2V-5B base | 32 GB | **download from HF directly** (`Wan-AI/Wan2.2-TI2V-5B`) — do not re-upload |

**Do not upload the full 137 GB tree.** The offline evaluation touches exactly 14 episodes. The
dataset class globs whatever is present and its start-up gates are ratio-based, so a subset
containing only these episodes passes. Copy with paths preserved:

```
# T2 -> tree rlbench_selfgen_512_aug_wide
close_jar/variation1/episodes/episode0
insert_onto_square_peg/variation1/episodes/episode0
light_bulb_in/variation1/episodes/episode0
meat_off_grill/variation1/episodes/episode0
open_drawer/variation1/episodes/episode0
push_buttons/variation1/episodes/episode0
reach_and_drag/variation1/episodes/episode0
put_item_in_drawer/variation1/episodes/episode0

# T3 -> tree rlbench_unseen_tasks_512_aug
basketball_in_hoop/variation0/episodes/episode19627
close_box/variation0/episodes/episode210343
close_drawer/variation0/episodes/episode124879
close_microwave/variation0/episodes/episode237383
take_lid_off_saucepan/variation0/episodes/episode174048
toilet_seat_down/variation0/episodes/episode139673
```

> **T3 episode names are random seeds, not `episode0`.** Re-derive them on the target machine
> rather than trusting this list — the selection is "first episode of each task, sorted", so a
> different subset changes it. `scripts/wait_and_eval_all.sh` prints the list it will use before
> it starts; run the dry check in §4 and copy from there.

Each episode directory must keep all of: `actions.npy`, `meta.json`, `handles.json`,
`scene_segments.json`, `seg_targets.json`, and `view1..view4/` each with `rgb/video.mp4`,
`depth.npz`, `mask.npz`, `camera_params.json`. **`scene_segments.json` is mandatory** — the
dataset refuses to start without it under the `scene_roles` protocol. (`seg_targets.json` is only
read under the `referring` protocol and the unseen-task tree has none; that is fine.)

Expected layout on the target machine:

```
<repo>/data/rlbench_selfgen_512_aug_wide/      # or a symlink
<repo>/data/rlbench_unseen_tasks_512_aug/
<repo>/checkpoints/Wan-AI/Wan2.2-TI2V-5B/      # or a symlink
<repo>/outputs/fusion0__seed42_fi3_512_aug_wide_sr_F1/checkpoint-5000/step5000.ckpt
```

**No credentials in the repo.** The training launcher reads a W&B key from a `.env` outside the
tree; evaluation needs no key at all. Do not copy `.env`, and do not commit any token.

---

## 2. Hardware

- **1 GPU, >= 31 GB free.** Measured peak for the offline stages is ~31 GB; the arm trained at
  38.8 GB peak across four ranks but evaluation is single-rank and shorter-sequence.
- ~40 GB disk for reports and generated videos.
- Offline stages need **no simulator**. Closed-loop does — see §7.

---

## 3. Environment

```bash
conda create -n fusion_eval python=3.10.20 -y && conda activate fusion_eval
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.57.3 deepspeed==0.16.9 diffsynth==1.1.9
pip install imageio imageio-ffmpeg opencv-python pillow scikit-image lpips einops safetensors
```

**Do not install `flash-attn`.** The start-up log prints
`Flash Attention library "flash_attn" not found, using pytorch attention implementation`, which
reads as a warning but is not one: PyTorch 2.6's SDPA already dispatches to FlashAttention for
this shape. Verified on the origin host — `torch.ops.aten._fused_sdp_choice` returns backend 1
(FLASH_ATTENTION) for `[1, 24, 28160, 128]` bf16, 54.9 ms versus 88.1 ms for the mem-efficient
path. The external package buys nothing and costs a long build.

**`PYTHONPATH` must be the repo root**, and `cd` alone is not enough under `torchrun`. An editable
install of any distribution also named `actionimages` will shadow the `training` package;
`train.py` asserts on this, but set it explicitly:

```bash
cd <repo> && export PYTHONPATH=$PWD
```

---

## 4. Verify before spending GPU

All CPU-only, about a minute:

```bash
cd <repo> && export PYTHONPATH=$PWD

# 1. canvas layout, mask roles, and train/eval parity for all five regimes
python tests/test_fusion_axis.py          # expect ALL_FUSION_AXIS_TESTS_PASSED
python tests/test_forward_unchanged.py    # expect ALL_FORWARD_UNCHANGED_TESTS_PASSED
python tests/test_eval_assembly.py        # expect EVAL_ASSEMBLY_OK
python tests/test_policy_conditioning.py  # expect ALL_POLICY_CONDITIONING_TESTS_PASSED

# 2. the data subset loads and its start-up gates pass
python - <<'PY'
from training.dataset.rlbench_selfgen import RLBenchSelfgenDataset
for root in ("data/rlbench_selfgen_512_aug_wide", "data/rlbench_unseen_tasks_512_aug"):
    ds = RLBenchSelfgenDataset(base_path=root, num_frames=41, frame_interval=3,
            height=512, width=512,
            template_mix="video+depth+segmentation+normal+action@1.0",
            prompt_tag_style="explicit", variations="all",
            segmentation_mode="scene_roles", strict_getitem=True)
    print(root, "->", len(ds), "episodes")
PY

# 3. which episodes the run will actually use (dry: it then waits for a GPU, so interrupt it)
timeout 12 env OUT_DIR=outputs/fusion0__seed42_fi3_512_aug_wide_sr_F1 TAG=dry \
  bash scripts/wait_and_eval_all.sh > /tmp/dry.txt 2>&1; grep -E "^T2 |^T3 |EVAL_TARGET" /tmp/dry.txt
```

Expect from step 2: `seg coverage OK`, `scene_roles OK: zero unknown pixels`. A failure there is a
data-subset problem, not a code problem — most likely a missing `scene_segments.json`.

---

## 5. Flags that silently measure the wrong thing if left at their defaults

This is the section to re-read if a number looks implausible.

The two scripts have **different** defaults, and the dangerous ones are all in `eval/rollout.py`.
Verified against the source, not from memory:

| Flag | Must be | `modality_mode_grid.py` | `eval/rollout.py` | What goes wrong if wrong |
|---|---|---|---|---|
| `--res` / `--res` | **512** | 512 (ok) | **256** | The arm trained at 512; at 256 you measure a resolution mismatch, not a policy. |
| `--frame_interval` / `--frame-interval` | **3** | 3 (ok) | **1** | Changes how much motion a 41-frame window spans. |
| `--segmentation_mode` | **scene_roles** | scene_roles (ok) | n/a | A checkpoint must be asked with the protocol it trained on -- the two protocols use different prompt tags *and* different pixels. |
| `--anchor-modality` | **fusion** | n/a | **video** | `video` asks a ten-segment checkpoint through a four-segment canvas: off-distribution, and the number then measures the format mismatch. |
| `--only` | the fusion template | all templates | n/a | Otherwise it also evaluates four-segment templates this arm never trained. |

So: the grid is safe at its defaults for `res`/`frame_interval`/`segmentation_mode` but needs
`--only`; **the rollout is unsafe at three of its defaults.**

`scripts/wait_and_eval_all.sh` sets all of these. If you invoke the underlying scripts yourself,
copy them from there rather than from memory.

---

## 6. Run it

```bash
cd <repo> && export PYTHONPATH=$PWD
setsid env OUT_DIR=outputs/fusion0__seed42_fi3_512_aug_wide_sr_F1 TAG=wide5000 \
  bash scripts/wait_and_eval_all.sh >> logs_eval_all_wide5000.txt 2>&1 < /dev/null &
```

It waits for a GPU with >= 31 GB free, then runs five stages. Each stage **waits and retries
independently** (up to 6 attempts): a stage releases its memory when it exits, so on a shared host
another tenant can take the card between stages, and one transient must not abort the rest.

| Stage | Work | Rough cost |
|---|---|---|
| `STAGE1_GRID_T2` | 4 regimes x 8 episodes | ~5 h |
| `STAGE1B_ONEOUT_T2` | 6 withheld modalities x 8 episodes | ~7 h |
| `STAGE1C_GRID_T3` | 4 regimes x 6 episodes | ~4 h |
| `STAGE1D_ONEOUT_T3` | 6 x 6 episodes | ~5 h |
| `STAGE2_CLOSEDLOOP` | 3 tasks x 10 trials | hours; needs the simulator (§7) |

Knobs: `N_TASK` (T2 episodes, default 8), `N_TASK3` (T3, default 6), `TRIALS` (closed-loop per
task, default 10), `NEED` (MiB free required, default 31000), `MAX_TRY` (default 6).

**Episodes are the outermost loop and `metrics.json` is rewritten after every one**, so a run cut
short still leaves every completed episode scored. Asking for more episodes than the machine can
finish costs nothing.

Progress: `tail -f logs_eval_all_wide5000.txt`. Useful greps: `_START gpu=`, `_DONE`, `_RETRY`,
`_GAVE_UP`, `EVAL_ALL_FINISHED`.

---

## 7. Closed-loop needs a simulator; the offline stages do not

`eval/rollout.py` drives RLBench in CoppeliaSim and needs `COPPELIASIM_ROOT` plus PyRep,
RLBench and `xvfb-run` for headless GL. That is a much larger install than the offline stages.

**If the simulator is not available, run the offline stages and skip closed-loop.** Comment out
the `STAGE2_CLOSEDLOOP` line in `scripts/wait_and_eval_all.sh`; stages 1–1D are self-contained and
carry the mutual-completion matrix, which is the headline.

Note: rendering is software GL (llvmpipe) unless the host has a GL driver. Export
`LP_NUM_THREADS=4` if you touch data generation — one unconstrained worker takes ~34 cores for
very little, and limiting it is ~9x cheaper in core-seconds.

---

## 8. Outputs

```
reports/modality_mode_grid/wide5000_grid_t2/metrics.json      # per-episode + aggregate
reports/modality_mode_grid/wide5000_oneout_t2/metrics.json    # the matrix
reports/modality_mode_grid/wide5000_grid_t3/metrics.json
reports/modality_mode_grid/wide5000_oneout_t3/metrics.json
reports/closedloop/rollout_wide5000_cl.json
logs_eval_*.txt, logs_cl_*.txt
```

`metrics.json` carries `results` (one entry per `episode|template|mode`) and `aggregate` (per
template/mode/modality: `n`, `median`, `mean`, bootstrap 95% CI, and the raw values). Each grid
directory also has a `grid.png` contact sheet for the first episode only.

---

## 9. Reading the numbers — three traps

**(a) Depth AbsRel is meaningless without `decode_coverage`, which sits next to it.** Depth is
encoded as a path through the RGB cube; a generated image that leaves the path decodes to `NaN`,
those pixels are dropped, and AbsRel is computed over whatever remains. On the origin host an
early checkpoint reported AbsRel 1.202 at **coverage 0.043** — an error over 4% of pixels, not a
depth measurement. Always quote the pair. The same applies to segmentation's `unknown_px`.

**(b) A `fully_given` segment is a VAE round-trip of ground truth, not generation.** Under
`rgb_given` the video PSNR (~30.7 dB on the origin host) is a floor, and reporting it as
generation quality would be wrong. The JSON marks each segment `fully_given` / `absent`.

**(c) One episode is not a result.** This repository has a precedent: a headline 1.31x ratio in
`ARM7_RESULTS.md` reversed once it was re-tested with adequate power, after it had already been
quoted. That is why this protocol runs 8 and 6 episodes with bootstrap CIs. Quote `n` and the
interval, not the median alone.

What to look for in the matrix: for each modality, quality under `full_anchor` (own anchor, upper
bound) versus `out:<modality>` (absent, four others present) versus `rgb_only` (absent, only video
present). **Monotone with the number of co-present modalities = cross-modal completion is real.
Flat = they share a decoder**, and the pre-registered response is to stop rather than scale.

---

## 10. Context the numbers should be read against

From the origin host, already measured; see `paper_fusion/NUMBERS.md` for provenance of each.

- **Training-time evidence that the anchor dependence is breaking.** Per-modality loss ratio
  `absent / anchored`, first 1000 steps -> last 1000 of 5000: depth 1.776 -> 1.266, action
  1.374 -> 0.860, segmentation 1.390 -> 1.097, video 1.635 -> 1.409, **normal 1.705 -> 1.855**
  (the only one getting worse). Four of five improve. Action's full-run ratio is **0.945**, i.e.
  action is no harder without its own anchor — the observations carry it.
- **Rotary-position extrapolation is not the bottleneck.** Segments 5–9 sit at latent positions
  55–109, far past the backbone's pretraining range, yet the same-modality loss changes by only
  0.3–1.9% from view 0 to view 1, and one modality improves. Systematic position degradation
  would be uniform and large.
- **An earlier single-episode read at step 2000 looked negative** (coverage collapsing when a
  modality is absent). It sampled before the trend above became visible. Do not take it as the
  baseline expectation for step 5000.
- `normal` is analytically a function of `depth`. It is the sharpest probe in the matrix — a
  deterministic right answer exists — and also the one modality whose training trend is adverse.

---

## 11. Known traps in this repository

- **`grep` here is ugrep**, whose `-q -v` semantics differ from GNU grep and can silently take the
  wrong branch in shell scripts. Prefer explicit tests.
- **Never edit a running `.sh`.** Bash reads a script by byte offset as it executes; editing one
  mid-run has made a live shell resume inside a comment and restart a job.
- **`pkill -f <pattern>` will match the shell running it** if the pattern appears in your own
  command line. Kill by exact PID and exclude `/bin/bash -c` entries.
- **A non-empty `--output_dir` makes training resume from it** and silently ignore the warm-start
  checkpoint. Not relevant to evaluation, but relevant if you train here.
- **The unseen-task tree names episodes by random seed**, not `episode0`. A hardcoded `episode0`
  matches nothing there and the tier is silently skipped with a one-line `SKIP`.
