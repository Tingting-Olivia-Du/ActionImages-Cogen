# Observation-space ablation — handoff

You are running **arm0 – arm6** of an observation-space ablation on a second machine. This
document is the whole contract: what the repo does, what the arms are, how to get the data and
weights, how to launch, and the failures that have already cost this project days. Read the
gotchas (§7) before the first launch — every one of them is a mistake that has actually
happened here.

---

## 1. What this repo is

A fork of **ActionImages** (Wan2.2-TI2V-5B video-diffusion backbone) that trains **one**
checkpoint to co-generate *perception* and *action* in a single sequence, with **no task heads
and one loss**.

The trick: an action is written **as pixels**. A 6-DoF end-effector pose is rendered into each
camera view as a coloured blob (position in the red channel, a small axis for rotation, a blue
pedestal for gripper state). So "predict the action" becomes "predict some more frames", and
the same diffusion objective covers RGB, depth, surface normals, segmentation and action alike.

A training sample is a **template**: a canvas of segments, e.g.

```
depth+action   ->   D0 | A0 | D1 | A1        (modality view0 | action view0 | modality view1 | action view1)
```

A **conditioning plan** then decides what is given vs predicted. Evaluation and closed-loop
rollout both use `IIII`, which gives only the first latent frame of each segment — that is what
a real rollout supplies.

Key files:

| path | what it is |
|---|---|
| `scripts/train_arm.sh` | **one arm = one `--template_mix`**; everything else identical across arms |
| `training/templates.py` | template parsing, the A axis (`ACTION_MASK_MIX_PRESETS`) and M axis |
| `training/dataset/base.py` | `CombDataset`: `name@ratio` mixing across data sources |
| `training/dataset/rlbench_selfgen.py` | the on-disk contract every source conforms to |
| `training/dataset/maniskill3.py` | ManiSkill3 source (byte-identical layout, `frame_interval` pinned to 1) |
| `training/percep/` | depth / normal / segmentation codecs (pixels ↔ metric values) |
| `eval/eval_action.py` | offline action scoring (r_peak, position, rotation, gripper) |
| `eval/rollout.py` | closed-loop RLBench rollouts (needs CoppeliaSim; **not** needed for training) |

---

## 2. What you are running, and why

The ablation is an **observation-space ladder**. `video+action` is pinned at 0.4 on every rung
that has perception, so the action budget never changes; the only thing that varies is **which
visual spaces the remaining 0.6 is spent on**.

| arm | template mix | rung |
|---|---|---|
| `arm0` | `video+action@1.0` | control: no perception at all |
| `arm1` | `video+action@0.4, depth+action@0.6` | one space — depth |
| `arm2` | `video+action@0.4, segmentation+action@0.6` | one space — segmentation |
| `arm3` | `video+action@0.4, normal+action@0.6` | one space — normals |
| `arm4` | `video+action@0.4, depth@0.3, segmentation@0.3` | two spaces |
| `arm5` | `video+action@0.4, normal@0.3, segmentation@0.3` | two spaces |
| `arm6` | `video+action@0.4, normal@0.3, depth@0.3` | two spaces |

(`arm7` = all three at 0.2 and `arm8` = uniform 0.25×4 are running on the origin machine. Do
not start those.)

**Every arm uses `ACTION_MASK_MIX=A1`.** It is declared per-arm in `scripts/train_arm.sh`, so
you do not pass it — but do not override it either. Holding A1 constant across the whole ladder
is what makes the rungs comparable: any difference is the observation-space menu and nothing
else. If you ever see `action_mask_mix=A0` in a launch log, stop and report it.

Arms that contain `segmentation` also declare `SEG_MODE=scene_roles` (a dense 9-role map, not a
target-only mask). This is likewise part of the arm definition, not a preference.

---

## 3. Machine and container

The origin machine is a **Docker container** (`/.dockerenv` present) on:

- Ubuntu 20.04.6, CUDA 12.1 toolkit, driver **610.43.02**
- 8 × **NVIDIA RTX 6000 Ada**, 49 GB each
- conda 23.9.0; the training env is **`ttd_train`, Python 3.10**

You do not need an identical machine — you need **4 GPUs of ≥48 GB per arm** (the run is
bf16 + gradient checkpointing + DeepSpeed ZeRO-2 with Adam offloaded to CPU, and it sits at
~33 GB/GPU). It should also work on 80 GB cards; on 24 GB cards it will not fit.

Environment (`requirements.txt` is the source of truth; the pinned versions actually in use):

```
torch 2.6.0+cu124   transformers 4.57.3   diffsynth 1.1.9
deepspeed 0.16.9    accelerate 1.5.2      wandb 0.28.1
```

```bash
conda create -n ttd_train python=3.10 -y && conda activate ttd_train
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .            # the distribution MUST be named "actionimages"; train.py asserts on it
```

**Do not install `flash-attn`.** The start-up line `Flash Attention library "flash_attn" not
found, using pytorch attention implementation` reads as a warning and is not one: measured on
the origin host, `torch.ops.aten._fused_sdp_choice` returns backend 1 (FlashAttention) for this
model's `[1, 24, 28160, 128]` bf16 attention — 54.9 ms, against 88.1 ms for the memory-efficient
path. Torch's own SDPA already dispatches to Flash for this shape. The external package buys
nothing and costs a very long build. It is commented out in `requirements.txt` for this reason.

### 3.1 Container

The origin host runs bare in a container rather than from an image built for this project, but
a reproducible image is four lines. The pins below are transcribed from the host, not chosen:

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
# python 3.10.20, then:
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir transformers==4.57.3 deepspeed==0.16.9 diffsynth==1.1.9 \
        accelerate safetensors einops imageio imageio-ffmpeg opencv-python-headless \
        pillow scikit-image lpips wandb numpy scipy
```

`libgl1`, `libglib2.0-0` and `ffmpeg` are needed: the loader decodes `.mp4` per episode, and
OpenCV pulls the GL libraries even in the headless build.

Run it with `--gpus '"device=0,1,2,3"' --ipc=host --shm-size=32g`. The dataloader uses 4 workers
per rank with pinned memory, and the default 64 MB `/dev/shm` will deadlock it.

Mount the data and the weights rather than baking them in — they are ~200 GB together:

```bash
docker run --rm -it --gpus '"device=0,1,2,3"' --ipc=host --shm-size=32g \
    -v /host/data:/app/data -v /host/checkpoints:/app/checkpoints \
    -v /host/outputs:/app/outputs -e WANDB_API_KEY \
    <image> bash scripts/train_arm.sh arm1
```

One known patch: DeepSpeed's bf16 path does not skip non-finite gradients upstream. If you hit
an overflow that will not clear, the origin repo's history contains
`chtc/patch_deepspeed_bf16_overflow.py` (removed in this cleanup) — ask before reintroducing it,
since none of the 6k runs so far have needed it.

**W&B**: `train_arm.sh` refuses to start without a key, because `wandb.init()` runs *after* the
12.8 GB checkpoint has loaded and a missing key would waste that load. Either export
`WANDB_API_KEY`, or point `TTD_ENV` at a file containing `WANDB_API=...`, or pass
`EXTRA_ARGS='--report_to none'`.

---

## 4. Data

Two sources, **5,710 training episodes over 23 tasks**, mixed
`rlbench_selfgen_512_aug_wide@0.7, maniskill3@0.3`. That ratio equalises exposure per episode
(RLBench 4.24 draws/episode vs ManiSkill 4.11 at 6k steps) and per task (≈4.4 % vs ≈4.3 %).

Both live under `data/` with the **same byte-level layout**, which is why one loader reads both:

```
data/<tree>/<task>/variation0/episodes/episode<N>/
    view{1..4}/rgb/video.mp4      # 512x512
    view{1..4}/depth.npz          # float16, METRES
    view{1..4}/mask.npz           # uint16 segmentation ids
    view{1..4}/camera_params.json # per-frame 4x4 cam2world (RLBench convention) + intrinsics
    actions.npy                   # [T,8] xyz + quat(xyzw) + gripper
    meta.json                     # written LAST -> presence means the episode is complete
    scene_segments.json, seg_targets.json, handles.json
```

### 4.1 Download

```bash
cd <repo>/data
# --- RLBench: 16 tasks, 3,960 episodes at variation0, shipped as one tar per task
huggingface-cli download TingtingDu/rlbench_selfgen_512_aug_wide --repo-type dataset \
    --local-dir /tmp/rlb_dl
mkdir -p rlbench_selfgen_512_aug_wide && cd rlbench_selfgen_512_aug_wide
for t in /tmp/rlb_dl/archives/*.tar; do tar xf "$t"; done
cd ..

# --- ManiSkill3: 7 tasks, 1,750 episodes at variation0 (+175 at variation1), raw files
huggingface-cli download TingtingDu/maniskill3 --repo-type dataset \
    --local-dir maniskill3
```

Sizes: RLBench ≈ 137 GB, ManiSkill3 ≈ 32 GB. **Budget ≥ 250 GB free** — see §7 on disk.

> The RLBench upload was still in flight when this document was written. Verify all 16 tars are
> present before extracting; if a task is missing, re-run the download rather than training on a
> partial tree.

### 4.2 Verify before training

The loader prints its own census at startup — that line is the acceptance test:

```
[selfgen] variations='0': kept 3960/4200 episodes across 16 tasks     # RLBench
[selfgen] variations='0': kept 1750/1925 episodes across 7 tasks      # ManiSkill3
```

Expected per-task counts (250 each unless noted): `light_bulb_in` 249,
`place_shape_in_shape_sorter` 249, `sweep_to_dustpan` 212; all other RLBench tasks 250.
ManiSkill3: 250 per task for all seven.

A quick structural check:

```bash
find data/rlbench_selfgen_512_aug_wide -name meta.json | wc -l   # 4200 (all variations)
find data/maniskill3 -name meta.json | wc -l                     # 1925
```

**`variation0` is the training split.** `variation1` is held out and must never be trained on —
`train_arm.sh` passes `--variations 0` for you; do not change it.

---

## 5. Warm start

Every arm starts from the released ActionImages checkpoint, so the question the ablation asks is
"does adding perception damage an **already-trained** action prior", not "what is easier to
learn from scratch".

```bash
# 12.8 GB, the DiT weights only
huggingface-cli download anyeZHY/ActionImages step125750.ckpt --local-dir checkpoints/official

# 32 GB base model: the VAE, the T5 text encoder and the tokenizer that the DiT plugs into.
# The .ckpt above is the denoiser ONLY -- without this the pipeline cannot encode pixels or text.
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
    --local-dir checkpoints/Wan-AI/Wan2.2-TI2V-5B \
    --include "diffusion_pytorch_model*.safetensors" "models_t5_umt5-xxl-enc-bf16*.pth" \
               "Wan*_VAE.pth" "google/*" "*.json"
```

The paths are exact: `ModelConfig(local_model_path="checkpoints", model_id="Wan-AI/Wan2.2-TI2V-5B")`
globs `checkpoints/Wan-AI/Wan2.2-TI2V-5B/<pattern>` and is created with `skip_download=True`, so
a missing file does not trigger a download — it yields an empty file list and a confusing
`AttributeError: 'list' object has no attribute 'endswith'`. If you see that error, this step is
what is wrong.

---

## 6. Launch

One arm at a time per 4-GPU group. Identical everything except the arm name:

```bash
cd <repo>
GPUS=0,1,2,3 \
DATASET="rlbench_selfgen_512_aug_wide@0.7,maniskill3@0.3" \
STEPS=6000 CKPT_EVERY=1000 SAVE_OPTIM=False \
OUT="$PWD/outputs/arm1_joint2src_seed42_fi3_6k" \
  setsid nohup bash scripts/train_arm.sh arm1 \
  > outputs/logs/arm1.log 2>&1 < /dev/null &
```

Then **verify it detached** — this is not optional, see §7.1:

```bash
ps -o pid,pgid,sid,cmd -p <pid>     # PGID and SID must both equal the PID
```

Defaults you should not change (they are part of the comparison): `SEED=42`,
`FRAME_INTERVAL=3`, `RES=512`, `--num_frames 41`, `per_device_train_batch_size=1`,
`gradient_accumulation_steps=1`, `lr 5e-7` with 1000 warmup steps, `--variations 0`,
`--full_param True`, DeepSpeed `configs/zero2_offload.json`.

**Effective batch is 4 per step** (4 GPUs × 1 × 1), so 6,000 steps = 24,000 samples.

### 6.1 What a healthy start looks like

```
weights: warm-start from .../checkpoints/official/step125750.ckpt
mask axes: perception_mask_mix=M2 (INERT: all N templates contain <action>)  action_mask_mix=A1
Loading models from: ['checkpoints/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-...safetensors', ...]
[selfgen] variations='0': kept 3960/4200 episodes across 16 tasks
[selfgen] variations='0': kept 1750/1925 episodes across 7 tasks
... {'loss': 0.03-0.06, 'grad_norm': 0.1-0.4, 'learning_rate': ...}
```

Check all four: **A1**, a non-empty model file list, both dataset censuses, and loss in the
0.03–0.06 band. Throughput here is ~17 s/it with two arms sharing the machine, ~8 s/it for a
single arm on 4 GPUs — so one arm is roughly **13–28 h** for 6k steps.

### 6.2 Checkpoints

`CKPT_EVERY=1000` and `SAVE_OPTIM=False` give one ~12.8 GB `stepN.ckpt` per 1,000 steps
(~77 GB per arm for the full run). With `SAVE_OPTIM=True` each checkpoint also writes ~108 GB
of DeepSpeed optimiser state and a save transiently needs ~240 GB free — that is how two runs
died here. Leave it `False`; the cost is that a crash resumes with fresh Adam moments.

Resume is automatic: relaunching the same command finds the newest `stepN.ckpt` in `OUT` and
continues. `scripts/train_watchdog.sh` does this on a 180 s poll and refuses to restart when
free disk is under 40 GB.

---

## 7. Gotchas — all of these have already broken this project

### 7.1 `nohup` is not enough; use `setsid`
On 2026-09-18 both 6k runs died at step ~1480 with
`SignalException: got signal: 15`. The supervising shell's **process group** was torn down and
SIGTERM reached `torchrun`. `nohup` only ignores SIGHUP. Launch with
`setsid nohup … < /dev/null &`, and verify `PGID == SID == PID`. A launcher script must not
`wait` — it becomes the parent that gets torn down.

### 7.2 `CUDA_VISIBLE_DEVICES` inherited from a sourced env file
An env script exported `CUDA_VISIBLE_DEVICES=0`. Every worker then saw **one** device: jobs
pinned to GPU ≥ 1 crashed with `ordinal N is not valid (device count: 1)` and the rest piled onto
GPU 0. If you source anything before launching, `unset CUDA_VISIBLE_DEVICES` or use `env -u`.

### 7.3 Disk
The volume here hit **632 KB free** mid-run. Budget: data ≈ 170 GB + base model 32 GB + warm
start 13 GB + ~77 GB per arm of checkpoints. Keep ≥ 40 GB headroom at all times — the watchdog
enforces this, a bare relaunch does not.

### 7.4 Do not let a local build overwrite tracked files
Unrelated to training, but the same class of error: check `git status` before committing and
never `git add -A` blindly.

### 7.5 The base model is not optional
See §5. `Loading models from: []` in the log means the Wan files are missing; the run will die
several minutes later with a `'list' object has no attribute 'endswith'` traceback.

---

## 8. What to report back

For each arm:

1. `OUT/checkpoint-{1000..6000}/step*.ckpt` — the 12.8 GB DiT weights. **step6000 is required;
   step4000 is wanted too** (it is the budget point the earlier runs used, so it keeps the new
   ladder comparable with them).
2. The launch log, including the four startup lines from §6.1 — they are the proof that the arm
   trained on what it claims.
3. The W&B run id, or the loss curve if W&B was off.
4. Any step at which the run was restarted, and why.

Do **not** run evaluation on your side unless asked. Offline scoring (`eval/eval_action.py`)
needs held-out trees that are not in this handoff, and closed-loop (`eval/rollout.py`) needs a
CoppeliaSim install. Both run on the origin machine, where the trees already exist.

---

## 9. If you change anything

The ladder only means something if the arms differ **in one variable**. If you have to change a
knob — a different GPU count, a different batch size, anything — say so explicitly in the report
rather than absorbing it, and apply it to **every** arm, not just the one that needed it. A rung
trained under a different effective batch is not on the same ladder.
