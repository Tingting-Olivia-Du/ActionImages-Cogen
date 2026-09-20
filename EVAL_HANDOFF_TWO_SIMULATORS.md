# Closed-loop evaluation on two simulators — handoff

You are running the closed-loop evaluation of a batch of checkpoints on a second machine.
This document is the whole contract: what to pull, what to install, which tasks to run, the
exact protocol, and the failures that have already cost this project days. Read §7 before the
first launch — every gotcha there is a mistake that has actually happened.

Scope: **closed-loop success rate on RLBench and ManiSkill3**, for checkpoints at the 4,000-
and/or 6,000-step budget points. Offline (open-loop) perception scoring is a different pipeline
and is not covered here.

---

## 0. Prerequisite: the code must be pushed first

The evaluation code for this is **not in any published commit yet**. On the origin machine
(`/workspace/1228_tingting/ActionImages-Cogen`, branch `main`, remote
`github.com/Tingting-Olivia-Du/ActionImages-Cogen`) the following are new or modified and
must be committed and pushed **before** the second machine clones or pulls:

| path | state | what it is |
|---|---|---|
| `eval/rollout_env_maniskill.py` | new | ManiSkill3 env wrapper, duck-typed against `RolloutEnv` |
| `eval/rollout_maniskill.py` | new | ManiSkill3 rollout loop |
| `eval/rollout.py` | modified | GT replay now holds the final pose (§4.3) |
| `scripts/maniskill3_gen.py` | modified | adds the `PlaceSphere-v1` task definition |
| `scripts/run_cl_phase2.sh` | new | the campaign dispatcher (both simulators, one queue) |
| `scripts/run_unseen14_campaign.sh` | new | RLBench-only dispatcher (superseded, kept for reference) |
| `scripts/run_unseen14_gt_replay.sh` | new | the RLBench GT-replay gate |
| `scripts/cl_json_complete.py` | new | the single definition of "this job is finished" |
| `scripts/cl_campaign_status.py` | new | done/total for a queue file |
| `scripts/make_unseen14_tables.py` | new | the results table (safe to run mid-campaign) |
| `scripts/eval_watchdog.sh` | new | auto-resume guardian |

```bash
cd /workspace/1228_tingting/ActionImages-Cogen
git add eval/rollout_env_maniskill.py eval/rollout_maniskill.py eval/rollout.py \
        scripts/maniskill3_gen.py scripts/run_cl_phase2.sh scripts/run_unseen14_campaign.sh \
        scripts/run_unseen14_gt_replay.sh scripts/cl_json_complete.py \
        scripts/cl_campaign_status.py scripts/make_unseen14_tables.py scripts/eval_watchdog.sh \
        EVAL_HANDOFF_TWO_SIMULATORS.md
git commit -m "Closed-loop evaluation on two simulators: ManiSkill3 harness, campaign dispatcher, GT gate"
git push origin main
```

`git status` first, and do not `git add -A`: the working tree also holds unrelated
paper-figure scripts and ~100 GB of reports and checkpoints that must not be committed.

The second machine then pulls and installs the package in place:

```bash
git clone https://github.com/Tingting-Olivia-Du/ActionImages-Cogen.git
cd ActionImages-Cogen && pip install -e .       # the distribution MUST be named "actionimages"
```

---

## 1. What "closed loop" means here, and what it needs

A rollout is receding-horizon: the policy is handed one anchor RGB frame plus the instruction,
generates a 41-frame canvas, the action segments are decoded into world-frame end-effector
poses, those are executed, and at the end of the chunk the model is asked again. The cap is
`1.5x` the demonstration length.

**The most useful fact for planning: RLBench closed-loop needs NO rendered dataset.** Scenes
come from `task.get_demos(live_demos=True)` + `reset_to_demo`, i.e. they are generated live by
the simulator. So adding RLBench tasks costs GPU time only — no tree generation, no disk.
ManiSkill3 is the opposite: its scenes are reproduced from a stored episode tree (§3.2).

You need, per checkpoint:

- the `stepN.ckpt` DiT weights (12.8 GB);
- the Wan2.2-TI2V-5B base model (32 GB) — the `.ckpt` is the denoiser ONLY, without the base
  there is no VAE and no text encoder;
- CoppeliaSim 4.1.0 + PyRep + RLBench (RLBench side);
- ManiSkill3 + SAPIEN + the `data/maniskill3_heldout` tree (ManiSkill side).

---

## 2. Environment

One env serves both simulators. Build it **on a persistent volume**, not under `/opt/conda`:
on the origin machine every env under `/opt/conda/envs` was lost with a container restart,
taking `ttd_train` and `ttd_rollout` with it.

```bash
conda create -p /mnt/persistent/envs/ttd_eval python=3.10 -y
PIP=/mnt/persistent/envs/ttd_eval/bin/pip

$PIP install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
$PIP install "numpy==1.26.4" transformers==4.57.3 diffsynth==1.1.9 "imageio[ffmpeg]" \
    safetensors einops sentencepiece protobuf modelscope ftfy trimesh matplotlib \
    opencv-python-headless scikit-image scipy lpips pandas accelerate cffi lxml defusedxml

# --- version locks that are NOT optional, see the reasoning below
$PIP install "diffusers==0.33.1" "huggingface-hub>=0.34.0,<1.0"
$PIP install -c <(printf 'torch==2.6.0\ntransformers==4.57.3\ndiffusers==0.33.1\nhuggingface-hub==0.36.2\nnumpy==1.26.4\n') xfuser
$PIP install --no-deps https://github.com/facebookresearch/vggt/archive/refs/heads/main.zip

# --- RLBench side
export COPPELIASIM_ROOT=/path/to/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
$PIP install https://github.com/stepjam/PyRep/archive/refs/heads/master.zip
$PIP install --no-deps https://github.com/stepjam/RLBench/archive/refs/heads/master.zip
$PIP install gymnasium pyquaternion natsort

# --- ManiSkill side
$PIP install --no-deps sapien==3.0.3 mplib==0.1.1 toppra h5py pytorch_kinematics \
    fast_kinematics transforms3d dacite tabulate tyro arm_pytorch_utilities pytorch_seed ipython
$PIP install --no-deps -e /path/to/ManiSkill     # editable clone of haosulab/ManiSkill
```

System packages (apt), all of which were missing on a fresh container:

```bash
apt-get install -y libglib2.0-0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 \
    libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xkb1 \
    libxcb-shape0 libxcb-xfixes0 libsm6 libice6 libxext6 libxrender1 libfontconfig1 \
    libdbus-1-3 libxi6 libxtst6 libgl1 libglu1-mesa git
```

**Why the version locks.** `training/models/wan_video_dit.py` imports `xfuser` at module
scope, so it is required even though sequence parallelism is off. `xfuser` needs
`diffusers>=0.33`, but `diffusers>=0.34` requires `huggingface-hub>=1.23`, which
`transformers==4.57.3` refuses. `diffusers==0.33.1` with `huggingface-hub==0.36.2` is the only
point that satisfies all three. `inference.py` imports `vggt` at module scope as well.

**Do not install `flash-attn`.** Torch's own SDPA already dispatches to FlashAttention for this
model's attention shape (measured 54.9 ms against 88.1 ms for the memory-efficient path); the
package buys nothing and costs a very long build.

**Rendering.** RLBench runs under `xvfb-run -a` with software GL — plain xvfb is enough for
rollouts (the eval-tree *generation* path is the one that needs GPU EGL via VirtualGL).
SAPIEN needs a Vulkan ICD; if `/usr/share/vulkan/icd.d` is empty, point at the one SAPIEN
ships:

```bash
export VK_ICD_FILENAMES=<env>/lib/python3.10/site-packages/sapien/vulkan_library/nvidia_icd.json
```

Smoke test before anything else — it needs no GPU and no checkpoint:

```bash
xvfb-run -a python -u eval/rollout.py --gt-replay --tasks close_box --num-trials 1 \
  --variation 0 --res 256 --arm-action-mode planning --max-steps-factor 1.5 \
  --tag smoke --no-record-video
```

---

## 3. The task lists

### 3.1 RLBench — 14 never-trained tasks

None of these appears in the 16-task training tree. **Nothing needs to be downloaded for
these**: the scenes are generated live by CoppeliaSim (§1). The rendered tree
`TingtingDu/rlbench_unseen_tasks_512_aug` on the Hub is for the OFFLINE perception evaluation,
which this document does not cover — a closed-loop run never reads it.

```
close_box          close_drawer       close_microwave     take_lid_off_saucepan
toilet_seat_down   basketball_in_hoop meat_on_grill       light_bulb_out
take_item_out_of_drawer               take_money_out_safe put_rubbish_in_bin
stack_cups         open_window        wipe_desk
```

GT-replay ceiling measured on the origin machine, 20 trials each, corrected protocol (§4.3),
**271/280 = 96.8%**. Four tasks do not reach a full ceiling and must be read against it:

```
wipe_desk 14/20     basketball_in_hoop 19/20     stack_cups 19/20     take_money_out_safe 19/20
```

The 16 tasks that ARE in the training tree, for reference — do not evaluate on them and do not
confuse `meat_off_grill` (trained) with `meat_on_grill` (held out), or `light_bulb_in`
(trained) with `light_bulb_out` (held out):

```
close_jar  insert_onto_square_peg  light_bulb_in  meat_off_grill  open_drawer
place_shape_in_shape_sorter  push_buttons  put_groceries_in_cupboard  put_item_in_drawer
put_money_in_safe  reach_and_drag  slide_block_to_target  stack_blocks  stack_wine
sweep_to_dustpan  turn_tap
```

### 3.2 ManiSkill3 — two blocks

Both need the episode tree, because a ManiSkill rollout reproduces a stored episode's scene
AND its cameras (the tree stores the per-episode camera parameters the model was trained to
see, and the policy is anchored on frames rendered through them). Download both from the Hub
into `data/`:

```bash
cd <repo>/data
huggingface-cli download TingtingDu/maniskill3_heldout --repo-type dataset \
    --local-dir maniskill3_heldout          # 3 tasks x 50 episodes, ~2.1 GB
huggingface-cli download TingtingDu/maniskill3 --repo-type dataset \
    --local-dir maniskill3                  # 7 tasks, variation0 + variation1, ~32 GB
```

Verify before running: the loader's own census is the acceptance test.

```bash
find data/maniskill3_heldout -name meta.json | wc -l    # 150
find data/maniskill3 -name meta.json | wc -l            # 1925  (1750 var0 + 175 var1)
```

Only `variation1` of `data/maniskill3` is used here — `variation0` is the training split and
must never be evaluated on. If you would rather regenerate than download,
`scripts/maniskill3_gen.py` writes the same trees from the ManiSkill demonstrations.

**Held-out TASKS** (`data/maniskill3_heldout`, variation0):

```
pull_cube        GT 20/20   -> evaluate
place_sphere     GT 20/20   -> evaluate
lift_peg_upright GT  0/20   -> DO NOT evaluate, harness floor (see below)
```

**Trained tasks, held-out SEEDS** (`data/maniskill3`, variation1):

```
pick_cube        GT 20/20   -> evaluate
stack_cube       GT 20/20   -> evaluate
pull_cube_tool   GT 20/20   -> evaluate
push_cube        GT 20/20   -> evaluate
stack_pyramid    GT 14/20   -> optional; report the 70% ceiling with it
peg_insertion_side GT 1/20  -> DO NOT evaluate, harness floor
plug_charger     GT  0/20   -> DO NOT evaluate, harness floor
```

**Why three tasks are excluded.** The harness executes absolute Cartesian end-effector poses,
because that is what the model predicts. The three excluded tasks are solved in the reference
demonstrations by contact-force transients that a Cartesian pose tracker cannot reproduce:
`lift_peg_upright` stands the peg up by pivoting it against the table while the gripper itself
rotates only 35 deg, and the two insertion tasks need sub-millimetre compliant motion. This was
diagnosed, not assumed — scene reproduction is exact (rendered frame 0 matches the stored video
to MAE 2.4/255), `actions.npy` matches the recomputed TCP poses to 0.00 mm / 0.000 deg, position
tracks to 5 mm and rotation to 2.3 deg, and repeating each waypoint 2/3/4 times does not help.
A model score on these tasks would measure the harness, not the model.

---

## 4. The protocol

### 4.1 Flags, and which of them are not free choices

```bash
python -u eval/rollout.py \                    # RLBench (wrap in `xvfb-run -a`)
  --ckpt <stepN.ckpt> --tag <tag> --tasks <task> --variation 0 --num-trials 20 \
  --res 512 --cfg 7.5 --steps 50 --frame-interval 3 --prompt-tag-style explicit \
  --axis-solver sphere --arm-action-mode planning --max-steps-factor 1.5 \
  --skip-anchor-frames 4 --execution-horizon 41 --max-ik-fail-streak 5 \
  --anchor-modality video --seed 42 --record-video --out reports/closedloop_unseen14

python -u eval/rollout_maniskill.py \          # ManiSkill3 (no xvfb)
  --ckpt <stepN.ckpt> --tag <tag> --tasks <task> \
  --tree data/maniskill3_heldout --variation 0 --num-trials 20 \
  --res 512 --cfg 7.5 --steps 50 --frame-interval 1 --prompt-tag-style explicit \
  --axis-solver sphere --max-steps-factor 1.5 --skip-anchor-frames 4 \
  --execution-horizon 41 --max-ik-fail-streak 5 --seed 42 --record-video \
  --out reports/closedloop_maniskill
```

- **`--frame-interval` is 3 on RLBench and 1 on ManiSkill.** Not a preference: the loader pins
  the ManiSkill source to stride 1 (`training/dataset/base.py`), because those episodes are
  49-103 frames at 20 Hz and stride 3 would run off the end of most of them. A stride-3
  ManiSkill rollout scores the checkpoint on a stride it never saw for that source.
- **`--prompt-tag-style explicit`** for every fine-tuned arm (they were trained with
  `<video><action> ` prefixes). The released upstream checkpoint never saw those tags: score it
  with `--prompt-tag-style none --cfg 10.0 --frame-interval 4`, which is what it was trained
  with. Mixing these up measures prompt-distribution shift, not capability.
- **`--skip-anchor-frames 4`** drops the chunk frames that merely reconstruct the pose the arm
  is already in; the dropped slots are refilled with a ramp so the chunk keeps its length.
- **`--max-ik-fail-streak 5`** is a harness policy, not a model property. It can move a number
  by 26 pp on its own. Keep it at 5 and never compare across different values.
- **`--record-video`** is on by default and worth the disk (~330 MB per 100 rollouts). Without
  it a failure is a single scalar with no way to tell "the world model is wrong" from "the
  action decode is wrong". Two videos are written per trial: what the simulator executed, and
  what the model imagined.

### 4.2 Scenes are paired across checkpoints

`trial_seed()` hashes only `(task, variation, trial)` with blake2b — not the checkpoint, not
the process. Two checkpoints scored with the same `--num-trials` therefore face **identical
scenes**, which is what licenses the paired McNemar test. (It uses blake2b and not Python's
`hash()` because string hashing is salted per process, which would silently give each arm its
own scenes while every individual run still looked fine.)

### 4.3 Run the GT-replay gate first

GT replay executes each scene's own demonstration through the identical controller, with no
model and no GPU. A task near zero there is a limit of the actuation harness and must be
excluded from the aggregate rather than charged to the model.

```bash
TAGPFX=gtgate bash scripts/run_unseen14_gt_replay.sh                       # RLBench, 14 tasks, CPU
python -u eval/rollout_maniskill.py --gt-replay --tree data/maniskill3_heldout \
  --variation 0 --tasks pull_cube --num-trials 20 --no-record-video --tag msgt_pull_cube
```

**GT replay holds the final pose** through the same `1.5x` budget the model gets, rather than
stopping when the demonstration's frames run out. This is a correction made on 2026-09-20 and
it matters: a Cartesian tracker lags its target, so anything still settling when the last
waypoint is issued has not settled yet. On `place_sphere`, which needs the sphere at rest
within 5 mm, the ceiling was **12/20 stopping at the demo's end and 20/20 holding** — eight
"failures" that were the budget, not the harness. If you compare against a ceiling measured
before that date, it is understated.

### 4.4 Reading the result

```bash
python scripts/make_unseen14_tables.py     # safe mid-campaign; short cells print as x/n
```

It prints per-task cells, Wilson intervals (which do not go negative at the low rates most of
these tasks sit at), the aggregate with and without the harness-floor gate, and an exact
McNemar between the two arms over the paired scenes.

---

## 5. Running a batch of checkpoints

`scripts/run_cl_phase2.sh` is a work queue over `(model, backend, task)`, dispatched across
GPUs. Edit `ckpt_of()` and `MODELS` to name your checkpoints; the order of `MODELS` is the
priority order, because a queue that never fully drains should still produce the most important
numbers first.

```bash
MODELS="myrun_4k myrun_6k" GPUS="0 1 2 3 4 5 6 7" \
  setsid nohup bash scripts/run_cl_phase2.sh > logs/phase2.log 2>&1 < /dev/null &
setsid nohup bash scripts/eval_watchdog.sh >> logs/eval_watchdog.log 2>&1 < /dev/null &
```

- **One job per GPU.** A rollout holds ~30 GB; two do not fit on a 49 GB card.
- **Workers pull from a shared flock'd queue**, so per-task cost differences (5x across these
  tasks) do not leave cards idle.
- A job is finished only when its JSON carries `--num-trials` scored results
  (`scripts/cl_json_complete.py`). Re-running the dispatcher skips finished jobs, so it is
  resumable and idempotent.
- The watchdog restarts a dead dispatcher, refuses to start anything under 40 GB free, and
  **rewinds the queue index to 0 on restart** — a worker killed mid-job has already advanced
  the shared index past its job, and a naive restart would skip it forever.

### Cost

Measured per trial, including scene setup: RLBench averages ~400 s (306 s for `close_drawer`
up to 575 s for `basketball_in_hoop`), ManiSkill ~572 s. So per checkpoint:

```
RLBench   14 tasks x 20 trials  ~= 31 GPU-hours
ManiSkill  6 tasks x 20 trials  ~= 19 GPU-hours
                                  ~= 50 GPU-hours, or ~7 h on 8 GPUs
```

---

## 6. What to report back

Per checkpoint:

1. `reports/closedloop_unseen14/rollout_*.json` and `reports/closedloop_maniskill/rollout_*.json`
   — each carries its own `args` block, so the protocol is auditable after the fact.
2. The GT-gate JSONs from the same machine. Do not reuse a ceiling measured elsewhere: it
   depends on the CoppeliaSim build and the controller.
3. The videos, or at least the failures. `reports/*/videos/<tag>/{executed,generated}/`.
4. The output of `scripts/make_unseen14_tables.py`.
5. Any task you dropped, and its GT-gate number.

---

## 7. Gotchas — every one of these has already broken this project

### 7.1 `pgrep -f <script>` matches shells that merely QUOTE the script
Every Claude Code / CI tool call runs as `bash -c '<command string>'`, and those wrappers
survive as orphans. If the string wrote a script with a heredoc, its command line contains that
script's whole text, so `pgrep -f` matches it forever. Two supervisor scripts here waited on
`while pgrep -f "<script>"; do sleep; done` and could **never** exit, so a campaign handover
silently never happened. Skip any candidate whose `argv[1]` is `-c` (read
`/proc/<pid>/cmdline`) and skip your own process group; see `alive()` in `eval_watchdog.sh`.

### 7.2 `nohup` is not enough; use `setsid`
Both 6k training runs died at step ~1480 with `SignalException: got signal: 15` when the
supervising shell's process group was torn down. `nohup` only ignores SIGHUP. Launch with
`setsid nohup ... < /dev/null &` and verify `PGID == SID == PID`.

### 7.3 Never edit a shell script that is currently running
bash reads a script by byte offset and resumes there, so changing the file's length under a
running dispatcher makes it execute from a wrong offset. Either restart it, or edit a copy and
`mv` it into place — the rename swaps the inode and the running process keeps reading the old
one.

### 7.4 Two dispatchers with separate queues will claim the same job
Starting a second dispatcher while the first is still draining had both of them claim the same
task; they wrote the same JSON and one truncated the other's 16 scored trials to zero. One
queue at a time, or drain the first (push its index to the end) before starting the second.

### 7.5 `lazy_render` on the ManiSkill env means NO pixels at all
`ManiSkillRolloutEnv(lazy_render=True)` builds the env with `obs_mode="none"`, which is worth
minutes per episode on a GT replay — but `obs_mode` also decides which textures the cameras
capture, so `views_rgb` then raises `KeyError: 'rgb'`. This voided 18 model jobs, because the
flag had only ever been exercised by `--gt-replay --no-record-video`, the one path that never
asks for a pixel. `rollout_maniskill.py` now derives the flag from
`(gt_replay and not record_video)`, and `_sensors()` raises a named error instead. **Whenever
you change a rendering path, verify it on a real model rollout, not only on GT replay.**

### 7.6 `CUDA_VISIBLE_DEVICES` inherited from a sourced env file
An env script exported `CUDA_VISIBLE_DEVICES=0`; every worker then saw one device and jobs
pinned to GPU >= 1 crashed with `ordinal N is not valid`. `unset CUDA_VISIBLE_DEVICES` in the
rc, or use `env -u`.

### 7.7 Disk
Budget the base model (32 GB) plus 12.8 GB per checkpoint plus a few GB of videos, and keep
40 GB of headroom. The origin volume hit 632 KB free mid-run once.

### 7.8 Aggregate success rates from different task sets are not comparable
Not even at identical n. Mid-campaign the arm0 column read 50.0% and the arm7 column 19.6% —
purely because arm0 had only run the six easiest tasks at that point. Always compare per task,
or over the paired intersection.
