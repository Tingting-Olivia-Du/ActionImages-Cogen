# Perception by simulator replay — handoff for the second machine

You are filling **Table 2 of the paper** (`Tables/tab_main.tex`, "Perception under simulator
replay"): how good each arm's *generated perception* is, scored against a ground truth the
simulator produces by **executing that model's own decoded actions**.

This document is the whole contract: why the obvious target is wrong, how to get the code onto
a machine that cannot pull from GitHub, what to download, what to run, and the traps that have
already cost time here. **Read §9 before quoting any number.**

Scope: perception under `IIII`, five unseen tasks, arms 0–7. Closed-loop success rate is
Table 1 and a different pipeline (`EVAL_HANDOFF_TWO_SIMULATORS.md`).

---

## 1. Why the demonstration is not the ground truth

Under `IIII` the model gets frame 0 plus an instruction and generates the rest. It invents its
own future. Scoring generated frame *t* against the **demonstration's** frame *t* measures two
things added together:

```
AbsRel(generated, demo)  =  perception error  +  trajectory divergence
```

The second term is neither small nor separable, and it differs per checkpoint — so two arms
scored that way are not comparable at all. The `X+action` templates make it unavoidable:
`depth+action` is `D0 | A0 | D1 | A1`, no RGB anywhere in the canvas, so no conditioning pins
the future while leaving depth to be predicted. Arm 7's `A1` mixture spends 0.00 on `fiii`, so
even the cross-view route is out of distribution for it.

**So we change the ground truth, not the metric.** Execute the model's own decoded actions in
the simulator and render what they actually produce:

```
target  =  D_sim(s0, â_1:t)
```

The model is no longer charged for failing to reproduce a demonstration it was never required
to reproduce. The world went where the model sent it; what remains is whether the geometry it
drew matches the geometry its own plan produced.

### What this measures, and what it does not

Accuracy **conditional on the model's own trajectory**. Genuine accuracy — the simulator
renders real metric geometry, real occlusion, real object poses, and the model cannot influence
that referee except through its actions. Shape, contact and occlusion cannot be faked.

It does **not** measure whether the imagined future is useful. A model that plans to do nothing
and correctly renders a static scene scores well. Always quote it beside closed-loop success
(Table 1). A perception number alone is not a claim that the model is useful.

## 2. The pipeline

```
 frame 0 in the anchor modality  +  instruction
        │
        ▼
 one 41-frame canvas:  D0 | A0 | D1 | A1     (anchor=depth; V/S/N analogous)
        │              └ imagined depth  └ imagined action
        ▼
 decode the action segments → 41 end-effector poses
        │
        ▼
 execute them in the simulator, one env.step() per frame
        │
        ├─► after every executed step: render TRUE depth + mask + RGB   ← the target
        ▼
 compare imagined frame k  ↔  simulator render after executing step k
```

Both sides are metres, 512×512, pixel-aligned in the same camera (the simulator's own
extrinsics/intrinsics are read back per trial and stored). Direct subtraction — no alignment,
no fitted scale, no correspondence search.

---

## 3. Getting the code onto a machine with no GitHub access

Everything needed is in **`handoff/simreplay_bundle.tar.gz`** (12 KB). Copy it by any means
(scp, USB, paste as base64 — it is small on purpose).

```bash
# on THIS machine
cd /workspace/1228_tingting/ActionImages-Cogen
base64 -w0 handoff/simreplay_bundle.tar.gz > /tmp/bundle.b64   # if you must paste it

# on the TARGET machine, at the repo root
base64 -d /tmp/bundle.b64 > simreplay_bundle.tar.gz            # if pasted
mkdir -p handoff && tar xzf simreplay_bundle.tar.gz -C handoff

git apply --check handoff/simreplay_core.patch && \
git apply        handoff/simreplay_core.patch                  # 3 files modified
cp handoff/score_sim_replay.py  scripts/
cp handoff/run_percep_simreplay.sh scripts/ && chmod +x scripts/run_percep_simreplay.sh
```

If `git apply` refuses because that machine's copies have drifted, apply the three edits by
hand — they are small and each is described below. `patch -p1 --fuzz=3 < handoff/simreplay_core.patch`
also usually works.

### The three edits, in case you apply them manually

| file | edit | why |
|---|---|---|
| `eval/rollout_env.py` | new ctor kwarg `extra_renders: Sequence[str] = ()`; OR it into `_need_depth` / `_need_mask` | **Decouples what the simulator renders from what the policy is anchored on.** Previously `_need_depth = anchor_modality in ("depth","normal","all")`, so an RGB-anchored run rendered no depth. Arm 0 has no non-RGB template and *must* run `anchor=video`, so its half of the comparison was unmeasurable. Only ever ADDS renders, so existing campaigns stay byte-identical. |
| `eval/rollout_env_maniskill.py` | same kwarg; OR it into `need_extra` which selects `obs_mode="rgb+depth+segmentation"` | same reason, ManiSkill side |
| `eval/rollout.py` | `--dump-percep`, `--dump-max-trials`; per-trial `.npz`; `extra_renders=("depth","mask")` when dumping | writes the raw arrays the scorer needs |

### Verify the patch took

```bash
python - <<'PY'
import inspect, eval.rollout_env as E
assert "extra_renders" in inspect.signature(E.RolloutEnv.__init__).parameters
import ast; ast.parse(open("eval/rollout.py").read())
print("patch OK")
PY
```

Then a **no-GPU, no-checkpoint** end-to-end smoke test (~6 min, needs xvfb + CoppeliaSim):

```bash
xvfb-run -a python eval/rollout.py --gt-replay --tasks close_microwave --variation 0 \
  --num-trials 1 --tag smoke --res 512 --frame-interval 3 --arm-action-mode planning \
  --max-steps-factor 1.5 --dump-percep --out /tmp/smoke
python -c "
import numpy as np; d=np.load('/tmp/smoke/percep_dump/smoke/close_microwave_v0_t0.npz')
assert {'sim_depth','sim_mask','sim_rgb','cmd','ach'} <= set(d.files), d.files
e=np.linalg.norm(d['cmd'][:,:3]-d['ach'][:,:3],axis=-1)
print('smoke OK; controller floor median %.4f m (expect <0.001)'%np.median(e))"
```

That last number is the **controller's** noise floor and must be sub-millimetre. It was 0.0003 m
here. If it is not, stop — the whole protocol rests on the simulator faithfully executing what
it is told.

## 4. What to download

### Checkpoints

Arms 0–7 at a matched step budget (4,000 here). Write a manifest at the repo root, one line
per arm, which is how the launcher finds them:

```
# ckpts.txt
arm0 /abs/path/outputs/arm0_.../checkpoint-4000/step4000.ckpt
arm1 /abs/path/outputs/arm1_.../checkpoint-4000/step4000.ckpt
...
arm7 /abs/path/outputs/arm7_.../checkpoint-4000/step4000.ckpt
```

**Every arm must be the same number of steps and the same `ACTION_MASK_MIX`.** Arm 0 on this
machine is `A0` while arm 7 is `A1`, which confounds any arm0-vs-arm7 comparison; arms 1–6 are
all `A1` and are the cleaner counterpart. Check `outputs/*/launch logs` before trusting a row.

### Data trees

RLBench rollouts build scenes live and need no tree. You still want
`data/rlbench_unseen_tasks_512_clean` for the **expert-ceiling** row (§6), which reads real
frames. ManiSkill needs its held-out tree for the same reason.

### External expert models

Cached to `~/.cache/huggingface` on first use; pre-download on a machine with no egress:

| modality | model | note |
|---|---|---|
| depth | `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf` | **metric** → zero free parameters, transformers-native |
| normal | `jingheya/lotus-normal-g-v1-1` | needs the Lotus repo on `sys.path` (see `scripts/probe_normal_specialists.py`) |
| segmentation | `CIDAS/clipseg-rd64-refined` | receives the scene-role vocabulary |

```bash
huggingface-cli download depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf
huggingface-cli download jingheya/lotus-normal-g-v1-1
huggingface-cli download CIDAS/clipseg-rd64-refined
```

UniDepth V2 would be an equivalent metric depth expert but ships its own package, which is not
installed here; DA V2 Metric was chosen purely on availability. Either is fine — but §9.4.

---

## 5. What Table 2 needs

Reproduced from `Tables/tab_main.tex` so you can see exactly which cell each job fills. Native
output spaces come from `scripts/train_arm.sh` (V=video, D=depth, N=normal, S=segmentation):

```
arm0 V       arm1 V+D     arm2 V+S     arm3 V+N
arm4 V+D+S   arm5 V+N+S   arm6 V+D+N   arm7 V+D+N+S (XGenAct)
```

```latex
\begin{tabular}{@{}llccccccc@{}}
\textbf{Arm (training outputs)} & \textbf{Prediction route}
  & Close microwave & Toilet seat down & Close box & Meat on grill & RL avg.
  & Pull cube & All avg. \\
\midrule
\multicolumn{9}{@{}l}{\textbf{Future RGB} --- LPIPS $\downarrow$} \\
Arm 0 (V) .. Arm 7 (V,D,N,S)   & direct video              & ... all eight arms ... \\
\midrule
\multicolumn{9}{@{}l}{\textbf{Metric depth} --- dynamic-region AbsRel $\downarrow$} \\
Arm 1 (V,D) / Arm 4 (V,D,S) / Arm 6 (V,D,N) / Arm 7   & direct co-generation & ... \\
Arm 7 (V,D,N,S)                & RGB $\rightarrow$ depth expert     & ... \\
Arm 0 (V)                      & RGB $\rightarrow$ depth expert     & ... \\
\emph{Expert ceiling}          & GT RGB $\rightarrow$ depth expert  & ... \\
\midrule
\multicolumn{9}{@{}l}{\textbf{Surface normal} --- dynamic-region mean cosine $\uparrow$} \\
Arm 3 (V,N) / Arm 5 (V,N,S) / Arm 6 (V,D,N) / Arm 7   & direct co-generation & ... \\
Arm 7 / Arm 0                  & RGB $\rightarrow$ normal expert    & ... \\
\emph{Expert ceiling}          & GT RGB $\rightarrow$ normal expert & ... \\
\midrule
\multicolumn{9}{@{}l}{\textbf{Segmentation} --- dynamic-region mIoU $\uparrow$} \\
Arm 2 (V,S) / Arm 4 (V,D,S) / Arm 5 (V,N,S) / Arm 7   & direct co-generation & ... \\
Arm 7 / Arm 0                  & RGB $\rightarrow$ segmentation expert & ... \\
\emph{Expert ceiling}          & GT RGB $\rightarrow$ segmentation expert & ... \\
\end{tabular}
```

**Every cell is `prediction / copy-anchor floor` on the dynamic region.** Keeping the floor in
every cell is deliberate: simulator replay is conditional on each model's own trajectory, and a
model that imagines no motion would otherwise look artificially good.

Three structural rules from the table's own comments:

- **A direct row exists only for a modality in that arm's training mixture.** Do not prompt
  arm 1 for normals. The launcher refuses this.
- **Only two cascades are in the main table** — arm 7 (same weights, isolates direct vs
  sequential) and arm 0 (RGB-only control). All-arm cascade sweeps go in the appendix.
- **Video has no cascade**: RGB is already the intermediate a cascade would consume. It is
  scored directly against simulator RGB, with copy-anchor as the floor.

### Tasks

| simulator | tasks |
|---|---|
| RLBench | `close_microwave`, `toilet_seat_down`, `close_box`, `meat_on_grill` |
| ManiSkill3 | `pull_cube` |

All absent from the training tree. Variation 0, 5 trials each, **the same trial count for every
method**. `RL avg.` macro-averages the four RLBench task means; `All avg.` is only filled once
`pull_cube` exists, and macro-averages the five task means rather than pooling frames.

---

## 6. Running it

```bash
# defaults: all 20 native (arm, anchor) pairs x 4 RLBench tasks, 5 trials, GPUs 0-3
bash scripts/run_percep_simreplay.sh

CONFIGS="arm1_depth arm4_depth arm6_depth" TASKS=close_box bash scripts/run_percep_simreplay.sh
GPUS="0 1" TRIALS=5 bash scripts/run_percep_simreplay.sh
```

The launcher resolves checkpoints from `ckpts.txt`, refuses any `(arm, anchor)` pair the arm
never trained, refuses a missing checkpoint, and does both **before** starting a rollout rather
than 25 minutes in. It is a `flock` queue: it resumes, and re-running skips finished tags.

The full matrix is **20 configs × 4 tasks = 80 RLBench jobs**. Measured here: 18–44 min per job
(5 trials), so ≈ 40 GPU-hours, ≈ 10 h wall-clock on four GPUs. Storage ≈ 30 MB per trial.

Protocol is byte-identical to the unseen14 campaign (cfg 7.5, fi 3, explicit tags, sphere
solver, planning, factor 1.5, skip-anchor 4, ik-streak 5, seed 42, res 512), so trials pair by
`scene_seed` with `reports/closedloop_unseen14/`.

### Scoring

```bash
python scripts/score_sim_replay.py --gpu 1        # -> reports/percep_simreplay/scores.json
```

≈ 44 s per trial for a direct depth row, ≈ 5 s for a cascade row. Rescoring reads only the
`.npz` dumps — **never re-run a rollout to change a metric.**

### The expert ceiling row

`GT RGB → expert` needs no rollout: run the expert on **real** frames of the same episodes.

```bash
DATA_TREE=$PWD/data/rlbench_unseen_tasks_512_clean \
EPS_FILE=/tmp/eps.txt SPLIT_TAG=_simreplay5 \
python scripts/probe_depth_baselines.py da2metric 20
```

where `/tmp/eps.txt` is a comma-separated list like
`close_microwave/variation0/episodes/episode0,...`. Output lands in `reports/external/`.
**Run this first** — see §9.4; it decides whether the cascade rows can be read at all.

### The dump

One `.npz` per trial, ≈ 30 MB compressed at 512²:

| key | shape | what |
|---|---|---|
| `canvas` | `[n_replan, 4T, H, W, 3]` u8 | the raw generated canvas, all segments |
| `sim_depth` | `[n_step, V, H, W]` f16 | metric depth after each executed step |
| `sim_mask` | `[n_step, V, H, W]` u16 | handle map |
| `sim_rgb` | `[n_step, V, H, W, 3]` u8 | **lossless** RGB — the target for the future-RGB row |
| `cmd`, `ach` | `[n_step, 8]` f32 | commanded vs **achieved** end-effector pose |
| `step`, `replan`, `k`, `is_ramp` | `[n_step]` | index bookkeeping (§8) |
| `extrinsics`, `intrinsics` | `[V,4,4]`, `[V,3,3]` | the simulator's own, read back live |

Raw, never through the video writer: the depth codec is a colour path and H.264 chroma
subsampling rewrites metric depth silently. The same applies to the RGB row — an LPIPS scored
on mp4 output measures the codec as much as the model.

---

## 7. Reading the output

```
percep_arm7_depth_close_microwave  absrel  full 0.0475/0.0973  dyn 0.2424/0.6207 (14.8%)  gap 0.00058
                                           └ model/floor ┘     └ model/floor ┘   └frac┘   └ exec gap ┘
```

**`full` vs `dyn`.** The camera is fixed within an episode, so 84–88 % of pixels never move and
a full-frame average is mostly a region where "assume nothing changed" is *exactly right*. The
dynamic region is pixels whose **counterfactual** ground truth moved more than 1 cm relative to
the chunk's reference render — the arm, the object it moves, and the occlusion they carry. 1 cm
is the codec's resolution scale, not a tuned threshold. **Table 2 reports the dynamic region.**

**`floor` = copy-anchor.** The model's own anchor frame copied to every step, scored through the
identical pipeline — "assume the scene is static". Not a rival model: the bar below which a
number means nothing. Every counterfactual metric can be passed by a frozen prediction, and this
is what catches that. A cascade gets its own floor: the expert run on the model's *anchor RGB*.

**`gap` = exec gap**, median distance between commanded and **achieved** end-effector position.
The imagined depth belongs to the pose the model DREW; the render belongs to the pose the
controller REACHED. That gap is the protocol's noise floor. It is reported per horizon bin
because it *grows*:

| k | exec gap (one trial) |
|---|---|
| 1–9 | 0.5 mm |
| 10–19 | 6 mm |
| 20–30 | **118 mm** |

Under GT replay it is 0.3 mm throughout, so this is not the controller — the model's
far-horizon poses get progressively harder to execute. **Headline numbers come from bins where
the gap is still millimetric**; far bins go in an appendix with the floor stated. Cells whose
replay fails the gate are left **blank**, not interpreted.

`k` is the index into the generated chunk — frame *k* of the imagined future, and equally the
*k*-th executed step since the last replan. At `frame_interval=3` and 20 Hz, one *k* = 0.15 s.

### Frames dropped, and why neither is a choice

- **`k == 0`** — the anchor, handed over as ground truth. Scoring it is free marks.
- **`k < 4` (`is_ramp`)** — `skip_anchor_frames=4` overwrites the start of every chunk with
  `ramp_from_current`, because the VAE round-trips the near-vertical home pose with the approach
  axis flipped ~174° (the "wiggle at the start of every rollout"). Those executed poses are not
  what the model drew.

---

## 8. Reference numbers from this machine

Arm 7 and arm 0 at 4,000 steps, four RLBench tasks, 5 trials, clean domain, depth only.
`reports/percep_simreplay/scores.json`; see its `README.md`.

| route | full model/floor | dyn model/floor |
|---|---|---|
| arm 7 direct co-generation (4-task mean) | **0.0598** / 0.0992 | **0.3238** / 0.7051 |
| arm 7 cascade (RGB → DA2-metric) | 0.3762 / 0.4501 | 0.4192 / 0.6985 |
| arm 0 cascade (RGB → DA2-metric) | 0.3729 / 0.4528 | 0.4504 / 0.6903 |
| DA2-metric on **real** frames | 0.3947 raw / 0.1109 median-aligned (ρ = +0.980) | — |

`exec_gap` medians 0.6–11 mm. Use these to check a new machine reproduces the arm 7 depth row
before trusting arms 1–6.

## 9. Gotchas

1. **Never quote a full-frame number alone.** One unrepresentative trial had the model at 0.0630
   against a 0.0644 floor and it looked worthless; over four tasks it beats the full-frame floor
   by ~40 % and the dynamic-region floor by 54 %. Single trials swing hard.
2. **Never quote any number without its floor**, and never a far-horizon number without the exec
   gap. See §7.
3. **`percep_arm0_video_*` is NOT arm 0's perception.** Arm 0 is `video+action@1.0` and cannot be
   prompted for depth. Those rows are `arm 0's RGB + an external expert`. Set the cascade aside
   and arm 0 has no perception number, by construction. The same holds for every
   (arm, modality) pair outside that arm's native set.
4. **The cascade rows are saturated by the expert, not by the design.** DA V2 Metric scores
   AbsRel **0.395 raw on REAL frames** of these tasks (ρ = +0.980, median-aligned 0.111) — its
   structure is fine, its metric scale is broken on synthetic renders. Both cascades land near
   0.38 regardless of whose RGB they read, so those rows currently measure a zero-shot domain
   gap. **Run the expert-ceiling row first**; if the ceiling is already worse than an arm's
   direct row, the cascade proves nothing about sequencing. The expert is presently given
   **zero** free parameters; `Tables/tab_percep.tex`'s convention would grant a two-parameter
   affine fit **to the anchor frame only**, which is NOT yet implemented and would move it
   toward ~0.11. Report both columns; the conclusion flips between them.
5. **Fine-tuned vs zero-shot.** Our arms are trained on exactly this render distribution; the
   experts are not. A cascade row bounds how hard the task is; it does not rank the designs. The
   only clean fix is training a `video+depth` expert on the same tree — no arm in the ladder
   does (arms 0–8 are all `X+action`).
6. **The rollout harness is the CLEAN domain.** `reset_to_new_demo` uses vanilla RLBench with
   fixed cameras; Colosseum randomization is not wired in, though training used it. This
   pipeline cannot fill randomized-domain rows, and RGB/LPIPS partly measures appearance shift.
7. **Check the GT-replay gate first.** A 1-trial `--gt-replay` of `close_microwave` under
   `planning`/factor 1.5/fi 3 timed out at 0 % while arm7_4k scores 85 % on the same task. Not
   yet chased. If the ground-truth ceiling is broken, no model number above it is interpretable.
8. **`--dump-percep` costs sim time on EVERY step** (three extra renders × two views). Do not
   leave it on for a success-rate campaign.
9. **`close_box` on arm 7 currently has 4 valid trials, not 5** (the `†` in Table 2). Every
   method must use the same trial count per task; rerun before the final comparison.

## 10. Not implemented — the remaining work

In priority order. Each is needed for a block of Table 2 that is currently blank.

1. **Segmentation scoring.** `scripts/score_sim_replay.py` has depth and normal branches; the
   seg branch is missing. It needs the live scene-role LUT that `eval/rollout.py:240` already
   builds for anchoring (it hard-fails on unmapped handles, which is the one place seg can go
   silently wrong), applied to `sim_mask`, then `scene_role_iou`.
2. **Future-RGB scoring.** `sim_rgb` is now dumped losslessly; the scorer has no LPIPS branch
   yet. This unlocks the eight-row RGB block — the only block where all arms including arm 0
   compare symmetrically, with no external model involved.
3. **ManiSkill3 `pull_cube`.** `rollout_env_maniskill.py` takes `extra_renders`, but
   `eval/rollout_maniskill.py`'s loop does not write the dump. Port the ~20 lines from
   `eval/rollout.py`. Until then every ManiSkill cell and `All avg.` stays blank.
4. **The anchor-frame affine column for cascades.** §9.4.
5. **Normal and segmentation expert rows** (Lotus-G, CLIPSeg) in the scorer's cascade branch,
   which currently only knows depth.
