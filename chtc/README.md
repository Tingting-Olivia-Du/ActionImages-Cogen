# Training the fusion arm on CHTC (UW-Madison)

Continues the **10-segment full-modality canvas** arm
(`video+depth+segmentation+normal+action`, both views) on CHTC GPU Lab, because the origin host's
eight cards are persistently held by other tenants and the arm is under-trained.

Patterned on `/workspace/ttdu/openpi/chtc`, with three differences forced by this workload:
it self-resumes across evictions, it keeps the global batch fixed regardless of how many GPUs the
scheduler grants, and it selects GPUs by **memory** rather than by capability.

---

## 1. The two decisions you asked about

### Resume from 5k, or restart? — **Warm-start from `step5000.ckpt`, fresh optimiser.**

Three options existed; this is why the middle one wins.

| | Keeps the 5k? | Card count | Verdict |
|---|---|---|---|
| DeepSpeed **resume** from `checkpoint-5000` | yes, optimiser included | **exactly 4, same node** | ✗ impossible here |
| **Warm-start from `step5000.ckpt`** (weights only) | yes, weights | **any** | ✓ **use this** |
| Restart from ActionImages `step125750` | no | any | ✗ throws away 48 GPU-hours |

The blocker for a true resume is concrete: `checkpoint-5000/global_step5000/` contains
`bf16_zero_pp_rank_0..3_optim_states.pt` — the ZeRO-2 optimiser state sharded **four** ways.
DeepSpeed cannot load four shards into one rank, and you expect to be granted one card. A resume
is therefore off the table, not merely inconvenient.

Warm-starting from the 12 GB `step5000.ckpt` keeps every weight the 5,000 steps produced and costs
only the Adam moments, which at a constant `5e-7` rebuild within the warmup. Use a **new**
`--output_dir`: `train.py` refuses a weights-only resume into a populated directory (it would
restart the step counter at 0 and overwrite the existing checkpoints), and `train_fusion.sh`
prints which of warm-start/resume the weights actually came from — read that line.

### Is F1 the main experiment? — **Yes, and F0 is the control it needs.**

`fusion0` = schedule **F1**, the arm whose result the paper is about. F1 is what trains the
`absent` role — 45% of its samples withhold the non-RGB modalities entirely, which is both the
deployment condition and the only setting where cross-modal completion is a prediction problem
rather than copying an anchor.

But **F1 alone is not a publishable main result.** The obvious reviewer question is whether the
*canvas* did it or the *dropout* did it, and `fusion0-anchor` (F0: same ten-segment canvas, every
segment keeps its anchor, no dropout) is the only thing that answers it. Budget for both:

```bash
condor_submit chtc/train.sub                                   # F1  -- priority
condor_submit chtc/train.sub arm=fusion0-anchor exp=f0ctl      # F0  -- the control
```

If only one can run, run F1 — but report it as preliminary until F0 exists.

---

## 2. Why one GPU is fine, and which GPUs qualify

Measured on the origin host: **36.6 GB** allocator peak at four ranks (n=5000 steps, from
`mem/peak_gb`). ZeRO-2 shards gradients, so fewer ranks means a larger per-card peak:

| ranks | estimated peak | L40/L40S 45 GB | A100-80 / H100 / H200 |
|---|---|---|---|
| 4 | 38 GB (measured) | ✓ | ✓ |
| 2 | 41–44 GB | marginal | ✓ |
| **1** | **47–56 GB** | **✗** | **✓** |

(The range is because DeepSpeed may keep bf16 or fp32 gradient partitions; the checkpoint file
names suggest bf16, the wider figure is the upper bound. **Check `mem/peak_gb` in W&B on the first
job** and treat the table as an estimate until then.)

So the submit file asks for `gpus_minimum_memory = 80000` and **does not** set
`gpus_minimum_capability = 9.0`. Restricting to 9.0 would limit you to H100+H200 (24 cards) and
exclude the 8 A100-80GB for no benefit — this job is memory-bound, not SM-feature-bound. A100 is
roughly 1.6x slower than H100 here and still far faster than the origin host. Widening the pool
from 24 to 32 cards matters more than per-card speed when a single card is hard to get.

**Throughput estimate:** ~35–45 s/step on one H100 with accumulation (the origin host did 35 s/step
on four RTX 6000 Ada). A 7-day `long` job is therefore ~14–17k steps. Reaching 20k takes two
submissions — which is fine, because the job self-resumes (§4).

---

## 3. Batch and learning rate

**The global batch is pinned to 4 no matter how many cards you get.** `train_fusion.sh` derives
`GRAD_ACCUM = 4 / NPROC`, so 4 cards → accum 1, 2 cards → accum 2, 1 card → accum 4. It prints
`global batch = NPROC(n) x per_device(1) x accum(a) = 4` — if that does not say 4, stop.

This matters because batch is part of the recipe: steps 1–5,000 ran at 4, and a continuation at a
different batch is a different experiment.

### On lowering the learning rate — the curves say **don't**

You suspected from the W&B curves that with batch 1 a warmup to below `3e-7` might be better. The
reasoning is right for a genuine batch-1 run (4x noisier gradients warrant smaller steps), but
accumulation means the global batch never drops to 1, so it does not apply.

And the recorded curves argue the opposite, from the 5,000-step run at `5e-7`:

| phase | lr | grad_norm median | p90 | max | loss median |
|---|---|---|---|---|---|
| 0–10% (warmup) | 5e-9 → | 0.716 | 2.640 | 6.567 | 0.0801 |
| 10–20% | 2.55e-7 | 0.339 | 0.731 | 0.866 | 0.0577 |
| 20–50% | 5e-7 | 0.295 | 0.522 | 3.085 | 0.0483 |
| 50–80% | 5e-7 | 0.288 | 0.611 | 1.811 | 0.0445 |
| 80–100% | 5e-7 | 0.269 | 0.504 | 0.821 | 0.0423 |

- `grad_norm` **falls monotonically** (0.716 → 0.269) and only **5.0%** of steps touch the
  `max_grad_norm = 1.0` clip — concentrated in warmup (p90 2.64 in the first tenth, 0.504 in the
  last). **The spikes you saw are almost certainly the warmup phase**, not the steady-state rate.
- Loss is **still descending at the end** (0.0445 → 0.0423). No plateau — consistent with your
  "under-trained" read.
- Max `grad_norm` in the final fifth is 0.821, **below** the clip. Nothing is unstable.

A rate that produces a shrinking gradient norm, almost no clipping, and a still-falling loss is
**conservative, not too high.** Lowering it to `3e-7` would slow a run whose problem is
insufficient training.

**Default: keep `5e-7`** (continuity with steps 1–5,000). If you would rather spend the CHTC
budget reaching convergence sooner, `1e-6` is the option the curves support — the median gradient
norm would land near 0.55, still well under the clip — and a fresh optimiser is the natural place
to change it. That is a recipe change, so put it in the run name:

```bash
condor_submit chtc/train.sub exp=f1_lr1e6 \
  extra_args="--learning_rate 1e-6"      # passed through to train_fusion.sh as EXTRA_ARGS
```

Warmup stays at 1,000 steps. It was 20% of a 5,000-step run and becomes 5–7% of a 20k one, which
is appropriate; and a fresh Adam state wants a warmup regardless.

---

## 4. What the wrapper does that the openpi one does not

`chtc/run_train.sh`:

1. **Self-resumes.** On start it looks for `/staging/tdu35/fusion/<exp>.tar` and unpacks it, so an
   evicted 7-day job continues instead of restarting. `+is_resumable = true` (the openpi template
   has `false`) declares this to HTCondor and also makes the job eligible for backfill capacity.
2. **Prunes before bundling.** Only the newest `global_step*` is kept; the rest are deleted. A
   resume tip is ~120 GB and older ones can never be resumed from anyway, so without this the
   bundle is too large to transfer.
3. **Applies the DeepSpeed bf16 patch.** Upstream DeepSpeed does not skip non-finite gradients
   under bf16 — it surfaces as a `Got Long` crash rather than a skipped step. `pip install`
   does not carry the fix, so it is re-applied at job start and the job **exits 5 if it fails**
   rather than training without it. (`chtc/patch_deepspeed_bf16_overflow.py`, copied from
   `/workspace/ttdu/ActionImages/scripts/`, the aligned source where it was verified.)
4. **Extracts then deletes each tarball**, so peak disk is one copy of the 137 GB dataset rather
   than two.

---

## 5. Run it

```bash
# --- once: build and push the image (needs Docker locally) ---
docker build -t aicogen_train -f chtc/train.Dockerfile .
docker tag aicogen_train <dockerhub_user>/aicogen_train:fusion
docker push <dockerhub_user>/aicogen_train:fusion
#   then set container_image in chtc/train.sub

# --- once: stage the inputs (run ON the access point, pulls from HF inside the cluster) ---
ssh tdu35@ap2002.chtc.wisc.edu
bash chtc/stage_to_chtc.sh        # dataset 137 GB + base model 32 GB + step5000.ckpt 12 GB

# --- every submission ---
export WANDB_API_KEY=...          # from a chmod-600 file; getenv=True passes it in
condor_submit chtc/train.sub                                # F1, 20k steps
condor_submit chtc/train.sub arm=fusion0-anchor exp=f0ctl   # F0 control

condor_q                          # status
condor_tail -f <cluster_id>       # stream stdout

# --- bring results back and evaluate ---
bash chtc/pull_checkpoints.sh fusion0_<cluster_id>
OUT_DIR=outputs/fusion0_<cluster_id> TAG=chtc20k bash scripts/wait_and_eval_all.sh
```

### Check these three lines in the job's first minute

```
global batch = NPROC(1) x per_device(1) x accum(4) = 4      <- must say 4
weights: warm-start from .../step5000.ckpt                  <- NOT "RESUME from"
[selfgen] scene_roles OK: zero `unknown` pixels             <- data staged correctly
```

---

## 6. Credentials — do not commit any

`chtc/train.sub` is in git and submit files are plain text. `getenv = True` passes
`WANDB_API_KEY` in from your submit shell; nothing else is needed, and evaluation needs no
credentials at all.

**The lab `.env` at `/workspace/ttdu/.env` contains a GitHub PAT, a W&B key, an HF token and your
CHTC password in plaintext.** It lives outside this repository, so a `git push` of the repo will
not include it — but do not copy it into the tree or into the Docker image (`train.Dockerfile`
does `COPY . /app`, so anything in the working tree ends up in a pushed image). Consider switching
CHTC to SSH keys and rotating that password if it has ever been pasted anywhere.

---

## 7. GPU Lab limits (from CHTC docs, as of 2025)

| Job length | Max runtime | Per-user GPU cap |
|---|---|---|
| short | 12 h | 2/3 of GPU Lab |
| medium | 24 h | 1/3 of GPU Lab |
| **long** (used here) | **7 days** | **4 GPUs** |

| GPU | VRAM | Capability | Count | Usable at 1 rank |
|---|---|---|---|---|
| L40 / L40S | 45 GB | 8.9 | 46 | **no** (needs 47–56) |
| A100 | 80 GB | 8.0 | 8 | yes |
| H100 | 80 GB | 9.0 | 8 | yes |
| H200 | 141 GB | 9.0 | 16 | yes, most headroom |

Verify against <https://chtc.cs.wisc.edu/uw-research-computing/gpu-jobs> before relying on the
counts; the openpi README's table is where these came from and hardware changes.
