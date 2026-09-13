# HANDOFF — fusion arm (full-modality canvas), as of 2026-09-13

For picking this up in a fresh session on another machine. **Read §1 and §2 first**: §1 is what
the work is, §2 is the one thing blocking it.

Deeper documents, in the order you will want them:

| Document | What it is |
|---|---|
| `chtc/README.md` | **CHTC training** — the immediate task. The two design decisions and why. |
| `HANDOFF_EVAL.md` | **Evaluation on another machine** — what to upload (13 GB, not 150), flag traps, how to read the numbers. |
| `paper_fusion/NUMBERS.md` | **Every measured number with provenance**, and what is still PENDING. |
| `paper_fusion/iclr2026_fusion.tex` | The manuscript. 7 red `\TODO{}` slots = the unmeasured results. |
| `paper/EXPERIMENT_STATUS_zh_aug29.md` §46.1–46.13 | Running lab log for this direction, newest last. |

---

## 1. What this is, in one page

Prior arms in this repo pack **four** segments into the latent canvas (one visual modality +
action, two views) and let a prompt tag select the modality. A sample draws one template and a
batch may not mix templates, so **two modalities never co-exist in a training step**.

Worse, the reading of those arms is confounded. Every segment that is not fully supplied still
gets its own clean first latent frame — an **anchor** — and the repo's own tag-swap ablation found
the anchor is what the model acts on: swapping the prompt tag moves metrics 1–2%, **removing the
anchor costs ~9×**. So "the prompt selects the modality" is mostly "the model copies that
modality's anchor", and the model is only ever scored in a regime deployment cannot supply — a
robot has RGB and nothing else.

This direction changes two things:

1. **One canvas, every modality.** Ten segments, `V0 D0 S0 N0 A0 | V1 D1 S1 N1 A1`, 110 latent
   frames, 28,160 tokens. Concatenation is along the latent *time* axis, so the model sees one
   110-frame video and must infer from position and content which stretch is which. Every pair of
   modalities is now inside the same dense attention.
2. **A third conditioning role: `absent`.** Beyond "fully given" and "anchored", a modality may
   receive *no* clean frame, so it must come from the others. A per-sample schedule (the **F
   axis**, presets F0/F1/F2) spends 45% of F1's samples with the non-RGB modalities absent —
   simultaneously the deployment condition and the only setting where cross-modal completion is a
   prediction problem rather than a copy.

**The headline object is the mutual-completion matrix**: each modality's quality against the
number of co-present modalities. Monotone ⇒ they supervise each other. **Flat ⇒ they merely share
a decoder**, and that outcome is pre-registered (paper §5.1) as a reason to *stop* rather than
scale — the same way an earlier privileged-distillation plan in this repo died when its assumed
teacher–student gap turned out not to exist.

---

## 2. Where it stands, and the one blocker

**Blocker: the origin host's 8 GPUs are persistently held by outside tenants.** Nothing is wrong
with the code. Measured from the origin host: `ap2002.chtc.wisc.edu` is unreachable on both :22 and
:443 while `github.com:22` is fine — so it is CHTC filtering by source IP, not a local egress
block. That machine is on UMD's VPN and two full-tunnel VPNs do not coexist. Hence the split:

| Where | Does what |
|---|---|
| your laptop, on **UW VPN** | `ssh tdu35@ap2002`, `condor_submit`, `condor_q`, `condor_tail` — a handful of commands |
| **inside CHTC** | `bash chtc/stage_to_chtc.sh` pulls 137 GB from HF into `/staging` — never through your laptop |
| any machine with Docker | `docker build`/`push` — Docker Hub is public, no UW network needed |

### Done

- The canvas and the F axis are implemented and unit-tested. `tests/test_fusion_axis.py` (7
  assertions) includes **train/eval mask parity for all five regimes** — without it a checkpoint
  can be scored under a conditioning it never trained on, silently.
- **Two arms trained.** Base tree (788 episodes): stopped at step 3,202 to free GPUs.
  **Scaled tree (3,960 episodes): 5,000 steps, `TRAIN_EXIT=0`, all 5 checkpoints on disk.** That
  second one is the arm of record: `outputs/fusion0__seed42_fi3_512_aug_wide_sr_F1`.
- **Dataset scaled 5.0×** and verified per-modality against the smaller tree (identical
  statistics, so the trees differ in scale alone). See §4 for the bug this surfaced.
- **Strong training-time evidence, no GPU needed** (`paper_fusion/NUMBERS.md`): the
  `absent/anchored` loss ratio **falls for four of five modalities** over training (depth
  1.776→1.266, action 1.374→0.860); action's full-run ratio is **0.945 — below 1**, i.e. no harder
  without its own anchor. **Rotary-position extrapolation is ruled out.** `normal` is the one
  modality getting *worse*, which matters because it is analytically a function of depth.
- Manuscript written with real numbers where measured and 7 explicit `\TODO{}` where not.

### Not done

- **The mutual-completion matrix, the closed-loop action number, and the F0 control.** All three
  are GPU-blocked, not code-blocked. `scripts/wait_and_eval_all.sh` is armed and waiting.
- The arm is **under-trained**: loss was still descending at step 5,000 with no plateau. Hence the
  CHTC continuation to 20k.

---

## 3. Next actions, in order

1. **Request `/staging` quota if you have not** — ≥500 GB (137 dataset + 32 base model + 12
   warm-start + checkpoints). <https://chtc.cs.wisc.edu/uw-research-computing/quota-request>.
   Approval takes a day or two, so send it before anything else.
2. **Push this repo to GitHub**, then follow `chtc/README.md` §5. Submit **F1 first**
   (`condor_submit chtc/train.sub`), then the **F0 control**
   (`arm=fusion0-anchor exp=f0ctl`). F1 alone is not a publishable main result — the reviewer
   question "was it the canvas or the dropout?" is answerable only with F0.
3. **Check three lines in the job's first minute** (they catch the three ways this goes silently
   wrong):
   ```
   global batch = NPROC(1) x per_device(1) x accum(4) = 4      <- must be 4
   weights: warm-start from .../step5000.ckpt                  <- NOT "RESUME from"
   [selfgen] scene_roles OK: zero `unknown` pixels             <- data staged correctly
   ```
4. **Evaluate whatever checkpoint exists**, on any GPU you can get — `HANDOFF_EVAL.md`. The
   offline stages need no simulator; only closed-loop does.
5. **Fill the `\TODO{}` slots** in `paper_fusion/iclr2026_fusion.tex` as numbers land, adding a
   MEASURED row to `paper_fusion/NUMBERS.md` for each. That directory's rule: **no number in the
   `.tex` without a provenance row.**

---

## 4. Decisions already made, with the reasoning (do not silently reverse these)

- **Warm-start from `step5000.ckpt`, not DeepSpeed resume.** `checkpoint-5000/global_step5000/`
  holds `bf16_zero_pp_rank_0..3` — optimiser state sharded **four** ways, and DeepSpeed cannot load
  four shards into one rank. With CHTC likely granting one card, a true resume is impossible, not
  merely awkward. Warm-starting keeps every weight and loses only Adam moments.
- **Global batch pinned to 4 via accumulation.** `GRAD_ACCUM = 4 / NPROC`. Batch is part of the
  recipe (steps 1–5,000 ran at 4); this decouples it from the allocation.
- **Keep lr `5e-7`.** The curves say it is conservative, not high: grad_norm falls 0.716→0.269,
  only 5.0% of steps clip and those are in warmup, final-fifth max 0.821 is *below* the clip, and
  loss is still descending. `1e-6` is the option the curves support if you want convergence
  sooner; `3e-7` would slow a run whose problem is under-training.
- **Select GPUs by memory, not capability.** One rank needs 47–56 GB (extrapolated from 36.6 GB at
  four ranks), so L40S at 45 GB is out. `gpus_minimum_memory=80000` with **no** `capability=9.0`,
  because 9.0 would exclude 8 A100-80GB for nothing — memory-bound, not SM-feature-bound.
- **T1 (variation-0) is excluded from evaluation.** Only T2 (unseen variation, seen task) and T3
  (never-trained task). Consequence: **no in-distribution baseline**, so T2/T3 are absolute
  quality, not "degraded by X% versus seen". T2-vs-T3 and the matrix (a within-episode contrast)
  survive.
- **`normal` stays in the canvas** even though it is analytically a function of depth. It is the
  sharpest probe — a deterministic right answer exists — not a fifth information source. Any text
  implying it adds information is wrong.
- **Action dilution is accepted and stated, not tuned away.** Action is 26.3% of predicted latent
  positions here against 50% in a four-segment template. Do **not** add per-modality loss weights:
  that is a new free parameter and makes the arm incomparable to every other.

---

## 5. Open questions for you

1. **Two ~108 GB optimiser tips are reclaimable** if those arms will not be continued:
   `arm7u/checkpoint-10000/global_step10000` and
   `fusion0__..._wide_sr_F1/checkpoint-5000/global_step5000`. Deleting either forecloses resuming
   that arm — and this repo has a precedent for wanting that (arm6@10000 was continued 3,000 steps
   to make arm6m1). Disk was 313 GB free at last check, so there is no urgency.
   **Keep `step5000.ckpt` regardless** — that is the warm start CHTC needs.
2. **How many steps?** `chtc/train.sub` defaults to 20,000 (from 5,000). One 7-day job is ~14–17k
   steps on an H100, so 20k is two submissions with self-resume.
3. **Is the base-tree arm (stopped at 3,202) worth finishing?** Per the current scope decision it
   is not a comparison target. If that changes, compare `checkpoint-3000` to `checkpoint-3000`,
   never `wide@5000` to `base@3000` — `ARM7_RESULTS.md` has a headline number that reversed
   because of exactly that error.

---

## 6. Traps that have already cost time

Repo-specific, all of them hit during this work:

- **`grep` here is ugrep.** Its `-q -v` semantics differ from GNU grep and can silently take the
  wrong branch in a shell script.
- **Never edit a running `.sh`.** Bash reads a script by byte offset as it executes; editing one
  mid-run has made a live shell resume inside a comment and restart a job.
- **`pkill -f <pattern>` matches the shell running it** when the pattern appears in your own
  command line. This killed a watcher's own launcher once (exit 144) and, another time, made a
  safety check refuse to proceed. Kill by exact PID and exclude `/bin/bash -c` entries.
- **A non-empty `--output_dir` makes training resume from it** and silently ignore the warm-start
  checkpoint. `train_fusion.sh` prints which one the weights really came from — read that line.
- **Do not install `flash-attn`.** The start-up line `Flash Attention library "flash_attn" not
  found` is not a warning: PyTorch 2.6's SDPA already dispatches to FlashAttention for this shape
  (verified, backend 1, 54.9 ms vs 88.1 ms mem-efficient). The package buys nothing.
- **A layout transcribed twice will drift, silently.** Two evaluation paths each carried their own
  copy of the canvas layout; after `absent` was added, one mislabelled which modality a metric
  belonged to and the other **decoded a robot pose out of a segmentation map and reported it as a
  policy**. Neither raised. Both now call `templates.inference_plan` — one function. Treat any
  "mirrors X exactly" comment as a latent defect.
- **`nvidia-smi` polling misses transient peaks.** A 4 s poll under-reported a peak by 4.5 GB and
  in the dangerous direction. `train.py` now logs `mem/peak_gb` from the allocator; use that.
- **The unseen-task tree names episodes by random seed**, not `episode0`. A hardcoded `episode0`
  matches nothing there and the whole tier is skipped with a one-line `SKIP`.
- **Scaling data surfaces annotation gaps that smaller trees hide.** The room ceiling appears in 3
  of 4,200 episodes and never in the 788-episode tree; a per-task handle union then propagated
  those 3 to all 540 episodes of two tasks, and the start-up check would have rejected the entire
  tree. Fixing it required first proving the smaller tree was bit-identically unaffected.

---

## 7. Credentials

`/workspace/ttdu/.env` on the origin host holds a GitHub PAT, a W&B key, an HF token **and a
plaintext CHTC password**. It sits outside this repository, so pushing the repo does not include
it — but `chtc/train.Dockerfile` does `COPY . /app`, so anything you copy into the working tree
ends up in a pushed image. Keep it out. Evaluation needs no credentials; training needs only
`WANDB_API_KEY`, passed through by `getenv = True`. Consider SSH keys for CHTC and rotating that
password.
