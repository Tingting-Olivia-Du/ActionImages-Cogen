# Robustness plan, 2026-08-29

## Status at ~7h11m (checked 11:11 UTC, jobs launched 03:57 UTC) — welcome back

Both jobs have been healthy the entire window: no Traceback/OOM/NaN in either log, all 5
expected processes still alive, checkpoints landing on schedule. Neither will be *done* when you
read this — both were always going to run past 8h (see the per-track ETA math below) — but both
have made solid, checkable progress.

**Track A (M1 mixture continuation, GPUs 5,7):** step **1275/3000** (42.5%), ~7h10m elapsed,
~20.2s/it steady. Warmup (1000 steps) completed around the 6h09m checkpoint, LR has been at the
full `5e-7` since. Loss band the whole run: roughly **0.03-0.10**, no drift, no spikes — visually
indistinguishable in shape from arm6's own training loss. Checkpoints saved so far:
`checkpoint-500`, `checkpoint-1000` (both landed clean). `checkpoint-1500` should land within the
next ~10-15 minutes of wall time at this rate. At ~20.5s/it the remaining 1725 steps are another
**~9.8h** — total run will finish around **~17h** after launch (~21:00 UTC today). Nothing to do
here yet; the interesting analysis (does `FIFI` stay stable across these checkpoints the way it
didn't for arm6 under M2?) needs `scripts/modality_mode_grid.py` run against
`checkpoint-{500,1000,1500,...}` once there are enough steps to see a trend — see the "What to
check" note further down.

**Track B (closed-loop campaign then tag-swap, GPU 0):** **70/100 rollouts** done, `errors: []`
throughout. Per-task success so far (n so far / 20 target each):

| task | success | n so far |
|---|---|---|
| close_box | 5/20 | 20 (task complete) |
| close_drawer | 9/20 | 20 (task complete) |
| push_buttons | 12/20 | 20 (task complete) |
| meat_off_grill | 5/10 | 10 (in progress) |
| open_drawer | — | 0 (not started) |

Overall so far: 31/70 = 44.3%. **These are provisional** — meat_off_grill and open_drawer aren't
finished, and per-task rates on n=20 (once complete) are the number to compare against
`tab:closedloop`'s official/arm4 columns, not this running total. At the observed ~6.2 min/rollout,
the remaining 30 rollouts (10 meat_off_grill + 20 open_drawer) are **~3.1h** away, then stage 2
(tag-swap across 4 episodes) is a further **~2.4h** — Track B should fully wrap up roughly **~5.5h**
from now (~16:40 UTC).

**Bottom line:** nothing crashed, nothing needed relaunching, both tracks are on the ETA originally
projected. Re-run the "How to check in when you're back" commands at the bottom of this file for
the latest numbers — they'll have moved on since this snapshot. I'll keep doing periodic
unattended health checks for a bit longer in case you're still away, tapering off since 8h has
now effectively elapsed.

---

Written while you're away for ~8h. Goal: make the draft's central claims harder to break, not
just add more results. Two jobs are running now on the three GPUs that were idle (0, 5, 7) when
I checked; GPUs 1/2/3/4/6 were busy with other tenants' jobs and were left alone. Everything
below is designed to be safe to run unattended: no destructive ops, no new output dirs that
collide with existing ones, disk-bounded checkpoint retention, and every training run is an
`--init_ckpt_path` warm-start into a **fresh** output directory (never a resume into arm6's own
dir), so a crash can't corrupt arm6's existing checkpoints.

## Where the draft is thin (why these experiments, not others)

1. **arm6 (the headline four-stream model) has no closed-loop table.** `tab:closedloop` in the
   draft only reports arm4 (action+depth, two streams). arm6 has only ever gotten an n=4/task
   smoke test (`reports/closedloop/rollout_arm6_9000.json`, and only at step 9000, not the final
   10000). If a reviewer asks "does the versatile model still act well in closed loop, or did
   you only check open-loop trajectory decoding (tab:action)?", the draft currently has no
   answer at the same evidentiary bar (n=20/task, paired McNemar) as arm4 got.

2. **The "prompt selects the modality" claim (sec:templates) is contradicted by the team's own
   unpublished ablation.** `scripts/tag_swap_ablation.py` already ran once (step 9000, one
   episode, `open_drawer`): holding the anchor frame fixed at a target modality and lying about
   the prompt tag moves the metric by ~1-2%, while `f0f0` (anchor removed, tag correct) collapses
   it by ~9x (depth AbsRel 0.087 -> 0.816). That means **the anchor frame's visual content
   selects the modality, not the prompt tag** — the tag is close to decorative under `IIII`. This
   is not in the draft anywhere (checked: no mention of "tag swap" or this mechanism in
   `paper/draft_aug_28.md`), and section 3.4's framing ("announced in the text prompt") reads as
   though the tag is doing the selection. This needs to be either (a) reported honestly with the
   anchor-dominance finding and the claim in 3.4 qualified, or (b) the evidence needs to be solid
   enough (more than n=1 episode) to be sure it's real before deciding how to frame it. Right now
   it's n=1 episode at a non-final checkpoint — not solid enough to build a paper claim on either
   way.

3. **`sec:modegrid`'s central causal claim ("the plan distribution decides which questions the
   checkpoint can answer... a mixture that wants both must pay for both") has never been tested
   by actually intervening on the plan distribution.** Both M0 (discriminative-heavy) and M2
   (world-model-heavy, the shipped default) are extremes. `SEGMENTATION_SCENE_ROLES_PLAN.md`
   already defines an M1 = 45/10/45 "balanced" preset and flags it ("M1 仍值得作为中间点扫") as
   worth sweeping — and never runs it. Without a midpoint, "a mixture that wants both must pay
   for both" is a claim about two data points, not a curve. If M1 recovers `FIFI` stability while
   keeping `IIII` mostly intact, the trade is softer than the draft claims and the recipe should
   probably ship M1, not M2. If M1 fails the same way M2 does, that's a much stronger version of
   the "cannot have both on 788 episodes" claim.

None of these require a new codec or new data — they reuse existing, previously-debugged scripts
with different arguments. That's deliberate: an 8-hour unattended window is the wrong place to
debug a brand-new modality codec against the admission-gate standard the paper itself sets.

## What's running right now

### Track A — GPUs 5, 7 — does the plan-mixture trade-off move with a balanced mixture?

```
scripts/run_track_a_m1_continue.sh
  GPUS=5,7 SEED=42 STEPS=3000 CKPT_EVERY=500 SAVE_TOP_K=-1 \
  INIT_CKPT=outputs/.grid_pin/step10000.ckpt \
  PERCEPTION_MASK_MIX=M1 \
  OUT=outputs/arm6m1__seed42_fi3_512_aug_sr \
  bash scripts/train_arm.sh arm6
-> logs_arm6m1.txt
```

Same template mix as arm6 (`video+action@0.4,depth@0.2,segmentation@0.2,normal@0.2`), same seed,
data tree, resolution, frame interval, LR, warmup — the **only** thing that changed is
`perception_mask_mix`: M2 (90% `IIII` / 5% `FIII` / 5% `FIFI`) -> M1 (45/10/45), and the starting
point (arm6's own step10000, not the official step125750), so any difference in the `FIFI`/`IIII`
trade is attributable to the mixture and not to a different warm start or recipe. Fresh optimizer
state, fresh output dir — this is a warm-start `INIT`, not a `RESUME`, so it does not touch
arm6's own checkpoint directory or its DeepSpeed optimizer state at all.

- 3000 steps, checkpoints every 500 (6 checkpoints, ~13GB each, ~80GB total — cheap against the
  1.3TB free on `/workspace`).
- At arm6's own observed rate (~21s/it, 2 GPUs) this is **~17.5h end to end** — it will still be
  running when you're back, and that's fine. `CKPT_EVERY=500` means you get a checkpoint every
  ~2.9h to read progress from even mid-run.
- **What to check when you're back:** run `scripts/modality_mode_grid.py` (or the eval-harness
  equivalent) on `outputs/arm6m1__seed42_fi3_512_aug_sr/checkpoint-{500,1000,1500,...}` for the
  same 4 episodes `tab:crosstask` uses, and look at `FIFI` depth AbsRel across steps. If it stays
  well under arm6's `1.335` blowup on `open_drawer` while `IIII` AbsRel doesn't drift far from
  arm6's ~0.085-0.16 band, M1 is the better recipe and that's worth a rerun-from-scratch someday.
  If `FIFI` still blows up, you have a stronger two-point-isn't-enough-but-here's-a-third-point
  result for the "central trade" paragraph.

### Track B — GPU 0 — two stages, chained

```
scripts/run_track_b_closedloop_then_tagswap.sh -> logs_arm6_closedloop10000_and_tagswap.txt
```

**Stage 1 (running now):** full closed-loop campaign for arm6@step10000, same 5 tasks as
`tab:closedloop` (`close_box close_drawer push_buttons meat_off_grill open_drawer`), n=20/task
(100 rollouts total, paired scene seeds against the same protocol CAMPAIGN_200 used for arm4).
`eval/rollout.py` checkpoints its JSON after every trial, so
`reports/closedloop/rollout_arm6_10000.json` is readable mid-campaign, not just at the end.
Estimated from the n=4 smoke test's wall time (~5.6 min/rollout at 512²): **~9-10h** for all 100.
It will likely still be running at the 8h mark — that's fine, check the JSON's `results` list
length for progress. When it's done, you have a table directly comparable to `tab:closedloop`:
does the four-stream checkpoint still act as well (or better/worse) than the two-stream arm4 did,
against the same official-init baseline?

**Stage 2 (queued after stage 1 exits):** extends `scripts/tag_swap_ablation.py` from n=1 episode
(`open_drawer`, step 9000) to the same 4 held-in episodes `tab:crosstask` already uses
(`open_drawer`, `push_buttons`, `stack_wine`, `turn_tap`), at the **final** checkpoint (10000).
~12 generations/episode x ~180s = ~36min/episode, ~2.4h total. Writes
`reports/tag_swap/arm6_10000_<episode>/metrics.json` per episode. Run
`python scripts/tag_swap_report.py arm6_10000_<episode>` per episode when back, or aggregate all
4 into one table modeled on `tab:crosstask`'s per-episode rows. **This is the evidence that
decides whether sec:templates' "announced in the prompt" framing needs a caveat.**

## Why closed-loop runs before tag-swap on the same GPU

Stage 1 is the bigger, more obviously-missing piece (a whole table the draft needs), and its
JSON is readable incrementally; stage 2 is cheap and self-contained and loses nothing by starting
late. If GPU 0 only gets through ~70% of the closed-loop campaign by the time you're back, that's
still more informative than a half-finished tag-swap sweep would be, and stage 2 will pick up on
its own once stage 1's process exits.

## Safety notes

- GPUs 1, 2, 3, 4, 6 were at 22-45GB used / 0-100% util when I checked (other tenants) and were
  never touched. GPUs 0, 5, 7 were <1.2GB used / 0% util; re-checked immediately before each
  launch (this shared node has previously OOM'd a job on GPU 0 when another tenant's process
  landed there mid-run — see `logs_heldout_10000.txt` — so if Track B dies with a CUDA OOM
  traceback, it's very likely a transient contention issue, not a bug in the ablation script; just
  relaunch `scripts/run_track_b_closedloop_then_tagswap.sh` after checking `nvidia-smi`).
- Neither track writes into `outputs/arm6__seed42_fi3_512_aug_sr/` (arm6's own directory stays
  read-only throughout: Track A's `OUT=` is a sibling directory, Track B only reads
  `outputs/.grid_pin/step10000.ckpt` and `outputs/arm6.../checkpoint-10000/step10000.ckpt`).
- Track A's checkpoint budget (~80GB, 6 checkpoints, `SAVE_TOP_K=-1` i.e. keep-all-6) is bounded
  because `STEPS=3000` is bounded — it will not keep growing past 3000 steps. `/workspace` had
  1.3TB free at launch.
- Both logs (`logs_arm6m1.txt`, `logs_arm6_closedloop10000_and_tagswap.txt`) are being watched by
  a background Monitor for `Traceback`/`OOM`/error signatures during the first ~10 minutes after
  launch to catch a bad launch early; after that I'm relying on periodic wakeups plus the fact
  that both scripts exit non-silently (`TRAIN_EXIT=$?` / `rollout exit=$?`) into their log files.

## Explicitly NOT attempted, and why

**Idea #2 as originally framed (use generated depth/segmentation as a closed-loop success
verifier) needs Track B's rollout videos/frames to exist first** — there's nothing to verify
against yet. Once `reports/closedloop/rollout_arm6_10000.json` is populated, the natural next
step is: for each rollout's terminal frame, query the same checkpoint for `<video><scene-seg>`
under `FIII` (reads segmentation from the observed, not imagined, final RGB — this is exactly the
`FIII` row of `tab:versatility`, mIoU 0.354 at step 1500), then test whether a simple rule on the
`target`/`goal` role masks (e.g. overlap, or target-role pixel count crossing a threshold)
predicts the simulator's own ground-truth success label better than chance. This is a genuinely
new contribution angle (the paper currently only shows the model *can* generate segmentation, not
that the generation is *useful* for anything downstream) but the per-task success rule needs
per-task RLBench semantics that shouldn't be designed blind/unattended — that's a next-session
task once Track B's data exists, not a background job.

**Idea #1 as "add a new modality" (e.g. optical flow, tying directly into the FlowWAM related-work
citation and the unfilled "\todo{}" in the Unified Framework contribution bullet) was considered
and deferred**, not because it's a bad idea — it's probably the single best way to fill that
`\todo{}` — but because a new modality needs a new invertible codec that clears the same
admission-gate bar depth/normal/segmentation did (Table `tab:gate`), and that verification loop
needs a human in it. Building and shipping an unverified codec unattended risks violating the
paper's own methodological standard. Recommend as the first thing to scope next session; RLBench's
simulator can likely provide exact GT scene flow (not just RAFT-estimated flow) from per-object
poses + depth + camera motion, which would be strictly more rigorous than what FlowWAM itself
does and is a natural companion to the existing Barron/Gray-code depth codec.

**Idea #3 (real-robot deployment) was not attempted and should not be run unattended under any
circumstance** — physical actuation with no one present to hit an e-stop is a safety issue
regardless of how confident the sim-to-real transfer looks. This needs you (or someone) physically
present. Sim-side prep that *is* safe to do offline, if useful later: check whether the depth/
normal codecs' admission-gate fidelity holds on real camera intrinsics/distortion (not just the
RLBench pinhole model) using a handful of already-captured real RGB-D frames, if any exist. I did
not find a real-robot data path in this repo to test against, so this is a placeholder for next
session, not something run tonight.

## How to check in when you're back

```
tail -50 logs_arm6m1.txt                                   # Track A training progress
tail -50 logs_arm6_closedloop10000_and_tagswap.txt          # Track B progress
python -c "import json; d=json.load(open('reports/closedloop/rollout_arm6_10000.json')); print(len(d['results']), 'rollouts done'); print(d.get('summary'))"
ls outputs/arm6m1__seed42_fi3_512_aug_sr/checkpoint-*        # which steps have landed
nvidia-smi                                                   # confirm both still alive / no OOM
```

---

## Batch 2, 2026-08-29 22:00 UTC — official baseline + specialists + held-out harness

Track A (M1) and Track B (arm6@10000 closed-loop + tag-swap) both finished clean
(`TRAIN_EXIT=0`; `errors: []`, 100/100 rollouts; all 4 tag-swap episodes done). Full plan and
rationale for this next batch: `/root/.claude/plans/sunny-exploring-scone.md`. Comparing against
`paper/U-CoGen_experiment_plan.md` surfaced that the repo's *existing* official/arm4 closed-loop
reference data (`reports/closedloop/archive_pre_anchorfix/`) predates a `eval/rollout.py` fix
(mtime 2026-08-22, `--skip-anchor-frames` defaulting to 4) — i.e. **the numbers currently in
`tab:closedloop` were generated by a since-changed harness** and aren't valid to compare against
freshly-generated arm6 data. Per the user's call to use the official checkpoint (not arm0) as the
closed-loop control, a *fresh* official run under the current harness was required regardless.

Four things launched/queued (GPUs 0, 3, 4, 5, 7 were all idle when checked):

1. **`scripts/run_official_closedloop_postfix.sh`** (GPU 0, ~9-10h) — official step125750,
   n=20/task, same 5 tasks as arm6's campaign, but with the *official* protocol from
   `CAMPAIGN_200.md` (`cfg=10.0 frame_interval=4 prompt_tag_style=none`), **not**
   `queue_closedloop.sh`'s hardcoded fine-tuned-arm protocol (`cfg=7.5 fi=3 explicit`) — using the
   wrong one would run official out of its trained distribution and bias the comparison. Chains
   into `eval/aggregate_closedloop.py --compare` against `rollout_arm6_10000.json` on success.
   Log: `logs_official_closedloop_postfix.txt`.
2. **`scripts/run_specialist_depth.sh`** (GPUs 3,4, ~23h for 4000 steps) — `video+depth@1.0`,
   warm-started fresh from official step125750 like every other arm. `outputs/specialist_depth__seed42_fi3_512_aug_sr`.
3. **`scripts/run_specialist_seg.sh`** (GPUs 5,7, ~23h) — `video+segmentation@1.0`,
   `SEG_MODE=scene_roles` explicitly set (the fallback default is `referring`, which would not be
   comparable to arm6's seg protocol). `outputs/specialist_seg__seed42_fi3_512_aug_sr`.
   Both specialists checkpoint every 500 steps (STEPS=4000): the step~2000 checkpoint gives a
   depth/seg-update count matched to arm6's own 0.2×10000=2000 updates at its final checkpoint
   ("Panel B" in the experiment-plan doc), and step4000 is the specialist's own ceiling reading
   (arm6's own depth AbsRel already plateaus by step 4000: 0.087→0.086→0.085).
4. **`scripts/heldout_batch_eval.py`** (new script, queued via its own `--wait_min 720` GPU-polling
   loop — no separate waiter needed) — batch quantitative eval across
   EVAL_PLAN.md §3.3's 16-episode set (8 seen/variation0 + 8 unseen/variation1, verified to exist
   in the current `data/rlbench_selfgen_512_aug` tree), 4 modalities, `IIII` mode only, with
   bootstrap-CI aggregation split by seen/unseen. This is the first executable version of the
   batch eval EVAL_PLAN.md designed back on 2026-08-11 but never coded (its blocker, the pipeline
   not supporting arbitrary template/stream assembly, was resolved since by
   `modality_mode_grid.py`/`tag_swap_ablation.py`, whose GPU-claiming/scoring functions this
   script reuses directly rather than reimplementing). CPU-side smoke test passed (all 16
   episodes resolve, `getitem(force_template=...)` doesn't degrade on any of the 4 templates on a
   representative unseen episode). Currently polling for a free GPU (log:
   `logs_heldout_batch_eval.txt`); will start automatically once GPU 0 frees (after job 1) or
   sooner if another GPU opens up.

**Deferred to the batch after this one** (per user's explicit prioritization): M0 full 10000-step
retrain (needs a cold start from step125750, not a cheap continuation like M1 was — ~58h/2GPU) and
a `video+normal@1.0` specialist to complete Table 1's three-specialist set.

### Check-in commands for this batch

```
tail -50 logs_official_closedloop_postfix.txt
tail -50 logs_specialist_depth.txt; tail -50 logs_specialist_seg.txt
tail -50 logs_heldout_batch_eval.txt
python -c "import json; d=json.load(open('reports/closedloop/rollout_official_125750_postfix_n20.json')); print(len(d['results']),'done'); print(d['errors'])"
cat reports/closedloop/compare_arm6_vs_official_postfix.json   # once job 1 finishes
cat reports/heldout_batch_eval/arm6_10000_heldout16/summary.json   # once job 4 finishes
ls outputs/specialist_depth__seed42_fi3_512_aug_sr/ outputs/specialist_seg__seed42_fi3_512_aug_sr/
nvidia-smi
```
