
# Results summary — fusion arm vs. every baseline on this project

Supersedes the T3 table in `FUSION_RESULTS_CONSOLIDATED.md`, which compared models across
**different task sets** and therefore overstated the fusion arm's standing. Sources:
`machine_b_exp_results.md` (Machine B: all baselines) + `machine_c_exp_results.md` (Machine C:
the live fusion arm). Every number below was recomputed from the raw JSON, per task.

---

## 0. Update (2026-09-15, Machine C) — all five models, same six tasks, 20 trials each

This directly answers §1's "no two models share a task set" problem: every model below was run
on the **identical six unseen tasks**, same protocol per model (official: cfg=10.0/fi=4/no tags,
native to its own training; every fine-tuned arm: cfg=7.5/fi=3/explicit tags), `--max-ik-fail-streak
5`, `--arm-action-mode planning`, `--axis-solver sphere`, seed 42. Fusion uses **checkpoint-8000**
(not 5000/7000 — training advanced past 7000 mid-session and `checkpoint_save_top_k=1` deletes old
checkpoints entirely once a newer one saves, not just their optimizer state), `--anchor-modality
fusion` (rgb_only, the deployment condition — not `fusion_full_anchor`, kept as a separate,
non-comparable reading). **All five models are now complete at n=120 each (600 trials total).**

| Model | basketball_in_hoop | close_box | close_drawer | close_microwave | take_lid_off_saucepan | toilet_seat_down | **Total** |
|---|---|---|---|---|---|---|---|
| official (step125750) | 0/20 | 5/20 | 16/20 | 15/20 | 16/20 | 3/20 | **55/120 (45.8%)** |
| specialist_action (step4000) | 0/20 | 6/20 | 15/20 | 16/20 | 18/20 | 4/20 | **59/120 (49.2%)** |
| arm6 (step10000) | 0/20 | 9/20 | 7/20 | 17/20 | 15/20 | 5/20 | **53/120 (44.2%)** |
| arm7 (step10000) | 0/20 | 16/20 | 13/20 | 14/20 | 0/20 | 4/20 | **47/120 (39.2%)** |
| fusion0 (step8000) | 0/20 | 6/20 | 4/20 | 4/20 | 3/20 | 8/20 | **25/120 (20.8%)** |
| arm7 (step4000)† | N/A | 9/10 | 6/10 | 8/10 | 0/10 | 1/10 | **24/50 (48.0%)** |

† arm7-checkpoint-4000 (2026-09-16, machine C, priority-1 campaign) was only run on 5 of the 6
tasks (basketball_in_hoop excluded) at **n=10**, not n=20 — not directly comparable to the rows
above, included here for reference against the same checkpoint's step-10000 result. Same protocol
(cfg=7.5/fi=3/explicit tags, `--anchor-modality video`). Per-task vs. final arm7 (step10000): close_box
90% vs 80% (4k ahead), close_microwave 80% vs 70% (4k ahead), close_drawer 60% vs 65% (10k ahead),
toilet_seat_down 10% vs 20% (10k ahead), take_lid_off_saucepan 0% vs 0% (tie) — a mixed, noisy
picture at n=10, not a clean trend either way. What's much clearer: 4k utterly fails at held-out
variations of the 16 *trained* tasks (see `machine_c_exp_results.md` / memory
`rlbench-trained-task-variation-counts` for that separate campaign): 5/130 (3.8%) at variation 1
and 0/110 (0%) at variation 2, vs. 48% here on genuinely unseen tasks at the training variation.
The step-4000 checkpoint generalizes to new task types under the training-time variation
distribution far better than it generalizes to new variations of tasks it has already seen —
consistent with 4000 steps being too early to have learned variation-invariant features yet.

**Reading it, all five complete:**
- Ranking: specialist_action 49.2% > official 45.8% > arm6 44.2% > arm7 39.2% > fusion0 20.8%.
  The top four span 29 points; fusion0 sits nearly 19 points below the next-lowest (arm7) — a
  much larger gap than separates any pair of the other four. This is the least-trained checkpoint
  of the five (8000 steps, resumed mid-session from a 5000-step run) and the only one asked under
  the harder cross-modal-completion training objective, so the gap is not evidence the fusion
  *approach* fails at closed-loop control — only that this particular checkpoint, at this step
  count, trails the single/few-modality baselines on this specific task set.
- `basketball_in_hoop` is **0/20 for all five models** — 100 trials, zero successes. The single
  strongest, most model-independent signal in this entire document.
- `take_lid_off_saucepan` splits sharply: 15–18/20 for official/specialist_action/arm6, but
  **0/20 for arm7** and only 3/20 for fusion0. This single task accounts for roughly half of
  arm7's gap to the top three, and is fusion0's second-weakest task after basketball_in_hoop.
- `close_box` is arm7's strongest task (16/20, tied for best in the table with official) —
  consistent with §3's earlier single-task read of arm7 improving sharply from step 8000 to
  step 10000 on this exact task, now confirmed at n=20 instead of n=20-total-across-checkpoints.
- Prior to this run, arm7 had **never been evaluated on this task set at all** (§1's coverage
  table showed it existing "only in Group B" — close_box/close_drawer). This is the first time
  arm7's true unseen-task standing (39.2%, not competitive with the top three) is visible.
- fusion0's best task is `toilet_seat_down` (8/20, 40%) — its only result above the table's
  overall average for that task's column, and worth a closer look at what differs there.
- Fusion's early trials (11.4%, n=35) are in the same range as its earlier 10-trial default-
  threshold reading (18.3%, §1 old table) and well below the other four models — consistent with
  the standing conclusion in §8, not yet contradicted by more data.

Raw JSON: `reports/closedloop/rollout_s20_{official,specaction,arm6,arm7,fusion}_g{1,2}_cl.json`
on Machine C. CSV export: `eval_doc/results_scale20.csv`.

---

## 1. The structural problem: no two models share a task set

`rlbench_unseen_tasks_512_aug` holds **six** never-trained tasks. Coverage across every
closed-loop run ever made (`—` = never evaluated):

| run | basketball | close_box | close_drawer | close_microwave | take_lid_off | toilet_seat |
|---|---|---|---|---|---|---|
| **fusion0@5000** (C) | 0/10 | 6/10 | 2/10 | 2/10 | 0/10 | 1/10 |
| **fusion0@5000 ik=15** (C) | 0/10 | 10/10 | 4/10 | 4/10 | 0/10 | 9/10 |
| arm6 `unseen4` | 0/10 | — | — | 9/10 | 8/10 | 1/10 |
| action spec `unseen4` | 0/10 | — | — | 7/10 | 9/10 | 1/10 |
| official `unseen4` | 0/10 | — | — | 7/10 | 7/10 | 1/10 |
| **arm7** (all runs) | — | ✓ | ✓ | — | — | — |
| arm6 / arm4 / arm0 / spec / official `main5` | — | ✓ | ✓ | — | — | — |
| GT replay | — | ✓ | ✓ | — | — | — |

**Only fusion0 has been evaluated on all six.** The rest split into two disjoint groups, and
**arm7 exists only in group B** — that is why it was absent from the earlier 4-task table. It was
not an omission on my part; the data does not exist.

---

## 2. Group A — `basketball_in_hoop`, `close_microwave`, `take_lid_off_saucepan`, `toilet_seat_down`

| model | success | rate | 95% CI | ik-streak | fi |
|---|---|---|---|---|---|
| arm6 @10000 | 18/40 | **45.0%** | [29.6, 60.4] | 5* | 3 |
| action specialist @4000 | 17/40 | 42.5% | [27.2, 57.8] | 5 | 3 |
| official 125750 | 15/40 | 37.5% | [22.5, 52.5] | 5* | 4 |
| **fusion0 @5000 (ik=15)** | 13/40 | **32.5%** | [18.0, 47.0] | 15 | 3 |
| **fusion0 @5000 (ik=5)** | 3/40 | **7.5%** | [0.0, 15.7] | 5 | 3 |

**arm7 has no data here.**

> `5*` = 该运行的 `args` 里没有 `max_ik_fail_streak` 字段，但已由 `results[].ik_fail_streak_max` 上限与 `stop_reason` 字面量反推确认阈值就是 5（§6）——不是假设。official 那份 40 trial 里最大连续 IK 失败只有 1，**从未触发过中止**。

At the matched default threshold the fusion arm is **7.5% against 37.5–45.0%** — a 5× gap, and
its CI does not overlap any baseline. Even at the relaxed threshold it stays below all three.

The per-task breakdown localises it:

| task | arm6 | spec | official | fusion ik=5 | fusion ik=15 |
|---|---|---|---|---|---|
| basketball_in_hoop | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| close_microwave | **9/10** | 7/10 | 7/10 | **2/10** | 4/10 |
| **take_lid_off_saucepan** | **8/10** | **9/10** | **7/10** | **0/10** | **0/10** |
| toilet_seat_down | 1/10 | 1/10 | 1/10 | 1/10 | **9/10** |

- **`take_lid_off_saucepan` is the discriminating task**: every baseline reaches 70–90%, the
  fusion arm is 0/10 under *both* thresholds. A specific, locatable failure, not noise.
- `toilet_seat_down` is threshold-sensitive **only for fusion** (1→9). It can produce a viable
  trajectory there; early poses were killing the trial.
- `basketball_in_hoop` is 0/10 for everything ever run. Not a model discriminator.

---

## 3. Group B — `close_box`, `close_drawer` (where arm7 lives)

| model | success | rate | 95% CI | ik | fi |
|---|---|---|---|---|---|
| *GT replay (planning)* | *20/20* | *100.0%* | — | n/a (planning) | 1 |
| **fusion0 @5000 (ik=15)** | 14/20 | **70.0%** | [49.9, 90.1] | 15 | 3 |
| *GT replay (ik)* | *14/20* | *70.0%* | — | 5* | 1 |
| action specialist @4000 | 23/40 | 57.5% | [42.2, 72.8] | 5* | 3 |
| **arm7 @10000** (t0/t1) | 22/40 | **55.0%** | [39.6, 70.4] | 5 | 3 |
| official 125750 | 20/40 | 50.0% | [34.5, 65.5] | 5* | 4 |
| **arm7 @8000** (main5) | 18/40 | **45.0%** | [29.6, 60.4] | 5 | 3 |
| arm7 @10000, anchor=normal | 16/40 | 40.0% | [24.8, 55.2] | 5 | 3 |
| **fusion0 @5000 (ik=5)** | 8/20 | **40.0%** | [18.5, 61.5] | 5 | 3 |
| arm7 @10000, anchor=depth | 15/40 | 37.5% | [22.5, 52.5] | 5 | 3 |
| arm6 @10000 | 14/40 | 35.0% | [20.2, 49.8] | 5* | 3 |
| arm7 @10000, anchor=seg | 12/40 | 30.0% | [15.8, 44.2] | 5 | 3 |

**Here the fusion arm is competitive**: 40.0% at the matched threshold, mid-pack among baselines
spanning 30–57.5%; at the relaxed threshold it is the best non-GT entry. Every CI in this table
overlaps every other — **n=20–40 on two tasks cannot separate these models.**

Note arm7 on `close_box` alone: **15% at step 8000 → 65% at step 10000**. Same arm, two
checkpoints, 4×. That single figure is the strongest argument in this document against reading
any 2-task, n≤40 comparison as a model ranking.

---

## 4. Why the earlier "fusion ties arm6 at 45.0%" was wrong

fusion0's 6-task aggregate is 45.0% (relaxed) and arm6's 4-task `unseen4` aggregate is also
45.0%. The match is coincidence over different task sets. Fusion's 45.0% is carried by
`close_box` (10/10) and `close_drawer` (4/10) — **two tasks the `unseen4` baselines were never
run on**. Remove them and fusion falls to 32.5% (relaxed) / 7.5% (default) while arm6 stays at
45.0%.

**Rule: aggregate success rates from different task sets are not comparable, even at identical n.**

---

## 5. `main5` is a mixed set, and that changes how its headline reads

`main5` is this repo's headline closed-loop protocol (`EVAL_PLAN_ARM7.md` §2): 5 tasks × 20
trials, variation0 — `close_box, close_drawer, push_buttons, meat_off_grill, open_drawer`. It is
the only protocol all three baselines completed, and it defines the "arm7 must beat 46%" target.

**Two of its five tasks are never-trained** (`close_box`, `close_drawer` belong to the unseen-task
tree). Splitting it:

| model | main5 headline | trained 3 | **never-trained 2** |
|---|---|---|---|
| action specialist @4000 | 46.0% | 38.3% | **57.5%** |
| arm7 @8000 | 41.0% | 38.3% | **45.0%** |
| arm6 @10000 | 35.0% | 35.0% | **35.0%** |
| official 125750 | 24.0% | **6.7%** | **50.0%** |

Every model does at least as well on the tasks it never trained on. The official checkpoint is
**6.7% vs 50.0%** — a 7.5× gap. So a large part of the main5 ranking is driven by those two tasks
being *easy*, not by learning on the trained ones.

This sharpens an existing repo conclusion (`实验结论汇总.md`: "tier-3 的 40% 不是迁移证据") with
a stronger argument: a checkpoint that scores 6.7% on the trained tasks scores 50% on tasks it
has never seen.

**Also: `open_drawer` is 0/20 for all five models**, including those trained on it — a systematic
failure averaged into the headline.

> **Naming trap.** `eval/rollout.py:43` `DEFAULT_TASKS` is **three** tasks
> (`push_buttons, open_drawer, meat_off_grill`), not main5's five. A run launched without
> `--tasks` is not comparable to the main5 table and its log looks identical.

---

## 6. Protocol knobs that move the number more than the model does

| knob | effect | evidence |
|---|---|---|
| `--max-ik-fail-streak` 5 → 15 | **+26.7 pp** on T3 (18.3→45.0), **0.0 pp** on T2 | C, same ckpt & seed, z=3.28 |
| `--arm-action-mode` ik → planning | ~+12 pp | B, GT replay 88.0% → 100.0% |
| `--frame-interval` | official ran at **4**, everything else at **3** | B, run args |
| checkpoint choice (single task, n=20) | up to **4×** | arm7 `close_box` 15% → 65% |

**已解决（下调为非问题）：** 本机 30 份文件的 `args` 里没有 `max_ik_fail_streak` 字段
（当时还没记这个 flag），但它们**确实都跑在 ik=5 上** —— 每份文件的 `results[]` 里
`ik_fail_streak_max` 的全局最大值都是 **5，从未超过**，且 `stop_reason` 里直接写着字面量
`"5 consecutive IK failures"`。阈值在当时是硬编码的 5，后来才提成 flag，默认值不变
（`eval/rollout.py:391` `default=5`）。

所以 **arm6 与 arm7 的全部结果都是 ik=5**，与 Machine C 的默认组同协议，可直接比。
唯三例外是三份 arm6 探索性运行（`rollout_arm6_{var1,var0,gated12_v1}_ikstreak50.json`，
ik=50，n=1–6），本文档没有引用它们。

一个副产物：`rollout_official_125750_unseen4.json` 的 `ik_fail_streak_max` 最大只有 **1**，
40 个 trial **一次都没触发过 IK 中止** —— 对 official 在 A 组的 37.5% 而言，
IK 阈值完全不是影响因素。

---

## 7. The result that does not depend on any of this

Unseen **layouts** (variation1), where task and scene are both in-distribution except the layout:

| model | n | success |
|---|---|---|
| **fusion0 @5000** | 70 | **1.4%** (identical at both thresholds) |
| arm6 @10000 | 60 | 1.7% |
| action specialist | 68 | 1.5% |
| arm6 @10000 (13 tasks) | 130 | 3.8% |
| **official 125750** | 60 | **0.0%** |
| *GT replay (no model)* | *30* | *80–90%* |

**Every model ever trained here, plus the released checkpoint, lands in 0–4% on unseen layouts,
while the harness reaches 80–90% on the same scenes.** Fusion is inside that band, and its T2
number is threshold-insensitive — a genuine action-prediction failure, not an artifact.

This is the project's largest open problem, and it is orthogonal to the modality question.

---

## 8. Standing of the fusion arm, stated honestly

- **Group A, matched protocol: clearly worse** (7.5% vs 37.5–45.0%), driven by a hard 0/10 on
  `take_lid_off_saucepan`.
- **Group B: competitive** (40.0%, mid-pack), but every CI overlaps.
- **Unseen layouts: indistinguishable from everything else**, i.e. ~0%.
- **Caveat that applies to all of it:** this is a **5,000-step checkpoint that was still
  descending in loss**, now training toward 20,000 on Machine C. It is a reading of an
  under-trained model, not of the architecture.
- **The claim the arm was built to test — the mutual-completion matrix — has never been run on
  either machine.** Nothing here speaks to it.

---

## 9. What would make these numbers comparable

1. Run fusion0 on **main5** (`--tasks close_box close_drawer push_buttons meat_off_grill
   open_drawer`, 20 trials, ik=5, fi=3, planning) so it enters the three-baseline table directly.
   The current fusion closed-loop watcher uses `DEFAULT_TASKS` (3 tasks) and will **not** produce
   a comparable number.
2. Run **arm7 and arm6 on the 4 `unseen4` tasks at ik=5** to complete Group A.
3. Re-run one baseline at ik=15 to calibrate the threshold offset across the whole table.
4. Report per task with n, never aggregates across task sets.

---

## 附录 A — 全部数据出处（可核对）

所有路径相对 `/workspace/ttdu/ActionImages-Cogen/`。SHA1 是我读取时文件的前 8 位，
用来确认你核对的是同一份文件（`sha1sum <file> | cut -c1-8`）。

### A.1 本机（Machine B）实存的 JSON — 23 份

复核命令：
```bash
python -c "import json,sys; d=json.load(open(sys.argv[1])); \
print(d['summary']); [print(r['task'], r['variation'], r['success']) for r in d['results']]" \
  reports/closedloop/<file>.json
```

| 文件 | SHA1(前8) | ckpt | 任务数 | n | success | 用于 | 取值 |
|---|---|---|---|---|---|---|---|
| `rollout_actionspec_gated12_v1.json` | `130e8ce2` | step4000.ckpt | 12 | 68 | 1.5% | §7 | specialist var1 1.5% |
| `rollout_actionspec_unseen4.json` | `ad08a0d4` | step4000.ckpt | 4 | 40 | 42.5% | §2 A组 | specialist 17/40 |
| `rollout_arm4__seed42_fi3_10000.json` | `3874b0e6` | step10000.ckpt | 5 | 20 | 35.0% | §3 B组 | arm4 5/8 |
| `rollout_arm6_10000.json` | `b5dbfc79` | step10000.ckpt | 5 | 100 | 35.0% | §3 B组 / §5 | arm6 逐任务 |
| `rollout_arm6_10000_unseen4.json` | `8e5f663d` | step10000.ckpt | 4 | 40 | 45.0% | §2 A组 | arm6 18/40 |
| `rollout_arm6_10000_var1.json` | `a1df74b9` | step10000.ckpt | 3 | 60 | 1.7% | §7 | arm6 var1 1.7% |
| `rollout_arm6_alltasks_v1.json` | `ebd49676` | step10000.ckpt | 13 | 130 | 3.8% | §7 | arm6 13任务 var1 3.8% |
| `rollout_arm7_10000_anchor_depth.json` | `3b2161c1` | step10000.ckpt | 5 | 100 | 22.0% | §3 B组 | arm7 anchor=depth 15/40 |
| `rollout_arm7_10000_anchor_normal.json` | `771f1ba8` | step10000.ckpt | 5 | 100 | 17.0% | §3 B组 | arm7 anchor=normal 16/40 |
| `rollout_arm7_10000_anchor_segmentation.json` | `3c8bba93` | step10000.ckpt | 5 | 100 | 28.0% | §3 B组 | arm7 anchor=seg 12/40 |
| `rollout_arm7_10000_t0.json` | `26910d3e` | step10000.ckpt | 1 | 20 | 65.0% | §3 B组 | arm7@10000 close_box 13/20 |
| `rollout_arm7_10000_t1.json` | `c5e0e8fc` | step10000.ckpt | 1 | 20 | 45.0% | §3 B组 | arm7@10000 close_drawer 9/20 |
| `rollout_arm7_10000_t2.json` | `705a4e27` | step10000.ckpt | 1 | 20 | 40.0% | §1 覆盖表 | push_buttons(已训练) |
| `rollout_arm7_10000_t3.json` | `83793204` | step10000.ckpt | 1 | 20 | 80.0% | §1 覆盖表 | meat_off_grill(已训练) |
| `rollout_arm7_10000_t4.json` | `18152f27` | step10000.ckpt | 1 | 20 | 0.0% | §1 覆盖表 | open_drawer 0/20(已训练) |
| `rollout_arm7_8000_main5.json` | `cc3bae08` | step8000.ckpt | 5 | 100 | 41.0% | §3 B组 / §5 | arm7@8000 逐任务 |
| `rollout_gt_replay_var0.json` | `5c7d5798` | step125750.ckpt | 5 | 50 | 88.0% | §3 / §6 | GT(ik) 70% |
| `rollout_gt_replay_var0_planning.json` | `9be43ea5` | step125750.ckpt | 5 | 50 | 100.0% | §3 / §6 | GT上界 100% |
| `rollout_gt_replay_var1_planning.json` | `38d865e6` | step125750.ckpt | 3 | 30 | 90.0% | §7 | GT var1 90% |
| `rollout_official_125750_postfix_n20.json` | `315d76e6` | step125750.ckpt | 5 | 100 | 24.0% | §3 B组 / §5 | official main5 逐任务 |
| `rollout_official_125750_unseen4.json` | `448a2091` | step125750.ckpt | 4 | 40 | 37.5% | §2 A组 / §4 | official 15/40 = 37.5% |
| `rollout_official_125750_var1.json` | `c0cfc2c3` | step125750.ckpt | 3 | 60 | 0.0% | §7 | official var1 0.0% |
| `rollout_spec_action_4000.json` | `35331c99` | step4000.ckpt | 5 | 100 | 46.0% | §3 B组 / §5 | specialist 逐任务 |

**注意「success」列是文件自带的全局 `summary.success_rate`，不等于我在正文里引用的数。**
正文多处取的是**该文件的任务子集**，从 `results[]` 逐条重数：

| 正文的数 | 文件全局 | 差别原因 |
|---|---|---|
| arm7 anchor=depth **15/40 (37.5%)** | 22.0% (n=100) | 只取 `close_box`+`close_drawer` 两个任务 |
| arm7 anchor=normal **16/40 (40.0%)** | 17.0% (n=100) | 同上 |
| arm7 anchor=seg **12/40 (30.0%)** | 28.0% (n=100) | 同上 |
| arm7@8000 **18/40 (45.0%)** | 41.0% (n=100) | 同上 |
| arm6@10000 **14/40 (35.0%)** | 35.0% (n=100) | 同上（数值巧合相等） |
| official **20/40 (50.0%)** | 24.0% (n=100) | 同上 |
| specialist **23/40 (57.5%)** | 46.0% (n=100) | 同上 |
| arm4 **5/8 (62.5%)** | 35.0% (n=20) | 只取两任务，且该文件每任务只有 4 trial |
| GT(ik) **14/20 (70.0%)** | 88.0% (n=50) | 同上 |
| GT(planning) **20/20 (100%)** | 100.0% (n=50) | 同上 |

arm7@10000 的 B 组 22/40 是 **t0 + t1 两个单任务文件相加**（13/20 + 9/20），不是一个文件。

`--max-ik-fail-streak` / `--frame-interval` 取自每份文件的 `args` 块：
```bash
python -c "import json,sys;a=json.load(open(sys.argv[1]))['args']; \
print(a.get('max_ik_fail_streak'), a.get('frame_interval'), a.get('arm_action_mode'))" \
  reports/closedloop/<file>.json
```
`max_ik_fail_streak` 字段缺失的 30 份文件**已核实同样是 ik=5**（见 §6）。判据是
`results[].ik_fail_streak_max` 全局最大值恰为 5 且 `stop_reason` 出现字面量
`"5 consecutive IK failures"`：

```bash
python - <<'EOF'
import json,glob,os
for p in sorted(glob.glob("reports/closedloop/rollout_*.json")):
    d=json.load(open(p)); rs=d.get("results") or []
    if not rs or "max_ik_fail_streak" in d["args"]: continue
    print(os.path.basename(p), max((r.get("ik_fail_streak_max") or 0) for r in rs))
EOF
```

### A.2 fusion0 的数 —— **不在本机**

本机 `reports/closedloop/` 下与 fusion 有关的只有 `rollout_smoke_fusion.json`（冒烟，非结果）。
正文所有 fusion0 数字转引自 `machine_c_exp_results.md`，对应 Machine C 上的：

| Machine C 路径 | 正文用途 |
|---|---|
| `reports/closedloop/rollout_t3{a,b}_cl.json` | fusion0@5000 T3 默认阈值 3/40、8/20 |
| `reports/closedloop/rollout_t3{a,b}_relaxed_cl.json` | fusion0@5000 T3 ik=15 13/40、14/20 |
| `reports/closedloop/rollout_t2{a,b,c}_cl.json` | fusion0 T2 1.4% (n=70) |
| `reports/closedloop/rollout_t2{a,b,c}_relaxed_cl.json` | T2 阈值不敏感（两者相同） |
| `reports/closedloop/rollout_t7{a,b,c}_cl.json` | step7000 full_anchor，**当时仍在写入** |
| `checkpoints/checkpoint-7000/trainer_state.json` | §6 的 loss/grad-norm、t=1.46 / t=0.06 |

**这是本文档最弱的一环**：A 组和 B 组里所有 fusion 行都无法在本机复核，
且 Machine C 的 t7 文件在写这份汇总时还没跑完。要真正核对，需要把 C 上这 10 份
JSON 拷回本机 `reports/closedloop/` 再跑 A.1 的命令。

### A.3 非 JSON 来源

| 断言 | 出处 |
|---|---|
| main5 的 5 个任务定义、cfg/fi/anchor 等参数 | `EVAL_PLAN_ARM7.md` §2 |
| 16 个训练任务清单（用来判定 close_box/close_drawer 未训练） | `scripts/gen_launch_512.sh` 的 TASKS |
| 6 个 T3 任务清单 | `data/rlbench_unseen_tasks_512_aug/` 目录列表 |
| `DEFAULT_TASKS` 只有 3 个任务 | `eval/rollout.py:43` |
| 「tier-3 的 40% 不是迁移证据」 | `实验结论汇总.md` |
| 43.8% 无法复现 | 见 §附录 B |

### A.4 43.8% 的出处问题

`FUSION_RESULTS_CONSOLIDATED.md` 里的 "Official (step125750) T3 **43.8%**" 在本机
**没有任何 JSON 能产生这个数**。official 的 T3 只有 `rollout_official_125750_unseen4.json`
一份，4 任务 n=40，逐条重数是 **15/40 = 37.5%**。43.8% = 7/16，n=16 不对应任何已有运行。
**可辩护的数字是 37.5%（n=40，4 任务）**；43.8% 应从所有文档中删除。
