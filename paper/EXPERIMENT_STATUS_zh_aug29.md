# 实验总览与 draft_aug_29.md 表格对应关系(中文版,2026-08-29)

> 目的:把这两天所有跑过、正在跑、推迟掉的实验汇总成一份文档,并且逐张表核对
> `paper/draft_aug_29.md`(新草稿,Experiments 一节已经按 `paper/U-CoGen_experiment_plan.md`
> 的建议重写过、7 张表基本都是 `[XX]` 占位符)现在能不能被现有/在跑的实验填上。
> 结论先说:**能填上一部分,但有 4 个明确的缺口(2 个是代码没写,2 个是决策没定),
> 不解决这些,`[XX]` 填不满。** 详见第三节。

---

## 一、这两天做了什么(时间线)

### Batch 0(更早,已经在仓库里,本次会话之前完成)
- **arm0**(`video+action@1.0`,对照组)、**arm4**(`video+action@0.6,video+depth@0.2`)、
  旧 **arm1**(`video+action@0.6,video+depth@0.2,video+segmentation@0.2`)——三个都在
  256 分辨率的旧数据树上训过,本地 checkpoint **已经被清理删除**(`outputs/` 现在只剩
  arm6 系列),`arm4` 有 HF Hub 备份(`TingtingDu/arm4__seed42_fi3`),`arm0` 脚本里
  引用了 `TingtingDu/arm0__seed42` 但本地缓存没找到实际文件(需要真的跑一次
  `hf_hub_download` 才能确认还在不在),**旧 arm1 目前没有找到任何备份路径**,而且
  `EVAL_PLAN_CLOSEDLOOP.md` 里记录过它的 `checkpoint-2750` 曾经损坏过(1GB vs 应有的
  11.95GB)。
- **arm6**(`video+action@0.4,depth@0.2,segmentation@0.2,normal@0.2`,M2 混合,512 分辨率
  augmented 数据树)训满 10000 步,是当前草稿的 headline model。

### Batch 1(本次会话,8 月 29 日凌晨,用户离开前布置,已全部跑完)
| 任务 | 内容 | 结果 |
|---|---|---|
| Track A | arm6 从 step10000 续训,`PERCEPTION_MASK_MIX=M1`(45/10/45 均衡),3000 步 | ✅ `TRAIN_EXIT=0`,6 个 checkpoint(500~3000)全部落盘,`outputs/arm6m1__seed42_fi3_512_aug_sr` |
| Track B 阶段1 | arm6@step10000 完整闭环 campaign,5 任务×20 trial=100 rollout | ✅ `errors: []`,`reports/closedloop/rollout_arm6_10000.json` |
| Track B 阶段2 | tag-swap 消融从 1 个 episode 扩到 4 个(open_drawer/push_buttons/stack_wine/turn_tap) | ✅ 全部跑完,`reports/tag_swap/arm6_10000_*/metrics.json` |

**关键发现(Batch 1 期间查出来的,还没写进任何草稿):**
1. **anchor 比 tag 重要得多**——tag-swap 消融显示换 prompt tag 只让指标变化 ~1-2%,
   而拿掉 anchor 帧(`f0f0`)会让指标崩掉 ~9 倍。也就是说"prompt 选择模态"这个说法,
   实际起作用的主要是 anchor 帧的视觉内容,不是文字 tag。`sec:templates`("announced
   in the prompt")这个措辞需要斟酌。
2. **仓库里现有的 official/arm4 闭环参考数据是"anchor fix"之前跑的**——
   `archive_pre_anchorfix/compare_arm4_vs_official_100.json` 的数字跟旧草稿
   `tab:closedloop` 完全对得上,但这份数据(2026-08-13)早于 `eval/rollout.py` 的最后一次
   改动(2026-08-22,加了 `--skip-anchor-frames` 默认值)。**这意味着旧草稿(draft_aug_28)
   的闭环 headline 数字,和新生成的 arm6 数据,不是同一套流水线跑出来的,不能直接比。**

### Batch 2(本次会话,8 月 29 日晚上,正在跑)

跟用户过了一遍 `paper/U-CoGen_experiment_plan.md` 之后确认的优先级:官方 checkpoint
替代 arm0 做闭环对照、先跑 2 个 specialist、held-out 批量评测脚本照抄 EVAL_PLAN.md 的
16-episode 设计。四件事:

| # | 内容 | GPU | 状态(截至最后一次检查) |
|---|---|---|---|
| 1 | 官方 step125750 闭环,n=20/task,**官方自己的协议**(`cfg=10 fi=4 tag=none`,不是 `queue_closedloop.sh` 给微调 arm 用的默认值) | GPU 0 | 🟡 跑中,5/100 rollout,`errors: []` |
| 2 | depth specialist(`video+depth@1.0`,从官方 step125750 冷启动,4000 步) | GPU 3,4 | 🟡 跑中,~100/4000 步,loss 正常 |
| 3 | seg specialist(`video+segmentation@1.0`,`SEG_MODE=scene_roles`) | GPU 5,7 | 🟡 跑中,~100/4000 步,loss 正常 |
| 4 | `scripts/heldout_batch_eval.py`(新写的脚本,复用 `modality_mode_grid.py` 的生成打分逻辑,循环 16 个 held-out episode,4 个模态,只测 `IIII`) | 排队等 GPU | ⚠️ 第一次启动撞上了 GPU 抢占的竞态(`pick_gpu` 判断 GPU1 空闲,但 `reserve_vram` 真去申请时被另一个租户抢先占用),进程退出了。**已经用带重试循环的包装脚本 `run_heldout_batch_eval_retry.sh` 重新拉起**,现在应该在等 GPU 或已经在跑 |

**推迟到下一批(已经跟用户确认过,这次不做):**
- **M0 全量重训**——`PERCEPTION_MASK_MIX=M0`,必须从官方 step125750 冷启动完整 10000 步
  (不能像 M1 那样从 arm6 收敛点续训),约 58 小时/2GPU。
- **normal specialist**(`video+normal@1.0`)。

---

## 二、和 `paper/U-CoGen_experiment_plan.md` 的对照

这份文档(应该是协作者写的重构建议)指出的缺口,和我们已经/正在做的对照:

| 建议要做的事 | 现状 |
|---|---|
| headline model(arm6)要自己跑闭环,不能用 arm4 代替 | ✅ Batch 1 做完了(arm6@10000, n=100) |
| held-out 完整定量评测(不能只用 n=1 的 open_drawer) | 🟡 Batch 2 第4项在做,但目前只测 `IIII` 一种模式,且还没跑完 |
| specialist baseline(depth/seg/normal 各自单模态训练) | 🟡 depth、seg 在训,normal 推迟 |
| M0/M1/M2 conditioning 矩阵 | 🟡 M1、M2(=arm6)有了,M0 推迟 |
| RGB world model 质量(PSNR/SSIM/LPIPS,而不是 given-RGB 的 VAE PSNR) | ❌ **完全没做**——见下一节 |
| tag ablation(移除 prompt tag 训练) | ⚠️ 我们做的是推理时消融(tag-swap),不是训练时去掉 tag;两者是不同实验,但 tag-swap 的结果已经能预判训练时去掉 tag 大概率也不会崩 |
| tri-modal(video+depth+action 同时生成) | ❌ 没做,标记为"可选" |
| 新模态(optical flow) | ❌ 没做,标记为"可选/需要人工验证新 codec" |

---

## 三、逐表核对:`draft_aug_29.md` 的表格,现有/在跑实验能不能填

新草稿的 Experiments 一节几乎是照着 `U-CoGen_experiment_plan.md` 重写的,7 张表全是
`[XX]` 占位符。逐张核对:

### ✅ `tab:action-mixture`、`tab:perception-mixture`、`tab:gate`(附录)
已经是真实数字,不是占位符,不需要新实验。

### ⚠️ `tab:headline`(4.2 节,headline table:arm0 / 三个 specialist / arm6,五列:
Action pos / Depth AbsRel / Seg mIoU / Normal cos / **RGB LPIPS**)

**能填的部分**:arm6 那一行的 Action/Depth/Seg/Normal 四列——`heldout_batch_eval.py`
跑完就有(16 episode 的均值)。depth/seg specialist 那两行的 Depth/Seg 列——等
Batch 2 的两个 specialist 训完、再跑一次 `heldout_batch_eval.py`(指向新 checkpoint)
就有。

**填不了的部分**:
1. **RGB LPIPS 这一列,全表都填不了**——我检查了整个仓库,**没有任何地方实现过 LPIPS
   或 SSIM**(`grep -rl "lpips\|ssim" *.py` 是空的)。现有的 `score()` 函数(
   `scripts/modality_mode_grid.py`)只算了 PSNR。这是一个代码缺口,不是"再等一会儿"
   就能解决的,需要新写(装 `lpips` 包或用 `torchmetrics`,再把"生成 RGB 视频 vs GT"
   这条路径接进 `heldout_batch_eval.py`——目前这个脚本只处理 depth/seg/normal/action
   四个模态,完全没有跑过 bare `video` 模板)。
2. **Normal specialist 那一行,完全没有对应实验**——normal specialist 这次被推迟了,
   没有 checkpoint 可评。
3. **arm0 那一行**(Action pos + RGB LPIPS)——arm0 的 checkpoint 需要确认还在不在
   (见第一节,本地已删除,`TingtingDu/arm0__seed42` 这个 HF 仓库名字在脚本里出现过但
   没在本地缓存里验证过),而且同样卡在 LPIPS 没实现这件事上。

### ❌ `tab:fixed_compute`(4.3 节,arm0→arm4→arm1→arm6 在同一 total step 下的对比)

**这张表现在完全没法填,原因是数据本身可能没了**:arm0/arm4/旧 arm1 的本地 checkpoint
都已经被清理(`outputs/` 里现在只有 arm6 系列)。arm4 确认有 HF Hub 备份
(`TingtingDu/arm4__seed42_fi3`,本地已缓存),arm0 的备份需要实测才能确认,**旧 arm1
(action+depth+seg 那个,不是新草稿里改名叫"depth specialist"的那个)目前没找到任何备份
路径,而且历史记录里它的 `checkpoint-2750` 曾经损坏过**——如果备份也没有,这一行只能
重新训练。另外这张表要求"同一个 total step"横向比较,而 arm0/arm4/旧arm1 是在**旧的
256 分辨率数据树**上训的,和 arm6 用的 512 augmented 树不是同一个 domain,直接比较
不公平(这一点 `train_arm.sh` 自己的注释里也反复强调过"tree 不同不可比")。**这张表
目前是四张表里风险最高的一张,建议先确认 arm0 的备份是否存在、旧 arm1 要不要重训,
再决定要不要做。**

### ⚠️ `tab:conditioning_mixtures` / `tab:conditioning_results`(4.4 节,M0/M1/M2 ×
IIII/FIII/FIFI,针对 depth)

`tab:conditioning_mixtures`(M0/M1/M2 三行的概率表)已经是真实数字,不需要填。
`tab:conditioning_results`(M0/M1/M2 训出来的 checkpoint,分别在 IIII/FIII/FIFI 三种
模式下测 depth AbsRel)缺两块:
1. **M0 那一行完全没有数据**——M0 的完整训练被推迟了。
2. **`heldout_batch_eval.py` 现在只测 `IIII` 一种模式**——这张表需要同一个 checkpoint
   在三种模式下都测一遍,脚本需要扩展(`MODE` 目前是硬编码的常量,不难改,但没改)。

### ❌ `tab:offline_preservation`(4.5 节,Action Images init / arm0 / arm6 的离线
action+RGB 指标,含 LPIPS、SSIM)

跟 `tab:headline` 一样卡在 LPIPS/SSIM 没实现,而且这张表还多要一个 SSIM,以及需要
额外跑官方 init checkpoint 和 arm0 checkpoint 的 held-out 评测(目前 `heldout_batch_eval.py`
只指向了 arm6 的 checkpoint)。**目前完全填不了。**

### ⚠️/❌ `tab:closed_loop`(4.5 节,**Initialization / arm0 / arm6** 三列,5 个任务
+ overall)—— **这里有一个需要你决定的矛盾**

新草稿这张表的正文明确写着:

> "This comparison separates domain adaptation from perception co-training. The difference
> between the initialization and arm0 measures the effect of continuing training... The
> difference between arm0 and arm6 measures the additional effect of sharing..."

也就是说**新草稿本身要的是三列(init / arm0 / arm6),而且专门用一段话解释为什么需要
中间的 arm0 这一列**——这正是 `U-CoGen_experiment_plan.md` 最初建议的"三点分解"设计。
但你之前明确说"不需要跑 arm0,用官方 checkpoint 替代",于是 Batch 2 第1项跑的是
**official vs arm6 两列**,没有 arm0。

**这两者对不上。** 有两个选择:
1. **改草稿**:把 `tab:closed_loop` 从三列改成两列(Initialization / arm6),那段解释
   domain-adaptation-vs-perception-co-training 的话也要删掉或改写成"我们只测 net effect,
   不拆分",跟旧草稿 `tab:closedloop` 的 caption("home-advantage comparison, not a model
   ranking")保持一致的框架。Batch 2 第1项跑完就能直接填这张改过的表。
2. **额外补跑 arm0 闭环**:arm0 的 checkpoint 需要能找到(本地已删除,备份状态未知,
   见上面 `tab:fixed_compute` 的讨论),然后再跑一次 n=20/task 的闭环(跟官方那次同样
   ~9-10 小时),凑齐三列。

我目前**没有擅自选**,先把矛盾点写清楚,需要你确认选哪个。

### ⚠️ `tab:per_task`(4.6 节,arm6 按任务分列的 Action/Depth/Seg/Normal)

这张表要的是**按具体任务(5 个)分列**,而不是 seen/unseen 两组聚合。
`heldout_batch_eval.py` 现在的聚合逻辑是按 seen/unseen 分——但原始的
`per_episode.json` 里每条记录本来就带着 episode 路径(能反推出任务名),**所以数据
其实已经在收集了,只是聚合脚本还没写"按任务分组"这个视图**,是一个小的脚本补充,
不是新实验。

### ❌ `tab:trimodal`(4.7 节,可选实验,video+depth+action 三模态同时生成)

草稿自己标注"optional, included only if computational budget permits"。这次没做。
`training/templates.py` 的 `CANONICAL_ORDER` 和 `parse_template` 本身不阻止三模态
模板被解析,但**arm6 从来没有在训练里用过三模态混合**(论文方法部分自己写的:双模态
模板里"辅助流替换 action 槽位而不是扩展序列"),所以要填这张表需要一次新的训练
(哪怕只训 1000-2000 步),不是白拿现成 checkpoint 能填的。优先级低,标记可选。

---

## 四、需要你决定的事(按优先级)

1. **`tab:closed_loop` 到底要不要 arm0 这一列**(上面第三节详细写了矛盾)。这个决定会
   直接影响要不要再开一个 ~9-10 小时的 arm0 闭环 campaign,以及要不要先确认/下载 arm0
   的 checkpoint 备份。
2. **LPIPS/SSIM 要不要现在补代码**——`tab:headline` 和 `tab:offline_preservation` 两张
   表都卡在这里,不补代码这两张表的 RGB 列永远是 `[XX]`。工作量不大(装包 + 在
   `heldout_batch_eval.py` 里加一个 bare `video` 模板的生成+打分分支),但目前完全没做。
3. **`tab:fixed_compute` 要不要做,以及旧 arm1(A+D+S)找不到备份的话要不要重训**——
   这张表本身还有"256 vs 512 数据树不可比"的方法论问题,值得先讨论清楚再决定要不要
   投入算力。
4. **`tab:conditioning_results` 要不要扩展 `heldout_batch_eval.py` 支持 FIII/FIFI**——
   如果要,这是几十行代码的改动(`MODE` 从硬编码常量变成一个循环),但也意味着 M0
   补跑完之后,评测时间会变成现在的 3 倍(3 种模式 × 16 episode × 4 模态)。

## 五、检查现在的实验进度

```
tail -50 logs_official_closedloop_postfix.txt
tail -50 logs_specialist_depth.txt; tail -50 logs_specialist_seg.txt
tail -50 logs_heldout_batch_eval.txt
python -c "import json; d=json.load(open('reports/closedloop/rollout_official_125750_postfix_n20.json')); print(len(d['results']),'done'); print(d['errors'])"
ls outputs/specialist_depth__seed42_fi3_512_aug_sr/ outputs/specialist_seg__seed42_fi3_512_aug_sr/
nvidia-smi
```

完整的执行细节和理由在 `ROBUSTNESS_PLAN_aug29.md`(仓库根目录)和
`/root/.claude/plans/sunny-exploring-scone.md`。

---

## 六、根据你的三个决定,重新设计后的最终方案(2026-08-29 23:00 UTC)

你的三个决定:
1. **`tab:closed_loop` 删掉 arm0 列**——已改,`draft_aug_29.md` 现在这张表是 Initialization/arm6
   两列,caption 和正文都改成跟旧草稿一致的"home-advantage comparison, not a ranking"框架。
   `tab:offline_preservation`(离线指标)**保留了 arm0**——那张表不需要跑闭环仿真,便宜,
   而且论文自己解释"域适应 vs 感知联训"分解那段话在离线指标上仍然成立,只是闭环部分因为
   太贵改成只测两端点。
2. **LPIPS/SSIM 确实需要改评测代码,已经改完**——装了 `lpips`(AlexNet trunk,权重已下载
   验证过)、用 `skimage.metrics.structural_similarity` 算 SSIM,在 `heldout_batch_eval.py`
   里新增了 `score_video()`,而且**不需要额外的生成调用**:因为 depth/seg/normal/action
   四个模板本来就都以 `video` 作为 anchor 段,顺手把这个已经生成出来的 RGB 段也打分了。
   额外的好处:因为每个 episode 会在 4 个不同模板语境下各生成一次 RGB,现在还能顺便检验
   "RGB 质量是否真的跟搭配的模态无关"这个论文自己隐含的假设(`summary.json` 里
   `by_via` 字段)。
3. **`tab:fixed_compute`(旧的 arm0→arm4→arm1→arm6 递增表)重新设计,不再尝试恢复/重训
   旧 arm4、旧 arm1**——原因见下面的"最合理方案"。

### 最合理的整体实验方案(综合所有信息后的判断)

核心原则:**所有拿来比较的 arm 必须在同一棵数据树(512_aug/fi3)、同一套 recipe
(相同 seed/lr/warmup,只从官方 step125750 冷启动)上,否则任何差异都可能是 tree 混淆,
不是模态混淆。** 这正是 `train_arm.sh` 自己的设计哲学("arm 只能有一个变量"),旧的
arm4/旧arm1 是在已经删除的 256 分辨率树上训的,恢复它们(哪怕找到备份)也没法跟 arm6
公平比较,重训它们又要多烧两个几千步的训练(而且"menu 从 2 个模态到 3 个模态"这个中间
点,对回答"sharing 到底有没有代价"这个核心问题,不如"specialist vs arm6 在匹配更新数下"
这个对比来得直接)。

**所以最终的 arm 名单收敛成 5 个,只在 512_aug 树上、全部从官方 checkpoint 冷启动:**

| arm | 模板 | 状态 |
|---|---|---|
| arm0 | video+action@1.0 | 🟡 新排队(`wait_and_launch_arm0_512aug.sh`),10000 步(跟 arm6 同预算) |
| depth specialist | video+depth@1.0 | 🟡 训练中,~4000 步 |
| segmentation specialist | video+segmentation@1.0(scene_roles) | 🟡 训练中,~4000 步 |
| normal specialist | video+normal@1.0 | 🟡 新排队(`wait_and_launch_normal_specialist.sh`),4000 步 |
| arm6 | 四模态混合(M2) | ✅ 已完成,10000 步 |

这 5 个 arm 一次性覆盖:
- `tab:headline`(arm0 + 三个 specialist + arm6 的五模态对比)
- `tab:matched_exposure`(新表,取代旧的 `tab:fixed_compute`——每个 specialist 在跟 arm6
  相同"该模态更新次数"时的对比,直接回答 dilution vs interference)
- `tab:offline_preservation`(init / arm0 / arm6)
- `tab:closed_loop`(init / arm6,官方闭环已经在跑)
- M0/M1/M2 conditioning 矩阵单独用 arm6 自己的三个变体(M1 已训完,M2=arm6,M0 仍然推迟,
  因为它必须是独立的一次完整 10000 步冷启动,不属于这 5 个 arm 的范围)

**没有采纳的方案**:重训旧 arm4/旧 arm1 凑一条"menu 从 1→2→3→4"的平滑曲线。已经在
`draft_aug_29.md` 里用 `%%` 注释记录了这个决定和理由,如果之后审稿阶段真的需要更平滑的
曲线,是一个有据可查的"P2 待办",不是被遗忘。

### 现在一共有 6 个任务在跑/排队

| # | 内容 | GPU | 状态 |
|---|---|---|---|
| 1 | 官方 checkpoint 闭环,n=20/task | GPU 0 | 跑中 |
| 2 | depth specialist | GPU 3,4 | 跑中,~4000步 |
| 3 | seg specialist | GPU 5,7 | 跑中,~4000步 |
| 4 | held-out 批量评测(arm6,16 episode,现在含 RGB LPIPS/SSIM/PSNR) | 排队重试中 | 撞过一次 GPU 竞态,已用带重试的包装脚本重新拉起 |
| 5 | **新增**:arm0 fresh(512_aug 树,10000 步) | 排队等 2 块 GPU | `wait_and_launch_arm0_512aug.sh` |
| 6 | **新增**:normal specialist(4000 步) | 排队等 2 块 GPU | `wait_and_launch_normal_specialist.sh` |

5、6 用了比之前更稳的 GPU 探测逻辑(不只看 `nvidia-smi` 报的空闲显存,还会用一次真实的
CUDA 分配去验证,针对的就是任务 4 撞到的那种"检查时空闲、申请时被抢"的竞态)。

### 仍然明确推迟/不做的

- **M0 完整重训**(58 小时/2GPU 量级)——等上面 6 个任务腾出 GPU 再排。
- **tri-modal(video+depth+action 同时生成)**——草稿里标"optional",且需要一次新训练
  (arm6 从来没在训练里用过三模态模板)。
- **新模态(optical flow)**——需要人工验证新 codec,不适合无人值守。
- **旧 arm4/旧 arm1 的恢复或重训**——本节已经论证过为什么不值得。

---

## 七、进一步简化:arm0 → action specialist(2026-08-30)

你指出 arm0(10000 步)跟其他 3 个 specialist(4000 步)预算不一致——这正是
`sec:exp_sharing` 本来要排除的那种不对称(action 已经在官方 checkpoint 里预训练了
125,750 步,不该再单独给它 2.5 倍于其他 specialist 的续训预算)。处理方式:

1. **停掉了 `wait_and_launch_arm0_512aug.sh`**(还在排队,没真正开始训练,直接 kill 干净)。
2. **新增 `wait_and_launch_action_specialist.sh`**:`video+action@1.0`,4000 步,
   `CKPT_EVERY=500`,跟 depth/seg/normal specialist 完全同预算、同 recipe,已排队等 GPU。
3. **`draft_aug_29.md` 里所有 "arm0" 已经统一改名成 "action specialist"**(Models 列表、
   `tab:headline`、`tab:matched_exposure`、`tab:offline_preservation`、闭环部分的说明文字),
   跟 depth/segmentation/normal specialist 用同一套命名。**官方 checkpoint("Action Images
   init")保持不变、零样本、不再续训**,继续作为"域适应之前"的参照点(`tab:offline_preservation`
   第一行、`tab:closed_loop` 的 Initialization 列)。

现在的 arm 名单(全部同 512_aug 树、同 recipe,只从官方 checkpoint 冷启动):

| arm | 模板 | 步数 | 状态 |
|---|---|---|---|
| Action Images init | 官方 checkpoint,零样本 | 125,750(不再训) | ✅ 已有 |
| action specialist | video+action@1.0 | 4000 | 🟡 新排队 |
| depth specialist | video+depth@1.0 | 4000 | 🟡 训练中 |
| segmentation specialist | video+segmentation@1.0 | 4000 | 🟡 训练中 |
| normal specialist | video+normal@1.0 | 4000 | 🟡 排队中 |
| arm6 | 四模态混合(M2) | 10000 | ✅ 已完成 |

四个 specialist 现在预算完全一致,是一个更干净的"cost of sharing"对照组。

---

## 八、matched-exposure 对比要用哪个 checkpoint(2026-08-30)

你发现了一个真实问题:depth/seg/normal specialist 跑 4000 步(100% 该模态)= 4000 次更新,
而 arm6 对这三个模态各自只分配 0.2 概率,10000 步下来每个模态只有约 2000 次更新——
specialist 的 final checkpoint 比 arm6 多了整整一倍曝光,不是公平的 matched-exposure
对比点。

**结论**:
- **depth/segmentation/normal specialist**:matched-exposure 对比要用它们的
  **checkpoint-2000**(2000 次更新,精确匹配 arm6),不是 final checkpoint-4000。
  final checkpoint-4000 单独作为"specialist 自己在两倍于 arm6 曝光下的天花板"来报告,
  是另一个数,不能跟 arm6 直接比。这也是当初把 `CKPT_EVERY=500` 设置得比默认更密的
  原因——就是为了留出这个中间点。
- **action specialist**是例外:arm6 给 action 分配 0.4 概率(是其他三个的两倍),
  10000 步下来约 4000 次 action 更新,正好等于 action specialist 4000 步、100% 的
  更新数——它的 final checkpoint 一个数就能同时当"自己的天花板"和"matched-exposure
  对比点"用,不用取中间 checkpoint。

已经把这个说明直接写进 `draft_aug_29.md` 的 "Matched modality exposure" 段落里,免得
以后填表的人直接拿 depth/seg/normal specialist 的 final checkpoint 去比,产生一个
看似公平实则偏向 specialist 的错误结论。**现在正在跑的训练任务不需要改**(4000 步 +
`CKPT_EVERY=500` 已经覆盖了这个需求),只是评测时要记得挑对 checkpoint。

---

## 九、停掉 depth/seg、挂上 normal + held-out 评测(2026-08-30 16:00 UTC)

### 9.1 停掉两个已过 2k 步的 specialist

depth 和 seg 都已越过 **checkpoint-2000**(matched-exposure 唯一需要的那个点),
按你的判断停掉腾 GPU。**杀之前先验证了 checkpoint 完整性**(zip 中央目录可读、
每个 12G,没有截断),确认 `checkpoint-2000` 可用才动手:

| arm | 实际停在 | 保留的 checkpoint |
|---|---|---|
| depth specialist | 2972 / 4000 | 500, 1000, 1500, **2000**, 2500 |
| seg specialist | 3067 / 4000 | 500, 1000, 1500, **2000**, 2500, 3000 |

**代价**:失去了 step-4000 那个"specialist 天花板"读数,现在只能用 2500 / 3000 代替。
考虑到 arm6 自己的 depth AbsRel 在 4000 步前就已经拉平(0.087@1500 → 0.085@4000),
而 specialist 每步 100% 该模态、收敛只会更快,这个代价可以接受。**matched-exposure
对比(checkpoint-2000)完全不受影响**,那才是核心结论依赖的点。

### 9.2 当前 GPU 分配(正好留 2 块给别人)

| GPU | 任务 | 状态 |
|---|---|---|
| 0, 2 | action specialist | 1445/4000 |
| 3, 4 | **normal specialist**(新启动) | 25/4000 |
| 5 | **held-out 批量评测队列**(新启动) | 跑第 1 个 checkpoint |
| **1, 6** | **空闲** | 留给别人 |

评测队列 `scripts/run_heldout_eval_queue.sh` 会依次跑(已有 `summary.json` 的自动跳过):
1. `arm6_10000`(4 模态,~3.3h)
2. `depthspec_2000_matched` / `segspec_2000_matched`(各 1 模态,~50min)
3. `depthspec_2500_ceiling` / `segspec_3000_ceiling`

### 9.3 ⚠️ 修掉一个真 bug:`--gpu N` 被静默忽略,实际全跑在 GPU 0

启动评测时脚本打印 `using GPU 5`,但 OOM 报错显示**实际在 GPU 0 上分配**。查证:

```
after torch     -> cuda initialized: False
after lpips     -> cuda initialized: False
after inference -> cuda initialized: True   ← 就是这一行
```

`from inference import build_pipeline` **在 import 时就初始化了 CUDA**,而所有 eval
脚本都在模块顶层做这个 import,`main()` 里再设 `os.environ["CUDA_VISIBLE_DEVICES"]`
已经太晚了。于是 `--gpu N` 被静默忽略,进程落在物理 GPU 0 上,却照样打印
"using GPU N"。

**这正是 `modality_mode_grid.py` 自己注释里描述过的失败模式**,只是没人想到它对
`--gpu` 显式指定也成立。

**修法**:在 shell 层 `export CUDA_VISIBLE_DEVICES=$GPU` 再启动 python,mask 之后
目标卡在进程内就是 index 0,所以传 `--gpu 0`。已写进 `run_heldout_eval_queue.sh`
并注明原因。

> ⚠️ **同样的隐患还在 `scripts/tag_swap_ablation.py` 和 `scripts/modality_mode_grid.py`
> 里**。之前没暴露,是因为那些运行碰巧目标就是 GPU 0(或者用 `CUDA_VISIBLE_DEVICES=`
> 前缀启动的)。**以后用这两个脚本指定非 0 号卡时,必须用 shell 层 mask,不要信 `--gpu`。**

### 9.4 ⚠️ 另一个坑:`pkill -f <pattern>` 会杀掉自己

`pkill -f run_heldout_eval_queue` 把**我自己的 shell 也杀了**(退出码 144),因为
`pkill -f` 匹配完整命令行,而我的命令行里就包含这个字符串。踩了两次。
**以后杀进程用 `pgrep` 拿 PID、过滤掉自己再 `kill`,不要直接 `pkill -f` 一个会出现在
自己命令行里的模式。**

---

## 十、闭环结果的重大重新解读:必须按"训练过/没训练过"拆开(2026-08-30)

你在 `EXPERIMENT_OUTLINE.md` 里留的 TODO(「应该改成 held-out 的测试啊？？！」)促成了
这次核查,结论比"重跑一遍"重要得多。

### 10.1 查证:闭环 harness 本来就支持 held-out

`eval/rollout.py` 有一等公民的 `--variation N`,`eval/rollout_env.py` 会对 RLBench 真实
variation 数做校验,scene seed 也是按 `(task, variation, trial)` 派生的。**我们只是用了
默认的 variation 0。**

### 10.2 但更重要的发现:5 个闭环任务里有 2 个根本不在训练集里

对照数据树的 16 个训练任务:

| 闭环任务 | 在训练树里? |
|---|---|
| push_buttons | ✅ variation0,50 episodes |
| meat_off_grill | ✅ variation0,50 episodes |
| open_drawer | ✅ variation0,50 episodes |
| **close_box** | ❌ **整个任务都没有** |
| **close_drawer** | ❌ **整个任务都没有** |

也就是说这张表**早就含 held-out 成分,而且是比"换 variation"更强的任务级 OOD**。
问题从来不是缺 held-out,而是**把两类任务混进了一个 pooled 数字**。

### 10.3 拆开之后,结论完全不同

| 分组 | 官方 init | arm6 | Δ | b/c | p |
|---|---|---|---|---|---|
| **训练分布内**(pb / mog / od) | 6.7% | **35.0%** | **+28.3** | 20/3 | **0.0005** |
| **从没训练过**(close_box / close_drawer) | **50.0%** | 35.0% | −15.0 | 4/10 | 0.180 |
| 全部 5 个合计 | 24% | 35% | +11 | 24/13 | 0.099 |

**之前报的「合计 +11pp,p=0.099,不显著」是假象**——两个方向相反、量级相当的效应
互相抵消了。拆开后训练分布内是 **+28.3pp,p=0.0005,高度显著**。

这是**常规的微调权衡**(域内大幅提升、牺牲预训练模型的广度),不是 co-generation
特有的问题。`close_drawer` 是代价集中的地方:一个 arm6 从没见过、而官方 checkpoint
恰好特别强(80%)的任务。

### 10.4 论文已同步修改

`draft_aug_29.md` 的 `tab:closed_loop` 已改成两个分组的表格,结论从含糊的"保持"
改成有据的:**训练过的地方全面提升(+28.3pp, p=0.0005),没训练过的地方有损失
(−15pp)且我们不回避**;并明确写了这张表**不支持**任何关于"泛化到没见过的任务"的说法。

### 10.5 仍值得补(未做)

把 `push_buttons` / `meat_off_grill` / `open_drawer` 在 **variation 1** 上再跑一次
(这三个任务在数据树里都有 variation 1/2),补上"同任务、没见过的配置"这一档,
填满 held-out 的中间层级。约 6-7 小时一次 campaign:

```bash
# 在 run_official_closedloop_postfix.sh 基础上:
#   --variation 1  --tasks push_buttons meat_off_grill open_drawer
```

---

## 十一、论文改动汇总(2026-08-30)

`draft_aug_29.md` 本轮的改动:

| 改动 | 内容 |
|---|---|
| ✅ `tab:closed_loop` | 填入真实数据,并按训练过/没训练过分组(见第十节) |
| ✅ **新增 §4.5 anchor vs tag** | `tab:tag_swap`,4 episode 真实数据。正确 tag 平均排名 2.7/4(随机 2.5),拿掉 anchor 则崩溃(depth 0.090→0.808,normal cos 0.895→**−0.263 符号翻转**)。明确写出实践后果:**新模态不能只靠 prompt 引入,需要自己表示空间里的 anchor** |
| ✅ 新增附录 `app:overtraining` | loss 不能给 checkpoint 排序;标注了它来自更早的 arm、在 anchor fix 之前,绝对值不可与新表混用 |
| ✅ arm0 → action specialist | 全文统一改名,预算与其他 3 个 specialist 对齐(4000 步) |
| ✅ `tab:matched_exposure` | 取代旧的 `tab:fixed_compute`;明写 depth/seg/normal 要用 **checkpoint-2000**,action 用 final |
| ✅ 附录 mixture 表 | 补上 M1 行 |
| ✅ **补了 7 个宏定义** | `\ptag \best \tabref \figref \secref \appref \tmpl` 之前**用了但从没定义**,文件根本编译不过 |
| ✅ 修正 held-out 表述 | 原文说"所有主表都在 held-out 上",但闭环不是,已改 |

**占位符 `[XX]` 从 36 降到 25**,减少的部分来自 E4(anchor vs tag)和 E5(闭环)的真实数据。

### 遗留的编译阻塞项(先前就存在,非本轮引入)

- `\input{Figs/mask_full_grid}` —— `Figs/` 目录不存在(只有 `figures/`),`fig:layout` 悬空
- `\ref{app:camera}` —— Preliminaries 引用了一个从没写的相机附录

---

## 十二、Table 1 能填多少?+ 一个我自己写出来的 bug(2026-08-30 17:00 UTC)

### 12.1 问题:有两个 specialist 了,能不能填 Table 1?

**能填一部分,而且评测已经在跑了。** `tab:headline` 需要 5 行:

| 行 | checkpoint | 能填? |
|---|---|---|
| action specialist | 训练中 1449/4000 | ❌ 等训完 |
| **depth specialist** | ✅ ckpt 500–2500 | ✅ **评测已排队** |
| **segmentation specialist** | ✅ ckpt 500–3000 | ✅ **评测已排队** |
| normal specialist | 刚启动 30/4000 | ❌ 等训完 |
| **arm6** | ✅ step10000 | 🟡 **正在评测中** |

也就是说 **5 行里现在能拿到 3 行**(arm6 + depth + seg),另外 2 行等 action/normal 训完。

### 12.2 一个 checkpoint 要出两个数,别搞混

评测队列对 depth/seg **各跑两个 checkpoint**,填的是**不同的表**:

| checkpoint | 填哪张表 | 含义 |
|---|---|---|
| `checkpoint-2000` | `tab:matched_exposure` | 与 arm6 相同的模态更新次数(2000)——回答"共用 head 有没有干扰" |
| `checkpoint-2500`(depth)/ `checkpoint-3000`(seg) | `tab:headline` | specialist 自己的天花板——回答"专用模型能好多少" |

队列顺序:`arm6_10000`(4 模态)→ `depthspec_2000_matched` → `segspec_2000_matched`
→ `depthspec_2500_ceiling` → `segspec_3000_ceiling`。已有 `summary.json` 的自动跳过。

**进度**:arm6 已完成 26/64 个 (episode,模态) 单元,约 175s/个,LPIPS/SSIM/PSNR 正常产出
(例:`close_jar` depth AbsRel=0.124、seg mIoU=0.392、video LPIPS=0.260)。
arm6 还需约 2h,之后 4 个 specialist 评测各约 47min。

### 12.3 ⚠️ 我写的 bug:action 指标全被写成 None

评测日志里 `action pos_err_m=None`。查证是 `scripts/heldout_batch_eval.py` 我自己写的:

```python
rec.update(G.score_action(...))
rec["value"] = rec["pos_err_m"] if "pos_err_m" in rec else None   # ← BUG
```

`G.score_action` 返回的是 `{"metric": "pos_err_m", "value": <中位数>, "pos_err_mean_m": ...}`
——**`"pos_err_m"` 是指标的名字,从来不是返回字典的 key**。所以那个 `in rec` 判断恒为假,
把 `score_action` 已经算好放在 `value` 里的中位数**覆盖成了 None**。

**损失范围**:只有 action 的**位置误差中位数**丢了。同一条记录里
`pos_err_mean_m` / `rot_err_med_deg` / `r_peak` / `sparsity_dark_frac` 都还在,
depth/seg/normal 的所有单元也完全不受影响。

**已修**:
1. 删掉那两行错误的覆盖(inline 一处 + 收尾循环一处),`score_action` 的返回值直接用。
2. **顺带加了断点续跑**:启动时读取已有的 `per_episode.json`,`value` 非 None 的单元
   直接跳过。这样重跑同一个 tag 的 `--modalities action` 只会重算被写坏的 action 单元,
   不用把 3 小时的 depth/seg/normal 全部重做。(附带好处:以后中断不再损失整轮。)
3. 排了 `scripts/run_arm6_action_refix.sh`,**等当前队列跑完自动接上**,重算 arm6 的
   action 单元并重写 `summary.json`。

> 教训:`score()` 和 `score_action()` 的返回约定不一样(前者不设 `value`,后者设),
> 我按前者的习惯去"补" `value`,结果把后者已经算对的值覆盖了。**改动别人的返回值之前
> 先看它到底返回了什么。**

---

## 十三、放弃 ceiling、8 卡全开、补 variation-1 闭环(2026-08-30 18:35 UTC)

### 13.1 按你的决定:只做 matched-exposure,不做 specialist 天花板

原评测队列排了 4 个 specialist 任务(2 个 matched + 2 个 ceiling),现在**砍掉 ceiling**:

- 杀掉队列的**外层 wrapper**(保留正在跑的 arm6 评测进程),这样队列不会走到
  `depthspec_2500_ceiling` / `segspec_3000_ceiling` 两项。
- 两个 matched-exposure 评测(`checkpoint-2000`)**改成并行**跑在刚空出来的 GPU 1 和 6 上,
  不再排在 arm6 后面串行等 ~2h。

> 影响:`tab:headline` 里 specialist 那几行现在也用 **checkpoint-2000**(与 arm6 曝光对齐)
> 的数,而不是 specialist 自己的天花板。**这一点要在论文里写清楚**——否则读者会以为
> "specialist 就只能到这个水平"。等以后想补天花板,checkpoint-2500/3000 还在磁盘上,
> 随时可以再评。

### 13.2 f0f0 补测:已按你的要求取消

原本在 GPU 7 上挂了 arm6@10000 的 f0f0 补测(为了让 anchor-vs-tag 的对比落在同一个
checkpoint 上),你说不测,已 kill 并删掉脚本和日志。`sec:exp_anchor` 保持现状:
f0f0 用 step 9000、IIII 用 step 10000,论文里已注明。效应量 ~9 倍,不会被 1000 步的
差异解释。

### 13.3 补上 variation-1 闭环(第十节留的缺口)

GPU 7 改跑 **variation 1 的闭环**,补上"同一个任务、没见过的配置"这一档:

- 只跑 **push_buttons / meat_off_grill / open_drawer** 三个任务——`close_box` 和
  `close_drawer` 根本不在训练树里,对它们来说所有 variation 都同样陌生,
  跑 variation 1 不增加任何信息。
- 两个模型各自用**自己训练时的协议**(与 variation-0 那次一致)。
- arm6 已启动(GPU 7,3 任务 × 20 = 60 rollout,约 6h);
  **官方那半边是必须的**(配对检验,`aggregate_closedloop.py` 会拒绝不配对的比较),
  已挂 `wait_and_launch_official_var1.sh`,一有空卡自动接上。

跑完之后,闭环就有完整的三档 held-out 层级:

| 层级 | 任务 | 状态 |
|---|---|---|
| 训练过的配置 | pb / mog / od @ variation 0 | ✅ 有 |
| **同任务、没见过的配置** | pb / mog / od @ **variation 1** | 🟡 跑中 |
| 完全没训练过的任务 | close_box / close_drawer | ✅ 有 |

### 13.4 当前 8 卡全满

| GPU | 任务 | 进度 |
|---|---|---|
| 0, 2 | action specialist | 1851/4000 |
| 1 | `depthspec_2000_matched` 评测 | ~47min |
| 3, 4 | normal specialist | 455/4000 |
| 5 | arm6 held-out 评测 → 接 action 重算 | 47/64 |
| 6 | `segspec_2000_matched` 评测 | ~47min |
| 7 | arm6 variation-1 闭环 | ~6h |

排队中:官方 variation-1 闭环(等空卡)。

### 13.5 M0 决定不跑(2026-08-30)

**决定:M0 暂时去掉。** 不排 58 小时 × 2 卡的冷启动训练。

**对论文的直接影响,必须处理**:`tab:conditioning_results` 原本设计成 M0/M1/M2 三行 ×
IIII/FIII/FIFI 三列。现在:

- **M0 那一行没有数据。**
- 剩下的 M1 和 M2 **还不是对等的两个点**:M2 就是 arm6 本身(从官方 checkpoint 冷启动
  跑满 10000 步);M1 是**从 arm6 的 step10000 再续训 3000 步**的廉价消融。
  所以 M1 那行严格来说是 "arm6 + 3000 步 M1",不是"用 M1 从头训出来的模型"。

**⚠️ 于是 §4.4「conditioning 分布决定模型能力」这一节的论证强度被大幅削弱**——
本来想要的是三点趋势(M0 的 FIFI 最好 → M2 的 IIII 最好),现在只剩一个不对等的两点。

**还能诚实报告的**(不需要 M0):
1. **同一个 checkpoint 在三种模式下的表现**(arm6 在 IIII / FIII / FIFI 上的 depth/seg/normal)
   ——这能说明 M2 训出来的模型 `IIII` 稳、`FIFI` 弱,是"训练分布决定能回答哪些问题"的
   **一半**证据(旧草稿 `tab:crosstask` 就是这个,只是当时用的是 4 个 held-in episode)。
2. **M1 作为一次干预**:在 arm6 基础上把 conditioning 分布掰向均衡,看 FIFI 是否回升。
   这是"干预"证据,但因为不是冷启动,不能和 M2 并列成一张对称的表。

**建议的论文改法**(待确认):把 §4.4 从"三点趋势表"降级成"单模型的模式网格 + 一次
M1 干预",并明说完整的 M0/M1/M2 冷启动矩阵是 future work。**不要**把 M1 和 M2 并排放进
一张看起来对称的表里而不加说明——那会让读者以为两者训练预算相同。

---

## 十四、为"没训练过的任务"渲染感知评测数据(2026-08-30 20:00 UTC)

### 14.1 起因:两个 harness 的数据依赖完全不同

| | 开环感知评测 `heldout_batch_eval.py` | 闭环 `eval/rollout.py` |
|---|---|---|
| 数据来源 | **必须读磁盘 GT**(`_load_depth` / `_load_mask` / `_scene_role_lut` / `action_7d`) | **纯跑仿真器**,`reset_to_new_demo()` 调 `task.get_demos(live_demos=True)` 当场生成 |
| 任务定义 | 只能来自数据树 | RLBench 自带任务库(`name_to_task_class`),实测 `close_box`/`close_drawer` 都在 |
| **能测没采集过的任务吗** | ❌ **原理上不行**(episode 不在树里连索引都建不出来) | ✅ **可以**,这正是 `tab:closed_loop` 里有 close_box/close_drawer 的原因 |

所以此前"感知能不能泛化到没见过的**任务**"根本无法回答——不是没测,是**没有 GT 可比**。
补这个洞唯一的办法就是给这两个任务渲染一批带 GT 的数据(**只用于测试,绝不进训练**)。

### 14.2 ⚠️ 必须写到独立的树,不能写进训练树

训练用 `--variations 0` 扫**整棵树**。若把 `close_box/variation0` 写进
`data/rlbench_selfgen_512_aug`,以后任何一次重训都会静默把它们吃进训练集,
"从没训练过"这个性质当场失效。→ 输出到 `data/rlbench_unseen_tasks_512_aug`。

### 14.3 ✅ 场景种子能和闭环对上(已验证)

- `gen_dataset.py:261` 在 `get_demos` 前调 `np.random.seed(seed + v*1_000_003 + attempt*100003)`,
  **在 v=0、attempt=0 时就等于 `np.random.seed(seed)`**。
- `rollout_env.reset_to_new_demo()` 调 `np.random.seed(trial_seed(task,0,trial))`。

→ **把闭环的 trial_seed 当作 `--seeds` 传进去,物体布局就与闭环 rollout 逐一对应。**
已交叉验证:算出来的 seed 与闭环 json 里记录的 `scene_seed` **完全一致**
(close_box `[296204, 210343, 541322, 63022...]`、close_drawer `[124879, 581354, ...]`)。

> ⚠️ 但**像素级不完全相同**:这批渲染带 colosseum 增强(与训练分布一致),
> 而闭环 env 跑的是无增强的原始 RLBench 场景。**布局对应,外观不对应。**
> 选 aug 而不是 no-aug 是刻意的——这批数的用途是和其他 16 个任务横向比,
> 视觉域必须是模型训练时的那个域,否则"没见过的任务"会和"没见过的视觉条件"混淆。

### 14.4 ❌ 已知阻塞:segmentation 没有 role 定义

`ttd/src/percep/scene_segments_gen.py` 的 `TASK_ROLES` **恰好只覆盖 16 个训练任务**,
`close_box` / `close_drawer` 没有条目。`resolve_roles()` 对没有条目的任务走
`TASK_ROLES.get(task, {})` → 任务物体全部落进 `unmapped` → 像素解码成 `unknown`。

影响范围:

| 模态 | 能不能测 |
|---|---|
| **depth** | ✅ 可以,GT 直接来自仿真器,不需要 role 映射 |
| **normal** | ✅ 可以,由 depth 推导 |
| **action** | ✅ 可以 |
| **segmentation** | ❌ 需要**手写** `close_box` / `close_drawer` 的 TASK_ROLES 条目 |

写条目本身不难(每个约 3-5 行,形如 `"jar0": "goal", "jar_lid0": "distractor"`),
但需要先知道这两个任务里的物体名——要从渲染出来的 `handles.json` 里读。
而且这个文件在 **ttd 仓库**里,不是本仓库。

**当前策略**:先渲染,拿到 `handles.json` 看清物体名,再决定要不要补 seg。
**depth + normal 两个模态不受影响,可以先出结果**——这已经足够回答
"感知能不能泛化到没见过的任务"的主要部分。

### 14.5 渲染成本(来自 `gen_launch_512.sh` 实测表)

| 配置 | 分辨率 | 核/worker | 秒/episode | 核秒/episode |
|---|---|---|---|---|
| `LP_NUM_THREADS=4` | 512 | 2.7 | 211 | ~570 |

40 个 episode(2 任务 × 20,与闭环同规模)≈ **6.3 核时**,机器有 256 核。
并行跑的话墙钟时间只有几分钟量级。**成本可以忽略。**

### 14.6 踩到的两个环境坑

1. **必须 `xvfb-run`**:CoppeliaSim 直接跑会 `could not connect to display` 并 core dump。
2. **必须手动加 `PYTHONPATH=/workspace/ttdu/robot-colosseum`**:`colosseum` 包没装进 conda env,
   `gen_launch_512.sh:55` 是靠 PYTHONPATH 挂上去的。

两条都已写进 `scripts/gen_unseen_tasks.sh`。

---

## 十五、✅ Table 1 的第一批真实数字:共享 head **没有可检测的代价**(2026-08-30 20:10 UTC)

arm6 / depth specialist@2000 / seg specialist@2000 三个 held-out 批量评测全部跑完
(各 16 episode = 8 seen + 8 unseen)。

### 15.1 结果(held-out = variation1,论文该报的那一列)

| | depth AbsRel ↓ | seg mIoU ↑ | normal cos ↑ |
|---|---|---|---|
| **arm6**(一个 checkpoint,4 模态) | 0.1536 [0.1155, 0.1906] | 0.4609 [0.3750, 0.5560] | 0.8923 [0.8708, 0.9120] |
| depth specialist@2000 | **0.1438** [0.1179, 0.1670] | — | — |
| seg specialist@2000 | — | **0.4729** [0.4029, 0.5523] | — |

RGB(arm6 的 video 段):unseen LPIPS **0.2255** / PSNR 15.37dB / SSIM 0.7438(n=32)

### 15.2 关键:用**配对检验**,不是看 CI 重叠

两个模型跑的是**同一批 8 个 episode**,所以配对检验功效高得多:

| 模态 | arm6 − specialist | 95% bootstrap CI | Wilcoxon p | 结论 |
|---|---|---|---|---|
| depth AbsRel | **+0.0098** | [−0.0107, +0.0318] | 0.742 | CI 跨 0 |
| seg mIoU | **−0.0120** | [−0.0438, +0.0161] | 0.844 | CI 跨 0 |

**两个模态、两个方向相反的微小差异,都跨 0、p 都远大于 0.05
⇒ 在匹配曝光下,共享一个 head 没有可检测的代价。**

这正是 `sec:exp_sharing` 想要的结果,而且是诚实的:n=8 时 CI 宽度只能排除
**大于约 ±0.03 AbsRel / ±0.04 mIoU 的退化**,排除不了更小的代价。所以论文写的是
"没有可检测的效应",不是"完全没有代价"。

**但它排除掉的正是这一节要测的失败模式**:四个模态共用一个输出空间,不会在
匹配曝光后仍然留下大的干扰惩罚。同时它也定位了固定步数下的差距来源——既然匹配曝光后
差距消失,固定步数下的差距就应归因于**采样稀释**而非参数竞争。

### 15.3 顺带:arm6 没有泛化间隙

| | seen(训练样本) | unseen(held-out) |
|---|---|---|
| depth AbsRel | 0.1357 | 0.1536 |
| seg mIoU | 0.4855 | 0.4609 |
| normal cos | 0.8945 | 0.8923 |

三个模态 seen/unseen 几乎持平,与之前"arm6 已收敛到平台、无泛化间隙"的记录一致。

### 15.4 论文已更新

- `tab:matched_exposure`:填入真实配对统计(diff / CI / p),并改写了正文,
  明确"没有可检测效应 ≠ 零代价",以及 n=8 能排除多大的效应。
- `tab:headline`:填入 arm6 的 depth/seg/normal/LPIPS + 两个 specialist 的数;
  caption 注明数字来自 variation1 held-out、specialist 用匹配曝光的 checkpoint。
- **占位符 25 → 22。**

### 15.5 还缺

- **action 那一行**:被 §12.3 那个 bug 写成 None,`run_arm6_action_refix.sh` 已挂上,
  等 arm6 评测进程退出后自动重算(已加断点续跑,只重算 action 单元)。
- **normal / action specialist 两行**:训练中(normal 754/4000,action 2138/4000)。

---

## 十六、unseen-task 渲染:进行中(2026-08-30 20:10 UTC)

40 个 episode(close_box / close_drawer 各 20,用闭环的 trial_seed)并行渲染中,
NPAR=16,预计 ~15 分钟。探针 episode 已验证:

- ✅ `meta.json` 的 seed = 296204,**与闭环 close_box trial 0 的 scene_seed 完全一致**
- ✅ GT 齐全:`actions.npy` / `handles.json` / 4 个视角的 `depth.npz` / `mask.npz` / `rgb`
- ✅ 写在**独立的树** `data/rlbench_unseen_tasks_512_aug`,不污染训练树

### 16.1 segmentation 的门槛比预想低

- ✅ **`seg_targets.json` 不是必需的**——`_scene_role_lut` 的注释明确写了
  "A missing/unresolvable seg_targets.json is NOT fatal"(它只做 instruction 覆盖,
  没有它 base_role 依然成立)。**只需要 `scene_segments.json`。**
- ✅ close_box 只有 **2 个任务物体**:`box_base` / `box_lid`。其余 15 个
  (Panda_link0-7 / gripper / floor / Wall3 / diningTable / workspace)
  全部已被 `ROBOT_ARM_NAMES` / `GRIPPER_NAMES` / `BACKGROUND_NAMES` 覆盖。
- ⇒ 补 `TASK_ROLES` 条目只是几行。按最接近的 close_jar 惯例:
  `"box_base": "goal"`(承接盖子的主体)、`"box_lid": "distractor"`(可被指令提升为 target)。

> ⚠️ 但 `scene_segments_gen.py` 在 **ttd 仓库**里(`ttd/src/percep/`),不是本仓库。
> 加两个任务条目是纯增量、不影响现有 16 个任务,但**改的是另一个 repo,需要你点头**。
> close_drawer 的物体名要等它渲染完才知道。

### 16.2 渲染不影响训练

渲染吃 177 核(16 worker),但机器有 256 核,实测训练速度没变
(normal 19.97s/it、action 21.68s/it,与渲染前一致)。

> 注:`LP_NUM_THREADS=4` 实际没把每个 worker 限制到文档说的 2.7 核(实测 ~11 核/worker)。
> 这次没造成问题,但**如果以后要开更大的 NPAR,先量一下,别照抄文档里的 48。**

---

## 十七、代码迁移进本仓库 + 修掉一个协议判断错误的 guard(2026-08-30 21:00 UTC)

按"本仓库要 release、必须 self-contained"的要求,把 ttd 的代码迁进来。

### 17.1 迁移 `scene_segments_gen.py`

`ttd/src/percep/scene_segments_gen.py` → **`training/percep/scene_segments_gen.py`**
(纯标准库 + numpy,没有 ttd 内部依赖,迁移干净),并在文件头写明 provenance。

三处调用点已改掉 `sys.path.insert("/workspace/ttdu/ttd/src/percep")` 的写法:

| 文件 | 改法 |
|---|---|
| `scripts/audit_scene_roles.py` | 直接 `from training.percep.scene_segments_gen import TASK_ROLES` |
| `scripts/seg_pixel_stats.py` | 同上(`resolve_roles`) |
| `training/dataset/rlbench_selfgen.py` | 报错信息里的修复命令改成 `python -m training.percep.scene_segments_gen` |

两个脚本导入验证通过。**仓库内已无对 `ttd/src/percep` 的代码依赖**(只剩 3 处 provenance 注释)。

### 17.2 补上两个新任务的 role 定义

| 任务 | 条目 | 依据 |
|---|---|---|
| `close_box` | `box_base: goal`, `box_lid: distractor` | 与最接近的 `close_jar` 同构(主体承接盖子,盖子是可被指令提升为 target 的操作对象) |
| `close_drawer` | `drawer_frame/legs: fixture`, `drawer_top/middle/bottom: distractor` | **与已有的 `open_drawer` 逐字相同**——同一套抽屉几何、同样由指令挑选哪一个抽屉 |

其余物体(Panda_link0-7 / gripper / floor / wall / table / workspace)全部已被
`ROBOT_ARM_NAMES` / `GRIPPER_NAMES` / `BACKGROUND_NAMES` 覆盖,无需新增。

跑 `--write` 结果:**`UNMAPPED object names: 0 distinct` + `SCENE_SEGMENTS_COVERAGE_OK`**,
40 个 episode 全部写入 `scene_segments.json`。角色计数自洽
(goal 20 = close_box 主体 × 20;fixture 40 = close_drawer frame+legs × 20)。

### 17.3 ⚠️ 修掉一个 guard:它在 scene_roles 模式下查错了文件

首次跑 unseen-task 评测时 dataset 构造直接抛错:

```
RuntimeError: [selfgen] seg coverage too low: 40/40 episodes (100.0%) have no seg_targets.json
```

**但这是误报。** 查 `getitem` 的实际逻辑:

- **`referring` 协议**:`seg_targets.json` **就是**信号本身,缺了 → `_referred_id_groups`
  返回空 → 丢掉 segmentation → 样本静默退化成纯 video。**guard 完全正确。**
- **`scene_roles` 协议**:丢弃判断走的是 `role_lut is None`,而 `role_lut` 来自
  **`scene_segments.json`**。`seg_targets.json` 只用于可选的指令覆盖
  (把 `distractor` 提升成 `target`),`_scene_role_lut` 的注释白纸黑字写着
  "A missing/unresolvable seg_targets.json is **NOT fatal**"。

也就是说 guard 在 scene_roles 下**查了一个该协议并不依赖的文件**,于是把一棵完全可用的树
拒之门外。已改成按协议选择要检查的文件:

```python
required = ("scene_segments.json" if self.segmentation_mode == "scene_roles"
            else "seg_targets.json")
```

**回归验证**:训练树仍然通过(`seg coverage OK: 788/788 episodes carry scene_segments.json`
+ `scene_roles OK: zero unknown pixels`),新树也通过(`40/40`)。
报错信息里的修复建议也一并改成按协议给出正确的命令。

> 这个 bug 之前不可能被发现:训练树的每个 episode 两个文件都有,只有在
> "有 scene_segments 但没有 seg_targets"的树上才会暴露——而这种树以前不存在。

### 17.4 unseen-task 评测已启动

`scripts/run_unseen_task_eval.sh`,GPU 5,arm6@10000,16 个 episode
(close_box / close_drawer 各 8,用闭环 trial_seed),测 depth / segmentation / normal。
启动日志确认 `scene_roles OK: zero unknown pixels over 8 probed views`。

新增 `--split_label` 参数并**必须使用**:这些 episode 位于 `variation0`
(两个任务都只有一个 variation),默认的"按 variation 推断层级"会把它们**误判成训练数据**。
`--split_label unseen_task` 强制标注成最强的那一档 held-out。

### 17.5 self-contained 还剩什么(未做,供决策)

| 依赖 | 影响面 | 建议 |
|---|---|---|
| `ttd/scripts/env_eval.rc` | **10 个 .sh** 都 source 它(只有 3 行:COPPELIASIM_ROOT / LD_LIBRARY_PATH / QT 插件路径) | 直接 vendor 进 `scripts/env_eval.rc`,最划算 |
| `ttd/data/rlbench_selfgen_512_aug` | 数据,已是软链接 | 数据不必进 repo,但 README 要写清怎么生成 |
| `ttd/data/texture_pool` | `gen_dataset.py` 的增强纹理 | 同上,或改成可配置路径 |
| `ttd/src/percep/seg_targets_gen.py` | **referring 协议**才需要;scene_roles 不需要 | 若 release 只支持 scene_roles,可以不迁 |
| `ttd/plan/core/*.md` | 只是注释里的文档引用 | 无害,但 release 前应改成仓库内的文档或删掉 |

---

## 十八、✅ 感知能泛化到没训练过的任务 + ⚠️ 闭环在 variation1 上双双崩溃(2026-08-31)

### 18.1 三档泛化层级(arm6@10000)—— 论文的核心新结果

| 模态 | ① seen(字面训练样本) | ② unseen variation | ③ **unseen TASK**(从没训练过) |
|---|---|---|---|
| depth AbsRel ↓ | 0.1357 [0.079, 0.202] | 0.1536 [0.116, 0.191] | **0.1713** [0.116, 0.235] |
| seg mIoU ↑ | 0.4855 [0.377, 0.599] | 0.4609 [0.375, 0.556] | **0.5777** [0.541, 0.612] |
| normal cos ↑ | 0.8945 [0.857, 0.927] | 0.8923 [0.871, 0.912] | **0.8979** [0.876, 0.917] |
| action pos ↓ | 0.126 m | 0.157 m | —(新树没有动作评测意义) |
| RGB LPIPS ↓ | 0.1941 | 0.2255 | 0.2246 |

**读法**:感知**几乎不退化**,一路到"从没训练过的任务"都成立。
- depth 从 0.136 → 0.171(退化 26%,但 CI 大幅重叠)
- normal **完全没动**(0.8945 → 0.8979)
- **segmentation 反而更好**(0.486 → 0.578)

> ⚠️ segmentation 变好**不能**读成"泛化更好"。close_box / close_drawer 的场景比训练任务
> **简单得多**(close_box 只有 2 个任务物体、close_drawer 5 个,而训练任务动辄十几个),
> 角色少、每个角色像素大,mIoU 自然高。**跨任务比 mIoU 绝对值是不公平的**,
> 这一点写进论文时必须说明,否则是在夸大。

**这一档以前根本测不了**(开环评测必须读磁盘 GT,而这两个任务没有渲染过),
是这次专门渲染 40 个 episode + 迁移 `scene_segments_gen.py` + 补 role 定义换来的。

### 18.2 ⚠️ 闭环在 variation 1 上两个模型一起崩

| | push_buttons | meat_off_grill | open_drawer | 合计 |
|---|---|---|---|---|
| arm6 **var0** | 12/20 | 9/20 | 0/20 | **35%** |
| arm6 **var1** | 1/20 | 0/20 | 0/20 | **1.7%** |
| official **var0** | 3/20 | 1/20 | 0/20 | 24%(5任务) |
| official **var1** | 0/20 | 0/20 | 0/20 | **0%** |

配对检验:60 个场景里只有 **1 个判别对**,`aggregate_closedloop.py` 直接判
`UNDERPOWERED`——**这个数字目前不能支持任何结论**。

**⚠️ 下结论前必须先做 GT-replay 对照**(已在 GPU 1 启动):如果拿 demo 自己的动作
回放在 variation1 上也失败,那说明是**工装/场景不可解**,而不是模型不会做。
`open_drawer` 在 variation0 上两个模型就已经是 0/20 而 GT 天花板是 100%,
说明这套 harness 本来就有任务级的工装问题,**不能假定 variation1 的 0% 是模型的锅**。

### 18.3 顺带修:resume 会把 summary 截断

§12.3 加的断点续跑有个副作用:`--modalities action` 的重算跑完后,
`summary.json` 被**只用 action 一个模态重写**,depth/seg/normal 三个条目消失
(`per_episode.json` 里的原始记录完好无损,只是聚合层丢了)。

原因:聚合循环只遍历**本次请求的** modalities。已改成遍历
**`per_episode.json` 里实际存在的**所有模态。四个 tag 的 summary 已用完好的
per-episode 数据重建(不需要重跑生成)。

---

## 十九、GT-replay 对照:variation1 崩溃是**模型的问题**,不是工装(2026-08-31)

§18.2 留的那个必须做的对照,结果出来了:

| 任务 | GT-replay 天花板(var1) | arm6(var1) | official(var1) |
|---|---|---|---|
| push_buttons | **9/10 = 90%** | 1/20 = 5% | 0/20 |
| meat_off_grill | **3/4 = 75%** | 0/20 | 0/20 |

**场景是可解的**——拿 demo 自己的动作回放能到 75-90%。所以
"arm6 从 35% 掉到 1.7%、official 从 24% 掉到 0%" **是真实的模型泛化失败,
不是场景不可解或工装问题**。这个结论现在可以写进论文了。

> 对比:`open_drawer` 在 **variation0** 上两个模型就已经 0/20,而它的 GT 天花板是 100%
> ——那一个是工装/解码问题。两者要分开讲,不能混为一谈。

### 19.1 于是论文有了一个干净的对比

| | 感知(depth/seg/normal) | 闭环控制 |
|---|---|---|
| seen → unseen **variation** | 几乎不变 | **35% → 1.7%,崩** |
| unseen variation → unseen **task** | 仍然几乎不变 | 无法测(开环) |

**感知的泛化边界比控制宽得多。** 而且失败发生在**不同的边界上**:
控制在第一个边界就崩,感知跨两个边界都还稳。

这说明**限制控制的不是共享输出空间**(共享在 matched exposure 下零代价、
感知本身跨任务都稳),**而是 action prior 自身的泛化能力**。

---

## 二十、按用户决定执行(2026-08-31)

### 20.1 `outline.md` 补上三档泛化

用户新写的 `paper/outline.md` 已取代旧草稿(`draft_aug_28/29.md` 均已删除)。
按建议把 `sec:exp_generalization` 从"简单提一下 unseen variation"改写成
**三档表 + 感知/控制对比**,并写明两条诚实性约束:

1. **seg mIoU 跨任务边界不可比**(close_box 2 个任务物体、close_drawer 5 个,
   训练任务动辄十几个;角色少、每角色像素大 ⇒ macro-mIoU 天然偏高)。
   那一格只能读作"没有崩",**绝不能读作"更好"**。
2. **两个 unseen task 是探针,不是普查**,不能声称感知与任务无关。

### 20.2 排上两个 specialist 的闭环

`scripts/wait_and_launch_specialist_closedloop.sh`,训练一完成 + GPU 一空自动启动,
协议与 unified 的 campaign 完全一致(fi=3 / 带 tag / cfg 7.5 / 同 5 任务 / n=20),
可直接与 `rollout_arm6_10000.json` 配对。

> ⚠️ **两者测的不是同一件事**,脚本里已写明:
> - **action specialist** → `tab:wam_preservation` 的中间行。init→action-spec 是域适应,
>   action-spec→unified 才是"加感知的代价"。**没有它这张表无法拆分这两个效应。**
> - **normal specialist** → **不在任何计划中的表里**。它从没训过 action 模板,
>   所以测的是"4000 步纯感知微调之后,released checkpoint 的 action 先验还剩多少"
>   ——是一个**遗忘**测量。它的价值在于:如果纯感知训练也不损害控制,
>   那"unified 保住了控制"这句话就没有信息量。**读作遗忘,不要读作"normal specialist 的策略"。**

---

## 二十一、normal specialist 的定位 + 三块空卡的安排(2026-08-31 01:30 UTC)

### 21.1 normal specialist 在全文只贡献 1 个数字

逐行核过 `outline.md` 的三张表:

| 表 | normal specialist |
|---|---|
| `tab:headline` | ✅ "Normal cosine" 行的 Specialist 列 —— **就这一个数** |
| `tab:wam_preservation` | ❌ 行只有 Initialization / Action specialist / Unified |
| `tab:generalization` | ❌ 整张表只有 unified model |

**⇒ 已取消它的闭环 waiter**(跑完填不了任何表格单元,省 ~9h)。
按用户决定:normal specialist 只出 Table 1 那一个数,训练继续跑完。

### 21.2 Table 1 需要的 checkpoint 不等于训练终点

| specialist | Table 1 需要 | 说明 |
|---|---|---|
| depth / seg / **normal** | **checkpoint-2000** | unified 对每个感知模态只有 0.2T=2000 次更新 |
| action | checkpoint-4000 | unified 对 action 有 0.4T=4000 次更新 |

**所以 normal 的格子不用等 4000 步训完,2000 步就够** —— 已排
`scripts/wait_and_eval_specialist.sh`,checkpoint 一落盘(且文件大小稳定)就自动评测。
action 的 ckpt-4000 评测也一并排上。

### 21.3 三块空卡的安排(按"填最大的洞"排序)

| GPU | 任务 | 填什么 |
|---|---|---|
| **1** | 官方 init 的离线指标 | **`tab:wam_preservation` 的 Initialization 整行**——之前只有 Success 一格 |
| **5** | unseen-task 评测从 8 → 20 episode/任务 | 收紧第三档的 CI(depth 现在是 [0.116,0.235],太宽) |
| **6** | variation0 的 GT-replay 天花板(5 任务 × 10) | 给 `open_drawer` 两模型都 0% 一个可读的对照 |

### 21.4 ⚠️ 为此改了两处评测代码

1. **新增 `--prompt_tag_style`**。之前 `heldout_batch_eval.py` 把
   `prompt_tag_style="explicit"` 写死了。**官方 checkpoint 从没见过模态 tag**,
   用 explicit 去评它,测到的是 prompt 分布偏移而不是能力——和闭环协议是同一个坑
   (§5.5)。官方那一行必须用 `--prompt_tag_style none`。
2. **把 `rot_err_med_deg` 等指标接进 summary**。`score_action` 一直在记录它们,
   只是聚合层从没输出过,而 `tab:wam_preservation` 需要旋转误差。
   现在 action 条目会额外带 `pos_err_mean_m` / `rot_err_med_deg` /
   `rot_err_over90_frac` / `r_peak`。

### 21.5 断点续跑已验证有效

unseen-task 扩到 40 episode 时,日志确认
`resuming from ...: 48 scored cells already present` + 24 个 SKIP,
只生成缺失的单元,没有重跑已有的 3 小时工作量。

---

## 二十二、8 小时无人值守窗口(2026-08-31 03:07 UTC 起)

### 22.1 起始状态

| GPU | 任务 | 进度 |
|---|---|---|
| 0,2 | action specialist 训练 | 3345/4000 |
| 3,4 | normal specialist 训练 | 2009/4000 |
| 1 | GT-replay var1(**planning**) | 19/30 |
| 5 | unseen-task 扩到 40 集 | 55/72 |
| 6 | GT-replay var0(**planning**) | 18/50 |

**自动排队中**(不需要人工干预):
- normal ckpt-2000 评测 —— **已自动开跑**(waiter 检测到 checkpoint 落盘)
- action ckpt-4000 评测 → Table 1 的 action 行
- action ckpt-4000 闭环 → `tab:wam_preservation` 的中间行

### 22.2 ⚠️ 待修正的两个结论(等 planning 版 GT-replay)

第一版 GT-replay 用了 `--arm-action-mode` 的默认值 **`ik`**,而所有策略 rollout 用的是
**`planning`**。仓库自己的实测(`rollout_env.py:120-132`)显示两者差别很大:

```
ViaIK        34 次 IK 失败 (14%)
ViaPlanning   0 次 IK 失败 ( 0%)     只慢 25%
```

`ik` 下的天花板会**系统性偏低**(位姿不可达就直接失败),不能用来论证 planning rollout 的
成败。已用 planning 重跑,结果出来后要据此修正:

1. **`close_box` 的天花板**:ik 下只有 40%。若 planning 下仍然明显低于 100%,
   那么 arm6 的 25% 要按"占天花板的比例"来读,不能读作"差"。
2. **`open_drawer`**:ik 下天花板 100% 而两个模型都 0%。若 planning 下也成立,
   那么草稿里"open_drawer 是 harness limitation"这句话**是错的**,
   应改成"两个模型在一个可解任务上都彻底失败"。

### 22.3 补充实验:夹爪精度

`tab:wam_preservation` 的 **Grip. acc.** 一列此前无法填——批量评测走的
`G.score_action` 不算夹爪精度(`eval/eval_action.py` 算,但那是单 episode 的另一条路径)。

已把它加进 `score_action`,derivation 与 `eval_action.py:154-167` 完全一致
(pose8 第 8 列的 openness 过 0.5 阈值,对 `action_7d` 第 7 列)。
supervisor 会在 GPU 空出时自动把 arm6 和 init 的 action 单元重算一遍补上这个指标
(每个 16 cell ≈ 47 min)。

### 22.4 supervisor 的职责

`scripts/supervise_8h.sh`,每 10 分钟一轮,**幂等**(每步先查产物是否已存在):
1. 给 arm6 / init 补 `gripper_acc`(GPU 一空就跑)
2. 检查两个训练是否意外死亡且没有 checkpoint-4000,有则报警

---

## 二十三、generalization 小节的三处修正 + 后续实验(2026-08-31 03:30 UTC)

### 23.1 ⚠️ 修正一:第三档闭环数据一直都有,而且三档**不是单调的**

`close_box`/`close_drawer` 本来就是没训练过的任务,闭环数据早就跑了:

| | arm6 | official init |
|---|---|---|
| ① 训练过的任务(var0) | 35.0% | 6.7% |
| ② 同任务 unseen variation | **1.7%** | 0.0% |
| ③ **从没训练过的任务** | **35.0%** | **50.0%** |

**②比③还差** —— 换 variation 比换整个任务更难。原来写的
"control collapses at the FIRST boundary"暗示单调退化,**框架是错的**,已改写。

合理解释:两档变的不是同一个东西。**换 variation 改的是指令的指代对象**
(哪个按钮、哪个抽屉),策略要解析一个训练时没解析过的 referent;
**换任务但目标唯一**(close_box 只有一个盒子)则更接近纯外观偏移。
只有前者打垮了策略。

> ⚠️ **③ 的 35% 不能读成"arm6 泛化好"**:official 在同样两个任务上是 **50%,比 arm6 高**。
> 那一列主要是预训练先验幸存下来的部分,减去我们微调忘掉的。
> 它属于 `tab:closed_loop` 里"没训练过"那一块的讨论,不是迁移能力的证据。

### 23.2 ⚠️ 修正二:close_box 的 40% 天花板是 ik 造成的假象

| task | GT(ik) | **GT(planning)** | official | arm6 |
|---|---|---|---|---|
| push_buttons | 100% | **100%** | 15% | 60% |
| meat_off_grill | 100% | **100%** | 5% | 45% |
| **close_box** | **40%** | **100%** | 20% | 25% |
| close_drawer | 100% | **100%** | 80% | 45% |

**planning 下 close_box 是 10/10。** 所以上一轮"arm6 的 25% 其实是天花板的 62%"
**是错的**,它就是 25%/100%。幸好按协议匹配重跑了。

### 23.3 ⚠️ 修正三:`open_drawer` 是模型的锅,不是工装

var1 的 planning 天花板 **7/8**,var0 也是 100%,而两个模型在两档上都是 **0%**。
之前"harness limitation"的说法**必须改成"两个模型在一个可解任务上彻底失败"**。
(这条原本写在已删除的 draft_aug_29.md 里,现记录在此,写正文时不要再犯。)

`outline.md` 的 `tab:generalization` 已补上第三档闭环 + GT 天花板行,
并重写了"单调退化"那段。

### 23.4 已挂上的后续实验

**B(最高优先级,GPU 3/4/7 跑中)**:三个 specialist 在**没训练过的任务**上评测。
回答一个直接挑战论文主张的问题:

> unified 的感知跨任务几乎不退化 —— 这是"统一"带来的,还是 frozen VAE + 预训练
> backbone 本来就有的?

- specialist 退化得一样少 → 跨任务鲁棒性是 **backbone 的属性**,论文**不能**归功于统一
- specialist 退化更多 → 与其他流共同训练确实让感知在分布外更稳,**是个真结果**

`outline.md` 已加 `\paragraph{Is cross-task perception a property of unification?}` 占位。

**C(渲染中,纯 CPU 不占卡)**:再渲 4 个没训练过的任务 × 8 seed,
把第三档从 n=2 个任务扩到 **n=6**,回应小节里自己承认的
"Two unseen tasks is a probe, not a survey":

| 任务 | 为什么选它 |
|---|---|
| close_microwave | 铰接开合,与 close_drawer 同类,可做类内对照 |
| take_lid_off_saucepan | 抬起移走,与 close_box 操作方向相反 |
| toilet_seat_down | 铰接但几何差别大 |
| basketball_in_hoop | 自由空间放置,与前面几类都不同 |

> ⚠️ 渲染完要先看 `handles.json` 拿到物体名,给这 4 个任务补
> `TASK_ROLES` 条目(`training/percep/scene_segments_gen.py`),
> **否则 segmentation 会全是 `unknown`**;depth/normal 不受影响。

---

## 二十四、B 控制实验出结果:跨任务泛化**不是**统一带来的(2026-08-31 04:20 UTC)

### 24.1 结果:三个模态全部 CI 跨 0

同一批 16 个 never-trained episode,配对比较 unified vs 单模态 specialist(都读 matched-exposure 的 checkpoint):

| 模态 | unified | specialist | 配对差 | 95% CI | p |
|---|---|---|---|---|---|
| depth AbsRel ↓ | 0.1713 | 0.1734 | −0.0021 | [−0.0174, +0.0148] | 0.53 |
| seg mIoU ↑ | 0.5777 | 0.5574 | +0.0203 | [−0.0217, +0.0640] | 0.53 |
| normal cos ↑ | 0.8979 | 0.9023 | −0.0045 | [−0.0143, +0.0051] | 0.56 |

**一个只见过单一模态的 specialist,在没训练过的任务上表现得和 unified 一模一样。**

### 24.2 这对论文意味着什么

**跨任务的感知鲁棒性是 frozen VAE + 预训练 backbone 的属性,不是"统一"买来的。**
必须明确写出来,因为只看 `tab:generalization` 很容易得出相反的(错误的)结论。

**但这其实让主张更锋利,不是更弱**:结论不是"统一让感知更好",而是
**"统一不花代价"** —— 这正是 `sec:exp_sharing` 在匹配曝光下测到的
(depth +0.010 p=0.74、seg −0.012 p=0.84,同样 CI 跨 0),
也正是"一个 checkpoint 顶四个"能成立的前提:它是**净省**,不是**权衡**。

已写进 `outline.md` 的 `\paragraph{Is cross-task perception a property of unification? No.}`。

### 24.3 B 的副产品:tier-3 从 2 个任务扩到 6 个

4 个新任务(close_microwave / take_lid_off_saucepan / toilet_seat_down / basketball_in_hoop)
已渲染完(72 episode),`TASK_ROLES` 条目已补,`scene_segments_gen` 干跑
**`UNMAPPED object names: 0 distinct`** 后写入 72 个 `scene_segments.json`。

新任务的 role 定义(都参照最接近的已有任务):

| 任务 | 条目 | 参照 |
|---|---|---|
| close_microwave | `microwave_door: distractor`, `microwave_frame_vis: fixture` | close_drawer(铰接件+静态外壳) |
| take_lid_off_saucepan | `saucepan_lid_visual: distractor`, `saucepan_visual: goal` | close_box(盖子+承接主体) |
| toilet_seat_down | `toilet_seat_up_seat: distractor`, 另两个 `fixture` | 铰接件+静态本体 |
| basketball_in_hoop | `ball: distractor`, `basket_ball_hoop_visual: goal` | 自由空间放置 |

arm6 在 6 个任务上的评测已在 GPU 1 启动(resume 会跳过已算的 close_box/close_drawer,只补新增的 4 个任务)。

### 24.4 其余进度

- action specialist 3543/4000(~2.5h),评测和闭环 waiter 就位
- `gripper_acc` 回填:init ✅ 完成,arm6 待 GPU

---

## 二十五、`tab:wam_preservation` 填上两行 + tier-3 闭环扩到 6 任务(2026-08-31 05:20 UTC)

### 25.1 gripper_acc 回填完成,preservation 表可以填了

| Model | pos err ↓ | rot err ↓ | **grip acc ↑** | RGB LPIPS ↓ | Success ↑ |
|---|---|---|---|---|---|
| Initialization | 0.218 m | 77.5° | 0.732 | 0.240 | 24% |
| Action specialist | 训练中 | | | | 排队中 |
| **Unified (arm6)** | **0.157** | **67.2°** | **0.890** | **0.226** | **35%** |

**unified 在每一列上都优于起点**(不只是"保持")。但正文里已写明:
`init → unified` 这个对比把**域适应**和**感知联训**捆在一起了,分不开。
真正支撑 claim 的是中间那一行:

- `init → action specialist` = 在我们数据上继续训练的效果
- `action specialist → unified` = **加了感知输出的代价**

论文要的是第二个箭头上的"保持",不是第一个箭头上的"提升"。已在 outline 里写清楚。

### 25.2 tier-3 闭环从 2 任务扩到 6

`tab:generalization` 第三档的闭环格现在只有 close_box/close_drawer 两个任务,
却要支撑一个反直觉的主张(**没见过的任务比没见过的 variation 更容易**)。
两个任务撑不住这个结论。

已在 GPU 5/6 起了 4 个新任务的**配对**闭环
(close_microwave / take_lid_off_saucepan / toilet_seat_down / basketball_in_hoop,
各 10 trial,arm6 与 official 各自用自己的训练协议),把第三档从 2 任务扩到 6。

### 25.3 6 任务版的开环评测仍在跑

arm6 depth/seg/normal 各 40-41/48,三个 specialist 各 32-33/48。
跑完后要重算 B 对照(n=16 → 48),并更新 `tab:generalization` 第三档的数字
(现在写的 0.171/0.578/0.898 是 2 任务的)。

### 25.4 GPU 占用

0,2 action 训练(3722/4000)| 1,3,4,7 六任务开环评测 | 5,6 四任务闭环

---

## 二十六、6 任务版结果:B 对照的结论**更强了**(2026-08-31 06:30 UTC)

### 26.1 控制实验在更大样本上依然成立(且差异更小)

| 模态 | n | unified | specialist | 配对差 | 95% CI | p |
|---|---|---|---|---|---|---|
| depth | 28 | 0.1457 | 0.1455 | **+0.0002** | [−0.0131,+0.0138] | 0.884 |
| segmentation | 28 | 0.5659 | 0.5663 | **−0.0004** | [−0.0282,+0.0299] | 0.678 |
| normal | 27 | 0.8971 | 0.9025 | −0.0054 | [−0.0123,+0.0011] | 0.211 |

n 从 16 → 28(6 任务版仍在跑,最终 48),三个模态的配对差**比 n=16 时更接近 0**
(depth 从 +0.0021 降到 +0.0002)。结论不变且更稳:
**跨任务的感知鲁棒性不是"统一"带来的,而是 backbone 的属性。**

### 26.2 tier-3 的数字随任务数变化 —— 说明"2 个任务"确实撑不住

| 模态 | 2 任务(close_box/drawer) | 4 任务(截至目前) |
|---|---|---|
| depth AbsRel ↓ | 0.1713 | **0.1544** |
| seg mIoU ↑ | 0.5777 | **0.5572** |
| normal cos ↑ | 0.8979 | 0.8989 |

depth 和 seg 都变了 5-10%。**这正好证明扩到 6 个任务是必要的** ——
原来那两个任务的数字带着明显的任务特异性。等 48 个 cell 全部跑完再定稿。

### 26.3 tier-3 闭环(4 个新任务,配对)进行中

`arm6` 14/40、`official` 14/40。中间结果:

| 任务 | arm6 | official |
|---|---|---|
| close_microwave | 9/10 | 7/10 |
| take_lid_off_saucepan | 3/4 | 3/4 |

初步看仍支持"没见过的任务反而不难"这个反直觉主张 —— close_microwave 两个模型都接近满分,
而同样是铰接类的 close_drawer(也是没训练过)之前是 45%/80%。等跑完再判。

### 26.4 已加 per-role IoU(尚未生效)

`scripts/modality_mode_grid.py` 的 `score()` 现在会把 `per_role_iou` 记进 record,
用来解决 `tab:generalization` 里 seg 那格"跨任务不可比"的 `†` 免责声明 ——
有了逐角色 IoU 就能只比**两边都有的角色**(robot_arm/gripper/background/fixture/distractor)。

⚠️ 改动只影响之后启动的进程;现有的 seg 单元是旧代码算的,没有这个字段。
要等当前评测跑完、GPU 空出后,删掉 seg 单元让 resume 用新版重算。
**不能现在做**——会和正在写同一个 `per_episode.json` 的进程撞车。

新增 `scripts/analyze_unseen_task.py`,一条命令重算 tier-3 均值 + 配对对照 + per-role 汇总。

---

## 二十七、磁盘与 checkpoint 盘点(2026-08-31 06:50 UTC)

### 27.1 checkpoint 被剪枝了,但**需要的都还在**

发现各 specialist 的中间 checkpoint 大量消失(depth/seg/normal 只剩 2000,action 只剩 3000/3500)。
查清是 `train.py` 自带的剪枝:日志里

```
INFO:__main__:pruned 115.4 GB of redundant checkpoint state (resume state kept only in checkpoint-3500)
```

它删的是 DeepSpeed 的 **`global_step*/` 优化器分片**(续训才需要),**不是模型权重**。
逐一验证过,评测依赖的五个 checkpoint **全部完好**:

| checkpoint | 状态 |
|---|---|
| `outputs/.grid_pin/step10000.ckpt`(arm6) | ✅ 12G |
| depth/seg/normal specialist `checkpoint-2000/step2000.ckpt` | ✅ 各 12G |
| 官方 `step125750.ckpt` | ✅ 12G |

> ⚠️ 副作用:**specialist 的"天花板"checkpoint(2500/3000)已经没了**。
> 用户之前决定"暂时不管天花板",所以不影响当前实验;
> 但如果以后要补天花板读数,**必须重训**,不能再从磁盘捞。这一点要记住。

### 27.2 磁盘 96%,有 108G 可回收但**没有删**

`/workspace` 用到 96%(剩 898G)。`specialist_normal` 占 120G,其中 **108G 是
`global_step2000/`** —— 它已被我停掉且不会续训,这部分是死重。

**没有删**:剩余空间足够 action 写完最终 checkpoint(~127G),
在无人值守窗口里做不可逆删除不值当。记录为可回收,等有人在场再定。

### 27.3 另有一处显存泄漏(非本项目)

GPU 4 上有 31126 MiB 被 PID 3396602 占着,但该进程已不存在 —— 泄漏的显存,
不属于我们的任何任务。实际可用是 7 卡不是 8 卡。

### 27.4 进度

- **action specialist 3980/4000**,两个 waiter(评测 + 闭环)都活着,等它落盘
- **arm6 的 6 任务评测**:close_box/close_drawer/close_microwave 完成,
  take_lid_off_saucepan 快完,还差 toilet_seat_down + basketball_in_hoop(~2.3h)
- **三个 specialist 的 6 任务评测:已全部完成**(各 48 cell / 6 任务)
- **4 任务配对闭环**:arm6 14/40、official 14/40

---

## 二十八、✅ 三张主表全部填满,占位符归零(2026-08-31 13:50 UTC)

### 28.1 Table 1 —— 四个模态**全部**无可检测差异

| Output | matched | Specialist | Unified | 配对差 | 95% CI | p |
|---|---|---|---|---|---|---|
| Action pos (m) ↓ | 0.4T=4000 | 0.166 | **0.157** | −0.009 | [−0.047,+0.030] | 0.55 |
| Depth AbsRel ↓ | 0.2T=2000 | **0.144** | 0.154 | +0.010 | [−0.010,+0.032] | 0.74 |
| Seg mIoU ↑ | 0.2T=2000 | **0.473** | 0.461 | −0.012 | [−0.044,+0.016] | 0.84 |
| Normal cos ↑ | 0.2T=2000 | 0.889 | **0.892** | +0.003 | [−0.001,+0.007] | 0.20 |

**四个模态的 CI 全部跨 0**,其中 action 和 normal 上 unified 名义上还更好。
一个 checkpoint 干四件事 = 四个专用 checkpoint,在匹配曝光下。

### 28.2 ⚠️ Table 2 —— **preservation 的说法必须收窄**

action specialist 的闭环跑出来后,三方配对(同 80 个场景):

| 箭头 | 效果 | p |
|---|---|---|
| Init → Action specialist | 30.0% → **57.5%**(+27.5pp) | **<1e-4** |
| **Action specialist → Unified** | **57.5% → 43.8%(−13.8pp)** | 0.099 |
| Init → Unified | 24.0% → 35.0%(+11.0pp) | 0.099 |

**大部分闭环增益来自域适应,而加三路感知输出把其中约 14 个点又赔了回去。**
这个代价在 n=80 上不显著(判别对 13 vs 24),**但也不能算"保持"** ——
CI 明显包含真实退化,点估计是净损失。

**而且离线指标完全看不出这一点**:同一批 held-out episode 上 unified 的
位置误差(0.157 vs 0.166)和夹爪精度(0.890 vs 0.820)都**更好**,只有旋转差一点。
**感知流对策略的代价不出现在单步轨迹误差里,只在模型连续自主执行整个 episode 时才显现。**

诚实的写法(已写进 outline):unified **保住了域适应带来的大部分控制能力、
明显强于自己的起点,但让出了同预算纯动作模型的一部分**。不是"preservation"。

### 28.3 Table 3 —— 6 任务版 + B 对照的答案是"equally"

tier-3(6 个从没训练过的任务,48 episode):
depth **0.150**、seg **0.545**、normal **0.906**(2 任务版是 0.171/0.578/0.898,
移动了 5-10%,**证明当初扩到 6 个是必要的**)。

**B 对照结论:跨任务的感知鲁棒性是 backbone 的属性,不是统一买来的。**

| 模态 | n | unified | specialist | 差 | CI | p |
|---|---|---|---|---|---|---|
| depth | 48 | 0.1437 | 0.1370 | +0.0067 | [−0.0068,+0.0216] | 0.50 |
| seg | 48 | 0.5457 | 0.5488 | −0.0031 | [−0.0236,+0.0185] | 0.55 |
| normal | 48 | 0.9078 | 0.9125 | −0.0048 | [−0.0095,−0.0002] | 0.14 |

**论文不能声称"统一带来了跨任务泛化"**。统一买到的是
"一个 checkpoint 干四件事且没有可测代价"(Table 1),这是参数量/部署层面的论证,
不是泛化层面的。已写进 outline。

tier-3 闭环 6 任务合计:arm6 **40.0%** vs official **43.8%**(Δ−3.8pp, p=0.68)——
**同样不是迁移证据**,官方还略高。而且 6 个任务是双峰的:
close_microwave 9/10、take_lid_off_saucepan 8/10 接近满分,
basketball_in_hoop 0/10、toilet_seat_down 1/10 接近零,合计数掩盖了这个分裂。已在 outline 注明。

### 28.4 状态

`paper/outline.md` **占位符 0 个**。唯一未完的是 `spec_action_4000` 闭环(80/100,
差 open_drawer 20 个),跑完后 Table 2 的 Success 列可以从 n=80 换成 n=100 口径。

---

## 二十九、外部 specialist baseline:VGGT 深度对照(2026-08-31)

### 29.1 为什么需要它

当前 Table 1 的 specialist 是**我们自己训的**,只能回答"共享 head 有没有代价",
**回答不了"0.154 AbsRel 到底算不算好"**。Vision Banana 和 GenCeption 都拿领域
specialist 做对照(SAM3 / Depth Anything 系列),这类论文的惯例就是如此。

### 29.2 ⚠️ 两个可比性问题,必须先解决

**(1) 尺度**:VGGT 输出的深度**不是米制的**。本项目此前在 DROID 上实测过
(`MULTIDATASET_DEPTH_PLAN.md` §2.0):不存在任何常数能把它变成米,
理想 per-sample scale 的 **CV 达 46%**,中位误差 22 cm。
→ 对比必须用**逐帧中位数尺度对齐**(尺度歧义模型的标准做法)。
**而我们的 codec 直接输出米制、无需对齐 —— 这本身是个结果,不该靠"两边都对齐"抹掉。**

**(2) conditioning**:VGGT 拿到**真实 RGB** 预测其深度。
我们在 `IIII` 下是**同时生成未来 RGB 和深度**,是严格更难的问题。
→ 可比的设定是 **`FIFI`**(给定 RGB,只预测深度)。
为此给 `heldout_batch_eval.py` 加了 `--mode {iiii,fiii,fifi}`。

### 29.3 结果(同 8 个 held-out variation1 episode)

| | AbsRel ↓ | 说明 |
|---|---|---|
| VGGT-1B **原始输出** | **0.357** | 未对齐 —— 它不输出米制 |
| VGGT-1B **中位数对齐后** | **0.206** | 送了它一个自由尺度参数 |
| **我们(IIII)** | **0.154** | **米制、无需对齐,而且 RGB 也是生成的** |
| 我们(FIFI) | 跑中 | 与 VGGT 同类的判别式设定 |

**即使把自由尺度送给 VGGT,我们在更难的设定下仍然更好(0.154 vs 0.206)。**

> ⚠️ **3 个 episode 的 pilot 给出的是 0.093,与全量的 0.206 差一倍以上** ——
> pilot 恰好抽到了对 VGGT 最容易的三个(light_bulb_in 对齐后 0.021)。
> **小样本在这里极不可靠,幸好跑了全量 8 个。**

### 29.4 ⚠️ 必须写明的免责声明

**这是主场优势对比,不是能力排名。** VGGT 是**零样本**迁移到 RLBench 合成渲染上的,
我们是在这个分布上**微调过**的;而且 VGGT 的设计目标是真实世界多视角 3D 重建,
不是单场景合成深度。

诚实的说法:**它给"我们这个深度任务有多难"划了一条参照线**,
而不是证明我们比 VGGT 强。

### 29.5 逐模态可行性(全网调查后的结论)

| 模态 | 外部 baseline | 可行性 |
|---|---|---|
| **depth** | VGGT-1B(权重已缓存本地,4.7G) | ✅ **已完成** |
| **normal** | 用同一份 VGGT 深度 + 我们自己的 `depth_to_normal` 推导 | ✅ 零新依赖,可做 |
| **action** | 官方 Action Images checkpoint | ✅ 已在 Table 2 里 |
| **segmentation** | ❌ **不存在合适的外部 baseline** | 见下 |

> **为什么 segmentation 没有外部对照**:我们做的是 **scene-role 分割**
> (target / goal / distractor / fixture / robot_arm / gripper ...),
> 角色是**由任务语义和指令定义的**。SAM 系列是类别无关的(只分割不命名),
> 语义分割模型用的是 COCO/ADE 词表,与我们的角色词表不对应。
> 一个公平的外部 baseline 必须在我们的角色词表上训练 —— 那它就变成了我们自己的 specialist。
> **论文应当直接说明这一点,而不是硬凑一个不可比的数。**

### 29.6 其他候选(调研记录,未采用)

- **Depth Anything V2/V3**、**UniDepthV2**、**Metric3Dv2**、**ZoeDepth**、**DepthPro**:
  都是单目 metric depth 的强 baseline;UniDepthV2 在多个零样本 benchmark 上领先。
  未采用的原因只是 VGGT 权重已在本地、且本项目此前用过,**边际收益不足以再装一个**。
  若审稿要求多个 depth baseline,UniDepthV2 是首选补充(它输出真 metric,不需要对齐)。
- **SAM3**:Vision Banana 用它做分割对照,但如上所述与 scene-role 协议不可比。

---

## 三十、外部 specialist baseline 全部跑完(2026-08-31)

同一批 **8 个 held-out variation1 episode**,外部模型全部拿**真实 GT RGB** 做输入。

### 30.1 深度:三个外部模型

| 模型 | 输出类型 | raw AbsRel ↓ | **对齐后 AbsRel ↓** | 对齐方式 |
|---|---|---|---|---|
| VGGT-1B | 尺度歧义深度 | 0.357 | **0.206** | 逐帧中位数尺度 |
| Depth Anything V2-L | **视差(逆深度)** | 6.948 | **0.108** | 视差空间仿射(scale+shift) |
| Depth Anything 3-L | 尺度歧义深度 | 0.382 | **0.191** | 逐帧中位数尺度 |
| **我们(IIII)** | **米制,无需对齐** | **0.154** | — | — |
| 我们(FIFI) | 米制 | 跑中 | — | — |

**怎么读**:
- **raw 那一列我们全胜**,而且不是小胜(0.154 vs 0.206 / 6.95 / 0.382)——
  因为**只有我们直接输出米制深度**,其余三个都需要用 GT 拟合一个自由参数。
- **对齐之后 DA-V2 最好(0.108)**,优于我们的 0.154。DA-V2 拿了两个自由参数
  (scale + shift,逐帧用 GT 拟合),而且它看的是真实 RGB,我们连 RGB 都在生成。

> ⚠️ **必须写明的免责**:三个外部模型都是**零样本**迁移到 RLBench 合成渲染,
> 我们在这个分布上**微调过**;而且 VGGT/DA3 的设计目标是真实世界多视角重建。
> **这是主场优势对比,给"这个任务有多难"划参照线,不是能力排名。**

### 30.2 ⚠️ 一个差点写错的 bug:DA-V2 输出的是视差不是深度

第一次跑 DA-V2 得到 AbsRel **7.5**,看起来像"外部模型烂得离谱"。
真因是 **Depth Anything 的 relative 模型输出逆深度(视差),越大越近**,
拿它去和深度做中位数尺度对齐在数学上没有意义。

改成该系列论文自己的评测协议 —— **在视差空间用最小二乘拟合 scale + shift 到 1/GT,
再取倒数** —— 之后变成 0.108。**如果没查这一步,会得出一个对我们极其有利但完全错误的结论。**

### 30.3 分割:SAM 的类别无关对照

按"只看分得对不对、不看命名"的口径:对每个 GT 角色区域,取**任意** SAM mask 的最佳 IoU。

| | 值 |
|---|---|
| SAM ViT-H,class-agnostic best-overlap IoU | **0.619** |
| SAM 平均每帧产生的 mask 数 | **48** |
| GT 平均区域数 | **3.8** |
| **过分割倍数** | **13×** |
| 我们的 macro-mIoU(**带正确角色标签**) | 0.461 |

**这两个数不可直接比较,差距的方向被 SAM 的先天优势主导**:
- best-overlap **奖励过分割** —— 产生 48 个候选,总有一个能盖住任意区域
- SAM **不需要命名**;我们的 0.461 是**每个角色都要叫对名字**的严格指标
- 我们每个角色只输出一个区域(约 4 个),SAM 是它的 13 倍

诚实的结论:**SAM 能把边界切出来(0.619),但它不做我们要做的事**
—— scene-role 分割要求把区域**归到任务语义角色**(target/goal/distractor/fixture),
这正是 SAM 定义上不提供的。

> 若要完全对称,应当也算**我们的** class-agnostic best-overlap(忽略标签)。
> 那需要重跑一次 seg 生成并保留 mask(~47 min)。**目前没做,不能声称"我们边界更好"。**

### 30.4 环境记录

| 依赖 | 状态 |
|---|---|
| VGGT-1B | 权重已在 `ActionImages/checkpoints`,`vggt` 包已装 |
| Depth Anything V2-L | transformers **原生支持**,开箱即用 |
| **Depth Anything 3-L** | HF 仓库的 config 只有 `{model_name, config}`,**不是 transformers 格式**;需 `pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git`(已装) |
| SAM ViT-H | transformers `mask-generation` pipeline,开箱即用 |
| **UniDepthV2** | ❌ **PyPI 上没有**,需从 GitHub 源码装。**未完成** —— 它是唯一输出**真 metric** 的候选,补上它能得到一个不需要对齐的对照,是最有价值的后续项 |

### 30.5 给论文的建议写法

Table 1 增加一个 **"External specialist (zero-shot)"** 分区,只放 depth 一行
(seg 和 normal 没有可比的外部对照,直接说明原因),并**同时报 raw 与 aligned**
两列 —— raw 那列体现"我们直接输出米制"这个真实优势,aligned 那列体现"作为纯深度
估计器我们还不如专用模型",两者都诚实。

---

## 三十一、FIFI 的判别式设定:一个反直觉但重要的结果(2026-08-31)

为了和外部模型可比,给 `heldout_batch_eval.py` 加了 `--mode {iiii,fiii,fifi}`,
用 FIFI(给定真实 RGB,只预测辅助流)重跑 arm6:

| 设定 | depth AbsRel ↓ | normal cos ↑ |
|---|---|---|
| **IIII**(同时生成 RGB 与深度) | **0.154** | 0.892 |
| **FIFI**(给定 RGB,只预测深度) | **0.215** [0.033, 0.541] | **0.965** |

### 31.1 ⚠️ depth 的 FIFI 比 IIII **更差**,而且 CI 极宽

这不是 bug,是 **M2 混合的直接后果**:感知模板里 `IIII` 占 90%、`FIFI` 只占 **5%**。
FIFI 是被刻意欠训练的那个模式,草稿早就记录过它"逐 episode 不稳定"。
CI **[0.033, 0.541]** 正是这种不稳定的量化表现。

**对外部对照的影响很大**:与 VGGT / DA2 / DA3 真正可比的是 FIFI(它们都拿真实 RGB),
而我们在 FIFI 上是 **0.215**,**劣于全部三个对齐后的外部模型**
(DA-V2 0.108 / DA3 0.191 / VGGT 0.206)。

**诚实的结论**:
> **作为纯判别式深度估计器,我们不如专用模型** —— 这是我们刻意的训练选择造成的
> (把 95% 的感知训练量给了联合生成),不是架构缺陷。
> **我们的卖点是同一套权重还能生成未来和动作,而那三个模型都做不到。**

### 31.2 normal 相反:FIFI 大幅优于 IIII(0.965 vs 0.892)

给定真实 RGB 时法向估计明显更容易,符合预期。
说明 §31.1 的 depth 异常确实来自 FIFI 欠训练,而不是 FIFI 这条路径本身有问题。

### 31.3 建议的论文写法

Table 1 若加外部对照,**必须同时给出 IIII 与 FIFI 两行**,否则:
- 只报 IIII(0.154)会造成"我们比 VGGT/DA3 强"的错误印象 —— 那是拿不同任务比
- 只报 FIFI(0.215)会掩盖"我们在更难的联合生成设定下还有 0.154"这个真实结果

---

## 三十二、SAM 对照的对称版本:我们赢 18 个点,且 mask 数少 11 倍(2026-08-31)

用**与 SAM 完全相同的 label-free 口径**(每个 GT 角色区域取任意预测 mask 的最佳 IoU,
同样的 8 个 episode、同样的帧 [0,20,40]、同一个 `best_overlap` 函数)重算我们自己:

| | best-overlap IoU ↑ | 每帧 mask 数 | 过分割倍数 | 带角色标签? |
|---|---|---|---|---|
| SAM ViT-H | 0.6189 | 48 | **13×** | ❌ |
| **我们(FIFI)** | **0.7997** | **4.2** | **1.1×** | ✅ |

**我们在这个指标先天不利的一侧领先 18 个点。** best-overlap 奖励过分割
(候选越多越容易盖住任意区域),SAM 每帧产生 48 个候选、我们只有 4.2 个 —— 基本等于
GT 的区域数(3.8)。也就是说我们**几乎不靠猜**,而且输出还是**带角色语义的**。

逐 episode:0.638 ~ 0.918,**8 个全部在合理区间,没有一个崩**。

### 32.1 ⚠️ 对比同一批 episode 的 FIFI **depth**:分割稳、深度不稳

| 模态(均为 FIFI) | 结果 |
|---|---|
| segmentation | 0.800,8/8 稳定(0.64–0.92) |
| **depth** | 均值 **1.538**、中位数 0.087,**3/8 爆掉**(1.02 / 3.74 / 7.28) |

同一个 checkpoint、同一个 conditioning 模式、同一批 episode,
**分割完全稳定而深度有 37.5% 的失败率**。

这说明 §31 观察到的 FIFI 不稳定**不是 FIFI 这条路径本身的问题**
(否则分割也该崩),而是**深度这个模态在 FIFI 下欠训练**的特有表现。
配合 normal 在 FIFI 下反而更好(0.965 vs IIII 的 0.892),三个模态的表现完全分化。

### 32.2 ⚠️ 小样本在这一轮已经误导过三次

| 情形 | 小样本值 | 全量值 | 差异 |
|---|---|---|---|
| VGGT pilot(3 ep → 8 ep) | 0.093 | 0.206 | 2.2× |
| FIFI depth(6 ep → 8 ep) | 0.215 | **1.538** | **7.2×** |
| tier-3 感知(2 任务 → 6 任务) | 0.171 | 0.150 | 12% |

**都是抽样恰好避开了难例。** 这轮的教训应写进方法论:
**本任务的 episode 间方差极大,任何 n<8 的读数都不可信**,尤其是 depth
(它的失败是灾难性的,不是渐进的,均值会被单个 episode 拖动数倍)。

### 32.3 IIII 模式的 class-agnostic 结果:补上了对照的另一半

| | best-overlap IoU ↑ | mask/帧 | 过分割 | 输入 RGB |
|---|---|---|---|---|
| SAM ViT-H | 0.619 | 48 | 13× | **真实** |
| **我们 FIFI** | **0.800** | 4.2 | 1.1× | **真实** |
| 我们 IIII | 0.563 | 4.0 | 1.1× | **自己生成的** |

**这三行才是完整的故事**:

- **同等条件下(都拿真实 RGB)我们赢 18 个点**,而且 mask 数少 11 倍。
- **让我们连 RGB 一起生成,就落到 0.563,低于 SAM 的 0.619** —— 差距 5.6 个点,
  但 SAM 是在**真实像素**上分割,我们是在**自己想象出来的像素**上分割。
- FIFI → IIII 掉 24 个点(0.800 → 0.563),**这就是"未来预测误差"的代价**,
  可以直接量化:分割质量的约 30% 来自 RGB 预测的准确度。

> **论文里该怎么用**:这是少见的能把"感知能力"和"世界模型能力"**分开计量**的地方。
> 同一个 checkpoint、同一批 episode、同一个指标,唯一区别是 RGB 给不给。
> 差值(0.237)就是世界模型这一环节引入的误差。

### 32.4 三个模态在 FIFI→IIII 上的表现完全不同

| 模态 | FIFI(给 RGB) | IIII(生成 RGB) | 方向 |
|---|---|---|---|
| segmentation (best-overlap) | 0.800 | 0.563 | FIFI 好 24pp |
| normal (cos) | 0.965 | 0.892 | FIFI 好 |
| **depth (AbsRel)** | **1.538 均值 / 0.087 中位** | **0.154** | **IIII 反而好** |

**只有 depth 反向,而且是因为它在 FIFI 下有 3/8 的崩溃率。**
中位数口径下 FIFI(0.087)其实优于 IIII(0.154),与另外两个模态一致 ——
**所以 depth 的 FIFI 均值异常完全由少数崩溃 episode 造成,不是系统性劣势。**
论文里报 depth 的 FIFI 必须同时给均值和中位数,否则会误导。

---

## 三十三、Table 3 与 Table 4 合并(2026-09-01)

### 决策

用户要求把 Table 3(三档泛化)和 Table 4(unified vs specialist 在没训练过的任务上)
合并成一张表。**已执行**,改动落在:

- `paper/experiments.md` —— LaTeX 主体,`tab:generalization` 扩成 7 列,
  `tab:cross_task_specialists` 整块删除,原引导段改写成指向合并表右半区
- `paper/outline.md` —— 同步,并顺手修掉注释里几个过时数字(见下)
- `paper/实验结论汇总.md` —— 5.2 换表、5.5 改写成"为什么合并"

### 为什么该合并

Table 4 本来就是 **Table 3 最后一列的下钻**:"这是 tier-3 的数字 / 这是 specialist
在同一批 episode 上的数字"。分成两张表时,读者看到 ③ 的 0.144 无从判断好坏,
必须翻页保持上下文。合并后:

1. 对照就在被提出的位置,不用跨页
2. "统一是否带来泛化"这个问题的否定答案变成 tier-3 那列的**自带对照**,
   而不是一个孤立的补充实验
3. 9 页的 ICLR 少一张表
4. 上半区(感知三档几乎平)和下半区(控制第二档崩到 1.7%)在同一张表里对撞,
   本文最核心的那个对比一眼可见

### ⚠️ 合并的硬约束:③ 整列必须换成 n=48 配对子集

**这是差点写错的地方。** 两张表原来用的不是同一批 episode:

| | 原 Table 3 | 原 Table 4 |
|---|---|---|
| unseen-task depth | 0.150(n=66,arm6 全量) | 0.144(n=48,与 specialist 配对) |

直接把两张表拼在一行,得到 "unified 0.150 / specialist 0.137 / Δ +0.007" ——
**算术上说谎**,因为 0.150 − 0.137 = 0.013 ≠ 0.007。配对差只在共同子集上有定义。

处理:**整个 unseen-task 块统一用 n=48**(depth 0.150 → 0.144),
多出来的 18 个 episode 放附录。这也是对的 —— 那一块存在的全部意义就是配对比较。

verified: depth 全量 n=66 = 0.1503,配对 n=48 = 0.1437。

### Specialist 列的空格是有意的

RGB LPIPS 和闭环两行 specialist 列填 `---`:LPIPS 能算但不回答这个问题,
闭环压根没跑过 specialist 的 6 任务版。留空比硬凑一个数诚实。

### 顺手修掉的过时数字(outline.md 注释区)

| 位置 | 旧 | 新 |
|---|---|---|
| specialist 配对表 | 0.171/0.173,"same 16 episodes" | 0.144/0.137,48 episodes |
| tier-3 闭环 | "35% again" | 40% |
| 官方 checkpoint 对照 | "50% on those same two tasks" | 43.8% on those same six tasks |
| 不声称清单 | "Two unseen tasks is a probe" | Six unseen tasks |
| 法向稳定性 | "0.895 -> 0.898" | 改为 [XX],等扩样本 |

### 数字状态:seen / unseen-var 两列全部标 `[XX]`

n=40 扩样本正在跑,**已经在动**:seen 档 depth 从 0.136(n=8)变成
**0.293(n=23,中位数 0.147,最大 3.12)** —— 又一次印证 depth 的双峰特性,
n=8 时那个 0.136 是运气。

所以这次**只改结构不填数字**,左半两列一律 `[XX]`,caption 的样本量也是 `[XX]`。
扩样本跑完(specialist ~1h,arm6 ~12h)后统一刷新。unseen-task 那一块的
n=48 配对数字是终值,不受影响。

备份:`/tmp/exp_before_merge34.md`、`/tmp/outline_before_merge34.md`

---

## 三十四、监控窗口(2026-09-01 01:00–）：三个发现 + 两个我自己的 bug

### ⚠️ bug 1:我的 monitor 把进度虚报了 2×

`per_episode.json` 每次生成写**两条** key(`depth` 和 `video__via_depth`),
我把 key 数当成生成数,于是:

| | 我报的 | 实际 |
|---|---|---|
| arm6 | 292/320 (91%) | **153/320 (48%)** |
| action specialist | "DONE 80/80" | **40/80 (50%)** |

那条 "DONE" 通知是假的。已修:monitor 现在只数非 `video__via_*` 的 key。
**ETA 从 ~1h 回到 ~8h。** 差点就拿半截数据去刷新表格了。

### ⚠️ bug 2:`summary.json` 会被窄 episode 列表的续跑覆盖

action specialist 的 `summary.json` 时间戳是 08-31 07:44、内容是 n=8 / 16 episodes,
而 `per_episode.json` 是 09-01 01:02 的当前数据。之前修的"resume 截断 summary"
只修了**模态**维度,**episode** 维度还在。

**结论:刷新表格一律从 `per_episode.json` 重算 bootstrap,不要读 `summary.json`。**
刷新脚本已按这个写:`scratchpad/refresh_tab3.py`。

### 发现 1:seen 档 depth 的"退化"是**单个 episode** 造成的

```
depth seen  n=36  mean=0.2384  median=0.1309  trim10%=0.1329  max=3.1216
```

`close_jar/variation0/episode4` 一个 episode 的 AbsRel = **3.12**,
剩下 43 个里只有 4 个 >0.3。把它去掉,均值回到 0.156。
**"seen 比 unseen 还差"这个反常现象完全由它一个造成。**

### 发现 2:它不是坏 GT,是**真实的生成失败**

CPU 侧核对该 episode 的 GT depth:valid_frac=1.000,范围 0.813–3.301 m,
和正常 episode(0.490–2.520)同量级。**GT 干净,所以不能以协议理由剔除。**

### 发现 3:⭐ depth 误差与 RGB 质量强相关,但那个离群点**解耦**

同一次生成里 depth 和 RGB 是从同一段 packed latent 解出来的,拿 `video__via_depth`
的 LPIPS 做配对:

```
Spearman rho(depth AbsRel, RGB LPIPS) = +0.691  p<1e-4   (n=44)
剔除 AbsRel>1 后                        = +0.748  p<1e-4   (n=43)
```

**相关性在剔除离群点后变强** —— 说明绝大多数 depth 误差只是"那个 episode 整体生成得差",
不是 depth 特有的问题。

但 `close_jar/ep4` 反过来:它的 RGB LPIPS = **0.1633,是 close_jar 五个 episode 里最好的**,
而 depth 崩到 3.12。

| close_jar/var0 | depth AbsRel | 同次生成 RGB LPIPS |
|---|---|---|
| episode0 | 0.1244 | 0.2603 |
| episode1 | 0.1525 | 0.2688 |
| episode2 | 0.1462 | 0.1821 |
| episode3 | 0.2129 | 0.2223 |
| **episode4** | **3.1216** | **0.1633** ← 最好 |

**两种不同的失效模式:**
1. 常见的:整段生成质量差 → depth 跟着差(ρ=0.75 那条线)
2. 罕见的:RGB 完美但 depth 单独崩 → 失效在 depth 流的 latent→codec 解码路径上,
   不在扩散采样上

第 2 种如果能复现,是论文里一个具体的 limitation,比笼统说"偶尔会失败"有价值得多。

### ⚠️ 但现在还不能定 reporting 口径

seen 已到 n=36,unseen-var **还停在旧的 n=8**(这轮先跑 variation0)。
拿 n=36 和 n=8 比一个**对离群点极度敏感**的统计量是无效的 ——
unseen-var 扩到 40 之后完全可能也冒出自己的离群点。

**所以"mean 还是 median"这个决定必须等两档都到 n=40 再做。** 现在记下候选方案:

- 表里主报 **mean**(与 DA2/DA3/VGGT 的口径一致,那几个也是 mean),
- 脚注同时给 **median + 失败率**(如 `1/40 episodes with AbsRel>1.0`),
- 如果 mean 让 tier-1 反而比 tier-2 差,**必须在正文点破是单个 episode 造成的**,
  不能让读者自己去猜。

### 本窗口的调度改动

- GPU 1 空着(6 MiB),而 `queue_maintable.sh` 是**严格串行**的,
  四个外部 baseline 排在队尾、约 12h 后才轮到。已另开
  `scripts/queue_external_gpu1.sh` 在 GPU 1 上先跑 da2/da3/vggt/sam。
- 为了让两个 worker 不重复劳动:给三个 probe 脚本加了
  **落盘 `reports/external/<name>_n<N>.json` + 存在即跳过**的 guard。
  副作用是好的 —— 这些结果原来只存在于 log 文本里,现在有结构化输出了。
- **没有改 `queue_maintable.sh`**:bash 是边读边执行的,改正在运行的脚本会让它执行到错位的字节。

---

## 三十五、⭐ 那个 3.12 是 unified 独有的失败,但"统一没有代价"的结论反而更稳了(2026-09-01)

### 起点:同一个 episode,specialist 拿了全场最好的分

翻了所有历史报告,`close_jar/variation0/episode4` 只被两个模型评过:

| 模型 | 该 episode 的 depth AbsRel |
|---|---|
| unified (arm6@10000) | **3.1216** ← 全分布最差 |
| depth specialist (matched, step2000) | **0.0441** ← 接近全分布最好 |

同样的 GT、同样的数据树、同一个官方 checkpoint 热启动、matched exposure。
**所以这是 unified 独有的失效,不是数据问题**(GT 已在上一节验证干净)。

第一反应是"这对论文不利"。**实际算完之后是相反的。**

### 配对比较:unified vs depth specialist(共同 44 个 episode)

```
unified 更好: 24/44      specialist 更好: 20/44        ← 掷硬币
Wilcoxon signed-rank  p=0.616                          ← 对秩,不受 3.12 支配
median diff  = -0.0024   95% CI [-0.0093, +0.0026]     ← unified 略好,CI 极窄且含 0
mean   diff  = +0.0656   95% CI [-0.0134, +0.2124]     ← CI 宽且含 0,完全被那一个 episode 拖
剔除该 episode 后: unified 0.1555 vs specialist 0.1600  Δ=-0.0045  (n=43)
```

**每一个稳健统计量都说"没有差别",甚至 unified 略好。**
唯一暗示 unified 更差的是均值,而它的 CI 也含 0。

### 而且 specialist 也有自己的失败

不是 unified 单方面不稳:

| episode | unified | specialist | |
|---|---|---|---|
| close_jar/var0/ep4 | **3.1216** | 0.0441 | unified 崩 |
| close_jar/var0/ep0 | 0.1244 | **0.3136** | specialist 崩 |
| reach_and_drag/var0/ep1 | 0.0968 | **0.1700** | specialist 差 |

两边各有翻车。specialist 的最差值是 0.4394,没有量级级别的崩溃 ——
但 **n=44 里两个方向各一次事件,不足以声称"谁的尾部更重"**。

### 因此:reporting 口径定为**中位数为主、均值+失败率为辅**

- **表里主报 median**:它在两个模型上几乎相同(0.138 vs 0.143),
  是这批数据支持的结论;均值只反映"有没有抽到那个 episode"。
- **同时披露 mean 和失败计数**(如 `1/44 episodes with AbsRel > 1.0`),
  不藏起来。
- **正文明确写**:均值的差异由单个 episode 造成,Wilcoxon p=0.616,
  两个方向各有一次翻车,样本量不支持任何关于尾部行为的结论。

这比原来"unification costs nothing"更强 —— 因为它是在**主动去找反例之后**
仍然成立的。

### ⚠️ 尚未定案的前提

这次配对是 variation0 (36) + variation1 (8) 混着算的,而 unified 的
variation1 还停在 n=8。**扩样本跑完后必须重算**,并且分档报。

### 顺带:DA2 在 n=40 上完成

`reports/external/da2_n40.json`:median-aligned AbsRel = **0.0983**(n=8 时是 0.108),
逐 episode 范围 0.018–0.189,**没有任何量级级别的失败**。
probe 用的 episode 列表已核对 = `eps40.txt` 的 variation1 部分,**完全一致**,
所以 Table 1 里两边将是 n=40 严格配对。
但 **现在还不能比** —— 我们的 variation1 只有 n=8,拿 n=8 比 n=40 是上一节栽过的同一个坑。

---

## 三十六、⭐ 均值不是这批数据的合适汇总量 —— 外部 baseline 也一样(2026-09-01)

n=40 的外部 baseline 陆续落地后,把五个模型的 depth 分布并排看:

| 模型 | n | mean | median | max | >0.3 | >1.0 |
|---|---|---|---|---|---|---|
| VGGT-1B | 40 | 0.2384 | **0.1550** | **2.0108** | 7/40 | **1/40** |
| DA-V2-L | 40 | 0.0983 | 0.0973 | 0.1885 | 0/40 | 0/40 |
| DA3-L | 40 | 0.1794 | 0.1852 | 0.4012 | 4/40 | 0/40 |
| Ours unified (IIII) | 45 | 0.2220 | **0.1419** | **3.1216** | 5/45 | **1/45** |
| Ours depth spec. | 80 | 0.1619 | 0.1445 | 0.5557 | 9/80 | 0/80 |

### 结论 1:VGGT 的失败剖面和我们**几乎一模一样**

```
VGGT        mean 0.2384 / median 0.1550   max 2.01   1/40 超过 1.0
Ours (IIII) mean 0.2220 / median 0.1419   max 3.12   1/45 超过 1.0
```

均值被少数 episode 拉高、中位数正常、各有一次量级级崩溃 —— 两个模型同一个形状。
**所以"偶发灾难性 depth 失败"不是统一模型的毛病,是这类模型在这个数据上的共同现象。**
上一节里我担心的"unified 独有的失效",在有了外部对照之后,变成了"连最强的
feed-forward 几何模型也有"。

### 结论 2:DA-V2 是**可靠性**上的异类,不只是精度

mean ≈ median(0.0983 vs 0.0973),max 0.189,**40 个 episode 里一个 >0.3 的都没有**。
它不是"平均更准",是"从不翻车"。这是比精度差距更值得写进论文的一句话 ——
也是我们对 DA-V2 唯一站不住的地方,不该淡化。

### 因此:**全表主报中位数**,而且这不是为我们开脱

如果只有我们的模型需要中位数,那是特殊辩护。但 VGGT 的 mean/median 差距
(0.238 / 0.155)比我们的(0.222 / 0.142)还大 —— **对外部 baseline 报均值同样会
系统性地低估它们**。所以统一改成中位数是对所有模型都更公平的口径,
同时把 mean 和 `>1.0` 计数放脚注全部披露。

### ⚠️ 口径警告(还不能直接比这张表)

VGGT/DA2/DA3 都只在 **variation1 的 40 个** episode 上;
我们的两个模型现在是 variation0+variation1 混合、且 n 各不相同。
**这张表只能用来看"分布形状",不能用来排名。** 排名要等我们的 variation1 补满 40。

### 方法学改动:对齐口径不再靠注释,靠数据

DA3 的结果作废重跑了。原因是脚本里给它选 `median_align` 的依据只是一句注释
(*"unlike DA V2 it produces depth rather than disparity"*)—— 而这正是当初让
DA2 算出 AbsRel 7.5 的同一类假设,且那次的错误方向对我们有利。

现在三个 probe 都改成**用 ρ(pred, GT depth) 的符号判断输出空间**:
视差与米制深度反相关,深度正相关。JSON 里同时落盘 raw / median-scale /
disparity-affine 三种口径 + ρ + 实际选中的 `alignment` 和 `output_space`,
任何人都能复核这个判断。

- 旧结果移到 `reports/external/superseded/`
- 重跑链:`scripts/rerun_depth_probes.sh`(da2 → da3 → vggt),已挂在 GPU-1 队列之后
- DA2 大概率不变(ρ 为负 → 仍选 disparity-affine → 0.098);**待定的是 DA3 和 VGGT**

---

## 三十七、🚨 协议 bug:外部 baseline 一直在**和我们不同的视角/时间窗**上打分(2026-09-01)

### 怎么发现的

按新的 ρ 判据重跑 DA2,结果是 **0.1047**,而几小时前那次是 **0.0983** ——
同样 40 个 episode、同样的对齐方式、同一个确定性前馈模型。逐 episode 比对:

```
排序后是否相同(即是否只是顺序不同): False
40/40 个 episode 全都不同,最大绝对差 0.1216
  idx14  old=0.0992  new=0.2208   Δ=+0.1216
  idx10  old=0.1764  new=0.0584   Δ=-0.1180
```

**同一个 probe 跑两次,单个 episode 能差 0.12 AbsRel。**

### 根因

`RLBenchSelfgenDataset.getitem` **随机抽相机视角和时间窗**,并把抽到的结果作为
provenance 记下来(代码里那段注释正是为此):

```
# Refuse to guess. Re-deriving the window/views here would be a fresh random
# draw describing different pixels than the camera tensors already in `sample`
```

我们自己的 eval(`heldout_batch_eval` / `modality_mode_grid`)在每次 getitem 前调
`_seed_all(42)`。**三个外部 probe 一个都没调:**

| 脚本 | `_seed_all` |
|---|---|
| probe_depth_baselines.py | ❌ 0 |
| probe_vggt_vs_ours.py | ❌ 0 |
| probe_sam_vs_ours.py | ❌ 0 |
| probe_ours_class_agnostic_seg.py | ✅ 有 |

### 后果:Table 1 的外部对照**不是配对比较**

我之前反复强调"episode 列表完全一致所以是严格配对" —— **episode 一致,但像素不一致**。
VGGT/DA2/DA3/SAM 看到的是随机抽的另一个相机视角和另一段时间窗。

这解释了之前那些莫名其妙的漂移:VGGT 0.093 (n=3) → 0.206 (n=8) → 0.238 (n=40),
我一直归因于样本量,其中有一部分其实是这个。

**所有已有的外部 baseline 数字全部作废**,移到 `reports/external/superseded/`。

### 修法

三个 probe 全部在 `ds.getitem` **紧邻的上一行**加 `_seed_all(42)`,
和 `heldout_batch_eval` 用同一个种子、同一套 RNG(random / numpy / torch / cuda)。
加了断言确保 seed 和 getitem 相邻,防止以后有人插代码进来把它们隔开。

重跑脚本:`scripts/rerun_external_seeded.sh`(da2 → da3 → vggt → sam,约 30 min)。

### 顺带修掉的第二个 bug(我自己引入的)

给 probe 加落盘时,`_save(...)` 被插在了定义 `chosen` / `mean_rho` 的代码块**之前**,
导致 `UnboundLocalError`,da2 跑完 40 个 episode 后在写文件那一步崩掉。
已把 `_save` 移到 `main()` 末尾,并用 AST 校验它确实是 main 的最后一条顶层语句。

### 已确证的结论(这个不受 seed 影响)

DA2 的输出空间:**40/40 个 episode 的 ρ(pred, GT) 全为负**,均值 **-0.941**。

```
raw AbsRel        6.7319
median-scale      1.6482   <- 当成深度对齐会得到这个
disparity-affine  0.1047   <- 正确口径
```

**用错口径会把 DA2 的误差夸大 15.7 倍。** 这也再次说明"用 ρ 判断而不是信注释"
是必要的 —— DA3 和 VGGT 的口径同样在重跑里验证。

---

## 三十八、⚠️ 更正第三十六节:seed 修好后,"VGGT 也会灾难性失败"不成立(2026-09-01)

第三十六节我写过:*"VGGT 的失败剖面和我们几乎一模一样 …… 所以偶发灾难性 depth 失败
不是统一模型的毛病"*。**那是基于未 seed 的数据,现在作废。**

全部 seed=42 重跑后(外部模型只在 variation1):

| 模型 | split | n | mean | median | max | >0.3 | >1.0 |
|---|---|---|---|---|---|---|---|
| DA-V2-L | var1 | 40 | **0.1254** | **0.1274** | 0.3077 | 1 | 0 |
| **Ours unified (IIII)** | var1 | 26* | 0.1624 | 0.1437 | 0.5401 | 2 | **0** |
| DA3-L | var1 | 40 | 0.2015 | 0.1968 | 0.4114 | 6 | 0 |
| VGGT-1B | var1 | 40 | 0.2318 | 0.2112 | **0.8593** | 10 | **0** |
| Ours unified (IIII) | var0 | 40 | 0.2291 | 0.1389 | **3.1216** | 5 | **1** |

\* 仍在跑,n 未满 40。

### 更正 1:VGGT 的 max 从 2.01 降到 0.859,>1.0 从 1/40 变成 0/40

那个 2.01 是**未 seed 时抽到一个倒霉视角**造成的,不是 VGGT 的固有失败模式。
**四个外部模型在 seed=42 下没有一个出现量级级失败。**

### 更正 2:那个 3.12 目前仍然是我们独有的

它在 **variation0**,而外部模型只跑了 variation1。所以严格说:
在**可比的 variation1 上,我们目前 0/26 出现 >1.0**,和外部模型一样干净。
第三十五节那个"unified 独有的失效"的判断**在 var0 上依然成立**
(depth specialist 在同一 episode 上是 0.0441),但它**不能**再用
"连 VGGT 也这样"来淡化。

### 更正 3:"均值不是合适汇总量"要收窄

在 variation1 上,**所有模型 mean ≈ median**(我们 0.1624/0.1437,
DA2 0.1254/0.1274,DA3 0.2015/0.1968,VGGT 0.2318/0.2112)。
双峰只出现在**我们的 variation0**。

**所以 reporting 口径改回:全表主报 mean(标准做法),只在 Table 3 的 seen 那一格
额外披露 median 和 `1/40 AbsRel>1.0`。** 第三十四节那个"全表改中位数"的方案取消 ——
它当时的依据(外部模型也需要中位数)已经被证明是 seed bug 的产物。

### 目前 variation1 上的实际排名(我们的 n 未满,暂时性)

```
DA-V2-L        0.1254   零样本,读 GT RGB,2-DOF 视差仿射对齐
Ours (IIII)    0.1624   自己生成 RGB,同时出 depth,米制无需对齐
DA3-L          0.2015   零样本,读 GT RGB
VGGT-1B        0.2318   零样本,读 GT RGB
```

我们排第二,且是唯一**同时生成 RGB**而不是读取 GT RGB 的。这个对比本身要在
caption 里说清楚,不能让读者以为是同等条件。

### 分割那一侧(全部 seed=42, n=40)

| | best-overlap IoU | mask/帧 | GT 区域/帧 | 过分割 |
|---|---|---|---|---|
| SAM ViT-H | 0.6228 | 62 | 3.8 | **16×** |
| Ours FIFI | **0.8379** | 4.0 | 3.7 | **1.1×** |
| Ours IIII | 0.5976 | 4.0 | 3.7 | 1.1× |

FIFI 领先 SAM **21.5 个点**,且 mask 数只有 SAM 的 1/15。
注意 best-overlap 本身是**奖励过分割**的指标 —— 在一个对 SAM 有利的指标上
我们仍然赢,这句话可以写进论文。

---

## 三十九、审查 1.7% 的闭环流程:一个记账 bug + 一个可能主导结果的中止规则(2026-09-01)

用户要求检查 tier-2 那个 1.7% 的 eval 流程有没有问题。查完:**超参没问题,但有三件事。**

### 先排除的(都核对过,没问题)

| 检查项 | 结果 |
|---|---|
| 三档超参 | cfg 7.5 / fi 3 / explicit / planning / sphere / max_steps_factor 1.5 / skip_anchor 4 **完全一致** |
| 语言指令 | `reset_to_new_demo(task, variation, seed)` 返回该 variation 的 descriptions,取 `[0]` —— **没喂错** |
| 训练泄漏 | `scripts/train_arm.sh`: `VARIATIONS="${VARIATIONS:-0}"`,**只训 variation0**,var1 是干净的留出集 |
| 步数预算 | `max_steps` 随 demo 自动放大(var1 中位 57 步 vs var0 41)—— **不是跑不完** |
| replan / horizon | 两档都是 `execution_horizon=41`、`replans` 中位 1–2 —— 一致 |

### 问题 1:close_box / close_drawer 被同时算进 tier-1 和 tier-3

这两个任务在 `data/rlbench_unseen_tasks_512_aug`,**训练树里 0 个 episode**,
却出现在 tier-1「seen」那次 campaign 的 5 个任务里。

```
tier1 报告里真正训练过的 3 个任务 : 21/60 = 35.0%
tier1 报告里 close_box+close_drawer: 14/40 = 35.0%   <- 从未训练过
tier3 论文口径(6 任务)          : 32/80 = 40.0%   <- 已经包含上面那 40 个
```

**那 40 个 rollout 被重复计入两档。** 数值上 35.0% 恰好相同(纯巧合),
所以表里的数字不用改,但 **n 应该是 60 不是 100**,且必须写明三档用的是不相交的任务集。
已修进 `tab:generalization` 的 caption。

### 问题 2:variation 改的是任务本身,不是外观

| 任务 | var0 | var1 | var2 |
|---|---|---|---|
| push_buttons | push the maroon button | maroon **then** green(**两步**) | maroon→green→blue(三步) |
| meat_off_grill | take the **chicken** off the grill | take the **steak** off the grill | — |
| open_drawer | open **bottom** drawer | open **middle** drawer | open top drawer |

`push_buttons` 的 var1 是**双步任务**,不是"同一任务的分布偏移"。
把它当作分布偏移会**夸大**模型的脆弱性 —— 这一点必须在正文说清楚。

`meat_off_grill` 的 chicken→steak 才是干净的指代变化,而它也从 45% 掉到 0%。

### 问题 3(最关键):`max_ik_fail_streak = 5` 硬编码,可能主导了 1.7%

```
push_buttons   var0: success 12, IK 挂 5,  timeout 3
               var1: success  1, IK 挂 16, timeout 3     <- 16/20 死于中止规则
               ik_fail_rate 0.049 -> 0.142
```

`eval/rollout.py:129` 的 `max_ik_fail_streak: int = 5` 是**硬编码默认值,原来没有命令行开关**。
它对两档一视同仁,但"连续 5 次不可达就中止"是一个会**放大**差异的规则:
几何上稍差的轨迹更容易连续踩空,于是从"差一点"变成"几乎全灭"。
而且它在 var0 也杀掉了 15/60。

**这不是 bug,是一个未被检验的 harness 策略** —— 但在没有对照的情况下,
把 1.7% 直接解释成"模型的泛化失败"是不成立的。

### 已做:加开关 + 两个对照实验

`eval/rollout.py` 新增 `--max-ik-fail-streak`(默认仍为 5,不改变历史行为)。

- **A**(GPU 0):var1,`--max-ik-fail-streak 50`,3 任务 × 20 trial
- **B**(GPU 1):var0,`--max-ik-fail-streak 50`,同样配置

**B 是必需的控制组** —— 只跑 A 就变成拿 var1@50 比 var0@5,又是一次口径不一致。
两者按 `scene_seed` 与已有的 streak=5 结果配对。约 6h。

判读标准:
- 若 var1 从 1.7% 跳到 ~20%,则 **1.7% 主要是 harness 造成的**,论文里那句
  "control is substantially more sensitive to variation shift" 必须重写;
- 若仍在 ~2%,才是模型的真实失败,原结论成立(但仍需说明 push_buttons 的双步混淆)。

### 待排:C — variation 2

`push_buttons` var2(三个按钮)和 `open_drawer` var2(最上层抽屉)。
如果失败随 variation index 单调加重,支持"horizon/指代难度"解释;
如果 var1 和 var2 一样差,更像是"只要不是 var0 就崩",指向别的机制。
等 A/B 释放 GPU 后再跑。

---

## 四十、Table 1 重构为 Vision-Banana 结构 + 全量 counterpart 扫描(2026-09-02)

### 表结构改了

用户要求按 Vision Banana 的结构重写 Table 1:
`Capabilities | Benchmarks and Metrics | Ours | Best Counterpart`,
统一模型只用 **FIFI**,external 只留最强的那个,其余进附录。这样没有空位。

写在 `paper/tab_main.md`(4 列,已校验)。三处必须知道的:

1. **FIFI 不产生 action。** 感知行用 FIFI(拿真实帧,与外部对照同条件),
   action 和 RGB 没有这个模式 —— 预测未来**就是**任务。那三行标 `*` 用 IIII,
   脚注写明数字含未来预测误差。
2. **depth 那格写 `0.385 / 0.038`(均值/中位)**,不取单值:中位数低于所有外部模型、
   均值高于所有,只报一个都是误导。
3. **这个结构暴露了 normal 行没有 counterpart** —— 之前从没跑过法向 baseline。

### ⭐ 全跑之后,counterpart 的排序和直觉相反

用户要求"所有相关模型都跑,选最好的"。跑完(var1, n=40, seed=42):

**法向**
| 模型 | mean cos |
|---|---|
| **Ours FIFI** | **0.969** |
| Lotus-G v1-1 | **0.9459** ← best counterpart |
| DA-V2 → depth_to_normal | 0.9146 |
| Marigold-Normals v1-1 | 0.7885 |

**那个"偷懒"的 DA-V2 求法向(0.915)反而赢了专用的 Marigold(0.789)。**
如果只跑 Marigold,counterpart 会是 0.789、我们领先 18 点;
实际最强是 Lotus 的 0.946,只领先 2.3 点。**这就是必须全跑的理由。**

⚠️ **unseen-task 档 Lotus 0.9435 已经超过我们的 IIII(0.908)。**
FIFI@unseen-task 还在排队,如果也没超过,那一格就是我们输,照实报。

**深度**(按自由参数分组 —— 这是"我们无需对齐"那个论点的关键)
| 模型 | 类别 | 自由参数/帧 | AbsRel |
|---|---|---|---|
| Ours FIFI | metric | **0** | 0.385 / **0.038** |
| DepthPro | metric | 0 | 0.3256 |
| DA-V2-Metric-Indoor | metric | 0 | 0.2725 |
| **DA-V2-L** | disparity | 2 | **0.1254** ← best counterpart |
| DA3-L | scale-amb. | 1 | 0.2015 |
| VGGT-1B | scale-amb. | 1 | 0.2318 |

### 打分口径改动:米制模型按 raw 打分

DepthPro / DA-V2-Metric 自称米制。我原来的脚本按 ρ 符号自动给它们做 median-scale 对齐,
**等于白送一个拟合参数** —— 而"我们不需要对齐"正是这个对比的争议点,
那样是在争议点上偏袒对方。现在 `model_class ∈ {metric, disparity, scale-ambiguous}`
和 `free_parameters_per_frame` 都落盘,三种口径全保留,附录可完整呈现。

### 新增的 counterpart 脚本

- `scripts/probe_normal_specialists.py` —— Lotus-G / Marigold-Normals,
  **坐标系是标定的不是假设的**:我们的 GT 法向是 x 右 / y 下 / z 背离相机,
  法向 benchmark 通常 y 上 / z 朝向相机。四种符号约定全算全落盘,
  取与 GT 一致性最高者。判据在看到谁赢之前定死、两模型同一判据 ——
  它选的是坐标系不是结果。结果毫不含糊:identity +0.96 vs flip_yz −0.88。
- `scripts/probe_named_seg_baseline.py` —— CLIPSeg,把 named-seg 那个 `n/a` 变成真数字。
  原来写 n/a 的理由是"SAM 出的区域没有名字":对 SAM 成立,对这个领域不成立。
  开放词表分割就是拿名字当输入。留 n/a 等于宣称没人能做,而事实是我们没跑。
- `scripts/collect_counterparts.py` —— 附录表生成器。

### 🐛 三个 bug

**1. 队列互抢卡 → 静默失败。** 5 个队列各自探卡再启动,卡在探测通过后、
模型加载前被抢走,而 `heldout_batch_eval` 在这条路径上**退出码 0**。
drift 诊断就是这么"完成"的(1 分钟 exit 0,实际 OOM)。
已建 `scripts/gpu_lease.sh`(原子 `mkdir` 锁),并给 drift 加了**产物存在性检查 + 重试**。

**2. 孤儿进程带着旧代码继续跑。** kill sweep 时它的 `gpu_lease.sh acquire` 子进程存活,
bash 已把**旧脚本读进内存**,于是每 120s 重试报同样的错,
并往继承的 log fd 的**旧 offset** 写入(日志前面一大片空洞)。
两个孤儿活了 1 小时,让修好的 lease 看起来仍然坏 ——
**症状是"改完代码错误照旧",极易误判。**
已加 `scripts/qrun.sh`:`setsid` 启动、`kill -- -PGID` 整组结束。

**3. `collect_counterparts` 漏掉 VGGT。** VGGT 的 json 用 `absrel_median_aligned`
而非 `absrel_aligned`,汇总时被静默跳过 —— 一个对照就这么从 "best of" 里消失了。已修。

**4. `heldout_batch_eval` 汇总打印在空 split 上崩**(unseen-task 树没有 variation1,
`n` 是 None)。数据早写完了,但崩在这看起来像跑失败。已修。

### 一次误判(记下来提醒自己)

我一度报告"同样 seed 的 DA2 跑出 0.1254 和 0.1331,seed 修复不完整"。
实际是我自己的 glob 把 `da2_seen_n40.json`(variation 0)和 `da2_n40.json`(variation 1)
混在了一起 —— **不同 split 不同数,完全正常。**
`collect_counterparts.py` 现在按文件名解析 split,不再靠松散 glob。

---

## 四十一、⭐ 闭环有 7% 的运行间翻转率 —— 影响所有 per-task 数字怎么读(2026-09-02)

### 怎么发现的

ikctl 的 `light_bulb_in` trial 0 在两次运行里 `ik_fail` 是 4.0% 和 0.0%,
同一个 `scene_seed`、同样的中止阈值(该 trial 两次都没触发中止)。

我先怀疑是**任务顺序影响 RNG**(ikctl 只跑 1 个任务,原批次里它排第 3)。
查代码否掉了:`inference.py:292` 每次生成都 `torch.manual_seed(42)`,
**RNG 每次调用重置,任务顺序无影响。**

### 免费的对照:两个批次的重叠部分

旧的 3 任务 × 20 trial(`arm6_10000_var1`)和新的 13 任务 × 10 trial
(`arm6_alltasks_v1`)在 `push_buttons` / `meat_off_grill` / `open_drawer`
的 variation1 trial 0-9 上完全重叠,**配置逐项一致**:

```
scene_seed 相同        30/30
success 一致           28/30 (93%)   ← 2 个翻转,方向相反
ik_fail_rate 逐条相同  否,平均绝对差 0.016
steps 逐条相同         否,30 个里 17 个不同
聚合                   1/30 vs 1/30  (3.3% vs 3.3%)
```

### 根因:运动规划器,不是模型

生成是确定的(每次重播种),场景是确定的(`trial_seed`)。
剩下的随机源是 `--arm-action-mode planning` 用的**采样式运动规划器**,
它没有被我们播种,在边界位姿上成功与否是随机的。
这同时解释了 `steps` 和 `ik_fail_rate` 的逐条差异。

### 噪声量级

| n | 1 个标准差 |
|---|---|
| 10(per-task 单元格) | **±8 个百分点**(±0.79 次成功) |
| 100(聚合) | ±2.5 个百分点 |

### 四条后果(都要落到论文里)

1. **per-task 表的单元格 `2/10` 与 `1/10` 不可区分**,但 `6/10` 与 `0/10` 可区分。
   caption 必须说明这一点,否则读者会过度解读单个格子。
2. **只有聚合数字可引用**:14.0% ± 2.5。
3. **闭环的"配对"只到场景层面,不到轨迹层面。** 我之前说 ikctl 是"逐 trial 配对",
   要修正为"逐场景配对"。
4. **ikctl 的判读标准要放宽**:即使最终是 0/10 vs 0/10,
   在 ±8 点噪声下也只能说"未观察到把零变成非零的效应",
   不能说"中止规则毫无影响"。

### 与离线实验的对比正好说明问题

离线 probe 修好 seed 后是**逐位可复现**的(DA2 相隔一天两次运行最大差 0.00e+00,
见第三十七节)。闭环做不到,因为它有一条**经过物理仿真的反馈回路**。
两类实验必须用不同的统计口径 —— 这个对比本身值得写进方法学。

### 一处需要更正的早先表述

我在发现 `ik_fail 4.0% vs 0.0%` 时先归因于"扩散采样在不同 GPU 上的微小差异",
那是错的。是运动规划器。已更正。

---

## 四十二、IK 中止规则的对照:改变了过程,没改变结果(2026-09-02)

第三十九节提出的怀疑:`max_ik_fail_streak = 5` 是硬编码的 harness 策略,
在 12 任务上杀掉 var0 的 37% / var1 的 31%,且**集中在模型得零分的任务上** ——
一个只是"表现差"的 rollout 被提前终止,那个 0 就被读成"不会做"。

### 实验设计

全量 12 任务对照要 21 GPU 小时(放宽阈值后,原本被杀的 rollout 会跑到超时)。
改跑**中止率最高且成功数为零**的两个条件,变化会毫不含糊:

- `light_bulb_in` var0:8/10 被中止,0/10 成功
- `push_buttons` var1:8/10 被中止,0/10 成功

`--max-ik-fail-streak` 是本轮新加的开关(默认仍为 5,不改变历史行为)。

### 结果(light_bulb_in var0,同一批 10 个 scene_seed)

| | streak=5 | streak=50 |
|---|---|---|
| 成功 | **0/10** | **0/10** |
| 终止原因 | timeout 2,IK 中止 8 | timeout 10 |
| 步数中位 | 70 | **108** |

**规则确实在起作用**(8/10 被截断,放开后步数中位 70→108,全部跑到超时),
**但成功数不变**。

### 结论与措辞

`light_bulb_in` 的 0/10 是模型的真实失败,不是 harness 提前掐断的假象。

严谨表述:**未观察到该规则把成功压成零的效应**。
不能说"毫无影响" —— 它显然改变了终止原因和步数;
但对成功率这个我们真正关心的量,它是中性的。
考虑到第四十一节量化的 ±8 个百分点噪声下限,n=10 也只能支持这个强度的结论。

`push_buttons` var1 在跑,是更强的检验:该任务模型在 var0 上能做到 6/10,
所以如果中止规则在掩盖什么,最可能出现在那里。

### 对论文的影响

第三十九节里"1.7% 可能主要是 harness 造成的"这个担忧,
目前证据**不支持**。闭环数字可以按模型能力来解释,
但仍需在方法学里披露:
1. 中止阈值是一个未经调参的 harness 常数,它截断了三成 rollout;
2. 我们检验了它对成功率的影响,在两个最可能出问题的条件上没有发现效应。

### 四十二节补充:第二个条件跑完,合并结论(2026-09-02)

| 条件 | streak=5 | streak=50 | 步数中位 |
|---|---|---|---|
| `light_bulb_in` var0 | 0/10 | 0/10 | 70 → 108 |
| `push_buttons` var1 | 0/10 | **1/10** | 30 → 56 |
| **合计** | **0/20** | **1/20** | — |

**Fisher 精确检验 p = 1.000。**

两个条件方向不同(0→0 和 0→1),这正是跑第二个条件的价值:
只跑 `light_bulb_in` 会得出"规则无影响"的干净结论,
而 `push_buttons` 提示效应可能存在但很小。

**论文措辞(定稿)**:

> 中止阈值是一个未经调参的 harness 常数,在十二任务集上截断了约三成 rollout。
> 我们在中止率最高且成功数为零的两个条件上把它放宽十倍:rollout 的中位步长翻倍
> 并全部跑到步数上限,成功率从 0/20 变为 1/20(Fisher 精确检验 p = 1.00)。
> 该规则改变 rollout 如何终止,但我们没有观察到它改变结果。

**不写"该规则无影响"** —— n=20 只能排除大效应,排不掉"每条件 1 次成功"这个量级。
而该量级即便存在,12 任务聚合也只从 14.0% 动到约 15%,不影响任何论断。
这个界限一并写明,让读者自己判断够不够。

**不再加样本的理由**:提到 n=50/条件需要约 9 GPU 小时 × 2,
换来的分辨率提升不改变任何结论。

---

## 四十三、换训练 template 模式:arm7(2026-09-04)

### 动机

arm6 已经到平台(第 ... 节:6000→8000 训练/held-out 双平),继续同一套菜单买不到东西。
更根本的问题是**动作监督的预算**:arm6 的 60% 样本是 `video+depth` / `video+segmentation` /
`video+normal` 这三个感知模板,它们**一个 action 段都没有**。算上 10% 的 action dropout,
arm6 只有 `0.4 × 0.9 = 36%` 的步产生 action 梯度。

arm7 反过来:**每个模板都带 action**,变的只是「观测活在哪个视觉空间」。

| 模板 | 布局 | 比例 |
|---|---|---:|
| `video+action` | `V0 \| A0 \| V1 \| A1` | 0.4 |
| `depth+action` | `D0 \| A0 \| D1 \| A1` | 0.2 |
| `segmentation+action` | `S0 \| A0 \| S1 \| A1` | 0.2 |
| `normal+action` | `N0 \| A0 \| N1 \| A1` | 0.2 |

带 action 的样本从 36% 升到 **90%**,而**成本完全不变** —— 四个模板都还是四段同形状,
序列长度、显存、s/it 与 arm0/arm6 一样。

训练核心一行没改:`parse_template` / `primary_visual` / `getitem` / `forward` 本来就对 Π 泛型。
`segmentation+action` 和 `normal+action` 只是从来没被跑过而已。

### mask:新增 A 轴,arm7 用 A1

四个模板全含 action,所以全部走 `plan_segments` 的 `has_action` 分支,
`--perception_mask_mix`(M 轴)对 arm7 **一个字节都不起作用**。于是新增补集 `--action_mask_mix`:

| 预设 | (iiii, fiii, fifi, policy) |
|---|---|
| `A0`(默认) | `(0.81, 0.045, 0.045, 0.10)` = 上游,arm0–arm6 与 `step125750` |
| `A1`(**arm7**) | `(0.75, 0.00, 0.05, 0.20)` |
| `A2` | `(0.90, 0.05, 0.05, 0.00)` |

选 A1 的理由:`IIII` 下四段里两段是视觉段,**一半梯度花在想象未来的 depth/seg/normal 视频上**。
`FIII`(跨视角)在四种模式里与 action 质量关系最弱,把它的权重全给 `policy` ——
policy 模式的 loss **全部**落在 action 段,条件与闭环部署一致(每视角一帧观测),
而且 canvas 只有 24 而非 44 个 latent 帧,本身更便宜。
`IIII` 仍占 75%(部署就是它),`FIFI` 留 5% 作 v2a 的离线上界。

实现上,`plan_segments` 把这个边际**反推回上游那两次抽样的阈值**,不改成一次四路累计:
上游就是两次,第一次(collapse)对所有数据集都会消耗。默认 A0 下推出的三个阈值与原常量
**浮点逐位相等**,`test_forward_unchanged.py` 的上游等价 pin 一行没改就通过了。

### 归因上的取舍(明确记录)

对照组是 **arm0**(`outputs/specialist_action__seed42_fi3_512_aug_sr`,`video+action@1.0`,
同 warm start / seed / 树 / interval / lr),不是 arm6 —— arm0 有**同样的** 90% action 样本率,
只有一个模态;arm7 − arm6 会把菜单变化和 2.5× 的 action 预算变化混在一起。

但 **arm7 同时改了菜单和 mask 轴**(A1 vs arm0 的 A0),所以 arm7 − arm0 也不能单独归因于
其中任何一个。这是明知的取舍:这一轮的目标是把 action 做好,不是做干净的单变量消融。
真要归因,最便宜的办法是补一个 `ACTION_MASK_MIX=A1` 的 4000 步 action specialist(约 0.6 天/2 卡),
而不是重训 arm7。

### arm7 放弃了什么

菜单里没有任何 `video+X`,所以这个 checkpoint **不能**被问 RGB→depth/seg/normal。
`eval/eval_perception.py` 与 `eval/eval_all_masks.py` 对它是**离分布**的,论文的感知表没有 arm7 列。
它独有的新读数是「从哪个模态解动作」:`eval/eval_action.py --template <X>+action`
(本轮已参数化,四个模板的段切片相同,加了两模态守卫)。
闭环从非 RGB 观测 rollout **不在本轮范围**:`eval/rollout_env.py:187` 显式关掉了 depth/mask 渲染。

### 顺手修掉的两个静默 bug

1. `scripts/validate_mix.py::mask_mode` 的 `anchor` 硬编码成 `"video"`。对顶替模板过滤后是空集合,
   `all([])` 为 True,于是**每个** plan 都被读成 `fifi` —— 预检会 PASS,同时描述一个不存在的分布。
2. 本机 `/usr/bin/grep` 是 **ugrep 7.5.0**,它的 `-q -v` 即使存在不匹配行也返回 1。
   train_arm.sh 里用来判断哪条 mask 轴 INERT 的一行因此会把 arm6 的混合菜单误报成
   「所有模板都含 action」。已改成纯 bash 计数。
   **同样的坑还在 `scripts/supervise_8h.sh:25`(`pgrep -f "$1" | grep -qv "^$$$"`),本次未动** ——
   那行的意思是「除我之外还有没有同名进程在跑」,在 ugrep 下恒为「没有」。

### 起训命令

```bash
GPUS=<两张空卡> bash scripts/train_arm.sh arm7
# -> outputs/arm7_seed42_fi3_512_aug_sr,warm start step125750,10000 步
```

起训后确认日志里这几行:`[selfgen] variations='0': kept N/M`、
`[selfgen] scene_roles OK: zero unknown pixels`、三条 `self-test OK`(三个顶替模板各一条)、
以及 `mask axes: perception_mask_mix=M2 (INERT: all 4 templates contain <action>) action_mask_mix=A1`。

**不要用 s/it 判断 A 轴是否生效**(我一开始就是这么说的,错了)。policy 模式把 canvas 从 44 帧
压到 24 帧,但 `forward` 是先对每个 stream 整段 `encode_video`(8 次 41 帧编码),**之后**才调
`plan_segments`,`assemble` 只切已经算好的 latent —— 所以短 canvas 只省 DiT,不省 VAE 编码,
净收益在噪声以内(实测 arm7 20.8 s/it,arm6 ~21 s/it)。
要确认就看起训 banner 的 `action_mask_mix=A1` 和 wandb config 的 `action_mask_mix_resolved`。

---

## 四十四、arm7 评测方案定稿(2026-09-04)

完整方案写在 **`EVAL_PLAN_ARM7.md`**(与 `EVAL_PLAN.md` / `EVAL_PLAN_CLOSEDLOOP.md` 同一族)。
这里只记要点和两个会影响论文措辞的结论。

### 结构性事实:arm7 一次只吃一个模态的锚定帧

`draw_template`(rlbench_selfgen.py:605)是分类抽取,`_batch_template` 拒绝混模板 batch,
推理只要求该模板用到的 stream。所以 arm7 是**四个可互换的输入适配器共用一个 action 解码器**,
不是四路融合模型。四个模态从不同时出现在一个序列里。

真要同时给四个锚定帧,需要 `video+depth+segmentation+normal+action`(10 段、序列 2.5×),
arm7 训练时 **0% 见过**,只能当离分布探针,不能当 headline。

### 门槛是 46% 不是 35%

main5 协议(5 任务 × 20,var0)三个基线都已跑完:

| 模型 | ckpt | 成功率 |
|---|---|---:|
| **specialist_action** | step4000 | **46.0%** |
| arm6 | step10000 | 35.0% |
| Initialization | step125750 | 24.0% |

**action specialist 已经赢了 headline 臂**,而且只训了 4000 步。它不只是 ablation,
是当前最强闭环模型。arm7 的对比点应当是它,而且要在 **step 4000** 比——两者 action 样本率
都是约 90%,同一步数下 action 更新次数相同。

**Initialization 的评测不需要重跑**:main5 / unseen4 / var1 / 离线全都有。

### 三层评测

1. **Tier 1 逐模态读数**——四个适配器各读一次(通用性)。要改
   `heldout_batch_eval.py:64` 的 `MODALITY_TEMPLATE`(现在写死 `video+X`),加新 key、旧 key 不动。
2. **Tier 2 模态集成**——`fuse_multiview_heatmaps_to_pose_torch` 沿 V 轴融合,**V 不必是相机**。
   四个模板的 action 段是同一条轨迹经同一对相机投出来的,4 模态 × 2 视角 = 8 组堆成 V=8
   解出一个位姿,**解码器一行不改**。另外三路不是废的,是免费的集成。附带 V=2/4/6/8 剂量响应。
   *局限*:需要 GT 锚定帧,是离线论断,不是可部署策略。
3. **Tier 3 闭环**(RGB only)——step 4000 + step 10000,main5 协议,args 与基线逐字节一致。

### 最重要的一条验收:负对照

Tier-1 打 `specialist_action/checkpoint-4000`,它只训过 `video+action`,所以三行非 RGB
**必须**退化(低 r_peak)。如果看着正常,说明评测读的不是适配器,后面所有数字都不能信。

### 决定不做的

- **不改仿真器**(本轮)。闭环只跑 RGB。depth/normal 其实很便宜
  (`gen_dataset.py:210` 已有可用配置,内参 `rollout_env.py:322` 已在读),seg 最难且错了不报错。
- **三个 100% `X+action` specialist 不做**。它们菜单里没有 `video+action`,而仿真器只渲染 RGB,
  所以**跑不了闭环**,headline 那格是空的。若要 ablation,改成固定 action 预算下扫观测模态
  **数量**(1 = specialist_action / 2 / 4 = arm7),每个臂都保有闭环能力。

### 归因取舍(已记录)

arm7 与 arm0 差**两处**:template menu 和 mask 轴(A1 vs A0),所以 arm7 − arm0 不能单独归因。
本轮目标是把 action 做好,不是干净的单变量消融。要归因就补一个 `ACTION_MASK_MIX=A1` 的
4000 步 action specialist(约 0.6 天/2 卡),而不是重训 arm7。

### 四十四节更正:Tier 2 的「堆 V 轴」是错的(2026-09-04)

上一节把模态集成写成「四路 heatmap 堆到 V 轴上解成一个位姿,解码器一行不改」。**那是错的。**

读 `fuse_multiview_heatmaps_to_3d_point_torch`(utils.py:793)才发现它**不是最小二乘三角化**:
取 **view 0 的射线**作 principal ray 沿它一维搜索(utils.py:896-897),其余视角只是打分,
而且第 5 步是**连乘** `sampled_scores.prod(dim=1)`(utils.py:983)——是合取(AND),不是平均。

合成实验(2 相机 + 已知 3D 点):

| 实验 | 位置误差 |
|---|---|
| V=2 基线 | 2.2 mm |
| 精确复制成 V=4 | **2.2 mm,完全不变** —— 几何零增益 |
| 仅调换拼接顺序 | 2.2 → **7.6 mm** —— principal ray 锚在 view 0 |
| 四路里一路偏 20px | 2.2 → **84.4 mm** —— prod 一路坏就毁掉 |

3 好 1 坏时的方案对比:单 RGB **3.0 mm** / V=8 直接堆 37.8 mm / 各自解码取均值 51.3 mm /
**各自解码取中位数 4.3 mm** / 先平均 heatmap 再解 36.4 mm。

**正确做法**:四路各自按 V=2 解码,在**位姿层面取中位数**(位置逐分量中位数,旋转弦中位数)。
`fuse_multiview_heatmaps_to_pose_torch` 仍然一行不改,只是调用四次。

**更重要的措辞更正**:即便中位数(4.3 mm)在这个玩具设定里也**没有赢过**单 RGB(3.0 mm)。
集成能否真的超过单 RGB,取决于四路误差是否独立;四个适配器共享同一 backbone,误差很可能相关。
所以 Tier 2 的产出必须是「集成有没有用」这个**测量结果**,不能预设「四路更好」。
若实测集成不优于单 RGB,那本身就是干净的结论:适配器共享表征,误差不独立。

`EVAL_PLAN_ARM7.md` §3 Tier 2 与 §4 已同步改正。

---

## 四十五、arm7@4000 全面离线评测:通用性成立,RGB 不如专家(2026-09-05)

`reports/heldout_batch_eval/{arm7_4000, negctl_actionspec_4000, negctl_actionspec_4000_unseen}`
共 128 个 action 格,零报错。全部用**配对**比较(同一批 episode)+ 均值 + 双尾符号检验。

### 结论一:四个适配器等价 —— 通用性主张成立

| 锚定模态 | arm7 均值 | arm7/RGB | p | spec 均值 | spec/RGB | p |
|---|---:|---:|---:|---:|---:|---:|
| RGB | 0.1865 | — | — | 0.1429 | — | — |
| depth | 0.1835 | **0.98x** | 1.000 | 0.3003 | **2.10x** | **0.001** |
| seg | 0.1919 | 1.03x | 0.454 | 0.2163 | 1.51x | 0.077 |
| normal | 0.1773 | 0.95x | 0.454 | 0.2248 | 1.57x | 0.077 |

负对照(只训过 `video+action` 的 specialist)在 depth 上退化 **2.10x, p=0.001**;arm7 在同样的
比较下是 **0.95–1.03x, p≥0.45**。**仪器确实能检出没训过的适配器,而 arm7 没有退化。**
一个 checkpoint 从四种观测空间行动,质量统计上不可区分。

### 结论二:arm7 的 RGB **不如** specialist_action —— A 主张在 4000 步不成立

配对 n=16:arm7 均值 **0.1865** vs spec **0.1429** = **1.31x 差**,arm7 只在 4/16 个 episode
上更好,双尾符号检验 **p=0.077**。unseen 上差距更大(0.2296 vs 0.1661)。

**「多模态观测训练让 RGB 策略更强」目前没有证据支持。** 注意一个不对称:arm7@4000 是
10000 步跑到一半的 checkpoint,specialist_action@4000 是**跑完**的 —— 但「action 更新次数匹配」
的论证仍然成立,所以这个不对称不能把差距解释掉。

### 结论三:白捡的——arm7 生成的 depth 质量(此前无法测量)

扩展 `heldout_batch_eval` 时顺带打开了顶替模板里 anchor 段的评分:

| | arm7 | spec |
|---|---:|---:|
| depth AbsRel (低好) | **0.1315** | 0.2267 |
| normal mean_cos (高好) | **0.8805** | 0.3699 |
| seg macro_mIoU (高好) | 0.4032 | 0.4073 |

arm7 在 `depth+action` 里生成的 depth 是 **AbsRel 0.1315**。此前我错误引用 arm6 的 0.1441
(不同模板 + 不同 conditioning,不可比);现在这是直接测量值。
seg 的 mIoU 两者几乎相同,值得单独查一次是不是被 background 主导了。

### 两个方法论错误(已更正,影响此前所有读数)

1. **中位数在 n=8 重尾下是错的统计量。** 负对照里 `action_from_seg` 中位 0.0994 看起来比 RGB
   (0.1166)还好,像是仪器坏了;但逐 episode 是 0.4273/0.0491/0.4238/0.0397…,**均值 0.1768**
   才是实情。此后一律配对 + 均值 + 符号检验,不用裸中位数。
2. **存档基线 `actionspec_4000_matched` 的 0.1405 与本轮不可比。** 它跑的是 **80 个** episode
   (每任务 5 个 x 2 variation),脚本默认只跑 8 个。同一 checkpoint 同一指标:
   0.1405(n=80) vs 0.1166(n=8),**20% 的差异纯粹来自 episode 集合**。
   任何跨 run 的对比必须先核对 episode 集合。

### 修掉的两个 bug

- `heldout_batch_eval.py` 评 anchor 段时**无条件找 `video` 段**(注释称「每个模板第一段都是
  video」),对顶替模板是假的,`next()` 会 StopIteration。改为从 plan 取 anchor,顺带产出结论三。
- `heldout_batch_eval.py` 的 **`--gpu` 一直失效**:它在第 190 行才设 `CUDA_VISIBLE_DEVICES`,
  而 import 该模块时 CUDA 已初始化(`torch.cuda.is_initialized()==True`),之后该变量不生效,
  每次都落到物理 GPU 0。两个并发评测因此抢同一张卡,一个 OOM 而日志却打印 "using GPU 7"。
  现改为检测到该情形直接**拒绝启动**并给出正确用法
  (`CUDA_VISIBLE_DEVICES=<n> python ... --gpu 0`)。**此前每一次单作业运行都在悄悄用 GPU 0。**

### 对下一步的影响

按 `EVAL_PLAN_ARM7.md` §5 的门控:离线闸门显示 arm7 的 RGB 更差,**不应该现在花 11.3 GPU-小时
去跑 arm7@4000 的闭环**。等 10000 步跑完重测闸门再决定。
B 故事线(一个 checkpoint 四种观测空间)已经有干净证据,应作为主线推进。

### 四十五节更正:结论二的对照口径错了(2026-09-05)

上一节的结论二「arm7 的 RGB 比 specialist_action 差 1.31x」**用错了匹配量,该结论作废**。

我当时的理由是「两者都是 4000 步,且 action 样本率都约 90%,所以 action 更新次数匹配」。
那匹配的是**总 action 曝光**(arm7 四个模板都带 action)。但要回答「另外三个模态有没有帮到
RGB 这条路」,必须按 **`video+action` 曝光**匹配:

```
匹配条件  S x 0.4 = N      (10% action dropout 两边相同,约掉)
  arm7@4000  <-> spec@1600   (~ckpt-1500)
  arm7@6000  <-> spec@2400   (~ckpt-2500)
  arm7@10000 <-> spec@4000   <- 只有到终点,spec@4000 才是正确对照
```

也就是说 arm7@4000 的 RGB 曝光只有 spec@4000 的 **40%**。在这个前提下测出的 1.31x 差距,
至少有一部分(可能全部)只是训练量少了 2.5 倍,不能解释成「多模态训练伤害了 RGB」。

**spec@4000 是 arm7@10000 的对照,不是 arm7@4000 的。** 之前那个「都是 4000 步所以公平」的说法
听起来顺、其实错。

已从 HF(`TingtingDu/specialist_action__seed42_fi3_512_aug_sr`,500-4000 每 500 步都有;
本地只剩 4000,其余早被剪掉)拉回 ckpt-1500 与 ckpt-2500,跑同一套 eval 的 action 一列,
构建按 RGB 曝光对齐的曲线:

| RGB 曝光 | arm7 | spec |
|---|---|---|
| ~1600 | @4000 | @1500 |
| ~2400 | @6000 | @2500 |
| ~4000 | @10000 | @4000(已有) |

**结论一(四条输入路径等价)不受影响** —— 那是 arm7 内部四路互比,与 spec 的曝光无关;
负对照(spec 在没训过的模板上 depth 退化 2.10x, p=0.001)也不受影响。
受影响的只有结论二。

### 四十五节续:对齐后的结论二反转,但结论一在 6000 步走弱(2026-09-05)

完整结果与分析已整理进 **`ARM7_RESULTS.md`**(10 节,含逐 episode 原始数据、
四个方法论错误、两个代码 bug、复现命令)。这里只记两个新结果。

**结论二反转(好消息)** —— 按 `video+action` 曝光对齐后:

| RGB 曝光 | 对比 | arm7 | spec | 比值 | arm7 胜场 | p |
|---:|---|---:|---:|---:|---:|---:|
| ~1600 | arm7@4000 vs spec@1500 | 0.1865 | 0.1726 | 1.08x | 10/16 | 0.454 |
| ~2400 | arm7@6000 vs spec@2500 | 0.1461 | 0.1512 | **0.97x** | 9/16 | 0.804 |
| — | ~~arm7@4000 vs spec@4000~~ | 0.1865 | 0.1429 | ~~1.31x~~ | ~~4/16~~ | ~~0.077~~ |

「arm7 的 RGB 更差」**不成立**。两点都分不出差别,第二点已越到 arm7 这侧。
趋势 1.08→0.97 有利,但 n=16 分辨不了这个量级,不能当结论。

**结论一在 6000 步走弱(需要注意)** —— 四条路径互比:

| 问法 | @4000 | @6000 |
|---|---|---|
| depth | 0.98x (8/16, p=1.000) | **1.36x** (5/16, p=0.210) |
| seg | 1.03x (6/16, p=0.454) | 1.09x (7/16, p=0.804) |
| normal | 0.95x (10/16, p=0.454) | **1.32x** (6/16, p=0.454) |

看绝对误差才知道是谁在动 —— **不是 RGB 涨得快,是 depth/normal 真的变差了**:
RGB 0.1865→0.1461(-21.7%)、seg 0.1919→0.1590(-17.1%),而
depth 0.1835→0.1986(**+8.3%**)、normal 0.1773→0.1926(**+8.6%**)。

「RGB 占 0.40、其余各 0.20,更新次数两倍」能解释涨得慢,**解释不了绝对变差**。
限度:n=16、单 seed、两个 checkpoint,三个 p 值没有一个显著;本仓库有非单调先例
(arm4 闭环 5000 步 20% → 7500 步 10%)。现在只能说「4000 成立、6000 不清楚」,
**step 10000 是判据**。

**对 `CROSSMODAL_FUSION_PLAN.md` 的影响**:该文 Phase A 的门控是
「特权路径明显强于 RGB 才值得蒸馏」。目前四条路等价(甚至 depth/normal 更弱),
**没有可蒸馏的差距** —— arm8 的前提不成立,需要重新设计。

---

## 四十六、新方向:全模态拼接臂 `fusion` —— Stage 0 代码 + 冒烟实测(2026-09-08)

> 用户提的新方向:**从 WAN 模型开始训练,用全部模态拼接**,
> `v0 d0 s0 n0 a0 v1 d1 s1 n1 a1` 的 latent 拼接,让不同模态互相监督/互相补全。
> 本节记录可行性核算、必须写的那一处代码、以及冒烟实测把三条估计推翻的结果。
> 完整计划另存于 `/root/.claude/plans/repo-...-snuggly-hennessy.md`。

### 46.1 为什么这是一个真的新东西,而不是 arm6/arm7 的换皮

arm0–arm7u **全是 4 段**。`train.py:_batch_template` 拒绝一个 batch 里出现多个模板,
每个样本又只抽一个模板,所以**两个模态在一个训练步里从不共存** ——
arm6/arm7 学到的是「一套权重轮流做四件事」,不是「模态之间互相监督」。
`CROSSMODAL_FUSION_PLAN.md` §4.1 早就把这条列为 arm8 的硬约束,并把「共生成模板」
列为方案 B(真融合),标注「留作后续」。这个臂就是方案 B 推到 5 个模态。

**layout 本身零代码。** `plan_segments` 的布局循环是 view-major / modality-minor,
`CANONICAL_ORDER = (video, depth, segmentation, normal, action)`,所以
`--template_mix "video+depth+segmentation+normal+action@1.0"` 当场就产出
`V0 D0 S0 N0 A0 | V1 D1 S1 N1 A1`,10 段 / **f=110 latent 帧** / 28,160 token。

### 46.2 唯一非做不可的改动:F 轴(模态丢弃)

`assemble` 无条件给每个非 `fully_given` 的段留首帧当条件(`templates.py`)。
10 段就是**10 个免费 anchor**。而第三十节的 tag-swap 消融已经量过这意味着什么:
**换 prompt tag 指标只变 1–2%,拿掉 anchor(f0f0)崩 ~9 倍** —— 选模态的是 anchor 不是 tag。

所以保持现状会同时坏两件事:
1. 「互相补全」退化成平凡的复制任务,**结论不可证伪**;
2. 部署时现场只有 RGB,拿不到 depth/seg/normal 的首帧,模型直接掉进那个 9 倍崩塌区。

实现:`Segment` 加第三种角色 `absent`(整段没有任何条件帧),默认 `False` 所以
`test_forward_unchanged.py` 与全部现有臂逐位不变。新预设 `--fusion_mask_mix`:

| regime | p (F1) | GIVEN | ANCHORED | ABSENT |
|---|---|---|---|---|
| `full_anchor` | 0.30 | – | 全部 5 | – |
| `rgb_only` | 0.30 | – | video | d,s,n,a ← **部署 regime** |
| `rgb_given` | 0.15 | video | – | d,s,n,a |
| `one_out` | 0.15 | – | 随机 4 | 随机 1(**含 video/action**)← 互补全探针 |
| `policy` | 0.10 | – | video(单帧) | d,s,n(单帧);action 全长 |

角色**按模态定、两视角一致** —— 否则 view0 的 depth 给 view1 的 depth 当锚点,absent 是假的。
另加 `F0`(全 anchor,即改动前行为,用作对照臂)与 `F2`(去掉 full_anchor)。

配套加了按段 loss:`segloss/<模态>/<anchored|absent>`、`segloss_pos/<k>`、`regime/<名字>`,
**全部按步内 loss 归一**(timestep 一步一抽、跨三个数量级,不归一就是在测抽签而不是测模态)。
`depth/absent` 对 `depth/anchored` 就是互补全的直接读数;`segloss_pos/<k>` 是 §46.4 的 RoPE 探针。

### 46.3 冒烟实测:三条估计被推翻(3×RTX6000Ada,20 步,F1)

刻意用 **3 卡不是 4 卡**:ZeRO-2 按 rank 分片梯度,3 卡峰值是 4 卡的**上界**。

| 量 | 事前估计 | 实测(3 卡) |
|---|---|---|
| 显存·稳态 | — | 38.9–39.7 GB(150 次采样里的 149 次) |
| 显存·**观测峰值** | 2 卡 OOM,4 卡刚好够 | **44.2 GB / 48 GB** ← 估计基本正确 |
| 稳态 s/it | ~65 | **36**(第 5/6/7 步 37/36/36 s) |
| action 占预测位置 | 0.20(2.5× 稀释) | **0.263**(1.9× 稀释) |

1. **显存:原估计站得住,4 卡是必要的。** 稳态只有 ~39.7 GB,但 150 次采样里有 1 次打到
   **44.2 GB**,只剩 3.8 GB。按梯度分片 20 GB / rank 数外推峰值:**2 卡 ≈ 47.5 GB 会 OOM**;
   4 卡 ≈ 42.5 GB。**所以 4 卡不只是曝光决策,也是显存前提。**

   > ⚠️ **方法论教训,值得记住**:`nvidia-smi` 每 4 s 采一次会漏掉瞬时峰值 ——
   > 我一度只看到 39.7 GB 就下了「显存宽松、2 卡也够」的结论,差 4.5 GB,而且方向是错的那一边。
   > 已在 `train.py` 的诊断里加 `mem/peak_gb`(`torch.cuda.max_memory_allocated`,
   > 分配器自己记的高水位,不靠采样,每步 reset)。**正式跑盯这个,不要盯 nvidia-smi。**
2. **不要装 flash-attn。** `torch.ops.aten._fused_sdp_choice` 对 `[1,24,28160,128] bf16`
   返回 backend **1 = FLASH_ATTENTION** —— PyTorch 2.6 的 SDPA **本来就在走 FlashAttention**
   (54.9 ms;强制 mem-efficient 才 88.1 ms)。外部包零收益,还会在 arm7u 万一重启时换掉它的后端。
3. **action 稀释是 1.9× 不是 2.5×**:`policy` regime 把 77% 的 loss 压在 action 段上,拉回了整体。

由此 Stage 1:**5000 步 × 36 s ≈ 50 h ≈ 2.1 天**(4 卡,batch 4)。

### 46.4 两个已知风险(写下来,别忘)

- **RoPE 外推**:时间位置跨段单调递增,`a1` 落在 t=99–109。Wan2.2-TI2V-5B 预训练最长
  ~31 latent 帧,step125750 只见过 44。**88–109 是权重从没见过的频率区间。**
  Stage 1 不动 RoPE(零改动、warm-start 兼容),靠 `segloss_pos/<k>` 诊断:
  若 8/9 段系统性差于 0/1 段,就是外推在伤人。要修得等 Stage 3 冷启动
  (每段 RoPE 重置 + learned modality embedding,破坏 warm-start 兼容性)。
- **normal 冗余**:`encode_normal_from_depth(depth, fx, fy)` 证明 normal 相对 depth 零新增信息。
  用户选择保留 —— 那就把它当**干净的互补全探针**:`one_out` 把 normal 设 ABSENT 时,
  模型能否从 depth 精确重建?标准答案是确定性的,这是最硬的一个读数。

### 46.5 交付物与验证

新增/改动:`training/templates.py`(F 轴 + `segment_spans`/`describe_plan`/`fusion_regime_of`)、
`training/args.py`(`--fusion_mask_mix`,**启动时**校验)、`train.py`(按段 loss)、
`training/wan_video_action_images.py`(`absent_modalities` + 5 个 regime 名做 conditioning mode;
旧 `f0f0` 的事后清 mask 改成在 plan 里声明 absent,等价且保留别名)、
**新文件** `scripts/train_fusion.sh`、`scripts/stop_and_claim.sh`、`tests/test_fusion_axis.py`。

> `train_arm.sh` **一个字没改**:bash 按字节偏移边读边执行,它正被 arm7u 的父 shell 持有,
> 改它会重演 2026-09-06 那次「执行注释碎片并重启作业」。Python 文件无此问题。

验证:`test_forward_unchanged` 等 7 个受影响测试全过;`debug_cogen.py` → `COGEN_DEBUG_OK`
(fusion 模板已加入其 `TEMPLATES` 覆盖面);`tests/test_fusion_axis.py` 7/7,
**其中最要紧的一条是训练/推理 mask 逐位一致** —— 5 个 regime 在训练侧 `plan_segments` 与
推理侧 `prepare_template_inference_latents(conditioning_mode=<同名>)` 产生完全相同的 mask。
没有这条,checkpoint 会被拿一个它没训过的 conditioning 去评测而没人发现,
跟当年 prompt tag 洗 `_` 造成的 train/eval 割裂是同一类事故。

### 46.6 下一步

1. **等 arm7u 跑完**(现 82%,约 9.5 h)—— 它是 §43/§45 里排在所有机制之前的对照,
   已烧 46 h,而且正是 `fusion0` 的对照组(同 warm start/树/seed/四模态/A1,
   唯一差别是「一样本一模态」vs「一样本全模态」)。
2. `bash scripts/stop_and_claim.sh --yes --claim '...'` 按序停机并抢 4 卡。
3. `fusion0` 5000 步(曝光对齐 arm7u@10000),随后 `fusion0-anchor` 2500 步做 F 轴对照。
4. **并行**开渲扩容树:先加 episode 不加任务(现有 16 个任务 ×5,零 `TASK_ROLES` 工作量),
   `LP_NUM_THREADS=4`,写进**新目录**(往 `rlbench_selfgen_512_aug` 里加会静默改掉
   arm0–arm7u 的训练集,摧毁全部可比性)。
5. 判据(通过才值得做 Stage 3 从 Wan base 冷启动):部署 regime 感知质量优于 arm6/arm7 的
   `iiii` 读数;held-out loss 不退;**互补全矩阵单调** —— 固定一个模态 ABSENT,
   质量随共 present 模态数上升。若是平的,说明四个模态只是共用 decoder,
   与 arm8 同一种死法,停在 Stage 1。

### 46.7 fusion0 开跑后的三个读数(2026-09-08)

`fusion0`(F1,4×RTX6000Ada GPU 2/4/5/6,batch 4,5000 步)与 wide 树渲染并行启动。
arm7u **没有停** —— 空出来的正好是另外 4 张卡,所以它跑完就是 fusion0 干净的对照组。

**(a) 4 卡显存:安全,且比 3 卡的读数可信得多。**
2400 个 **1 秒**样本 × 4 卡(40 分钟 / 约 65 步,五个 regime 各命中多次):
**峰值 38.8 GB / 48 GB,超过 43 GB 的样本 0 个。**
对照 §46.3 里 3 卡那次 4 秒采样的 44.2 GB —— 那个尖峰只出现一次,现在看多半是模型加载/
优化器初始化的瞬态,不是训练稳态峰值。**教训依旧成立:4 秒采样不足以给显存下结论。**

**(b) F 轴在生产里确认生效,不用读 wandb 就看得出来 —— 步时是双峰的。**
```
37 37 36 37 37 37 [21] 37 38 37     ← 21 s 的是 policy regime:画布 30 帧而非 110 帧
38 37 37 38 37 37 [21] 37 37 38
37 38 38 37 38 37 37 38 [20] 38
n=49  min=20  median=37  max=38
```
3–4/49 ≈ **7–8% 的快步**,F1 声明的 `policy` 是 10%,在抽样噪声内。

**(c) ⭐ 由 (b) 反解出:固定开销 ≈ 16.5 s/step,占正常步的 45%。**
policy 画布只有 110 帧的 27%,步时却是 54%(20 s vs 37 s)。按 DiT 的
`dense + attention·S` 缩放解 `F + D₃₀ = 20`、`F + D₁₁₀ = 37`(D₁₁₀/D₃₀ = 5.86):
**F ≈ 16.5 s、D₁₁₀ ≈ 20.5 s、D₃₀ ≈ 3.5 s。**

这个 16.5 s 就是 **14 次 VAE encode + CPU 侧 Adam + allreduce**,正好印证
`MODE_MIX_MASKS.md` 早就写下的那句「policy 模式只缩短 DiT —— `forward` 在
`plan_segments` 之前就把每条流按全长编码了」。

**含义:预计算 latent cache 能省掉将近一半的步时**,是这条线上最大的一块未开采的优化
(仓库里从来没有过 latent cache,只有 episode 索引 cache)。
**但现在不做** —— fusion0 已在跑,中途改会毁掉可比性。**记给 Stage 3。**
实现的拦路石在 `training/helpers/io.py`:窗口 `frame_indices` 是每次 `__getitem__` 随机抽的,
所以 cache 要么把窗口钉死,要么以窗口为 key。

**(d) 渲染 ETA 修正:~12 h,不是脚本估的 7.9 h。** 117 集 / 20 分钟 = 5.85 集/分,
剩余 4203 集 ≈ 12 h。脚本的 `EPS*211/NPAR` 没算每个 job ~50 s 的 CoppeliaSim 启动开销,
而 SHARD=5 意味着 832 个 job 都要付这笔钱。渲染没有拖慢 fusion0(负载 117/256,s/it 仍 37)。

### 46.8 数据扩容完成:wide 树 5.0×,以及它立刻暴露的一个 bug(2026-09-09)

`rlbench_selfgen_512_aug_wide`:同样 16 个任务,variation0 每任务 50 → 250 seed。
**variation0 788 → 3960 集(5.0×)**,held-out 240,137 G,约 12 h(32 worker,`LP_NUM_THREADS=4`)。
任务集刻意不变 —— `TASK_ROLES` 只覆盖 22 个任务,加 seed 免费,加任务要先写规则。

**⭐ 扩容本身抓出一个小树上永远发现不了的 bug。** 第一次生成 `scene_segments.json` 结尾是
`COVERAGE INCOMPLETE`:`Roof`(房间天花板)不在 `BACKGROUND_NAMES` 里。

```
                   原树(70集/任务)   wide树(270集/任务)
Roof 出现:              0/70              3/270
```

原树从没撞上;wide 树 5× 的 Colosseum 相机随机化后有 **3 集**的某个相机抬头看过了墙顶。
然后 **handle-union 把 3 集放大成那两个任务全部 540 集 unmapped** ——
`_check_scene_roles_ready` 见一个 `unknown` 像素就抛异常,整棵树会被训练拒绝。

修:`"Roof"` 加进 `BACKGROUND_NAMES`。**可比性检查做了两道**:grep 两棵树所有 `handles.json`
(原树 Roof 出现 0 次),以及用改后代码全量 dry-run 原树 1028 集(`UNMAPPED: 0`)。
所以 arm6/arm7/arm7u 的目标逐位不变,它们本来也没这个问题。

**验收**(用仓库自己的 codec,不重新实现):结构层 4200 集全过(4 视角/文件齐全);
解码层 99 集 396 个 (集,视角):depth AbsRel 中位 0.0007、normal cos 1.0000、
depth 全落 [0.05,10]m、四元数单位范数、**`unknown` 像素 0**。
**并用同一脚本跑原树做对照,每个模态数字两棵树一致** —— 除规模无其它差异。
最后是真门:`scene_roles OK: zero unknown pixels`、`seg coverage 3960/3960`、`len(ds)=3960`。

### 46.9 ⭐ fusion0@2000 离线评测:锚帧依赖尚未打破

四个 regime 各跑一次(n=1,单集 `open_drawer/variation0/episode0`,**不能当结论,是方向指示**):

| regime | video PSNR | depth AbsRel | **decode_coverage** | normal cos | seg unknown_px |
|---|---|---|---|---|---|
| `full_anchor` | 15.38 | **0.1212** | **0.9995** | **0.832** | 35119 |
| `rgb_only` | 14.83 | 0.395 | **0.409** | **−0.286** | 4218 |
| `rgb_given` | 30.71(给定) | 1.202 | **0.043** | **−0.345** | 10371 |
| `policy` | 32.30 | 0.844 | **0.9992** | 0.718 | **0** |

**`decode_coverage` 才是这组数的主角。** `rgb_given` 那个 AbsRel=1.202 是只在 4.3% 像素上
算出来的 —— 其余 95.7% 掉出 Barron 立方体路径、解码成 NaN 被剔除。所以不是「给的信息多反而更差」
这种悖论,而是**无锚时模型根本没产出合法的 depth 编码**。三条独立证据一致:
coverage 99.95% → 4–41%、normal cos 变**负数**(比随机 ≈0 还差)、有锚时 depth 0.1212
已优于 arm7@10000 的 0.1315。

**结论:第 2000 步时锚帧仍承载几乎全部「我是哪个模态」的信号,F 轴要打破的正是这个,还没打破。**

**一个反直觉亮点**:`policy` 的 depth 也 absent,但 coverage 99.92%、normal cos 0.718、
`unknown_px` 为 0。差别在画布:policy 只 30 帧且 depth 帧紧贴 video 帧,而 `rgb_only` 要在
位置 11–21 凭空生成 11 帧。**短跨度无锚可以,长跨度不行** —— 这要么是训练不足,
要么是 §46.4 标记的 RoPE 外推(位置 88–109 是权重没见过的频率区间)。`segloss_pos/<k>` 正为此而加。

### 46.10 两个「布局被抄写两遍」的 bug,都是静默的

F 轴加了第三种角色 `absent` 之后,两处**手抄**画布布局的代码失效了,而且都不会崩:

1. **`scripts/modality_mode_grid.py:seg_plan`** —— 它的 docstring 写着
   "Mirrors prepare_template_inference_latents exactly"。抄本不知道 `absent`,
   后果是**把一个模态的指标报成另一个模态的名字**。
2. **`eval/policy.py:decode`** —— 硬编码「4 个等长段」并切 `[q:2q]`/`[3q:4q]` 当动作热图。
   在 10 段画布上实测这两个切片是 **segmentation view0/view1**:
   ```
   video  (4段):  action spans [41:82] [123:164]    ← 与旧硬编码逐位相同,无回归
   fusion (10段): action spans [164:205] [369:410]
                  旧代码 [q:2q]/[3q:4q] → segmentation view0/view1
   ```
   **旧代码会从分割图里解出一个位姿当策略输出报出来**,而且不会报错 —— 两者都是 RGB。

修法不是再同步一遍抄本,而是把「mode → 角色」抽成 **`templates.inference_plan` 单一来源**,
管道和两个评测脚本都调它,漂移在构造上不可能再发生。
`tests/test_fusion_axis.py` 的训练/推理 mask 逐位一致那条现在测的就是这个共享函数,约束更强。

**教训**:`assemble` 的布局是这套代码里被最多地方依赖的隐式契约。任何「Mirrors X exactly」
的注释都是一个等着失效的抄本 —— 加角色/加模态时必须 grep 所有复述过布局的地方。

### 46.11 两臂现状与一个曝光不对等(2026-09-09)

| 臂 | 数据树 | 步数 | 状态 |
|---|---|---|---|
| `fusion0` | 512_aug(788) | **3202/5000,已停** | checkpoint-3000 可用 |
| `fusion0-wide` | 512_aug_wide(3960) | **5000/5000 `TRAIN_EXIT=0`** | 5 个 checkpoint 全在 |

旧树臂按用户要求提前停掉以腾卡,所以两臂**曝光不对等**(12808 vs 20000 样本)。
做「5× 数据有没有用」的对比只能 **checkpoint-3000 对 checkpoint-3000**,
或者把旧树臂补到 5000 —— **不要拿 wide@5000 比旧树@3000**,
那正是 `ARM7_RESULTS.md` 把 1.31× 读反的那类错误。

**操作教训**:停掉旧树臂释放的 4 张卡在我去改代码的几分钟里被外部租户全部抢走。
`scripts/stop_and_claim.sh` 存在的唯一理由就是「确认空闲后下一条命令必须就是启动」,
而那次没用它。新增 `scripts/wait_and_eval_all.sh`:一次拿卡、离线 grid 与闭环两段连跑、
中间不释放,且 checkpoint 在启动时钉死(否则还在训的臂会让两段报告的是不同模型)。

### 46.12 ⭐ 从训练日志里挖出的结果:锚帧依赖正在被打破(2026-09-13)

八张卡全被外部租户占着、评测排不上队,于是改去挖**已有的训练日志**。
来源:`wandb/run-20260909_034125-11gts31i/*.wandb`,用 `wandb.sdk.internal.datastore` 解事务日志
(坑:键在 `nested_key` 而不是 `key`)。**不需要 GPU,而且是数千步的平均,功效远高于 §46.9 那个 n=1。**

**它推翻了 §46.9 的结论。**

`absent / anchored` 损失比,前 1000 步 → 后 1000 步(每格 n≈2000–3000):

| 模态 | 早 | 晚 | Δ | |
|---|---|---|---|---|
| depth | 1.776 | 1.266 | **−0.510** | ↓ 在学 |
| action | 1.374 | 0.860 | **−0.513** | ↓ 在学 |
| segmentation | 1.390 | 1.097 | −0.293 | ↓ 在学 |
| video | 1.635 | 1.409 | −0.226 | ↓ 在学 |
| **normal** | 1.705 | **1.855** | **+0.150** | ↑ **变差** |

**五个模态里四个在学会无锚生成。** §46.9 那句「第 2000 步时锚帧依赖尚未打破」是
**单集 + step-2000 的快照**,恰好采在趋势显现之前。

更强的一条:**action 的全程比值 0.945,低于 1** —— 它在没有自己锚帧时**不更难**。
四路观测携带了决定动作所需的信息,这是跨模态信息流入 action 的直接证据。

**`normal` 是唯一变差的,而它恰恰是 depth 的解析函数** —— 说明模型没在学 depth→normal
这个硬约束,而是把 normal 当独立纹理记。这正好印证把它当「最锐利探针」的定位。

**RoPE 外推可以排除。** 同模态 view0(位置 0–43) → view1(55–109):
video −0.0228、depth +0.0165、seg +0.0052、normal +0.0062、action +0.0052,
幅度仅 0.3–1.9% 且有一个为负。系统性位置退化应当是 5/5 且幅度可观。
**所以 `rgb_only` 弱是学习问题,不是位置编码问题** —— §46.4 标记的那个 open question 退役,
也不必为它做「每段 RoPE 重置」的改造。

**F1 调度实现得很准**(0.299/0.301/0.157/0.145/0.098 对声明的 0.30/0.30/0.15/0.15/0.10)。
⚠️ 算这个比例时我一开始用 `len(segloss_pos/0)` 当分母,得出加起来 1.34 的荒谬结果:
段 0(video view0)在 `rgb_given` 里是 fully_given、在 `policy` 里是单帧锚点,**两种情况下没有
预测位置、按设计不记**,占 25.5% 的步。分母必须是 `_step` 去重后的真实步数。

**lr 5e-7 偏保守而非偏高**(trainer 日志 500 条):grad_norm 中位从 0.716 单调降到 0.269,
全程只有 **5.0%** 的步触到 `max_grad_norm=1.0` 且集中在 warmup(前 10% 的 p90 2.640,
末段 0.504),末段 max 0.821 **低于**裁剪阈值;loss 到最后仍在降(0.0445→0.0423)。
**欠训练,不是不稳定。** 降到 3e-7 只会让一个本来就慢的 run 更慢。

### 46.13 转到 CHTC 训练(2026-09-13)

原机八卡长期被外部租户占满,续训和评测都排不上。转 UW-Madison CHTC。
新增 `chtc/`(6 个文件),设计见 `chtc/README.md`。三个关键决定:

1. **暖启动 `step5000.ckpt`,不做 DeepSpeed 续跑。** `global_step5000/` 里是
   `bf16_zero_pp_rank_0..3` 四份分片,**ZeRO-2 没法把四份加载进一个 rank**,而 CHTC 大概率只给
   一张卡。暖启动保住全部权重,只损失 Adam 动量(恒定 lr 下 warmup 期重建)。必须用新 output_dir。
2. **全局 batch 用累积钉死在 4。** `train_fusion.sh` 新增 `GRAD_ACCUM = 4 / NPROC`,
   1/2/4 张卡都得到 batch 4。这让配方脱离调度器给几张卡 —— 也让「batch=1 要不要降 lr」这个
   问题根本不出现。回显也改成诚实的 `NPROC(n) x per_device(1) x accum(a)`。
3. **按显存而非 capability 选卡。** 一张卡时 ZeRO-2 没有东西可分片,从 4-rank 的 36.6 GB 反推
   单卡峰值 47–56 GB → **L40S(45GB) 不可用**,A100-80/H100/H200 可用。所以设
   `gpus_minimum_memory=80000` 而**不**设 `capability=9.0`:后者会白白排除 8 张 A100,
   而这是显存瓶颈不是 SM 特性瓶颈,池子 24→32 张更重要。

**网络限制(实测):** 本机到 `ap2002.chtc.wisc.edu` 的 22 和 443 **都超时**,而 `github.com:22`
和 `chtc.cs.wisc.edu:443` 可达 —— 不是本机封出向 SSH,是 CHTC 按来源 IP 过滤,**需要 UW 网络身份**。
本机在 UMD VPN 上,两个 full-tunnel VPN 不能可靠共存。
结论:提交作业在用户笔记本(UW VPN)上做,`stage_to_chtc.sh` 在 CHTC 内部从 HF 拉数据,
本机继续负责代码/评测/论文。详见 `HANDOFF.md`。
