# U-CoGen 实验总纲(Experiment Outline)

> 常驻参考文档。记录**全部**计划中的实验:每个实验回答什么问题、填论文哪张表、
> 现在什么状态、怎么跑、坑在哪。最后更新 2026-08-30。
>
> 相关文档:
> - `paper/draft_aug_29.md` —— 论文正文(实验部分已按本文档重构)
> - `paper/U-CoGen_experiment_plan.md` —— 协作者写的实验部分重构建议(本文档的上游)
> - `paper/EXPERIMENT_STATUS_zh_aug29.md` —— 逐表 gap 分析 + 决策记录(过程性文档)
> - `ROBUSTNESS_PLAN_aug29.md` —— 8/29 两批无人值守任务的执行记录

---

## 0. 实验设计的一条铁律

**所有拿来横向比较的 arm,必须在同一棵数据树(`rlbench_selfgen_512_aug`)、
同一套 recipe(seed 42 / lr 5e-7 / 1000 warmup / fi=3 / 512²)、同一个起点
(官方 `step125750` 冷启动)上训练,只允许 `--template_mix` 这一个变量不同。**

这是 `scripts/train_arm.sh` 自身的设计哲学(它把 tree/fi/seg_mode 都编进输出目录名,
就是为了让不可比的 arm 不可能被误当成可比)。历史上的 arm4 / 旧 arm1 是在已删除的
256² 树上训的,**即使找回备份也不可比**,所以已经从论文中移除(`draft_aug_29.md`
里留了 `%%` 注释说明原因,不是遗忘)。

---

## 1. 模型清单(最终版,5 个 + 1 个零样本参照)

| 名称 | template_mix | 步数 | 输出目录 | 状态 |
|---|---|---|---|---|
| **Action Images init** | 官方发布权重,零样本 | 125,750(不再训) | `starVLA/.../step125750.ckpt` | ✅ 已有 |
| **action specialist** | `video+action@1.0` | 4000 | `outputs/specialist_action__seed42_fi3_512_aug_sr` | 🟡 训练中(GPU 0,2) |
| **depth specialist** | `video+depth@1.0` | 4000(**实际停在 2972**) | `outputs/specialist_depth__seed42_fi3_512_aug_sr` | ✅ 已停,ckpt 500–2500 |
| **seg specialist** | `video+segmentation@1.0`(scene_roles) | 4000(**实际停在 3067**) | `outputs/specialist_seg__seed42_fi3_512_aug_sr` | ✅ 已停,ckpt 500–3000 |
| **normal specialist** | `video+normal@1.0` | 4000 | `outputs/specialist_normal__seed42_fi3_512_aug_sr` | 🟡 训练中(GPU 3,4) |
| **arm6(headline)** | `action@0.4, depth@0.2, seg@0.2, normal@0.2`(M2) | 10000 | `outputs/.grid_pin/step10000.ckpt` | ✅ 已完成 |

**conditioning 混合变体**(只改 `--perception_mask_mix`,template_mix 与 arm6 相同):

| 变体 | IIII / FIII / FIFI | 步数 | 状态 |
|---|---|---|---|
| M2 = arm6 本身 | 0.90 / 0.05 / 0.05 | 10000 | ✅ |
| M1 | 0.45 / 0.10 / 0.45 | 3000(从 arm6@10000 续训) | ✅ `outputs/arm6m1__...` |
| M0 | 0.10 / 0.00 / 0.90 | 10000(必须冷启动) | ❌ **已决定暂时不跑**(2026-08-30),~58h/2GPU |

> ⚠️ **M1 与 M0 不对称**:M1 是从 arm6 收敛点续训 3000 步的廉价消融;M0 在这棵树+
> 这个 template_mix 上从没训过,没有收敛点可续,必须从 `step125750` 冷启动跑满
> 10000 步才和 M2 可比。这个不对称在写 `tab:conditioning_results` 时必须说明,
> 否则 M1 那一行是"arm6 + 3000 步 M1",不是"从头用 M1 训的模型"。

---

## 2. 实验清单(按论文表格组织)

### E1 —— held-out 批量定量评测 → `tab:headline`、`tab:per_task`

**问题**:一个 checkpoint 能不能在**没见过的 task variation** 上同时输出五种模态?

**做法**:`scripts/heldout_batch_eval.py`(本次新写)。复用
`scripts/modality_mode_grid.py` 的生成/打分逻辑,循环 `EVAL_PLAN.md` §3.3 定义的
16 个 episode(8 个 seen/variation0 + 8 个 unseen/variation1,已核实全部存在),
4 个模态,只测 `IIII`,输出 mean + 95% bootstrap CI,按 seen/unseen 分组。
RGB 指标(PSNR/SSIM/LPIPS)顺带算——因为四个模板本来就都以 `video` 段为 anchor,
不需要额外生成调用。

**要跑几次**:每个 checkpoint 一次。至少需要 arm6@10000、四个 specialist、
官方 init,共 6 次 × ~3.3h。

**状态**:❌ 脚本写完并 CPU 冒烟测试通过,但**一直没抢到 GPU**(重试脚本跑满 200 次
退出)。这是当前**最高优先级**的待跑项——`tab:headline` 是论文的头号表。

```bash
nohup bash scripts/run_heldout_batch_eval_retry.sh > logs_heldout_batch_eval.txt 2>&1 &
# 或指定 checkpoint:
python scripts/heldout_batch_eval.py --ckpt <ckpt> --tag <name> --gpu <N>
```

> ⚠️ `tab:per_task` 需要**按任务**分组,而脚本现在按 seen/unseen 分组。
> 原始 `per_episode.json` 每条都带 episode 路径,能反推任务名,**只需要补一个聚合
> 视图,不需要重跑生成**。

---

### E2 —— 四个 specialist vs arm6 → `tab:matched_exposure`

**问题**:共用一个 head 到底有没有代价?代价里多少是"参数干扰",多少只是
"采样稀释"(每个模态被抽到的次数变少)?

**做法**:按**模态特定更新次数**对齐比较,不是按 total step 比。

**⚠️ 这里最容易出错——必须挑对 checkpoint:**

| 模态 | arm6 的更新次数 | specialist 该用哪个 checkpoint |
|---|---|---|
| action | 10000 × 0.4 = **4000** | **checkpoint-4000**(final,正好匹配) |
| depth | 10000 × 0.2 = **2000** | **checkpoint-2000**(不是 final!) |
| segmentation | 10000 × 0.2 = **2000** | **checkpoint-2000** |
| normal | 10000 × 0.2 = **2000** | **checkpoint-2000** |

depth/seg/normal specialist 的 **final checkpoint-4000 是两倍于 arm6 的曝光**,
拿它跟 arm6 比会得出一个看似公平实则偏向 specialist 的错误结论。final checkpoint
另有用途:报告"specialist 在两倍曝光下的天花板"。`CKPT_EVERY=500` 就是为了留出
checkpoint-2000 这个中间点。

**状态**:🟡 三个 specialist 训练中,normal 待重排。评测依赖 E1 的脚本。

---

### E3 —— conditioning 分布矩阵 → `tab:conditioning_results`

**问题**:训练时的 conditioning 分布,是不是决定了成品 checkpoint **能回答哪些问题**?
(这是方法部分第二个贡献 $p(\Pi,\rho)=p(\Pi)p(\rho|\Pi)$ 的直接验证)

**做法**:M0 / M1 / M2 三个训练分布 × `IIII` / `FIII` / `FIFI` 三种测试模式,
在完整 held-out 集上测 depth。预期趋势:M0 的 `FIFI` 最好,M2 的 `IIII` 最好。
如果成立,就直接支持 introduction 里那句 "perception is not a modality, it is a
$(\Pi,\mathcal{C})$ pair"。

**状态**:🟡 M2 有了(=arm6),M1 有了,**M0 完全没跑**。而且
`heldout_batch_eval.py` 现在 `MODE` 是硬编码的 `iiii`,要支持三种模式需要改成循环
(几十行,不难,但评测时间 ×3)。

**建议**:也可以画成 Pareto 图(横轴 FIFI AbsRel,纵轴 IIII AbsRel,三个点),
比表格更有概括性(见 `U-CoGen_experiment_plan.md` §11)。

---

### E4 —— anchor vs tag 消融 → `tab:tag_swap`(§4.5)✅ **已完成**

**问题**:到底是 **prompt tag** 还是 **anchor 帧** 在选择模态?

**做法**:两个互补的探针。
1. **换 tag**:固定所有 pipeline 看得见的东西(段布局、streams、anchor 帧)在目标模态,
   **只改文本**,然后按目标模态解码打分。`scripts/tag_swap_ablation.py`
2. **拿掉 anchor**(`f0f0`):保持 tag 正确,去掉 anchor 帧,让 prompt 成为唯一的
   模态指示。**不是训练模式**,只作对照。

**结果(arm6@10000,4 个 episode × 3 个目标 × 4 个 tag)**:

- **tag 几乎不起作用**:对角线(tag 正确)在 4 个 tag 中平均排名 **2.7/4**,
  随机排序是 2.5/4——正确的 tag 并不比错误的 tag 好,好几行反而更差。
  多数行的跨 tag 波动 <5%,在重复生成的噪声范围内。
- **anchor 起决定作用**:去掉 anchor 后同一 checkpoint 崩溃——
  depth AbsRel `0.090 → 0.808`,seg mIoU `0.478 → 0.056`,
  normal cos `0.895 → −0.263`(**符号翻转**,法向量指反了)。

**这个发现改变了什么**:参数层面的 claim 完全不受影响(一个 checkpoint、一个 head、
一个 loss、五种可解码输出、零额外参数)。但**机制**跟原来假设的不一样:在 `IIII` 下,
是每个辅助段的**首帧**(conditioning 轴上的对象)在标识模态,不是 template tag。
原因在构造里就能看出来:因为 `assemble()` 无条件给每个预测段发一个干净的首帧,
tag 在训练时从来不需要承载信息,所以它也就没学会承载。

**实践后果**(已写进论文,不留给读者自己发现):新模态**不能只靠 prompt** 在推理时
引入,它需要一个自己表示空间里的 anchor 帧。"task set 是数据混合的属性而非架构的属性"
这个 claim 依然成立,但**在已学会的任务之间做选择的接口是 conditioning plan,不是文本**。

> ⚠️ 小瑕疵:`f0f0` 是在 step 9000 测的,`IIII` 基线是 step 10000。要完全干净应该在
> 同一 checkpoint 重测一次 f0f0(~15 分钟,1 GPU)。不影响结论方向(9 倍的效应量
> 不可能被 1000 步的差异解释),但值得补。

---

### E5 —— 闭环控制 → `tab:closed_loop`(§4.6)✅ **已完成**

**问题**:加了三路稠密感知输出之后,策略还能不能用?

**做法**:receding-horizon 控制,5 任务 × 20 scene seed = 100 rollout,
**seed 配对**,逐任务精确 McNemar。每个模型用**自己训练时的协议**跑
(arm6: fi=3 / 带 tag / cfg=7.5;官方: fi=4 / 无 tag / cfg=10),
强行统一协议只会测出分布偏移而不是控制能力。

**结果(按"训练过 / 没训练过"分开读,这是关键)**:

`close_box` 和 `close_drawer` **根本不在我们的 16 个训练任务里**——arm6 从来没见过
它们的任何 variation。另外三个在训练树里,这里测的是 variation 0。混在一起报是错的:

| 任务 | 官方 init | arm6 | Δ | b/c | p |
|---|---|---|---|---|---|
| *— 训练分布内(variation 0)—* | | | | | |
| push_buttons | 15% | **60%** | +45 | 11/2 | **0.022** |
| meat_off_grill | 5% | **45%** | +40 | 9/1 | **0.021** |
| open_drawer | 0% | 0% | 0 | 0/0 | 1.000 |
| **小计** | **6.7%** | **35.0%** | **+28.3** | 20/3 | **0.0005** |
| *— 从没训练过 —* | | | | | |
| close_box | 20% | 25% | +5 | 4/3 | 1.000 |
| close_drawer | **80%** | 45% | −35 | 0/7 | **0.016** |
| **小计** | **50.0%** | **35.0%** | **−15.0** | 4/10 | 0.180 |
| **全部 5 个合计** | 24% | 35% | +11 | 24/13 | 0.099 |

**怎么读**:合计的 +11pp / p=0.099「不显著」是**假象**——它把两个方向相反、量级相当的
效应平均掉了。拆开之后:
- **训练分布内**:6.7% → 35.0%,**+28.3pp,p=0.0005**(高度显著)
- **没训练过的任务**:50.0% → 35.0%,−15pp,p=0.18(不显著)

这是**常规的微调权衡**(域内大幅提升、牺牲一部分预训练模型的广度),不是 co-generation
特有的问题。`close_drawer` 是代价集中的地方——一个 arm6 从没见过、而官方 checkpoint
恰好特别强(80%)的任务。`open_drawer` 两个模型都是 0%(GT 重放天花板 100%),是工装
问题,不贡献任何判别对。

**这张表支持的 claim**:加三路稠密感知输出**不以控制能力为代价**——一个把 60% 训练概率
花在非动作输出上的 checkpoint,在它训练过的地方全面优于自己的起点。它**不支持**任何
关于"泛化到没见过的任务"的说法,那里诚实的读法是有损失,而我们本来也没针对它训练。

> ✅ **你在文档里问的 TODO「应该改成 held-out 的测试啊？？！」——已查证并处理:**
> 1. **闭环 harness 完全支持 held-out**:`eval/rollout.py` 有一等公民的 `--variation N`,
>    `rollout_env.py` 会对 RLBench 的真实 variation 数做校验,scene seed 也是按
>    `(task, variation, trial)` 派生的。我们只是用了默认的 variation 0。
> 2. **但这张表其实已经包含 held-out 成分,而且是比 variation 更强的那种**:
>    close_box / close_drawer 是**任务级**的 held-out(整个任务从没训练过),
>    比"同一个任务的另一个 variation"更 OOD。之前的问题不是缺 held-out,
>    而是**把两类任务混在一个 pooled 数字里**。现已拆开报告。
> 3. **仍然值得补的**:把 `push_buttons` / `meat_off_grill` / `open_drawer` 在
>    **variation 1** 上再跑一次(这三个任务在数据树里都有 variation 1/2),
>    就能得到"同任务、没见过的配置"这一档,填满 held-out 的中间层级。
>    代价 ~3 任务 × 20 trial × 2 模型 ≈ 一次 6-7 小时的 campaign。
>    命令:在 `scripts/run_official_closedloop_postfix.sh` 基础上加 `--variation 1`
>    并把 `--tasks` 改成这三个。

> ⚠️ 另一个诚实性要点(已写进论文):官方是零样本迁移到我们的 rig,arm6 是域内微调过的,
> **这是主场优势对比,不是模型能力排名**。

---

### E6 —— 离线动作 + RGB 世界模型质量 → `tab:offline_preservation`

**问题**:加感知之后,动作精度和 RGB 未来预测质量各自掉了多少?

**做法**:init / action specialist / arm6 三行,在 held-out 上测
位置误差、旋转误差、夹爪精度、LPIPS、SSIM。这里**保留三行分解**
(闭环那张表因为太贵只测两端点),因为离线多测一个 checkpoint 只多一次解码。

- init → action specialist = 域适应的效果
- action specialist → arm6 = 感知联训的额外效果

**状态**:❌ 依赖 E1 脚本 + action specialist 训完。
LPIPS/SSIM 的代码**已经写好**(`score_video()`,装了 `lpips` + 用
`skimage.metrics.structural_similarity`),这是本次新增的——之前全仓库没有任何地方
实现过 LPIPS/SSIM,`tab:versatility` 里的 given-RGB PSNR 其实是 frozen VAE 的
重建天花板,不是预测质量。

---

## 3. 可选 / 低优先级(算力允许再做)

| 实验 | 回答什么 | 代价 | 状态 |
|---|---|---|---|
| **E7 三模态共生成** `video+depth+action` | 区分"prompt 切换"和真正的"co-generation"——感知和动作在**同一次生成**里共存 | 需要一次新训练(arm6 从没训过三模态模板) | ❌ 论文标 optional |
| **E8 新模态(optical flow)** | 证明"加一个输出是数据操作而非架构操作" | 新 codec + 需要过 admission gate(人工验证) | ❌ 不适合无人值守 |
| **E9 训练时去掉 tag** | 验证 tag 是不是语义路由器 | 一次训练 | ⚠️ **E4 已经能预判结果**:既然 anchor 才是选模态的,训练时去掉 tag 大概率不会崩。做之前先想清楚它还能加什么信息 |
| **E10 depth codec 对比** | grayscale / 普通 colormap / Gray-code path 过 VAE 后的 AbsRel | 纯 CPU/VAE,便宜 | ❌ P1 |
| **E11 深度↔法向一致性** | 从预测 depth 算出的法向 vs 直接预测的法向,cos 相似度 | 纯后处理,不用新生成 | ❌ 附录级,能回应"normal 是 depth 的确定性函数"的质疑 |

---

## 4. 当前 GPU 占用与队列

| GPU | 任务 |
|---|---|
| 0, 2 | action specialist |
| 3, 4 | depth specialist |
| 5, 7 | segmentation specialist |
| 1, 6 | **空闲**(留给别人) |

**排队待跑(按优先级)**:
1. **E1 held-out 批量评测**(1 GPU × ~3.3h × 6 个 checkpoint)—— 最高优先级,
   `tab:headline` 完全依赖它
2. **normal specialist 重排**(2 GPU × ~23h)—— 崩过一次,见下节
3. **M0 完整训练**(2 GPU × ~58h)—— E3 的缺口
4. E4 的 f0f0 在 step10000 补测(1 GPU × ~15min)

---

## 5. 已知的坑(每一条都真实踩过)

### 5.1 GPU 抢占竞态,有两种,防法不同

- **短窗口**(检查 → 申请,几秒):`nvidia-smi` 说空闲,但真去 allocate 时被抢。
  防法:用一次**真实 CUDA 分配**做探测(`wait_and_launch_*.sh` 里的 `probe_gpu()`),
  而不是只读 `nvidia-smi`。
- **长窗口**(进程启动 → DeepSpeed 真正上卡,1.5~2 分钟):这是 **normal specialist
  崩掉的原因**。`torchrun` + DeepSpeed 路径要经过 conda 激活、import、建模型,
  这段时间显存不会被预占,外部租户可以在这个窗口里冒出来吃满卡。
  **启动前几秒的探测防不住这一种**——这是训练脚本路径的结构性空档,不是探测逻辑的 bug。
  (对比:`modality_mode_grid.py` 那类脚本在最开始就用 `reserve_vram()` 占住显存再慢慢
  加载,所以只有短窗口问题。)

### 5.2 显存要看 `memory.used`,不要看 `utilization`
一个租户占着 45GB 但 util 0% 的卡,照样会 OOM 你。

### 5.3 output_dir 非空 = 静默续训
`train.py` 会优先 resume 而不是 warm-start。所有 `wait_and_launch_*.sh` 都有
"目录非空就拒绝启动"的保护。**normal specialist 的目录现在是空的,可以直接重排。**

### 5.4 prompt tag 不能含下划线
`train.py:287` 会洗掉 `_`/`.`,但推理路径不洗。`<scene_seg>` 曾因此让 train/eval
用了不同的字符串,已改成 `<scene-seg>`。

### 5.5 `queue_closedloop.sh` 的参数是给微调过的 arm 用的
它硬编码 `cfg=7.5 fi=3 explicit`。**官方 checkpoint 必须用它自己的协议**
(`cfg=10 fi=4 none`),否则等于让它在没训练过的 prompt 分布下跑,数字被人为拉低。
见 `scripts/run_official_closedloop_postfix.sh`。

### 5.6 anchor fix 之前的闭环数据不可用
`reports/closedloop/archive_pre_anchorfix/` 里的数据早于 `eval/rollout.py`
2026-08-22 的改动。**已废弃的旧草稿(draft_aug_28)的闭环表用的就是这批数据**
(pooled 31% vs 20%, p=0.052),不能和新生成的数据混用。本次 `tab:closed_loop` 的
官方基线是在当前 harness 下重新跑的,数字不同(24% vs 35%, p=0.099)。

---

## 5.9 论文当前的编译阻塞项(与实验无关,但迟早要修)

这几个是 `draft_aug_29.md` 里**先前就存在**的悬空引用,不是本轮改动引入的:

| 问题 | 位置 | 说明 |
|---|---|---|
| `\input{Figs/mask_full_grid}` | L525 | `Figs/` 目录不存在(只有 `figures/`),`fig:layout` 因此悬空 |
| `\ref{app:camera}` | L260 | Preliminaries 引用了一个从没写的相机附录 |
| `gen_inst` / `headings` / `others` / `sample-table` | L138, L1350 | 都在 `%` 注释里,无害 |

本轮补上了之前**用了但从没定义**的宏(`\ptag` `\best` `\tabref` `\figref`
`\secref` `\appref` `\tmpl`)——在此之前这份文件是编译不过的。

---

## 6. 论文表格 ↔ 实验 对照速查

| 表 | 内容 | 依赖实验 | 状态 |
|---|---|---|---|
| `tab:headline` | 4 specialist + arm6 五模态对比 | E1 + E2 | ❌ 等 E1 |
| `tab:matched_exposure` | 匹配更新次数的 specialist vs arm6 | E1 + E2 | ❌ 等训练完 |
| `tab:conditioning_mixtures` | M0/M1/M2 概率表 | —— | ✅ 真实值 |
| `tab:conditioning_results` | M0/M1/M2 × IIII/FIII/FIFI | E3 | ❌ 缺 M0 |
| `tab:tag_swap` | anchor vs tag | E4 | ✅ **真实值** |
| `tab:offline_preservation` | init/action-spec/arm6 离线指标 | E6 | ❌ 等 E1 |
| `tab:closed_loop` | init vs arm6 闭环 | E5 | ✅ **真实值** |
| `tab:per_task` | arm6 逐任务 | E1(补聚合视图) | ❌ 等 E1 |
| `tab:trimodal` | 三模态共生成 | E7 | ❌ optional |
| `tab:gate` | codec admission gate | —— | ✅ 真实值 |
| `app:overtraining` | loss 不能给 checkpoint 排序 | 历史观测 | ✅ 已写进附录 |

**当前 36 → 25 个 `[XX]` 占位符**,减少的部分来自 E4 和 E5 的真实数据。
