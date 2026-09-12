# EVAL_PLAN_ARM7 —— arm7 的评测方案

> 日期：2026-09-04。对象：`outputs/arm7__seed42_fi3_512_aug_sr`
> （`video+action@0.4, depth+action@0.2, segmentation+action@0.2, normal+action@0.2`，
> `--action_mask_mix A1`，`scene_roles`）。写这份文档时训练在 step 2092/10000，
> 19.2 s/it，剩余约 42 小时。
>
> 配套阅读：`MODE_MIX_MASKS.md` §9.5（M/A 两条 mask 轴与 A1 的理由）、
> `CHANGES_TEMPLATES.md` 附录（A 轴与 arm7 的改动记录）、
> `EVAL_PLAN_CLOSEDLOOP.md`（闭环协议与指标定义）、
> `paper/EXPERIMENT_STATUS_zh_aug29.md`（流水账日志）。
>
> 本文件只覆盖 **arm7 的评测**。ablation 臂延后，理由见 §7。

---

## 0. 决定评测形态的那个结构性事实

**arm7 是四个可互换的输入适配器共用一个 action 解码器，不是把四路融合起来的模型。**
这一条决定了下面所有设计，先把证据摆出来：

| 证据 | 位置 |
|---|---|
| `draw_template` 是**分类抽取**，每个样本只拿一个模板 | `training/dataset/rlbench_selfgen.py:605` |
| `_batch_template` **拒绝**一个 batch 里出现多个模板 | `train.py:_batch_template` |
| 推理只要求该模板用到的 stream：`video+action` 只需 `{"video"}` | `training/wan_video_action_images.py:576-578` |
| 闭环实际传的就是单路 RGB | `eval/policy.py:188-189` |

所以每个训练步都是一条 4 段序列 `X0|A0|X1|A1`，其中 X **有且只有一个**；四个模态**从不**同时
出现在一个序列里。arm7 一次只被问一个模态，靠 prompt tag + 锚定帧像素选路。

真正需要同时给四个锚定帧的模板是 `video+depth+segmentation+normal+action`——10 段、序列约
2.5×、attention 约 6.3×，**arm7 训练时 0% 见过**，不在本方案内。

**两个已定决策**：

1. 另外三路 action 当**集成（ensemble）**用（§3）。
2. **不改仿真器**，闭环只跑 RGB，四模态相关的全部放在离线（§4、§7）。

---

## 1. arm7 的完整设置

| 项 | 值 |
|---|---|
| `template_mix` | `video+action@0.4, depth+action@0.2, segmentation+action@0.2, normal+action@0.2` |
| `action_mask_mix` | **A1** =（iiii .75, fiii 0, fifi .05, policy .20） |
| `perception_mask_mix` | M2 —— **完全失效**，菜单里没有不含 action 的模板 |
| `segmentation_mode` | `scene_roles`（标签 `<scene-seg>`） |
| 数据树 | `rlbench_selfgen_512_aug`，`--variations 0` → 788 episode / 16 任务 |
| 分辨率 | 512×512，`num_frames` 41 |
| **步长 stride** | **`frame_interval` 3** → 一个窗口覆盖 **6.0 秒**运动，等效 **6.7 Hz** |
| 初始化 | warm-start `anyeZHY/ActionImages/step125750.ckpt` |
| 训练计划 | `max_steps` 10000，lr 5e-7，warmup 1000，`constant_with_warmup` |
| 优化 | ZeRO-2 offload、`full_param True`、bf16、梯度检查点、accum 1、clip 1.0 |
| batch | `per_device_train_batch_size 1` × 2 卡 |
| checkpoint | 每 1000 步、全留；`keep_optimizer_last_only` 剪枝后总占用约 257G |
| seed | 42 |

> **checkpoint 占用已实测**：一个目录在它是 resume tip 时是 120G（12.8G 权重 +
> 108G ZeRO-2 优化器状态），下一个 checkpoint 落盘后被剪到 12G。日志实证：
> `pruned 115.4 GB of redundant checkpoint state (resume state kept only in checkpoint-2000)`。
> `scripts/train_arm.sh` 的投影公式本次已按这个稳态改正（原来按 13G/个 算，漏掉了 tip 的 108G）。

**任何评测都必须传** `--frame_interval 3`、`--res 512`、`--segmentation_mode scene_roles`、
`--prompt_tag_style explicit`。这几项不一致是**静默**的，会直接让读数作废
（`--frame_interval` 不匹配 = 条件分布离分布；`scene_roles` 与 `referring` 是两套不同的
prompt tag **和**不同的像素）。

---

## 2. 已有的基线，以及真正的门槛

闭环 **main5 协议**（5 任务 × 20 trial = n=100，variation0，cfg 7.5 / fi 3 / explicit /
sphere / planning / factor 1.5 / anchor 4）。这是唯一一个三个基线都已跑完的协议：

| 模型 | ckpt | 成功率 | 来源 |
|---|---|---:|---|
| **specialist_action** | step4000 | **46.0%** | `reports/closedloop/rollout_spec_action_4000.json` |
| arm6 | step10000 | 35.0% | `reports/closedloop/rollout_arm6_10000.json` |
| Initialization（官方） | step125750 | 24.0% | `reports/closedloop/rollout_official_125750_postfix_n20.json` |

**arm7 要赢的是 46%，不是 35%。** specialist_action（HF:
`TingtingDu/specialist_action__seed42_fi3_512_aug_sr`）已经赢了 headline 臂，而且只训了 4000 步。
它不只是一个 ablation，它是当前最强的闭环模型。

离线对应物也都在：`reports/heldout_batch_eval/{init_125750, actionspec_4000_matched,
arm6_10000}`，以及各自的 `_unseen_task` 版本。

**结论**：Initialization 的评测**已经跑过**（main5 / unseen4 / var1 / 离线全都有），不需要重跑。

**两个残缺的 campaign**（2026-09-04 被停）：`init_gated12_v1`（没写出 summary）、
`actionspec_gated12_v1`（120 个只跑了 68 个）。只有在论文决定用 gated12-v1 协议时才需要补跑。

---

## 3. 三层评测设计

### Tier 1 —— 逐模态 action 读数（通用性）

同一个 episode 问四遍，每个适配器一遍。回答「一个 checkpoint、四种观测空间」。

```bash
python eval/eval_action.py --ckpt <arm7 ckpt> \
  --template {video,depth,segmentation,normal}+action \
  --frame_interval 3 --res 512 --segmentation_mode scene_roles
```

`--template` 本次已建好；四个模板的 `X0|A0|X1|A1` 切片完全相同（段序 view-major、action 在
视角内最后），并在**启动时**加了两模态守卫，不会跑完 50 步扩散才报错。

留出集上带 bootstrap CI 的版本走 `scripts/heldout_batch_eval.py`。**这里要改一处代码**：
`MODALITY_TEMPLATE`（`scripts/heldout_batch_eval.py:64`）写死的是 `video+X` 感知模板：

```python
MODALITY_TEMPLATE = {"depth": "video+depth", "segmentation": "video+segmentation",
                     "normal": "video+normal", "action": "video+action"}
```

把 arm7 这一族加成**新的 key**（如 `action_from_depth: "depth+action"`），
**原有四个 key 一个字节不动**——否则以前每一次运行都不再可复现。

指标现成，不用新写：`pos_err_m`、`rot_err_med_deg`、`r_peak`、`gripper_acc`、
`sparsity_dark_frac`。

### Tier 2 —— 模态集成（训四个适配器的回报）

> **2026-09-04 修正**：本节最初写的是「四路 heatmap 堆到 V 轴上解成一个位姿，解码器一行不改」。
> **那是错的**，实测见下。正确做法是各自解码后在位姿层面取中位数。

#### 先搞清楚解码器实际在做什么

`fuse_multiview_heatmaps_to_3d_point_torch`（`training/utils.py:793`）**不是最小二乘三角化**，
是「一维搜索 + 投票」：

| 步 | 做什么 | 位置 |
|---|---|---|
| 1 | 每视角取 heatmap 的加权 2D 质心 | utils.py:865-872 |
| 2 | 反投影成世界系射线 | utils.py:889-893 |
| 3 | **取 view 0 的射线作 principal ray**，沿它采 D 个候选点 | utils.py:896-897 |
| 4 | 把候选点投影到**全部** V 个视角，双线性采样各自 heatmap → (B,V,D) | utils.py:950-979 |
| 5 | **沿 V 轴连乘** `sampled_scores.prod(dim=1)` | **utils.py:983** |
| 6 | argmax 选深度 | utils.py:986 |

**解一定落在 view 0 的射线上**；其余视角只是给深度打分的否决器。第 5 步是 `prod`，
即**合取（AND）**，不是平均。

#### 所以直接堆 V=8 是错的（合成实验实测）

| 实验 | 位置误差 |
|---|---|
| V=2 基线 | 2.2 mm |
| 精确复制成 V=4 | **2.2 mm，完全不变** |
| 仅调换拼接顺序 | 2.2 → **7.6 mm** |
| 四路里一路偏 20px | 2.2 → **84.4 mm** |

1. **几何零增益**：四个模态共用同一对相机，只有 2 个相机中心。复制视角是恒等操作
   （`prod` 复制 = 平方，单调，argmax 不变）。
2. **顺序敏感**：principal ray 恒取 view 0，位置被锚定在拼接时排第一的那个模态。
3. **`prod` 一路坏就毁掉**：任何一路在重投影点响应趋零，就把该候选乘成 0。集成本该对单个坏
   成员鲁棒，`prod` 恰好相反。

3 好 1 坏（一路偏 20px）时的方案对比：

| 方案 | 误差 |
|---|---|
| 只用单 RGB | **3.0 mm** |
| V=8 直接堆 | 37.8 mm（比单 RGB 差 12 倍） |
| 各自解码取**均值** | 51.3 mm（均值不抗离群） |
| **各自解码取中位数** | **4.3 mm** |
| 先平均 heatmap 再 V=2 解 | 36.4 mm |

#### 正确做法

**四路各自按 V=2 解码，在位姿层面取中位数**：

- 位置：三个语义点（R/G/B 通道）各自的 3D 坐标取**逐分量中位数**，或几何中位数（Weiszfeld）。
- 旋转：先各自解出 `R_hat`，再取弦中位数（四元数对齐半球后取中位数并重正交化）。
- 报剂量响应：从 1 路（RGB）到 2/3/4 路，看误差随集成规模怎么变。

代价：4 次解码而不是 1 次（解码是纯几何，比扩散采样便宜得多，可忽略）。
**`fuse_multiview_heatmaps_to_pose_torch` 本身仍然一行不用改**——只是调用它四次。

#### 这一节要回答的问题（不是它的前提）

**四路的误差是否独立？** 四个适配器共享同一个 backbone，如果错误高度相关，集成收益会很小
——上面的玩具实验里中位数（4.3 mm）也没有赢过单 RGB（3.0 mm）。
所以 Tier-2 的产出应当是「集成有没有用」这个**测量结果**，而不是预设「四路更好」。
若实测集成不优于单 RGB，那本身就是一个干净的结论：适配器共享表征，误差不独立。

**局限照旧**：集成需要 GT 的 depth/seg/normal 锚定帧，是离线论断，不是可部署策略。

### Tier 3 —— 闭环，只用 RGB（headline）

`video+action`，main5 协议，参数与 §2 三个基线**逐字节一致**，这样数字直接进同一张表。
`scripts/queue_cl_one.sh` 已经把协议和 GPU 租借封好了。

评哪些 checkpoint：

- **step 4000**——与 specialist_action **action 更新次数完全匹配**。两者都有约 90% 的样本带
  action（arm7 是四个模板全含 action，specialist 是单模板含 action），所以同一步数下看过的
  action 更新次数相同。这是公平的对比点。
- **step 10000**——arm7 的上限。

两个 campaign，各约 **11.3 GPU-小时**（实测 6.78 分钟/rollout 均值，取自 arm6 那次 130 个
rollout 的 `rollout_seconds`）。

**不能只看终点**：arm4 历史上出现过 5000 步 20% → 7500 步 10% 的先涨后崩
（`reports/closedloop/rollout_arm4__seed42_fi3_{5000,7500}.json`）。

---

## 4. 要改的代码

1. `scripts/heldout_batch_eval.py:64`——`MODALITY_TEMPLATE` 加 `X+action` 族的新 key，
   原有 key 不动。
2. **新增**：集成解码。对同一个 episode 跑四次生成（四个模板各一次，之间用
   `_seed_all(seed)` 对齐到同一窗口/视角对），每次**照常按 V=2 解码**得到一个
   `pose8`，然后在**位姿层面**做稳健合并：位置逐分量中位数，旋转取弦中位数。
   再按 1/2/3/4 路输出剂量响应。
   **复用** `eval/eval_action.py` 的切片（`q = len(arr)//4`，action 段在 index 1 和 3）与
   `heldout_batch_eval.py:245` 的 `_seed_all` 对齐；`fuse_multiview_heatmaps_to_pose_torch`
   本身不改，只是调用四次。
   ⚠️ **不要把四路堆到 V 轴上**——`prod` 聚合是合取不是平均，实测一路坏会把误差从 3 mm
   推到 38 mm（见 §3 Tier 2）。
3. `eval/rollout_env.py`、`eval/policy.py`、训练代码：**都不动**。

---

## 5. 排期

目前只有 GPU 5 空闲（arm7 占着 2 和 3，其余被别的容器占用；判断依据是 `memory.used` 而不是
`utilization`）。

| 时间 | 做什么 | 成本 |
|---|---|---|
| 现在 | 写 Tier-1 + Tier-2 代码，并拿 `specialist_action/checkpoint-4000` 空跑一遍做负对照（§6.1） | CPU + 约 1 GPU-h |
| 约 11 小时后（ckpt 4000 落盘） | arm7@4000 的 Tier-1 + Tier-2 离线；Tier-3 闭环 @4000 | 约 3h + 11.3 GPU-h |
| 约 42 小时后（训练结束） | 1000..10000 整条曲线的 Tier-1 + Tier-2；Tier-3 @10000 | 约 30h + 11.3 GPU-h |

---

## 6. 验收

### 6.1 负对照（最重要的一条）

Tier-1 打 `outputs/specialist_action__seed42_fi3_512_aug_sr/checkpoint-4000`：它**只训过**
`video+action`，所以 `depth+action` / `segmentation+action` / `normal+action` 三行必须给出
**低 `r_peak`、退化的 `pos_err`**。

如果它们看着正常，说明评测读的根本不是适配器，而是别的东西——这条不过，后面所有数字都不能信。

### 6.2 其余

| # | 检查 | 不过意味着 |
|---|---|---|
| 2 | Tier-2 四次调用的 `view_indices` / `frame_indices` 一致 | 在融合不同场景 |
| 3 | 集成路径取 V=2 子集，复现单独跑 `video+action` 的 Tier-1 数字（浮点噪声内） | `extr`/`intr` 平铺写错了 |
| 4 | arm7 的 Tier-3 `args` 块，除 `ckpt`/`tag` 外等于 `rollout_spec_action_4000.json` 的 | 对比不成立，先 diff 再信 |
| 5 | 每次离线调用都传 fi 3 / res 512 / scene_roles / explicit | 离分布读数 |

---

## 7. 延后的事项

### 7.1 Ablation 臂

已讨论但**不在本方案内**。给以后留个记录：

**「depth+action@1.0 / seg+action@1.0 / normal+action@1.0 三个 100% specialist」跑不了 RGB 闭环**
——它们菜单里没有 `video+action`，而仿真器只渲染 RGB（`eval/rollout_env.py:187` 显式
`depth=False, mask=False`）。所以它们只会产生几行离线数字，而 headline 那一格是**空的**。

若要 ablation，建议改成**在固定 action 预算下扫观测模态的数量**，这样每个臂都保有闭环能力：

| 观测模态数 | 菜单 | 闭环 |
|---|---|---|
| 1 | `video+action@1.0` | 已有 46.0% |
| 2 | `video+action@0.6, depth+action@0.4` | 需新训 |
| 4 | arm7 | 在跑 |

### 7.2 仿真器出 depth / normal / seg

现成路子是有的——`scripts/gen_dataset.py:210` 已经在用
`CameraConfig(rgb=True, depth=True, mask=True, image_size=RES, depth_in_meters=True)`。

| 模态 | 难度 | 说明 |
|---|---|---|
| depth | 低 | 翻一个 CameraConfig 开关 + `encode_depth`（与训练同一个 codec）。注意 `depth_in_meters=True` |
| normal | 低 | 由 depth 加焦距算出（`encode_normal_from_depth`）；内参 `eval/rollout_env.py:322` 已经在读了 |
| segmentation | **高** | 要现场查 `simGetObjectName` 建 handle→role LUT（`training/percep/scene_segments_gen.py:resolve_roles` 只需 `{handle: name}`），再叠 `seg_targets` 的 target 提升。三个里唯一一个**错了不报错**的 |

代价是每步多一次 depth 渲染——`rollout_env.py:185-187` 的注释写明当初关掉就是为了这个。

### 7.3 归因上的已知取舍

arm7 与 arm0（`specialist_action`，A0）差**两处**：template menu 和 mask 轴（A1 vs A0）。所以
arm7 − arm0 的差异**不能单独归因**于其中任何一个。这是明知的取舍——本轮目标是把 action 做好，
不是做干净的单变量消融。真要归因，最便宜的办法是补一个 `ACTION_MASK_MIX=A1` 的 4000 步 action
specialist（约 0.6 天 / 2 卡），而不是重训 arm7。
