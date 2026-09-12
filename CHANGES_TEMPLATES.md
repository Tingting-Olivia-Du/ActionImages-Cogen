# CHANGES_TEMPLATES —— 从「模态顶替」到「任务模板」的完整改动记录

> 日期：2026-08-10。
> 设计依据：`/workspace/ttdu/ttd/plan/core/16-multitask_template_design.md`
> （上游基线仍是 `291f71cd74b154b8eaf49020035b84c7e044b542`；整个 fork 尚未 commit，
> 所以 `git diff` 只能给出「fork vs 上游」，给不出「本次 vs 上次」——本文件就是那份记录。）
>
> 配套阅读：`FORK_CHANGES.md`（fork vs 上游的全量改动；其 §0/§1 描述的顶替式设计已被本次取代，
> 该文件顶部已加指路标记）。

---

## 0. 一句话

上一版让 depth/seg **顶替** `sample["video"]`，代价是一个 depth 样本里**根本没有 RGB**——那是
depth 空间的世界模型，不是 GenCeption / Vision Banana 定义的**感知**（RGB 进 → depth 出）。
本次把「视觉槽位装什么」升级为**任务模板**：数据集按模态输出多路 `streams`，prompt 写出模态集合
Π，mask 决定哪些段是给定的。于是 `video+depth`（RGB 整段给定 → depth 预测）第一次可表达，
而且**段数与官方基线相同、显存成本为零**。

```
上一版   [ v1_depth | v1_action | v2_depth | v2_action ]   4 段，序列里没有 RGB
本次新增 [ v1_rgb   | v1_depth  | v2_rgb   | v2_depth  ]   4 段，rgb 整段给定 = 感知
本次新增 [ v1_rgb | v1_depth | v1_act | v2_rgb | v2_depth | v2_act ]   6 段 = co-supervision
```

---

## 1. 核心概念：两个正交轴

| 轴 | 谁控制 | 出处 |
|---|---|---|
| **Π —— 哪些模态在序列里** | prompt 标签 | GenCeption §3.5 / Vision Banana §2：一个样本一个任务，靠 prompt 选，靠**数据混合比例**平衡 |
| **哪些段给定 / 哪些段预测** | mask | Action-Images §3.3 自己的 mask strategy（i2va / a2v / v2a / video-only） |

「感知」不是一个模态，是一个 **(Π, mask) 组合**：`Π={video, depth}` + video 段整段给定。
顶替式布局在原理上表达不了它，这就是必须改 `forward` 的唯一理由。

模板名 = Π 按 canonical 序（video → depth → segmentation → action）用 `+` 连接：

| 模板 | 段数 @256² | 语义 |
|---|---|---|
| `video+action` | 4（2816 token） | 官方配方 |
| `video+depth` | 4（2816） | **真·感知**，与基线同成本 |
| `video+segmentation` | 4（2816） | referring seg |
| `video+depth+action` | 6（4224，~2.25× attention） | co-supervision，doc 15 §2.5 判定「不可达」的那一格 |
| `depth+action` | 4（2816） | 上一版的顶替臂。**是世界模型，不是感知**，报告时必须如此措辞 |
| `video` | 2（1408） | 官方的 video-only 模式（action dropout 的落点） |

---

## 2. 逐文件改动

### 2.1 `training/templates.py`（新文件，~270 行）

模板系统的全部逻辑，**不依赖模型、可在 CPU 上单测**：

| 导出 | 作用 |
|---|---|
| `CANONICAL_ORDER` / `VISUAL_MODALITIES` / `ACTION` | 段序与模态分类 |
| `parse_template` / `format_template` / `parse_template_mix` / `draw_template` | 模板名解析、`name@ratio` 混合、逐样本抽取 |
| `primary_visual` | 「上游称之为 video 段」的那个槽位。含 `video` 的模板返回 `video`；`depth+action` 返回 `depth` —— 正是这一条让顶替臂在同一套 mask 规则下复现旧行为 |
| `prompt_prefix` / `seg_tag` | 按 Π 组装 prompt；`style="none"` 复现上游文本 |
| `Segment` / `plan_segments` | 采样一次 (段序, 段长, 给定/预测) 计划 |
| `assemble` | 计划 + 各段 latents → `(latents, camera_emb, masks)`，**累积偏移** mask |

`plan_segments` 里的概率全部是从上游 `train.py:286-340` **逐字转写**的常量：
`SINGLE_FRAME_VISUAL_PROB=0.1`、`MODE_FIRST_SEGMENT_GIVEN_PROB=0.9`、
`MODE_ALL_VISUAL_GIVEN_PROB=0.95`。新增的只有 `PERCEPTION_VIDEO_GIVEN_PROB=0.9`
（无 action 的感知模板：90% 概率把锚定视觉整段给定，10% 保留联合生成的多样性，
免得模型学成「RGB 永远白给」）。

### 2.2 `train.py`

**`ActionImagesModel.forward`** —— 本次唯一改动训练核心的地方：

| 改动 | 原因 |
|---|---|
| 路由判据 `torch.sum(action_7d**2) != 0` → 读 `inputs["template"]` | 让 prompt 成为**事前决定**而非**事后描述**（doc 15 §6.2）。数值判据只保留为**防御性**检查：template 含 action 但 `action_7d` 全零（bridge）时仍退回视觉-only |
| 硬编码 4 段 `torch.cat` → `plan_segments` + `assemble` | 段数变成 Π×视角 的函数 |
| 相机 latent 从 src/tgt 两份硬编码 → `for v in range(num_views)` 循环 | 段数变化后必须按视角取、按段长切片 |
| 视觉 latent 从 `video[:, :, :T]` / `[T:]` → 遍历 `streams` 的每个视觉模态 × 每个视角 | 一个样本可以同时有 rgb 和 depth |
| 新增 `_batch_template()` 守卫 | 不同模板段数不同，无法 stack；混批时**报错**而不是静默选一个 |
| 新增 streams 缺失检查 | prompt 承诺了某模态但 batch 没带它的像素时立即失败 |
| **loss 一行未改** | 所有模态都在 `[-1,1]` 的 RGB-like 空间，统一 masked MSE（doc 15 §4.4） |

**`ActionImagesDataCollator.__call__`**：新增转发 `template` 与 `streams`
（此前两者都被静默丢弃）。`streams` 按 key 逐个 stack，key 集合不一致时报错。

**`train()`**：`CombDataset(...)` 改传 `template_mix` / `prompt_tag_style` /
`action_dropout_prob`；wandb config 同步记录这三项（取代 `visual_modality_mix`）。

### 2.3 `training/args.py`

新增 3 个字段 + 1 个弃用别名：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--template_mix` | `video+action@1.0` | 模板混合。默认即上游配方 |
| `--prompt_tag_style` | `explicit` | `explicit` 把 Π 写进**每个**样本的 prompt（含基线的 `<video><action>`）；`none` 复现上游文本，**只给回归测试用** |
| `--action_dropout_prob` | `0.1` | 复现上游 `random.random() < 0.9` 的 action gate（见 §3.2） |
| `--visual_modality_mix` | `None`（弃用） | 旧值 `video\|depth\|segmentation` 自动翻译成 `<modality>+action` 模板；与 `--template_mix` 同时给会**报错**而不是猜 |

`__post_init__` 增加了这三项的取值校验（非法的 tag style / 越界概率在启动时就炸）。

### 2.4 `training/dataset/rlbench_selfgen.py`

- `__init__` 参数 `visual_modality_mix` → `template_mix` + `prompt_tag_style` + `action_dropout_prob`。
- 删除 `_parse_modality_mix` / `_draw_modality`（逻辑搬进 `templates.py` 并泛化）。
- `getitem(index, force_modality=...)` → `getitem(index, force_template=...)`，返回值新增：
  - `sample["streams"]`：`{模态名: [C, 2T, H, W]}`，只装该模板点到的视觉模态；
  - `sample["template"]`：canonical 序的模板名。
  - `sample["video"]` **保留**，等于 `streams[primary_visual(Π)]`——既是 forward 推形状用的引用，
    也是旧代码的兼容别名。顶替式模板下它就是 depth，与上一版完全一致。
- seg 解析失败的降级从「整个样本退回 video」改为「**从 Π 里摘掉 segmentation**」；
  摘完若没有视觉模态了再补 `video`。
- `_self_test()` 从「只测一个感知模态」扩成「**菜单里每个模板各测一个样本**」，
  并且对 seg 会**连试最多 5 个 episode**——把「seg 在这棵树上根本解析不出来」（真故障）
  与「episode 0 恰好解析不出来」（正常）区分开。
- `action_7d` / `action_8d` 仍然**不清零**（上一版就已如此，这是 co-supervision 的前提）。

### 2.5 `training/dataset/base.py`

- `CombDataset.__init__` 参数换成 `template_mix` / `prompt_tag_style` / `action_dropout_prob`。
- **修了一个静默 bug（见 §3.4）**：`variations` 现在真的传给 `RLBenchSelfgenDataset` 了。

### 2.6 测试

| 文件 | 变化 |
|---|---|
| `tests/test_forward_unchanged.py` | **换了种类**：从「比对 `forward` 源码文本」改成「**行为等价**」。文件里逐字转写了上游 `train.py:286-346`，然后对每条决策路径（单帧变体 / p<0.9 / 0.9≤p<0.95 / p≥0.95 / 非 rlbench）比对 `latents` / `camera_emb` / `masks` **逐位相同**。另加：段序 view-major、canonical 序与拼写无关、感知模板的 mask、6 段 co-gen 布局与相机按视角复用、顶替模板复现旧行为、非法模板被拒 |
| `tests/test_selfgen_dataset.py` | 重写。新增：`video+depth` 同时带 rgb 与 depth 两路；`depth+action` **不带** RGB；prompt 与 streams 必须一致；`prompt_tag_style=none` 复现上游文本；action dropout 让 template 与 prompt **一起**掉 `<action>`；混合菜单每个模板都出现 |
| `tests/test_prompt_tags.py` | 扩到 5 个模板的全写前缀；断言「模板可从文本中还原」；`none` 风格；seg 无 color map 必须报错；token 预算 |
| `tests/test_selfgen_alignment.py` / `test_depth_roundtrip_e2e.py` / `test_new_episodes.py` | 改读 `sample["streams"][...]`，模板名改为 `video+depth` / `video+segmentation` |
| `debug/visualize_pipeline.py` | 改用 `force_template`；depth 侧用 `depth+action`（`sample["video"]` 就是 depth，可视化逻辑零改动） |

### 2.7 脚本

- **`scripts/train_arm.sh`**：臂由 `--template_mix` 唯一定义，提供三个具名臂
  `arm0`（`video+action@1.0`，对照）/ `arm1`（0.6/0.2/0.2 低比例感知）/
  `arm2`（0.5 action + 0.5 co-gen），也接受任意 mix 字符串。
  **默认改为 warm-start** `step125750`（10k 步）；`INIT_CKPT=` 留空则是阶段 2 的 from-base（125k 步）。
  checkpoint 间隔 250 步（要看清 r_peak 的「先涨后崩」形状，不是只看两端）。
- **`scripts/smoke_arm.sh`**：参数从模态名改为模板名/整条 mix；默认也 warm-start；
  头注释写明预期显存关系（4 段的三个模板应当相等，6 段的明显更高）。
- **`scripts/run_tests.sh`**：`python -u`（此前无输出缓冲，长测试看不到进度），注释更新。

---

## 3. 行为差异：即使配置「看起来一样」也变了的地方

这一节最重要——下面每一条都会改变训练，而且不会在 loss 曲线上表现出来。

### 3.1 每个样本的 prompt 都带标签了

`prompt_tag_style=explicit`（训练默认）给**所有**模板写全 Π，包括基线的 `<video><action> `。
上一版刻意让 video 无前缀以保持与上游文本逐字相同。

- **代价**：文本分布对每个样本都偏离了预训练先验。
- **为什么可接受**：所有臂用同一套写法，成本是**共模的**，在臂间差分中抵消；本研究的结论全部是臂间差分。
- **代价的兜底**：`--prompt_tag_style none` 复现上游文本，L2 数值回归挂在这上面。
- **实测开销**（UMT5）：`<video><action>` +5 token，`<video><depth><action>` +7，
  `<video><seg: a=red, b=green>` +14；最长 25 token，远低于 512 预算；无 `<unk>`。
- **评测注意**：`step125750` 从未见过任何标签，它的零样本读数**必须继续用无标签 prompt** 读。
  报告时要写清「零样本参照 = 无标签，各臂 = 有标签」。

### 3.2 action gate 从 forward 搬到了 dataset

上游：`if torch.sum(action_7d**2) != 0 and random.random() < 0.9:` —— 10% 的样本即使有动作
也走纯 video 分支。这条现在是 dataset 里的 `action_dropout_prob=0.1`，命中时模板从
`video+action` 变成 `video`，**prompt 同步掉 `<action>`**。

- 上游版本下，那 10% 的样本文本仍然描述着一个序列里并不存在的 action 流。搬过来才能让
  「prompt 就是控制信号」这句话字面成立。
- **副作用**：随机数消耗顺序变了（dataset worker 的 RNG vs forward 的全局 RNG），
  所以**逐 RNG-trace 的复现是不可能的**。等价性因此定义为「**给定相同决策**，张量逐位相同」，
  这正是 `test_forward_unchanged.py` 现在测的东西。

### 3.3 `<depth><action>` 的语义定性变了（代码行为不变）

`depth+action` 模板的张量与上一版完全相同（`test_forward_unchanged.py` 里有专门一条比对）。
变的是**怎么称呼它**：它是 depth 空间的世界模型，不是 perception。
`video+depth` 才是论文意义上的感知。写作与报告必须区分这两者。

### 3.4 `--variations` 此前被静默丢弃 —— 已修

`CombDataset.__init__` 接收 `variations` 参数，但构造 `RLBenchSelfgenDataset` 的那个 lambda
**没有把它传下去**。`train.py` 传了、`CombDataset` 收了、然后丢了。

后果：任何用 `train_arm.sh`（默认 `VARIATIONS=0`）起的训练，**实际上在全部 variation 上训练**
——也就是在自己的留出测试集上训练。`_apply_variation_filter` 的 docstring 恰好预言了这个失败：
「a "train" run silently trains on its own test set and every held-out number is meaningless」。

- **影响面**：训练集规模从「以为的」468 个 episode 变成实际的 ~2148 个。
- **修复后**：`--variations 0` 真的生效。构造时会打印
  `[selfgen] variations='0': kept N/M episodes across K tasks` —— 起训后请在日志里确认这一行。
- **对既有 checkpoint 的影响**：本次修复之前用 `train_arm.sh` 跑出的任何结果，其
  train/test 划分都不成立，不能作为留出评测的依据。

### 3.5 弃用 `--visual_modality_mix`

旧命令仍能跑（自动翻译成 `<modality>+action` 模板），但与 `--template_mix` 同时给会报错。
迁移表：

| 旧 | 新 |
|---|---|
| `--visual_modality_mix video@1.0` | `--template_mix video+action@1.0` |
| `--visual_modality_mix depth@1.0` | `--template_mix depth+action@1.0`（世界模型）或 `video+depth@1.0`（感知） |
| `--visual_modality_mix video@0.5,depth@0.5` | `--template_mix video+action@0.5,depth+action@0.5` |

### 3.6 `test_forward_unchanged.py` 换了种类，不是被删弱

旧版比对源码文本，并且自己写着「如果你在做 6 段的 `<video><depth><action>`，这个测试**就应该**
失败——那是重新协商基线的正确时机」。那个时机到了。新版比对**行为**，覆盖面反而更大
（旧版只能说明「代码没变」，新版说明「不同段数下每条决策路径的张量都对」）。

---

## 4. 明确**没有**改的东西

- **loss**：`se.sum()/valid.sum()` 逐元素等权，零行改动。
- **codec**：`training/percep/depth_codec.py` / `seg_codec.py` 一行未动（`build_seg_prompt`
  仍在，只是 dataset 改走 `templates.seg_tag`）。
- **相机条件**：Plücker → VAE → 每段按视角复用，语义与上游相同（`cam_emb` 只区分视角、不区分模态）。
- **provenance / P0-1 不变量**：`frame_indices` / `view_indices` / `view_dirs` 照旧，
  感知 GT 仍从产生这些像素的同一窗口、同一视角读。
- **VAE / DiT / pipeline / inference.py**：未改。
- **官方 mask 语义**：单帧变体、三档 mode-mix 的概率与含义逐字保留。
- **`_save_checkpoint`**：仍是上游实现（脚本用 `--save_safetensors False` 压掉 fp32 副本）。

---

## 5. 怎么验证

```bash
cd /workspace/ttdu/ActionImages-Cogen
export PYTHONPATH=/workspace/ttdu/ActionImages-Cogen
conda activate ttd_train

bash scripts/run_tests.sh          # CPU 全套
```

关键几条：

| 检查 | 期望 |
|---|---|
| `tests/test_forward_unchanged.py` | 5 条决策路径全部 `upstream-equivalent`，无 GPU、秒级 |
| `tests/test_prompt_tags.py` | 5 个模板前缀 token 开销 ≤14，最长 prompt 25 token，无 `<unk>` |
| `tests/test_selfgen_dataset.py` | `video+depth` 两路流、`depth+action` 无 RGB、action dropout 同步掉标签 |
| `tests/test_depth_roundtrip_e2e.py` | AbsRel < 0.1%，跨视角负控制 > 30% |
| L2 数值回归 | `--template_mix video+action@1.0 --prompt_tag_style none`，官方 HF 数据 + `--learning_rate 0` + 20 步，与 pristine `/workspace/ttdu/ActionImages` 比 `max\|Δloss\| < 1e-5` |

GPU 冒烟（挑空闲卡；用户会腾出 GPU 4/7）：

```bash
GPUS=4 bash scripts/smoke_arm.sh video+action
GPUS=4 bash scripts/smoke_arm.sh video+depth          # 峰值显存应与上一条相等
GPUS=4 bash scripts/smoke_arm.sh video+depth+action   # 6 段，应明显更高
```

正式臂（阶段 1，warm-start，每臂约 1.4 天 @2 卡 14.5 s/it）：

```bash
GPUS=4,7 bash scripts/train_arm.sh arm0    # 对照，必须跑
GPUS=4,7 bash scripts/train_arm.sh arm1    # 低比例感知
GPUS=4,7 bash scripts/train_arm.sh arm2    # co-supervision（视 arm0/1 结果再定）
```

起训后立刻在日志里确认两行：
`[selfgen] variations='0': kept ...`（§3.4 的修复生效）和
`[selfgen] self-test OK: template=...`（每个模板都构得出样本）。

---

## 6. 明确**没有**做的事

- **eval 脚本移植**：`g0_openloop_rlbench_v2.py` / `g0_perception_zeroshot.py` 仍在 ttd 侧，
  尚未搬进 fork。搬的时候要顺手修 `g0_perception_zeroshot.py:126` 的真 bug
  （用 `sorted(glob(...))` 重新推导视角，与 dataset 的**未排序** glob 顺序不一致）——
  本 fork 的 provenance 字段让这类错位不可能再发生，直接用它。
  **在此之前没有任何 r_peak / AbsRel / IoU 读数**，也就没有零样本基线。
- **阶段 2 from-Wan-base**：脚本已支持（`INIT_CKPT=`），但按计划要等代码冻结、阶段 1 有结论后再起
  （125k 步 ≈ 21 天/臂 @2 卡）。
- **batch > 1 的跨模板混批**：不同模板段数不同，无法 stack。现在是显式报错而非静默出错。
  真要放开，需要按模板分桶采样。
- **8 段全拼**（`video+depth+segmentation+action`）：`parse_template` 接受它，但 2× 序列 / 4× attention，
  并且论文的平衡机制是混合比例而不是拼接——不建议进菜单。
- **bridge / droid / 混合数据**：只用 `rlbench_selfgen@1.0`。两者没有 depth/mask GT，
  模板系统会把它们当作 `video+action`（无 `template` 字段 → forward 回退默认）。
- **cache 路径**：仍无 `cached_dataset.py` / `--use_cache`。感知 target 本来就不缓存（D-042）。

---

## 附:A 轴与 arm7(2026-09-04)

### A 轴 —— `--action_mask_mix`

`--perception_mask_mix`(M 轴)只管**不含** `<action>` 的模板。含 action 的模板一直用
`train.py` 上游那三个硬编码常量,没有开关。arm7 的菜单四个模板全含 action,于是 M 轴一个字节
都不起作用 —— 这条轴就是它的补集。

| 预设 | (iiii, fiii, fifi, policy) | 用途 |
|---|---|---|
| `A0` | `(0.81, 0.045, 0.045, 0.10)` | **默认**,从上游常量推导,arm0–arm6 与 `step125750` 都是它 |
| `A1` | `(0.75, 0.00, 0.05, 0.20)` | arm7 |
| `A2` | `(0.90, 0.05, 0.05, 0.00)` | 90/5/5,无 policy 分支 |

`plan_segments` 把这个边际**反推回上游那两次抽样的阈值**,而不是改成一次四路累计抽样。这是
刻意的:上游就是两次抽样,且第一次(collapse)对所有数据集都会消耗(短路顺序)。改成一次会保住
边际分布却改变同一个种子落到哪个分支 —— 正是 `test_forward_unchanged.py` 存在的意义。默认 A0
下,推导出的三个阈值与 `SINGLE_FRAME_VISUAL_PROB` / `MODE_FIRST_SEGMENT_GIVEN_PROB` /
`MODE_ALL_VISUAL_GIVEN_PROB` **浮点逐位相等**,所以这条轴在默认路径上是彻底的 no-op。

改动文件:`training/templates.py`(常量 + `parse_action_mask_mix` + `plan_segments` 参数)、
`training/args.py`、`train.py`(5 处,与 `perception_mask_mix` 一一对应)。
wandb config 现在同时记录 `action_mask_mix` 与 `action_mask_mix_resolved` ——
`None` 在 config 里读起来像「没有 mask 策略」,而它其实是 A0。

### arm7

```
--template_mix video+action@0.4,depth+action@0.2,segmentation+action@0.2,normal+action@0.2
--action_mask_mix A1  --segmentation_mode scene_roles
```

arm6 有 60% 的样本是 `video+X` 感知模板,**一个 action 段都没有**;算上 action dropout,只有
`0.4 × 0.9 = 36%` 的步产生 action 梯度。arm7 让**每个**模板都带 action(90%),而且成本完全不变
—— 四个模板都还是四段同形状。

**对照组是 arm0 不是 arm6。** `outputs/specialist_action__seed42_fi3_512_aug_sr`
(`video+action@1.0`,同样的 warm start / seed / 树 / interval / lr)有**同样的** 90% action
样本率但只有一个模态。注意 arm7 同时改了菜单和 mask 轴(A1 vs A0),所以 arm7 − arm0 的差异不能
单独归因于其中一个;要归因就得再跑一个 `ACTION_MASK_MIX=A1` 的 action specialist。

**arm7 放弃了什么**:菜单里没有任何 `video+X`,所以这个 checkpoint **不能**被问 RGB→depth/seg/
normal。`eval/eval_perception.py` 和 `eval/eval_all_masks.py` 对它是离分布的。它独有的新读数是
「从哪个模态解动作」,用 `eval/eval_action.py --template <X>+action`。

### 其他

- `eval/eval_action.py` 新增 `--template`(四选一)与 `--segmentation_mode`。段切片
  `q = len(arr)//4`、action 段在 index 1 和 3 对四个模板都成立(段序 view-major、action 在
  视角内最后),但加了 `len(parse_template(...)) == 2` 的守卫,免得六段模板静默切错像素。
  报告文件名和 video 文件名都带上 template,否则四次读数会互相覆盖。
- `scripts/validate_mix.py::mask_mode` 修了一个会静默误报的 bug:`anchor` 原本硬编码成
  `"video"`,对顶替模板过滤后是空集合,`all([])` 为 True,于是**每个** plan 都被读成 `fifi`。
  现在从 plan 里取第一个视觉段。
- `scripts/train_arm.sh` 起训时打印哪条 mask 轴是 INERT。这里**不能**用 `grep -qv`:本机
  `/usr/bin/grep` 是 ugrep 7.5.0,它的 `-q -v` 即使存在不匹配行也返回 1,会把 arm6 的混合菜单
  误报成「所有模板都含 action」。改用纯 bash 计数。
  (同样的坑还在 `scripts/supervise_8h.sh:25`,本次未动。)
- `scripts/smoke_arm.sh` 的默认数据树从 `rlbench_selfgen@1.0` 改为 `rlbench_selfgen_512_aug`,
  分辨率按树推导(与 train_arm.sh 同一段逻辑)。原默认指向的 256 v2 树早已被删,
  `data/rlbench_selfgen` 是**悬空软链** —— 也就是说这个脚本此前根本跑不起来。
  同时新增 `SEG_MODE` / `FRAME_INTERVAL` / `PERCEPTION_MASK_MIX` / `ACTION_MASK_MIX` 环境变量并
  透传,否则 smoke 验的是一个和真实臂不同的配置。
