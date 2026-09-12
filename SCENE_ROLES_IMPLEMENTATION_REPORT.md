# Scene Role Segmentation —— 方案修订与实现报告

> 日期：2026-08-13
> 仓库：`/workspace/ttdu/ActionImages-Cogen`
> 配套方案：[SEGMENTATION_SCENE_ROLES_PLAN.md](SEGMENTATION_SCENE_ROLES_PLAN.md)（v2）
> 状态：**Phase A + B2 代码完成并通过测试；§3.2 VAE 否决门实测通过；§3.0 闭环门项目决定放行；
> M 轴已定 M2 (90/5/5)。未启动任何训练**

本报告分三部分：一、方案改了什么以及为什么；二、代码改了什么、怎么验证的；三、没做的部分和
下一步。所有数字都标注了实测方法，没有引用未验证的估计。

---

## 第一部分：方案修订

### 1.1 修订轨迹

初稿的结论是"segmentation 学不好是因为标签太稀疏，改成稠密场景角色即可"。评审过程中有四次
实质性转向，每次都由代码或数据核对触发：

| # | 触发 | 修订 |
|---|---|---|
| 1 | 核对 `comparisons_all_segments/` 与 `templates.py` | 分清 in-distribution 列：`video+segmentation` 走 fork 自己的 perception 分支，90% 是 FIFI，不是官方的 90% IIII |
| 2 | 用户指出官方 90% 是 IIII | 确认两者都对但适用范围不同；写成对照表 |
| 3 | 用户指出 IIII 会生成"更后面的场景" | 发现窗口覆盖任务比例在 36%–100% 之间浮动，模型从首帧无法判断节奏 → IIII 的 IoU 混淆了"分割差"和"时刻不对" |
| 4 | 用户指出交付物是闭环成功率 | 核对 `EVAL_PLAN_CLOSEDLOOP.md`，发现 **rollout 用 `template="video+action"`，segmentation 流根本不在序列里** → 全文围绕"辅助流"重构 |

第 4 条是最根本的一次，它改变了方案的价值论证方式。

### 1.2 §0：交付物定位（新增，全文最重要）

`EVAL_PLAN_CLOSEDLOOP.md` §2.5 与 §7 确认闭环 rollout 协议为：

```text
template     = "video+action"      ← 序列里只有 video 和 action
conditioning = i2va（只给首帧）      ← 即 IIII
```

**segmentation 永远不会在 rollout 中被生成或消费。** 因此：

- seg 影响交付物的唯一路径是「更好的辅助监督 → 共享 backbone 表征更好 → 世界模型更准 →
  闭环成功率更高」，一条长而弱的因果链；
- 指标必须分三层：**L1 闭环成功率（唯一去留判据）/ L2 开环 action / L3 seg IoU（诊断）**；
- 曾据此新增 **§3.0 闭环前置门**（arm1 vs arm0）。**2026-08-14 项目决定放行**：辅助流有用是
  本研究的核心假设，不应被降级成一道可否决全项目的工程门；且 `CAMPAIGN_200.md` 的闭环证据
  方向为正。证据边界见方案 §3.0（那是 arm4 vs **官方**，不是 vs arm0，也没隔离出辅助流）。

同时它给 M 轴一个原则性理由：rollout 跑在 IIII + 生成式 régime，而当前辅助任务 90% 训练在
FIFI 判别式 régime，**训练与部署 régime 不匹配，表征迁移路径很弱**。

### 1.3 §1.2：IIII 的时间错位混淆（新增）

窗口为 `num_frames=41 × frame_interval=3` = 121 原生步，而 episode 长度实测 71–334：

| task | T | 窗口覆盖比例 | 重复末帧 |
|---|---:|---:|---:|
| push_buttons | 71 | 100% | 16 |
| open_drawer | 96 | 100% | 8 |
| close_jar | 157 | 77% | 0 |
| put_item_in_drawer | 291 | 42% | 0 |
| stack_blocks | 334 | 36% | 0 |

从首帧无法判断节奏，模型只能回归平均值，长 episode 上系统性超前。IoU 对时间错位是**断崖式**
敏感的，位置误差是平滑的——这解释了为什么 seg 的 IIII 数字（0.02）最不可信，而 action 的
IIII 数字仍有参考价值。**FIFI 干净是因为整段 RGB 给全后时间演化被钉死。**

### 1.4 修正的四处技术错误

| 错误 | 修正 | 依据 |
|---|---|---|
| palette 把最小色距 98.3 分配给 `distractor`/`tool` | `tool` ↔ `unknown` 换色 | 实测两两距离；`TAU=40` 正按 98.3 标定，且两者在 sweep_to_dustpan 必然共现 |
| `decode` 对全部 9 色做 nearest | 限制在本 episode 实际出现的角色子集 | 现有 `decode_known_color` 已有此不变量并注明理由 |
| `instruction_roles` 存进新 JSON | 只存 instruction 无关的 `base_role`；target 仍由 `build_referring_spec` 运行时解析 | 避免两个真相源；自动继承 `dropped_groups` 统计 |
| 要求重生成 2818 个 episode | 降级为硬失败校验 | 实测 `handles.json` 覆盖率 >99% |

### 1.5 补上的实验设计缺口

- **样本量**：≥20 held-out episode（variation1/2，共 478 个可用）× 3 seed，报 95% CI；
- **arm 编排**：S0/S1b/S1/S2 同批跑，不把 S2 挂在 S1 后面；新增零标注成本的 S1b；
- **M 轴先于 S 轴**：改一个参数 vs 16 个 resolver；
- **预先声明的失败准则**，含最可能的失败模式："L3 涨、L1 不动"→ 二级假设被否定。

---

## 第二部分：代码实现

### 2.1 改动清单

| 文件 | 性质 | 内容 |
|---|---|---|
| `training/percep/seg_codec.py` | 增量 | scene-role palette、`build_role_lut`、`encode_scene_roles`、`scene_role_labels`、`decode_scene_roles`、`build_scene_prompt`、`scene_role_iou` |
| `training/percep/__init__.py` | 增量 | 导出上述符号 |
| `training/templates.py` | 增量 | `scene_seg_tag`、`SCENE_SEG_PROTOCOL`、`prompt_prefix(segmentation_mode=)`、`PERCEPTION_MASK_MIX_*`、`parse_perception_mask_mix`、`plan_segments(perception_mask_mix=)`、single-frame perception 模式 |
| `training/args.py` | 增量 | `--segmentation_mode`、`--perception_mask_mix` |
| `training/dataset/base.py` | 增量 | `segmentation_mode` 透传 |
| `training/dataset/rlbench_selfgen.py` | 增量 | `_load_scene_segments`、`_scene_role_lut`、编码分支、prompt 分支 |
| `train.py` | 增量 | 参数解析与透传；wandb config 记录 |
| `ttd/src/percep/scene_segments_gen.py` | **新增** | 16 任务角色解析器 + 覆盖率校验 |
| `scripts/seg_pixel_stats.py` | **新增** | 逐任务像素占比统计（§4） |
| `tests/test_scene_seg_codec.py` | **新增** | 10 项 codec 验收 |
| `tests/test_scene_seg_dataset.py` | **新增** | 9 项真实 episode 集成验收 |
| 数据树 2818 × `scene_segments.json` | **新增** | 由生成器写入 |
| `scripts/train_arm.sh` | 增量 | `PERCEPTION_MASK_MIX` 环境变量，**新 arm 默认 M2** |
| `scripts/vae_roundtrip_seg.py` | **新增** | §3.2 否决门诊断（含 RGB PSNR 对照） |

**库层改动全部是增量的**：`--segmentation_mode` 默认 `referring`，`templates.py` 的
`PERCEPTION_MASK_MIX_DEFAULT` 保持 M0，两者都复现改动前的精确行为。

**实验层的默认另算**：`scripts/train_arm.sh` 的 `PERCEPTION_MASK_MIX` 默认 **M2 (90/5/5)**
（2026-08-14 决定，见 2.11）。这是刻意的两层划分——库默认负责可复现性，启动器负责当前实验
选择，与 `FRAME_INTERVAL`（脚本 3 / `args.py` 1）同构。

### 2.2 关键设计决定

**palette 换色（实测驱动）**。九个角色的两两 sRGB 距离，最紧的五对：

```
 98.3  distractor  vs unknown     ← 换色后
109.2  tool        vs fixture
109.4  target      vs unknown
135.2  target      vs fixture
136.4  robot_arm   vs gripper
```

换色前 98.3 落在 `distractor`/`tool` 上——两个语义相近且必然共现的前景类。换色后落在
`distractor`/`unknown` 上，而 `unknown` 在合法训练数据里出现次数必须为 0，代价为零。
`tests/test_scene_seg_codec.py::test_tightest_pair_is_not_two_cooccurring_foreground_roles`
把这条不变量钉死了，未来任何重排 palette 的改动都会触发它。

**未映射 handle → `unknown` 而非 background**。`build_role_lut` 把 LUT 初始化为
`UNKNOWN_LABEL`。标注缺口必须可见；折进占 85% 的背景类里等于永远不会被发现。

**单一真相源**。`scene_segments.json` 只存 `base_role`；`target` 在
`_scene_role_lut` 里由 `build_referring_spec` 运行时覆盖上去。角色优先级（§6.3）因此就是
组合顺序，不需要单独的优先级表——实际只有 target 这一级会冲突。

**M 轴的分支顺序是 load-bearing 的**（见 2.4 的回归）。

### 2.3 新增的 single-frame perception 模式

```text
IIII              : V0(full,给首帧) | S0(full,给首帧) | V1(full,给首帧) | S1(full,给首帧)
single-frame perc : V0[:1]          | S0(full,给首帧) | V1[:1]          | S1(full,给首帧)
```

它与 IIII 语义不同：IIII 要同时生成 RGB 和 seg 视频，此模式下 RGB 段被截掉，模型**只**生成
seg 视频。是"模型是否需要先渲染 RGB 才能产出 seg"的干净消融。

**同时验证了方案预测的崩溃**：把 perception 模板原样路由进 `has_action` 分支会让四个段全被
截成 1 帧，每段唯一那帧又都是 condition，触发 `assemble` 的
`assert not bool(masks.all())`。实测确认：

```
single-frame perception: seq_len 24  cond 4  pred 20   OK
naive all-visual collapse -> AssertionError: every latent position is a condition frame
```

### 2.4 一个真实回归及其修复（重要）

初版 M 轴实现按 `(iiii, fiii, fifi, single)` 的元组顺序依次判断累积概率。**边缘分布完全正确**
（M0 实测 90.1% FIFI / 9.9% IIII），但它把随机值→结果的映射反了：历史代码是
`if rng.random() < 0.9: FIFI`，低值 → FIFI；新代码低值 → IIII。

`tests/test_forward_unchanged.py::test_perception_template_gives_rgb_whole` 用
`ScriptedRandom([0.0])` 钉死了这个映射，立刻失败。这不是测试过时——**同一个 seed 会训练出
不同的 mask 序列**，正是该测试存在的意义。

修复是把 FIFI 分支放在第一位判断，而不是修改测试：

```python
# BRANCH ORDER IS LOAD-BEARING, and is not the tuple order. ...
if p < p_fifi:                    given_modalities = {anchor}   # FIFI
elif p < p_fifi + p_iiii:         pass                          # IIII
elif p < p_fifi + p_iiii + p_fiii: give_first_segment = True    # FIII
else:                             single_frame_anchor = True
```

修复后 `test_forward_unchanged` 全绿，且分布不变。

### 2.5 第二个真实 bug：target 覆盖吞掉 goal/tool（影响 9/16 任务）

初版 `_scene_role_lut` 把 `seg_targets.json` 里**所有**被指令引用的 handle 都覆盖成 `target`。
测试全绿，但抽查 `place_shape_in_shape_sorter` 的 `present_roles` 时发现少了 `goal`，追查后
确认是设计错误。

`seg_targets.json` 列的是"指令**提到**的所有物体"，不是"被操作的物体"。实测 **9/16 任务列了
两到三个**：

```
put_item_in_drawer   ['item', 'bottom drawer']            ← item 是 target，drawer 是 goal
sweep_to_dustpan     ['broom', 'dirt', 'dustpan']         ← tool / target / goal 三个都有
reach_and_drag       ['cube', 'stick', 'target']          ← target / tool / goal
```

全覆盖成 `target` 会把 receptacle 和 implement 一起涂红，**恰好摧毁本协议存在的理由**。
`sweep_to_dustpan` 会同时失去 broom(tool) 和 dustpan(goal)。

修复是按两个来源各自知道的信息组合，而不是让一个无条件覆盖另一个：

- `seg_targets.json` 知道"指令提到了这个物体"（并解析 episode-random 颜色）；
- `scene_segments.json` 知道"这个物体是容器 / 工具 / 装置"；
- **只有 base_role 为 `distractor`（可操作物体）的被引用 handle 才提升为 `target`**；
  被引用的 goal 仍是 goal。

这才是 §6.3 `target > goal > tool > fixture` 链的正确读法：链条用于消解一个物体同时符合多个
角色时的歧义，不授权用较不具体的角色覆盖较具体的角色。

连带修正了 `open_drawer`：三个抽屉盒原本全标 `fixture`，导致该任务解析不出任何 target。它们
在 open_drawer 里是**被操作对象**（指令选其一），应为 `distractor`；而在 `put_item_in_drawer`
里同样的几何体是**目的地**，应为 `goal`。同样的物体、不同的角色、由任务决定 —— 这正是角色表
必须逐任务写而不能共用一张 name→role 映射的原因。

新增 `test_every_task_resolves_a_target_and_keeps_goal_tool_distinct` 覆盖全部 16 个任务，
双向断言：每个任务都解析出 target，且没有任务因覆盖而丢失 goal/tool。实测全绿。

### 2.6 元数据生成结果

`scene_segments_gen.py` 对 16 个任务写了显式的 name→role 规则，不用 substring 猜测
（`seg_targets_gen.py` 的审计已证明同名前缀可能表示不同实例）。

全树 2818 episodes 实测：

```
handles per base_role:  robot_arm 22035 | background 11767 | gripper 8454
                        distractor 8242 | goal 3085 | fixture 2532 | tool 457
UNMAPPED object names: 0 distinct
SCENE_SEGMENTS_COVERAGE_OK
```

`--verify-masks`（打开每个 `mask.npz` 检查每个**实际渲染**的 handle 都被映射）在 400 episodes
× 4 视角上同样零缺口。

过程中发现并解决的唯一缺口：`reach_and_drag` 的 `target0` 在 2818 个 episode 中只有 14 个出现。
核对 `seg_targets.json` 后确认它是拖拽目的地（`goal`），出现率低是因为该任务已记录的 ~94%
遮挡率，不是覆盖 bug。

### 2.7 稀疏度实测（`scripts/seg_pixel_stats.py`）

16 任务 × 4 episode × 2 视角，输出 `reports/seg_pixel_stats.csv`：

```
referred_target（今天的监督）  : 中位 0.97%，min 0.22% (close_jar)，max 28.96% (put_groceries)
non-background（scene_roles） : 中位 12.19%，min 9.74%，max 39.68%
```

数据集层面的端到端测量（`test_scene_roles_is_denser_than_referring`，64×64、真实 episode）：
**referring 0.13% → scene_roles 31.91%，249×**。

注意这两个数不矛盾也不该混用：前者是全树逐任务中位数，后者是单个 open_drawer 样本在模型分辨率
下的非黑像素比——open_drawer 的抽屉部件占 22%，属于前景最多的一档。

跨任务方差同样有两个口径，报告时必须写明用的哪个：

- 按"所有任务物体"（不含机器人）：`reach_and_drag` 1.05% vs `put_groceries_in_cupboard`
  33.33%，约 **32×**；
- 按"非 background"（含机器人）：**4.1×** —— 机器人稳定占 ~9%，把比值拉平了。

方案 §8.4 要求逐样本归一化权重而不是按任务硬编码，依据是前一个口径。

### 2.8 测试结果

**新增测试全绿：**

```
tests/test_scene_seg_codec.py    10 项 → SCENE_SEG_CODEC_OK
tests/test_scene_seg_dataset.py   9 项 → SCENE_SEG_DATASET_OK
```

codec 侧覆盖：palette 形状与唯一性、最紧色对不变量、未映射→unknown、**无 VAE roundtrip
每类 IoU = 1.0**、1 像素目标不被删除、子集 decode 不被缺席角色抢像素、background 强制参与、
unknown 可检测、nearest resize 不产生插值颜色、prompt tag 与 referring 可区分。

dataset 侧覆盖：**referring 模式逐位一致**（6/6 样本 text 与像素完全相同）、流形状与值域、
prompt tag、每个像素都是合法 palette 色、**训练数据零 unknown 像素**、target 覆盖 base_role、
跨视角/跨时间角色稳定、稠密度提升、**全部 16 任务都解析出 target 且不丢失 goal/tool**。

**既有测试套件**（`export PYTHONPATH=/workspace/ttdu/ActionImages-Cogen` 后全量跑）：

| 测试 | 结果 | 说明 |
|---|---|---|
| test_aggregate_closedloop | PASS | |
| test_checkpoint_pruning | PASS | |
| test_decode_6dof | PASS | |
| test_depth_codec | PASS | |
| test_depth_roundtrip_e2e | PASS | GPU VAE 往返 |
| test_eval_assembly | PASS | |
| **test_forward_unchanged** | **PASS** | **一度真实回归，已按 2.4 修复** |
| test_frame_interval | PASS | |
| test_policy_conditioning | PASS | |
| test_prompt_tags | PASS | |
| test_seg_codec | PASS | |
| test_scene_seg_codec | PASS | 新增 |
| test_selfgen_alignment | PASS | |
| test_scene_seg_dataset | PASS | 新增 |
| **test_new_episodes** | **PASS（放宽超时后）** | 首轮 400s 超时被杀，非失败。见下方输出 |
| **test_selfgen_dataset** | **PASS（放宽超时后）** | 同上，首轮 400s 超时被杀。`ALL_SELFGEN_DATASET_TESTS_PASSED` |
| test_rollout_env | FAIL（环境） | `COPPELIASIM_ROOT` 未设，需 `source ttd/scripts/env_eval.rc`；与本次改动无关 |

**最终结果：17 项中 16 项 PASS，唯一的 FAIL 是环境问题。**

`test_rollout_env` 在启动 CoppeliaSim 之前就因 `COPPELIASIM_ROOT` 未设而退出，与本次改动无关。

`test_new_episodes` 与 `test_selfgen_dataset` 首轮显示 FAIL，实为被我设的 400s 超时杀掉——两者
都只打印了 import 警告，没有产生任何断言结果。放宽超时后单独重跑**均全绿**。

其中 `test_new_episodes` 正是覆盖 referring seg 路径的那个测试，所以这条结果直接支撑
"旧协议未被破坏"：

```
DEPTH_OK   alignment + roundtrip on 16 eps, worst AbsRel=0.0985%
SEG_OK     alignment + roundtrip on 16 eps, worst IoU=1.0000
           (44 instance-views scored, 8 not visible -> skipped)
SEG_COVERAGE_OK  1398/1398 variation0 episodes
ALL_NEW_EPISODE_TESTS_PASSED
```

`test_selfgen_dataset` 同样全绿，覆盖模板混合、prompt tag 一致性、双视角编码与 action dropout：

```
PERCEPTION_TEMPLATE_OK / COSUPERVISION_OK / SUBSTITUTION_TEMPLATE_OK
BOTH_VIEWS_ENCODED_OK (same draw: views=[3, 1] frames=55..95)
MIXED_MIX_OK {'video+depth': 20, 'video+segmentation': 22, 'video+action': 18}
ACTION_DROPOUT_OK / MIX_VALIDATION_OK
ALL_SELFGEN_DATASET_TESTS_PASSED
```

**首轮曾出现的 `ModuleNotFoundError: No module named 'training.percep'`** 是 `PYTHONPATH` 未设
导致的 editable-install 遮蔽（`train.py` 的 `assert_own_training_package` 专门记录了这个陷阱），
设置后即消失，与本次改动无关。

### 2.9 §3.2 VAE roundtrip 诊断（已执行）

新增 `scripts/vae_roundtrip_seg.py`，实现方案 §3.2 的硬性否决门：把 GT segmentation 视频过一遍
VAE encode→decode，**不经任何模型**，测每类 IoU。这是任何训练模型的能力上界。

**两个实现细节是必须的，都不是可选优化：**

1. **逐视角编码。** Wan VAE 时间压缩 4×，`T_lat = 1 + (T-1)/4`，只有 `T ≡ 1 (mod 4)` 才整除。
   `num_frames=41` 每视角满足，但拼接后的 82 帧不满足——整段喂进去会静默返回 81 帧。脚本按
   `train.py` 的做法逐视角编码。

2. **RGB 对照（PSNR）。** 见下方 2.10，这是本次最重要的一条教训。

**结果（48 episodes / 14 tasks / 256² / 41 帧，完整输出见 `reports/vae_roundtrip_seg.txt`）：**

```
--- sanity control ---
RGB PSNR            median=29.59 dB   min=28.12 dB        ← VAE 确实在工作

--- referring（今天的协议）---
target instances    n=82  median=0.9910  mean=0.9496  p10=0.9523  min=0.4957

--- scene_roles（提议的协议）---
background  0.9987 | robot_arm 0.9921 | fixture 0.9912 | distractor 0.9903
goal        0.9801 | target    0.9646 | gripper   0.9494 | tool      0.8816   (median)

--- referring target 按 GT 面积分桶（小目标才是这道门要回答的问题）---
<0.25%     n=33  median=0.9776  mean=0.8885  p10=0.5361  min=0.4957
0.25-1%    n=21  median=0.9930  mean=0.9904  p10=0.9872  min=0.9514
1-5%       n=21  median=0.9945  mean=0.9894  p10=0.9769  min=0.9756
>5%        n= 7  median=0.9963  mean=0.9958  p10=0.9938  min=0.9920
```

**判定：通过。** `target` roundtrip IoU 中位 **0.9910**，远高于 §3.2 的 0.8 阈值。

**这条结果否掉了方案 §1.3 一级假设的后半句**——"经 VAE 8× 空间 / 4× 时间压缩后信号进一步
衰减"。在 256² 下，中位大小 1% 的目标过 VAE 几乎无损。segmentation 没学好的原因**不在 VAE**。

**但有两个必须记录的细节，都不构成否决：**

1. **最小目标桶有长尾。** `<0.25%` 桶的中位数 0.9776 很好，但 **p10 = 0.5361、min = 0.4957**
   ——约 10% 的极小目标掉到 0.5 附近。逐任务看，最差值全部来自 `sweep_to_dustpan`
   （min 0.4957），也就是 dirt：5 个 4–8 px 的 handle。这正是 `seg_codec.py` docstring 里
   早就记录过的那类物体。**结论：VAE 对绝大多数目标无损，但对个位数像素的目标确实会吃掉一半。**
   报告 seg 指标时应把 `<0.25%` 桶单列，不要让它被中位数掩盖。

2. **dense 协议反而略微降低了 target 的 roundtrip 上界。** 同一批 episode：
   referring `target` 中位 **0.9910** / p10 0.9523，scene_roles `target` 中位 **0.9646** /
   p10 0.8570。原因是颜色邻居变近了——referring 里 target 只跟黑色竞争（色距 >230），
   scene_roles 里它周围是 goal/fixture/distractor（最近色距 98–135）。
   这是 dense 表示的一项**实测代价**，方案初稿只论证了收益、没预料到这一项。幅度不大
   （2.6 个百分点），但它是所有 scene_roles 结果的上界，必须在报告 S1/S2 时一并给出。

覆盖率说明：均匀取样只落到 14/16 个任务（`put_groceries_in_cupboard`、`stack_blocks` 未被
抽中，因为各任务 episode 数不等）。两者的 target 面积分别是 28.96% 和 1.12%，都不在最小桶，
不影响上面的结论。

### 2.10 一次假否决：破损的 VAE 看起来和"方案该毙掉"一模一样

首次运行报告 `target` roundtrip IoU = **0.028**，落在 §3.2 的否决区（< 0.5），结论是
"VAE 正在摧毁目标信号，应暂停 Phase B/C"。

**这个结论是错的。** 破绽在同一张表里：`background` 占约 85% 的像素，IoU 却只有 **0.0018**。
一个连大面积背景都重建不出来的 VAE 对 RGB 也会完全不可用——物理上不成立。

根因是我自己的加载代码：

```python
vae.load_state_dict(sd, strict=False)   # 键名未经 converter，全部不匹配
except Exception: pass                   # 失败被吞掉
```

`strict=False` 加上裸 `except: pass`，让**随机初始化的 VAE** 一路跑到了结论。正确路径是
`WanVideoVAE38.state_dict_converter().from_civitai(sd)` 再 `load_state_dict(..., strict=True)`。

修复后同一批数据：**RGB PSNR ≈ 30 dB，`target` roundtrip IoU = 0.998**——与假结论完全相反。

**已加入两道防线，防止同类错误再次冒充结论：**

- `load_state_dict(..., strict=True)`，且移除了 try/except：部分加载是 bug，不是警告；
- **RGB PSNR 对照**：若 RGB roundtrip < 15 dB，脚本直接 `ABORT` 并声明上面所有 IoU 无意义，
  而不是给出否决判词。一个随机 VAE 对**每一类**都给近零 IoU，读起来和真正的否决一模一样；
  唯一能区分两者的就是这个对照。

这条教训的普遍形式：**加载失败必须表现为崩溃，不能表现为发现。** 任何"全线极差"的诊断结果，
第一反应应该是怀疑仪器，而不是宣布结论。

### 2.11 两项项目决定（2026-08-14）及其代码后果

#### 决定 1：§3.0 闭环前置门放行

理由：**"辅助感知流提升世界模型/策略"是本研究的核心假设，不是待检验的中间结论**；把它降级
成一道可能否掉全项目的工程门，是把研究问题当成风险来管。另有 `CAMPAIGN_200.md` 的闭环证据
方向为正（arm4 31/100 vs 官方 20/100，`close_drawer` +40 pp p=0.0078、`push_buttons` +35 pp
p=0.0391，合计 p=0.052）。

**但这条证据的边界已如实写进方案 §3.0，避免日后被当成更强的东西引用：**

- 那是 **arm4 vs 官方 checkpoint**，不是 arm4 vs arm0。`CAMPAIGN_200.md` §2.2 自己写明
  "显著性说明的是'领域内微调有效'，不是 arm4 比官方模型强"——arm4 in-domain、官方 zero-shot，
  分辨率与 cfg 也不同，**没有隔离出"辅助流"这一个变量**；
- 那条辅助流是 **depth** 不是 segmentation，迁移到 seg 属类比；
- `outputs/arm0__seed42_fi3` 已存在，真正能隔离辅助流的 arm0 vs arm4 配对闭环比较成本很低，
  将来需要单独证据时跑它即可。

代码无改动，仅方案与本报告的状态标记更新。

#### 决定 2：perception 训练改为 90% IIII（M2）

即之前讨论的 M2 = (IIII 90, FIII 5, FIFI 5)，与官方 `video+action` 分支逐字对齐。

**落地是刻意的两层：**

| 层 | 值 | 理由 |
|---|---|---|
| `templates.py` `PERCEPTION_MASK_MIX_DEFAULT` | **M0** (10/0/90) | `test_forward_unchanged` 钉住的历史行为，必须继续意味着"旧行为可复现" |
| `scripts/train_arm.sh` `PERCEPTION_MASK_MIX` | **M2** (90/5/5) | 实验选择落在命令行 + wandb config，可追溯 |

与 `FRAME_INTERVAL`（脚本 3 / `args.py` 1）同构。复现旧 arm：`PERCEPTION_MASK_MIX=M0 bash scripts/train_arm.sh ...`

**实测验证：**

```
library default (templates.py) = (0.1, 0.0, 0.9, 0.0)      ← 仍是 M0
  library default        FIFI=90.1%  IIII= 9.9%
  launcher default M2    IIII=89.7%  FIII= 4.9%  FIFI= 5.4%
  video+action mix=None  IIII=81.2%  1I1I=10.0%  FIII=4.4%  FIFI=4.4%
  video+action mix=M2    IIII=81.2%  1I1I=10.0%  FIII=4.4%  FIFI=4.4%   ← 完全不受影响
train_arm.sh: 默认 -> M2 ; PERCEPTION_MASK_MIX=M0 覆盖 -> M0
test_forward_unchanged / test_prompt_tags / test_scene_seg_codec  全部 PASS
```

**必须记住的后果（已写进方案 §11）：** FIFI-seg 的训练曝光量从约 18% 降到约 2%，所以
**评估重心必须从 FIFI 移到 IIII + aligned IoU**。FIFI 仍要报，但作为"无时间混淆的上界探针"，
并标注"训练暴露量仅 5%，数值偏低属预期，不代表能力退化"。**不要因为 FIFI 数字下降就判定
M2 失败**——那是确定结果，不是失败信号；M 轴的判据在 L1。

---

## 第三部分：没做的部分与下一步

### 3.1 明确未实现

| 项 | 原因 |
|---|---|
| ~~**§3.0 闭环前置门**~~ | ✅ **已放行**（2026-08-14 项目决定）。辅助流有用是本研究的核心假设，不作为可否决全项目的门；已有 `CAMPAIGN_200.md` 闭环证据方向为正。证据边界见方案 §3.0 |
| **§8.4 loss 空间权重（Phase C）** | 方案要求先跑 S1 看是否需要；且链路（collator key → forward → `assemble` offset 对齐）改动面大，不应与表示改动同批引入 |
| ~~**§3.2 VAE roundtrip 诊断**~~ | ✅ **已完成，通过**（见 2.9/2.10）。`scripts/vae_roundtrip_seg.py` |
| **§3.4 aligned IoU** | 需要复用已生成的 IIII 输出，属评估脚本，未写 |
| **§11 eval 侧 per-role 指标** | `eval/eval_perception.py` 仍只算 referring 的 per-instance IoU |
| **golden tensor 固化（§8.1）** | 用等价方式替代了：`test_referring_mode_is_bit_identical` 直接比对同一份代码在默认/显式参数下的输出。真正的跨版本 golden 需要在改动前采集，已错过时机 |

### 3.2 需要人工确认的判断

`scene_segments_gen.py` 里 16 个任务的 `base_role` 是我从物体名和任务语义推断的，**方案 §14
Phase A 要求逐任务人工确认 contact sheet，这一步没有做**。几个我认为最需要复核的：

- `put_item_in_drawer`：三个抽屉都标 `goal`，但只有指令指定的那个才是真 goal，其余更接近
  `fixture`。当前实现不区分"非目标抽屉"；
- `close_jar`：`jar0`/`jar1` 都标 `goal`，同上；
- `insert_onto_square_peg`：三根 `pillar` 都标 `goal`；
- `put_money_in_safe`：`safe_door` 标 `fixture`、`safe_body` 标 `goal`，可争议；
- `light_bulb_in`：seg_targets 的 "bulb" 组同时含 `light_bulb0`(distractor→target) 与
  `bulb0`(goal)，覆盖后灯泡红、灯座绿 —— 抽查确认语义正确，但属于"一个引用组跨两个角色"的
  特殊情形，值得留意。

这类"同类多实例，指令选其一"的情形，当前靠 `target` 覆盖处理，未被指令选中的同类物体保持
`goal`/`distractor`。方案 §6.3 允许这样，但值得人工过一遍 contact sheet。

### 3.3 建议的下一步顺序

1. ~~**§3.2 VAE per-class roundtrip**~~ ✅ **已完成，通过**（target 中位 0.9910，RGB 对照
   29.6 dB）。VAE 不是瓶颈，一级假设中"VAE 压缩衰减"那半句被否掉；
2. ~~**§3.0 闭环前置门**~~ ✅ **已放行**（项目决定，见方案 §3.0）；
3. ~~**M 轴扫描**~~ ✅ **已定 M2 (90/5/5)**，`scripts/train_arm.sh` 默认；
4. **两道门都已放行**，现在的瓶颈是 Phase A 的人工 contact sheet 确认（§3.2 本节）与 S 轴训练。

### 3.4 复现命令

```bash
cd /workspace/ttdu/ActionImages-Cogen
export PYTHONPATH=/workspace/ttdu/ActionImages-Cogen        # 必须，否则 editable install 遮蔽
conda activate ttd_train

# 元数据（默认 dry run；--write 才落盘；--verify-masks 打开每个 mask.npz 校验）
python /workspace/ttdu/ttd/src/percep/scene_segments_gen.py --write
python /workspace/ttdu/ttd/src/percep/scene_segments_gen.py --verify-masks --limit 400

# 稀疏度统计
python scripts/seg_pixel_stats.py --episodes-per-task 4 --max-frames 20

# 测试
python tests/test_scene_seg_codec.py
python tests/test_scene_seg_dataset.py
python tests/test_forward_unchanged.py

# VAE 否决门（§3.2）
python scripts/vae_roundtrip_seg.py --episodes 48 --num-frames 41

# 训练。新 arm 默认已是 M2 (90% IIII)，无需显式指定
bash scripts/train_arm.sh "video+action@0.6,video+depth@0.2,video+segmentation@0.2"        # S0, M2
EXTRA_ARGS="--segmentation_mode scene_roles" \
  bash scripts/train_arm.sh "video+action@0.4,video+depth@0.2,video+segmentation@0.4"      # S1, M2
PERCEPTION_MASK_MIX=M0 bash scripts/train_arm.sh ...                                       # 复现旧 arm
```

---

## 附：一句话结论

代码层面 scene_roles 已可用且对现有流程零影响（默认参数逐位一致，全树 2818 episode 零标注
缺口，19 项新测试全绿，既有套件 17 项中 16 项 PASS、唯一 FAIL 是缺 CoppeliaSim 环境变量）。但方案本身的价值取决于一个尚未检验的前提：**辅助流对闭环成功率有
贡献**。在 §3.0 的前置门给出结果之前，不建议投入 Phase A 的人工确认和 Phase C 的 loss 改造。
