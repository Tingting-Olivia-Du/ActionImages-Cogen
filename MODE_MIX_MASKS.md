# ActionImages / Cogen mask 与 mode-mix 速查表

本文统一说明官方 ActionImages 与 ActionImages-Cogen 的 segment 布局、mask 语义和随机采样概率。

## 1. Notation

以两个相机视角为例：

| 记号 | 含义 |
|---|---|
| `V0`, `V1` | view 0 / view 1 的完整 video latent segment |
| `D0`, `D1` | view 0 / view 1 的完整 depth latent segment |
| `S0`, `S1` | view 0 / view 1 的完整 segmentation latent segment |
| `A0`, `A1` | view 0 / view 1 的完整 action latent segment |
| `F` | Full：整个 segment 都是干净条件，不计算 loss |
| `I` | Initial：segment 完整存在，但只有第一 latent 帧是条件，其余帧需要预测 |
| `1` | 序列中只保留该 segment 的第一 latent 帧；这一帧是条件 |

`0/1` 是**视角编号**，不是时间下标。`V0[:1]` 表示只保留 `V0` 的第一 latent 帧；Python
切片 `[:1]` 等价于 `[0:1]`，包含下标 0，不包含下标 1，并保留长度为 1 的时间维。

mask 的布尔语义：

| mask | 输入模型时 | loss |
|---|---|---|
| `True` | 使用干净 latent | 不计算 |
| `False` | 使用加噪 latent | 计算去噪 loss |

## 2. 标准布局

官方 `video+action` 与 Cogen 基线采用相同布局：

```text
V0 | A0 | V1 | A1
```

Cogen 的通用顺序是 view-major、modality-minor，模态固定按
`video -> depth -> segmentation -> action` 排列：

| 模板 | 两视角布局 |
|---|---|
| `video` | `V0 | V1` |
| `video+action` | `V0 | A0 | V1 | A1` |
| `video+depth` | `V0 | D0 | V1 | D1` |
| `video+segmentation` | `V0 | S0 | V1 | S1` |
| `depth+action` | `D0 | A0 | D1 | A1` |
| `segmentation+action` | `S0 | A0 | S1 | A1` |
| `normal+action` | `N0 | A0 | N1 | A1` |
| `video+depth+action` | `V0 | D0 | A0 | V1 | D1 | A1` |

最后三个四段模板是 **arm7 的菜单**（与 `video+action` 一起，比例 0.4/0.2/0.2/0.2）。
它们在各自的视觉空间里是世界模型（序列里根本没有 RGB），但**都带 action 段** ——
这正是 arm7 与 arm6 的区别：arm6 的 60% 样本是 `video+X` 感知模板，一个 action 段都没有。

## 3. 官方推理任务

布局均为 `V0 | A0 | V1 | A1`：

| `task_type` | V0 | A0 | V1 | A1 | 任务含义 |
|---|:---:|:---:|:---:|:---:|---|
| `i2va` | I | I | I | I | 从每段初始帧联合生成后续 video 和 action |
| `v2a` | F | I | F | I | 给定完整双视角 video，预测 action |
| `a2v` | I | F | I | F | 给定完整双视角 action，预测 video |

官方推理接口显式支持 `a2v`，但官方训练的随机 mode-mix 中没有严格的 `a2v` 分支。

## 4. 官方训练与 Cogen `video+action`

排除 single-frame 分支后，官方训练和 Cogen 使用同一个随机数 `p`：

| `p` 范围 | 条件概率 | V0 | A0 | V1 | A1 | 含义 |
|---|---:|:---:|:---:|:---:|:---:|---|
| `p < 0.90` | 90% | I | I | I | I | `i2va`：联合生成 video 和 action |
| `0.90 <= p < 0.95` | 5% | F | I | I | I | 给定完整 source-view video，生成 target-view video 和动作 |
| `p >= 0.95` | 5% | F | I | F | I | `v2a`：给定完整双视角 video，预测动作 |

后两个分支的唯一 mask 差异是 `V1`：

| 分支 | 已有完整信息 | 仍需预测 | 主要能力 |
|---|---|---|---|
| `0.90 <= p < 0.95` | `V0` | `A0`, `V1`, `A1` 的后续 | 单视角条件下的跨视角视频与动作联合生成 |
| `p >= 0.95` | `V0`, `V1` | `A0`, `A1` 的后续 | 从完整多视角视频反推动作 |

中间分支连接了 `i2va` 和 `v2a`：条件信息从“只有各段初始帧”，增加为“完整 `V0`”，再增加为
“完整 `V0+V1`”。精确的 `90%/5%/5%` 是官方代码采用的经验性比例；仓库中没有给出理论推导或消融依据。

## 5. RLBench single-frame policy mode

对于保留 action 的 RLBench 样本，先以 10% 概率进入 policy mode：

```text
普通布局：V0      | A0 | V1      | A1
policy：  V0[:1] | A0 | V1[:1] | A1
```

| 模式 | V0 | A0 | V1 | A1 | 含义 |
|---|:---:|:---:|:---:|:---:|---|
| single-frame policy | 1 | I | 1 | I | 视觉段只保留第一 latent 帧，action 段保留完整长度并预测后续 |

命中该分支后不会再抽取上节的 `p`。

## 6. action dropout 与最终概率

官方首先以 90% 概率保留 action；Cogen 将同一决策移到 dataset，参数为
`action_dropout_prob=0.1`，使模板和 prompt 同步去掉 `<action>`。

对于有有效 action 的 RLBench 原始样本，最终概率为：

| 最终模式 | 计算 | 概率 |
|---|---:|---:|
| action dropout，变成 video-only | `0.10` | 10% |
| single-frame policy | `0.90 * 0.10` | 9% |
| `i2va` | `0.90 * 0.90 * 0.90` | 72.9% |
| 只完整给定 `V0` | `0.90 * 0.90 * 0.05` | 4.05% |
| `v2a` | `0.90 * 0.90 * 0.05` | 4.05% |

如果只观察“已经保留 action”的 RLBench 样本，则相应比例为 `10% / 81% / 4.5% / 4.5%`。

以上是 `A0`（默认）。arm7 用 `A1`，同一张表变成 `20% policy / 75% IIII / 0% FIII / 5% FIFI`；
再乘上 10% 的 action dropout，arm7 每个模板的最终分布是
`10% video-only / 18% policy / 67.5% IIII / 0% FIII / 4.5% FIFI`。

## 7. video-only

action 缺失、全零或发生 action dropout 后，布局变为：

```text
V0 | V1
```

| V0 | V1 | 含义 |
|:---:|:---:|---|
| I | I | 每个视角仅第一 latent 帧为条件，预测各自后续视频 |

不能把唯一模态全部设为 `F`，否则所有位置都是条件，整步没有预测目标，loss 会成为 0。

## 8. Cogen perception 模板

不含 action 且包含多个模态时，Cogen 以 90% 概率完整给定主视觉模态，以 10% 概率保留联合生成。

### `video+depth`

布局：`V0 | D0 | V1 | D1`

| 模式 | 概率 | V0 | D0 | V1 | D1 | 含义 |
|---|---:|:---:|:---:|:---:|:---:|---|
| perception | 90% | F | I | F | I | 给定完整 RGB，预测 depth 后续 |
| joint generation | 10% | I | I | I | I | 联合生成 RGB 与 depth 后续 |

### `video+segmentation`

布局：`V0 | S0 | V1 | S1`

| 模式 | 概率 | V0 | S0 | V1 | S1 | 含义 |
|---|---:|:---:|:---:|:---:|:---:|---|
| perception | 90% | F | I | F | I | 给定完整 RGB，预测 segmentation 后续 |
| joint generation | 10% | I | I | I | I | 联合生成 RGB 与 segmentation 后续 |

这里的 10% 避免模型形成“只要 RGB 出现在模板里，完整 RGB 就永远免费提供”的固定假设。

## 9. Cogen 含 action 的通用化

Cogen 把官方规则推广到其他含 action 模板。`anchor` 是模板中 canonical 顺序最靠前的视觉模态：

| 模板 | `anchor` | `p >= 0.95` 时完整给定 |
|---|---|---|
| `video+action` | video | `V0`, `V1` |
| `depth+action` | depth | `D0`, `D1` |
| `segmentation+action` | segmentation | `S0`, `S1` |
| `normal+action` | normal | `N0`, `N1` |
| `video+depth+action` | video | `V0`, `V1`；depth 仍为 `I` |

**注意**：任何按 `anchor == "video"` 硬编码来判定 mask mode 的分析代码，对这三个顶替模板都是错的
—— 过滤后是空集合，`all([]) is True`，于是每个 plan 都被读成 `fifi`。
`scripts/validate_mix.py::mask_mode` 曾有这个 bug，已改为从 plan 里取第一个视觉段。

以六段 `video+depth+action` 为例：

| `p` 范围 | V0 | D0 | A0 | V1 | D1 | A1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `p < 0.90` | I | I | I | I | I | I |
| `0.90 <= p < 0.95` | F | I | I | I | I | I |
| `p >= 0.95` | F | I | I | F | I | I |

## 9.5 两条 mask 轴：M 与 A

两条轴**划分**模板空间，互不重叠，任何一个模板只会命中其中一条：

| 轴 | 命令行 | 适用于 | 四元组含义 |
|---|---|---|---|
| **M** | `--perception_mask_mix` | **不含** `<action>` 的模板（`video+depth` 等） | `(iiii, fiii, fifi, single_frame)` |
| **A** | `--action_mask_mix` | **含** `<action>` 的模板（`video+action`、`depth+action` 等） | `(iiii, fiii, fifi, policy)` |

所以一个菜单如果全部由含 action 的模板组成（arm7），`--perception_mask_mix` **完全失效**；
反之全是感知模板时 `--action_mask_mix` 失效。`scripts/train_arm.sh` 起训时会显式打印哪一条是
INERT，免得日志读起来像两条都在生效。

### A 轴预设

| 预设 | `iiii` | `fiii` | `fifi` | `policy` | 用在哪 |
|---|---:|---:|---:|---:|---|
| `A0` | 0.81 | 0.045 | 0.045 | 0.10 | **默认**。逐位复现上游常量；arm0–arm6 与官方 `step125750` 都是它 |
| `A1` | 0.75 | 0.00 | 0.05 | 0.20 | **arm7**。见下 |
| `A2` | 0.90 | 0.05 | 0.05 | 0.00 | 90/5/5，不要 policy 分支 |

`A0` 不是手写的常数，而是从 §10 的三个上游常量**推导**出来的
（`p_policy = 0.1`，`p_iiii = 0.9 × 0.9`，`p_fiii = 0.9 × 0.05`，`p_fifi = 0.9 × 0.05`），
`plan_segments` 再把它**反推回两次抽样的阈值**。所以默认路径下抽样次数、顺序、阈值与加这条轴之前
逐位相同 —— `tests/test_forward_unchanged.py::test_action_mask_mix_a0_is_the_upstream_default`
在每个阈值边界上逐一比对了这一点。

**保持两次抽样（而不是一次四路累计）是有意的**：上游就是两次，且第一次的 collapse 抽样对
**所有**数据集都会消耗（短路顺序）；改成一次抽样会保住边际分布，却改变同一个种子落到哪个分支。

### 为什么 arm7 用 A1

arm7 的目标是**把 action 做好**，而 `IIII` 下四段里有两段是视觉段，也就是**一半梯度花在
「想象未来的 depth/seg/normal 视频」上**。四种模式按「对 action 的价值」排序：

- `policy`：loss **全部**落在 action 段上，而且条件与闭环部署完全一致（每个视角一帧观测）。
  canvas 是 24 而非 44 个 latent 帧（短 ~45%，attention ~30%），所以它本身还更便宜。
- `IIII`：部署时用的就是它（`eval/policy.py` 传 `fully_given_modalities=[]`），必须占大头。
- `FIFI`：v2a，loss 也全在 action 上，但「完整观测视频」是部署拿不到的条件。留 5% 作离线上界。
- `FIII`：跨视角模式，四者中与 action 质量关系最弱 —— A1 把它的权重全给了 `policy`。

**关于开销，别指望从 s/it 上看出来**：policy 模式确实把 canvas 从 44 帧压到 24 帧，但
`forward` 是**先**把每个 stream 整段 `encode_video`（视觉 ×2 视角、action ×2、camera 的
moment/direction 各 ×2，共 8 次 41 帧编码），**之后**才调 `plan_segments`，`assemble` 只是把
已经算好的 latent 切成 `[:, :, :1]`。所以短 canvas 只省 DiT 的前向/反向，一点也不省 VAE 编码，
净收益被摊薄到实测噪声以内（arm7 20.8 s/it vs arm6 ~21 s/it，看不出差别）。

**要确认 A 轴真的生效，看 config 不要看 s/it**：起训 banner 的
`mask axes: ... action_mask_mix=A1` 那一行，以及 wandb config 里的 `action_mask_mix` /
`action_mask_mix_resolved`（后者是解析后的四元组，正因为 `None` 在 config 里读起来像
「没有策略」才要记录解析值）。

## 10. 常量速查

| 常量 | 值 | 实际作用 |
|---|---:|---|
| `SINGLE_FRAME_VISUAL_PROB` | `0.1` | RLBench 含 action 时，视觉段只保留第一 latent 帧 |
| `MODE_FIRST_SEGMENT_GIVEN_PROB` | `0.9` | 含 action 三路选择的第一个累计阈值；不是“第一段完整给定的概率” |
| `MODE_ALL_VISUAL_GIVEN_PROB` | `0.95` | 含 action 三路选择的第二个累计阈值 |
| `PERCEPTION_VIDEO_GIVEN_PROB` | `0.9` | 不含 action 的多模态模板中，完整给定 `anchor` 的概率 |

前三个含 action 的常量现在只是 `ACTION_MASK_MIX_PRESETS["A0"]` 的**来源**；
真正被 `plan_segments` 比较的阈值来自 `--action_mask_mix`，默认解析成 A0 即这三个数。

相关实现：

- `training/templates.py::plan_segments`：采样 segment 长度和 given/predicted 方案。
- `training/templates.py::assemble`：拼接 latent/camera，并将 `F/I` 转换为布尔 mask。
- 官方 `ActionImages/train.py`：固定 `V0 | A0 | V1 | A1` 布局的原始训练逻辑。
- 官方 `training/wan_video_action_images.py`：推理时的 `i2va/v2a/a2v` 显式 mask。
