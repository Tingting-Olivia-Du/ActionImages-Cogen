# RLBench 官方数据与 Selfgen 数据的采样频率对比

本文对比以下两个数据加载器在视频、动作、相机参数及感知模态上的采样方式：

- `training/dataset/rlbench.py`：官方发布的 RLBench 多视角数据。
- `training/dataset/rlbench_selfgen.py`：项目自行生成的 RLBench 数据。

核心结论是：两者即使都向模型提供 41 帧，其覆盖的真实时间范围也不同。

| 数据集 | 磁盘视频频率 | 磁盘动作频率 | loader 中的动作处理 | `frame_interval=1` 时的有效频率 | 41 帧的严格时间跨度 |
|---|---:|---:|---|---:|---:|
| 官方 RLBench | 约 5 Hz | 20 Hz | `actions[::4]` | 约 5 Hz | 8.0 秒 |
| RLBench Selfgen | 20 Hz | 20 Hz | 不降采样 | 20 Hz | 2.0 秒 |

因此，相同的 41 个模型时间点在官方数据中覆盖约 8 秒，在 selfgen 数据中只覆盖约 2 秒，时间跨度相差 4 倍。

## 1. 基本概念

### 1.1 控制频率

RLBench 的原始控制或 observation 频率为 20 Hz，即每秒产生 20 条记录。相邻记录的时间间隔为：

```text
1 / 20 = 0.05 秒
```

### 1.2 帧数与时间间隔数

`N` 个采样点之间只有 `N - 1` 个时间间隔。例如 41 帧的时间戳为：

```text
t0, t1, t2, ..., t40
```

从第一帧到最后一帧共有 40 个间隔。因此，严格的时间戳跨度应按下面的公式计算：

```text
时间跨度 = (帧数 - 1) × 相邻帧时间间隔
```

如果按“每个采样点占据一个时间槽”估算序列容量，则可能使用 `帧数 × 时间间隔`。两种口径都可以出现，但含义不同，不能混用。

## 2. 官方 `RLBenchMVDataset` 的采样方式

官方 loader 在 `training/dataset/rlbench.py` 中加载动作：

```python
def get_8d_action(self, episode_path, frame_indices=None):
    actions = self._load_8d_actions_base(episode_path)
    actions = actions[::4]
    if frame_indices is not None:
        actions = actions[frame_indices]
    return actions
```

磁盘上的 `actions.npy` 保存原始 20 Hz 动作，但官方发布的视频已经约按 4 倍降频。因此需要先执行：

```python
actions = actions[::4]
```

把动作从 20 Hz 降到约 5 Hz，以便与视频帧对齐：

```text
20 Hz / 4 = 5 Hz
```

### 2.1 `[::4]` 的实际含义

假设原始动作索引为：

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, ...
```

执行 `[::4]` 后得到：

```text
0, 4, 8, 12, 16, ...
```

原始控制频率为 20 Hz，所以这些记录的时间戳是：

```text
0.00, 0.20, 0.40, 0.60, 0.80, ... 秒
```

相邻模型时间点之间跨过 4 个原始控制周期：

```text
4 × 0.05 = 0.20 秒
```

### 2.2 视频帧和动作的索引关系

设视频窗口起点为 `s`，并且 `frame_interval=1`。41 个视频索引为：

```text
s, s+1, s+2, ..., s+40
```

动作先经过 `[::4]`，所以它们对应的原始动作索引为：

```text
4s, 4s+4, 4s+8, ..., 4s+160
```

映射关系如下：

```text
视频帧 s     <-> 原始动作 4s
视频帧 s+1   <-> 原始动作 4s+4
视频帧 s+2   <-> 原始动作 4s+8
...
视频帧 s+40  <-> 原始动作 4s+160
```

模型最终只接收 41 个动作点，并没有接收全部 164 条原始动作值。每两个相邻动作点之间跨越了 4 个原始控制周期，因此这 41 个动作点描述了一条时间上更稀疏、跨度更长的轨迹。

## 3. 官方数据的 41 帧为什么覆盖 8 秒

官方对动作执行 `[::4]` 后，相邻采样点的时间间隔为：

```text
4 / 20 = 0.20 秒
```

41 帧之间有 40 个时间间隔，因此从第一帧到最后一帧的严格时间戳跨度为：

```text
(41 - 1) × 0.20
= 40 × 0.20
= 8.0 秒
```

也可以直接使用原始动作索引计算：

```text
选中的原始动作索引：0, 4, 8, ..., 160

时间跨度：
(160 - 0) / 20
= 8.0 秒
```

统一写成公式为：

```text
(41 - 1) × 4 / 20 Hz = 8.0 秒
```

### 3.1 “41 帧对应 164 个控制步”的准确含义

文档中有时使用：

```text
41 × 4 = 164 个控制步
```

这是按 41 个采样槽、每个槽代表 4 个原始控制周期计算的序列容量口径。它不表示 loader 实际读取了 164 个动作值。

对于长度为 164 的原始动作数组，其索引范围是 `0..163`。执行 `[::4]` 后读取的是：

```text
0, 4, 8, ..., 160
```

一共正好 41 个值，而最后的 `161、162、163` 不会被取到。

因此应区分以下三个概念：

- 原始数组可以有 164 条控制记录；
- loader 实际取出 41 个动作点；
- 第一个动作点到最后一个动作点的严格时间戳跨度为 8.0 秒。

如果按 41 个 0.2 秒的采样槽估算容量，会得到 8.2 秒；如果按第一条和最后一条记录的时间戳差计算，则是严格的 8.0 秒。本文讨论运动时间跨度时采用后者。

## 4. `RLBenchSelfgenDataset` 的采样方式

Selfgen loader 继承官方 loader，但覆盖了 `get_8d_action()`：

```python
def get_8d_action(self, episode_path, frame_indices=None):
    actions = self._load_8d_actions_base(episode_path)
    if frame_indices is not None:
        actions = actions[frame_indices]
    return actions
```

这里没有执行：

```python
actions = actions[::4]
```

原因是 selfgen 数据在生成时把同一个 `demo` 中的每一条 observation 同时写入所有模态：

```text
actions.npy
view*/rgb/video.mp4
view*/depth.npz
view*/mask.npz
view*/camera_params.json
```

视频明确以 20 fps 写出。动作、RGB、深度、mask 和相机参数都遍历同一个 `demo`，所以磁盘中的长度满足：

```text
len(actions)
= len(video)
= len(depth)
= len(mask)
= len(camera_params)
= len(demo)
```

它们都是原生 20 Hz，索引一一对应。

### 4.1 Selfgen 的索引关系

在 `frame_interval=1` 下，假设选择的 41 个帧索引为：

```text
s, s+1, s+2, ..., s+40
```

那么所有模态都使用同一组索引：

```text
RGB            s+i
action         s+i
camera         s+i
depth          s+i
segmentation   s+i
```

不存在隐式的 4 倍索引变换。

## 5. Selfgen 的 41 帧为什么只覆盖 2 秒

Selfgen 保持 20 Hz，相邻帧的时间间隔为：

```text
1 / 20 = 0.05 秒
```

41 个采样点之间有 40 个间隔，因此严格时间跨度是：

```text
(41 - 1) × 0.05
= 40 × 0.05
= 2.0 秒
```

对应原始索引：

```text
0, 1, 2, ..., 40
```

按索引计算同样得到：

```text
(40 - 0) / 20 = 2.0 秒
```

文档中有时写“41 帧 = 2.05 秒”，这是按 41 个 20 Hz 时间槽计算：

```text
41 / 20 = 2.05 秒
```

如果计算第一帧到最后一帧的时间戳差，则严格结果仍是 2.0 秒。

## 6. 两者的直观对比

在 `num_frames=41`、`frame_interval=1` 时：

```text
官方数据，原始动作索引：
0, 4, 8, 12, ..., 160
|<----------- 8.0 秒 ----------->|

Selfgen 数据，原始动作索引：
0, 1, 2, 3, ..., 40
|<-- 2.0 秒 -->|
```

两者最终交给模型的动作张量形状相同：

```text
action_7d: [41, 7]
action_8d: [41, 8]
```

但时间语义不同：

- 官方数据的 41 个点更稀疏，覆盖约 8 秒；
- selfgen 的 41 个点更密集，只覆盖约 2 秒。

## 7. `frame_interval` 与 `[::4]` 是两层不同操作

### 7.1 `[::4]` 的作用

`[::4]` 只出现在官方动作 loader 中，其作用是把 20 Hz 动作对齐到约 5 Hz 的官方视频：

```text
原始动作 20 Hz
    |
    | actions[::4]
    v
对齐后动作 5 Hz
```

它解决的是磁盘视频和磁盘动作频率不一致的问题。

### 7.2 `frame_interval` 的作用

`frame_interval` 用于从磁盘视频中抽取训练窗口：

```python
frame_indices = [
    start_idx + i * frame_interval
    for i in range(num_frames)
]
```

它会同时作用于已经建立对应关系的视频、动作和相机参数。它解决的是训练窗口希望以多大时间步长取帧的问题。

因此二者不能简单视为同一个降采样操作。

## 8. 任意 `frame_interval` 下的采样公式

设：

```text
F = frame_interval
s = 随机窗口起点
i = 模型序列中的位置，i = 0..N-1
```

### 8.1 官方数据

视频索引为：

```text
j_i = s + iF
```

它对应的原始控制索引为：

```text
a_i = 4j_i = 4(s + iF)
```

相邻模型时间点跨越 `4F` 个原始控制周期，有效采样频率为：

```text
20 / (4F) = 5/F Hz
```

`N` 帧的严格时间跨度为：

```text
(N - 1) × 4F / 20 秒
```

### 8.2 Selfgen 数据

视频索引和原始控制索引相同：

```text
j_i = s + iF
a_i = j_i
```

相邻模型时间点跨越 `F` 个原始控制周期，有效采样频率为：

```text
20/F Hz
```

`N` 帧的严格时间跨度为：

```text
(N - 1) × F / 20 秒
```

### 8.3 41 帧的具体结果

| `frame_interval` | 官方有效频率 | Selfgen 有效频率 | 官方 41 帧跨度 | Selfgen 41 帧跨度 |
|---:|---:|---:|---:|---:|
| 1 | 5 Hz | 20 Hz | 8 秒 | 2 秒 |
| 2 | 2.5 Hz | 10 Hz | 16 秒 | 4 秒 |
| 4 | 1.25 Hz | 5 Hz | 32 秒 | 8 秒 |

对 selfgen 设置 `frame_interval=4`，可以使它在时间间隔上接近官方数据的 `frame_interval=1`。但是如果模型是在 selfgen 的 `frame_interval=1` 上训练的，评测时直接改成 4 会改变帧间运动幅度和动作变化分布，构成分布外输入。

同时，不应为了“对齐官方”而把官方 loader 也设置为 `frame_interval=4`。官方已经通过 `[::4]` 做了一次 4 倍动作降采样，再使用 `frame_interval=4` 后，相邻模型时间点将跨越：

```text
4 × 4 = 16 个原始控制周期
```

有效频率只剩：

```text
20 / 16 = 1.25 Hz
```

## 9. 7D 动作是否也保持正确频率

Selfgen 没有覆盖父类的 `get_7d_action()`，但不会因此重新执行官方的 `[::4]`。

父类方法内部调用：

```python
actions = self.get_8d_action(
    episode_path,
    frame_indices=None,
)
```

Python 会根据 `self` 的实际类型动态分派。当对象是 `RLBenchSelfgenDataset` 时，这里调用的是 selfgen 覆盖后的 `get_8d_action()`，所以不执行 `[::4]`。

Selfgen 中两种动作的实际路径分别是：

```text
action_8d：
原始 20 Hz 动作
  -> 不降采样
  -> 按 frame_indices 取帧

action_7d：
原始 20 Hz 动作
  -> 不降采样
  -> 四元数转换为 XYZ 欧拉角
  -> 按 frame_indices 取帧
```

所以 selfgen 的 `action_7d` 和 `action_8d` 都与 20 Hz 视频保持一一对应。

## 10. 对训练和评测的影响

虽然两个 loader 最终返回的张量形状相同，但模型学习到的时间范围并不相同。

### 官方数据

- 41 个时间点约覆盖 8 秒；
- 动作轨迹较稀疏；
- 一次生成更可能覆盖任务的大部分过程；
- 模型学习的是相对长 horizon 的轨迹。

### Selfgen 数据

- 41 个时间点约覆盖 2 秒；
- 动作轨迹更密集、更细腻；
- 大多数完整 episode 无法由一次 41 帧生成覆盖；
- 完成较长任务通常需要滚动预测或 receding-horizon replanning。

因此，不能仅因为两个数据集都使用 `num_frames=41`，就认为它们具有相同的动作 horizon。`num_frames` 只描述模型接收多少个离散时间点，真正的时间范围还取决于磁盘数据频率、loader 内部降采样以及 `frame_interval`。

## 11. 最终结论

官方数据在磁盘上是“约 5 Hz 视频 + 20 Hz 动作”。`RLBenchMVDataset` 通过 `actions[::4]` 把动作降到约 5 Hz，使动作和视频对齐。因此在 `frame_interval=1` 时，41 个模型时间点从第一帧到最后一帧覆盖：

```text
(41 - 1) × 4 / 20 = 8.0 秒
```

Selfgen 数据则把视频、动作、相机、深度和 mask 全部以原生 20 Hz 一一写出，loader 不执行 `[::4]`。因此同样的 41 个时间点只覆盖：

```text
(41 - 1) / 20 = 2.0 秒
```

所以两者的根本区别可以概括为：

> 官方 loader 的一个模型时间步对应 4 个原始控制周期，而 selfgen loader 的一个模型时间步只对应 1 个原始控制周期。在相同 `num_frames` 和 `frame_interval` 下，官方数据的时间 horizon 是 selfgen 的 4 倍。
