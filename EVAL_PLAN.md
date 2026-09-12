# EVAL_PLAN —— arm0 / arm1 的评测方案

> 日期：2026-08-11。对象：`outputs/arm0__seed42`（`video+action@1.0`，对照臂）与
> `outputs/arm1__seed42`（`video+action@0.6, video+depth@0.2, video+segmentation@0.2`）。
> 两臂只差 `--template_mix` 一个 flag（数据树/seed/lr/warm-start/prompt 风格全同，
> `scripts/train_arm.sh` 保证），所以任何指标差异都可归因到**任务菜单**。
>
> 上游背景：`FORK_CHANGES.md` §5 "eval 脚本移植" 至今**未做**；ttd 的两个脚本
> (`ttd/scripts/eval/g0_openloop_rlbench_v2.py`、`g0_perception_zeroshot.py`) 是顶替式设计
> 时代写的，不能直接跑在本仓库的模板数据集上。本文件是移植 + 协议 + 判据的完整方案。

---

## 0. 一句话与判据

要产出的是**三条曲线**，不是三个数：

| 曲线 | 内容 | 回答什么 |
|---|---|---|
| **C1 动作能力 vs 步数** | `r_peak_mean`（arm0 / arm1，seen & unseen 分开） | 加感知**伤不伤**已训练好的动作先验 |
| **C2 同「动作样本数」对齐的 C1** | x 轴换成 arm1_step × 0.6 | 差异是**干扰**还是仅仅**动作数据变少** |
| **C3 感知能力 vs 步数** | arm1 的 depth AbsRel / seg IoU，对照零样本基线与 codec 地板 | 感知到底**学没学到** |

**判据（`FORK_CHANGES.md` §6.4 的可执行版本）**：

- **在学**：arm1 的 depth AbsRel 在 step≥1000 时显著低于「官方 checkpoint 零样本 AbsRel」，
  且随步数单调下降；seg IoU 显著高于零样本。
- **无干扰**：C2 上 arm1 与 arm0 的 `r_peak_mean` 配对差在 16 个 episode 上的
  中位数 95% bootstrap CI **跨 0**。
- **有干扰**：C2 上 arm1 显著低于 arm0（Wilcoxon 配对 p<0.05 且中位差 > 10 个 r_peak 单位），
  且 C1 上差距比 C2 更大（说明不只是动作样本变少）。
- **两臂同归于尽**：两条曲线都从官方零点往下掉 → 那是 ttd `TODO_post_3k_overfit.md` 的
  续训漂移复现，与感知无关；此时结论是"这个配方本身在 470→1398 episode 的数据上会漂"，
  arm0 的存在正是为了让这句话可以被说出来。

---

## 1. 待测对象与完整性

| arm | 可用 checkpoint（step） | 备注 |
|---|---|---|
| arm0 | 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500 | 250/500 已被裁剪 |
| arm1 | 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500 | **2750 已损坏，见下** |

**⚠️ B1 —— `arm1__seed42/checkpoint-2750/step2750.ckpt` 是坏的。**
文件 1.00 GB（其余全部 11.95 GB），`torch.load` 报
`PytorchStreamReader failed reading zip archive: failed finding central directory`
——写盘中途被截断。**不要评它，也不要从它续跑**（`find_latest_checkpoint` 会挑到它，
再启 arm1 前需先删除或改名该目录）。已核实其余 18 个 checkpoint 均可 `torch.load`，
945 keys / 6.42B 参数 / bf16 / 含 `cam_encoder`，与 `build_pipeline` 组装的 `pipe.dit`
结构对得上（`strict=True` 可加载）。

**公共比较网格 = {750, 1000, 1250, 1500, 1750, 2000, 2250, 2500}，8 个点。**
arm1 的 250/500 额外画在 C1 上（单臂早期形状），但不进配对检验。

两臂都从 `anyeZHY/ActionImages/step125750.ckpt` 暖启动 ⇒ **曲线的 x=0 点 = 官方 checkpoint
在 explicit prompt 下的读数**（见 §5 R2），这不是"另一个模型"，就是两臂的初始化。

---

## 2. 指标定义

### 2.1 动作（主指标）

在生成的 action-image 上，逐帧逐视角：

| 指标 | 定义 | 为什么 |
|---|---|---|
| **`r_peak_mean`** | 红通道逐帧空间最大值，对帧与视角取均值 | **主指标**。动作被编码成一个高斯 blob，`r_peak` 低 = 模型不再画得出可解码的动作图。ttd 观察到的坍塌（官方 106 → A1@2500 47.0 → A1@3000 3.7）就是这个量掉下去，而 loss 曲线看不出来 |
| `r_peak_weak_frac` | `r_peak < 100` 的帧占比 | 阈值 100 沿用 ttd，避免均值被少数强帧撑住 |
| `pos_err_strong_median` | 仅在两视角 `r_peak ≥ 100` 的帧上算的解码位置误差中位数（米） | **无条件的 `pos_err` 会骗人**：blob 消失后 argmax 是噪声，误差反而可能"稳定"在 0.45 m 左右（官方 i2va 就是 0.4976 m）。条件化后它才是"能画出动作时画得准不准" |
| `pos_err_mean_m` | 无条件均值 | 只为与 ttd 历史数字（reports/g0_rlbench_v2）保持可比，不作判据 |
| **`axis_err_deg`** | 解码方向向量与 `R(gt_euler)·x̂` 的夹角（度），同样只在强帧上取中位数 | 姿态误差。官方 7D 的 `[3:6]` 本来就是这个量，定义见 §2.1.1 |
| `gripper_acc` | **`median(B) > 32`** 与 GT 开合比对 | 次要。**不能**用官方的 `any(B>128)` 规则——实测恒为 closed，§2.1.1 有证据 |

协议：**i2va**（每段只给首帧，模型自己生成动作）= 主协议；**v2a**（RGB 整段给定）= 副协议，
分离"看不见"与"画不出"。**a2v** 只跑一次做**工装自检**：动作段整段给定 ⇒ 解码结果只反映
「编码-解码往返 + VAE 往返」的地板，**与 checkpoint 无关**。地板值在阶段 A 用固定 `near/far`
实测一次并冻结（量级应为 cm 级、`r_peak` 应接近 255）；此后任何 checkpoint 的 a2v 偏离该地板
都说明相机/视角/窗口接错了，而不是模型的问题。

#### 2.1.0 官方仓库的 decode 与"评估"现状（本方案的唯一抄写来源）

**官方 `/workspace/ttdu/ActionImages` 里没有评估。** 全仓库 `grep` 不到任何 metric、GT 比对、
成功率或 benchmark，README 里连 evaluation / benchmark 字样都没有，`scripts/` 只有两个训练脚本。
唯一消费生成结果的地方是 `inference.py:298 export_action_point_cloud_from_pred_video`，
它的产物是一个**给人眼看的 `.ply` 点云**，不是数字。

官方那条 decode 链路（**本 fork 逐字未改**：`inference.py` / `training/utils.py` /
`training/helpers/io.py` / `vis/` 与官方 diff 为空）：

```python
# inference.py:311-328
heatmaps_1 = raw_video[T//4 : T//2]      # 第 2 段 = view1 的 action image
heatmaps_2 = raw_video[3*T//4 :]         # 第 4 段 = view2 的 action image
heatmaps   = torch.stack([h1, h2], dim=1)                      # [T, V=2, H, W, 3]
action_points = fuse_multiview_heatmaps_to_7d_point_torch(
    heatmaps, extr[..., :3, :].reshape(2,-1,3,4).permute(1,0,2,3),
              intr.reshape(2,-1,3,3).permute(1,0,2,3),
    near=0.5, far=1.0, num_depth_samples=512, apply_edge_smoothing=False)
trimesh.PointCloud(action_points[:, :3], colors=coolwarm(t)).export(...)   # 只用 [:3]
```

三个直接影响本方案的事实：

1. **`near=0.5, far=1.0` 是官方硬编码的常数**，按他们 xarm 桌面 demo 的尺度定的，
   **不是**从 GT 推的。→ 本方案照官方口径用**固定常数**，只按 RLBench 的 rig 重新标定：
   实测 192 个 view-episode / 33900 个采样点的「相机中心 → 末端」距离是
   `p0=0.642 m, p50=1.176 m, p100=1.793 m`，故取 **`near=0.6, far=1.8`,
   `num_depth_samples=512`, `apply_edge_smoothing=False`**，全部 checkpoint / 全部 arm 共用。
   这样 `pos_err` 里**没有任何 GT 泄漏**，绝对值可以直接报。
2. **`action_points[:, :3]` 是点云导出这一步的截断，不是解码器的能力上限。**
   `fuse_multiview_heatmaps_to_7d_point_torch` 返回的是完整 7 维，只是 `trimesh.PointCloud`
   只吃 xyz。这 7 维**能不能拿来算指标要逐维看**，见 §2.1.1。
3. **`r_peak` / `pos_err` / `gripper_acc` 这些指标官方一个都没有**，本方案自定义，
   §2.1 与 §2.1.1 是它们的完整定义。阈值 `r_peak < 100`、`b_med > 32` 同理是本方案自定的
   判读常数，**不是**官方或论文口径，报告里必须这么措辞。

另一条官方输出是 `combine_action_video`（`training/helpers/io.py`）：把 4 折拼成
`[第1段 | 第2段 | 两者 0.5 混合]` 的画面存 mp4——纯可视化，把 action image 半透明叠回 RGB 上。
本方案的定性检查直接用它，不要另写。

#### 2.1.1 `pos_err` 到底怎么算的（不是 cos similarity）

编码侧 `project_actions_7d_to_5d_torch_batch` 把 GT 位姿写成**三个 3D 点**再投影到每个视角：

| 通道 | 3D 点 | 含义 |
|---|---|---|
| R | `pos` | 末端位置 |
| G | `pos + R(euler)·(+x)·0.1` | 工具 x 轴（"normal"） |
| B | `pos + R(euler)·(−z)·0.1` | 工具 up 轴，**外加**一层背景亮度承载 openness |

`project_action_2d_to_hw_torch` 把每个 2D 点画成高斯 blob（`sigma=0.05`），这就是 action image。

解码侧 `fuse_multiview_heatmaps_to_7d_point_torch`（`training/utils.py:1070`）：

1. 每个视角的 R 通道归一化成概率图 → 求**加权 2D 质心**（不是 argmax）；
2. 过质心投射射线，沿主视角射线在 `[near, far]` 上采 512 个深度；
3. 取**把各视角 heatmap 响应同时最大化**的那个 3D 点 → 世界系 `xyz`（多视角软三角化）；
4. G 通道同样解一遍 → 方向轴；B 通道**只**用来判开合（`median > 32`）。

`pos_err = ‖dec[:3] − gt_a7[:3]‖₂`，**世界坐标系下的欧氏距离，单位米**。全链路没有任何余弦相似度，
也没有在图像空间比像素——两张 action image 从不直接相减。

#### 解码出来的 7 维，逐维能不能当指标

`fuse_multiview_heatmaps_to_7d_point_torch` 返回 `[x, y, z, dir_x, dir_y, dir_z, gripper]`，
GT 是 `[x, y, z, euler_x, euler_y, euler_z, openness]`——**同样是 7 维，但后 4 维不同构**，
函数 docstring 自己写明了这一点（"We model the 7D action […euler…] via two 3D points"）。

| 维 | 解码得到 | 能否与 GT 比 |
|---|---|---|
| `[0:3]` | 世界系 xyz（米） | ✅ 直接比 → `pos_err` |
| `[3:6]` | **单位方向向量** `normalize(end − start)` | ✅ **可以比，但要先转**：GT 的 euler → `R(euler)·(+x)`，这正是编码器 `project_actions_7d_to_5d_torch_batch` 用的那根轴。两个单位向量取夹角 → **`axis_err_deg`**（本方案新增，见下） |
| `[6]` | 夹爪开合 | ❌ **官方实现是退化的**，见下 |

**新增指标 `axis_err_deg`** = `arccos(clip(⟨dir_pred, R(gt_euler)·x̂⟩, −1, 1))`，逐帧、单位度，
与 `pos_err` 一样只在两视角 `r_peak ≥ 100` 的帧上取中位数。**不需要写任何新解码代码**，
官方 7D 里第二段本来就是这个量——上一版方案漏掉了它，等于白丢一个免费的姿态指标。

**⚠️ 官方的夹爪位是常数 1，不能用。** `fuse_multiview_heatmaps_to_7d_point_torch:1132` 的规则是
「任一视角任一像素 `B > 128` 即判 closed」，而 B 通道里画着 **up 轴的高斯 blob**（峰值 255）。
实测（`project_action_5d_to_rgb_torch`，256²，open / closed 各一帧）：

```
OPEN    B: max=255.0  median=64.0  #px>128=697   官方 any(B>128) -> closed(1)
CLOSED  B: max=255.0  median= 0.0  #px>128=697   官方 any(B>128) -> closed(1)
```

两种情况都有 697 个像素超过 128 ——**blob 自己就把规则触发了**，openness 永远读成 closed。
真正承载开合的是 B 通道的**背景电平**（`project_action_5d_to_rgb_torch:451-452`：
`B<=0.25` 的像素填 `0.25·openness`，即 open→64、closed→0），blob 的 max 看不见它、
中位数才看得见。故本方案的 `gripper_acc` 用 **`median(B) > 32`** 判开合；
这不是"沿用 ttd"，而是因为官方那条规则在数值上不成立，**报告里要如此说明**。

**仍然缺的：绕前向轴的滚转角。** B 通道的 up-blob 从头到尾只被拿去做阈值，没有被三角化，
所以拼不出完整旋转矩阵。补法约 30 行（B 通道先扣掉 openness 背景电平再走同一套
`fuse_multiview_heatmaps_to_3d_point_torch`，然后用 forward/up 两轴 Gram-Schmidt 拼 R），
**这是闭环的前置条件**（见 §2.4），但**不是** `axis_err_deg` 的前置条件。

**其余注意事项：**
- **深度 sweep 范围用固定常数 `near=0.6 / far=1.8`**（§2.1.0 标定），照官方硬编码 `0.5/1.0`
  的口径、只换 rig 尺度。**不要**用逐 episode 从 GT 推导的 `near/far`——那会把真值信息喂进解码器，
  跨臂比较虽然是共模项，但 `pos_err` 的绝对值就不能报了。
- **`sigma=0.05` 的 blob 半径 ≈ 12 px @256**，所以质心法对"blob 变胖变淡"很敏感——
  这正是 `r_peak` 比 `pos_err` 先动的原因。

### 2.4 关于 RLBench 闭环成功率：**先做门禁，暂不做主指标**

结论：**基础设施是现成的，但这一轮不做闭环，把它挂在一个明确的门禁后面。** 理由是三条硬约束，
不是懒。

**现状（已核实）**：`ttd_eval` conda 环境里 PyRep + RLBench 装好了，
CoppeliaSim 4.1 在 `/workspace/ttdu/ttd/opt/`，`source ttd/scripts/env_eval.rc` 后
`import rlbench` 正常，114 个任务类全在（含本方案 8 个任务）。上游 ActionImages 仓库
**没有**任何仿真 rollout 代码——README 的 inference 只做开环生成 + 一个还是 TODO 的 Blender 可视化。

**约束 1 —— 解码器给不出可执行的位姿。** 见 §2.1.1：现在只解出「位置 + 一根轴」，
RLBench 的 `EndEffectorPoseViaPlanning` 要 `[x,y,z,qx,qy,qz,qw,grip]`。绕轴的滚转角没解，
IK 目标就是欠定的。**必须先补 6-DoF 解码，并用 a2v 自检验证旋转往返误差**
（现在的 0.008 m 自检只覆盖位置）。

**约束 2 —— 41 帧装不下任何一个 episode。** 实测 variation0 的 episode 长度：
中位 **168** 步、最短 59（push_buttons）、最长 397（stack_blocks），
**`steps ≤ 41` 的比例是 0.000**。单次生成只覆盖中位 episode 的 ~24%，
所以"生成一次、开环回放、看成功率"这条便宜路径**不成立**，必须做 receding-horizon 分块重规划。

**约束 3 —— 成本比开环高两个数量级，且很可能撞地板。**
每次重规划 = 一次 35 s 的扩散生成。执行满 41 帧再重规划是最省的算法，中位 episode 也要 ~4 次
= 2.3 min GPU；正常的"只执行前一半"要 8–16 次 = 5–10 min，再加仿真时间。
一个有统计意义的成功率格子要 ≥20 次 rollout ⇒ **每个 (checkpoint, task) 格子 2–3 小时**。
18 个 checkpoint × 8 任务的曲线彻底不可能，连 2 checkpoint × 2 任务的端点检查都要 10–20 小时。
更要命的是**地板效应**：官方 checkpoint 在 close_jar 上 `r_peak_weak_frac = 0.61`
（61% 的帧根本画不出可解码的动作图），这样的策略大概率 0% 成功，
0% vs 0% 分不开两个臂——而分开两个臂正是这次实验的全部目的。

**因此的执行顺序（门禁）**：

1. 先跑完 §6 的阶段 B（官方 ckpt 的开环参照点）。
2. **门禁**：若官方 ckpt 在 seen 集上 `r_peak_weak_frac < 0.3` 且
   `pos_err_strong_median < 0.05 m`（即先验确实能画出稳定、准确的动作），
   则闭环有分辨力，投入约 2 人日补 6-DoF 解码 + 相机 rig 对齐 + rollout 循环，
   只在 **arm0/arm1 各取 step 750 与 2500 两个端点** × 4 个任务 × 20 rollout 上做端点检查。
3. 否则（大概率）：在报告里写明"闭环成功率在本先验强度下无分辨力，
   已用 R2 的 `r_peak_weak_frac=…` 证明"，把闭环留到 stage-2（from-Wan-base 的 125k 步臂）之后。

**如果真要做，还有一条必须先解决的隐患**：rollout 时喂给模型的两个视角必须与
`rlbench_selfgen_v2` 的渲染 rig 完全一致（每个 episode 的 `view*/camera_params.json` 记着
逐帧 extrinsics/intrinsics）。RLBench 默认相机不是这个 rig，必须按数据里的参数配置
`VisionSensor`，否则每一步 rollout 都是分布外输入，测出来的 0% 不能归因给模型。

### 2.2 感知（arm1 的存在理由）

| 任务 | 指标 | 地板 / 参照 |
|---|---|---|
| `video+depth` | **AbsRel**（decode 回米制，clip 到 `[MIN_VALID, MAX_VALID]=[0.05,10]`） | codec 往返地板 **0.089%**（`tests/test_depth_roundtrip_e2e.py` 实测）；官方零样本值待测（R3） |
| | **`valid_frac`** | 必报。`path_decode` 对不在编码流形上的颜色返回 NaN，只在有效像素上算 AbsRel 会让"只画对一小块"的模型赢。同时报 `absrel_penalized`（无效像素按最大误差计） |
| `video+segmentation` | **IoU**（`decode_known_color`，按 `build_referring_spec` 解析出的实例） | codec 往返地板 IoU=1.0；报 `gt_px`（<50 的 episode 直接剔除，见 §3.3）与 `pred_empty_frac` |

### 2.3 RGB 世界模型（第三根轴，便宜且必要）

i2va 生成结果里 RGB 段 vs GT 帧的 **PSNR**。arm1 把 40% 的算力挪去画 depth/seg，
"动作没坏但视频坏了"是一个真实的可能性，不测就看不见。CPU 侧即可算（GT 由同一 dataset
重建），不额外占 GPU。

---

## 3. 协议（必须逐条固定，否则曲线不可比）

### 3.1 采样参数

`res=256`（与训练一致，官方 512 先验的域适应是共模项）、`num_frames=41`、
`num_inference_steps=50`、`cfg_scale=7.5`、`seed=42`、bf16、单卡。
与 ttd 历史数字同参，可直接对照 `official/i2va r_peak_mean=106.0`。

### 3.2 视角与时间窗必须钉死

`base.py:132` 的 `random.sample(range(len(videos)), 2)` 选 cond/target 视角，
`training/helpers/io.py` 的 `random.randint` 选 41 帧窗口起点——两个**未播种**的抽样。
不钉死，两个 checkpoint 之间比的就不是同一段像素。

做法：沿用 ttd 的 monkeypatch（`view_pair=(0,1)`、`video_start_idx=0`），**并且**用本 fork
新增的一等公民 provenance 断言它生效了：

```python
assert sample["view_indices"] == [0, 1]
assert sample["frame_indices"][0] == 0
```

（provenance 字段是 `FORK_CHANGES.md` §2.4 删掉 ttd 那 115 行 monkeypatch 换来的；
ttd `g0_perception_zeroshot.py:126` 用 `sorted(glob(...))` 重推视角、与 dataset 的**未排序**
glob 顺序不一致的那个真 bug，在这里结构上不可能再犯——但前提是 eval 走 provenance，
不要自己再 glob 一次。）

### 3.3 Episode 集合（已核实全部存在）

**SEEN（variation0，训练见过；8 个）**
```
close_jar/variation0/episodes/episode0
insert_onto_square_peg/variation0/episodes/episode0
light_bulb_in/variation0/episodes/episode0
meat_off_grill/variation0/episodes/episode0
open_drawer/variation0/episodes/episode0
push_buttons/variation0/episodes/episode0
reach_and_drag/variation0/episodes/episode0
put_item_in_drawer/variation0/episodes/episode0
```
**UNSEEN（held-out variation，训练 `--variations 0` 从未见过；8 个）**
```
close_jar/variation1/episodes/episode0
insert_onto_square_peg/variation1/episodes/episode0
light_bulb_in/variation1/episodes/episode0
meat_off_grill/variation1/episodes/episode0
open_drawer/variation1/episodes/episode0
push_buttons/variation1/episodes/episode0
reach_and_drag/variation1/episodes/episode0
put_item_in_drawer/variation1/episodes/episode0
```

16 个 episode 是**配对检验的样本量**（同 episode、同视角、同窗口、同 seed，跨 arm/step 配对），
不是 4 个（ttd 当年用 4 个，配对检验没有功效）。seen/unseen 各 8 是为了让"过拟合 variation0"
与"能力下降"能分开读。

**seg 子集要先筛**：ttd 的注释记录了 `close_jar` 的 jar_lid 在多数 episode 里被遮挡，
GT mask 近乎全空，IoU=0 反映的是 GT 空、不是模型差。**阶段 A 用 CPU 预扫**上面 16 个
episode 的 GT mask 像素数，只保留 `gt_px/T ≥ 50 px/帧` 的进 seg 评测，并把入选名单冻结在
`eval/episodes_seg.json` 里。

### 3.4 prompt 必须与训练同分布

两臂都用默认 `--prompt_tag_style explicit`（已从 wandb metadata 核实：两个 run 的命令行都没传
该 flag），所以**每个** prompt 都带 `<video><action>` / `<video><depth>` / seg 颜色标签前缀。

⇒ eval **不要自己拼 prompt**，用 `ds.getitem(idx, force_template=...)` 返回的 `sample["text"]`。
它由 `templates.prompt_prefix` 生成，与训练逐字同源。

---

## 4. 需要写的代码（4 个文件 + 1 个 pipeline 补丁）

### ⚠️ B2 —— 推理管线不支持模板，感知评测**跑不了**

`training/wan_video_action_images.py:prepare_action_images_inference_latents` 只会拼两种布局：
视频-only（每视角 1 段）和 `[video|action]×V`（每视角 2 段），而且第二段的像素**只能**来自
`project_action_5d_to_rgb_torch`。`video+depth` 需要的
`[v1_rgb | v1_depth | v2_rgb | v2_depth]` 无法表达——不是参数问题，是没有注入任意 stream 的口子。

**解法（约 60 行，不动既有路径）**：新增
`prepare_template_inference_latents(...)`，复用 `training/templates.py` 里
**训练用的同一套** `plan_segments` / `assemble`（它们不依赖模型，纯 CPU 可测）：

1. 各 stream（`sample["streams"][m]`，`[C,2T,H,W]`）按视角切开 → VAE 编码 → `latents_by_modality[m][v]`；
2. action 段仍走 `project_actions_7d_to_5d → project_action_5d_to_rgb → VAE`；
3. 相机 Plücker latent 按视角算 → `cam_by_view`；
4. **plan 用确定性构造，不抽样**（见下表）→ `assemble(plan, latents, cam)` → `(latents, cam_emb, masks)`；
5. `seg_latent_spans` 由 plan 的累积偏移给出，`__call__` 后半段（去噪 + 分段解码）**一行不用改**。

`__call__` 只加一个可选 `template=` / `streams=` / `plan=` 入口做路由；`template=None` 时
走原路径，逐字不变。

**评测用的确定性 plan：**

| 协议 | 模板 | plan |
|---|---|---|
| i2va | `video+action` | 全部段 `fully_given=False, single_frame=False`（只给首帧） |
| v2a | `video+action` | `video` 段 `fully_given=True` |
| a2v（自检） | `video+action` | `action` 段 `fully_given=True` |
| depth | `video+depth` | `video` 段 `fully_given=True` ← 对应训练里 `PERCEPTION_VIDEO_GIVEN_PROB=0.9` 的主模式 |
| seg | `video+segmentation` | 同上 |

感知协议**必须**整段给 RGB：这正是本 fork 与顶替式设计的分界——"RGB 进 → depth 出"才是感知，
只给首帧就退化成了 depth 空间的世界模型（`CHANGES_TEMPLATES.md` §0）。

### 交付物

| 文件 | 内容 |
|---|---|
| `eval/harness.py` | `build_pipeline`（从 ttd `acc0a_wan_sample.py` 搬，路径改本仓库）、dataset 构造、episode 解析、视角/窗口钉死 + provenance 断言、确定性 plan 构造、report 写盘 |
| `eval/eval_action.py` | 一个 ckpt × 16 episode × {i2va, v2a}，**模型只加载一次**（ttd `g0_openloop_rlbench_batch.py` 的模式；per-run 加载会让 3 分钟的加载费用乘 16） |
| `eval/eval_perception.py` | 一个 ckpt × {depth 16 ep, seg 筛后名单} |
| `eval/aggregate.py` | 收 JSON → C1/C2/C3 曲线 + 配对 Wilcoxon + bootstrap CI + PSNR |
| `tests/test_eval_assembly.py` | **CPU 守卫**：`video+action` + i2va-plan 经新路径产出的 `masks` / `seg_latent_spans` 与 `prepare_action_images_inference_latents(task_type="i2va")` **逐元素相同**；v2a/a2v 同理。没有这条，新路径可以静默地与官方基线布局不同，而两臂的所有数字都建立在它上面 |

### ⚠️ P10 —— `PYTHONPATH` 说错就白跑

`ttd_train` 环境里有一个**同名** `actionimages` editable 安装，把 `training` 映射到
`/workspace/ttdu/ActionImages`。eval 脚本必须复用 `train.py` 的
`assert_own_training_package()`（或等价断言），否则 `import training.templates` 会
**静默**加载另一个仓库、连模板系统都没有，而脚本看起来跑得很正常。
ttd 的老脚本开头那几行 `sys.path.insert(0, "/workspace/ttdu/ActionImages")` **必须删掉**。

---

## 5. 参照点与控制组（没有这些，曲线不可解释）

| 编号 | 跑什么 | 读什么 |
|---|---|---|
| **R1** | codec 往返地板（无 GPU）：`tests/test_depth_roundtrip_e2e.py` / `test_seg_codec.py` 在**评测用的 16 个 episode** 上重跑 | AbsRel ≈ 0.089%、IoU ≈ 1.0。C3 的下界；模型不可能比它好 |
| **R2** | 官方 `step125750`，**explicit** prompt，全协议 | **曲线的 x=0 点**（= 两臂的初始化） |
| **R3** | 官方 `step125750`，**`prompt_tag_style="none"`**，i2va | R2−R3 = **加标签的代价**，即 doc 16 §3.1 "共模、会抵消"那个假设的实测值。这是最便宜的高价值控制。（附带：`r_peak` 不依赖 `near/far`，所以这一格也能与 ttd 历史读数 `close_jar/variation0/episode6 → 106.0` 粗对一下；仅作交叉参考，**不是**判据，ttd 那套用的是 GT 推导的 sweep 范围） |
| **R4** | 官方 `step125750` 跑 depth/seg 协议 | 感知的**零样本基线**。C3 的判据是"显著低于它"，没有它 AbsRel 是个孤零零的数 |
| **R5** | **arm0** 跑 depth/seg 协议（step 750 与 2500 两点即可） | 负控制。arm0 从没见过 `<depth>`，它的 AbsRel 应当 ≈ R4 且不随步数改善。若它也在变好 → 变化来自续训本身而非感知监督 |
| **R6** | a2v 自检（任一 ckpt 一次） | 编解码往返地板，阶段 A 实测后冻结。偏离 = 工装接错 |

---

## 6. 执行计划与算力预算

单次生成实测 ~35 s（41 帧 / 256² / 50 步 / 单卡，ttd 日志 714 次采样的众数）；
模型 + 12.8 GB ckpt 加载 ~3–4 min，**因此必须按 checkpoint 批处理**。

| 阶段 | 内容 | GPU | 预计 |
|---|---|---|---|
| **A** | 完整性与地板：18 个 ckpt 的 `torch.load` 体检、R1 codec 地板、seg 可见性筛选、`tests/test_eval_assembly.py` | 0 | ~30 min |
| **B** | 参照点 R2/R3/R4/R6（官方 ckpt，3 次加载） | 1 卡 | ~1.5 h |
| **C** | 动作主网格：(arm0 8 + arm1 10) × 16 ep × 2 协议 | 2 卡并行 | 18 × (16×2×35 s + 4 min) ≈ **7 GPU-h → ~3.5 h** |
| **D** | 感知：arm1 10 ckpt × (16 depth + ~6 seg) + R5 arm0 2 点 | 2 卡并行 | 12 × (22×35 s + 4 min) ≈ **3 GPU-h → ~1.5 h** |
| **E** | 聚合出图 + PSNR（CPU） | 0 | ~30 min |

**总计 ≈ 10 GPU-小时，2 卡约 5–6 小时。**
时间紧的话先砍网格不砍 episode 数：C 阶段只跑 {750, 1250, 1750, 2500} 4 个点
（≈1.6 h），配对功效不变，只是曲线粗一档；**不要**反过来砍成 4 个 episode，
那会让配对检验失去意义。

**GPU 选择**：本次查询时 1、2 号卡空闲（7 MiB），0/3–7 有外部租户。
`/workspace` 仅剩 372 G，**每次启动前重新 `nvidia-smi`**，且 mp4 可视化只对
seen 集的 2 个 episode 保存（否则 18 ckpt × 32 视频会吃掉几十 G）。

---

## 7. 分析与出图

1. **C1**：x=step，y=`r_peak_mean`（16 ep 的中位数 + IQR 带），arm0 / arm1 两条线 ×
   seen / unseen 两个子图，横向虚线 = R2（初始化点）。
2. **C2**：同上，arm1 的 x 乘 0.6（动作样本数对齐）。C1 与 C2 的差 = "少喂动作"能解释的部分。
3. **C3**：x=step，y=AbsRel（对数轴）与 IoU，画上 R4（零样本）、R1（codec 地板）、R5（arm0 负控制）。
4. **配对统计**：每个 step 上对 16 个 episode 做 arm0−arm1 的 Wilcoxon signed-rank，
   报中位差与 10000 次 bootstrap 的 95% CI。**不做**跨 arm 的训练 loss 比较——
   arm1 的 loss 是三种模板的混合平均，与 arm0 的 loss 不是同一个量。
5. 输出 `reports/eval_arms/` 下：逐次 JSON、汇总 CSV、三张 PNG、一份 `SUMMARY.md`
   （含每个判据的通过/不通过与实测数字）。

---

## 8. 陷阱清单（逐条已在上文定位）

| # | 陷阱 | 位置 |
|---|---|---|
| P1 | arm1 step2750 ckpt 已损坏，会被 `find_latest_checkpoint` 挑中 | §1 |
| P2 | 官方 ckpt 没见过 `<video><action>` 标签，与两臂不同分布 | §5 R2/R3 |
| P3 | 无条件 `pos_err` 在 blob 消失后仍"稳定"，会掩盖坍塌 | §2.1 |
| P4 | a2v 的数字与 checkpoint 无关，是工装自检不是指标 | §2.1 / R6 |
| P5 | 视角与时间窗是两个未播种抽样 | §3.2 |
| P6 | AbsRel 只在有效像素上算会奖励"大面积画废" | §2.2 |
| P7 | seg 的 GT 可能近乎全空，IoU=0 不代表模型差 | §3.3 |
| P8 | 逐 run 重载 5B 模型会让加载费用乘 16 | §4 |
| P9 | `/workspace` 只剩 372 G，mp4 会吃盘；GPU 是共享的 | §6 |
| P10 | `PYTHONPATH` 错 → 静默加载另一个仓库的 `training` | §4 |
| P11 | 新推理路径可能与官方布局不一致而无人察觉 | §4 `tests/test_eval_assembly.py` |
| P12 | 官方 7D 的 `[3:6]` 是单位方向向量、不是 euler，直接与 GT 相减是错的（要先转成同一根轴再取夹角）；绕轴滚转角仍缺 | §2.1.1 |
| P16 | 官方 `fuse_..._7d` 的夹爪位被 up-blob 触发、恒为 closed，照抄即得一个常数指标 | §2.1.1 |
| P13 | 若用逐 episode 从 GT 推导的 `near`/`far`，`pos_err` 绝对值带真值泄漏、不可报 | §2.1.0 / §2.1.1 |
| P15 | 官方仓库**没有任何评估代码**；`r_peak` / `pos_err` / 各阈值均为本方案自定义，措辞不得暗示是论文口径 | §2.1.0 |
| P14 | 41 帧装不下任何 episode（中位 168 步），开环回放测成功率不成立 | §2.4 |

---

## 9. 下一步

按 §6 的 A → B → C/D → E 顺序执行。阶段 A 全部为 CPU，可立即开始；
阶段 B 的 R3 复现出 `r_peak_mean ≈ 106` 是**放行阶段 C 的门禁**——
在工装被验证正确之前，两臂跑出来的任何数字都不可信。
