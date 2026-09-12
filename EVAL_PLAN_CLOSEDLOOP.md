# EVAL_PLAN_CLOSEDLOOP —— RLBench 闭环成功率方案

> 日期：2026-08-12。对象同 `EVAL_PLAN.md`：`outputs/arm0__seed42`（`video+action@1.0`）与
> `outputs/arm1__seed42`（`video+action@0.6, video+depth@0.2, video+segmentation@0.2`）。
>
> 本文件**取代 `EVAL_PLAN.md` §2.4**（"先做门禁，暂不做主指标"）。§2.4 列的三条硬约束
> 逐条复核后：**约束 1 是误判**（是解码器实现缺口，不是表征缺口；**已按论文 §3.2 补完并测过，
> 见 §1.5 与 §9**），**约束 2 的成因被找到了**（是我们自己的 `frame_interval=1` 与官方
> `actions[::4]` 分道扬镳，不是 41 帧的固有限制；**`rlbench_selfgen` 因此需要改，见 §2.2.1**），
> **约束 3 的对照对象定为 Table 3 的 `Ours` 行 = 20.6%**（手工公式解码那一行，正是 §1 补完的
> 那套解码器；**不**对照 `w/ action head` 的 36.7%）。
>
> `EVAL_PLAN.md` 的开环指标（`r_peak` / `pos_err` / C1–C3 曲线）在本方案里**降级为仪表与门禁**，
> 不再是交付物。交付物是**闭环成功率**。

---

## 0. 三条约束的复核结论（全部有实测支撑）

| 原约束 | 原结论 | 复核结论 | 证据 |
|---|---|---|---|
| **1. 解码器给不出可执行位姿** | 阻塞 | **误判，且已解除。** 编码侧写的是**三个 3D 点**，含两根独立轴 ⇒ 完整 SO(3) 本来就在图里；缺的只是解码器少三角化了一个通道。**按论文 §3.2 补完后整旋转测地误差中位 0.63°、p95 1.59°，位置 p95 4.7 mm，测试全绿** | §1 / §9 |
| **2. 41 帧装不下 episode** | 阻塞 | **成立，成因是我们的数据配方。** 官方 loader `actions[::4]`，41 帧 = **164 个控制步**；我们 selfgen 是 20 Hz 1:1、`frame_interval=1`，41 帧 = **41 步 = 2.05 s**。本轮用 replan 补；**`rlbench_selfgen` 的 stride 是 stage-2 第一优先级** | §2 / §2.2.1 |
| **3. 成本高 + 撞地板** | 暂缓 | **成本重算后可接受**（≈16 GPU-h 拿到 320 次 rollout）。对照基准 = Table 3 `Ours` 的 **20.6%**。地板效应仍是真风险，用**开环门禁 + 官方 ckpt 同协议对照**兜底 | §3 / §6 |

---

## 1. 约束 1：encode 给的是完整 action，decode 为什么不行

### 1.1 编码侧确实是完整的

`training/utils.py:256 project_actions_7d_to_5d_torch_batch` 把每个 7-DoF action 写成**三个 3D 点**，
分别投影进 R/G/B：

| 通道 | 3D 点 | 含义 |
|---|---|---|
| **R** | `pos` | 末端位置 |
| **G** | `pos + 0.1 · R(θ)·(+x)` | 第一根轴（代码变量名 `normal_3d`） |
| **B** | `pos + 0.1 · R(θ)·(−z)` | 第二根轴（代码变量名 `up_3d`），**再叠**一层背景电平承载 openness |

论文 §3.1 Eq.(1) 写的是同一件事（论文把 `R(θ)e_x` 叫 up、`R(θ)(−e_z)` 叫 normal，
**与代码变量名正好互换**——只是命名，几何完全一致）。

**两根独立轴 ⇒ 完整旋转矩阵。信息一点没丢。**

### 1.2 不行的是**shipped 解码器**，而且论文里就有正确版本

`training/utils.py:1070 fuse_multiview_heatmaps_to_7d_point_torch` 只做了两件事：

```python
state_6d = fuse_multiview_heatmaps_to_6d_point_torch(start=R, end=G, ...)   # 只三角化 R 和 G
closed_any_view = (B > 128).any()                                          # B 只当布尔阈值用
```

**B 通道的几何从头到尾没被三角化。** 所以 shipped 解码器输出的 `[3:6]` 只是一根单位方向向量，
绕它的滚转角欠定——这就是 `EVAL_PLAN.md` §2.4 说的"IK 目标欠定"。

但**论文 §3.2 明确解了三个点**（p.8）：

> 令 `q̂^pos, q̂^up, q̂^normal` 为解码得到的 3D 点。
> `p̂ = q̂^pos`，`ê^x = norm(q̂^up − q̂^pos)`，`ê^z = norm(q̂^pos − q̂^normal)`，
> 再取 `ê^y = ê^z × ê^x`，由此确定末端姿态 `θ̂`。

**所以这不是"我们自己发明补丁"，是把 shipped 代码补齐到论文口径。**

### 1.3 实测：补完之后精度是多少

CPU 实测（2 视角 rig、256²、`near=0.6/far=1.8`、512 采样、24 帧轨迹、GT 渲染图，
脚本见 §5 交付物 `tests/test_decode_6dof.py`）：

```
shipped : pos_err median 0.0041 m   axis(G) err median 0.40°   gripper 恒 = 1（退化）
补完后  : pos_err median 0.0041 m   axis(G) err median 0.39°   axis(B) err median 0.78°
          >>> 整旋转测地误差 median 0.62°，max 3.26°
          >>> 两根解码轴的夹角 median 89.94°（应为 90°，说明没有退化）
```

**⇒ 6-DoF 可执行位姿是现成的，补法约 40 行，零训练成本。** 输出直接 `Rot.from_matrix(R̂).as_quat()`
喂 RLBench 的 `[x,y,z,qx,qy,qz,qw,grip]`（scipy 与 PyRep 同为 xyzw 序，
与 `rlbench.py:114` 的 `R.from_quat` 用的是同一约定）。

**Gram-Schmidt 的取舍**：上面用 `ê^x` 作基准、把 `ê^z` 正交化。两根轴本来就近乎正交（89.94°），
谁作基准差别在 0.1° 量级；**固定选 `ê^x`（G 通道）作基准**，因为 G 通道没有 openness 背景电平污染。

### 1.4 gripper：不用固定阈值，改成估计背景电平

你说的对——**不该对整张 heatmap 用固定阈值**。但正确的替代也**不是换一个固定阈值**，
而是**估计 B 通道的背景电平再除以 0.25**。

**编码侧**（`utils.py:449-452`）：

```python
B = gaussian(up_point)          # 峰值 1.0
mask = B <= 0.25
B[mask] = openness * 0.25       # 背景电平 = 0.25（open）或 0（closed）
```

**shipped 规则 `any(B > 128)` 为什么恒为 closed**：up 点的高斯 blob 峰值就是 255，
实测 open / closed 两种帧**都有 697 个像素超过 128**，规则被 blob 自己触发。
实测 `acc = 0.500`（就是常数）。

**论文 Eq.(7) 是对的估计量，但字面写法有一个边界 off-by-one**：

```
ĝ = (1/0.25) · mean{ A(i,j,3) : A(i,j,3) < 0.25 }
```

编码侧用 `<= 0.25` 填充、填的值**正好是 0.25**；解码侧用**严格 `<`** 选集合。
实测：

```
OPEN   帧: #pixels < 0.25 =      0     #pixels <= 0.25 = 128217
CLOSED 帧: #pixels < 0.25 = 128214     #pixels <= 0.25 = 128214
```

**open 帧的选择集合是空的** ⇒ Eq.(7) 字面实现恒读 0，`acc = 0.500`，和 shipped 规则一样废。
改成 `<=` 后在干净渲染图上**精确**：`ĝ_open = 1.0000`、`ĝ_closed = 0.0000`。

**但生成图不是干净渲染图。** 加模糊 + 噪声（VAE + 扩散退化的替身）后，
选择集合会被截断而产生偏置：

| 噪声 σ | Eq.7 `<` | Eq.7 `<=` | **median(B)/0.25** |
|---|---|---|---|
| 0.00（仅模糊） | acc 0.50, ĝ_open 0.00 | acc 0.50, ĝ_open 0.00 | **acc 1.00, ĝ_open 1.00** |
| 0.02 | acc 1.00, ĝ_open 0.94 | acc 1.00, 0.94 | **acc 1.00, 1.00** |
| 0.05 | acc 1.00, 0.84 | acc 1.00, 0.84 | **acc 1.00, 1.01** |
| 0.10 | acc 1.00, 0.68 | acc 1.00, 0.68 | **acc 1.00, 1.01** |
| 0.20 | **acc 0.50, 0.44（翻转）** | **acc 0.50, 0.44** | **acc 1.00, 1.01** |

**⇒ 采用 `ĝ = median(B) / 0.25`，判开合用 `ĝ > 0.5`。**

措辞要点：这**和论文 Eq.(7) 是同一个估计量**（都在估背景电平），只是把
"阈值选集合 + 求均值"换成"稳健位置统计量"，因为前者在边界上退化、在模糊下有偏。
背景像素占 ~98%，中位数天然落在背景电平上。**这不是"沿用 ttd 的 median>32"那种拍脑袋阈值**，
是有推导、有对照实验的。报告里按这个说法写。

> ⚠️ 上表的 σ 是**合成噪声**，只用来给三条规则排序。**阶段 A 必须用真 VAE 往返
> 重测一遍**（`encode_video → decode_video` 打一个 GT action 段），把 `ĝ` 的实际分布画出来，
> 确认 0.5 这个判定点落在双峰之间。

### 1.5 §1 已实现（2026-08-12）

代码已落地，`tests/test_decode_6dof.py` 全绿。改了哪三处见文件末尾 §9。实测输出：

```
pos_err  median=0.0041 m   p95=0.0047 m
rot_err  median=0.63 deg   p95=1.59 deg   max=3.26 deg
GRIPPER_DECODE_EXACT_OK[paper]   ghat_open=1.0000 ghat_closed=0.0000
GRIPPER_DECODE_EXACT_OK[median]  ghat_open=1.0000 ghat_closed=0.0000
SHIPPED_GRIPPER_IS_CONSTANT_PINNED_OK  value=1.0 acc=0.500
SHIPPED_IS_STRICT_SUBSET_OK  axis gap max=0.040 deg, positions identical
rot_err median: strip_pedestal=0.63 deg   keep_pedestal=11.81 deg
```

**新发现（实现时才测出来）：openness 背景电平必须先扣掉再三角化蓝通道。**
不扣的话 open 帧整幅背景停在 0.25，把加权质心往画面中心拽，
**整旋转误差从 0.63° 恶化到 11.81°（19 倍）**。`strip_openness_pedestal=True` 是默认值。

**剩下一件阶段 A 必做的事**：经 **真 VAE 往返** 重跑上面两项，阈值放宽到
**rot p95 < 8°、pos p95 < 2 cm**，实测值写进 `reports/closedloop/codec_floor.json` 冻结——
**这是闭环成功率的地板**，任何 rollout 的解码误差都不可能比它小。
同时把 `ĝ` 的直方图画出来，确认 0.5 判定点落在双峰之间；若不落，
把 `gripper_mode` 从 `"paper"` 切到 `"median"`（§1.4 的噪声表说明后者更稳），并在报告里注明。

---

## 2. 约束 2：action chunking + replan

### 2.1 先说清楚 41 帧问题的真正成因

**官方 RLBench 数据是 4 倍时间下采样的：**

`training/dataset/rlbench.py:105`

```python
def get_8d_action(self, episode_path, frame_indices=None):
    actions = self._load_8d_actions_base(episode_path)
    actions = actions[::4]          # <<<<<<
```

实测本仓库 `data/rlbench/` 里的官方数据：

| episode | `actions.npy` | `view1/rgb/video.mp4` | 比值 |
|---|---|---|---|
| `open_box/variation0/episode0` | 149 | 38 | 3.92 |
| `open_box/variation0/episode1` | 141 | 36 | 3.92 |
| `open_box/variation0/episode10` | 136 | 34 | 4.00 |

**官方的 41 帧 = 164 个原始控制步。**

**我们的 selfgen 是 20 Hz 1:1**（`training/dataset/rlbench_selfgen.py:267-275` 显式覆盖掉
`[::4]`，注释写明"Selfgen writes everything at native 20Hz 1:1"），而 `train.py:688` 传
`frame_interval=1`。**我们的 41 帧 = 41 个控制步 = 2.05 s。**

实测 96 个 selfgen episode（8 任务 × 2 variation）：

```
episode 长度（20Hz 步）: min 59  p25 116  median 160  p75 204  max 339
  ≤ 41 步的比例 : 0.000      ← 我们一次生成盖不住任何 episode
  ≤ 164 步的比例: 0.531      ← 官方口径一次生成盖得住 53% 的 episode
```

**两个结论：**

1. `EVAL_PLAN.md` P14 说的"41 帧装不下 episode"**对我们的臂成立**，但它**不是表征的固有限制**，
   是我们数据配方与官方分道扬镳的后果。
2. 官方一次生成也只盖得住 53%，这正好解释了论文 Table 3 的成功率为什么只有 20.6%
   （见 §3）——**它就是按"一次生成打完"评的**。

### 2.2 本轮评测没得选：stride 必须是 1

有人会想"那我们评测时按 `[::4]` 喂不就对齐官方了"。**不行**：两个臂都是在
`frame_interval=1` 上训的，喂 stride-4 的条件帧是**分布外输入**，测出来的失败不能归因给模型。

**⇒ 本轮评测在 stride 1 上做，用 receding-horizon replanning 补齐 horizon。**

这一条本身就是本次实验的一个**独立发现**，必须写进报告——
我们的臂学到的是 2 秒 horizon 的策略，官方学到的是整段 episode 的策略，
**两者在闭环下的行为差异是可预期的，不能简单归因到 `template_mix`。**

### 2.2.1 ✅ 已实现：`--frame_interval`（2026-08-12）

这不只是评测口径问题，**是数据配方的缺陷**：我们的臂在 41 帧里只看到 2.0 s，
永远学不到官方那种"一次预测打完整段任务"的能力，而这正是 Action-Images 论文成立的前提
（§3.2：论文的 Table 3 就是靠单次预测覆盖 53% 的 episode 拿到 20.6% 的）。

**⚠️ 本节上一版写的"改法 A = 在 `get_8d_action` 里恢复 `[::4]`"是错的，已作废。**
selfgen 的视频本身就是 20 Hz，在 actions 上单独 `[::4]` 会把它砍成 ~40 条，
而 `frame_indices` 还在 20 Hz 空间取到 `s+40` → IndexError 或静默错位。

**正确的杠杆只有一个：`frame_interval`。** 因为 `frame_indices` 由第一路视频算出后，
**五个流全部复用同一份**——actions（`rlbench_selfgen.py:271-275`）、
camera（`:283-289`）、**depth（`:372`）**、**mask（`:375`）**、video（`load_video_frames`）。
所以只要调这一个数，五个流一起走。

#### 两个阈值

```
无 padding      需要 total ≥ 1 + (N−1)×F      # 窗口跨度
窗口数 > 1      需要 total >  N×F              # randint 的上界
窗口数         = max(0, total − N×F) + 1
```

实测 2818 个 selfgen episode（原生长度 min 58 / median 145 / max 777）：

| `F` | 严格跨度 | 有效频率 | 平均padding | >25%被填充 | **多于1个窗口** | 中位窗口数 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.0 s | 20 Hz | 0.0% | 0.0% | 100.0% | 105 |
| 2 | 4.0 s | 10 Hz | 0.5% | 0.2% | 96.5% | 64 |
| **3** ← 采用 | **6.0 s** | **6.7 Hz** | **5.1%** | **4.0%** | **61.3%** | **23** |
| 4 | 8.0 s | 5 Hz | 15.3% | 31.4% | 34.4% | 1 |
| *官方 @F=1* | *8.0 s* | *5 Hz* | *5.7%* | *4.5%* | *43.9%* | *1* |

**选 `F=3`**：padding 画像（5.1% / 4.0%）几乎与官方（5.7% / 4.5%）重合，
跨度 6.0 s 已接近官方 8.0 s，而窗口多样性还留着中位 23 个。
`F=4` 虽然严格对齐官方 5 Hz，但 65.6% 的 episode 只剩唯一窗口——
官方靠 **180k 条轨迹**买多样性，我们只有 **1398 个 variation0 episode**，
增广塌掉的代价比对齐 5 Hz 的收益大。

#### 顺带量到的两件事

- **夹爪混叠**：4 倍降采样下 selfgen 丢 12.88% 的开合跳变（1098 段 <4 步的常值段），
  官方丢 6.01%（458 段）。但 selfgen 的窗口起点**不锁死在 F 的倍数**（20 Hz 全帧在盘上），
  跨 epoch 会采到不同相位；官方的 `[::4]` 相位在生成时就烤死了。这一项 selfgen 反而占优。
- **上游的 off-by-(F−1)**：合法性判据用 `N×F`，实际只需 `1+(N−1)×F`，
  `F=3` 时白白少 2 个合法起点。影响很小，未改动（改了会动到官方路径的抽样序列）。

#### 使用

```bash
FRAME_INTERVAL=3 bash scripts/train_arm.sh arm0    # -> outputs/arm0__seed42_fi3
```

**默认仍是 1**，arm0/arm1 的行为与产物路径逐字节不变。改动清单见 §9.6。

### 2.3 不需要 clone 任何 repo；starVLA 里就有可抄的

四处现成参考，全在 `/workspace/ttdu/starVLA`：

| 参考 | 位置 | 抄什么 |
|---|---|---|
| **chunk-cache 调度（主参考）** | `examples/modelExtensions/Gemma4/eval_libero_local.py:269-285` | 标准写法：`if step % chunk_size == 0: 重新推理`；`cur_action = cached[step % chunk_size]`。**执行满一整个 chunk 再 replan**，最省算力的算法 |
| 契约文档 | `deployment/model_server/policy_wrapper.py:13-16` | 明确写着 chunk-cache 调度、gripper sticky、action ensembling 是**客户端**职责，不进模型侧。我们照这个切分 |
| 重叠 chunk / 实时分块 | `examples/modelExtensions/DiscreteDiffusion/README.md:44-62` | `action_horizon` vs `execution_horizon`，前缀长度 = `action_horizon − execution_horizon`。**只有当 §2.5 的 E=41 打不通时才升级到这个** |
| 接口形状 + bring-up 流程 | `docs/agent_skills/integrate-starvla-dataset/assets/templates/model2bench_interface.py` | `policy.run_policy(obs, prompt) -> (n_action_steps, action_dim)` 契约，以及**三步自检**：① dummy obs 无环境 ② 录制 episode 上闭环 ③ 真环境闭环。**这三步逐字采纳**，见 §6 阶段 B |

RLBench 侧不需要参考实现：rollout 循环本身约 80 行，
且 `EVAL_PLAN.md` §2.4 担心的"相机 rig 对不上"其实**不存在**——
`ttd/scripts/gen_dataset.py:22` 用的是 RLBench **内建具名相机**：

```python
VIEWS = {"view1": "front", "view2": "overhead", "view3": "left_shoulder", "view4": "right_shoulder"}
RES = [256, 256]
```

`view1/view2` = `front` / `overhead`，rollout 时打开同名 `CameraConfig(image_size=[256,256])` 即可，
**不用手工摆 `VisionSensor`**。且实测相机在 episode 内**静止**（extrinsics/intrinsics 逐帧全等），
所以每次 replan 读一次相机参数、沿时间 tile 41 份就行。

> ⚠️ 内参必须走 `obs.misc[f"{cam}_camera_intrinsics"]`（`gen_dataset.py:133` 的原路），
> **不要**从 FOV 自己推。实测记录下来的内参 `fx = fy = −351.68`（**负焦距**），
> 自己推会得到正号，投影整体翻转，每一步 rollout 都会静默错。

### 2.4 动作模式：`EndEffectorPoseViaIK`，不是 `ViaPlanning`

实测 96 个 episode 的**逐步增量**（20 Hz）：

```
|Δpos| : median 7.71 mm   p95 17.99 mm   max 35.05 mm
|Δrot| : median 0.41°     p95  3.58°     max 44.99°
```

**默认采用 `MoveArmThenGripper(EndEffectorPoseViaIK(absolute_mode=True), Discrete())`。**
IK 失败（不可达 / 奇异）时的处理已写死并记账：
**跳过该路点、`ik_fail` 计数 +1，连续失败 ≥5 个路点则提前终止该 rollout 并记为失败**。
`ik_fail_rate` 逐 rollout 上报——否则"模型画得不准"和"IK 解不出来"会混在同一个 0% 里。

> **⚠️ 更正本节上一版**："`ViaPlanning` 慢到不可用"是**错的**，实测（close_jar GT 重放）：
>
> | 模式 | ik_fail | 耗时 | 每路点 | 成功 |
> |---|---|---|---|---|
> | `ViaIK` stride1 (247 路点) | 34 (14%) | 49.8 s | 0.20 s | ✗ |
> | `ViaPlanning` stride1 (271 路点) | **0 (0%)** | 68.7 s | 0.25 s | ✗ |
> | `ViaPlanning` stride4 (68 路点) | 0 (0%) | **19.7 s** | 0.29 s | ✗ |
>
> `ViaPlanning` 只慢 25%/路点，而且**把 IK 失败清零**（消掉一个混淆项）。
> 仍默认 `ViaIK` 的唯一理由：上表是**良构的 GT 位姿**；模型画出不可达位姿时 `ViaIK` 立即失败，
> 规划器可能长时间搜索后才放弃，而 rollout campaign 是按墙钟计价的。
> `RolloutEnv(arm_action_mode=...)` / `--arm-action-mode` 两个都能选；
> 若实测 `ik_fail_rate` 高到影响结论，切 `planning`。

### 2.5 replan 协议（钉死）

```
H = 41              # action_horizon，模型一次生成的帧数，等于训练配置，不可变
E                   # execution_horizon，每次 replan 实际执行的帧数
max_steps = 2 × len(GT demo)      # 超时即失败
```

- **主协议 `E = 41`**（执行满一整个 chunk）——对应 starVLA `eval_libero_local.py` 的写法。
  最省算力，中位 episode 需 `ceil(160/41) = 4` 次生成。
- **副协议 `E = 21`**（执行一半）——中位 episode 8 次生成，算力翻倍。
  **只在主协议成功率 > 0 的 (arm, task) 格子上跑**，用来分离
  "chunk 尾部漂移"与"模型根本不会做这个任务"。
- 每次 replan 的条件输入 = **当前 front/overhead 两视角的真实观测帧**（i2va 协议：只给首帧），
  相机参数 tile 41 份，prompt 与训练同源（§2.6）。
- **不做** action ensembling / temporal aggregation。理由：加了就无法把结果归因到
  `template_mix`，而那是本次实验的唯一目的。留作 stage-2。

### 2.6 prompt 与训练同分布

两臂都是 `--prompt_tag_style explicit`。rollout 时**不要自己拼 prompt**，
调 `training/templates.py` 的 `prompt_prefix` 生成 `<video><action>` 前缀，
任务描述取 `meta.json` 的 `desc` **列表第 0 项**（钉死，不抽样）。
官方 ckpt 参照点（R2）同样用 explicit 前缀，理由见 `EVAL_PLAN.md` §5 R2。

---

## 3. 约束 3：论文 Table 3 / Table 4 到底是怎么做的

### 3.1 先纠正一件事：**Table 4 不是成功率表**

| 表 | 内容 | 是不是成功率 |
|---|---|---|
| **Table 2**（p.11） | Zero-shot：RLBench 4 任务 + Real 5 任务 | ✅ 成功率 |
| **Table 3**（p.12） | **RLBench in-domain 9 任务** | ✅ 成功率 |
| **Table 4**（p.12） | Video-and-Action **Joint Generation Quality** | ❌ **是生成质量**：PSNR / SSIM / FVD / LPIPS / 2DErr / 3DErr |

**Table 3 原始数字**（成功率 %）：

| Method | close box | close door | open door | phone base | open bottle | close drawer | open oven | open jar | wipe desk | **Avg.** |
|---|---|---|---|---|---|---|---|---|---|---|
| MV-Diffusion Policy | 20 | 40 | 15 | 20 | 5 | 50 | 10 | 0 | 0 | 17.8 |
| MolmoAct (zeroshot) | 5 | 10 | 0 | 0 | 5 | 10 | 0 | 0 | 0 | 3.3 |
| π0.5 | 10 | 0 | 5 | 5 | 45 | 65 | 0 | 0 | 0 | 14.4 |
| TesserAct | 40 | 25 | 5 | 15 | 20 | 70 | 5 | 5 | 0 | 20.6 |
| Cosmos-Policy | 40 | 15 | 0 | 15 | 30 | 80 | 0 | 0 | 0 | 20.0 |
| **Ours** | 55 | 60 | 0 | 0 | 5 | 60 | 5 | 0 | 0 | **20.6** |
| **w/ action head** | 80 | 65 | 15 | 20 | 40 | 80 | 15 | 5 | 10 | **36.7** |

**Table 4 原始数字**：Ours PSNR 23.48 / SSIM 78.62 / FVD 143.74 / LPIPS 0.209 /
**2DErr 1.61 / 3DErr 12.2×10⁻³**。

### 3.2 Table 3 的两行 `Ours` 分别是什么

论文 p.11 明确区分了两种解码：

> Besides **the reconstruction-based decoder that recovers actions from generated action
> images**, we also consider an **optional learned action head** […] a lightweight MLP that
> takes as input the output video latents, camera parameters, and decoded actions and
> observations, and train it to directly regress the continuous 7-DoF action sequence.

| Table 3 行 | 解码方式 | Avg. | 对我们的意义 |
|---|---|---|---|
| **`Ours`（倒数第二行）** | **§3.2 的手工公式解码**（就是 §1 补完的那套） | **20.6** | ✅ **我们的直接对照行**。同一条解码链路，无额外训练 |
| `w/ action head` | 额外训练的 MLP 回归头 | 36.7 | 我们没有这个头，**不对照**。它只说明"表征还能支撑更强的解码" |

**⇒ 本方案的对照基准是 Table 3 的 `Ours` 行：Avg 20.6%。**

### 3.3 与 Table 3 的可比性

**按本项目的判定：Table 3 的 `Ours` / `w/ action head` 为闭环，我们的闭环成功率直接与
Table 3 的 `Ours`（20.6%）对照。** 这是本方案采用的口径。

> ⚠️ **记录一处与论文正文的张力**，供写报告时自行斟酌，不影响上面的口径：
> 论文 p.10 §4.1 开头写 "all experiments in this subsection are conducted […] under
> **one-trial open-loop evaluation** […] the model must complete the task from a single
> forward prediction **without online replanning**"，而 Table 3 就在 §4.1 内；
> p.14 Limitations 又写 "has not yet been fully developed into a **closed-loop** policy"。
> 如果审稿人拿这两句来质疑对照的可比性，回应路径有二：
> ① 明确本文的"闭环"指 replan 频率而非论文措辞；
> ② 用 §6.3 的开环单次数字做**同协议**的第二组对照（成本只有 1/4）。
> **建议两组数都出**——开环单次那组无论如何都与 §4.1 的措辞严格同协议，是最安全的对照。

无论采用哪种读法，下面这几项**仍然不可比**，报告里要单独声明：
任务集不同（论文 9 个 in-domain 任务，我们 4 个）、分辨率不同（512² vs 256²）、
时间跨度不同（一帧 = 4 控制步 vs 1 控制步，§2.1）、backbone 训练量不同
（官方 100k 步 from Wan2.1-I2V-14B，我们从 `step125750` 续训 ≤2.5k 步）。

这同时解释了 §2.1 的观察：**官方一次生成覆盖 53% 的 episode，Table 3 的 Avg 只有 20.6%**，
两个数在同一个故事里自洽——horizon 覆盖不足是主要瓶颈。

### 3.4 能从官方抄的 / 必须自己定的

| 项 | 官方口径 | 我们 | 说明 |
|---|---|---|---|
| 指标 | task success rate | 同 | RLBench `task.success()` |
| 每格 trial 数 | **20**（Table 2/3 全部数字都是 5 的倍数） | **20** | 抄 |
| 去噪步数 | 50 | 50 | 抄 |
| CFG | **10.0**（Supp. §1 Inference Details） | **7.5** | ⚠️ **不抄**。7.5 是 `EVAL_PLAN.md` §3.1 定的、与 ttd 历史数字可比的值。**阶段 B 在 1 个任务上做 7.5 vs 10.0 的小 sweep**，选高的冻结，冻结值写进报告 |
| 分辨率 | 512² | **256²** | 不抄。两臂都是 256² 训的 |
| 帧数 | 41/视角/模态 | 41 | 抄（也没得选） |
| 视角数 | 2 | 2 | 抄 |
| 时间跨度 | 41 帧 = **164 控制步** | 41 帧 = **41 控制步** | §2.1，不可调和，用 replan 补 |
| 对照行 | Table 3 `Ours` = **20.6%** | 主对照 | §3.2。**不**对照 `w/ action head`（36.7%），我们没有那个 MLP 头 |
| **3DErr = 12.2 mm** | 有 | — | ✅ **这是最有用的一个官方数字**：它是官方模型在 in-domain RLBench 上的动作位置误差。我们 §1.3 实测的 codec 地板是 **4.1 mm**，所以官方模型的动作误差 ≈ 3× 地板。**我们任何 ckpt 的 `pos_err_strong_median` 远大于 12.2 mm，就说明动作先验比论文的弱**，闭环大概率 0% |

---

## 4. 指标与判据

### 4.1 主指标

| 指标 | 定义 |
|---|---|
| **`success_rate`** | `task.success()` 为真的 rollout 占比，每格 20 次 |
| `steps_to_success` | 成功 rollout 的控制步数中位数（效率，次要） |
| `n_replans` | 每次 rollout 的生成次数（成本记账） |
| **`ik_fail_rate`** | IK 解不出的路点占比。**必报**，否则模型误差与运动学不可达混在一起 |
| **`timeout_frac`** | 撞 `max_steps` 的占比。区分"做错了"与"没做完" |

### 4.2 同步记录的开环仪表（不是交付物，是归因工具）

每次 replan 的那一次生成，顺手记 `r_peak_mean` / `pos_err`（对齐到 GT demo 的最近邻时刻）/
`ĝ` 的直方图。**闭环 0% 时，这些数字决定报告里写哪句话**：

- `r_peak` 高、`pos_err` 小、但 success 0% ⇒ 解码/执行链路问题（查 `ik_fail_rate`、查相机 rig）。
- `r_peak` 低 ⇒ 动作先验坍塌，与 `EVAL_PLAN.md` C1 曲线对上。
- `ĝ` 不双峰 ⇒ 夹爪判读失效，§1.4 的规则在生成图上不成立。

### 4.3 判据

- **闭环有分辨力**：官方 ckpt（R2）在 4 个任务上 `success_rate` 合计 **> 10%**（≥8/80 次）。
  达不到 ⇒ 闭环在本先验强度下测不出两臂差异，按 §6 阶段 C 的**降级路径**走。
- **无干扰**：arm0 与 arm1 在同 step、同任务、同 seed 集上的成功率差，
  用 **20 次配对 rollout 的 McNemar 检验**，p ≥ 0.05 且 |Δ| ≤ 10 pp。
- **有干扰**：arm1 显著低于 arm0（McNemar p < 0.05）。
- **两臂同归于尽**：两臂都显著低于 R2 ⇒ 是续训漂移，与感知无关
  （对应 ttd `TODO_post_3k_overfit.md`）。

> **配对怎么配**：同一个 `(task, variation, episode_seed)` 在两臂上用**同一个 RLBench seed
> reset**，起始场景逐物体相同。这是 McNemar 成立的前提，rollout 循环里必须
> `np.random.seed(seed); env.reset()` 并把实际 seed 写进 JSON。

---

## 5. 需要写的代码

| 文件 | 内容 | 依赖 GPU | 状态 |
|---|---|---|---|
| `training/utils.py`（**改**） | 见 §9 | 否 | ✅ **已完成** |
| `tests/test_decode_6dof.py` | §1.5 的验收 | 否 | ✅ **已完成，全绿** |
| `scripts/run_tests.sh`（**改**） | 把新测试挂进 CPU 套件 | 否 | ✅ **已完成** |
| `eval/rollout_env.py` | RLBench 环境封装：`front`+`overhead` @256²、`EndEffectorPoseViaIK`、`reset_to_new_demo`、相机参数按 `obs.misc` 读取、`success()` / `ik_fail` 记账 | 否 | ✅ **已完成** |
| `eval/policy.py` | starVLA `model2bench_interface.py` 的契约：`run_policy(rgb, extr, intr, prompt) -> [41, 8]`。内部：两视角观测 → 条件张量 → `pipe(template="video+action", ...)` → 切第 2/4 段 → `fuse_multiview_heatmaps_to_pose_torch`（输出已经就是 RLBench 要的 8 维，不用再转） | 是 | ✅ **已完成** |
| `eval/rollout.py` | §2.5 的 receding-horizon 循环（抄 `eval_libero_local.py:269-285` 的 chunk cache），**模型只加载一次**、按 ckpt 批处理所有 task × trial；`--gt-replay` 走同一循环但喂 GT 动作 | 是 | ✅ **已完成** |
| `tests/test_policy_conditioning.py` | **CPU 守卫**：条件张量的三个约定 + canvas 布局 + 解码往返 + 视角互换负控制 | 否 | ✅ **已完成，全绿** |
| `tests/test_rollout_env.py` + `scripts/run_rollout_tests.sh` | GT 重放门禁（带仿真、不带模型/GPU） | 否 | ✅ **已完成** |
| `eval/aggregate_closedloop.py` | 收 JSON → 成功率表 + McNemar + 成本表 + 开环仪表 | 否 | 待做 |

### 5.1 ⚠️ 场景不能复现存盘 episode —— 改用 `reset_to_new_demo`（实测结论）

原方案假定"复刻 `gen_dataset.py` 的 seeding 就能重建存盘 episode 的场景"。**实测不成立：**

```
np.random.seed(seed + variation*1_000_003) → set_variation → reset()
  机械臂 home 位姿      与存盘 actions.npy[0] 差 1e-5 m  ← 这不是证据，home 位姿与场景无关
  渲染首帧 vs 存盘 mp4  MAE 19~24 / 255（reset 1/2/3 次都不行）
  用存盘 actions.npy 重放 → close_jar / push_buttons / open_drawer 全部 SUCCESS=False
```

CoppeliaSim 有 numpy seed 覆盖不到的仿真器侧状态。**结论：不要试图对齐存盘场景。**

**改用 `RolloutEnv.reset_to_new_demo(task, variation, scene_seed)`**：
`get_demos(live_demos=True)` 现场生成一条 demo，再 `reset_to_demo` 倒回该场景起点。三个好处：

1. **场景一定可解**（它有 demo），失败不能推给"初始状态本来就做不到"；
2. **同时拿到该场景的 GT 轨迹**，GT 重放门禁和开环参照指标都免费；
3. `scene_seed` 完全决定场景 ⇒ **两个臂面对同一批世界**，McNemar 的前提成立（`tests/test_rollout_env.py` 有 determinism 断言）。

代价：每次 rollout 多一次 demo 生成，实测 **15–63 s**（见 §6.1 重算）。

**已经不用做的**：`EVAL_PLAN.md` §4 的 ⚠️ B2（推理管线不支持模板）**已经解决**——
`training/wan_video_action_images.py:551 prepare_template_inference_latents` 已存在，
`__call__` 已有 `template=` / `streams=` / `fully_given_modalities=` 入口，
`tests/test_eval_assembly.py` 的 CPU 守卫也在。闭环只用 `video+action` + i2va plan，这条路已通。

**`PYTHONPATH` 陷阱照旧**（`EVAL_PLAN.md` P10）：`ttd_train` 里有同名 `actionimages` editable 装到
`/workspace/ttdu/ActionImages`。所有新脚本必须走 `REPO` 优先的 `sys.path`。

### 5.2 ✅ 环境已合一：`ttd_rollout`（单进程，不用拆 socket）

阶段 A 第 1 项已完成。原本担心要拆成 sim 进程 + 生成进程，实测**不用**：

- `ttd_eval`（PyRep 4.1.0.3 / RLBench 1.2.0，无 torch）与 `ttd_train`（torch 2.6 / diffsynth，无 PyRep）
  **都是 Python 3.10.20、numpy 都是 1.26.4**，且 PyRep 的扩展是 `_sim_cffi.abi3.so`（稳定 ABI）。
- 做法：`conda create -n ttd_rollout --clone ttd_train`，再 `pip install cffi` +
  `--no-deps` 装 `cloudpickle/farama-notifications/gymnasium/natsort/pyquaternion`，
  最后把 `pyrep/`、`rlbench/` 及其 dist-info 从 `ttd_eval` 的 site-packages 拷过去。
- **刻意不动 `ttd_train`**（它要跑 1.4 天的训练任务），克隆是零风险的保险。
- 验证：`torch 2.6.0+cu124 cuda=True` + `rlbench 1.2.0` + `diffsynth` 在同一解释器里 import 通过。

⇒ rollout 是**单进程**，`eval/rollout.py` 里模型和仿真器共用一个进程，不需要 IPC。
运行必须 `source /workspace/ttdu/ttd/scripts/env_eval.rc` 并用 `xvfb-run`（CoppeliaSim 即使
`headless=True` 也要一个 X display）；`scripts/run_rollout_tests.sh` 已经把这些包好。

---

## 6. 执行计划与算力预算

### 6.1 成本（用实测数字重算）

单次生成 35 s（256²/41 帧/50 步/单卡，ttd 714 次采样实测众数）。
**新增的两项实测开销**：`reset_to_new_demo` 的 demo 生成 **16–63 s/次**（§5.1），
仿真执行 **0.20 s/控制步**（ViaIK）。demo 长度实测 67–122 步（close_jar 例外，250）。

| 协议 | 生成次数/rollout | GPU/rollout | 仿真/rollout | 20 次/格 |
|---|---|---|---|---|
| **E = 41（主）** | 2–3 | 70–105 s | 31–65 s | **34–57 min** |
| E = 21（副） | 4–6 | 140–210 s | 同上 | 57–92 min |

**3 个可测任务 × 20 次 = 2.4 h / checkpoint（E=41）**。
`EVAL_PLAN.md` §2.4 写的"每格 2–3 小时"高估了一个任务格，但**整格网**因为多了 demo 生成
和 close_jar 被剔除，总量与原估算相当。

### 6.2 网格与总预算

| 阶段 | 内容 | GPU | 预计 |
|---|---|---|---|
| **A** | 环境合一（✅ §5.2）、6-DoF 解码补丁 + `test_decode_6dof.py`（✅ §1.5）、rollout 工装 + 4 个测试（✅ §9.7）、VAE 往返地板实测并冻结 | 0 | ~1 h 剩余 |
| **B** | ① `bash scripts/run_rollout_tests.sh`（✅ GT 重放门禁）② 官方 ckpt 单任务真闭环 + CFG 7.5/10.0 小 sweep | 1 卡 | **~2 h** |
| **门禁** | R2（官方 ckpt）3 任务 × 20 = 60 rollout，`success_rate > 10%` | 1 卡 | **~2.4 GPU-h** |
| **C** | 主网格：2 臂 × {step 750, 2500} × 3 任务 × 20 = **240 rollout** | 2 卡并行 | **9.6 GPU-h → ~5 h** |
| **D** | `eval/aggregate_closedloop.py` 出表 + McNemar | 0 | ~1 h |

**总计 ≈ 12 GPU-小时，2 卡约 6 小时。**

**任务：3 个**，不是 4 个。原定 `push_buttons` / `open_drawer` / `meat_off_grill` / `close_jar`，
**`close_jar` 被 GT 重放门禁剔除**（见 §6.4）。

### 6.4 ⚠️ `close_jar` 不可测——已从网格剔除

GT 重放实测（`reports/closedloop/gt_replay.json`，每任务 2 个场景）：

| 任务 | GT 重放成功率 | ik_fail | 跟踪误差中位 |
|---|---|---|---|
| `push_buttons` | ✅ 100% (2/2) | 0% | 1.1–1.4 mm |
| `open_drawer` | ✅ 100% (2/2) | 0% | 1.0–1.5 mm |
| `meat_off_grill` | ✅ 100% (2/2) | 0% | 1.7–1.8 mm |
| **`close_jar`** | **❌ 0% (0/2)** | **0%** | 1.1–1.4 mm |

**`close_jar` 的失败不是运动学问题**：这两个场景 IK 失败率就是 0%，末端跟踪误差 1.1 mm，
整条 GT 轨迹 180/167 个路点全部执行完，依然不成功；另跑的 `ViaPlanning` 对照
（`ik_fail` 0%，见 §2.4）同样失败。⇒ **"重放末端位姿"这条路本身拧不上盖子**——
拧盖依赖抓握时序与接触力，`Discrete` 夹爪 + EE 位姿重放复现不了。

**⇒ 在这个工装上，`close_jar` 的闭环成功率必然是 0%，而那是工装的数字、不是模型的。**
把它留在网格里只会往两个臂身上各加 20 个必然失败的 rollout，稀释 McNemar 的功效。
`tests/test_rollout_env.py` 会把任务分成 `measurable_tasks` / `unmeasurable_tasks` 两栏写进
`gt_replay.json`，campaign 只跑前者。**报告里要写明剔除了哪个任务、依据是哪个数字。**

**GPU**：本次查询 1、2 号卡空闲（7 MiB），0/3/6/7 有外部租户，4/5 满载。
`/workspace` 只剩 **460 G**。**每次启动前重新 `nvidia-smi`**；rollout 视频只对每格前 2 次 trial 存盘。

> ⚠️ `arm1__seed42/checkpoint-2750` 已损坏（1.00 GB vs 正常 11.95 GB，`torch.load` 报
> `failed finding central directory`）。`find_latest_checkpoint` 会挑中它，**跑之前先改名**。
> 详见 `EVAL_PLAN.md` §1 B1。

### 6.3 门禁不过时的降级路径（**不是"不做"**）

若 R2 < 10%：闭环对两臂无分辨力。此时**不撤退到"只写一句话"**，改交付：

1. **官方 ckpt 的闭环成功率本身就是结果**——它回答了"这个先验能不能闭环",
   是本仓库第一个 RLBench 闭环数字，值得单独报。
2. 把 320 次 rollout 的预算改投**开环单次成功率**（= 论文 Table 3 的协议，
   一次生成 41 帧、直接执行、看 `task.success()`）。这个**可以与 Table 3 同协议对照**
   （虽然任务不同、stride 不同），比闭环更接近官方口径，且成本降到 1/4。
3. 闭环留到 stage-2（from-Wan-base 的 125k 步臂）。

---

## 7. 陷阱清单

| # | 陷阱 | 位置 |
|---|---|---|
| **C1** | shipped `fuse_..._7d_...` 的 `[3:6]` 只有一根轴，滚转欠定；论文 §3.2 有正确版本，别自己发明 | §1.2 |
| **C2** | shipped `any(B>128)` 夹爪规则恒为 closed（blob 自己触发，697 px），acc=0.5 | §1.4 |
| **C3** | 论文 Eq.(7) 字面用严格 `<`，而编码侧用 `<=` 填在正好 0.25 —— open 帧选择集合为空，恒读 0 | §1.4 |
| **C4** | Eq.(7) 的"阈值选集合再求均值"在模糊/噪声下有偏（ĝ 1.00→0.44）；生成图上若 ĝ 不双峰就切 `gripper_mode="median"` | §1.4 / §1.5 |
| **C4b** | **三角化蓝通道前必须先扣掉 `0.25·ĝ` 的 openness 背景电平**，否则 open 帧整幅背景把加权质心往画面中心拽，整旋转误差 0.63°→**11.81°**。默认 `strip_openness_pedestal=True`，别关 | §1.5 |
| **C5** | 官方 loader `actions[::4]`，我们 selfgen 是 1:1 —— **同样是"41 帧"，时间跨度差 4 倍**。这是数据配方缺陷，stage-2 要改 `rlbench_selfgen` | §2.1 / §2.2.1 |
| **C6** | 本轮不能改用 stride 4 来"对齐官方"：两臂是 stride 1 训的，那是 OOD 输入。改 stride 必须连带重训 | §2.2 |
| **C6b** | 将来改 `rlbench_selfgen` 的 stride 时，video / depth / mask / camera / actions **五者必须共用同一个 `frame_indices`**；depth/mask 的取帧路径与 video 不是同一段代码，最容易漏 | §2.2.1 |
| **C7** | 相机内参 `fx = fy = −351.68` 是**负的**；自己从 FOV 推会得到正号，投影整体翻转且不报错 | §2.3 |
| **C8** | 用 `ViaPlanning` 逐 20 Hz 路点规划会慢到不可用（增量中位仅 7.7 mm）；用 `ViaIK` | §2.4 |
| **C9** | 不报 `ik_fail_rate` ⇒ 模型误差与运动学不可达混在同一个 0% 里 | §2.4 / §4.1 |
| **C10** | **Table 4 不是成功率表**，是生成质量（PSNR/FVD/2DErr/3DErr）；成功率只在 Table 2/3 | §3.1 |
| **C11** | 对照的是 Table 3 的 **`Ours` = 20.6%**（手工公式解码），**不是** `w/ action head` = 36.7%（那是额外训练的 MLP 头，我们没有） | §3.2 |
| **C11b** | 论文 §4.1 正文写的是 one-trial open-loop、Limitations 写未做闭环，与"Table 3 是闭环"的读法有张力。**同时出开环单次那组数**（成本 1/4）可以让对照无论按哪种读法都站得住 | §3.3 |
| **C12** | 官方 CFG 是 10.0，我们的历史值是 7.5 —— 不 sweep 就冻结等于埋了一个未测的自由度 | §3.3 |
| **C13** | McNemar 要求两臂用**同一个 scene_seed**，否则配对不成立。`aggregate_closedloop.py` 会检查并拒绝 | §4.3 |
| **C14** | ~~两个环境可能装不进一个~~ 已解决：`ttd_rollout` 单进程 | §5.2 |
| **C17** | **action 段第 0 帧是条件不是预测**；填 0 等于把末端锚在世界原点。必须喂当前真实位姿 | §9.7 第 0 条 |
| **C18** | **场景无法复刻存盘 episode**（渲染首帧 MAE 19–24，GT 重放全失败）。用 `reset_to_new_demo`，别试图对齐存盘场景 | §5.1 |
| **C19** | **`close_jar` 在本工装上不可测**（GT 重放 0%，且 ik_fail=0 说明不是运动学问题）。留在网格里只会稀释 McNemar 功效 | §6.4 |
| **C20** | `ĝ` 有 1.2% 的 uint8 量化地板（0.9882 而非 1.0），不能与 1.0 直接比 | §9.7 第 1 条 |
| **C15** | `arm1/checkpoint-2750` 已损坏且会被 `find_latest_checkpoint` 挑中 | §6.2 |
| **C16** | `PYTHONPATH` 错 → 静默 import `/workspace/ttdu/ActionImages` 的 `training` | §5 |

---

## 8. 下一步

阶段 A 已完成（1–3 全绿）：

1. ✅ 环境合一 → `ttd_rollout` 单进程（§5.2）。
2. ✅ `fuse_multiview_heatmaps_to_pose_torch` + `tests/test_decode_6dof.py`（§1.5）。
3. ✅ **GT 闭环重放门禁**：`bash scripts/run_rollout_tests.sh` →
   3 个任务 100%、`close_jar` 0%（已剔除，§6.4）、场景可复现。**不花一秒 GPU**，
   却同时验证了相机 rig、IK、成功判定、场景确定性和 replan 循环。

**剩下的：**

4. VAE 往返地板：把 §1.5 的解码精度经**真 VAE** 重测一次，冻结进
   `reports/closedloop/codec_floor.json`，同时确认 `ĝ` 直方图双峰（否则切 `gripper_mode="median"`）。
5. 阶段 B：官方 ckpt 单任务真闭环 + CFG 7.5 vs 10.0 小 sweep（§3.4）。
6. 门禁：官方 ckpt 3 任务 × 20 = 60 rollout，`success_rate > 10%`，
   过了才跑主网格；不过走 §6.3 的降级路径。

**跑法：**

```bash
# 门禁自检（无 GPU，~8 min）
bash scripts/run_rollout_tests.sh --trials 2

# 真闭环
source /workspace/ttdu/ttd/scripts/env_eval.rc
conda activate ttd_rollout
xvfb-run -a python eval/rollout.py --ckpt <path> --tag official \
    --tasks push_buttons open_drawer meat_off_grill --num-trials 20

# 配对比较
python eval/aggregate_closedloop.py --compare \
    reports/closedloop/rollout_arm0.json reports/closedloop/rollout_arm1.json
```

---

## 9. 代码改动记录（2026-08-12，§1 已落地）

三个文件，**没有改动任何既有函数的行为**——新增为主，唯一的既有代码改动是加注释。

### 9.1 `training/utils.py` —— 新增三个函数

| 函数 | 位置 | 作用 |
|---|---|---|
| `decode_gripper_openness_torch(blue, mode=...)` | 紧接 `fuse_multiview_heatmaps_to_7d_point_torch` 之后 | 论文 **Eq. (7)**：估计蓝通道背景电平 ÷ 0.25 → 连续 openness。`mode="paper"` 用 `<=`（原因见下）、`mode="median"` 用中位数（生成图上更稳） |
| `rotation_matrix_to_quaternion_xyzw(rot)` | 同上 | 旋转矩阵 → `(qx,qy,qz,qw)`。纯 torch、不引 scipy，序与 PyRep/RLBench `gripper_pose[3:7]` 一致 |
| `fuse_multiview_heatmaps_to_pose_torch(...)` | 同上 | **论文 §3.2 的完整解码器**。三通道全三角化 → `e_x`/`e_z` → `e_y = e_z × e_x` → 再正交化 → 返回 `[x,y,z,qx,qy,qz,qw,openness]`，直接就是 RLBench 末端位姿动作 |

三处**偏离论文字面、但有实测依据**的实现决定，都写在 docstring 里：

1. **Eq. (7) 用 `<=` 而不是论文的 `<`。** 编码侧 `project_action_5d_to_rgb_torch:451-452`
   用 `B <= 0.25` 填充、在 open 帧上填的值**正好是 0.25**；论文用严格 `<` 选集合，
   于是 open 帧选中 **0 个像素**（closed 帧选中 128214 个），公式恒读 0。
   实测断言在 `test_decode_6dof.py` 的 `PAPER_EQ7_STRICT_LT_IS_DEGENERATE_OK`。
2. **多一步再正交化。** 论文给的 `e_x`、`e_z` 解出来只是**近似**垂直（实测 89.94°），
   直接堆成矩阵不是合法旋转、IK 会拿到无意义目标。做法：保留 `e_x`（G 通道，没有 openness
   污染），`e_y = norm(e_z × e_x)`，再 `e_z ← e_x × e_y`。等价于以 `e_x` 为锚的 Gram-Schmidt。
3. **蓝通道三角化前先扣掉 `0.25·ĝ` 的背景电平**（`strip_openness_pedestal=True`）。
   论文没提这一步。不做的话整旋转误差 **0.63° → 11.81°**，见 §1.5。

### 9.2 `training/utils.py` —— `fuse_multiview_heatmaps_to_7d_point_torch` 只加注释

**逻辑一行未改**（它是官方基线，`inference.py:298` 的点云导出还在用，且那个消费者只取 `[:3]`，
两个缺陷都影响不到它）。docstring 里加了一段 `UPSTREAM FUNCTION -- DELIBERATELY LEFT AS-IS`，
写明两个缺陷（只三角化 R/G ⇒ 滚转欠定；`any(B>128)` 被 blob 触发 ⇒ 夹爪位恒为 closed，acc=0.500）
并指向新函数。

### 9.3 `tests/test_decode_6dof.py` —— 新增

按仓库既有约定写成 `python -u` 直跑的断言脚本（不是 pytest），末尾打 `ALL_DECODE_6DOF_TESTS_PASSED`。
七组断言：位姿往返可执行、解码旋转正交且 det=+1、四元数与 scipy 逐元素一致、
两种 gripper 模式都精确、Eq.(7) 严格 `<` 的退化（把这个坑钉住）、
**shipped 解码器的两个缺陷钉成回归测试**（断言它是错的——将来若有人"修好"了上游，
这两条会失败，提醒去复核建立在它上面的历史数字）、以及背景电平扣除确实有收益。

### 9.4 `scripts/run_tests.sh` —— 新增一行

`tests/test_decode_6dof.py` 挂进 CPU 套件，排在 codec 测试之前。

### 9.5 `train.py` —— 修一个**既有的语法错误**（与本次改动无关，顺手发现）

跑全量套件时暴露的：`train.py` 第 316–327 行有一个**顶格写在缩进方法体里**的 `'''…'''`
块注释（说明 `visual_latents` 结构的示意），它终止了缩进块，导致第 330 行
`IndentationError: unexpected indent`。

**后果：`train.py` 从 2026-08-11 01:07 起就无法被 import**，
`tests/test_checkpoint_pruning.py`（唯一 `from train import ...` 的测试）因此一直红着。
**训练入口本身也是坏的**——`--template_mix` 的任何续跑在这个文件被修好之前都起不来。

改法：把那段示意改写成与上下文同风格的 `#` 注释并正常缩进。**没有改动任何逻辑**，
`ast.parse` 通过，`test_checkpoint_pruning.py` 恢复全绿。

---

### 9.7 闭环 rollout 工装（2026-08-12）

| 文件 | 内容 |
|---|---|
| `eval/rollout_env.py`（新） | RLBench 封装。`front`+`overhead` @256²、`EndEffectorPoseViaIK`（可切 `planning`）、`reset_to_new_demo`、相机参数按 `obs.misc` 读、IK 失败**报告而非抛出**、`RolloutStats` 记账 |
| `eval/policy.py`（新） | starVLA `model2bench_interface.py` 的契约。`run_policy(rgb[2,H,W,3], extr[2,4,4], intr[2,3,3], prompt) -> [41,8]`。三个约定（extrinsics 绝对 c2w / camera 相对 view1 首帧 / intrinsics 负焦距）在构造时断言。附带 `last_debug` 记 `r_peak_mean` 等归因仪表 |
| `eval/rollout.py`（新） | receding-horizon 循环，形状抄 `eval_libero_local.py:269-285`。`--gt-replay` 走同一循环但喂 demo 自己的动作。JSON **增量写盘**（campaign 被 kill 也不丢） |
| `eval/aggregate_closedloop.py`（新） | **精确二项 McNemar**（不是卡方近似——判别对少时近似不成立）+ 配对 bootstrap CI。会**拒绝**场景 seed 不一致或无交集的配对 |
| `tests/test_policy_conditioning.py`（新，CPU） | 条件张量三约定 + canvas 布局 + 解码往返 + **视角互换负控制** |
| `tests/test_aggregate_closedloop.py`（新，CPU） | McNemar 对手算值 + CI 与 p 值一致性 + 三类拒绝路径 |
| `tests/test_rollout_env.py`（新，带仿真） | GT 重放门禁，见 §6.4 |
| `scripts/run_rollout_tests.sh`（新） | 上面这条的 runner，包好 `env_eval.rc` + `xvfb-run` + `ttd_rollout` |
| `scripts/run_tests.sh`（改） | 把两个 CPU 测试挂进快速套件 |

**实现时量到、文档里原本没有的三件事：**

0. **⚠️ 当前末端位姿是"条件"、不是"预测"。** `prepare_template_inference_latents` 和官方
   `prepare_action_images_inference_latents` 都会给**每个 segment 的第一帧 latent** 上锚
   （`wan_video_action_images.py:531`），所以 **action 段的第 0 帧是喂进去的**。
   官方推理用数据集里的真实首帧动作填它；闭环里的对应物就是**机器人当下的位姿**。
   最初写成 `torch.zeros(1,T,7)` ——那等于告诉模型"末端在世界原点、姿态是单位阵"，
   每个 chunk 都会先瞬移过去。已改成从 `env.current_pose8(obs)` 转 7 维填入，
   `run_policy` 现在**强制要求** `current_pose8`，缺了直接报错。
   `test_policy_conditioning.py` 有 `MISSING_CURRENT_POSE_REJECTED_OK` 与
   200 组随机旋转的 `pose8 → action7` 约定比对（xyzw 四元数 / xyz 欧拉，
   与 `dataset/rlbench.py:113-115` 逐位一致）。

1. **`ĝ` 有 1.2% 的 uint8 量化地板。** 编码侧背景电平是 `0.25×255 = 63.75`，生成帧是 uint8，
   截断成 63 ⇒ `ĝ_open = (63/255)/0.25 = 0.9882` 而不是 1.0。二值判定有 0.49 的余量、完全无害，
   但**报告里不能拿 `ĝ` 和 1.0 直接比**。`test_decode_6dof.py` 在 float 域看不到这一条，
   `test_policy_conditioning.py`（走 uint8 canvas）才看得到。
2. **`ViaPlanning` 不慢**，见 §2.4 的更正表。

### 9.6 `--frame_interval`（§2.2.1 的实现，2026-08-12）

五个文件。**默认值 1 使既有行为逐字节不变**，arm0/arm1 不受影响。

| 文件 | 改动 |
|---|---|
| `training/args.py` | 新增 `frame_interval: int = 1` 字段，注释里带上 §2.2.1 那张实测表 |
| `train.py` | 调用点 `frame_interval=1` → `args.frame_interval`；wandb config 增记 `frame_interval` 与 `window_span_seconds`（**两个只差这一项的 run 在 wandb 里本来分不开**） |
| `training/helpers/io.py` | ① 拒绝非正整数 `frame_interval`；② **修 `total_frames <= max_frames` 分支**——它原本完全无视 `frame_interval`，对短视频静默给出 stride-1 窗口，等于把两种时间尺度的样本混进同一个 run。改成先按 stride 取、再补末帧；**`F=1` 时与上游逐元素相同**（`min(i,total−1)` 复现 `range(total)+[total−1]*pad`），已用 89 个长度 × 3 个起点验证 |
| `scripts/train_arm.sh` | `FRAME_INTERVAL` 环境变量（默认 1）；**非 1 时 `OUT` 加 `_fi<N>` 后缀**——否则 `FRAME_INTERVAL=3` 跑 arm0 会命中脚本自己的 resume 分支、静默续跑那个 20 Hz 的臂；启动时打印跨度与频率 |
| `tests/test_frame_interval.py` | 新增，挂进 `run_tests.sh`。三组：`F=1` 上游 parity、短视频分支的 stride、真实 episode 上五流同步（逐元素比对 `actions.npy[frame_indices]`）+ `F=1` 负控制 |

实测输出：

```
UPSTREAM_PARITY_AT_INTERVAL_1_OK  (89 lengths x 3 starts, indices identical)
SHORT_VIDEO_BRANCH_STRIDES_OK  total=35 interval=3 -> [0,3,6,9,12]...[34,34,34]（上游给的是 stride 1）
SPAN_AND_WINDOW_THRESHOLDS_OK  no padding iff total >= 1+(N-1)*F; single window iff total <= N*F
FRAME_INTERVAL_VALIDATION_OK  (0, -1, 1.5, '3' all rejected)
SELFGEN_STREAMS_STRIDE_TOGETHER_OK  interval=3, 6 samples verified against actions.npy on disk
  last: episode5 len=96 -> window 0..95 (95 native steps = 4.75 s, 8 repeated tail frames)
INTERVAL_1_STILL_CONTIGUOUS_OK  window 32..72 (2.00 s)
```
