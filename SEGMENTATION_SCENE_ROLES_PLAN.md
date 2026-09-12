# RLBench 全场景角色分割方案 (Scene Role Segmentation v1)

> 状态：**Phase A / B2 已实现**（codec、dataset、M 轴、元数据生成器、19 项测试）；
> 训练与评估未启动。实现细节与实测结果见
> [SCENE_ROLES_IMPLEMENTATION_REPORT.md](SCENE_ROLES_IMPLEMENTATION_REPORT.md)
> 版本：v2（2026-08-12 修订，替换 v1 初稿）
> 适用仓库：`/workspace/ttdu/ActionImages-Cogen`

本版相对初稿的主要修订：证据章节全部替换为在全数据树上实测的数字；palette 分配、decode
作用域、元数据真相源、loss 加权的落地链路四处技术错误已修正；实验设计补上样本量、判定阈值
和失败准则；数据重生成一项按实测覆盖率降级为可选。

---

## 0. 交付物定位：segmentation 是辅助流，闭环成功率才是交付物

**这一节决定了后面所有取舍，先读。**

按 [EVAL_PLAN_CLOSEDLOOP.md](EVAL_PLAN_CLOSEDLOOP.md)，本项目的交付物是**闭环成功率**
（对照 Table 3 `Ours` = 20.6%），开环指标"降级为仪表与门禁，不再是交付物"。而闭环 rollout
的协议是（§2.5、§7 交付物表）：

```text
template   = "video+action"          ← 序列里只有 video 和 action
conditioning = i2va（只给首帧）        ← 即 IIII
每次 replan 的条件 = 当前两视角的真实观测帧
```

**segmentation 流在 rollout 序列里根本不存在，永远不会被生成或消费。**

由此推出三条贯穿全文的结论：

### 0.1 seg 的唯一收益路径是"辅助表征"

scene_roles 能影响交付物的唯一途径是：
**更好的辅助监督 → 共享 backbone 学到更好的场景表征 → `video+action` 的世界模型更准 → 闭环成功率更高。**

这是一条长而弱的因果链，中间每一环都有噪声。方案必须诚实地承认这一点，并据此
（a）把闭环成功率设为主指标，（b）把 seg IoU 降级为诊断量，（c）增加一道前置门（§0.3）。

### 0.2 这给 M 轴一个原则性理由（不只是"co-generation 更好听"）

rollout 跑在 **IIII + 生成式** régime 里。而当前 perception 分支 90% 是 FIFI，训练的是
"给定 RGB 判别式地读出 seg"。**辅助任务的训练 régime 与部署 régime 不匹配**，学到的表征
迁移路径很弱。

把 perception 也拉到 IIII 为主（§10.3 的 M2），辅助任务就与 `video+action` 跑在同一个
régime 下：同样从首帧生成、同样要维持时间一致性。**共享表征的迁移假设这时才成立。**

这比 §10.3 原来写的"解耦任务身份与 conditioning 等级"更根本，也更直接指向交付物。

### 0.3 前置门：辅助流到底有没有用？

如果 depth/seg 辅助流对闭环成功率**根本没有贡献**，那 scene_roles 无论把 seg IoU 做到多高
都没有收益路径。所以在投入 Phase A 的 16 个 task resolver 之前，必须先回答：

> **arm1（`video+action@0.6, video+depth@0.2, video+segmentation@0.2`）的闭环成功率，
> 是否高于 arm0（`video+action@1.0`）？**

这道门（§3.0）优先级**高于**本文其余全部诊断。它的三种结果对应三条完全不同的路：

| 结果 | 含义 | 行动 |
|---|---|---|
| arm1 > arm0（显著） | 辅助流有用，值得改进辅助流的质量 | 全速执行本方案 |
| arm1 ≈ arm0 | 辅助流不痛不痒，可能是质量太差（本方案的赌注），也可能是路径不通 | 先跑 M 轴（改一个常数），再决定 S 轴 |
| arm1 < arm0（显著） | 辅助流在**抢容量**，损害交付物 | **停止本方案**，转而研究如何减少辅助流干扰 |

注意 `EVAL_PLAN_CLOSEDLOOP.md` §0 自己标注了"地板效应仍是真风险"。若 arm0/arm1 闭环成功率
都接近 0，这道门给不出信号，此时退化为用开环 action 指标（`pos_err`）代理，并在报告里写明
结论强度降级。

### 0.4 对全文的影响索引

| 章节 | 因 §0 而改变的地方 |
|---|---|
| §1 | "没学好"仍成立，但它是**诊断**不是交付物缺口 |
| §3 | 新增 §3.0 闭环前置门，排在所有诊断之前 |
| §10 | arm 判定以闭环成功率为主；action 指标退化从"记录"升为**主判据** |
| §11 | 指标分三层：交付（闭环）/ 归因（action 开环）/ 诊断（seg IoU） |
| §14 | Phase A 的启动条件挂在 §3.0 的结果上 |

---

## 1. 背景：问题是什么，证据到哪一步

### 1.1 现象

`comparisons_all_segments/` 里 arm1（`video+action@0.6,video+depth@0.2,video+segmentation@0.2`）
的 segmentation 结果：

| checkpoint | IIII | FIII | **FIFI** |
|---|---:|---:|---:|
| arm1_step1000 | 0.029 | 0.458 | **0.545** |
| arm1_step1500 | 0.025 | 0.263 | **0.082** |
| arm1_step2000 | 0.023 | 0.272 | **0.141** |
| arm1_step2500 | 0.012 | 0.115 | **0.047** |

读这张表必须先分清哪一列是 in-distribution。**这里有一个极易误读的点**：官方
[ActionImages/train.py:319-326](/workspace/ttdu/ActionImages/train.py#L319-L326) 的 mode-mix 是
90% IIII / 5% FIII / 5% FIFI，但那条规则只适用于**带 action 流**的模板 —— 官方代码里只有
`[video_src, action_src, video_tgt, action_tgt]` 四个 segment，压根没有 segmentation 模态。

fork 的 [templates.py:215-233](training/templates.py#L215-L233) 分成两个分支：

| 模板 | 分支 | 90% 的配置 |
|---|---|---|
| `video+action`（官方配方，arm1 的 60%） | `if has_action`（L215-228，逐字复刻官方 90/5/5） | **IIII** |
| `video+depth` / `video+segmentation`（arm1 的 40%） | `elif len(mods) > 1`（L229-233，fork 新增的 perception 分支，`PERCEPTION_VIDEO_GIVEN_PROB = 0.9`） | **FIFI** |

perception 分支是 fork 相对官方的一处主动分歧，理由写在代码注释里："the point of the sample is
to read a modality off the RGB, so the RGB is given nearly always"。因此对
`video+segmentation` 而言：

- **FIFI 是 in-distribution（90%）**，把 RGB 整段给全，模型只需从 RGB 读出 segmentation。
- IIII 是剩下 10%：RGB 不给，模型自由生成场景。生成的视频与 GT episode 本就不是同一段像素，
  对着 GT mask 算 IoU 意义有限，0.02 不能直接当作"没学会"（详见 §1.2 的时间错位混淆）。
- FIII 需要拆开说，容易搞混。`video+segmentation` 的 segment 布局是（view-major, modality-minor）：

  ```
  [ video_v0 , seg_v0 , video_v1 , seg_v1 ]
      F         I        I         I         ← FIII
  ```

  - **训练时**：perception 分支下 FIII 从不出现（`give_first_segment` 只在 `has_action` 分支里
    设置）。arm1 训练时产生过的 FIII 样本全部来自 `video+action`，那些序列里确实没有 seg 段。
  - **但 eval 表的 FIII 列**是一条完整的 `video+segmentation` 序列 —— 第 2、4 位就是两个视角的
    seg 段，各给首帧、其余预测。seg **在**序列里，被预测。它 OOD 的地方仅仅是"这个 mask 组合
    没有被训练过"。
  - 因此 FIII = 0.458 是自洽的：view0 的 RGB 整段给全，场景被钉死，seg_v0 自然可预测。它比
    IIII 高一个数量级，正说明"RGB 可得时模型能做 RGB→seg"。

所以唯一可解释的是 FIFI 列。人工看视频的结论（FIFI 生成还行、其余不行）与这一点一致，并且
FIFI 自身 **0.545 → 0.047 单调塌陷**。结论：在它唯一被训练的配置下，segmentation 既没有练到位，
也没有稳住。

需要同时记住两个削弱因素，它们不改变"没学好"的结论，但决定了归因：

1. **训练暴露量极小**：seg 只占 arm1 的 20%，2500 步里约 500 步。
2. **评估样本量为 1**：整张表来自单个 **seen** episode `open_drawer/variation0/episodes/episode0`、
   单 seed。0.545↔0.047 的摆幅比任何预期的方案间差异都大。

### 1.2 IIII 的时间错位混淆（为什么只解读 FIFI）

IIII 那一列不能当作"没学会"的证据，因为它混进了一个结构性的时间歧义。

采样窗口是 `num_frames=41 × frame_interval=3` = 跨 121 个原生步，而 episode 长度（**实测**）
在 71–334 之间。一个窗口覆盖任务的比例因此是：

| task | T | 窗口覆盖比例 | 重复的末帧数 |
|---|---:|---:|---:|
| push_buttons | 71 | 100% | 16 |
| open_drawer | 96 | 100% | 8 |
| close_jar | 157 | 77% | 0 |
| put_item_in_drawer | 291 | 42% | 0 |
| stack_blocks | 334 | 36% | 0 |

IIII 只给每段第 0 帧，而**从第 0 帧无法判断这个窗口要覆盖整个任务还是只覆盖 36%** —— 两者首帧
一样。模型只能回归训练集的平均节奏，于是在长 episode 上系统性地"跑到未来"。
[helpers/io.py:44-47](training/helpers/io.py#L44-L47) 里短 episode 还会把末帧 hold 住最多 16 帧，
进一步加大节奏方差。

两个必须区分的层面：

- **训练时模型并不"生成"**。diffusion loss 在 `add_noise(GT_latents)` 上做去噪，条件永远是真实
  GT。节奏歧义在训练时表现为**不可约 loss**（同一条件对应多个合法未来），不是轨迹分叉。
- **推理时才分叉**。此时对着某个 GT episode 逐帧算指标就变成了混淆量。

IoU 对时间错位是**断崖式**敏感的：一个完全正确的 mask 偏 10 帧，IoU 就接近 0。所以
`IIII seg IoU = 0.02` 至少有两种读法 ——（a）没学会分割；（b）分割很好但画的是另一个时刻。
现有数据分不开。同一张表里 action 的 IIII（0.06–0.15 m）远差于 FIFI（0.013 m）是同一个混淆，
但位置误差对时间错位是平滑的（早到晚到仍在同一条轨迹附近），所以 action 受影响小得多。这个
不对称正说明 seg 的 IIII 数字最不可信。

**结论：FIFI 之所以是主指标，不只因为它 in-distribution，更因为它把两个 RGB 段整段给全，
整段时间演化被钉死，segmentation 成为给定像素的确定性函数，没有节奏自由度 —— 这是唯一
不含该混淆的测量。** 对应的诊断见 §3.4，评估协议见 §11。

> 附带发现（超出本方案范围，但与 §1.1 列出的候选根因有关）：这条节奏歧义给
> `video+action` 的 90% IIII 样本注入了不可约噪声，是 arm 级训练不稳定的一个独立候选解释。
> 可能的缓解方向是按 episode 长度归一化 `frame_interval`（让每个窗口覆盖固定的任务**比例**
> 而非固定的原生步数），或把 episode 进度显式喂给条件。建议单独立项，不要混进本方案。

### 1.3 本方案的假设（两级）

**一级假设（表征层）**：稀疏的 target-only 监督（中位数约 1% 前景像素、其余全黑）给模型的
有效梯度太少，且经 VAE 8× 空间 / 4× 时间压缩后进一步衰减，导致 segmentation 学得慢、学得不稳。
用稠密的、跨任务统一的场景角色标签替代，可以把有效监督面积提高一个数量级。

**二级假设（交付层）**：更好的辅助监督会通过共享 backbone 提升 `video+action` 的世界模型质量，
从而提高闭环成功率。

两级都**尚未被证明**，而且二级假设比一级弱得多——它是一条长因果链（§0.1）。**方案的价值完全
取决于二级假设，而二级假设的前提由 §3.0 的闭环前置门检验。**

§3 的其余 P0 诊断与主线并行执行，用于事后归因，不阻塞 Phase A。但如果 §3.2 的 VAE roundtrip
失败，dense 标签在数学上不可能解决一级假设的问题，届时必须先改分辨率。

### 1.4 目标

1. 使用 RLBench selfgen 已保存的完整 segmentation handle map，而不只是 target handles；
2. 把稀疏 target-only 监督改成稠密、跨任务统一的场景角色分割监督；
3. 保留 target 与 goal 的任务语义，使输出仍受自然语言指令控制；
4. 避免 raw simulator handle、不稳定实例顺序和过多颜色引入额外歧义；
5. 完整保留旧 referring 协议，保证旧 checkpoint 与已有实验逐位可复现；
6. **不损害 `video+action` 的闭环成功率**——这是硬约束，不是软目标（§10.5）。

---

## 2. 数据格式（已核对）

segmentation GT 是每个视角目录下的：

```text
<episode>/view{1..4}/mask.npz     # 不是 mask.npy
```

```python
handle_map = np.load("mask.npz")["mask"]   # [T, H, W], uint16, 值为 CoppeliaSim object handle
```

相关位置：

- 数据生成：`/workspace/ttdu/ttd/scripts/gen_dataset.py:128-131`
- 数据读取：[rlbench_selfgen.py:305-307](training/dataset/rlbench_selfgen.py#L305-L307)
- handle → 物体名：每 episode 的 `handles.json`
- 当前 referred target：每 episode 的 `seg_targets.json`（`protocol: rlbench-task-source-v1`）

当前编码链路（[rlbench_selfgen.py:374-379](training/dataset/rlbench_selfgen.py#L374-L379)）：

```text
mask.npz → 完整 handle map → seg_targets.json + instruction → 少量 target handles
        → np.isin → target 涂色，其余全黑
```

数据树规模（`data/rlbench_selfgen` → `ttd/data/rlbench_selfgen_v2`）：

| | episode 数 |
|---|---:|
| 全树 | 2818 |
| variation0（训练） | 1398 |
| variation1+2（留出） | 478 |

---

## 3. P0：前置门 + 四项诊断

§3.0 是**前置门**，优先级高于其余全部；§3.1–3.4 不需要任何标注工作，与主线并行，产出物同时
是后续实验必需的基线。硬性否决点有两个：§3.0 和 §3.2。

### 3.0 闭环前置门 —— **已放行（2026-08-14，项目决定）**

> **状态：放行，不阻塞 Phase A。** 依据是两条，必须分开记：
>
> 1. **这是本研究的核心假设，不是待检验的中间结论。** "辅助感知流提升世界模型/策略"是项目
>    的立项前提；把它降级成一道可能否掉全项目的门，是把研究问题当成工程风险来管，方向错了。
> 2. **已有闭环证据方向为正**：`reports/closedloop/CAMPAIGN_200.md`，100 rollout × 2，
>    arm4（`video+action@0.6,video+depth@0.4`）**31/100 vs 官方 20/100**，
>    `close_drawer` +40 pp（p=0.0078）、`push_buttons` +35 pp（p=0.0391），合计 p=0.052、
>    95% CI `[+1.0, +21.0]` pp。
>
> **但要诚实记下这条证据的边界，免得日后被当成比它更强的东西引用：**
>
> - 那是 **arm4 vs 官方 checkpoint**，不是 **arm4 vs arm0**。`CAMPAIGN_200.md` §2.2 自己写明
>   "显著性说明的是'领域内微调有效'，不是 arm4 比官方模型强"——arm4 in-domain、官方 zero-shot，
>   分辨率与 cfg 也不同。**它没有隔离出"辅助流"这一个变量**；能隔离的只有 arm0 vs arm4
>   （两者都 in-domain，只差 `video+depth@0.4`）。
> - 那条辅助流是 **depth**，不是 segmentation；迁移到 seg 属于类比，不是实测。
> - `outputs/arm0__seed42_fi3` 已存在，所以真正的 arm0 vs arm4 配对闭环比较成本很低，
>   将来若需要一条能单独支撑"辅助流有用"的证据，跑它即可。
>
> 下面保留原门禁设计，供将来需要该证据时直接执行。

#### （原设计，保留备用）辅助流对交付物有没有贡献

**问题**：arm1（含 depth+seg 辅助流）的闭环成功率是否高于 arm0（纯 `video+action`）？

**为什么排第一**：seg 流在 rollout 序列里不存在（§0），它影响交付物的唯一路径是辅助表征。
若该路径不通，把 seg IoU 从 0.5 做到 0.9 也换不来一点成功率。这道门决定后面是否值得投入
Phase A 的 16 个 task resolver。

**怎么跑**：复用 `EVAL_PLAN_CLOSEDLOOP.md` 已钉死的协议，不要新发明——
`eval/policy.py` 与 `tests/test_policy_conditioning.py` 已完成。主协议 `E=41`，
`max_steps = 2 × len(GT demo)`，每格 20 次 rollout，arm0/arm1 各跑同一组 (task, seed)。

**判读与行动**：见 §0.3 的三分支表。

**地板效应的兜底**：`EVAL_PLAN_CLOSEDLOOP.md` §0 已标注地板效应是真风险。若 arm0/arm1
成功率都接近 0，这道门无信号，退化为用开环 action 指标（`pos_err`、`r_peak_mean`）代理，
并在报告里显式写明"结论强度降级，二级假设未被检验"。**不要把无信号读成通过。**

### 3.1 扩大 S0 评估面

现有 seg 证据是 n=1 seen episode。用 §11 定义的评估协议（≥20 held-out episode × 3 seed × 2 视角）
重跑现有 arm1 的四个 checkpoint，得到带置信区间的 S0 曲线。没有这条曲线，S1 跑完也无法判断
差异是否超过噪声。

### 3.2 VAE per-class roundtrip（硬性判据）

把 GT 编码后的 segmentation 视频过一遍 VAE encode → decode，不经任何模型，测每类 IoU。
实现：`scripts/vae_roundtrip_seg.py`。

**两条实现要求，都不是可选项（实测踩过）：**

1. **逐视角编码。** Wan VAE 时间压缩 4×，`T_lat = 1 + (T-1)/4`，仅当 `T ≡ 1 (mod 4)` 整除。
   `num_frames=41` 每视角满足，拼接后的 82 帧不满足——整段喂进去会静默返回 81 帧。按
   `train.py` 的做法逐视角编码。

2. **必须带 RGB PSNR 对照，否则结果不可信。** 一个未正确加载的 VAE 会对**每一类**都给出近零
   IoU，读起来与真正的否决完全一样。首次实现时 `load_state_dict(..., strict=False)` 加裸
   `except: pass` 让随机初始化的 VAE 跑出了 `target IoU = 0.028` 的"否决"结论——破绽是
   `background`（占 85% 像素）也只有 0.0018，物理上不可能。修正加载后同一批数据得到
   `RGB PSNR ≈ 30 dB, target IoU = 0.998`，**结论完全相反**。
   因此脚本在 RGB roundtrip < 15 dB 时直接 ABORT 并声明所有 IoU 无效，而不给判词。
   通则：**加载失败必须表现为崩溃，不能表现为发现。**

- 若 **target-only 协议下 target 的 roundtrip IoU 已经很低**（判据：中位数 < 0.5），则信号是在
  VAE 里丢的，不是在监督量上丢的。dense 角色标签不改变 target 的像素数，救不了这个问题。
  此时必须先解决分辨率/压缩（提高 segmentation 流分辨率、或裁剪到目标区域），**暂停 Phase B/C**。
- 若 roundtrip IoU 高（> 0.8），VAE 不是瓶颈，本方案继续。

#### ✅ 已执行（2026-08-14）：**通过**

48 episodes / 14 tasks / 256² / 41 帧，完整输出 `reports/vae_roundtrip_seg.txt`：

| | 中位 | p10 | min |
|---|---:|---:|---:|
| RGB PSNR（对照） | 29.59 dB | — | 28.12 dB |
| referring `target` | **0.9910** | 0.9523 | 0.4957 |
| scene_roles `target` | 0.9646 | 0.8570 | 0.6590 |

**判定：0.9910 ≫ 0.8，VAE 不是瓶颈，方案继续。**

这条结果同时**否掉了 §1.3 一级假设的后半句**——"经 VAE 8×/4× 压缩后信号进一步衰减"。
在 256² 下中位大小 1% 的目标过 VAE 几乎无损，seg 没学好的原因不在 VAE。归因应转向
训练暴露量与训练稳定性（§1.1 的两个削弱因素）。

两个不构成否决、但必须带进后续报告的细节：

1. **最小目标桶有长尾。** 按 GT 面积分桶后 `<0.25%` 桶中位 0.9776 但 **p10 = 0.5361、
   min = 0.4957**；最差值全部来自 `sweep_to_dustpan` 的 dirt（5 个 4–8 px handle）。
   §11 报告 seg 指标时 `<0.25%` 桶必须单列，否则被中位数掩盖。

   | GT 面积 | n | median | p10 | min |
   |---|---:|---:|---:|---:|
   | <0.25% | 33 | 0.9776 | **0.5361** | 0.4957 |
   | 0.25–1% | 21 | 0.9930 | 0.9872 | 0.9514 |
   | 1–5% | 21 | 0.9945 | 0.9769 | 0.9756 |
   | >5% | 7 | 0.9963 | 0.9938 | 0.9920 |

2. **dense 协议反而略微降低 target 的 roundtrip 上界**（0.9910 → 0.9646，p10 0.952 → 0.857）。
   原因是颜色邻居变近：referring 里 target 只跟黑色竞争（色距 >230），scene_roles 里周围是
   goal/fixture/distractor（最近 98–135）。这是 dense 表示的一项实测代价，初稿只论证了收益。
   幅度不大，但它是所有 scene_roles 结果的**上界**，报告 S1/S2 时必须一并给出。

这一项成本约半天，却是唯一能否决整个方案的检查，优先做。

### 3.3 loss 归因

在现有 arm1 配置下，统计 segmentation segment 的 latent MSE 中 target 区域所占比例。这是
"target 信号被背景淹没"这一说法唯一的直接证据，也是 §8 权重初值的标定依据。

### 3.4 时间对齐 IoU（拆开 IIII 的混淆）

复用 `comparisons_all_segments/` 里**已经生成好的** IIII 输出，不需要重新采样，成本约 1 小时。
对每个预测帧 `t`，计算它与 GT 帧 `t-k … t+k`（k 取 5、10、20）的最大 IoU，再取全序列平均；
或先做 DTW 对齐再算逐帧 IoU。

判读：

- **aligned IoU ≫ raw IoU** → 模型学会了分割，只是时间不同步。IIII 的 0.02 是评估伪影，
  §1.1 里"没学好"的证据基础就只剩 FIFI 的塌陷一条，归因应转向训练稳定性而非标签稀疏度。
- **aligned IoU ≈ raw IoU** → 时间错位不是主因，IIII 确实画不出有效 mask，与 FIFI 的塌陷
  相互印证。

无论哪个结果，§10.5 的失败准则都不变；这一项只影响事后归因的写法。同时把
`best-offset` 的分布记录下来 —— 若系统性为正（预测超前于 GT），即 §1.2 预测的"跑到未来"。

---

## 4. 稀疏度实测

**实测方法**：全部 16 个任务的 `variation0/episodes/episode0`，view1 + view2 的**全部帧**（不是抽样帧），
按 `handles.json` 的物体名归类。`target` 取自该 episode 的 `seg_targets.json`。

| 任务 | referred target | 所有物体 | 机器人 | 静态区域 |
|---|---:|---:|---:|---:|
| close_jar | 0.23% | 7.56% | 10.45% | 82.00% |
| insert_onto_square_peg | 0.77% | 2.57% | 8.61% | 88.81% |
| light_bulb_in | 0.31% | 5.87% | 9.50% | 84.63% |
| meat_off_grill | 0.45% | 16.35% | 8.80% | 74.85% |
| open_drawer | 1.05% | 23.78% | 10.25% | 65.97% |
| place_shape_in_shape_sorter | 1.03% | 1.17% | 6.97% | 91.85% |
| push_buttons | 1.58% | 2.05% | 9.91% | 88.04% |
| put_groceries_in_cupboard | 27.50% | 33.33% | 6.15% | 60.52% |
| put_item_in_drawer | 3.61% | 27.33% | 8.89% | 63.78% |
| put_money_in_safe | 15.36% | 19.63% | 6.34% | 74.03% |
| reach_and_drag | 1.05% | 1.05% | 8.51% | 90.44% |
| slide_block_to_target | 0.39% | 0.96% | 9.38% | 89.65% |
| stack_blocks | 1.02% | 2.49% | 7.86% | 89.65% |
| stack_wine | 0.87% | 7.53% | 10.36% | 82.12% |
| sweep_to_dustpan | 6.18% | 11.29% | 8.67% | 80.04% |
| turn_tap | 0.50% | 2.79% | 10.79% | 86.42% |
| **中位数** | **1.03%** | **5.9%** | **9.0%** | **84.6%** |

必须从这张表读出三件事，初稿只读出了第一件：

1. target-only 监督确实稀疏：中位数约 1%。
2. **跨任务方差达 ~32 倍**（按"所有物体"列：`reach_and_drag` 1.05% vs
   `put_groceries_in_cupboard` 33.33%）。若改按"非 background"列算则是 4.1×，因为机器人稳定
   占 ~9% 拉平了比值 —— 两个口径都要报，不要混用：`reach_and_drag` / `slide_block_to_target`
   的所有物体加起来才 1%，
   而 `put_groceries_in_cupboard` 有 33%。任何全局固定的 loss 权重都会在两端之一失配，所以
   §8 的权重必须按 label 定义、逐样本归一化，不能按任务硬编码。
3. **"dense" 是相对的**：即使编码全部角色，background 仍占 60–92%（中位 85%）。本方案把
   有效前景从 ~1% 提到 ~15%，不是提到 100%。文档里不要再用"稠密"暗示背景问题被解决了。

第三点直接推出一个初稿没有面对的风险，见 §10.1。

**产出要求**：这张表由一个存进仓库的脚本在全树（2818 episodes）上重新生成，而不是靠临时抽样。
脚本路径 `scripts/seg_pixel_stats.py`，输出 CSV 进 `reports/`。

---

## 5. 为什么不能按 handle ID 上色

### 5.1 Handle ID 没有跨任务语义

同一数值在不同任务里代表完全不同的物体。**实测**：handle `44` 在多数任务是
`Panda_link2_visual`，在 `put_groceries_in_cupboard` 里则是另一个物体。所以
`color = palette[handle_id]` 和把 uint16 直接编码进 RGB 三通道都不可行。

### 5.2 Handle ID 不跨 episode 稳定

`sweep_to_dustpan` 的 dustpan / dirt / broom 等 handle 会随 episode 变化，按 ID 上色会让同一物体
在不同训练样本里随机变色。

### 5.3 一个逻辑物体常含多个 handle

`shape_sorter` + `shape_sorter_visual`；push button 的 target / top plate / wrap；dustpan 的多个
可见部件；Panda 的各 link、gripper、fingers。按 handle 上色会把一个物体拆成多个实例。

### 5.4 调色板容量

**实测**每场景可见 handle 数 17–25（中位 20）。当前 codec 只有 8 个前景色。把几十个 handle 映射到
相近颜色会降低 VAE 后的可分性。这正是必须先做语义合并、再上色的原因。

---

## 6. 表示：全场景功能角色分割

### 6.1 核心原则

颜色绑定跨任务统一的**功能角色**，不绑定 simulator handle，也不绑定"第几个物体"。

```text
uint16 handle map → object name → 合并为逻辑物体 → 功能角色 → 固定全局 palette → dense RGB
```

同一角色在所有任务、episode、帧、视角中永远同色。多个同角色物体共享颜色；本协议解决
semantic/role segmentation，不解决开放数量的 instance segmentation。

### 6.2 固定颜色协议

复用 `seg_codec.py` 已标定的 8 色 palette + 黑色背景。

| Label | 角色 | RGB | 定义 |
|---:|---|---|---|
| 0 | `background` | `(0, 0, 0)` | floor / wall / table / workspace 等静态环境 |
| 1 | `target` | `(230, 25, 75)` red | 指令要求直接操作、移动或触发的物体 |
| 2 | `goal` | `(60, 180, 75)` green | 容器、接收物体或目标位置 |
| 3 | `robot_arm` | `(0, 130, 200)` blue | Panda arm links |
| 4 | `gripper` | `(70, 240, 240)` cyan | gripper 与左右 fingers |
| 5 | `distractor` | `(255, 225, 25)` yellow | 非目标可动物体与同类干扰物 |
| 6 | `tool` | `(245, 130, 48)` → **`(240, 50, 230)` magenta** | broom、stick 等中间工具 |
| 7 | `fixture` | `(145, 30, 180)` purple | drawer frame、grill、tap body、rack 等装置 |
| 8 | `unknown` | `(240, 50, 230)` → **`(245, 130, 48)` orange** | 未映射 handle，仅用于暴露标注错误 |

**`tool` 与 `unknown` 相对初稿互换，理由是实测的两两 sRGB 距离**：

```
 98.3  distractor  vs tool        ← 初稿把最小间距分配给了两个必然共现且语义相近的前景类
109.2  fixture     vs unknown
109.4  target      vs tool
135.2  target      vs fixture
136.4  robot_arm   vs gripper
```

`seg_codec.py` 的 `TAU = 40` 正是按 98.3 这个最小值标定的（`TAU < 98.3/2`）。把 palette 里最挤的
一对留给 `distractor`/`tool` 是最坏选择 —— `sweep_to_dustpan` 里 broom(tool) 与干扰物同框。互换后
最紧的一对变成 `distractor`/`unknown`，而 `unknown` 在合法训练数据里出现次数必须为 0，代价为零。

`unknown` 不是正常训练类别。正式训练前必须通过覆盖率检查；橙色只用于让漏标在调试时可见，
不能像黑色一样静默隐藏。

**palette 已用满**：9 个 label 占掉 8 个前景色 + 黑色，零余量。Phase D 的 instance 协议或将来拆分
`table` / `wall` 都需要新增颜色，而新增颜色必须重新标定 `TAU`（现值绑定在当前最小间距上）。这一点
写在这里，避免后来者以为还能随手加类。

### 6.3 角色优先级

```text
target > goal > tool > distractor > fixture > {gripper, robot_arm} > background
```

实际只有前两级会发生冲突：`target`/`goal` 是 instruction 相关的，覆盖在 instruction 无关的
`base_role` 之上。`gripper` 与 `robot_arm` 的 handle 集合互不相交，不需要优先级。

示例：

- `close_jar`：lid = `target`，jar body = `goal`
- `put_item_in_drawer`：item = `target`，指定 drawer = `goal`，其余 drawer 部件 = `fixture`
- `put_groceries_in_cupboard`：指定 grocery = `target`，cupboard = `goal`，其他 groceries = `distractor`
- `push_buttons`：指令要求按下的按钮 = `target`，其余 = `distractor`
- `sweep_to_dustpan`：dirt = `target`，dustpan = `goal`，broom = `tool`
- `open_drawer`：指定 drawer = `target`，frame/legs 及其他 drawer 部件 = `fixture`

多个连续目标共用红色（例如需要按多个按钮）。动作先后由 instruction 和 action stream 表达，
不由 segmentation 颜色承担。

---

## 7. 元数据

### 7.1 单一真相源

初稿把 `instruction_roles`（target/goal）写进新文件 `scene_segments.json`，同时又声称
"runtime 只做字典查找"。这与现有设计冲突：`build_referring_spec(seg_targets, instruction)` 是
**运行时**按 instruction 解析 target 的 —— `push_buttons`、`put_groceries_in_cupboard` 这类任务的
target 依赖指令文本。把结果固化成 JSON 会产生第二个真相源，一旦漂移，§11.1 里
"target/goal 与 instruction 对齐"这条验收门就形同虚设。

**修正：`scene_segments.json` 只存 instruction 无关的部分。**

```json
{
  "protocol": "rlbench-scene-role-v1",
  "task": "close_jar",
  "instances": {
    "jar_lid":       { "handles": [87],                         "base_role": "distractor" },
    "jar":           { "handles": [81, 85],                     "base_role": "fixture" },
    "robot_arm":     { "handles": [39,40,41,42,43,44,45,46],    "base_role": "robot_arm" },
    "robot_gripper": { "handles": [31, 34, 35],                 "base_role": "gripper" },
    "environment":   { "handles": [10, 48, 52, 55],             "base_role": "background" }
  }
}
```

target / goal 的解析链路不变：

```text
seg_targets.json + instruction --build_referring_spec--> {instance_name: handles}
                                                          ↓ 覆盖到 base_role 之上
scene_segments.json (instances + base_role) ------------> 完整 role LUT
```

好处：target 只有一个真相源；`build_referring_spec` 现有的 `dropped_groups` 统计（定义了但本
episode 解析不出 handle 的组）自动继承，GT 完整性的缺口继续可观测。

`goal` 的归属是新增信息，`seg_targets.json` 里没有。它按任务写进生成器的 resolver（见 §7.2），
产物落到 `instances` 的 `base_role`（如 `close_jar` 的 jar → `goal` 而非 `fixture`），因为在给定任务
里 goal 通常是 instruction 无关的。**只有当某任务的 goal 确实随 instruction 变化时**，才在
`seg_targets.json` v2 里加一个 `goal` 段，走同一条运行时解析路径 —— 不要新开第三个文件。

### 7.2 生成器

新增离线脚本 `/workspace/ttdu/ttd/src/percep/scene_segments_gen.py`，包含：

1. 全局 robot / static handle 规则（按 `handles.json` 的名字前缀 `Panda_*`、
   `ResizableFloor*` / `Wall*` / `*Table*` / `workspace`）；
2. 16 个任务各自的 logical-instance resolver（显式，不靠 substring 猜）；
3. task-specific 的 goal / tool / fixture / distractor 定义；
4. 全 handle 覆盖检查与重叠检查；
5. 统计输出：每类像素占比、unknown handles、空 target/goal、每场景逻辑物体数。

不能只靠 object-name substring 自动推断全部角色：现有 segmentation 审计已证明同名前缀可能表示
不同实例，看似属于某物体的 handle 也可能实际渲染另一个部件。§5.1 里 handle `44` 的实测是同一
类问题的另一面。

### 7.3 数据重生成：降级为可选

初稿要求新数据扫描所有帧收集 handle 名，以防漏记。**实测**：随机 60 个 episode × 2 视角的全部帧，
`handles.json` 缺失只出现 **1 次**（`put_groceries_in_cupboard` 的 handle `44`），覆盖率 > 99%。

为不到 1% 的缺口重跑 2818 个 episode 的数据生成不划算。**修正**：

- 保留离线校验：生成器扫描 `mask.npz` 全部 unique ID，验证都能解析；无法解析时**报告 episode
  路径和 handle ID 并硬失败**，绝不静默映射成 background。这一条必须有。
- `gen_dataset.py` 改为扫描全部帧收集 handles —— 只对**将来新生成的数据**生效，不回填。
- 对现有数据的个别缺口，写一张小的补丁表；若某 episode 无法解析则从训练集中排除并计数。

---

## 8. Dataset / codec / loss 改动

### 8.1 向后兼容开关

```bash
--segmentation_mode referring     # 当前 target-only 协议，默认
--segmentation_mode scene_roles   # 新的 dense role 协议
```

旧命令、旧 checkpoint、已有评估继续走 `referring`；新实验显式选择。

**Phase B 开始之前必须先固化 golden tensor**：用固定 seed 跑一批现有 dataset 输出，存下张量
hash 到 `tests/golden/seg_referring_*.json`。否则改完就没有基线可以证明"逐位一致"（§12.3）。

### 8.2 Dataset

[rlbench_selfgen.py](training/dataset/rlbench_selfgen.py) 新增：

```python
_load_scene_segments(episode_path)                    # 读 scene_segments.json，lru_cache
_build_scene_role_lut(scene_segments, seg_targets, instruction)   # -> uint8[65536] LUT
_encode_scene_roles(handle_map, role_lut)
```

编码走纯 LUT：

```python
label_map = role_lut[handle_map]          # uint16 索引 -> uint8 label
rgb = SCENE_ROLE_PALETTE[label_map]
```

保持现有不变量：

- 两个采样视角都编码；
- 与 RGB 使用完全相同的 `frame_indices` / `view_indices` / `view_dirs`；
- resize handle map 只用 nearest（`_to_model_res`，先 resize 再 encode，绝不 resize 颜色）；
- 同一 clip 内颜色不变；
- 输出 `[C, 2T, H, W]`，范围 `[-1,1]`。

### 8.3 Codec

`training/percep/seg_codec.py` 新增独立接口，`encode_known_color` / `decode_known_color` 原样保留
给 referring：

```python
encode_scene_roles(label_map)                    # uint8 label -> uint8 RGB
decode_scene_roles(rgb, present_roles)           # -> {role: bool mask}
```

两点必须做到：

1. **不做连通域剪枝、腐蚀、小区域删除。** 理由已记在 `seg_codec.py` 的 docstring 里：
   Appendix A 的 `THETA_SIZE=2e-4`（256² 下约 13px）会把 `sweep_to_dustpan` 的 dirt（5 个 4–8px
   的 handle）全部删光，在无噪声 roundtrip 上实测 IoU = 0.0。
2. **`decode_scene_roles` 必须限制在本 episode 实际出现的角色子集上，不能对全部 9 色做 nearest。**
   这是初稿的一处退化：现有 `decode_known_color` 只在本场景用到的颜色里选，注释写明了原因
   —— 没用到的保留色会抢走歧义像素。本 episode 出现哪些角色，可以从
   `scene_segments.json` + instruction 确定性推出，训练和评估时都拿得到，没有理由丢掉这个性质。

### 8.4 Loss 与类别不平衡

完整 scene map 中 background 仍占 60–92%（§4）。只把背景显式编码而继续用等权 latent MSE，可能
得到"视觉上更稠密、但主要在学静态背景"的假改进。

空间权重初值（P0.3 的归因统计出来后再定稿）：

| 区域 | 权重 |
|---|---:|
| background | 0.25 |
| robot_arm / gripper / fixture / tool / distractor | 1.0 |
| target / goal | 4.0 |

落地细节 —— 初稿完全没提，而这里才是 Phase C 真正的工作量：

1. **链路**：权重图要从 dataset → `ActionImagesDataCollator`（显式 key 列表，必须加 key）→
   `forward` → 按 `assemble` 的 segment offset 对齐地插进
   [train.py:384-390](train.py#L384-L390) 的 `valid` 里。现在的 loss 是在**整条拼接后的 latent
   序列**上算的，权重图必须知道 segmentation 段落在哪些 offset 上。
2. **下采样用 max-pool，不用 bilinear/avg。** VAE 是 8× 空间、4× 时间压缩：256→32 的 latent 网格
   上，1% 的 target 只剩约 10 个 latent 位置。规则写死为"该 latent 位置覆盖的任一像素属于该类 →
   取该类权重（取最大）"，否则实现者会随手用 avg 把小目标抹平。
3. **归一化只在 segmentation segment 内部做。** "有效区域平均权重归一化为 1"若在整条序列上做，
   会连带缩放 video / depth / action 段的 loss，那 §10.4 里"相同 action/depth 配方"的前提就不成立，
   三个 arm 不再可比。
4. 只作用于 segmentation segment，不改 RGB / depth / action。
5. target/goal 权重是配置项，不是硬编码常量。
6. **诚实记录**：这是在 diffusion latent MSE 上加空间权重，不是分割的交叉熵。latent 是纠缠表示，
   "class weight map" 只是空间上的近似。这不影响做，但影响结果如何解释，报告里要写明。

---

## 9. Prompt 协议

```text
<scene_seg protocol=rlbench-scene-role-v1> close the red jar
```

（协议名与 `scene_segments.json` 的 `protocol` 字段统一为 `rlbench-scene-role-v1`；初稿两处不一致。）

不把物体和颜色逐一枚举进 prompt，因为 palette 已全局固定。这避免了：prompt 随场景物体数暴涨；
simulator 内部名泄漏进自然语言；同一物体因字典序变化而改变颜色；模型同时学 segmentation 和
任意颜色重映射。

旧协议不变：

```text
<seg: item=red, drawer=green> put the item in the bottom drawer
```

两者必须用不同 tag，否则模型无法判断该输出 target-only mask 还是 dense role map。

---

## 10. 实验设计

### 10.1 风险：dense 标签可能**稀释** target

必须正面写下这个风险，它决定了 arm 的编排。scene_roles 不改变 target 的绝对像素量（还是 ~1%），
改变的是前景竞争者从 0 涨到约 15%。在等权 MSE 下，target 占非平凡内容的份额是**下降**的。

因此 **S1（dense + 等权）恰恰是最可能让 instruction grounding 变差的配置**。初稿的编排是
"先跑 S1，若发现 background dominance 再做 S2"—— 但 S1 变差既可能是"dense 无效"也可能是
"dense 有效但被稀释"，这个 arm 单独无法区分，第一轮会白跑。

### 10.2 arm 编排（同批起，不串行）

| arm | Segmentation 表示 | Loss | 目的 |
|---|---|---|---|
| S0 | 当前 referred target-only | 等权 | 基线（P0.1 已产出） |
| **S1b** | target + goal + robot_arm + gripper，其余仍为黑 | 等权 | **便宜的中间点** |
| S1 | Scene Role v1（全 9 类） | 等权 | dense 标签本身 |
| S2 | Scene Role v1（全 9 类） | class-balanced | dense + 重加权 |

**S1b 是本版新增，它决定 Phase A 的规模。** 它拿到大部分稠密收益 —— robot_arm + gripper 稳定
占 9%（§4 全 16 任务方差极小），是跨任务语义最确定的类 —— 而标注成本接近 0：`handles.json` 里
`Panda_*` 前缀加 `seg_targets.json` 现有的 target 就够，**不需要 16 个 task resolver**。
稀释程度也远小于 S1。

若 S1b 已复现 S1 的大部分收益，§7.2 里 16 个 resolver 的多数工作可以直接砍掉。所以
**S1b 先跑，与 Phase A 的标注工作并行**。

S1 与 S2 同批起，不要用 S1 的结果当 S2 的门槛。

### 10.3 正交维度：perception 的 mask 比例（M 轴）

`PERCEPTION_VIDEO_GIVEN_PROB = 0.9` 是 fork 自己定的常数（§1.1），**从未被扫过**。它和标签
表示（S 轴）正交，而且改动量是一个常数 vs. 16 个 task resolver —— 性价比高得多，因此不排在
S 轴后面。

先厘清 IIII 在 `video+segmentation` 下给了什么：4 段各给首帧，即
`video_v0[0] + seg_v0[0] + video_v1[0] + seg_v1[0]`。**seg 首帧已经用颜色标出了哪个物体是
target**，所以 IIII 对 seg 不是无信号的，它训练的是另一种能力：

| 配置 | 训练的能力 |
|---|---|
| FIFI | 从**给定**的 RGB 读出 seg —— Vision Banana 式 perception |
| IIII | 自己生成视频，**同时**维持一条与之一致的 seg 流 —— co-generation |

两者不可互相替代。本仓库的目标是 co-generation，而 action 有 90% 的时间在 IIII régime；seg 若
只在 RGB 给全时被训练，两条流永远学不到在同一个生成过程中互相约束。

#### 当前设计的一个隐藏混淆（M 轴的最强动机）

现在一个 batch 里混着 `video+action`（90% IIII）和 `video+segmentation`（90% FIFI）：
**模板身份几乎完美预测了 conditioning 等级** —— 看到 `<seg:...>` tag 就意味着 RGB 会给全。
模型可以学到这个捷径，而它在任何别的配置下都不成立。把 perception 也改成三档 mask 分布，
就把"哪个任务"和"给多少条件"两个变量解耦了。这比"要不要练 co-generation"更硬，是 M 轴
本身值得做的理由。

另外 FIII 目前**训练里从不出现、评估里却在测**（0.458，§1.1）。三档结构让评估表的三列都有
对应的训练分布，同时给出一个"给多少用多少"的条件梯度课程。

#### 扫描点

三档比例记作 (IIII, FIII, FIFI)：

| arm | (IIII, FIII, FIFI) | 说明 |
|---|---|---|
| M0 | (10, 0, 90) | 历史值，无 FIII。`templates.py` 的库默认，供复现旧 arm |
| M1 | (45, 10, 45) | 两种能力均衡 |
| **M2** | **(90, 5, 5)** | **✅ 项目已选定：新 arm 的默认**（`scripts/train_arm.sh`） |

#### ✅ 决定（2026-08-14）：新 arm 一律用 M2

理由就是 §0.2：**闭环 rollout 跑 `video+action` + i2va（= IIII）**，辅助流若 90% 训练在 FIFI，
学的是"给定 RGB 判别式读出 depth/seg"，而部署从不使用那个 régime。M2 让辅助任务与主任务、
与部署跑在同一个生成式 régime 下，共享表征的迁移假设这时才成立。附带收益是模板身份不再预测
conditioning 等级（见下方"隐藏混淆"）。

落地方式（**故意分两层**）：

- `templates.py` 的 `PERCEPTION_MASK_MIX_DEFAULT` **保持 M0**。它是
  `tests/test_forward_unchanged.py` 钉住的那条历史行为，必须继续意味着"旧行为仍可复现"；
- `scripts/train_arm.sh` 的 `PERCEPTION_MASK_MIX` **默认 M2**，实验选择落在命令行和 wandb
  config 里，可追溯。与 `FRAME_INTERVAL`（脚本 3 / `args.py` 1）是同一套责任划分。
- 复现 2026-08-14 之前的 arm：`PERCEPTION_MASK_MIX=M0 bash scripts/train_arm.sh ...`

M1 仍值得作为中间点扫，但不再是"选默认值"的前提——默认已定。

#### single-frame 分支：现状、坑、以及一个值得加的新模式

**现状（已核对）**：[templates.py:247](training/templates.py#L247) 是
`single_frame=single_frame_visual and m in VISUAL_MODALITIES`，而
`VISUAL_MODALITIES = ("video", "depth", "segmentation")`。
[MODE_MIX_MASKS.md:92](MODE_MIX_MASKS.md#L92) 同样写的是"**视觉段**只保留第一 latent 帧，
action 段保留完整长度"。

所以在 `video+depth+action`（arm2）里，policy mode 下 **depth 也被截成 1 帧**，不保持完整 latent。
这是 fork 的一次推广：官方只有 video 一个视觉模态，"collapse video"与"collapse all visuals"
无差别；fork 引入 depth/seg 后实现成了后者。可能是有意的（policy 的语义是"只给一眼观测预测
动作"，给全 depth 视频就不算 policy），但**需要确认 arm2 的意图**。

**坑（仅针对原样路由）**：若把 perception 模板直接扔进 `has_action` 分支，`video+segmentation`
的四个段**全是 visual**，全截成 1 帧后每段唯一那帧又都是 condition，
[templates.py:301](training/templates.py#L301) 的 `assert not bool(masks.all())` 触发。
对 `video+action` 不会，因为 action 段保持全长。

**更好的做法 —— 新增 "single-frame perception" 模式**：让 `single_frame` 只作用于 anchor
（video），depth/seg 保持完整长度：

```text
IIII            : V0(full,给首帧) | S0(full,给首帧) | V1(full,给首帧) | S1(full,给首帧)
single-frame perc: V0[:1]         | S0(full,给首帧) | V1[:1]         | S1(full,给首帧)
```

它不崩（seg 段长度 > 1，存在预测位置），且**与 IIII 语义不同**：IIII 下模型要同时生成 RGB 视频
和 seg 视频；此模式下 RGB 段被截掉，模型**只**生成 seg 视频。这是一个干净的消融 ——
*模型是否需要先渲染出 RGB 才能产出 seg？* 建议作为 M 轴的第四档可选项纳入。

#### 代价：FIFI 曝光量的硬约束

FIFI 从 90% 降到 5% 是 18× 削减。seg 本身只占 arm1 的 20%：

| | FIFI-seg 占总步数 | 2500 步内 |
|---|---:|---:|
| M0（0.2 × 0.90） | 18% | ~450 步 |
| M2（0.2 × 0.05） | 1% | ~25 步 |

即便把 seg 份额提到 0.4 也只有 2%。**要维持今天的绝对曝光量需要 seg 份额 3.6，不可能。**
所以 M2 下 FIFI 必然大幅退化 —— 这不是风险而是确定结果，写进预期，不要当作失败信号。
真正的问题是：是否愿意用 FIFI 换 IIII 下的 co-generation 能力。

#### FIFI 退化 vs IIII 进化：这两笔账不对称

"FIFI 退了、IIII 不就进了吗"——会进，但**两笔账的可测性完全不同**，不能相互抵消着看：

| | 任务性质 | IoU 上限 | 可测性 |
|---|---|---|---|
| FIFI | **确定性**：给定 RGB，seg 是它的函数 | 1.0 | 干净 |
| IIII | **随机**：给定首帧+文本，存在多个合法未来 | 远低于 1，由数据的固有不确定性决定 | 带 §1.2 时间错位混淆 |

所以 M2 下你会看到"一个确定性指标明确下降 + 一个测不准指标小幅波动"。**不要把这读成净亏损，
也不要读成净收益。** 要看到 IIII 侧的真实进步，需要三个工具：

1. **aligned IoU**（§3.4）——去掉时间错位。
2. **RGB–seg 循环一致性**（新增）——把 IIII 生成的 RGB 再用 FIFI 模式喂回模型，比较得到的 seg
   与它在 IIII 时自己生成的 seg。**不需要 GT**，直接测 co-generation 的自洽程度，正是 IIII
   régime 该学会而 FIFI régime 教不了的东西。
3. **闭环成功率 / IIII 下的 action 指标**——§0 已确立的交付物。这是 M 轴的**最终判据**。

#### 其他配套

- 常被引用的不对称 ——"训练条件少、推理条件多"比反过来安全 —— 在这个 mask-conditioning
  设置里**并未被验证**（把 clean latent 送到模型预期为 noisy 的位置本身也是分布偏移）。
  所以扫，不要辩。
- **M 轴的判定用闭环成功率，不用 seg IoU**（§0.2）。seg IoU 只作为"辅助任务是否在学"的诊断。
  这一点与 S 轴不同：S 轴改的是标签内容，seg IoU 是其直接产物；M 轴改的是 régime，其收益按
  定义只在交付物上体现。
- M 轴与 S 轴不交叉全扫：先在 S0（现有 referring 标签）上扫 M0/M1/M2，选出最优 M 再跑
  §10.2 的 S 轴。M 只改 mask 分布，立刻能跑；S 轴要等 Phase A 的标注。
- 新增 `--perception_mask_mix`（三档比例）参数，替代硬编码的 `PERCEPTION_VIDEO_GIVEN_PROB`。

### 10.4 受控变量

所有 arm 必须保持相同：初始 checkpoint、episode 快照、variation split（训练 variation0，留出
variation1/2）、seed、步数、learning rate、sampling interval、`template_mix`、action/depth 配方。

**另外必须提高 seg 的训练暴露量。** §1.1 指出 arm1 里 seg 只占 20%（2500 步中约 500 步）。在
这个暴露量下比较标签表示，很可能测的是训练不足而不是表示差异。建议所有 seg arm 用
`video+action@0.4,video+depth@0.2,video+segmentation@0.4` 或延长步数，并在 S0 上用同样设置重跑，
保证可比。

### 10.5 判定阈值与失败准则

判据分两层，**上层优先**。这是 §0 的直接后果：seg IoU 涨了但闭环没涨，等于没做。

#### 上层：交付判据（决定去留）

- **主指标**：闭环成功率，协议见 `EVAL_PLAN_CLOSEDLOOP.md` §2.5（`E=41`，每格 20 次 rollout）。
- **硬约束（一票否决）**：任何 arm 的闭环成功率相对 arm0 退化，即视为失败，无论 seg 指标多好。
  §1.4 目标 6 是硬约束——辅助流不得以损害 `video+action` 为代价。
- 闭环无信号（地板效应）时降级为开环 action 指标：strong-frame median `pos_err` 相对 arm1
  退化 > 20% 即判失败，并在报告里写明结论强度已降级。

#### 下层：表征判据（决定归因）

- **主指标**：held-out target IoU 的 mean ± 95% CI（FIFI 协议，§11）。
- **判定 S1 > S0**：CI 不重叠，且 mean 提升 ≥ 0.05 绝对值。

#### 失败准则（预先声明，不许事后调整）

- 若 §3.0 的前置门判定"arm1 < arm0"→ **停止本方案**（§0.3）。
- 若 S1 与 S1b 均未超过 S0 的下层阈值，而 S2 超过 → 结论是"重加权有效，dense 本身无效"，
  保留 scene_roles 仅作为承载权重图的手段。
- 若 S1 / S1b / S2 都未超过 S0 → 一级假设被否定。回滚到 `referring`，转向 §1.1 暴露的另外两个
  候选原因（训练暴露量、训练稳定性/checkpoint 选择）。**不要"再调调看"。**
- **若某 arm 下层指标显著改善、上层指标不变** → 二级假设被否定：辅助表征质量与交付物无关。
  这是本方案最可能的失败模式（§0.1 的因果链长而弱），必须能被识别出来而不是被解释掉。
  此时结论是"scene_roles 是一个更好的 segmentation 协议，但不是提升闭环成功率的杠杆"，
  据此决定是否还要继续投入。

---

## 11. 评估指标

指标分三层。**报告必须按这个顺序呈现**，不能把诊断层的改善写在最前面（§0）。

| 层 | 指标 | 作用 |
|---|---|---|
| **L1 交付** | 闭环成功率（`EVAL_PLAN_CLOSEDLOOP.md` §2.5 协议） | **唯一的去留判据** |
| **L2 归因** | 开环 action：strong-frame median `pos_err`、`r_peak_mean`、`gripper_acc` | 闭环无信号时的代理；辅助流是否伤害主任务 |
| **L3 诊断** | 下面全部 segmentation 指标 | 辅助任务本身学得如何；**不能单独支撑结论** |

### 11.1 L3：segmentation 诊断指标

不用总体 pixel accuracy —— 全预测 background 也能拿到 85%。

**评估协议（初稿缺，是 S0/S1 可比性的前提）**：≥20 个 **held-out** episode（variation1/2，覆盖全部
16 个任务）× 3 seed × 2 视角。每个指标报 mean ± 95% CI。

**M2 选定后，FIFI 的角色变了，必须同时报两条 mask（这是 §10.3 决定的直接后果）：**

| mask | 在 M2 下的地位 | 怎么用 |
|---|---|---|
| **IIII** | **in-distribution（90%）**，与部署 régime 一致 | **主报**，但**必须**配 §3.4 的 aligned IoU + best-offset，否则被时间错位混淆吞掉 |
| FIFI | 仅 5% 训练，但仍是**唯一无时间混淆的确定性测量** | 作为**上界探针**报，标注"训练暴露量仅 5%，数值偏低属预期，不代表能力退化" |

不要再把 FIFI 当作唯一主指标——那是 M0 时代的写法。也不要因为 FIFI 数字下降就判定 M2 失败：
§10.3 已经算清 FIFI-seg 曝光量从 ~18% 降到 ~2%，**下降是确定结果而非失败信号**，
M 轴的判据在 L1（§11 三层表）。

1. `target IoU`（L3 主）
2. `goal IoU`
3. foreground macro mIoU
4. robot_arm / gripper IoU
5. tool / fixture / distractor 各类 IoU
6. boundary F-score
7. 小目标 recall，按 GT 面积分桶（分桶边界按 §4 实测分布定：<0.5%、0.5–2%、2–10%、>10%）
8. 跨帧颜色/身份一致性
9. 每视角单独统计后取 macro average
10. VAE encode/decode 后的 per-class roundtrip IoU（下界，来自 P0.2）

记录但不作主指标：background IoU、overall pixel accuracy、每类预测像素比例、预测为 unknown 的
像素比例。

IIII / FIII 两种 mask 继续记录，但**必须标注为 OOD 探针**（§1.1），不能与 FIFI 并列解读。
报告 IIII 时必须同时给出 §3.4 的 **aligned IoU 与 best-offset 分布**：不带对齐的 IIII IoU
混淆了"分割差"和"时刻不对"（§1.2），单独列出会被误读成模型能力结论。

### 11.2 M 轴专用：RGB–seg 循环一致性

M1/M2 把训练重心移到 IIII，而 IIII 的 IoU 既带时间混淆又有随机性天花板（§10.3）。补一个
不需要 GT 的自洽指标：

1. 用 IIII 生成 RGB 视频与 seg 视频；
2. 把生成的 RGB 当作条件，用 FIFI 模式再跑一次，得到 seg′；
3. 报 seg 与 seg′ 的 per-role IoU。

它测的是"模型生成的两条流是否互相一致"，正是 IIII régime 该学会、FIFI régime 教不了的能力。
与 L1 一起看：若循环一致性上升而闭环成功率不动，说明 co-generation 学到了但没有转化为交付物。

---

## 12. 验收门

### 12.1 数据层

- 每个 `mask.npz` 确认为 `[T,H,W] uint16`
- 所有实际出现的 handle ID 恰好映射到一个逻辑物体
- 所有逻辑物体恰好映射到一个 role
- 训练集 unknown handle 数为 0（否则硬失败，不得静默转 background）
- 角色颜色在两视角与整个时间窗口内一致
- target/goal 与 instruction/variation 对齐，且与 `build_referring_spec` 的输出交叉一致
- 新旧数据都能生成 `scene_segments.json`；无法生成的 episode 被显式排除并计数

### 12.2 Codec 层

- 无 VAE 的 encode/decode roundtrip 每类 IoU = 1.0
- 相邻类别不互相覆盖
- 1–2 像素小目标不被后处理删除
- resize 前后只产生合法 label，不产生插值颜色
- unknown label 能被显式检测
- `decode_scene_roles` 在限定角色子集下的行为经测试覆盖（§8.3.2）

### 12.3 Pipeline 层

- RGB 与 scene-role stream 使用相同帧和视角
- **`referring` 模式输出与 golden tensor 逐位一致**（§8.1）
- `scene_roles` 模式 prompt、template、stream 一致
- 旧 checkpoint 可在 `referring` 模式继续推理
- eval 按 protocol 选择正确 decoder，不混用

---

## 13. 文件改动

| 文件 | 改动 |
|---|---|
| `eval/`（§3.0 前置门） | **复用** `EVAL_PLAN_CLOSEDLOOP.md` 已建好的 `eval/policy.py`，不新写闭环 harness |
| `scripts/seg_pixel_stats.py` | **新增**：全树像素占比统计，产出 §4 的表 |
| `scripts/vae_roundtrip_seg.py` | **新增**：P0.2 的 per-class VAE roundtrip 诊断 |
| `scripts/seg_aligned_iou.py` | **新增**：P0.4 的时间对齐 IoU / best-offset，复用已有 IIII 输出 |
| `/workspace/ttdu/ttd/src/percep/scene_segments_gen.py` | 新增离线 logical-instance / role 元数据生成器 |
| `/workspace/ttdu/ttd/scripts/gen_dataset.py` | 新数据扫描全部帧收集 handles（不回填旧数据） |
| `training/percep/seg_codec.py` | 新增 role palette 与 `encode/decode_scene_roles`；`tool`/`unknown` 换色 |
| `training/dataset/rlbench_selfgen.py` | 元数据加载、LUT 构造、`scene_roles` 编码分支 |
| `training/templates.py` | 新增 `<scene_seg protocol=...>` tag |
| `training/args.py` | `--segmentation_mode`、`--perception_mask_mix`、loss 权重参数 |
| `training/templates.py`（M 轴） | perception 分支改为三档 mask 分布，**跳过 single-frame collapse** |
| `train.py` | Phase C：segmentation-only 空间权重（含 collator key 与 offset 对齐） |
| `eval/eval_perception.py` | per-role IoU、macro mIoU、boundary、面积分桶 recall、CI |
| `tests/golden/seg_referring_*.json` | **新增**：Phase B 前固化的 referring 基线 |
| `tests/test_scene_seg_codec.py` | palette、roundtrip、小物体、unknown、子集 decode |
| `tests/test_scene_seg_dataset.py` | 真实 episode 的对齐、覆盖率、角色一致性 |

---

## 14. 实施顺序

**Phase A 的启动条件已解除**（§3.0 放行）。剩余顺序仍按"成本从低到高"排。

| 阶段 | 内容 | 出口条件 |
|---|---|---|
| ~~**P0-gate**~~ | ~~§3.0 闭环前置门~~ **✅ 已放行**（项目决定 + arm4 闭环证据方向为正，边界见 §3.0） | 不再阻塞 |
| **P0-diag**（与 gate 并行，~2 天） | ~~3.2 VAE roundtrip~~ **✅ 已完成，通过（0.9910）**；3.1 扩大 S0 评估面；3.3 loss 归因；3.4 时间对齐 IoU | 3.2 已放行。剩余三项不阻塞 |
| ~~**M**（扫描）~~ | **✅ 已定：M2 (90/5/5)**，`scripts/train_arm.sh` 默认。M1 仍可作中间点补扫 | 不阻塞 A |
| **A**（标注） | `scene_segments.json` schema（仅 instances + base_role）；16 任务 resolver；全树生成；覆盖率与像素占比报告；人工 contact sheet 逐任务确认 | unknown handle = 0 |
| **B0**（先于 B） | 固化 referring golden tensor | 基线可复现 |
| **B1**（与 A 并行） | S1b：robot/gripper/goal 最小扩展，只用 `Panda_*` 规则 + 现有 seg_targets | 跑出 S1b，决定 A 的规模是否可缩减 |
| **B2** | 固定 palette codec；`--segmentation_mode scene_roles`；等权 loss；codec/dataset/对齐测试 | §12.1–12.3 全绿 |
| **C** | 空间权重图（含 collator/offset 链路）；S0 / S1b / S1 / S2 **同批**跑；按 §11 三层协议评估 | 按 §10.5 双层阈值判定 |
| **D**（可选） | 若确需区分同角色的物理实例，单独增加 `<instance_seg>` 协议与 handle-free matching 评估 | —— |

顺序的核心逻辑仍是**成本从低到高**。两道前置门都已放行（§3.0 项目决定，§3.2 实测 0.9910），
所以现在的瓶颈是 Phase A 的人工确认与 S 轴训练本身。

Phase D 的动态实例颜色**不得混入** `scene_roles`，否则会失去本方案最重要的跨任务颜色一致性。

---

## 15. 决策摘要

0. **交付物是闭环成功率，segmentation 是辅助流，rollout 序列里没有它**（§0）。所有下层指标
   的改善只有转化为 L1 才算数。**§3.0 的闭环前置门排在一切之前**：arm1 若不优于 arm0，
   本方案没有收益路径。
1. 使用 `mask.npz["mask"]` 的完整 handle map。
2. 不用 raw handle ID 或 episode 内排序决定颜色；先合并逻辑物体，再映射固定功能角色。
3. target 永远红、goal 永远绿，机器人/工具/装置各自固定色；静态环境统一黑色。
4. **`tool` 与 `unknown` 相对初稿换色**，因为最小色距 98.3 不能给两个共现的前景类。
5. **decode 限制在本 episode 实际出现的角色子集上**，不对全 palette 做 nearest。
6. **`scene_segments.json` 只存 instruction 无关信息**；target/goal 继续由
   `build_referring_spec` 运行时解析，保持单一真相源。
7. 全 handle 必须被处理，未覆盖项用橙色暴露并在训练前硬失败。
8. 保留 `referring` 协议，Phase B 前先固化 golden tensor 以证明逐位一致。
9. **不重生成 2818 个 episode**：实测 `handles.json` 覆盖率 > 99%，硬失败校验足够。
10. **S0 / S1b / S1 / S2 同批跑**，不把 S2 挂在 S1 的结果后面；S1b 先行，用来决定标注规模。
11. 评估用 ≥20 held-out episode × 3 seed，只解读 FIFI，主指标 target IoU 报 CI；
    **预先声明判定阈值与失败回滚条件**。
12. P0.2 的 VAE roundtrip 是唯一的硬性否决点，优先做。
13. **IIII 的 IoU 混淆了"分割差"和"时刻不对"**（窗口覆盖任务的 36%–100%，模型从首帧无法
    判断节奏）。报告 IIII 必须附 aligned IoU 与 best-offset；FIFI 之所以干净，是因为整段
    RGB 给全后时间演化被钉死。
14. **M 轴（perception mask 比例）先于 S 轴**：改一个参数 vs 16 个 task resolver，而且
    rollout 跑在 IIII régime，辅助任务与部署 régime 对齐才有迁移可言（§0.2）。
15. 最可能的失败模式是"L3 涨、L1 不动"——二级假设被否定。必须能识别它，而不是解释掉它。
