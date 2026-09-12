# CROSSMODAL_FUSION_PLAN —— 联合解码与跨模态监督

> 日期：2026-09-04。前置：`EVAL_PLAN_ARM7.md`（arm7 的三层评测）。
> 本文回答两个问题：
> **(1)** arm7 的四路读数能不能联合解码成一个更好的位姿？
> **(2)** 不同模态之间有没有可以互相提供监督信号的方式，值不值得为它新训一个臂？
>
> 结论先行：**(1) 只有在四路误差不独立时才有意义，必须先测；(2) 值得，但正确的框架不是
> 「集成」而是「特权信息蒸馏」**——理由见 §1，那一节改变了本文其余部分的全部设计。

---

## 1. 先做信息审计：四个模态到底谁带了什么

这一步不做，后面所有「联合」都是在猜。逐个查代码：

| 模态 | 像素从哪来 | 独有信息 | 相对 RGB 的地位 |
|---|---|---|---|
| `video` | RLBench RGB 渲染 | 纹理 / 颜色 / 光照 | **部署时唯一可得** |
| `depth` | `depth.npz` → `encode_depth` | 每像素**度量距离** | **特权**：单目 RGB 尺度歧义 |
| `segmentation` | `mask.npz`（仿真器 object handle 图）→ `resolve_roles` 按 `simGetObjectName` 的**物体名**赋角色 | 目标/目标位/机械臂/夹具的**真值语义** | **特权**：RGB 推不出 |
| `normal` | `encode_normal_from_depth(depth, fx, fy)` | **无** | **与 depth 冗余** |

两条结论，都能从代码直接读出来，不是推测：

**(a) `normal` 是 `depth` 的确定性函数。** `training/percep/normal_codec.py:56-61` 里
`encode_normal_from_depth` 只吃 `(depth, fx, fy)`，逐帧调 `depth_to_normal`。
所以 normal 相对 depth **零新增信息**，只是把高频几何（边缘、朝向）放大到 depth 绝对值压缩掉的
尺度上。**depth↔normal 之间的一致性是解析约束，不是可学的信息**。

**(b) `segmentation` 是特权信息。** 它来自仿真器的 handle 图加上
`simGetObjectName` 的物体名解析（`training/percep/scene_segments_gen.py:204 resolve_roles`），
是**真值**的物体身份与角色划分。RGB 不可能无损推出它。

### 这改变了 arm7 是什么

之前的说法是「四个可互换的适配器 = 冗余，所以可以集成」。**信息审计说不是。**
四路构成一个**信息层级**：

```
特权（部署拿不到）      depth  ── 度量几何
                       seg    ── 真值语义
                        │
                        │  normal = f(depth)，冗余
                        ▼
部署唯一可得            video  ── 纹理，几何与语义都得自己推
```

**所以 arm7 的三个非 RGB 适配器是「特权教师」，RGB 适配器是「学生」。**

这条解掉了 `EVAL_PLAN_ARM7.md` §0 里那个矛盾——「arm7 把 60% 训练花在闭环用不上的模态上」。
在特权框架下那 60% 不是浪费，**是教师**。

但要注意：**arm7 目前没有任何迁移机制**。四个适配器共享权重，却从未被要求彼此一致；
它们只是轮流用同一套权重做四件事。**把教师的信息真正推给学生，正是新臂要加的东西**（§4）。

---

## 2. 让联合解码/联合训练在数学上成立的那个精确约束

`forward` 里 action 段的像素**不是从磁盘读的，是渲染的**：同一份 `action_7d` 经**同一对相机**
投影（`project_actions_7d_to_5d_torch_batch` → `project_action_5d_to_rgb_torch`，
`train.py:369-377`）。四个模板用的是**同一份** `extrinsics_c2w` / `intrinsics_c2w`。

推论，两条都是精确的、不是近似：

1. `video+action` 里的 `A0` 与 `depth+action` 里的 `A0`，**监督目标逐像素相同**。
   所以两者预测之间的一致性损失是**良定义**的，不需要任何对齐启发式。
2. 同一帧下四路的 action 热图落在**同一个图像平面**，因此「四路 2D 质心的偏离」是一个
   直接可测的像素量，**不需要任何三角化**。

第 2 条正是你提的那个想法可以立刻实现的原因。

---

## 3. 推理期：联合解码（不重训，先做）

### 3.1 前置——先把误差拆成「方位」和「距离」

在谈联合之前必须先知道误差长什么样。解码器是「沿 view 0 的射线一维搜索」
（见 `EVAL_PLAN_ARM7.md` §3 Tier 2），所以位置误差天然可以分解成两个正交分量：

- **bearing（方位）误差**：2D 质心在图像平面上偏了多少像素 → 射线方向错了多少。
- **range（距离）误差**：沿射线选的深度错了多少米。

量化方式：把预测点投影回 view 0，与 GT 投影点的像素距离 = bearing；GT 点到 view 0 射线的
投影距离 = range。**这是整份计划里最便宜、信息量最大的一个测量**：

- 若误差主要在 **range** → 联合解码有很大空间（多一路观测正是用来定深度的）。
- 若误差主要在 **bearing** → 四路都在同一个图像平面上看错了地方，联合解码救不了，
  应该直接转 §4 的训练期方案。

### 3.2 跨模态一致性（你提的方案，一般化）

**B1. 同视角 2D 质心偏离**（最便宜，零几何）
对每个 (帧, 视角)，取四路 action 热图的加权质心，算两两像素距离。
产出：每帧一个「分歧度」标量。用途：离群模态检测、失败预警。

**B2. 4×4 跨模态三角化矩阵**（你提的「RGB 出主射线、depth 定深度」的一般化）

| | 列 = 谁给 view 1（打分/定深度） |
|---|---|
| **行 = 谁给 view 0（主射线/方位）** | video / depth / seg / normal |

- **对角线** = 模态内解码（现状）。
- **非对角** = 跨模态：例如「RGB 的 view 0 出主射线 × depth 的 view 1 打分」——正是你说的那个。

这张表回答一个很干净的问题：**四个适配器是否共享同一个几何框架？**
若非对角 ≈ 对角，说明四路的 action 表示在同一个坐标系里、可以互换；
若非对角显著更差，说明每个适配器各自漂了，那么「联合」从一开始就不成立。

**注意实现细节**：主射线恒取索引 0（`utils.py:896-897`），所以行/列的角色由**拼接顺序**决定，
不是由参数决定。必须显式构造，不能靠传参。

### 3.3 稳健联合解码（只在 B1/B2 说明有戏时做）

按 `EVAL_PLAN_ARM7.md` §3 Tier 2 的更正：**四路各自 V=2 解码，位姿层面取中位数**
（合成实验：一路坏时中位数 4.3 mm，直接堆 V=8 是 37.8 mm）。
再加一档：用 B1 的分歧度**门控**——先剔除离群模态再合并。

### 3.4 已经排除的一条路

**「读生成的 depth 图拿距离」不做。** 理由不是精度不够（那个判断我之前用 arm6 的 14.4% 下的，
**是错的**：同一模型同一模板换 mask mode，AbsRel 在 0.1441↔0.0366 之间摆 4 倍，
三重不可比），而是：距离读数要的是**夹爪那个像素**的深度，那里是细长、运动、自遮挡的结构，
是深度图上最难的位置，而 AbsRel 是全图平均，根本代表不了它。

**但保留一个变体作为诊断**：`depth+action` 在 **FIFI** 下是「完整 depth 给定 → 预测 action」，
此时深度图是 **GT**，没有生成误差。用它做「action 给方位 + GT 深度给距离」的单视角定位，
可以得到 range 误差的**下界**，从而判断 §3.1 的分解里 range 那部分还有多少可压缩空间。
A1 里 FIFI 占 5%，是分布内的。

---

## 4. 训练期：跨模态监督，新臂 arm8

### 4.1 现状的硬约束

arm7 **每个样本只抽一个模板**（`rlbench_selfgen.py:605`），而 `train.py:_batch_template`
**拒绝**一个 batch 里出现多个模板。所以**两个模态在一个训练步里从不共存**，
任何跨模态损失都必须先解决这一点。三条路：

| 方案 | 一个 step 里怎么让两个模态共存 | 部署 | 代价 |
|---|---|---|---|
| **A. 孪生蒸馏（推荐）** | 同一样本**跑两次前向**：一次特权模板、一次 RGB 模板 | **不变**，仍只要 RGB | <2× step（action 段与相机段的 VAE 编码可共享） |
| B. 共生成模板 | 用 `video+depth+action`（6 段），RGB 与 depth 在**同一条序列**里 | **要 depth** → 必须改仿真器 | 序列 1.5×、attention 2.25× |
| C. 解析约束 | `depth+normal+action`，强制 normal = f(depth) | 不变 | 低 |

**C 基本没价值**：§1 已证明 normal 相对 depth 零新增信息，这个约束模型可以平凡满足，
且与 action 无关。

**B 是「真融合」**——模型能同时看到 RGB 和 depth 来决定动作，这是「利用不同的信息」最字面的
实现。但它把部署条件改了（要 depth），与「不改仿真器」的既定决策冲突，且贵 2.25×。
**留作后续，不是首选。**

### 4.2 推荐方案：孪生跨模态蒸馏（arm8）

**每步做什么**

1. 抽一个样本，用 `_seed_all(seed)` 固定窗口与视角对。
2. 构两条分支，**同一窗口、同一相机、同一份 `action_7d`**：
   - **教师**：从 `{depth+action, segmentation+action}` 里抽一个（特权模态；normal 排除，
     因为它相对 depth 无新信息）。
   - **学生**：`video+action`（部署模态）。
3. 两条分支各算标准扩散损失。
4. 加一项蒸馏损失，**只作用在 action 段的预测上**，教师侧 `stop-grad`：

```
L = L_diff(student) + L_diff(teacher) + λ · || ε̂_action(student) − sg[ ε̂_action(teacher) ] ||²
```

**为什么只对 action 段**：§2 已证明两条分支的 action 段监督目标**逐像素相同**，
所以这一项是良定义的；视觉段的目标本来就不同（一个是 RGB 一个是 depth），不能拉齐。

**为什么 stop-grad 在教师侧**：部署只有 RGB，所以我们要的是**把特权信息推给学生**，
不是让学生把教师拉低。这是 LUPI / 教师-学生蒸馏的标准方向（腿足机器人里 privileged sim state
→ onboard sensing 就是这么做的）。对称一致性是另一个变体，可作消融。

**部署不变**：arm8 仍然是一个 checkpoint、RGB 单模态闭环，**不需要改仿真器**。
这正是它比方案 B 好的地方。

**代价**：两次前向，但 action 段的渲染+编码、相机 plücker 编码两条分支完全相同，可以只算一次，
所以实际 <2×。按 arm7 的 19.3 s/it 估，arm8 约 35 s/it，10000 步约 4 天；
若只在一部分步上开蒸馏（如 50%），可压到约 3 天。

**要改的代码**

| 位置 | 改什么 |
|---|---|
| `training/dataset/rlbench_selfgen.py:getitem` | 新增成对模式：一次返回教师/学生两套 `streams` 与 `template`，共用同一窗口/视角/`action_7d` |
| `train.py:_batch_template` | 放行「成对」这一种结构（仍然禁止任意混模板——那个守卫的理由没变） |
| `train.py:forward` | 跑两条分支；共享 action/相机编码；加蒸馏项 |
| `training/args.py` | `--crossmodal_distill_weight`（λ）、`--crossmodal_teacher`（哪些模板可当教师）、`--crossmodal_prob`（多少比例的步开蒸馏） |
| `scripts/train_arm.sh` | `arm8)` case |

**默认必须是关的**（λ=0 且 `crossmodal_prob=0`），这样 arm0–arm7 的复现性一个字节不变，
`tests/test_forward_unchanged.py` 的上游等价 pin 也不受影响。

### 4.3 这个臂的成败判据

- **主判据**：arm8 的 **RGB 闭环**成功率 vs arm7 vs specialist_action（46.0%）。
- **机制判据**：arm8 的 §3.1 bearing/range 分解相对 arm7 应当在 **range** 上改善更多
  ——因为 depth 教师给的正是度量几何。若改善出现在 bearing 上，说明机制解释错了。
- **诊断**：§3.2 的 4×4 跨模态矩阵，arm8 的非对角应当比 arm7 更接近对角（适配器被拉到同一框架）。

---

## 5. 顺序与门控（重要）

**§4 的新臂被 §3 的测量门控，不要并行开工。**

```
Phase A  arm7 Tier-1 逐模态误差 + bearing/range 分解        ~3 GPU-h
           │
           ├─ 若 RGB 适配器已经和特权适配器一样好
           │     → 没有东西可蒸馏，arm8 不做，转去追 §3.3 的稳健解码
           │
           └─ 若特权适配器明显更好（尤其 range 更准）
                 → 有可蒸馏的差距，继续
                       │
Phase B  一致性分析 B1 + 4×4 矩阵                            ~2 GPU-h（复用 A 的生成结果）
           │
           ├─ 非对角 ≈ 对角 → 四路同框架，联合解码可行，且蒸馏有基础
           └─ 非对角明显更差 → 各自漂了；此时蒸馏的价值更大（正是要拉齐），
                                但联合解码要放弃
                       │
Phase C  arm8 训练 + 闭环                                    ~3–4 天 + 11.3 GPU-h
```

Phase A/B 合计约 5 GPU-小时，就能决定要不要投那 3–4 天。**先做 A。**

---

## 6. 文献定位

### 6.1 最接近的先行工作，以及关键区别

**Multi-Modal Manipulation via Multi-Modal Policy Consensus**（arXiv 2509.23468, 2026）
——每个模态一个**独立 expert 网络**，router 学一致性权重自适应加权；在 RLBench 与真机上验证。

和 arm7/arm8 的区别不是细节，是**模态之间的关系类型**：

| | Policy Consensus | arm7 / arm8 |
|---|---|---|
| 网络 | 每模态一个独立 expert | **一个网络**，四个 prompt 选择的适配器 |
| 融合 | router 学权重 | 无（arm7）→ 蒸馏（arm8） |
| 模态关系 | **互补**（视觉 + 触觉，看到的是不同的东西） | **层级**：特权(depth/seg) → 部署(RGB)，外加一个冗余项(normal) |
| 「分歧」的含义 | 不一定是错误——两路看的本就不同 | **明确是错误**：同一场景同一 GT 轨迹，分歧必有一方错 |

这个区别对我们**有利**，值得写进 related work：在互补设定里，ensemble 分歧无法直接解释为
不确定性（Diff-DAgger 明确指出，多解任务里各 expert 各自笃定也会产生大分歧）；
在我们的**冗余/层级**设定里 GT 唯一，分歧就是误差，所以 §3.2 的一致性度量是可解释的。

### 6.2 方法出处

- **跨模态蒸馏**：Gupta, Hoffman, Malik，*Cross Modal Distillation for Supervision Transfer*,
  CVPR 2016——把有标注模态的表征当作无标注配对模态的监督信号。§4.2 是它在
  「特权→部署」方向上的实例。
- **特权信息学习 / 教师-学生**：腿足机器人里的标准做法——教师用部署拿不到的仿真真值状态训练，
  学生只用机载感知模仿。arm8 的 depth/seg 正是这种「仿真才有」的特权量。
- **组合式扩散策略**（product-of-experts / 分数聚合、router 组合）：`MCDP`、`FDP`、`LAG-Fusion`
  等——它们在**分布层面**组合多个策略。与 §3.3 的**位姿层面**稳健合并是两条不同路线；
  我们选后者是因为解码器把热图变成位姿的那一步是几何的、不是概率的。
- **稳健多视角三角化 / 离群剔除**：多帧多视角一致性比单帧 RANSAC 更能定 inlier——
  §3.3 的分歧门控就是这一类。

### 6.3 一句话定位

> 现有工作组合的是**互补**传感器，靠独立专家网络加 router；
> 我们研究的是**一个网络**在**同一场景的多种渲染**上共享一个动作解码器，
> 其中部分模态是仿真特权信息，因而问题从「怎么加权」变成
> **「特权模态能不能把信息推给部署模态」**。

---

## 7. 参考

- [Multi-Modal Manipulation via Multi-Modal Policy Consensus (2509.23468)](https://arxiv.org/abs/2509.23468)
- [Cross Modal Distillation for Supervision Transfer (Gupta, Hoffman, Malik, CVPR 2016)](https://arxiv.org/pdf/1507.00448)
- [Diff-DAgger: Uncertainty Estimation with Diffusion Policy (2410.14868)](https://arxiv.org/abs/2410.14868)
- [Modality-Composable Diffusion Policy (2503.12466)](https://arxiv.org/pdf/2503.12466)
- [Flexible Multitask Learning with Factorized Diffusion Policy (2512.21898)](https://arxiv.org/html/2512.21898)
- [Context-Aware Outlier Rejection for Robust Multi-View 3D Tracking (2412.16511)](https://arxiv.org/html/2412.16511v1)
- [DecAlign: Hierarchical Cross-Modal Alignment (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/file/f7f5f501282771c96bb3fedcc96bedfe-Paper-Conference.pdf)
