# CROSSMODAL_SUPERVISION_RESEARCH —— 不同信号如何互相监督

> 日期：2026-09-06。触发：arm7 的实测结果（`ARM7_RESULTS.md`）。
> 本文不是泛泛的文献综述，而是回答一个被数据逼出来的具体问题。
>
> 前置阅读：`ARM7_RESULTS.md`（实测）、`CROSSMODAL_FUSION_PLAN.md`（本文推翻了它的 §4.1 两处判断）。

---

## 0. 问题的精确形式

arm7 是**一个网络**，四条输入路径（RGB / depth / segmentation / normal），共用一个动作解码器。
四个模态是**同一场景的不同渲染**，监督目标是**同一条 GT 轨迹**。

实测（`ARM7_RESULTS.md`）给出两个约束条件，任何方案都必须与它们相容：

| 观察 | 数据 |
|---|---|
| **四条路径等价，没有「强模态」** | step 4000 汇总 24/48，p=1.000；6000/7000 比值 1.2–1.36 但均不显著 |
| **弱路径会绝对退化** | depth 0.1835→0.1986，normal 0.1773→0.1930（4000→7000），而 RGB 0.1865→0.1546 |

所以问题是：

> **当模态之间高度冗余（normal 甚至是 depth 的解析函数）时，互相监督还能提供什么？
> 以及它们为什么会互相伤害？**

---

## 1. 核心发现：两条主流范式对模态关系的要求**正好相反**

这是整份研究的支点，也是最容易搞错的地方。

### 1.1 Co-training：要求条件独立，**冗余会杀死它**

Blum & Mitchell 式的协同训练（一个视图为另一个视图打伪标签）依赖三条假设：
充分性、兼容性、以及**给定标签条件独立**。条件独立之所以是核心，是因为它保证
「一个视图犯的错与另一个视图犯的错不系统相关」——只有这样，视图 A 的高置信预测
才能给视图 B 带来 A 不知道的新信息。

文献一致指出：这个假设**实践中太强**，违反时协同训练性能下降；
当两个视图关于标签**冗余**而非互补时，理论保证基本失效。

### 1.2 对比 / 表示学习：要求冗余，**冗余正是它需要的**

Tosh, Krishnamurthy & Hsu（*Contrastive learning, multi-view redundancy, and linear models*）
的结论方向相反：**只要两个视图关于标签是冗余的**，在学到的表示上做线性函数
就在下游任务上近似最优。

也就是说，同一份「高度冗余」的性质：

```
        co-training           对比 / 表示对齐
冗余  →  理论保证失效     →    理论保证成立
```

### 1.3 arm7 落在哪一端

四条路径是同一场景的确定性渲染，关于动作**高度冗余**；`normal = encode_normal_from_depth(depth, fx, fy)`
更是**解析函数**（`training/percep/normal_codec.py:56-61`）。

**所以：伪标签式的互相监督在理论上被排除；表示层对齐在理论上被支持。**
这一条直接决定了 §4 里哪些方法值得试。

> 一个常被忽略的佐证：对比自监督（SimCLR/BYOL）的两个「视图」是同一张图的两次增强，
> **冗余度接近 100%**，却极其有效。这说明冗余本身不是问题——问题在于**用冗余去做什么**。
> 拿冗余去传播标签（co-training）是白费；拿冗余去逼出不变性（对比）是正解。

---

## 2. 我们的实测已经证伪了「特权信息」框架

`CROSSMODAL_FUSION_PLAN.md` §1 曾提出：depth/seg 是仿真器的**特权信息**（RGB 推不出），
所以应当把它们当**教师**、RGB 当**学生**，做 LUPI / 教师-学生蒸馏。

**实测把这个前提推翻了。** 蒸馏（无论是 Gupta-Hoffman-Malik 的跨模态监督迁移，
还是腿足机器人里的特权状态教师→机载感知学生）都要求**教师确实更强**。而 arm7 的四条路径
质量等价（24/48，p=1.000），特权模态的路径并不比 RGB 好。

**没有可蒸馏的差距。** 单向蒸馏方案（`CROSSMODAL_FUSION_PLAN.md` §4.2 的 arm8）
的前提不成立，应当搁置。

> 为什么「特权」没有转化成「更强」？一个合理解释：这些模态的特权性体现在**场景理解**上，
> 而动作解码所需的信息（夹爪在哪、往哪走）在四种渲染里同样可得。
> 特权信息若不落在**任务相关**的那部分，就不会转化成任务优势。
> 这也正是 *Factorized Contrastive Learning* 指出的问题：任务相关信息可能落在
> modality-unique 区域，也可能不落在——不能假设。

---

## 3. 退化现象的机制诊断

depth/normal 绝对变差，这不是新鲜事，文献里有对应的名字。

### 3.1 Modality laziness / greedy learner

多模态联合训练里，**收敛更快的模态会主导优化**，弱模态训练不足；深度模型的贪婪性
使它优先学「容易学的」模态，从而**压制**难学的模态。

与我们的数据吻合：训练里 RGB 占 0.40、其余各 0.20，RGB 的更新次数是其他三条的**两倍**，
而且 RGB 是预训练权重最熟悉的分布（warm start 自 `step125750`，那是纯 RGB 训练的）。

但要注意：**「更新少」只能解释涨得慢，解释不了绝对变差。** 绝对变差需要第二个机制。

### 3.2 梯度冲突 / 负迁移

负迁移的标准定义就是「因为学了别的任务，本任务性能反而下降」。
优化视角下它表现为**梯度冲突**：两个任务的梯度余弦相似度为负时，
沿一个任务的梯度走会损害另一个。

四条路径共用**全部**权重（arm7 没有任何按模态分开的模块），所以冲突无处可躲。
这与我们看到的「RGB 改善的同时 depth/normal 变差」在形式上完全一致。

### 3.3 但先别急着下结论：噪声地板

`ARM7_RESULTS.md` §4.4：相邻 checkpoint 之间同一路径的抖动实测 **5–20%**，
而要测的效应只有 20–30%。三个 p 值没有一个显著。
**「退化」目前是一个方向一致（6/6）但未达显著的观察**，n=80 的读数出来之前，
机制讨论只能是候选解释，不能当既成事实。

---

## 4. 方法族：按「需要什么前提」分类

关键不是方法本身，而是**每种方法要求模态之间是什么关系**。

| 族 | 代表 | 要求的前提 | arm7 满足？ |
|---|---|---|---|
| **A. 表示层对齐** | 对比学习 / 一致性正则 | 视图关于标签**冗余** | ✅ 强满足 |
| **B. 输出层互教** | Deep Mutual Learning | 无需 teacher，peer 对等 | ✅ 满足（四路等价正是对等） |
| **C. 单向蒸馏** | 跨模态蒸馏 / LUPI / 教师-学生 | 教师**确实更强** | ❌ **实测不满足** |
| **D. 解析约束** | depth↔normal 几何一致性 | 存在解析关系 | ✅ 满足（但作用机制与直觉不同，见 4.4） |
| **E. 优化层平衡** | OGM-GE / 梯度手术 | 存在模态不平衡 | ⚠️ 疑似满足（待 n=80 确认） |
| **F. 伪标签互标** | Co-training | 条件**独立** | ❌ **理论排除** |

### 4.1 A —— 表示层对齐（理论最站得住）

不是让一条路径去纠正另一条的**输出**，而是要求四条路径对同一场景产生**一致的内部表示**。
理论上这正是冗余视图该用的方式（§1.2）。

对 arm7 的具体形式：同一样本、同一窗口，走两条不同路径的两次前向，
在 DiT 的某一层特征上加对齐损失（余弦 / InfoNCE）。

**代价**：每步两次前向（action 段与相机段的 VAE 编码可共享，实际 <2×）。
**风险**：对齐层选错等于什么都没做；太靠近输入则强迫四种像素分布相同（不可能），
太靠近输出则退化成 B。

### 4.2 B —— 输出层互教（Deep Mutual Learning）

DML 的核心主张：**不需要预先存在的强教师**，一群对等的学生互相模仿，
效果甚至超过从一个更强但静态的教师蒸馏。每个学生带两个损失：常规监督损失 +
模仿同伴输出分布的损失。

**这正是 arm7 的处境**：四条路径对等（24/48），没有教师。
而且我们有一个别人没有的便利条件——

> **四条路径的 action 段监督目标逐像素相同**：action 像素是从**同一份 `action_7d`**
> 经**同一对相机**渲染出来的（`train.py:369-377`），四个模板共用同一份外参内参。
> 所以「让两条路径的 action 预测互相靠拢」是**良定义**的，不需要任何对齐启发式，
> 也不需要 DML 那样在类别分布上做 KL——直接在 action latent 上做 L2 即可。

这是本节最值得试的一条。

### 4.3 C —— 单向蒸馏：**已被实测排除**

见 §2。保留记录以免重复提出。

### 4.4 D —— 解析约束：我之前的判断是错的

`CROSSMODAL_FUSION_PLAN.md` §4.1 写过：「normal 相对 depth 零新增信息，
这个约束模型可以平凡满足，基本没价值」。**这个判断与文献相反。**

联合 depth + normal 估计的文献一致报告：**显式的几何一致性约束是有益的、且是必要的**，
仅靠多任务学习而不加约束，在深度变化剧烈的区域（褶皱、遮挡边界）得不到改善。

**为什么「零新增信息」仍然有用**——机制不是信息，是**误差谱重加权**：
`normal` 依赖 depth 的**局部梯度**，所以 normal 损失会重罚 L1/L2 深度损失几乎看不见的
**高频**误差。约束不提供新信息，它**改变误差在频谱上的分配**。

对 arm7 的含义：`depth+action` 与 `normal+action` 两条路径之间的一致性约束，
可能改善的是**几何细节**，而不是动作精度。是否值得，取决于动作误差里有多少来自
高频几何——这可以用 `ARM7_RESULTS.md` §4.4 提到的 bearing/range 分解去测。

### 4.5 E —— 优化层平衡（针对退化现象）

若 n=80 确认 depth/normal 确实退化，最直接的对策不是加损失，而是**改优化**：

- **OGM-GE**（*Balanced Multimodal Learning via On-the-fly Gradient Modulation*, CVPR 2022 Oral）：
  监控模态间的判别力差距，在反传时**抑制主导模态的梯度**，给弱模态留出优化空间。
- **梯度手术**：检测余弦相似度为负的冲突梯度并投影/重加权。
- **最便宜的一招**：直接改采样比例。arm7 现在是 0.4/0.2/0.2/0.2；
  改成 0.25×4 就消除了「RGB 更新次数两倍」这个最简单的解释。
  **这一条不需要任何新代码**——`--template_mix` 就是个命令行参数。

### 4.6 F —— 伪标签互标：**理论排除**

见 §1.1。冗余视图之间做 co-training 不会带来新信息。

---

## 5. 与最接近的先行工作的关系

| 工作 | 结构 | 与 arm7 的关键差异 |
|---|---|---|
| **Multi-Modal Policy Consensus**（2509.23468） | 每模态一个**独立 expert 网络** + router 学一致性权重 | 它融合的是**互补**传感器（视觉+触觉，看到不同的东西）；arm7 是**一个网络**、四条 prompt 选择的路径、模态**冗余**。互补设定里「分歧」不一定是错误；冗余设定里 GT 唯一，**分歧就是误差** |
| **4M / 4M-21**（NeurIPS 2023） | 单个 Transformer，把所有模态 token 化，做多模态掩码建模，any-to-any | 结构上最接近 arm7（一个模型、多模态、掩码、可任意条件生成）。但 4M 的目标是**通用视觉表征**，arm7 的目标是**单一动作输出**——所以 4M 不面对「四条路径解同一个动作、彼此该不该互教」这个问题 |
| **Deep Mutual Learning**（CVPR 2018） | 多个**独立网络**互相模仿 | arm7 是同一套权重的四条路径，比 DML 更紧耦合：互教在 DML 里是跨网络正则，在 arm7 里是**同一组参数内部**的自洽约束 |
| **跨模态蒸馏**（Gupta et al., CVPR 2016） | 有标注模态 → 无标注配对模态 | 要求教师更强；arm7 实测不满足 |

**arm7 的定位**：一个网络、多条**冗余**输入路径、**共享**输出头。
这个格子在文献里相对空——大多数多模态工作处理的是**互补**模态（要融合），
或者**独立网络**（要蒸馏/集成）。

---

## 6. 对 arm7 的具体建议

按「先验成功率 × 代价」排序：

### 6.1 先做：改采样比例（零代码，最强的对照）

`--template_mix video+action@0.25,depth+action@0.25,segmentation+action@0.25,normal+action@0.25`

如果 depth/normal 的退化在均匀采样下消失，那它就是**简单的更新次数不平衡**（§3.1），
不需要任何 §4 的机制。**这是必须先排除的最朴素解释**，而且它是一个命令行参数。

### 6.2 再做：action latent 互教（§4.2）

同一样本两条路径两次前向，在 action latent 上加对称 L2。
理由：监督目标逐像素相同（§4.2 的引用块），良定义、无需启发式；
DML 表明对等 peer 互教**不需要更强的教师**——正好匹配四路等价的实测。

**判据**：RGB 路径的闭环成功率（headline），以及四条路径的比值是否被拉回 1.0 附近。

### 6.3 视情况：OGM-GE 式梯度调制（§4.5）

只有在 6.1 证明退化**不是**采样比例造成的时候才值得做。

### 6.4 不做

- 单向蒸馏（§4.3，前提被实测否定）
- co-training 式伪标签（§4.6，理论排除）
- depth↔normal 解析约束（§4.4）——除非 bearing/range 分解显示动作误差里高频几何占大头

---

## 7. 这份研究改变了什么

| 之前的判断 | 现在 |
|---|---|
| 「depth/seg 是特权信息 → 当教师蒸馏给 RGB」（`CROSSMODAL_FUSION_PLAN.md` §1、§4.2） | **前提被实测否定**：四路等价，无教师可言 |
| 「normal 是 depth 的解析函数 → 一致性约束可平凡满足、没价值」（同上 §4.1） | **与文献相反**：约束有益，机制是**误差谱重加权**而非新信息 |
| 「集成/互相监督应该有用」 | 需要区分**用冗余做什么**：传播标签无效（co-training），逼不变性有效（对比 / 互教） |
| 未意识到退化现象 | 有了名字（modality laziness / 负迁移）和**零成本的首选对照**（均匀采样比例） |

---

## 8. 参考

- [Contrastive learning, multi-view redundancy, and linear models (Tosh, Krishnamurthy, Hsu)](https://arxiv.org/abs/2008.10150) —— 冗余视图下对比学习的理论保证
- [Factorized Contrastive Learning: Going Beyond Multi-view Redundancy (NeurIPS 2023)](https://openreview.net/forum?id=alLs7EtRJP) —— 任务相关信息可能落在 modality-unique 区域
- [A Survey on Multi-view Learning](https://arxiv.org/pdf/1304.5634) —— co-training 三假设与条件独立
- [Deep Mutual Learning (CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.pdf) —— 对等互教，无需强教师
- [Balanced Multimodal Learning via On-the-fly Gradient Modulation (CVPR 2022 Oral)](https://openaccess.thecvf.com/content/CVPR2022/papers/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.pdf) —— 抑制主导模态梯度
- [BalanceBenchmark: A Survey for Multimodal Imbalance Learning](https://arxiv.org/pdf/2502.10816) —— modality laziness 综述
- [Multi-Task Learning with Deep Neural Networks: A Survey](https://arxiv.org/pdf/2009.09796) —— 负迁移与梯度冲突
- [Cross Modal Distillation for Supervision Transfer (Gupta, Hoffman, Malik, CVPR 2016)](https://arxiv.org/pdf/1507.00448)
- [4M: Massively Multimodal Masked Modeling (NeurIPS 2023)](https://arxiv.org/abs/2312.06647)
- [Multi-Modal Manipulation via Multi-Modal Policy Consensus (2509.23468)](https://arxiv.org/abs/2509.23468)
- [Multi-task learning with cross-task consistency for improved depth estimation](https://arxiv.org/pdf/2311.18664) —— depth↔normal 几何一致性有益
