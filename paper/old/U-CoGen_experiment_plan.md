# U-CoGen 实验部分重构与实验计划

## 1. 总体结论：实验部分应该回答 5 个核心问题

你的 Method 已经比 Experiments 清楚很多。现在真正需要做的是让实验严格围绕论文的核心 claim 来组织，而不是围绕“已经跑出了哪些有趣数字”来组织。

这篇文章真正需要证明的不是“depth 能生成出来”，而是下面五件事：

| Research Question | 要证明什么 | 最关键实验 |
|---|---|---|
| **RQ1: Unified capability** | 一个 checkpoint、一个 output head、一个 loss，确实能做 action / depth / segmentation / normal / RGB | arm6 在完整 test set 上的五模态 quantitative evaluation |
| **RQ2: Cost of sharing** | 共用一个 head 没有造成严重 negative transfer | unified vs 单模态 specialist；arm0 → arm4 → arm1 → arm6 |
| **RQ3: Conditioning factorization** | template 和 conditioning plan 真的是两个独立控制轴，而且训练分布决定最终能力 | M0 / M1 / M2 × IIII / FIII / FIFI 矩阵 |
| **RQ4: Policy preservation** | 加入 perception 之后 action/world-action policy 没坏 | arm0 vs arm6 offline action + closed-loop |
| **RQ5: Generalization / extensibility** | 这不是记住训练 episode，而是能泛化；最好还能加新 modality 而不改 architecture | held-out variations；可选 new-modality experiment |

当前 Experiments 更像是在回答：

> “我在 open_drawer 的某几个 checkpoint 上观察到一些很有意思的现象。”

最终应该变成：

> **“这里有一套系统实验，逐项验证 unified shared-head world-action generation 的核心假设。”**

---

## 2. 当前实验部分最大的几个问题

### 2.1 大量 main-paper 结论来自 n=1 或 n=4

目前有：

- `One checkpoint, four prompts`: `n=1 episode`
- action table: `n=1 episode`
- conditioning depth: 4 episodes
- qualitative: 一个 `open_drawer`

这些可以作为 **qualitative demonstration / debugging observation**，但不应该作为 main quantitative evidence。

尤其是类似：

> “That a checkpoint whose training mixture is 60% non-action still recovers 1.9 cm inverse dynamics is our strongest evidence that the shared head is not being torn apart…”

reviewer 很容易直接质疑：

> “This is one episode.”

真正需要的是：

**16 tasks × held-out episodes/variations × multiple views 的平均值和 confidence interval。**

---

## 3. 最缺的一张表：真正的 headline table

建议主表不是当前的 `versatility` table，而是：

### Table 1 — One shared checkpoint vs specialists

| Model | # checkpoints | Action pos ↓ | Action rot ↓ | Depth AbsRel ↓ | Seg mIoU ↑ | Normal cos ↑ | RGB LPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| ActionImages init | 1 | … | … | – | – | – | … |
| Action-only continuation (arm0) | 1 | … | … | – | – | – | … |
| Depth specialist | 1 | – | – | … | – | – | … |
| Seg specialist | 1 | – | – | – | … | – | … |
| Normal specialist | 1 | – | – | – | – | … | … |
| **U-CoGen arm6** | **1** | **…** | **…** | **…** | **…** | **…** | **…** |

全部在：

> **the same held-out test set under IIII**

上评价。

这张表非常重要，因为 reviewer 第一反应一定是：

> “Okay, one head can do everything. But how much quality do you lose compared with separate models?”

当前实验还没有回答这个问题。

---

## 4. Specialist baseline 是必须补的

当前已经有：

- arm0: action
- arm4: action + depth
- arm1: action + depth + segmentation
- arm6: action + depth + segmentation + normal

但这还不能回答：

> 如果单独训练 depth，性能是不是 AbsRel 0.03，而 unified 是 0.10？

即使 unified 有一定性能损失，“一个模型做四件事”依然有价值，但必须把 tradeoff 诚实展示出来。

因此建议额外训练三个 specialist：

\[
\texttt{video+depth only}
\]

\[
\texttt{video+seg only}
\]

\[
\texttt{video+normal only}
\]

都从同一个 ActionImages checkpoint warm start。

不需要训练特别久，因为现有结果已经显示 perception 在 1500–4000 step 内可以迅速学出来。

---

## 5. 公平性问题：不能直接比较 global step

这是一个 reviewer 很可能会抓的点。

arm6 的训练分布是：

\[
p(A)=0.4,\quad p(D)=p(S)=p(N)=0.2
\]

假如 arm6 训练 5000 steps，大概只有：

- action updates = 2000
- depth updates = 1000
- segmentation updates = 1000
- normal updates = 1000

而 depth specialist 训练 5000 steps，则有 **5000 depth updates**。

因此直接比较：

> depth specialist @5000 vs arm6 @5000

不是纯粹的 sharing penalty，而是同时包含：

> **5× fewer depth samples.**

### 推荐做法：以 modality-specific updates 为横轴

例如 depth：

\[
N_D(t)=\sum_{i=1}^{t}\mathbf{1}[\Pi_i=\texttt{video+depth}]
\]

然后画：

> **Depth AbsRel vs number of depth updates**

比较：

- depth specialist
- arm4
- arm1
- arm6

segmentation 和 normal 同理。

action 则画：

> Action error / rollout success vs cumulative action updates

这样可以区分：

- **data allocation cost**
- **representation interference**

如果 arm6 在相同 depth update 数下与 specialist 接近，那是一个很强的结果：

> sharing the head itself causes little interference; most of the apparent gap comes from reduced task sampling frequency.

甚至有可能 arm6 更好，那就是 positive transfer。

---

# 6. 推荐的 Experiments 总体结构

## 6.1 Experimental setup

这里只写：

- Dataset
- train / validation / test split
- baselines
- training
- evaluation protocol
- metrics

不要在 setup 里讲结果。

### Train

variation-0 的训练 trajectories。

### Validation

variation-0 中重新生成的 held-out seeds，或独立 trajectories。

用于：

- checkpoint selection
- hyperparameter selection
- M0 / M1 / M2 selection

### Test

**all held-out variations**。

当前 draft 已经有：

> variation-0 is trained on and all other variations are held out

这个 split 本身非常漂亮。

但是后续主结果反而大量使用 held-in `open_drawer` episode，相当于浪费了最好的 generalization setup。

---

# 7. Section 4.2：One shared model predicts every modality

这一节应该放真正的 headline result。

不是当前 n=1 的 table，而是在整个 test set 上报告。

### 评价 regime

建议主要使用：

\[
\texttt{IIII}
\]

因为这是你真正的 world-model deployment mode。

### 报告指标

#### Action

至少包括：

- position error
- rotation error
- gripper accuracy / F1

`r_peak` 可以保留，但应该移到 appendix。

因为：

> “blob 还在”

并不等于：

> “policy 还在”。

#### Depth

- AbsRel
- 可选 RMSE / δ1

#### Segmentation

- macro mIoU
- 可选 role-wise IoU

#### Normal

- mean cosine
- angular error

#### RGB future

- PSNR
- SSIM
- LPIPS

---

# 8. “One checkpoint, four prompts” 应该保留，但作为 qualitative evidence

当前 Figure 非常适合作为 teaser。

标题甚至可以保留：

> **One checkpoint, four prompts.**

但是正文不要再把这一张图里的：

- AbsRel 0.088
- mIoU 0.435
- cosine 0.895

作为 quantitative evidence。

更合适的写法是：

> Figure X visually illustrates the prompt-switched behavior; Table Y evaluates the same capability over the full held-out test set.

这样 qualitative 和 quantitative 的角色就清楚了。

---

# 9. Section 4.3：What is the cost of sharing one output head?

这一节可以直接利用已有的：

\[
arm0\rightarrow arm4\rightarrow arm1\rightarrow arm6
\]

这是当前实验设计里最有潜力的一组实验。

### x-axis

Number of supported modalities:

\[
1,\ 2,\ 3,\ 4
\]

### y-axis

分别画：

- action performance
- depth performance
- segmentation performance
- normalized aggregate score（可选）

模型定义：

| Model | Training modalities |
|---|---|
| arm0 | A |
| arm4 | A + D |
| arm1 | A + D + S |
| arm6 | A + D + S + N |

但必须处理 sampling-rate confound。

### Panel A: Fixed compute

所有模型在相同 total steps 比较。

回答：

> 在同样训练预算下，task menu 变大有什么代价？

### Panel B: Matched task exposure

例如 action：

- arm0 @ 2000 action updates
- arm4 @ 2000 action updates
- arm1 @ 2000 action updates
- arm6 @ 2000 action updates

回答：

> 如果每个 task 得到相同训练量，sharing 本身有没有 interference？

这是非常成熟、非常有说服力的 experiment。

---

# 10. Section 4.4：Conditioning axis 是第二个真正有 novelty 的实验

当前的：

> “The conditioning axis is not free”

方向非常对，甚至比 closed-loop transient 更值得进 main paper。

但目前的问题是：

- 只有 4 个 episodes
- M0 vs M2 没有做完整 causal comparison

Method 明确提出：

\[
p(\Pi,\rho)=p(\Pi)p(\rho\mid\Pi)
\]

实验应该非常工整地验证这个 factorization。

建议训练：

| Mixture | IIII | FIII | FIFI |
|---|---:|---:|---:|
| M0 | 0.10 | 0 | 0.90 |
| M1 | 0.45 | 0.10 | 0.45 |
| M2 | 0.90 | 0.05 | 0.05 |

然后做：

### Train-mixture × Test-mode matrix

例如 depth：

| train | IIII ↓ | FIII ↓ | FIFI ↓ |
|---|---:|---:|---:|
| M0 | … | … | **…** |
| M1 | … | … | … |
| M2 | **…** | … | … |

全部在：

> hundreds of held-out clips

上计算。

如果趋势是：

\[
M0 \rightarrow \text{best FIFI}
\]

\[
M2 \rightarrow \text{best IIII}
\]

就能非常直接支持 Introduction 中的：

> perception is not a modality; it is a \((\Pi,\mathcal{C})\) pair.

---

# 11. 推荐增加 Pareto 图

横轴：

\[
\text{FIFI depth error}
\]

纵轴：

\[
\text{IIII depth error}
\]

三个点：

- M0
- M1
- M2

可以得到一个 tradeoff frontier。

这个比现在列出：

- `open_drawer`: 1.335
- `push_buttons`: 0.041

更有概括性、更像 main-paper result。

当前 episode-specific instability 很有意思，但更适合 appendix。

---

# 12. Section 4.5：Policy preservation 必须比较 arm0 和 arm6

当前 headline model 是：

> **arm6**

但 closed-loop table 目前是：

> **arm4**

所以目前真正证明的是：

> action + depth co-training 没有完全破坏 policy。

却没有证明：

> U-CoGen 的完整四流 checkpoint 能作为 WAM policy。

这是 World Action Model paper 里非常危险的缺口。

---

# 13. Closed-loop 最重要的三列

建议主表至少包含：

| Model | Purpose |
|---|---|
| ActionImages init | starting point |
| arm0 action-only continuation | domain-finetuning control |
| **arm6 U-CoGen** | proposed method |

这样可以区分两个效果：

\[
\text{init}\rightarrow arm0
\]

代表：

> RLBench domain continuation gain

而：

\[
arm0\rightarrow arm6
\]

才代表：

> perception co-training effect.

arm4 / arm1 可以作为 ablation。

---

# 14. 不一定要证明 perception 提升 action

这一点非常重要。

如果最后结果是：

| Model | Success |
|---|---:|
| arm0 | 44% |
| arm6 | 43% |

这完全可以接受。

你的 claim 可以是：

> **Adding three dense perception outputs does not materially degrade control.**

这已经是很强的结果。

你的论文并不是在 claim：

> Auxiliary perception improves robot control.

真正 novelty 是：

> one generative policy can support all these outputs without separate heads.

所以：

\[
arm6\approx arm0
\]

已经足够好。

不要为了故事强行 claim：

> perception improves policy.

---

# 15. Closed-loop sample size 建议

当前：

> 5 tasks × 20 scenes

作为 pilot 很合理。

正式版建议：

### 最好

\[
8\text{--}16\ tasks\times20\ seeds
\]

至少主模型：

- init
- arm0
- arm6

全部使用相同 scene seeds。

然后报告：

\[
95\%\ CI
\]

以及 paired statistics。

当前用 McNemar 是合理的。

---

# 16. 必须报告 RGB future quality

你叫：

> **World Action Model**

所以模型除了 action 外，还承担：

> video generation。

因此 arm6 不仅要证明 action 没坏，还应该证明：

> RGB world model 没坏。

建议 under IIII 报：

- PSNR
- SSIM
- LPIPS

FVD 不一定值得跑，尤其是 RLBench 数据规模不大时。

当前 table 里的：

> given-RGB PSNR

基本没有意义，因为那主要反映 frozen VAE reconstruction ceiling。

真正要比较的是：

> **predicted future RGB under IIII**

尤其是：

\[
arm0 \text{ vs } arm6
\]

如果 RGB LPIPS 基本不变，会进一步支持：

> shared output space does not destroy world modeling.

---

# 17. FIFI 和 IIII 实际上测的是两种不同能力

这是一个应该主动解释的 conceptual point。

## FIFI

RGB future 是 given 的。

所以：

\[
RGB\rightarrow depth
\]

主要测：

> **perception / measurement decoding ability**

## IIII

RGB future 本身也是 model-generated。

所以：

\[
initial\ scene
\rightarrow
future\ RGB + future\ depth
\]

AbsRel 同时包含：

- future prediction error
- perception generation error

因此不建议仅用 IIII AbsRel 宣称：

> “our depth estimator reaches X.”

更准确的术语是：

> joint future-depth generation error.

而 FIFI 才更接近：

> conditional depth estimation accuracy.

这与论文的 conceptual framing 完全一致。

---

# 18. 推荐的关键 ablation：Remove modality tags

Method 中一个重要设计是：

> template is announced verbatim in the prompt.

reviewer 很自然会问：

> prompt tag 真的重要吗？

可以做一个很小但非常干净的 ablation：

### U-CoGen w/o modality tags

仍然随机训练：

- video+depth
- video+segmentation
- video+normal
- video+action

但 prompt 只有：

> `slide the bottom drawer open`

没有：

> `<video><depth>`

由于所有 auxiliary modality 都 occupy 同一个 slot，模型无法知道第二个 segment 应该是什么。

理论上性能会明显混乱。

如果确实下降，就能直接证明：

> **The prompt tag, rather than an architectural router, identifies the output semantics.**

这是非常契合论文 novelty 的 ablation。

---

# 19. Codec experiment：当前基本够，但定位应该是 representation validation

当前 admission gate 已经有：

- depth VAE AbsRel 0.362%
- normal cosine 0.9906
- segmentation target IoU 0.965

这些结果很好。

它们证明：

> downstream model performance limitation is not caused by the frozen VAE destroying the encoded measurements.

这个实验应该保留。

建议正文只放一个 compact table，完整细节移 Appendix。

如果算力允许，可以补：

### Depth encoding ablation

比较：

- grayscale
- ordinary colormap
- RGB cube / Gray-code path

经过 frozen VAE 后的 AbsRel。

这可以进一步证明 codec 设计合理。

但优先级是 **P1，不是 P0**。

---

# 20. Reviewer 可能攻击：normal 是 depth 的 deterministic function

当前 surface normals 是：

> derived from metric depth.

因此 reviewer 可能会说：

> normal 并不是一个独立 sensor modality。

这不是致命问题，但 framing 要准确。

不要写：

> four independent perceptual signals.

更合适的是：

> heterogeneous robot-relevant output representations.

还可以增加一个 cross-representation consistency metric：

从 predicted depth 计算：

\[
\hat n_D
\]

与直接 predicted normal：

\[
\hat n_N
\]

比较：

\[
\cos(\hat n_D,\hat n_N)
\]

如果二者一致，就是一个很漂亮的结果：

> one model learns mutually consistent geometric representations.

因为 depth 和 normal 当前是不同 prompt 下的不同 generation，这个实验可以放 exploratory / appendix。

---

# 21. 当前数据描述有 inconsistency，需要马上修

Experiments Setup 目前写：

> per-frame metric depth and **handle masks**

但 Method segmentation 写：

> target, goal, robot_arm, gripper, distractor, tool, fixture, background...

这明显不一致。

reviewer 会问：

> 你到底只有 handle mask，还是有完整 role segmentation ground truth？

如果实际是 RLBench simulator semantic/object masks 再映射成 scene roles，就应该明确写：

> simulator-provided instance/object masks are deterministically mapped to the fixed scene-role vocabulary.

不能只写 handle masks。

---

# 22. 一个更深的 conceptual issue：现在不是“同时”生成 action + depth + seg + normal

当前训练 template 是：

\[
V+A
\]

或者：

\[
V+D
\]

或者：

\[
V+S
\]

或者：

\[
V+N
\]

也就是说：

> **one checkpoint can switch between action generation and perception generation**

但并不是：

\[
V+A+D+S+N
\]

同时 co-generate。

因此严格的 reviewer 可能会说：

> “This is unified multi-task generation, not simultaneous co-generation of robot actions and perception.”

标题：

> **Unified Co-Generation for World Action Models**

需要注意 wording。

---

# 23. 如果算力允许，强烈推荐一个 tri-modal template

例如：

\[
\boxed{\texttt{video+depth+action}}
\]

即：

\[
[V_0,D_0,A_0,V_1,D_1,A_1].
\]

只需要一个 experiment。

然后证明同一次 generation 可以 jointly produce：

- future RGB
- metric depth
- action

这会明显增强 “co-generation” 这个词的可信度。

不需要一次同时生成 D + S + N + A。

只要：

> **V + D + A**

就足以证明 perception 和 action 可以真正 coexist in one generation，而不只是 prompt switching。

如果这个实验太贵，则应该稍微收窄 abstract / introduction wording：

> a single model prompt-switches between world-action and world-perception generation

而不是暗示所有东西 simultaneous。

---

# 24. Bonus：新增一个从未训练过的 modality

Introduction 中目前有一句很大的 claim：

> “the task set is a property of the data mixture rather than of the network.”

如果想真正证明这句话，可以从 arm6 checkpoint 出发，再新增：

\[
\texttt{<flow>}
\]

例如 optical flow。

不修改：

- architecture
- output head
- loss

只：

- 定义 codec
- 加 `<flow>` tag
- 加 training data

训练 1–2k steps。

然后展示：

### 新任务

flow 学会了。

### 老任务

action / depth / segmentation / normal 基本保留。

如果成功，这会是整篇 paper 最漂亮的实验之一，因为它直接证明：

> **adding an output is a data operation rather than an architectural operation.**

这比简单写：

> “理论上以后还能加其他 modality”

强得多。

---

# 25. Loss-collapse 故事建议移出 main paper

当前现象：

> loss 最低的时候 rollout success 从 28.8% 掉到 6.7%，随后又恢复

科学上非常有意思。

但是和论文主要 contribution 没有直接关系。

它现在占据主文较大篇幅，而且可能把 reviewer 的注意力带向：

> Why is your training so unstable?

而不是：

> Wow, one head supports four robot output spaces.

### 建议主文只写一句

> Because diffusion loss is poorly correlated with closed-loop control, checkpoints are selected using a fixed validation rollout protocol.

### Appendix

放完整 loss/success curve。

这样更合适。

---

# 26. 四个 depth episode 的 instability 也更适合 Appendix

目前：

- `open_drawer FIFI = 1.335`
- `push_buttons FIFI = 0.041`

这是很好的 failure analysis。

但主实验应该是：

\[
M0/M1/M2
\times
IIII/FIII/FIFI
\]

在完整 dataset 上。

Appendix 再展示：

> under-trained conditioning modes can fail episodically.

这样逻辑会更稳。

---

# 27. 推荐的最终 Experiments 结构

## 4.1 Experimental setup

包括：

- Dataset
- train / val / test split
- models
- training
- metrics
- checkpoint selection

---

## 4.2 One model, every output

主 quantitative table：

> arm6 on RGB / action / depth / segmentation / normal over held-out test data.

然后放：

> “One checkpoint, four prompts” qualitative figure.

---

## 4.3 The cost of unifying the output space

比较：

\[
arm0,\ arm4,\ arm1,\ arm6
\]

以及单模态 specialists。

核心回答：

> Does performance degrade as the task menu grows?

最好用：

> modality-specific update count

做公平 comparison。

---

## 4.4 Template and conditioning are independent control axes

做：

\[
M0/M1/M2
\times
IIII/FIII/FIFI
\]

这是 conditioning contribution 的 main ablation。

---

## 4.5 Does perception co-training preserve the policy?

### Offline action metrics

\[
arm0\ vs\ arm6
\]

### RGB future metrics

\[
arm0\ vs\ arm6
\]

### Closed-loop

\[
init\ vs\ arm0\ vs\ arm6
\]

arm4 / arm1 作为 secondary ablation。

---

## 4.6 Generalization and extensibility

首先一定做：

> held-out RLBench variations.

算力允许再加：

- new flow modality
- V+D+A tri-modal generation

---

## Appendix

放：

- codec admission gate full details
- `r_peak`
- checkpoint/loss transient
- four-episode conditioning failures
- per-task results
- extra qualitative generations
- class-wise segmentation
- per-view metrics

---

# 28. 如果算力有限，建议实验优先级

## P0

### 1. arm6 在完整 held-out variation test set 上跑所有 modalities

这是现在最急缺的。

不要再以 `open_drawer n=1` 作为 main evidence。

### 2. closed-loop arm0 vs arm6

headline model 必须自己跑 rollout；arm4 不能替 arm6 证明 policy preservation。

### 3. 三个 perception specialists

- V+D
- V+S
- V+N

用于回答 unified performance cost。

### 4. M0 / M1 / M2 conditioning experiment

至少 depth 做完整 held-out test matrix。

它直接验证第二个方法贡献。

---

## P0 / P1

### 5. Predicted RGB quality: arm0 vs arm6

用 IIII 的：

- future PSNR
- SSIM
- LPIPS

不是 given-RGB VAE PSNR。

---

## P1

### 6. No-modality-tag ablation

非常干净地验证：

> prompt tag is the semantic router.

### 7. V+D+A tri-modal generation

如果成功，会明显强化 “co-generation”。

---

## P1 / P2

### 8. 新增 optical-flow modality

这是证明 architecture-free extensibility 的 killer experiment。

---

# 29. 如果只能再跑“三组训练”

如果 GPU budget 很紧，建议优先：

## 第一组：arm6 full evaluation + closed-loop

严格来说甚至不需要重新训练，只需要系统 evaluate。

这是最高 ROI。

## 第二组：Depth / Seg / Normal specialists

因为缺 specialist baseline 是目前最大的 reviewer hole。

## 第三组：M0 / M1 / M2

因为它验证你的 template-condition factorization，而不是简单证明“多任务模型可以 work”。

`no-tag` 和 tri-modal 可以在 supplementary / rebuttal budget 中再做。

---

# 30. 最终实验 narrative 应该是什么

不要写成：

> We generated depth.  
> We generated segmentation.  
> We generated normals.  
> Our action blobs still look alive.  
> Sometimes FIFI fails.  
> Sometimes rollout collapses.

而应该形成下面的逻辑：

> **A single shared generative head supports all robot outputs.**

然后：

\[
\Downarrow
\]

> **Its quality approaches modality specialists despite sharing all parameters.**

然后：

\[
\Downarrow
\]

> **Increasing the task menu causes little intrinsic interference once task exposure is controlled.**

然后：

\[
\Downarrow
\]

> **Template determines what is generated; conditioning determines what is observed, and changing the training distribution moves performance predictably across this axis.**

然后：

\[
\Downarrow
\]

> **Most importantly, the full unified model preserves RGB world prediction and closed-loop action performance.**

这个故事和当前 Method 的逻辑是完全对齐的。

---

# 31. 最重要的四张 main-paper 表 / 图

## Table 1 — U-CoGen vs specialists

五模态完整 quantitative comparison。

核心回答：

> 一个 shared checkpoint 的性能与专用模型相比损失多少？

---

## Figure 2 — arm0 → arm4 → arm1 → arm6

展示：

> performance vs modality-specific training exposure

核心回答：

> task menu 扩大后到底是 sampling dilution 还是 genuine interference？

---

## Table / Figure 3 — Conditioning matrix

\[
M0/M1/M2
\times
IIII/FIII/FIFI
\]

或 Pareto frontier。

核心回答：

> training conditioning distribution 是否系统地决定 checkpoint 最终能回答哪些问题？

---

## Table 4 — Closed-loop policy preservation

比较：

\[
init\ vs\ arm0\ vs\ arm6
\]

核心回答：

> 完整 U-CoGen 是否仍然保持 world-action policy 能力？

---

# 32. 总结

如果上述四组核心结果做扎实，实验部分会比当前版本强一个档次。

整篇论文的实验应该始终围绕一句话展开：

> **One head, one loss, prompt-selected heterogeneous robot outputs.**

真正需要建立的证据链是：

1. **一个模型确实能输出所有模态。**
2. **和 specialist 相比，sharing 的性能代价可接受。**
3. **task menu 扩大后的损失可以区分为 sampling dilution 与真实 interference。**
4. **template 与 conditioning plan 是两个可独立控制的轴。**
5. **完整 unified model 不破坏 RGB world prediction 和 closed-loop action policy。**
6. **最好进一步证明新 modality 可以只通过数据和 prompt 加入，而不需要改 architecture。**

做到这里，实验部分就会和 Method 的贡献完全对齐，也会更有能力抵抗 reviewer 对 baseline、公平性、generalization、policy preservation 和 “co-generation” 定义的质疑。
