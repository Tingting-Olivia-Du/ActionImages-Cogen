# XGenAct / ActionImages-Cogen 模型架构全解

> 审计日期：2026-09-25
>
> 论文工作区：`/workspace/1228_tingting/6a89bc13108d28a2448f410a`，commit `987a219`
>
> 代码工作区：`/workspace/1228_tingting/ActionImages-Cogen`，commit `a98192b`
> 两个工作区在审计时都已 `git fetch`，本地 `main` 与远端 `main` 均为 0 ahead / 0 behind。本文以当前工作树为准，同时用实际 checkpoint 的 tensor header 复核参数量。
## 1. 一句话结论

XGenAct 不是“RGB encoder + depth head + segmentation head + action head”的多头网络。它把 RGB、metric depth、surface normal、functional-role segmentation 和 7-DoF action 都确定性地画成三通道视频，用同一个冻结的 Wan2.2 VAE 编码，再把所选模态和两个视角的 latent segment 沿时间轴拼成一张 canvas；一个共享的、带逐层相机条件分支的 30-block DiT 对整张 canvas 做 flow matching，最后仍由同一个线性输出头和同一个冻结 VAE decoder 生成所有模态。

最重要的精确数字是：

- 原始 Wan2.2-TI2V-5B DiT：`4,999,787,712` 参数。
- 每个 DiT block 新增 camera encoder 和 projector，共新增 `1,415,761,920` 参数。
- 实际训练并保存在 XGenAct checkpoint 中的 DiT：`6,415,549,632` 参数，而不是严格意义上的 5B。
- 冻结 UMT5 encoder：`5,680,910,336` 参数。
- 冻结 Wan2.2 VAE38 encoder/decoder：`704,688,668` 参数。
- 完整加载管线：`12,801,148,636` 参数；主实验只训练其中的 `6,415,549,632` 个 DiT 参数。

论文中的“5B diffusion transformer”准确地描述了 backbone 型号，但没有计入 Action Images / 本仓库加入的逐层相机模块。更精确的表述应是：**Wan2.2-TI2V-5B backbone + 1.416B per-block camera-conditioning modules，最终 denoiser 为 6.416B。**

## 2. 完整文字架构图

下面以论文主模型 arm7 的一次 `m+action` 样本为例，其中 `m` 每步只取 RGB、depth、segmentation、normal 中的一种；主实验不是把四种感知模态同时塞进一次 forward。

```text
                                      language instruction + modality tag
                                      例如 <depth><action> open the drawer
                                                     │
                                                     ▼
                                      UMT5-XXL Encoder（冻结，5.681B）
                                      512 tokens × 4096
                                                     │
                                      Linear 4096→3072 → GELU → Linear 3072→3072
                                                     │
                                                     └──────────────┐
                                                                    │ cross-attention context

  两个随机相机视角 v0、v1                                             │
  每个 stream: [3, 41, 512, 512]                                    │
          │                                                         │
          ├─ RGB: identity codec                                    │
          ├─ depth: metric-depth → RGB-cube path                    │
          ├─ normal: (n+1)/2                                        │
          ├─ segmentation: fixed functional-role palette             │
          └─ action: 7-DoF → 3 projected 2D Gaussian heatmaps RGB    │
          │
          ▼
  同一个 Wan2.2 Video VAE38 Encoder（冻结，所有模态共享）
  [3,41,512,512] → [48,11,32,32]
          │
          ▼
  view-major / modality-minor packing
  [ m(v0) | action(v0) | m(v1) | action(v1) ]
  4 × 11 latent frames → Z: [B,48,44,32,32]
          │
          ├─ conditioning plan 生成 clean/noisy mask
          │    I: 仅 segment 第一个 latent frame 干净
          │    F: 整个 segment 干净
          │    1: segment 只保留第一个 latent frame
          │
          ├─ clean 位置保持 Z；prediction 位置变为 (1-σ)Z + σε
          │
          ▼
  Conv3d patch embedding: 48→3072, kernel=stride=(1,2,2)
  [B,48,44,32,32] → [B,3072,44,16,16]
                     → 11,264 tokens × 3072
          │
          │       相机支路（两个视角逐像素 Plücker rays）
          │       direction RGB ─┐
          │                       ├─ 同一个冻结 VAE encoder
          │       moment/3 RGB ──┘  → concat → C: [B,44,96,32,32]
          │                                     │
          │                 每个 DiT block 自己的 camera encoder：
          │                 AdaptiveAvgPool3d(48,16,16)
          │                 Flatten 12,288 → Linear 12,288→3,072
          │                 再复制到每个 16×16 spatial token
          │                                     │
          ▼                                     ▼
  ┌──────────────── DiT Block × 30（每层结构相同、权重不共享） ────────────────┐
  │ AdaLN(t) → 加 camera token → 24-head global self-attention + 3D RoPE       │
  │          → Linear projector → gated residual                               │
  │ affine LayerNorm → 24-head cross-attention(language) → residual            │
  │ AdaLN(t) → MLP 3072→14336→3072 (GELU-tanh) → gated residual                │
  └──────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
  time-conditioned output head
  LayerNorm(no affine) → Linear 3072→(48×1×2×2=192)
          │
          ▼
  unpatchify → predicted flow vθ: [B,48,44,32,32]
          │
          ├─ training: 仅 prediction mask 上计算 weighted flow-matching MSE
          │
          └─ inference: 50-step scheduler 仅更新 prediction mask，clean mask 固定
                    │
                    ▼
       按 segment 切开 → 同一个冻结 VAE38 Decoder
       [48,11,32,32] → [3,41,512,512]
                    │
       ┌────────────┼───────────────┬────────────────┬─────────────────┐
       ▼            ▼               ▼                ▼                 ▼
      RGB       metric depth     unit normal    role labels      action images
    identity    path inverse      2RGB-1         palette NN      多视角几何解码
                                                                    │
                                                                    ▼
                                                    xyz + quaternion + gripper
```

## 3. 输入、模态 codec 与 template

### 3.1 原始输入

主训练设置使用：

- 每个 episode 从可用相机中随机抽两个不同视角；若数据只有一个视角则复制为两个。
- 每个视角 `T=41` 帧、`H=W=512`。
- 主实验 temporal stride 为 3；RLBench 原始频率 20 Hz，因此窗口跨度为 `(41-1)×3/20 = 6.0 s`，有效频率约 6.67 Hz。
- 动作原始表示为 `[x,y,z,rx,ry,rz,gripper]`，即位置、XYZ Euler rotation、开合状态。
- 相机输入包含每帧 `3×4` camera-to-world extrinsic 和 `3×3` intrinsic。模型相机分支使用以第一个 condition camera 为基准的 relative pose；动作投影使用绝对 camera-to-world 参数。

### 3.2 所有模态都先变成 RGB 视频

| 模态 | 进入 VAE 前的确定性表示 | 学习型专用 encoder/head |
|---|---|---:|
| RGB | identity，再归一化到 `[-1,1]` | 无 |
| metric depth | Barron power transform 后沿 RGB cube 的 Gray-code Hamiltonian path 编码 | 无 |
| normal | `RGB=(n+1)/2`；解码后映回 `[-1,1]^3` 并单位化 | 无 |
| functional-role segmentation | 全数据集固定 palette；角色包括 background、target、goal、robot arm、gripper、distractor、tool、fixture | 无 |
| action | 三个 3D 点投影成三个 RGB Gaussian heatmap；蓝色低响应背景同时编码 gripper openness | 无 |

动作图的几何是：

```text
R channel: q_pos    = project(p)
G channel: q_up     = project(p + 0.1 * R_eef * [ 1, 0, 0])
B channel: q_normal = project(p + 0.1 * R_eef * [ 0, 0,-1])
Gaussian σ = 0.05 * min(H,W)
B <= 0.25 的背景写入 openness * 0.25
```

因此 action stream 对 VAE 来说也只是普通三通道视频。虽然函数名仍叫 `project_actions_7d_to_5d`，其实际中间 tensor 最后一维为 7：三个二维点共 6 个数，加一个 gripper 值。

### 3.3 主模型的 template 并不是五模态同时输入

论文主模型 arm7 的 template sampling 为：

| template | 概率 | packed order（两个视角） |
|---|---:|---|
| `video+action` | 0.4 | `video(v0), action(v0), video(v1), action(v1)` |
| `depth+action` | 0.2 | `depth(v0), action(v0), depth(v1), action(v1)` |
| `segmentation+action` | 0.2 | `seg(v0), action(v0), seg(v1), action(v1)` |
| `normal+action` | 0.2 | `normal(v0), action(v0), normal(v1), action(v1)` |

所以每个常规样本都是 **一种视觉/感知空间 + action**，固定四个 segment、相同计算量。四种感知任务通过不同训练步共享同一 DiT，而不是在一个主实验样本里同时出现。

代码也支持 `video+depth+segmentation+normal+action` 等 fusion canvas；五模态、双视角时会有 10 个 segment，但这不是论文表中 arm7 的主训练布局。

### 3.4 模态身份如何告诉模型

模态身份主要来自：

1. prompt 前缀，例如 `<video><action>`、`<depth><action>`、`<normal><action>`、`<scene-seg><action>`；
2. 固定的 segment 顺序；
3. 各模态自身的像素统计。

DiT 内部没有单独的 learned modality embedding、segment embedding 或 modality-specific adapter。代码的 canonical order 是：

```text
video → depth → segmentation → normal → action
```

segment 按 **view-major, modality-minor** 排列。

## 4. 冻结的 Wan2.2 VAE38 encoder / decoder

实际 checkpoint 匹配 `WanVideoVAE38`，不是旧版 16-channel、8× spatial compression 的 Wan VAE。其核心配置为：

| 配置 | 值 |
|---|---:|
| latent channels | 48 |
| encoder base dim | 160 |
| decoder base dim | 256 |
| channel multipliers | `[1,2,4,4]` |
| residual blocks / encoder stage | 2 |
| residual blocks / decoder stage | 3 |
| spatial compression | 16× |
| temporal compression | 4×，causal |
| VAE 参数 | `704,688,668`，全部冻结 |

### 4.1 VAE encoder

```text
RGB [B,3,T,H,W]
  │ patchify spatial 2×2（纯 reshape）
  ▼
[B,12,T,H/2,W/2]
  │ CausalConv3d 12→160, k=3
  ▼
Stage E0: 2×ResidualBlock 160→160 + spatial downsample ×2
Stage E1: 2×ResidualBlock 160→320 + spatial ×2, temporal ×2
Stage E2: 2×ResidualBlock 320→640 + spatial ×2, temporal ×2
Stage E3: 2×ResidualBlock 640→640, no downsample
  │
  ├─ middle: ResidualBlock(640) → single-head spatial AttentionBlock(640)
  │          → ResidualBlock(640)
  │
  └─ RMSNorm → SiLU → CausalConv3d 640→96
       → outer 1×1×1 CausalConv3d 96→96
       → split 48-d μ / 48-d log-variance
       → 实际 encode() 返回归一化后的 μ，不采样
```

每个 VAE `ResidualBlock` 是：

```text
RMSNorm → SiLU → CausalConv3d(k=3)
        → RMSNorm → SiLU → Dropout(0) → CausalConv3d(k=3)
        + identity / 1×1×1 shortcut
```

对 `41×512×512` 视频：

```text
[B,3,41,512,512] → [B,48,11,32,32]
T_l = 1 + (T-1)/4 = 11
H_l = H/16 = 32
W_l = W/16 = 32
```

VAE encoder 参数为 `149,627,776`；posterior 1×1 projection 为 `9,312`。

### 4.2 VAE decoder

```text
latent [B,48,T_l,H_l,W_l]
  │ inverse normalization
  │ outer 1×1×1 CausalConv3d 48→48
  │ CausalConv3d 48→1024, k=3
  ▼
middle: ResidualBlock(1024) → single-head spatial AttentionBlock(1024)
        → ResidualBlock(1024)
  ▼
Stage D0: 3×ResidualBlock 1024→1024 + spatial ×2, temporal ×2
Stage D1: 3×ResidualBlock 1024→1024 + spatial ×2, temporal ×2
Stage D2: 3×ResidualBlock 1024→512  + spatial ×2
Stage D3: 3×ResidualBlock 512→256, no upsample
  ▼
RMSNorm → SiLU → CausalConv3d 256→12
  ▼
unpatchify spatial 2×2 → RGB [B,3,T,H,W]
```

VAE decoder 参数为 `555,049,228`；decoder 前的外部 1×1 projection 为 `2,352`。整个 VAE 用 causal temporal convolution 和 feature cache 做顺序编码/解码。代码对单 latent-frame segment 解码时会临时复制为两个 latent frames，再只保留第一帧，以绕过 VAE38 单 latent 输入会被裁成零长度的问题。

## 5. 冻结的 text encoder

语言编码器是 UMT5-XXL 风格的 encoder-only T5：

| 配置 | 值 |
|---|---:|
| vocabulary | 256,384 |
| max text length | 512 |
| hidden / attention dim | 4,096 |
| FFN dim | 10,240 |
| heads | 64，head dim 64 |
| encoder layers | 24 |
| relative-position buckets | 32 |
| 参数量 | `5,680,910,336`，全部冻结 |

每层为：

```text
T5 RMSNorm
  → 64-head self-attention（Q/K/V/O 均无 bias，不使用 1/sqrt(d) scaling）
  → residual
T5 RMSNorm
  → gated-GELU FFN:
       (Linear 4096→10240 → GELU) ⊙ Linear 4096→10240
       → Linear 10240→4096
  → residual
```

每个 T5 block `192,948,224` 参数，24 层共 `4,630,757,376`；token embedding 单独占 `1,050,148,864`。prompt 输出 `[B,512,4096]`，有效文本结束后的 embedding 被置零，再由 DiT 的共享 `text_embedding` 投到 3072 维。

本模型配置 `has_image_input=False`、`require_clip_embedding=False`，因此实际管线没有加载或使用 CLIP image encoder。图像条件来自 VAE latent canvas 中保持干净的位置，不来自 CLIP image tokens。

## 6. Conditioning canvas 与 mask

令单个完整 segment 的 VAE latent 为 `[B,48,11,32,32]`。四 segment 常规 canvas 为：

```text
[m(v0) | action(v0) | m(v1) | action(v1)]
        11 + 11 + 11 + 11 = 44 latent frames
```

mask 和 latent 相同形状，`True` 表示 clean condition，`False` 表示需要生成。三个基础角色是：

- `F`：segment 所有 latent frames 都 clean；
- `I`：只给 segment 的第一个 latent frame，后面预测；
- `-`：segment 没有任何 clean frame；fusion experiments 使用；
- `1`：物理上把视觉 segment 截成一个 latent frame，且该帧是 clean；policy mode 使用。

论文主设置在 action 未被 drop 时使用 A1：

| plan | 条件概率 | canvas latent frames | 含义 |
|---|---:|---:|---|
| `IIII` | 0.75 | 44 | 四个 segment 都只给首 latent frame；共同生成未来感知与 action |
| `FIII` | 0.00 | 44 | 仅第 0 个 segment 全给定 |
| `FIFI` | 0.05 | 44 | 两个视角的视觉/感知 segment 全给定，action 只给首帧；inverse dynamics / v2a |
| `1I1I` policy | 0.20 | 24 | 两个视觉 segment 各缩成 1 帧，两个 action segment 各 11 帧 |

在选 plan 之前，action 另有 `0.1` 概率被 drop，样本变成双视角的单视觉模态 future generation。故相对最初 template draw 的总体概率为 10% action-free，67.5% IIII，4.5% FIFI，18% policy。

一个容易误解的实现事实是：**mask 没有作为额外 channel 或 mask embedding 输入 DiT。** 模型只通过“哪些位置是 clean latent，哪些位置被加噪”来辨认条件位置；`in_dim=48` 恰好就是 VAE latent channels，没有额外 mask channels。

## 7. 相机 encoder：Plücker 条件如何进入每个 DiT block

### 7.1 Plücker field

每个像素射线由 6D Plücker coordinate 表示：

```text
d = normalize(R_camera_to_world · K^{-1}[u,v,1])
m = camera_origin × d
plücker = [d, m]
```

实现将 `d` 和 `m/3` 分开当作两个三通道视频，分别通过同一个冻结 VAE encoder：

```text
direction: [B,3,41,512,512] → [B,48,11,32,32]
moment/3 : [B,3,41,512,512] → [B,48,11,32,32]
concat channels → [B,96,11,32,32]
permute         → [B,11,96,32,32] per view
```

相机 latent 按与 data latent 完全相同的 segment plan 重复和拼接。四 segment canvas 上是 `[B,44,96,32,32]`。

### 7.2 每个 block 都有独立 camera encoder

每个 block 的 camera module 是：

```text
AdaptiveAvgPool3d(output=(48,16,16))
Flatten(start_dim=2)
Linear(48×16×16=12,288 → 3,072)
```

注意输入 `[B,F,96,32,32]` 在 PyTorch `AdaptiveAvgPool3d` 看来是 `[N,C,D,H,W]`：packed latent-frame 数 `F` 被当作 channel 轴保留，`96×32×32` 被池化为 `48×16×16`。结果是 `[B,F,3072]`；随后在 DiT patch grid 的 `16×16` 空间复制成 `[B,F×16×16,3072]`，与 video tokens 一一对齐。

camera linear 在从纯 Wan base 初始化时为全零，所以初始时相机注入为零；`projector: Linear(3072→3072)` 初始化为 identity。官方 Action Images warm-start checkpoint 已经包含训练过的这两部分权重。

论文 appendix 说 camera tensor 与 latent sequence “concatenated”，从代码角度更精确的说法是：**camera latent 先按相同 segment 顺序对齐，然后在每个 DiT block 内投影并加到 self-attention 的输入 token 上；它不是拼到 DiT 输入 channel，也不是额外 camera tokens。**

## 8. DiT 总配置

| 参数 | 值 |
|---|---:|
| backbone | Wan2.2-TI2V-5B |
| `in_dim` / `out_dim` | 48 / 48 |
| model width `D` | 3,072 |
| FFN width | 14,336 |
| blocks | 30 |
| attention heads | 24 |
| head dim | 128 |
| patch size | `(1,2,2)` |
| timestep Fourier dim | 256 |
| text input dim | 4,096 |
| normalization epsilon | `1e-6` |
| self-attention position encoding | 3D RoPE |
| self-attention mask | 无；全局双向 attention |
| cross-attention context | UMT5 tokens |

### 8.1 输入投影

四 segment、512 分辨率时：

```text
x: [B,48,44,32,32]
Conv3d(48→3072, kernel=(1,2,2), stride=(1,2,2))
→ [B,3072,44,16,16]
→ flatten [B,11,264,3072]
```

policy canvas 是 `24×16×16 = 6,144` tokens。一个双视角、单模态、完整 22-latent-frame canvas 是 `5,632` tokens。五模态 fusion 的 110-frame canvas 则是 `28,160` tokens。

### 8.2 timestep conditioning

```text
timestep
  → 256-d sinusoidal embedding
  → Linear 256→3072 → SiLU → Linear 3072→3072 = t
  → SiLU → Linear 3072→(6×3072)
  → [shift_msa, scale_msa, gate_msa,
     shift_mlp, scale_mlp, gate_mlp]
```

每个 block 还有自己的可学习 `modulation[1,6,3072]`，与全局 timestep modulation 相加。即 AdaLN 参数既有 sample-dependent 的时间项，也有 layer-specific 的可学习基线。

### 8.3 3D RoPE

每个 attention head 128 维被分为：

```text
temporal 44 dims + height 42 dims + width 42 dims = 128
```

RoPE 坐标是在 **拼接后的整个 `(F,H,W)` grid** 上建立。每个 segment 不会重置 temporal position；后面的视角和模态继续使用更大的 packed-time index。这也是为何 segment packing order 是模型定义的一部分。

## 9. 一个 DiT block 的完整结构

30 层均为下面的结构，架构相同但参数不共享：

```text
输入 x ∈ R[B,S,3072]

1) Self-attention branch
   u = LayerNorm_no_affine(x)
   u = u * (1 + scale_msa) + shift_msa
   c = CameraEncoder_block(C)             # [B,S,3072]
   u = u + c

   q = RMSNorm(Linear_q(u))
   k = RMSNorm(Linear_k(u))
   v = Linear_v(u)
   q,k = 3D_RoPE(q,k)
   a = MultiHeadAttention(q,k,v), 24 heads
   a = Linear_o(a)

   x = x + gate_msa * Projector_block(a)

2) Text cross-attention branch
   u = LayerNorm_affine(x)
   q = RMSNorm(Linear_q(u))
   k = RMSNorm(Linear_k(text_context))
   v = Linear_v(text_context)
   a = MultiHeadAttention(q,k,v), 24 heads
   x = x + Linear_o(a)

3) Feed-forward branch
   u = LayerNorm_no_affine(x)
   u = u * (1 + scale_mlp) + shift_mlp
   f = Linear(3072→14336) → GELU(tanh approximation)
       → Linear(14336→3072)
   x = x + gate_mlp * f
```

实现会依次尝试 FlashAttention 3、FlashAttention 2、SageAttention，最后 fallback 到 PyTorch scaled-dot-product attention。Q/K 使用 RMSNorm，V 不归一化。cross-attention 没有 AdaLN gate；它直接 residual-add 到主支路。

### 9.1 单 block 精确参数量

| 子模块 | 参数量 |
|---|---:|
| self-attention（Q/K/V/O + Q/K RMSNorm） | `37,767,168` |
| cross-attention（Q/K/V/O + Q/K RMSNorm） | `37,767,168` |
| affine `norm3` | `6,144` |
| FFN `3072→14336→3072` | `88,097,792` |
| block modulation `[1,6,3072]` | `18,432` |
| 原始 Wan block 小计 | `163,656,704` |
| camera linear `12288→3072` | `37,751,808` |
| post-self-attention projector `3072→3072` | `9,440,256` |
| **扩展后单 block 总计** | **`210,848,768`** |

30 blocks 合计 `6,325,463,040` 参数。

## 10. 共享输出 head 与“decoder”的四种含义

### 10.1 DiT 输出 head

DiT 后没有 depth/action/segmentation 分支。唯一的 head 是：

```text
AdaLN using t and head.modulation[1,2,3072]
Linear 3072 → 48×1×2×2 = 192
unpatchify → [B,48,F,32,32]
```

head 共 `596,160` 参数。它预测的是 latent-space flow/velocity，而不是 RGB、depth label 或连续 action。

### 10.2 VAE decoder

这是唯一学习型像素 decoder，所有模态共享，且训练时冻结。它把每个 48-channel latent segment 还原为三通道视频。

### 10.3 确定性 modality decoder

- RGB：identity。
- depth：将生成 RGB 投影到最近的合法 RGB-cube path edge，再反变换为 metric depth；离 path 太远的像素标为 invalid。
- normal：`2*RGB-1` 后单位化。
- segmentation：映射到最近的固定 role color / label。

这些 decoder 无参数，不参加反向传播。

### 10.4 action geometric decoder

生成的两个视角 action RGB 分别给出位置点和两个轴端点。系统先从 RGB heatmap 中恢复三个二维峰，再用已知 intrinsics/extrinsics 多视角融合成三个 3D 点，构造正交旋转矩阵：

```text
e_x  = normalize(q_up - q_pos)
e_z  = normalize(q_pos - q_normal)
e_y  = normalize(e_z × e_x)
e_z' = e_x × e_y
R    = [e_x | e_y | e_z']
```

蓝色背景 pedestal 解码 gripper openness。最终输出 `[x,y,z,qx,qy,qz,qw,openness]`，没有 learned action MLP head。

## 11. 参数量完整审计

### 11.1 扩展 DiT

| DiT 部分 | 参数量 |
|---|---:|
| patch embedding | `592,896` |
| text projection MLP | `22,026,240` |
| timestep embedding MLP | `10,229,760` |
| timestep 6-way projection | `56,641,536` |
| 30 个扩展 DiT blocks | `6,325,463,040` |
| shared output head | `596,160` |
| **扩展 DiT 总计** | **`6,415,549,632`** |

拆成来源看：

| 来源 | 参数量 |
|---|---:|
| 原始 Wan2.2-TI2V-5B checkpoint | `4,999,787,712` |
| 30× camera encoder | `1,132,554,240` |
| 30× projector | `283,207,680` |
| camera additions 总计 | `1,415,761,920` |
| **最终 denoiser** | **`6,415,549,632`** |

### 11.2 完整 pipeline

| 组件 | 参数量 | 主实验是否训练 |
|---|---:|---:|
| UMT5 text encoder | `5,680,910,336` | 否 |
| Wan2.2 VAE38 | `704,688,668` | 否 |
| camera-augmented DiT | `6,415,549,632` | 是，全部参数 |
| **总计** | **`12,801,148,636`** | **训练 6.416B** |

### 11.3 checkpoint 实物验证

这些数字不是按论文名称估算，而是对本地权重 tensor shape 求和：

| 权重 | tensors | dtype | numel | 文件大小 |
|---|---:|---|---:|---:|
| Wan 原始 DiT safetensors | 825 | FP32 | `4,999,787,712` | 约 20.0 GB，3 shards |
| official Action Images `step125750.ckpt` | 945 | BF16 | `6,415,549,632` | 12.831 GB |
| arm7 `step4000.ckpt` | 945 | BF16 | `6,415,549,632` | 12.831 GB |
| Wan2.2 VAE38 | 196 | FP32 | `704,688,668` | 2.819 GB |
| UMT5 encoder | 242 | BF16 | `5,680,910,336` | 11.362 GB |

扩展 DiT 比原始 checkpoint 多 120 个 tensors，恰好是每层 `cam_encoder.weight/bias + projector.weight/bias` 四个 tensors，乘 30 层。

## 12. 训练目标与优化参数

### 12.1 Flow matching

训练 scheduler 有 1,000 个离散 timestep，shift 为 5。原始线性噪声比例 `σ_hat` 变换为：

```text
σ = 5 σ_hat / (1 + 4 σ_hat)
x_σ = (1-σ) z + σ ε
target = ε - z
```

clean mask 上把 `x_σ` 恢复为原始 `z`；loss 只在 prediction mask 上计算：

```text
L = w(σ) * sum((v_theta(x_σ)- (ε-z))² * prediction_mask)
             / sum(prediction_mask)
```

每个 batch 只抽一个 timestep，batch 内共享。`w(σ)` 是 scheduler 的 bell-shaped timestep weighting，经减去最小值并归一到均值 1。

### 12.2 实际 arm7 joint checkpoint 的训练配置

本地 `outputs/arm7_joint2src_seed42_fi3_6k` 的 W&B config 和 checkpoint 表明：

| 项 | 值 |
|---|---|
| warm start | official Action Images step 125,750 |
| data mixture | RLBench selfgen 0.7 + ManiSkill3 0.3 |
| template mixture | 0.4 RGB+action；depth/seg/normal+action 各 0.2 |
| segmentation | scene roles |
| frames / resolution | 41 / 512×512 |
| frame interval | 3 |
| GPUs | 4 × RTX 6000 Ada |
| micro batch | 1 / GPU |
| effective batch | 4，无 gradient accumulation |
| max steps | 6,000 |
| optimizer | DeepSpeed AdamW，ZeRO-2，optimizer CPU offload |
| learning rate | `5e-7` |
| warmup | 1,000 steps |
| schedule | constant after warmup |
| Adam β / ε / weight decay | `(0.9,0.999)` / `1e-8` / `0` |
| gradient clip | 1.0 |
| compute precision | bf16 mixed precision |
| checkpoint dtype | bf16 |
| memory | gradient checkpointing |
| seed | 42 |
| text CFG dropout | 5% prompt 置空 |
| action dropout | 10% |

`--full_param True` 使整个扩展 DiT 可训练，包括 patch/text/time projections、全部 attention/FFN、全部 camera modules 和输出 head。VAE 与 T5 由 `pipe.requires_grad_(False)` 冻结。代码中也存在部分微调模式；若 `full_param=False`，只解冻名称包含 `cam_encoder`、`projector`、`self_attn`、`text_embedding` 的模块，但这不是论文主实验设置。

训练对象先从 bf16 base/warm-start 权重加载，再把可训练参数转换到 FP32 交给 DeepSpeed mixed-precision 管理；forward/backward 使用 bf16 配置，最终单独保存的 DiT checkpoint 是 bf16。

## 13. 推理路径

1. 根据 template 分别编码每个 `(modality, view)` stream。
2. action template 中，把候选/占位 7-DoF action 投影为 action video 并编码。
3. 用 deterministic conditioning plan 构造 packed latent、camera latent 与 clean mask。
4. 未给定位置初始化为 Gaussian noise；给定位置保留 clean VAE latent。
5. UMT5 分别编码 positive/negative prompt；CFG 合并两次 DiT 输出。论文主评估使用 50 denoising steps、CFG 7.5、sigma shift 5。
6. 每一步 scheduler 只更新 `~mask`；clean latent 在整个采样过程中保持不变。
7. 按 segment span 切开最终 canvas，各自用同一个 VAE decoder 解码。
8. 感知 stream 用确定性 inverse codec；action stream 用多视角几何 decoder 得到可执行 pose/gripper。

推理支持 Unified Sequence Parallel：token sequence 沿 sequence 维切到多个 rank，camera embedding 在 block 内做相同切分，head 后 all-gather。它改变并行方式，不改变模型参数或数学结构。

## 14. 几个必须写清楚的架构判断

### 14.1 它是共享生成器，不是多头模型

depth、normal、segmentation、action 没有各自的 neural decoder/head，也没有各自的 loss。它们的差异只来自 deterministic codec、prompt tag、segment position 和训练数据内容。

### 14.2 DiT 不是 encoder-decoder Transformer

主干是单栈 30-layer diffusion transformer：每层 self-attention + text cross-attention + FFN。这里的“encoder / decoder”应分别指 UMT5 encoder 和 Video VAE encoder/decoder；DiT 自身没有 U-Net 式 down/up path，也没有 Transformer decoder 的 causal autoregressive mask。

### 14.3 VAE 是 causal，但 DiT attention 不是 causal

VAE 沿时间使用 causal 3D convolution；DiT 对 packed latent tokens 做全局双向 self-attention。因此不同 segment、不同视角、不同未来位置可以在同一次 denoising forward 中互相通信。

### 14.4 主模型是跨 step 的多任务共享，不是一次 forward 的全模态 fusion

arm7 每步从四个 `visual/perception + action` template 中抽一个。所谓 cross-task generation，核心是这些任务在训练步之间共享全部 6.416B DiT 参数。代码有真正的五模态 fusion canvas，但它是另一类实验设置。

### 14.5 “无 modality-specific 参数”成立，但并非“没有新增参数”

相对于 Wan base，模型新增了 1.416B camera-conditioning 参数；这些参数对所有模态共享，并不是 depth/seg/action 专用，所以论文关于“无 modality-specific head”的主张成立。但若描述总模型大小，不能只写 5B。

### 14.6 没有显式 condition mask 输入

模型没有 mask token、mask channel 或 separate condition encoder。clean/noisy latent 的数值差异就是条件机制。这使 training/inference 的 conditioning-plan 分布是否匹配尤其重要。

### 14.7 当前代码与生成 checkpoint 时的代码

arm7 joint checkpoint 的 W&B metadata 指向 commit `9d2a702`。从该 commit 到当前 `a98192b`，核心模型文件 `training/models/wan_video_dit.py`、`training/wan_video_action_images.py` 和 `training/templates.py` 没有架构差异；变化集中在 launcher、训练参数与后续数据/评估支持。实际 checkpoint shape 也与当前类定义严格一致。

## 15. 代码与论文依据索引

- DiT、attention、block、head、Wan2.2-TI2V-5B config：[`training/models/wan_video_dit.py`](training/models/wan_video_dit.py)
- camera module 注入、冻结/解冻策略、训练 forward 和 masked loss：[`train.py`](train.py)
- VAE encode/decode、inference canvas、denoising loop：[`training/wan_video_action_images.py`](training/wan_video_action_images.py)
- template、conditioning plan、segment packing：[`training/templates.py`](training/templates.py)
- action image、Plücker rays 与 action decoder：[`training/utils.py`](training/utils.py)
- 两视角抽样与相机相对位姿：[`training/dataset/base.py`](training/dataset/base.py)
- depth/normal/segmentation codec 接入：[`training/dataset/rlbench_selfgen.py`](training/dataset/rlbench_selfgen.py)
- 主训练 launcher：[`scripts/train_arm.sh`](scripts/train_arm.sh)
- 两数据源 arm0/arm7 实际 launch：[`scripts/launch_joint_arms.sh`](scripts/launch_joint_arms.sh)
- DeepSpeed 配置：[`configs/zero2_offload.json`](configs/zero2_offload.json)
- 论文方法：[`../6a89bc13108d28a2448f410a/sec/3_method.tex`](../6a89bc13108d28a2448f410a/sec/3_method.tex)
- 论文 architecture、optimization、camera appendix：[`../6a89bc13108d28a2448f410a/sec/appendix.tex`](../6a89bc13108d28a2448f410a/sec/appendix.tex)
- VAE/T5 的具体类定义来自当前训练环境中的 `diffsynth`：
  - `/workspace/1228_tingting/envs/ttd_train/lib/python3.10/site-packages/diffsynth/models/wan_video_vae.py`
  - `/workspace/1228_tingting/envs/ttd_train/lib/python3.10/site-packages/diffsynth/models/wan_video_text_encoder.py`
  - `/workspace/1228_tingting/envs/ttd_train/lib/python3.10/site-packages/diffsynth/schedulers/flow_match.py`
