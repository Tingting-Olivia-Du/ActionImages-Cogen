# TESTING —— 感知/动作管线的测试与逐环节可视化

> 日期 2026-08-09，2026-08-10 部分更新。配套 `CHANGES_TEMPLATES.md`（当前设计改了什么）与
> `FORK_CHANGES.md`（fork vs 上游的历史全量）。本文档回答：**怎么验证它是对的**，
> 以及**每个中间环节实际长什么样、数值是多少**。
>
> ⚠️ **2026-08-10**：顶替式设计已升级为任务模板（`--template_mix`）。下表里
> `test_forward_unchanged.py` 与 `test_prompt_tags.py` 两行描述的是旧版断言，
> 现行断言见 `CHANGES_TEMPLATES.md` §2.6 与 §5；其余各行仍然准确。
> 命令里的 `--visual_modality_mix X@1.0` 请读作 `--template_mix video+X@1.0`（感知）
> 或 `X+action@1.0`（顶替式世界模型）。
>
> 两套东西：
> - `tests/`（7 个文件，纯 CPU，~1 分钟）—— 断言式，进 CI。
> - `debug/visualize_pipeline.py` —— 逐环节出图，回答断言问不出来的问题。

```bash
cd /workspace/ttdu/ActionImages-Cogen && export PYTHONPATH=$PWD
conda activate ttd_train

bash scripts/run_tests.sh                                  # 断言套件
CUDA_VISIBLE_DEVICES=<空卡> python debug/visualize_pipeline.py   # 逐环节出图 -> debug/out/
python debug/visualize_pipeline.py --no-vae                # 只跑 CPU 环节
```

---

## 第一部分：断言套件 `tests/`

当前**7/7 全绿**。

| 文件 | 守住的不变量 | 实测 |
|---|---|---|
| `test_forward_unchanged.py` | 训练核心与上游 `291f71c` **源码逐字相同**；4 段拼接顺序与 4 个 mask 偏移是字面量校验；`forward` 里不得出现 `visual_modality`/`depth`/`perception` | 3 项全过 |
| `test_depth_codec.py` | Barron 变换双射、RGB-cube 路径双射与哈密顿性、量化往返、非法值处理 | 真实 episode 往返 **AbsRel 0.067%**（门槛 2%） |
| `test_seg_codec.py` | 多实例编解码、邻接、噪声鲁棒、referring spec 的颜色词消解与部分丢弃上报 | 真实 episode **IoU 1.0000** |
| `test_prompt_tags.py` | `<depth>` 活过 `train.py:225` 的 `.replace(".","")`/`.replace("_"," ")`；UMT5 不炸成一堆 subword、不撞 `<unk>`；episode 路径含 `"rlbench"` | `<depth>` **+3 token**，`<seg: ...>` +8 token，往返一致 |
| `test_selfgen_dataset.py` | **depth 样本的 `action_7d.abs().sum() > 0`**；两半都必须是 depth；混合 mix 能 stack；非法 mix 构造期即拒 | `action_7d.abs().sum() = 212.1` |
| `test_selfgen_alignment.py` | 从磁盘独立重算 camera/extrinsics/intrinsics 与样本比对，两模态×两视角 | 全过；**负控制**：交换 cond/target 被检出 |
| `test_depth_roundtrip_e2e.py` | `sample["video"]` 两半解码回米制 vs `depth.npz` | **AbsRel 0.089%**；**负控制**跨视角 **37.56%** |

### 两条断言值得单独说

**(1) `action_7d.abs().sum() > 0` on a depth sample。** ttd 的对应测试断言的正好相反。
那一行不是风格差异，是整个科学改动：ttd 里感知样本清零动作 ⇒ 感知监督与动作监督**永远不会**
出现在同一次梯度更新里 ⇒ 主 RQ（感知多任务是否干涉动作能力）只能靠"交替训练几百步后 r_peak
是否下降"间接推断。删掉清零，co-supervision 第一次可以被直接观测。

**(2) 每个负控制。** 两个 alignment/e2e 测试都带负控制，因为"通过"本身不构成证据——测试必须
被证明**有能力失败**。跨视角 37.56% vs 同视角 0.089%，相差 420 倍，说明比对是有判别力的。

---

## 第二部分：逐环节可视化 `debug/out/`

一个真实样本：`open_drawer/variation0/episodes/episode0`，视角 (view4, view3)，帧窗 [32, 72]。

| 图 | 环节 | 看什么 |
|---|---|---|
| `S1_gt_depth.png` | 磁盘上的 GT 深度（米） | 两行=两个视角，即 `<depth><action>` 要填的两段 |
| `S2_S3_encode_and_tensor.png` | encode → RGB → dataset 张量 | **第2行必须与第3行完全相同**；**第3行两半都必须不同于第1行** |
| `S4_codec_only_error.png` | 无 VAE 往返误差 | 地板值 |
| `S5_vae_roundtrip_depth.png` | **encode → VAE → decode** | 本次最重要的一张 |
| `S6_vae_roundtrip_rgb_control.png` | 同一 VAE 对自然 RGB | S5 的对照 |
| `S7_action_images.png` | 动作流（7d→5d→RGB） | 同一条 3D 轨迹按各视角相机重投影 |
| `S8_assembled_sequence.png` | 拼好的 4 段 + 条件帧 | 模型真正看到的东西 |
| `S9_latent_distribution.png` | latent 分布 depth vs RGB | 是否偏离预训练流形 |
| `S10_segment_latent_scales.png` | 四段各自的 latent scale | S9 的上下文 |

### 已验证的等式（S2/S3）

```
dataset 张量 vs 直接 encode_depth(GT)：max |diff| = 0 / 0（cond / target）
```
张量**就是** codec 输出，中间没有任何隐式的 resize / 归一化 / 通道重排。

### S8：布局本身

四段沿**时间轴**拼成一条长视频，一次 forward，一个 masked MSE：

```
段1 v1_depth  | 段2 v1_action | 段3 v2_depth  | 段4 v2_action
 ↑首帧干净      ↑首帧干净       ↑首帧干净       ↑首帧干净
```

图上可见：段1/段3 是两个**不同视角**的深度；段2/段4 是**同一条轨迹**按各自相机重投影的热斑
（形状不同、运动一致）。每段首帧绿框标出——模态身份就靠这一帧"自报家门"，不需要任何 head。

---

## 第三部分：本次 debug 的三个实质发现

### 发现 1（解决一个悬置的风险）：深度编码**能**活过 Wan VAE

`plan/core/9-depth-seg-codec-vs-genception.md` §8.2 把这条列为**正式采用前必须补做**的测试，
理由是"人工颜色曲线可能是自然视频 VAE 的分布外输入"（§3）。本次测了：

| | cond 半 (view4) | target 半 (view3) |
|---|---|---|
| AbsRel 过 VAE 前 | 0.0943% | 0.0948% |
| AbsRel 过 VAE 后 | **0.583%** | **0.602%** |
| 可解码像素占比 | 100.00% → **100.00%** | 100.00% → **100.00%** |

- 误差劣化 **6 倍**，但仍在 2% 门槛的 **1/3.4** 处。
- **没有任何像素被推离颜色路径**（`decode_depth` 的 NaN 占比 0）。这是原先最担心的失效模式。
- 像素级 L1：**depth 0.0072 < 自然 RGB 0.0325**。深度编码对 VAE 来说比自然视频**更容易**，
  不是更难——它分片光滑、几乎没有纹理。

S5 第 3/4 行显示误差**全部集中在深度不连续处（物体轮廓）**，物体内部近乎无损。这正是 VAE
糊边的典型行为，对本任务是良性的。

**归因纪律**：这是 VAE 重建下界，不是模型生成误差。DiT 的生成误差会远大于 0.58%。这条结论
只排除了"codec 过不了 VAE"这一种死法。

### 发现 2（一个看起来吓人、实际不是问题的数字）：depth latent 更宽，但在先验已处理的范围内

S9 单看很吓人：depth latent std **1.79** vs 自然 RGB **0.86**，2.1 倍，均值也偏移。第一反应是
"depth 段落在 DiT 预训练流形之外，warm-start 会打架"。

S10 把它放回上下文——**官方布局本来就是异质的**：

| 段 | latent std |
|---|---|
| v1_rgb / v2_rgb | 0.838 / 0.879 |
| **v1_action / v2_action** | **1.535 / 1.535** |
| v1_depth / v2_depth | 1.822 / 1.766 |

官方 `<video><action>` 自己就横跨 0.84 → 1.54（1.8 倍）。depth 的 1.79 只比 action 段高
1.16 倍，而 action 段是预训练 checkpoint **已经处理了 125,750 步**的东西。

**结论：depth 的 latent scale 在先验已处理的范围内，不是混淆因素。** 需要注意的是它仍略高于
两个臂共有的 action 段，所以 depth 臂起步时 loss 可能略高、收敛稍慢——那是预期，不是 bug。

### 发现 3（一个真 bug，被可视化抓到）：VAE 必须**逐段**编码

第一版可视化把 `[C, 2T, H, W]` 整条 82 帧丢进 VAE，得到 81 帧返回：Wan 的时间压缩是
`T_latent = (T-1)/4 + 1`，82 帧不整除。

但 `forward` **从来不这样做**——它对 `video[:, :, :T]` 和 `video[:, :, T:]` **分别**调用
`encode_video`（train.py:236-237），每段 41 帧 → 11 latent 帧，整除。把整条编码不只是形状错，
更会让两个视角在 VAE 内部**共享时间上下文**，那不是训练时发生的事。

已修，并在 `vae_roundtrip` 里加了形状断言。**这类错误任何断言测试都抓不到**，因为它只存在于
分析代码里；只有把中间产物画出来、发现帧数对不上才会暴露。

---

## 第四部分：还没做的验证（需要 GPU / 更长时间）

按重要性排序：

1. **L2 数值回归**：官方 HF 数据 + `--learning_rate 0` + 20 步，在 pristine
   `/workspace/ttdu/ActionImages` 与本仓库各跑一次，断言 `max|Δloss| < 1e-5`。
   L1（源码逐字）已经过了，这条是端到端的补充确认。
2. **冒烟跑**：`GPUS=<空卡> bash scripts/smoke_arm.sh depth`，再跑 `video`。判据：
   checkpoint `strict=True` 加载成功、30 步 loss 有限、量级同阶、**峰值显存一致**
   （4 段形状相同，显存本应先验相等；差距大说明有东西改了形状）。
3. **eval 脚本移植**：`g0_openloop_rlbench_v2.py`（r_peak）与 `g0_perception_zeroshot.py`
   （AbsRel/IoU）。移植时要修 ttd 版 `g0_perception_zeroshot.py:126` 的真 bug——它用
   `sorted(glob(...))` 重新推导视角，而 camera 张量来自 dataset 的**未排序** glob 顺序
   （实测某 episode 返回 `['view4','view2','view3','view1']`），两者不一致就是 P0-1 那类错位。
   本 fork 的 provenance 字段让它无法再发生。
4. **seg 臂**：codec 与 dataset 分支都已写好且单测通过，但从未端到端跑过。启用只需
   `--visual_modality_mix segmentation@1.0`。跑之前应先补一个 seg 版的
   `test_depth_roundtrip_e2e`（解码回 mask 比 IoU），以及 seg 的 VAE 往返——**seg 比 depth
   更值得担心**：它是离散色块，VAE 糊边会直接把边界像素判成错误类别，而 depth 的糊边只是
   数值误差。
5. **多 episode 统计**：现在所有数字来自 1 个 episode。S4/S5 应该扫 20-50 个 episode 出分布，
   确认 0.58% 不是这一集特有的。

---

## 附：观察到但**没有**处理的事

- **codec 的动态范围利用率低**。RLBench 场景深度约 0.9–4.2 m，而 codec 的有效区间是
  0.05–10 m 且经 Barron 变换。S1/S2 可见整个场景只落在颜色路径的一小段弧上（几乎全是
  红/橙/黄）。这意味着有效颜色分辨率远低于 codec 的理论上限，VAE 的任何扰动都会被放大成
  更大的深度误差。当前 0.58% 已经够用，所以没动——但如果将来 depth 精度成为瓶颈，
  **重新标定 `MIN_VALID`/`MAX_VALID` 到 RLBench 实际范围是第一个该试的杠杆**，比改模型便宜得多。
- **`BaseDataset.__getitem__` 的 50 次重试**已收紧为连续 5 次即 raise，但**没有**完全去掉。
  真·瞬时 IO 错误仍然应该被容忍。
- **`train.py:265` 的 `"rlbench" in inputs["path"][0]` 门**依赖数据软链的名字。已加
  `test_prompt_tags.py` 里的断言守住，但这本身是上游的一处脆弱设计，没有改动它。
