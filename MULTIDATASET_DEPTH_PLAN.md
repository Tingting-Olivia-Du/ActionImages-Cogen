# 三数据集 + depth 辅流：现状调查与修改方案

目标：`rlbench / droid / bridge` 三数据集混训，同时按比例混 `video+action` 与 `video+depth`，
并对 `policy(single-frame) / fifi / fiii / iiii` 四种 mask 模式做显式比例分配。

---

## 0. 决策纪要（已定，依据见后文）

| # | 决策 | 依据 |
|---|---|---|
| D1 | **走方案 A**：depth 只从 `rlbench_selfgen_512_aug` 抽 | droid 的 VGGT 深度实测不可标定（§2.0）；bridge 无米制锚点 |
| D2 | **不用官方 HF `rlbench`**，RLBench 侧只用 `rlbench_selfgen_512_aug` | 评测全在 selfgen 上；历史 arm 全是 selfgen；官方那 5 个任务永远评不到（§2.2） |
| D3 | 数据集比例锚定论文 Table 1：**selfgen .62 / droid .28 / bridge .10** | §4.1 |
| D4 | depth 占 selfgen 内部 **1/3** → 全局 `video+depth` ≈ 20.7% | §4.2 |
| D5 | 两条模板共用 mask 比例 **(iiii .81, fiii .045, fifi .045, policy/single .10)** | 用户指定；perception 侧零代码即可配置（§4.3） |
| D6 | 不做 `--action_mask_mix` | D5 的 action 侧数值就是现状硬编码值，改动 3 可延后 |

**一句话结论**：两条轴（模板混合、mask 混合）的机制本仓库已经有了，缺的是
(a) bridge/droid 数据没接进来、(b) 模板机制只对 `rlbench_selfgen*` 生效。
depth 本身没有任何官方基线可对照——它 100% 是本 fork 的新实验。

---

## 1. 现状调查

### 1.1 数据树的能力矩阵

| 树 | episodes | views | action | depth GT | 本方案是否使用 |
|---|---|---|---|---|---|
| `rlbench_selfgen_512_aug` | 1,068 → **788 实际参训**（`VARIATIONS=0` 后；1,028 个 4 视图 depth 齐全，其中 variation0 788 个） | 4 | ✅ | ✅ `view*/depth.npz` | ✅ **RLBench 侧唯一来源** |
| `droid` | 9,437 | 2 | ✅ | ❌ | ✅ |
| `bridge` | 17,709 | 3–5 | ❌（全零） | ❌ | ✅ 仅 video-only |
| `rlbench`（官方 HF） | 4,124 | 4 | ✅ | ❌ | ❌ **排除，见 §2.2** |
| `rlbench_selfgen` / `_512` | 0（软链悬空） | — | — | — | ❌ |

`data/` 下目前只有 `rlbench`、`rlbench_selfgen*` 四个软链，**没有 `bridge` / `droid`**。
好消息：`training/dataset/bridge.py` 和 `droid.py` 与 `ActionImages/` 下的**逐字节相同**，
所以 `/workspace/ttdu/ActionImages/data/{bridge,droid}` 里已建好的数据可以直接软链过来，
loader 一行都不用改。

### 1.2 已经有的机制（不需要重写）

* `training/templates.py`
  * `parse_template_mix("video+action@0.6,video+depth@0.4")` → 模板 + 累积概率
  * `draw_template()` 每个 sample 抽一个模板
  * `plan_segments()` 产出 `(modality, view, single_frame, fully_given)` 的段计划
  * `assemble()` 拼 latents / camera / masks，并断言"不能整条序列都是条件帧"
* `--template_mix`、`--perception_mask_mix`（含 M0/M1/M2 预设）、`--action_dropout_prob`、
  `--prompt_tag_style` 都已在 `training/args.py` 里
* `sample["streams"][modality]` + `sample["template"]` 的数据契约，collator 已转发
* depth 编解码：`training/percep/depth_codec.py`，metric depth → Barron 变换 → RGB cube 路径

### 1.3 四个缺口

**G1 数据没接** — `data/bridge`、`data/droid` 软链缺失。

**G2 模板机制只对 selfgen 生效** — `CombDataset.name_to_ctor`（`training/dataset/base.py:294`）
只把 `template_mix / prompt_tag_style / action_dropout_prob / ...` 传给 `rlbench_selfgen*`；
`rlbench / bridge / droid` 三个构造器不接这些参数。结果这三棵树的 sample 里没有 `template`
和 `streams`，`train.py:_batch_template` 走 fallback：

```python
if not names:
    return ("video", ACTION)   # datasets that predate templates (bridge/droid/rlbench)
```

也就是说**现在把 droid/bridge 加进 `--dataset_name` 也不会参与模板混合**，它们永远是
`video+action`，`--template_mix` 对它们完全无效。

**G3 action 模板的 mask 比例写死** — `plan_segments()` 里 perception 分支有可配的
`perception_mask_mix`，但 action 分支是硬编码常量：

```python
SINGLE_FRAME_VISUAL_PROB   = 0.1    # policy
MODE_FIRST_SEGMENT_GIVEN_PROB = 0.9 # p<0.90 -> iiii
MODE_ALL_VISUAL_GIVEN_PROB    = 0.95# 0.90..0.95 -> fiii;  >=0.95 -> fifi
```

实际边缘分布（rlbench，`is_rlbench=True`）：

| | policy | iiii | fiii | fifi |
|---|---|---|---|---|
| action 模板（现状） | 0.100 | 0.810 | 0.045 | 0.045 |
| perception 模板 M0（现状默认） | 0 | 0.10 | 0 | 0.90 |

注意两者**几乎是反的**：action 模板 81% 是 iiii，perception 模板 90% 是 fifi。
`templates.py` 自己的注释也指出了这个问题（M2 预设就是为了消掉它）。

**G4 `is_rlbench` 闸门** — `plan_segments(..., is_rlbench=True)` 决定 policy 模式是否生效。
non-rlbench 的树（droid/bridge）**永远抽不到 policy 模式**，但那个随机数照抽（保持与上游
RNG 对齐）。所以 droid 的 action 边缘分布是 `iiii 0.9 / fiii 0.05 / fifi 0.05`，没有 policy。

**G5 bridge 的零动作会静默降级** — `train.py:343`：

```python
if ACTION in modalities and torch.sum(action_7d**2) == 0:
    modalities = tuple(m for m in modalities if m != ACTION)
```

给 bridge 抽到 `video+action` 时会当场退化成 `video`。功能上没错，但**你请求的比例和实际
训练到的比例会对不上**，而且 prompt 里可能已经写了 `<action>` 标签（取决于 prompt 是在
dataset 里生成的还是 forward 里）。建议显式给 bridge 配 `video@1.0`，不要靠这条兜底。

---

### 1.4 官方论文怎么处理 depth：完全没有

查了 `ttd/papers/Action-Images.pdf`（25 页全文抽取）。**"depth" 全文只出现在第 7 页一次**，
而且不是模态，是讲**动作解码时的深度歧义**：从主视图的热力图往外投射一条光线，在近/远平面之间
采样候选 3D 点，再投影到侧视图打分选最优（式 9）。也就是说 depth 在官方方法里只是
"两视图三角化时被消解掉的那个自由度"，**没有任何 depth 监督、没有 depth 模态、没有 depth 头**。

> The main view provides the image-space anchor for ray casting, while the side view resolves
> the depth ambiguity by selecting the best match along the ray.

VGGT 在论文里出现 3 次，**全都不是 depth**：
1. **Bridge 的相机标注**——"BridgeV2 ... lacks camera labels and action-camera alignment.
   We estimate camera annotations with VGGT and use BridgeV2 for **video-only generation**."
2. 推理时给 in-the-wild 图片估计内外参
3. 定性图里用 VGGT 重建场景点云做可视化

顺带，论文这段也**逐条印证了我们已经做的数据处理**：

| 论文原话 | 我们的实现 |
|---|---|
| DROID "camera calibration is often noisy or incomplete in practice, so we filter out low-quality samples" | 只保留 `cam2base` 里两台外参都是 `source=="GT"` 的 9,437 集 |
| RLBench "we improve its visual diversity with Robot-Colosseum background augmentation" | 官方 HF release 即增强版；fork 侧另有 `rlbench_selfgen_512_aug` |
| Bridge "use BridgeV2 for **video-only** generation" | `bridge.py` 返回零动作 → 走 video-only 分支 |

**所以 depth 辅流 100% 是本 fork 的新增实验，没有官方基线可对照。** 这不是反对做它，
而是说它的收益必须自己用 arm 对比测出来，不能引用论文。

---

## 2. 核心设计问题：depth 只有 selfgen 有

### 2.0 实测：VGGT depth 能不能标定成 metric？答案是不能

思路是合理的：VGGT 的 depth 和它预测的相机在同一个（任意尺度的）坐标系里，而 DROID 的两台
外参相机之间的**真实米制基线**我们是知道的（来自 `cam2base`），所以

```
scale = ||C1_true - C2_true|| / ||C1_vggt - C2_vggt||
```

理论上就能把 VGGT depth 变成米。验证标尺也是现成的：夹爪的米制 3D 位置已知，
所以它在每台相机里的**真实深度**是精确已知的。

在 10 个 DROID episode、155 个夹爪采样点上实测（`scripts/probe_vggt_depth_droid.py`）：

```
baseline-derived scale : mean 0.7828  std 0.0327   (4.2% CV)
ideal per-sample scale : median 0.5840  IQR [0.493, 0.685]   CV 46%
centre-pixel err : median 22.4 cm
patch-p10   err : median 22.1 cm      (已排除"夹爪太细、差一个像素就打到桌面"的干扰)
```

两个结论：

1. **基线推出来的 scale 很稳**（跨 episode 只有 4.2% 变异），说明 VGGT 的内部归一化对这种
   双相机布局是一致的——这部分符合预期。
2. **但它和"理想 scale"对不上，而且理想 scale 本身散得一塌糊涂（CV 46%，IQR 0.49–0.69）。**
   这意味着 **不存在任何一个常数能把 VGGT depth 变成真实深度**——误差不是尺度问题，
   是深度场本身在这些位置就不准。中位绝对误差 22 cm，相对误差 44%。

   取 5×5 邻域的 p10（最近表面）也没改善（22.1 vs 22.4 cm），所以不是薄结构对不齐造成的。

> 诚实的边界：这个测量取的是**夹爪附近**的深度，那是 VGGT 最难的区域（细、反光、运动、常在画面边缘）。
> 大片平坦桌面的相对精度肯定好得多。但对操作策略来说，夹爪周围恰恰是唯一重要的区域；
> 而且"理想 scale 的 46% CV"这个指标本身与取哪个像素无关地说明了depth 场不是按常数比例缩放的。

`depth_codec.py` 编的是 metric depth（Barron 变换，硬编码 `MIN_VALID=0.05, MAX_VALID=10.0` 米）。
喂 22 cm 误差的伪 metric 进去，等于给 `<depth>` 这个 tag 教一个自相矛盾的 target。

### 2.05 唯一有 depth 的那棵树，任务和官方 rlbench 完全不重叠

| 树 | episodes | depth | 任务 |
|---|---|---|---|
| `rlbench`（官方 HF） | 4,124 | **0 个 depth 文件** | 5 个：close_box, close_laptop_lid, open_box, open_microwave, wipe_desk |
| `rlbench_selfgen_512_aug` | 1,068（**1,028** 个 4 视图 depth 齐全，100%） | ✅ `view*/depth.npz` | 16 个：close_jar, insert_onto_square_peg, light_bulb_in, … |

**两组任务的交集是空集。** 所以 depth 辅流不是"action 见过的场景的子集"，而是**另一批任务**。
如果评测在官方那 5 个任务上做，depth 信号从来没有和被评测的任务共现过——任何收益只能来自
"几何先验普遍有用"这种通用机制，而不是任务内的表征共享。这会显著削弱这个实验能得出的结论强度。

三个应对（建议 B）：

* **A. 就这么跑。** selfgen 内部自洽：同一批 1,028 个 episode 既出 `video+action` 也出
  `video+depth`，配对是干净的。结论范围限定为"在 selfgen 的 16 个任务内，depth 辅流是否帮助 action"。
* **B.（推荐）先用 selfgen 把官方那 5 个任务也渲一遍带 depth。**
  `ttd/scripts/gen_dataset.py --tasks close_box,close_laptop_lid,open_box,open_microwave,wipe_desk`
  ——任务名是参数不是硬编码，`CameraConfig(depth=True, depth_in_meters=True)`，直接产出
  `view*/depth.npz`（float16，米制）。这样 depth 和 action 共享任务分布，实验才问得出
  "depth 辅流对**这些**任务的 action 有没有帮助"。渲染成本参见 ttd 的 selfgen 并行度笔记。
* **C. 扩 selfgen 的 episode 数**（同一批任务，更多 seed）。解决数据量，不解决任务错配。

⚠️ 数据量本身也是个约束：1,028 个 episode 要承担**全部** depth 监督。按 §4.1 的
`selfgen@0.30`，一个 100k 步的 run 会从这 1,028 个 episode 里抽约 30k 个样本（约 29 遍）。
depth 任务看到的视觉多样性比 action 任务（~31k episodes）低一个量级以上。

### 2.1 修正后的 depth 可行性

| 树 | metric depth 可行性 |
|---|---|
| `rlbench_selfgen_512_aug` | ✅ 仿真渲染的真 metric depth（`view*/depth.npz`） |
| `rlbench`（官方 HF） | ❌ release 里根本没有 depth 文件 |
| `droid` | ⚠️ **VGGT 不行（已实测）**。真 metric 只能回到 ZED 双目 SVO 重算——贵，留作后续 arm |
| `bridge` | ❌ 单目 + 无任何米制锚点（没有 action-camera 对齐，这正是论文说它只能 video-only 的原因） |

**决策 D1：走方案 A** —— depth 只从 `rlbench_selfgen_512_aug` 抽。
选项 B（渲染官方 5 个任务补 depth）在 D2 之后已经**不再需要**：既然评测本来就在 selfgen 的
16 个任务上，depth 和 action 已经共享任务分布了。选项 C（扩 selfgen episode 数）仍然是
后续提升 depth 多样性的唯一途径，见 §4.4。

### 2.2 为什么排除官方 HF `rlbench`（决策 D2）

不是偏好，是三条查出来的事实：

1. **评测全在 selfgen 上。**
   * `eval/rollout.py:43` `DEFAULT_TASKS = ["push_buttons", "open_drawer", "meat_off_grill"]`
     —— 三个全是 selfgen 任务
   * `eval/rollout_env.py:224` 闭环 rollout 直接**复现
     `data/rlbench_selfgen/<task>/variation<v>/episodes/episode<s>` 的场景**
   * `eval_action.py` / `eval_perception.py` / `eval_all_masks.py` 的 `--data` 默认值都是
     `data/rlbench_selfgen`
2. **历史 arm 全是 selfgen。** `scripts/train_arm.sh:100`
   `DATASET="${DATASET:-rlbench_selfgen_512_aug}"`，`:253` `--dataset_name "${DATASET}@1.0"`。
   而 `train_arm.sh:28` 自己写了 "DATA TREE ... is **NOT a free knob** -- it changes what the
   arm is"：混入官方 rlbench 会让新 arm 与 arm0/arm4 不可比。
3. **官方那 5 个任务永远评不到。** 与 selfgen 的 16 个任务交集为空（§2.05），
   训练加进去只是稀释预算。

副作用是好的：去掉它之后 §2.05 的任务错配问题**自动消失**。

## 3. 修改方案

### 改动 1：接数据（5 分钟，零代码）

```bash
cd /workspace/ttdu/ActionImages-Cogen/data
ln -s /workspace/ttdu/ActionImages/data/bridge bridge
ln -s /workspace/ttdu/ActionImages/data/droid  droid
```

⚠️ `droid` 的 loader 会读 `data/droid/` 根目录下的 `cam2base_extrinsic_superset.json` 和
`camera_serials.json`——软链整个目录（不是只链 `processed/`）就都带过来了。

### 改动 2：per-dataset 模板菜单（主要改动）

问题：全局一个 `--template_mix` 无法表达"depth 只能从 selfgen 抽、bridge 不能有 action"。

如果保留全局 mix 再让每棵树静默过滤到自己可行的子集，**实际比例会和请求的不一致且不可见**
——这正是本仓库反复反对的那类静默降级（见 `rlbench_selfgen.py:_assert_seg_available`
"Refuse to silently train a seg template on episodes that have no seg ground truth"）。

建议加一个 **per-dataset 模板菜单**参数，语法沿用现有 `name@ratio` 风格：

```
--template_mix_per_dataset "
   rlbench_selfgen_512_aug=video+action@0.67,video+depth@0.33;
   droid=video+action@1.0;
   bridge=video@1.0
"
```

实现要点（`training/dataset/base.py`）：
1. 新增解析函数 `parse_per_dataset_template_mix(spec, default)` → `{ds_name: mix_str}`；
   没在里面出现的数据集回落到全局 `--template_mix`。
2. `name_to_ctor` 里把 `template_mix / prompt_tag_style / action_dropout_prob /
   segmentation_mode / strict_getitem` **也传给 `rlbench` / `bridge` / `droid`**。
3. 让 `RLBenchMVDataset / BridgeMVDataset / DROIDMVDataset` 继承与 selfgen 相同的
   模板逻辑。最省事的做法是把 selfgen 里那段"抽模板 → 建 prompt → 组 streams"提到
   `BaseDataset` 的一个 mixin/方法里，三个子类各自只声明**自己能提供哪些 modality**：

   ```python
   AVAILABLE_MODALITIES = ("video",)              # bridge
   AVAILABLE_MODALITIES = ("video", "action")     # rlbench, droid
   AVAILABLE_MODALITIES = ("video", "depth", "segmentation", "action")  # selfgen
   ```
4. **构造时校验**（loud，不静默）：菜单里出现该数据集提供不了的 modality → 直接报错，
   照抄 `_assert_seg_available` 的风格。这样"给 bridge 配 video+action"会立刻失败，
   而不是训练到一半才发现比例不对。

### 改动 3（可延后）：`--action_mask_mix`

**本次实验不需要。** 决策 D5 要的 action 侧数值 `(iiii .81, fiii .045, fifi .045, policy .10)`
**就是 `plan_segments` 里当前硬编码的值**（§4.3 已实测验证），所以 action 那条不用动。

只有将来要调 action 的 mask 比例时才做。届时注意：**分支顺序是 load-bearing 的**，
`templates.py` 已写明改顺序会让同一个 seed 训到不同的 mask 序列。做法应是
**保持现有分支结构、只把三个常量换成从 mix 元组算出的阈值**，这样默认值与现状逐决策等价，
`tests/test_forward_unchanged.py` 应当仍然通过。

### 改动 4：`is_rlbench` 闸门改成显式能力声明

现在 policy 模式靠 `is_rlbench` 这个布尔量闸控，语义上其实是"这个数据集的动作轨迹适合
做 single-frame policy 吗"。droid 有真动作，没有理由排除它。建议改名成
`allow_single_frame_policy`，由数据集声明，默认值保持 `rlbench=True, droid=False, bridge=False`
以复现现状，再由一个 flag 打开 droid。

---

## 4. 推荐的比例分配

### 4.1 数据集层：锚定论文 Table 1（决策 D3）

论文 Table 1 给的规模：

| | #Traj | #Views | Real | Action Ann. | Cam. Calib. | Cam. Motion |
|---|---|---|---|---|---|---|
| RLBench | 180k | 4 | ✗ | ✓ | ✓ | Diverse |
| DROID | 80k | 2 | ✓ | ✓ | ✓ | Static |
| BridgeV2 | 30k | 1–4 | ✓ | ✓ | **✗** | Static |

注意 Bridge 是**有动作标注但没有相机标定**——这才是它只能做 video-only 的真正原因：
没有标定就无法把 3D 动作投影成 action image。

按轨迹数均匀采样的隐含比例：**RLBench 62% / DROID 28% / Bridge 10%**。

但我们手上的绝对量和论文**正好相反**：

| | episodes | 若按 episode 均匀采样 | 论文隐含 |
|---|---|---|---|
| `rlbench_selfgen_512_aug` | 1,028 | 3.7% | 62% |
| `droid` | 9,437 | 33.5% | 28% |
| `bridge` | 17,709 | 62.9% | 10% |

所以**必须显式重采样**，不能让 `CombDataset` 按数据量走：

```
--dataset_name "rlbench_selfgen_512_aug@0.62,droid@0.28,bridge@0.10"
```

### 4.2 模板层（决策 D4）

```
rlbench_selfgen_512_aug = video+action@0.67, video+depth@0.33
droid                   = video+action@1.0
bridge                  = video@1.0          # 显式，不靠零动作兜底
```

bridge 显式配 `video@1.0` 而不是让 `train.py:343` 的零动作兜底去删 ACTION：兜底会让
**请求比例和实际训练比例对不上**，且属于本仓库一贯反对的静默降级。

### 4.3 mask 层：两条模板走同一套比例（决策 D5）

| | policy / single | iiii | fiii | fifi |
|---|---|---|---|---|
| `video+action` | 0.10 | 0.81 | 0.045 | 0.045 |
| `video+depth` | 0.10 | 0.81 | 0.045 | 0.045 |

**perception 这边零代码即可配置**，`parse_perception_mask_mix` 的元组顺序是
`(p_iiii, p_fiii, p_fifi, p_single_frame)`：

```
--perception_mask_mix "0.81,0.045,0.045,0.10"
```

action 那边这组数**就是当前硬编码值**，不用动（决策 D6）。

实测验证（各 40,000 次采样，`plan_segments` 直接统计实际产出的段计划）：

```
realised video+depth : {'iiii': 0.808, 'fiii': 0.0456, 'fifi': 0.0469, 'single': 0.0995}
realised video+action: {'iiii': 0.810, 'fiii': 0.0450, 'fifi': 0.0435, 'policy': 0.1014}
```

⚠️ **两个语义差异不是 bug：**

1. **perception 的第 4 档不是 policy，是"单帧 anchor"。** action 模板里 `single_frame` 把所有
   视觉段压成 1 帧、action 段保持全长（所以还有东西要预测）。perception 模板里所有段都是视觉的，
   全压就没有预测目标，`assemble()` 的 `assert not masks.all()` 会直接炸。所以 perception 的
   第 4 档只压 **anchor（RGB）**、depth 保持全长 = **"给一帧 RGB，生成整段 depth 视频"**。
2. **action 的 policy 档受 `is_rlbench` 闸控**（缺口 G4）。selfgen 是 rlbench 系，policy 正常生效；
   **droid 抽不到 policy**，它的真实分布是 `iiii .90 / fiii .05 / fifi .05`。要让 droid 也有
   policy，需要做改动 4。

### 4.4 折算到全局的实际份额

按 100,000 optimizer steps × accum 4 × 2 GPU = **800,000 samples**，
`action_dropout_prob=0.1`：

| 全局任务 | 份额 | 样本数 | 来源 episode 数 | 重复遍数 |
|---|---|---|---|---|
| `video+action` | **62.4%** | 499k | selfgen 1,028 + droid 9,437 | — |
| `video+depth` | **20.7%** | 165k | **只有 selfgen 788** | **210 遍** |
| `video`（纯视频） | **16.9%** | 135k | bridge 17,709 + action dropout | — |

各树的重复采样倍数：

| | 目标份额 | 样本数 | episodes | 遍数 |
|---|---|---|---|---|
| `rlbench_selfgen_512_aug` | 62% | 496k | **788** | **630** |
| `droid` | 28% | 224k | 9,437 | 24 |
| `bridge` | 10% | 80k | 17,709 | 4.5 |

**depth 的样本数不少（20.7%，辅助任务常见区间是 10–25%），稀缺的是 episode 多样性**：
它只见 **788** 个 episode（`VARIATIONS=0` 只放行 variation0），比 action 侧低一个量级，而且**再怎么调比例也解决不了**——
只能靠渲更多 selfgen（§2.1 选项 C）。

想加大 depth，只动 §4.2 里 selfgen 内部那个 `0.33`，selfgen 总曝光不变，纯粹在
action/depth 之间切：

| depth 占 selfgen | 全局 `video+depth` | 全局 `video+action` |
|---|---|---|
| 1/3（推荐） | 20.7% | 62.4% |
| 1/2 | 31.0% | 53.1% |

### 4.5 ⚠️ 实验设计上的混淆项，报告时必须写清楚

control arm（arm0）是 `rlbench_selfgen_512_aug@1.0`：selfgen 曝光 **778 遍**、
action 样本 **720k**。上面这个配置下 selfgen 只曝光 **482 遍**、action 样本 **499k**。

也就是说同样步数下你**同时改了两件事**：加了真实数据、减少了 selfgen 曝光。

诚实的 framing 是：**"同算力预算下，真实数据共训 + depth 辅流，值不值它挤掉的 selfgen 样本"**
——这本来也正是 co-training 该问的问题。但**不能**把结果说成"加 depth 带来的提升"。

要把两个因素拆开，需要三个 arm：

| arm | 配置 | 隔离的因素 |
|---|---|---|
| A0（已有） | `selfgen@1.0`，`video+action@1.0` | control |
| A1 | `selfgen@.62,droid@.28,bridge@.10`，全 `video+action`/`video` | 只加真实数据共训 |
| A2 | 同 A1，selfgen 内 `video+depth@0.33` | 在 A1 基础上只加 depth 辅流 |

A2 − A1 才是 depth 辅流的净效应；A1 − A0 是共训的净效应。只跑 A2 的话两者无法分离。

## 5. 验证步骤

1. **契约自检**：`RLBenchSelfgenDataset` 构造时已有 per-template 自检
   （`_self_test`，每个模板抽一个样本、检查 `streams` 齐全）。把它提到 BaseDataset
   之后，三棵新树也会自动被检查。
2. **比例实测**：照抄 `ActionImages/scripts/validate_datasets.py` 的做法，过 300 个
   batch 统计 `(dataset, template, mask_mode)` 的联合分布，和请求值对照。
   这一步能抓住"bridge 的 action 被静默丢掉"这类偏差。
3. **既有回归**：`tests/test_forward_unchanged.py` 必须仍然通过（A0 与现状等价的证明）。
   `scripts/run_tests.sh` 跑全量。
4. **短跑**：12 步 smoke，确认三棵树都能出样本、显存没变。
   ⚠️ `_batch_template` 禁止一个 batch 内混模板，`per_device_train_batch_size` 必须保持 1。

---

## 6. 改动清单（按依赖顺序）

| # | 文件 | 改动 | 规模 | 必需 |
|---|---|---|---|---|
| 1 | `data/` | 加 `bridge`、`droid` 软链 | 0 行 | ✅ |
| 2 | `training/templates.py` | `parse_per_dataset_template_mix()` | ~25 行 | ✅ |
| 3 | `training/dataset/base.py` | 模板逻辑（抽模板→建 prompt→组 streams）从 selfgen 提到 `BaseDataset` | ~80 行 | ✅ |
| 4 | `training/dataset/{rlbench,bridge,droid}.py` | 声明 `AVAILABLE_MODALITIES` | 各 1 行 | ✅ |
| 5 | `training/dataset/base.py` | `CombDataset` 给三棵树传模板参数 + 能力校验（loud） | ~40 行 | ✅ |
| 6 | `training/args.py` | `--template_mix_per_dataset` | ~15 行 | ✅ |
| 7 | `train.py` | 新参数接到 `CombDataset` | ~5 行 | ✅ |
| 8 | `scripts/train_mix.sh` | 新 launcher，参考 `train_rlbench_2gpu.sh` | ~60 行 | ✅ |
| 9 | `scripts/validate_mix.py` | `(dataset, template, mask_mode)` 联合分布实测 | ~80 行 | ✅ |
| 10 | `training/templates.py` | `--action_mask_mix` | ~40 行 | ⬜ 可延后（D6） |
| 11 | `training/templates.py` | `is_rlbench` → `allow_single_frame_policy` | ~15 行 | ⬜ 可延后（G4） |

### 目标启动命令

```bash
CUDA_VISIBLE_DEVICES=... MASTER_PORT=21001 \
bash scripts/train_mix.sh 2 \
  --dataset_name "rlbench_selfgen_512_aug@0.62,droid@0.28,bridge@0.10" \
  --template_mix_per_dataset "rlbench_selfgen_512_aug=video+action@0.67,video+depth@0.33;\
droid=video+action@1.0;bridge=video@1.0" \
  --perception_mask_mix "0.81,0.045,0.045,0.10" \
  --output_dir ./outputs/arm-A2-mix-depth
```

### 开跑前的三个硬约束

1. **`per_device_train_batch_size` 必须保持 1。** `train.py:_batch_template` 禁止一个 batch
   内混模板；且各数据集的 `extrinsics` dtype 不同（bridge float32 / droid float64），
   batch>1 会在 collate 阶段报 dtype 错。
2. **磁盘。** 一个完整 checkpoint 约 **127 GB**（`global_step*/` 三个文件
   38.5+38.5+38.4 GB，加 `step*.ckpt` 12 GB）。HF 是**先写新的再删旧的**，所以
   `save_top_k=k` 的峰值需求是 `(k+1)×127 GB`。`/workspace` 目前只剩约 100–220 GB
   （随其他任务波动），开跑前必须先腾空间并把 `--checkpoint_save_top_k` 设成 1，
   否则会在第一次存档时 ENOSPC——这正是 `mix-reproduce` 那次
   `PytorchStreamWriter failed writing file` 的原因。
3. **DeepSpeed bf16 patch。** 长跑前先跑
   `python scripts/patch_deepspeed_bf16_overflow.py`（在 ActionImages 侧已验证），
   否则单个非有限梯度会直接终止训练。
