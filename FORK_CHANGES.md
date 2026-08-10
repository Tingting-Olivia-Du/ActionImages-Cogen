# FORK_CHANGES —— 相对上游 ActionImages 的全部改动

> 上游基线：`291f71cd74b154b8eaf49020035b84c7e044b542`（`fix device mismatch`，与
> UMass-Embodied-AGI/ActionImages HEAD 一致）。
> 改动日期：2026-08-09。改动来源：`/workspace/ttdu/ttd`（percep codec + selfgen 数据层）。
> 设计依据：`/workspace/ttdu/ttd/plan/core/15-a4_io_current_vs_desired_action_perception_cogen.md`
>
> ---
>
> ## ⚠️ 2026-08-10：本文件的 §0 / §1 / §2.1 / §2.2 / §2.6 / §3 / §5 已被取代
>
> 顶替式设计（depth 顶替 video、`forward` 逐字不变）已升级为**任务模板**设计：
> dataset 输出多路 `streams`，prompt 写出模态集合 Π，mask 决定哪些段给定。
> 原因：顶替式布局下一个 depth 样本里**没有 RGB**，那是 depth 空间的世界模型，
> 不是两篇论文定义的感知（RGB 进 → depth 出）。
>
> **当前权威描述见 `CHANGES_TEMPLATES.md`**（逐文件改动 + 行为差异 + 迁移表），
> 设计依据见 `/workspace/ttdu/ttd/plan/core/16-multitask_template_design.md`。
>
> 下面保留原文不做历史删改（追溯价值优先），但读之前请先看上面那份。
> 仍然准确、未被取代的小节：§2.3（rlbench.py 的 view_dirs）、§2.4（base.py 的 provenance 与
> 重试收紧）、§2.5（percep 包的搬运）、§4（运行方式与软链）。
>
> ---

## 0. 一句话（**已被 CHANGES_TEMPLATES.md 取代**）

官方是 `<video><action>` 沿时间轴拼 4 段联合去噪；本 fork 让 **depth / segmentation 顶替
video 那两段**，得到 `<depth><action>` / `<segmentation><action>`，并且**不再清零动作**——
感知监督与动作监督第一次出现在同一个样本、同一次梯度更新里。

```
官方   [ v1_rgb   | v1_action | v2_rgb   | v2_action ]     4 段
本 fork [ v1_depth | v1_action | v2_depth | v2_action ]     4 段（形状/显存/loss 完全相同）
```

「不再清零动作」这一条仍然成立且仍然是关键；「顶替」与「forward 不变」两条已不再是当前设计。

---

## 1. 最重要的结论：`ActionImagesModel.forward` 一行都没改（**已不成立**）

> 2026-08-10：`forward` 现在按模板组装段序，见 `CHANGES_TEMPLATES.md` §2.2。
> 本节末尾预言的「那是重新协商基线的正确时机」正是那次改动。

`forward` 是**按位置**切视觉流的，从不看像素语义：

```python
video_src_latents = self.pipe.encode_video(video[:, :, :T, ...])   # train.py:236
video_tgt_latents = self.pipe.encode_video(video[:, :, T:, ...])   # train.py:237
latents = torch.cat([video_src_latents, action_src_latents,
                     video_tgt_latents, action_tgt_latents], dim=2)  # train.py:291
```

所以只要 dataset 把 depth 编码后的 RGB 写进 `sample["video"]` 的**两半**，拼出来的序列就
是 `[v1_depth | v1_action | v2_depth | v2_action]`。段数、段长、相机嵌入
`[c_v1, c_v1, c_v2, c_v2]`、mask `{0, T_l, 2T_l, 3T_l}`、loss 全部不变。

模型靠什么分辨每段是什么模态？靠 `noisy_latents[masks] = origin_latents[masks]` —— 每段首帧
是**干净的真值**，模态"自报家门"。这就是为什么换模态不需要新 head、新 loss 项、新分支
（doc 15 §3）。

**推论**：`<video><action>` 基线的 bit-exact 是**构造上成立**的，不是测出来的。
`tests/test_forward_unchanged.py` 把这一点变成 CI：它拿 `git show <上游commit>:train.py` 和
当前文件比对 `forward` / `get_inputs` / collator 的**源码文本**。如果哪天要做 6 段的
`<video><depth><action>`，这个测试**就应该失败**——那是重新协商基线的正确时机。

---

## 2. 逐文件改动

### 2.1 `train.py`（+40 行，均在 `train()` 与模块级；**训练核心未动**）

| 改动 | 为什么 |
|---|---|
| 新增 `assert_own_training_package()`，在 `train()` 开头调用 | `ttd_train` 环境里有一个**同名** `actionimages` 的 editable 安装，把 `training` 映射到 `/workspace/ttdu/ActionImages/training`。它是 `sys.meta_path.append` 的，所以只要本仓库根在 `sys.path` 上，stdlib `PathFinder` 就会赢——但如果没设，`import training` 会**静默**加载另一个仓库，本 fork 的所有改动全部失效而运行看起来完全正常。改成立即崩溃 |
| `CombDataset(...)` 传入 `visual_modality_mix` / `strict_getitem` | 把新参数接到数据层 |
| wandb config 增加 `visual_modality_mix` / `dataset_name` / `training_package` | 这三个都是"静默出错"源，记进 run 里而不是信任启动命令 |

### 2.2 `training/args.py`（+19 行，2 个字段）

- **`visual_modality_mix`**（默认 `"video@1.0"`）：视觉段用哪个模态。`name@ratio` 逗号分隔，
  语法与 `dataset_name` 一致。默认值就是上游行为，所以**不传这个 flag 的运行即基线**。
- **`strict_getitem`**（默认 `False`）：关掉 dataset 重试，冒烟跑用。

**没有**照搬 ttd 的 `args_core.py`：`use_cache` / `cache_root` / `checkpoint_probe_*` /
`perception_prob` / `use_selfgen_mixture` 对应的功能本 fork 都不做（见 §5）。

也**没有**照搬 ttd 的 P0-2 安全网（`perception_prob>0 且 batch_size>1 → raise`）。原因：
ttd 那里 perception 样本和 action 样本**张量形状/段数不同**，混在一个 batch 会被 batch 级的
`sum(action_7d**2)` 判据整体误路由。本 fork 里 depth 是**顶替** video 占同一个槽位，所有模态
形状完全一致、`sum(action_7d**2)` 对所有样本一律非零，混合 batch 能正确 stack——这个限制不再
成立。（`forward` 的 mode-mix 随机数与 `inputs["path"][0]` 仍是 batch 级的，所以实践上仍用
`per_device_train_batch_size=1`，但那是显存与配方的理由，不是正确性的理由。）

### 2.3 `training/dataset/rlbench.py`（+15 行）

`get_all_views` 增加 `view_dirs`，**与 `videos` 同步 append**，结束时写入 `self._last_view_dirs`。

为什么需要：感知模态必须从**产生这些像素的同一个视角**读 `depth.npz`/`mask.npz`。而
`glob.glob` 的顺序既不排序也不保证跨调用稳定（实测一个 episode 返回
`['view4','view2','view3','view1']`），事后重新 glob 去还原映射，正是 ttd 当年用
monkeypatch 绕开的那类错位。

**刻意不 `sorted()`**：排序会改变 `random.sample` 在给定 seed 下选中的视角，破坏与上游的一致性。

已知遗留（**刻意不修**）：上游 `extrinsics`/`intrinsics` 在另一个 `if` 里 append，所以
"有 camera_params.json 但没有 video.mp4"的视角会让两者错位。修它会改变行为、破坏 bit-exact
基线，因此只在注释里标注，由需要的调用方自行断言。

### 2.4 `training/dataset/base.py`（+55/-4 行）

**(a) `getitem` 返回值新增 3 个 key**：`frame_indices` / `view_indices` / `view_dirs`。

这是把 ttd 的 `_CaptureViewsAndFrames`（115 行、patch `glob.glob` + `random.sample` +
`load_video_frames` 的 scoped monkeypatch）**整个删掉**换来的。理由：

1. 它 patch 的是进程级全局名字，而 eval 脚本**也**在 patch `random.sample`；两者嵌套能工作
   是巧合，不是契约。
2. 它 115 行里约 45 行是运行期校验，存在的唯一原因就是它**看不到真值**。直接返回真值后这些
   校验全部不需要。
3. 新设计每个样本都要用到视角信息，不只是感知样本。返回 provenance 让 P0-1 不变量变成
   **结构性的**（"样本自带它是从哪个窗口/视角构造的"），而不是**被检查的**。

collator 只转发一份显式 key 列表，所以这三个字段永远到不了模型。bridge/droid 没有
`_last_view_dirs`，通过 `getattr` 默认拿到 `[]`。

**(b) `__getitem__` 重试收紧**：上游是失败 50 次、每次打印 `str(e)` 后换个随机 index。
系统性失败（比如感知路径对**每个** episode 都抛异常）在这个逻辑下表现为刷屏 + 静默改喂别的
样本——训练看起来很健康，学的东西完全不是你要的。改为：保留重试（真·瞬时 IO 错误），但
**连续 5 次失败即 raise**，并带上第一次的完整 traceback；`strict_getitem=True` 时完全不重试。

**(c) `CombDataset`** 新增 `rlbench_selfgen` 条目 + 两个透传 kwarg。

### 2.5 `training/percep/`（新包）

从 `ttd/src/percep/` **原样搬运** `depth_codec.py` / `seg_codec.py`（唯一改动：depth_codec
`__main__` 里的默认 glob 路径改指向本仓库的 `data/rlbench_selfgen` 软链）。

放在 `training/` 包内而不是顶层 `percep/`，这样 `from training.percep.depth_codec import ...`
和 `training.utils` 走同一套解析机制，删掉了 ttd 里的 `sys.path.insert(0, .../percep)` ——
那种 hack 正是"我到底在跑哪个仓库的代码"含糊不清的来源。

**没有搬**：`percep_tasks.py`（第二套注册表，`encode()` 签名从来没和
`_encode_perception` 对上过，两套注册表就是 ttd 当年漂移的原因）、`task_objects.py`
（自己的 docstring 就写着 DEPRECATED）、`normal_codec.py`（A3 预研）、`seg_targets_gen.py`
（495 行离线生成器，需要 RLBench/PyRep；16 个任务的 `seg_targets.json` 数据里已经有了）。

### 2.6 `training/dataset/rlbench_selfgen.py`（新文件，~300 行）

ttd 版本的重写。**逐字保留**：`_try_load_cache`/`_save_cache`（自采数据会增长，不能缓存
episode 索引）、`get_instruction`（读 meta.json）、`get_8d_action`（selfgen 是原生 20Hz
1:1，**不能** `[::4]` 降采样）、`get_camera_params`（非零填充的 frame key）、
`_load_depth`/`_load_mask`/`_load_seg_targets`/`_referred_id_groups`。

**删除**：`_CaptureViewsAndFrames`（§2.4）、`_load_handles`（已被 `seg_targets.json` 取代）、
`_debug_alignment`（已被一等公民 provenance 取代）。

**新增/改写**：

| 项 | 说明 |
|---|---|
| `visual_modality_mix` 解析 + 逐样本抽取 | 取代 ttd 的 `perception_prob` 二选一 |
| **不再清零 `action_7d`/`action_8d`** | ttd 在 `:360-361` 清零，这是"互斥"在数值上的载体。删掉它就解锁了 doc 15 §2.5 判定为"不可达"的那一格 |
| **两个视角都编码感知** | ttd 只编码 target 视角、cond 视角留 RGB。在"顶替"布局下那会让一个 depth run 里混进一段 RGB |
| `perception_task` → `visual_modality` | 语义反转了：不再是"这个样本是感知任务**而不是**动作任务"，而是"视觉槽位里装的是什么" |
| `_to_model_res()` 几何守卫 | ttd 假设"selfgen 渲染就是 256 所以不用 resize"，未强制。`--height 512` 时 RGB 半边是 512² 而 depth 半边是 256²，`torch.cat` 抛异常 → **被 50 次重试吞掉**变成无限刷屏。现在断言正方形，并且**先 nearest-resize 原始 GT 再编码**（对深度不连续处做 bilinear 会造出场景里不存在的深度；对编码后的颜色做 bilinear 会造出被 `path_decode` 投影到错误段的颜色） |
| `_self_test()` 构造期自检 | 直接调 `getitem(0)`（绕过重试包装），在 5B 模型加载前、wandb 起 run 前就炸 |
| `MODALITY_PROMPT` | `video` **刻意无前缀**：加 `<video>` 会改变**每个**样本的文本分布（包括基线），既破坏 bit-exact 又把 prompt 移出预训练先验的流形。只标注非默认模态 |

---

## 3. 测试（`tests/`，全部 CPU、无需 GPU，`bash scripts/run_tests.sh` 约 1 分钟）

| 文件 | 守住什么 |
|---|---|
| `test_forward_unchanged.py` | 训练核心与上游源码逐字相同；段序与 mask 偏移是字面量校验；`forward` 里不得出现 `visual_modality`/`depth`/`perception` 等字样 |
| `test_depth_codec.py` / `test_seg_codec.py` | 从 ttd 移植，仅改 import。实测 depth 往返 **AbsRel 0.067%**，seg IoU 1.0 |
| `test_prompt_tags.py` | `<depth>` 能活过 `train.py:225` 的 `.replace(".","")/.replace("_"," ")` 清洗；UMT5 只多切 **3 个 token** 且不撞 `<unk>`；episode 路径含 `"rlbench"`（`train.py:265` 的 10% 单帧条件变体以此为门，数据软链改名会**只对这个数据集**悄悄改掉训练配方） |
| `test_selfgen_dataset.py` | **`action_7d.abs().sum() > 0` on a depth sample** —— ttd 的对应测试断言的正好相反，这一行就是整个科学改动；两半都必须是 depth（同 seed 重抽保证比的是同一次 draw，否则这条断言是空的）；混合 mix 能正确 stack；非法 mix 串在构造期被拒 |
| `test_selfgen_alignment.py` | 从磁盘独立重算 camera/extrinsics/intrinsics 与样本比对，两个模态、两个视角都测。**带负控制**：交换 cond/target 必须被检出 |
| `test_depth_roundtrip_e2e.py` | 最高价值的一条：把 `sample["video"]` 两半解码回米制与 `depth.npz` 比。一条断言同时覆盖错视角/错窗口/错顺序/错归一化/错 resize/错通道序/半 RGB。实测 **AbsRel 0.089%**，负控制（跨视角）**37.6%** |

所有测试当前**全绿**。

---

## 4. 运行方式

```bash
cd /workspace/ttdu/ActionImages-Cogen
export PYTHONPATH=/workspace/ttdu/ActionImages-Cogen     # 必须；见 §2.1
conda activate ttd_train

bash scripts/run_tests.sh                 # CPU 测试全跑
GPUS=3 bash scripts/smoke_arm.sh depth    # 30 步冒烟 + 峰值显存
GPUS=3 bash scripts/smoke_arm.sh video    # 对照的 loss 量级 / 显存
GPUS=0,1,2,3 bash scripts/train_arm.sh depth   # 正式臂
GPUS=4,5,6,7 bash scripts/train_arm.sh video   # 对照臂（同数据、同 seed、同步数）
```

两个臂**只差 `--visual_modality_mix` 一个 flag**，所以 r_peak 的差异只能归因于视觉模态。

数据/权重软链（都被 `.gitignore` 覆盖）：
```
checkpoints            -> /workspace/ttdu/ActionImages/checkpoints
data/rlbench_selfgen   -> /workspace/ttdu/ttd/data/rlbench_selfgen_v2   # 名字必须含 "rlbench"
data/rlbench           -> /workspace/ttdu/ActionImages/data/rlbench     # 官方 HF，回归用
```

---

## 5. 明确**没有**做的事

- **segmentation 臂**：`seg_codec.py` 已搬运、单测已通过、`_encode_perception` 的 seg 分支和
  prompt 也已写好，但没有跑过。启用只需 `--visual_modality_mix segmentation@1.0`，不需要动
  训练端代码。
- **标签集合泛化**：没有 doc 15 的 Π 子集/`P(Π)` 分布、没有 `<video>`/`<action>` 原子标签、
  没有 given/predict 语法。`visual_modality_mix` 是三个互斥值上的扁平 `name@ratio`。
- **6 段 `<video><depth><action>`**：需要 doc 15 §4.3 的累积偏移 mask 循环，`forward` 就不能
  保持逐字不变了；序列翻 1.5 倍、attention 约 2.25 倍。
- **cache 路径**：无 `cached_dataset.py` / `--use_cache` / `--cache_root`。感知样本本来就不可缓存。
- **checkpoint 探针**：无 `checkpoint_monitor=r_peak`、无探针 subprocess。`_save_checkpoint`
  保持上游实现（**注意**：因此它仍会先调 `super()._save_checkpoint()` 写一份 fp32
  `model.safetensors`，脚本里带了 `--save_safetensors False` 压掉）。评测在选定 checkpoint 上手动跑。
- **bridge / droid / 混合数据**：只用 `rlbench_selfgen@1.0`。
- **eval 脚本移植**：`g0_openloop_rlbench_v2.py` / `g0_perception_zeroshot.py` 尚未搬过来。
  搬的时候要顺手修 ttd 版 `g0_perception_zeroshot.py:126` 的一个真 bug——它用
  `sorted(glob(...))` 重新推导视角，而 camera 张量来自 dataset 的**未排序** glob 顺序，
  两者不一致时就是 P0-1 那一类错位（本 fork 的 provenance 字段让它无法再发生）。
- **`inference.py` / `wan_video_action_images.py`**：未改。
- **loss 加权**：所有模态都在 `[-1,1]` 的 RGB-like 空间，统一 masked MSE，零行改动。

---

## 6. 下一步（按顺序）

1. `GPUS=<空卡> bash scripts/smoke_arm.sh depth`，再跑一次 `video`：验证 30 步 loss 有限、
   量级同阶、**峰值显存一致**（4 段形状相同，显存本应先验相等；差距大说明有东西改了形状）。
2. L2 数值回归：官方 HF 数据 + `--learning_rate 0` + 20 步，在 pristine `/workspace/ttdu/ActionImages`
   与本仓库各跑一次，`max|Δloss| < 1e-5`。
3. 移植两个 eval 脚本，在 `step125750` 上取三个参照点：
   video 的 `r_peak`（先验上限）、depth 的 `r_peak`（换成深度条件后还能不能动）、
   depth 的 AbsRel（零样本几何能力基线）。
4. 起两个臂。**"在学"的判据** = depth AbsRel 明显低于零样本基线，**同时** depth 臂的 r_peak
   接近 video 先验上限。这一对数字第一次是在真正的同批次 co-supervision 下读出来的，而不是
   ttd 那种交替单任务（doc 15 §6.2）。
