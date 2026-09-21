# 用模拟器回放评估感知质量 —— 交接文档

你要填的是**论文的 Table 2**（`Tables/tab_main.tex`，"Perception under simulator replay"）：
每个 arm **生成的感知**有多好。评分用的 ground truth 由模拟器产生，方法是**执行该模型自己
解码出的动作**，再渲染执行后的画面。

这份文档是完整的约定：为什么最直观的对比目标是错的、谁跑什么、跑什么命令、怎么读数字，以及已经踩过的坑。
**引用任何数字之前，先读 §9。**


---

## 0. 谁跑什么

**Table 2 的 arm 0–7 全部由你来跑。** arm 7 从 Hugging Face 下载（§4），所以表里每一行
都来自同一台机器、同一版代码、同一种训练配方。因此要跑：

- 所有**直出**行：Future RGB（arm 0–7）· 深度（1/4/6/7）· 法线（3/5/6/7）· 分割（2/4/5/7）；
- **专家天花板**扫描，它会写出胜出专家文件 `ceiling.json`；
- arm 7 的**级联**行（主表），以及**所有 arm 的级联**扫描（附录）；
- 在 HF 数据树上跑的**真实帧探针**（附录）。

天花板和任何 arm 都无关（专家读的是**模拟器**的画面，从不读模型生成的画面），所以用全部
video 锚定的 dump 来算：8 个 arm × 4 个任务 × 5 次试验。**`ceiling.json` 是"哪个专家胜出"的
唯一依据**，所有级联行都读它。

**暂不处理：ManiSkill3（`pull_cube`）。** 它的 rollout 循环还不会写 dump（§10）。所有 ManiSkill
格子和 `All avg.` 暂时留空。

---

## 1. 为什么不能用示教数据当 ground truth

在 `IIII` 条件下，模型只拿到第 0 帧和一条指令，其余全部自己生成，也就是说它自己编了一个未来。
如果拿生成的第 *t* 帧去对比**示教数据**的第 *t* 帧，量到的是两项之和：

```
AbsRel(生成, 示教)  =  感知误差  +  轨迹偏离
```

第二项既不小也拆不开，而且每个 checkpoint 都不一样，所以这样打分的两个 arm 根本没法比较。
`X+action` 模板让这个问题无法绕开：`depth+action` 的画布是 `D0 | A0 | D1 | A1`，里面没有任何 RGB，
不存在一种条件设置能既把未来钉住、又把深度留给模型去预测。

**所以我们换的是 ground truth，不是指标。** 在模拟器里执行模型自己解码出的动作，渲染这些
动作真正产生的结果：

```
目标  =  D_sim(s0, â_1:t)
```

模型不会再因为"没复现示教轨迹"而被扣分，它本来就没被要求复现。世界跟着模型的指令走，剩下
要比的只是：模型画出来的东西，和它自己的计划实际产生的东西是否一致。

**它测的是什么：** **以模型自己的轨迹为条件**的准确度。这是真正的准确度：真实的米制几何、
真实的遮挡、真实的物体位姿，裁判是模型只能通过动作去影响的模拟器。**它不测什么：** 这个未来
有没有用。一个计划原地不动、并且正确画出静止场景的模型也能拿高分。所以一定要和闭环成功率
（Table 1）一起引用。

## 2. 流程

```
 锚定模态下的第 0 帧  +  指令
        │
        ▼
 一张 41 帧的画布：  X0 | A0 | X1 | A1      X = video / depth / normal / segmentation
        │            └ 想象的 X     └ 想象的动作
        ▼
 解码动作段 → 41 个末端位姿 → 在 CoppeliaSim 里执行，每帧一步
        │
        ├─► 每执行一步：渲染真实的 depth + handle mask + RGB   ← 目标
        ▼
 对比想象的第 k 帧  ↔  执行第 k 步后模拟器的渲染
```

两边都是 512×512，在同一个相机下逐像素对齐（每次试验都把模拟器自己的外参/内参读回来存下）。
直接对比，不对齐、不拟合尺度、不做对应点搜索。

---

## 3. 把代码带过去，并证明完整到达

**整文件，不用补丁。** 补丁打到一个已经漂移的副本上，要么失败，更糟的是模糊匹配打到错误的
位置；整文件加 sha256，要么逐字节一致，要么明确不一致。

`handoff/simreplay_bundle.zip` 里有：

| 路径 | 内容 |
|---|---|
| `files/eval/rollout.py` | `--dump-percep`；写出原始数组，包括无损的 `sim_rgb` 和现场的 `seg_lut` |
| `files/eval/rollout_env.py`、`files/eval/rollout_env_maniskill.py` | `extra_renders` —— 不管锚定模态是什么都渲染 depth/mask |
| `files/scripts/score_sim_replay.py` | 打分脚本：直出 / 级联 / 天花板三类行 |
| `files/scripts/run_percep_simreplay.sh` | rollout 队列 |
| `files/tests/test_score_sim_replay_synthetic.py` | oracle 测试 |
| `files/PERCEP_SIMREPLAY.md` | 本文档 |
| `MANIFEST.sha256` | 上面每个文件的哈希 |
| `install.sh`、`verify.sh` | 按整文件安装（先备份被替换的文件），然后验证 |
| `fixtures/` | **在原机器上**生成的两个 dump，以及它们在原机器上的精确分数（真实模型那个来自 joint2src arm 7 —— 它测的是**打分脚本**，所以用哪个 checkpoint 无所谓） |

### 安装

```bash
unzip simreplay_bundle.zip -d simreplay_bundle
bash simreplay_bundle/install.sh /path/to/ActionImages-Cogen                  # 全部
bash simreplay_bundle/install.sh /path/to/ActionImages-Cogen --scorer-only    # 见下
```

每个被替换的文件都会先复制成 `<文件>.orig.<时间戳>`，本地改动不会丢。

**如果你正在跑 rollout 队列，用 `--scorer-only`。** 它只装打分脚本、测试和本文档，不碰
`eval/rollout*.py`。在队列运行中替换 `rollout.py`，下一个启动的任务就会用和上一个不同的代码，
同一张表的 dump 就来自两套代码。rollout 相关文件等队列跑完再装（如果需要的话）——已经写好的
dump 仍然有效（§3.2）。

### 如果是把整个 repo 打包过去

那就不需要 `install.sh`，`handoff/` 已经在 repo 里了。打包时要排除：`outputs/`、`data/`、
`checkpoints/`、`reports/`、`wandb/`、`logs/`、`ckpts.txt`。其中 `reports/` 里有原机器旧的
`percep_arm0_video_*` 结果，和你的任务同名，launcher 看到已存在的 `rollout_<tag>.json` 会直接
跳过；`ckpts.txt` 里是原机器的路径。**解压到一个新目录**，重建 `checkpoints/`、`data/` 软链和
你自己的 `ckpts.txt`，等当前队列跑完再切换，然后运行：

```bash
bash handoff/verify.sh /path/to/新目录/ActionImages-Cogen
```

### 验证 —— 没看到 `VERIFIED` 之前不要打分

```bash
bash simreplay_bundle/verify.sh /path/to/ActionImages-Cogen [--scorer-only]
```

按能证明的程度从弱到强，检查三件事：

1. **哈希。** 每个安装的文件都和打包的逐字节一致。
2. **导入和函数签名**，包括打分脚本依赖、但**没有**随包带过去的 repo 模块：
   `scripts/modality_mode_grid.py`（`seg_plan` 必须是返回 5 元组的版本，并且把 `depth+action`
   排成 `depth, action, depth, action`）以及 `training/percep/*`。其中某个文件版本漂移是现实中
   最可能出的问题：它不会报错，只会把画布切错，然后给出看起来合理的数字。
3. **用带过去的样本做跨机器复现** —— 同样的字节进，同样的数字出：
   - *Oracle。* 合成画布，里面"想象"的帧**就是**模拟器的帧，相当于一个完美的世界模型。它必须
     拿满分 —— LPIPS 0、mIoU 1、AbsRel ≈ 0.0007（codec 误差）、cos ≈ 0.99999 —— 且动态区域上
     copy-anchor 严格更差，**同时**和原机器的数字相差不超过 1e-4。这一步能抓住"执行步和画布帧
     错开一位"这种错误，哈希抓不到。
   - *真实模型。* 一个 arm 7 `depth+action` 试验必须精确复现 `0.25909 / 0.48902`（动态区域，
     预测 / floor）。
   - *只在完整模式下：* 用**安装后的** `rollout.py` 新跑一次 GT 回放，必须写出所有 key，且控制器
     误差低于 1 mm。（约 6 分钟，只用 CPU，不需要 GPU 和 checkpoint。）

### 3.2 本地自己改过的 rollout.py 写出的 dump

你在收到打包版本之前，自己改了 `rollout.py` 来保存现场的 seg LUT。打分脚本兼容这种
情况：它会查找 `seg_lut`、`role_lut`、`live_role_lut`、`lut`，或者任何名字里带 `lut`、长度为
65536 的数组；如果没有 `seg_present`，就从标注好的 mask 里推出 `present`。确认一次：

```bash
python -c "import numpy as np; print(np.load('<你任一个 seg 锚定的 .npz>').files)"
```

列表里必须有 `sim_depth`、`sim_mask`、`sim_rgb` 和一个 LUT key。如果缺 `sim_rgb`，你的
dump 就没法用于 Future RGB 那一行（§9.8）。

---

## 4. 需要准备什么

conda eval 环境、CoppeliaSim + `xvfb-run`、`checkpoints/` 下的 Wan2.2-TI2V-5B 底模，以及 repo
根目录下的 checkpoint 清单，每个 arm 一行：

```
# ckpts.txt
arm0 /abs/path/.../checkpoint-4000/step4000.ckpt
...
arm6 /abs/path/.../checkpoint-4000/step4000.ckpt
arm7 /abs/path/arm7__seed42_fi3_512_aug_sr/checkpoint-4000/step4000.ckpt
```

**从 Hugging Face 下载 arm 7**（公开仓库；12.83 GB；只需要 4000 步那个文件夹）：

```bash
huggingface-cli download TingtingDu/arm7__seed42_fi3_512_aug_sr \
  --include "checkpoint-4000/*" --local-dir /abs/path/arm7__seed42_fi3_512_aug_sr
ls -la /abs/path/arm7__seed42_fi3_512_aug_sr/checkpoint-4000/     # step4000.ckpt  12.83 GB
echo "arm7 /abs/path/arm7__seed42_fi3_512_aug_sr/checkpoint-4000/step4000.ckpt" >> ckpts.txt
```

关于这个 checkpoint，已知和未知的：

| | 取值 | 依据 |
|---|---|---|
| 模板 | `video+action@0.4`，`depth/segmentation/normal+action` 各 `@0.2` | `scripts/train_arm.sh` 的 arm7 |
| 数据 | RLBench `rlbench_selfgen_512_aug`，没有 ManiSkill | 名字里的 `_512_aug` |
| 分割 | `scene_roles` | 名字里的 `_sr` |
| frame interval | 3 | 名字里的 `_fi3` |
| 步数 | **10000** 步训练中的第 4000 步 | `trainer_state.json`（`max_steps` 10000） |
| `ACTION_MASK_MIX` | **未知** —— 名字早于 `_a1` 后缀的命名规则（2026-09-19），`trainer_state.json` 里也没存训练参数 | **需要对照那次训练的 wandb 配置确认** |

10k 训练里的第 4000 步和 6k 训练里的第 4000 步处在学习率计划的同一位置：学习率是
`constant_with_warmup`（1k 步预热后保持不变），优化过程不"知道"终点更远。各 arm 之间必须一致的
是步数、数据和 mask 配比 —— 把 8 行互相核对，并记在表注里（§9.5）。

**专家模型**

对照专家的选择参考 Vision Banana（[arXiv:2604.20329](https://arxiv.org/pdf/2604.20329)，Table 2–7）。
它在每个任务上比过的外部专家：

| 任务 | Vision Banana 比过的专家 |
|---|---|
| 语义分割（Cityscapes，零样本） | **SAM 3**、APE-D、OpenSeeD、X-Decoder |
| 实例分割 | SAM 3、OWLv2、DINO-X、Grounding DINO、LLMDet、APE-D |
| 米制深度 | **Depth Anything 3**、**Depth Pro**、UniK3D、MoGe-2 |
| 法线 | **Marigold**、**Lotus-2**、DSINE、StableNormal |

**已经接进打分脚本、可以直接用的：**

| 模态 | 名字（`--experts` 里用） | HF id | 每帧自由参数 | 备注 |
|---|---|---|---|---|
| depth | `da2metric` | `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf` | 0 | 米制 |
| depth | `depthpro` | `apple/DepthPro-hf` | 0 | 米制；Vision Banana 比过 |
| depth | `unidepth` | `lpiccinelli/unidepth-v2-vitl14` | 0 | 米制；**不在 PyPI 上** —— 见下 |
| normal | `marigold` | `prs-eth/marigold-normals-v1-1` | 0 | Vision Banana 比过 |
| normal | `lotus` | `jingheya/lotus-normal-g-v1-1` | 0 | Lotus 第一版；需要 Lotus 源码 —— 见下 |
| seg | `clipseg` | `CIDAS/clipseg-rd64-refined` | — | 文本条件的稠密分割，每个角色一句提示 |
| seg | `samclip` | `facebook/sam-vit-huge` + `openai/clip-vit-large-patch14` | — | **SAM + CLIP**：SAM 自动出 mask，CLIP 按角色描述给每个 mask 命名 |
| seg | `groundedsam` | `IDEA-Research/grounding-dino-base` + `facebook/sam-vit-huge` | — | **Grounded-SAM**：Grounding DINO 按角色短语找框，SAM 把框变成 mask |

三个分割专家都已用 oracle dump 跑通，天花板和级联读的是同一批像素。SAM 单独用进不了 mIoU 格子，
因为它输出的区域没有名字；配上 CLIP 或 Grounding DINO 给区域命名之后才能按角色打分。只看 SAM
边界质量的类别无关 bo-IoU 仍在附录探针里（`scripts/probe_sam_vs_ours.py`）。

```bash
for m in depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf apple/DepthPro-hf \
         lpiccinelli/unidepth-v2-vitl14 prs-eth/marigold-normals-v1-1 \
         jingheya/lotus-normal-g-v1-1 CIDAS/clipseg-rd64-refined \
         facebook/sam-vit-huge openai/clip-vit-large-patch14 IDEA-Research/grounding-dino-base; do
  huggingface-cli download "$m"
done
# UniDepth 不在 PyPI 上（`pip install unidepth` 找不到 —— 2026-09-21 实测）：
pip install git+https://github.com/lpiccinelli-eth/UniDepth
# Lotus 没有 pip 包；打分脚本从它的源码目录导入 `pipeline.LotusGPipeline`：
git clone https://github.com/EnVision-Research/Lotus /opt/lotus && export LOTUS_REPO=/opt/lotus
```

**Vision Banana 用过、还没接进来的 —— 你可以加。** 以下 ID 已在 2026-09-21 核实存在：

| 模态 | 专家 | 获取方式 | 权限 | 备注 |
|---|---|---|---|---|
| seg | **SAM 3** | HF `facebook/sam3` | **gated，需人工审批** | Vision Banana 的主分割对照；原生支持文本提示，最该加 |
| depth | Depth Anything 3（米制版） | HF `depth-anything/DA3METRIC-LARGE` | 公开 | 需要 `depth_anything_3` 包 |
| depth | MoGe-2 | HF `Ruicheng/moge-2-vitl-normal` | 公开 | 需要 `moge` 包；同时输出米制深度和法线 |
| depth | UniK3D | HF `lpiccinelli/unik3d-vitl` | 公开 | 需要 `unik3d` 包 |
| normal | Lotus-2 | HF `jingheya/lotus-2` | 公开 | Lotus 的新版本，Vision Banana 比的是这个 |
| normal | StableNormal | HF `Stable-X/stable-normal-v0-1` | 公开 | 需要它的源码包 |
| normal | DSINE | torch.hub `baegwangbin/DSINE` | 公开 | 不在 HF 上 |

**遇到 gated 的模型（目前只有 SAM 3），停下来告诉用户去 HF 页面申请权限**，不要换一个近似的
模型顶替。拿到权限后用 `hf auth login` 登录。

**SAM 3 的环境要求和现有 eval 环境冲突：** 官方包要求 Python ≥ 3.12、PyTorch ≥ 2.7、CUDA ≥ 12.6；
transformers 里的 `Sam3Model` / `Sam3Processor` 要 transformers ≥ 5.0。而 eval 环境是 Python 3.10 +
torch 2.6 + transformers 4.57。**不要为了 SAM 3 升级 eval 环境** —— Wan 的推理流水线依赖现在的版本。
正确做法是另建一个环境装 SAM 3，由打分脚本以子进程的方式调用（送 RGB 帧和角色短语，收回标签图）。
transformers 版的调用方式：

```python
from transformers import Sam3Processor, Sam3Model        # transformers >= 5.0
model = Sam3Model.from_pretrained("facebook/sam3").to("cuda")
processor = Sam3Processor.from_pretrained("facebook/sam3")
inputs = processor(images=image, text="robot arm", return_tensors="pt").to("cuda")
outputs = model(**inputs)
res = processor.post_process_instance_segmentation(
    outputs, threshold=0.5, mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist())[0]     # res["masks"], res["boxes"], res["scores"]
```

每个角色一句短语（同 Grounded-SAM 的短语表 `GDINO_PHRASE`），按分数从低到高把 mask 画进标签图，
没被任何 mask 覆盖的像素算 background —— 和 `groundedsam` 分支同一套规则。

**加一个新专家的做法：** 在 `scripts/score_sim_replay.py` 里对应的 `DEPTH_EXPERTS` /
`NORMAL_EXPERTS` / `SEG_EXPERTS` 字典加一项，再在 `depth_expert` / `normal_expert` / `seg_expert`
里加一个分支，输出格式和已有分支一样（深度：米制 `[H,W]`；法线：单位向量 `[H,W,3]`；分割：角色
标签图 `[H,W] uint8`）。加完**必须**跑一次：

```bash
python tests/test_score_sim_replay_synthetic.py handoff/fixtures/oracle_gt_close_microwave.npz --experts
```

它检查同一个专家在 oracle 画布上的级联和天花板分数完全相等，也就是两条路径读的是同一批像素。
注意这会改动 `score_sim_replay.py` 的哈希，`verify.sh` 第 1 步会报不一致 —— 这是预期的，把改动
告诉用户，由用户决定是否更新打包版本。

尽量多装：天花板扫描的候选越多，越不容易被说"级联只是输给了一个弱专家"。缺包的候选会
**被跳过并打印原因**，不会悄悄丢掉 —— 装上之后它就会进入扫描。只有**米制**深度模型能当级联候选：
尺度不确定的模型（DA3 相对深度版、DA V2 相对深度、VGGT、Lotus/Marigold 深度）需要对着 GT 拟合
尺度，而能让这件事公平的"只拟合锚定帧的仿射变换"还没实现（§10）。它们放在附录扫描里
（`scripts/probe_depth_baselines.py`）。

**可选 —— 附录用的真实帧探针：** 干净的未见任务数据树已经上传到 HF：

```bash
huggingface-cli download TingtingDu/rlbench_unseen_tasks_512_clean --repo-type dataset \
  --local-dir data/rlbench_unseen_tasks_512_clean \
  --include "close_microwave/*" "toilet_seat_down/*" "close_box/*" "meat_on_grill/*"
```

Table 2 **不需要**它（§6.3），它用于附录探针（§6.5）。

---

## 5. Table 2

照搬自当前的 `Tables/tab_main.tex`。**每个格子都是动态区域上的 `预测 / copy-anchor floor`。**
各 arm 训练过的输出空间（`scripts/train_arm.sh`）：

```
arm0 V       arm1 V+D     arm2 V+S     arm3 V+N
arm4 V+D+S   arm5 V+N+S   arm6 V+D+N   arm7 V+D+N+S (XGenAct)
```

```latex
\begin{tabular}{@{}llccccccc@{}}
\textbf{Arm (training outputs)} & \textbf{Prediction route}
  & Close microwave & Toilet seat down & Close box & Meat on grill & RL avg. & Pull cube & All avg. \\
\midrule
\multicolumn{9}{@{}l}{\textbf{Future RGB} --- LPIPS $\downarrow$} \\
Arm 0 (V) ... Arm 6 (V,D,N)        & direct video           & ...  \\
Arm 7 (V,D,N,S)                    & direct video           & ...  \\
\midrule
\multicolumn{9}{@{}l}{\textbf{Metric depth} --- dynamic-region AbsRel $\downarrow$} \\
Arm 1 / Arm 4 / Arm 6              & direct co-generation   & ...  \\
Arm 7                              & direct co-generation   & ...  \\
Arm 7                              & RGB $\rightarrow$ depth expert        & ... \\
\emph{Expert ceiling}              & GT RGB $\rightarrow$ depth expert     & ... \\
\midrule
\multicolumn{9}{@{}l}{\textbf{Surface normal} --- dynamic-region mean cosine $\uparrow$} \\
Arm 3 / Arm 5 / Arm 6              & direct co-generation   & ...  \\
Arm 7                              & direct co-generation   & ...  \\
Arm 7                              & RGB $\rightarrow$ normal expert       & ... \\
\emph{Expert ceiling}              & GT RGB $\rightarrow$ normal expert    & ... \\
\midrule
\multicolumn{9}{@{}l}{\textbf{Segmentation} --- dynamic-region mIoU $\uparrow$} \\
Arm 2 / Arm 4 / Arm 5              & direct co-generation   & ...  \\
Arm 7                              & direct co-generation   & ...  \\
Arm 7                              & RGB $\rightarrow$ segmentation expert & ... \\
\emph{Expert ceiling}              & GT RGB $\rightarrow$ segmentation expert & ... \\
\end{tabular}
```

表格本身注释里定下的规则：直出行**只**给该 arm 训练过的模态（其他组合 launcher 会拒绝）；video
没有级联（RGB 本身就是级联的输入）；`RL avg.` 是 4 个任务均值的宏平均；`All avg.` 等 `pull_cube`；
**每个方法每个任务的试验次数必须相同**（§9.4）。

---

## 6. 运行

### 6.1 Rollout

```bash
DRY_RUN=1 bash scripts/run_percep_simreplay.sh                # 先看要跑哪些任务，什么都不启动
bash scripts/run_percep_simreplay.sh                          # ckpts.txt 里列出的每个 arm 的所有训练过的模态
CONFIGS="arm1_depth arm4_depth" TASKS=close_box bash scripts/run_percep_simreplay.sh
OUT=$PWD/reports/percep_simreplay_v2 GPUS="0 1 2 3" bash scripts/run_percep_simreplay.sh
```

**每次正式启动前先跑一次 `DRY_RUN=1`。** 它做完全部检查、打印队列和每个 arm 对应的 checkpoint，然后
退出，不碰 GPU。默认的组合只来自 `ckpts.txt` 里列出的 arm —— 某个 arm 还没下载完，它会被略过，
而不是让整批启动失败；用 `CONFIGS=` 显式指定时，缺 checkpoint 会直接拒绝。注意 `GPUS=""` 会被当成
没设置、回落到默认的 `0 1 2 3`，要限制 GPU 请写明卡号。

一个任务 = `<arm>_<anchor> × <task>`，5 次试验。launcher 会在启动**之前**拒绝该 arm 没训练过的
`(arm, anchor)` 组合和缺失的 checkpoint；它支持断点续跑（`rollout_<tag>.json` 已存在的 tag 会跳过
—— 所以新一轮评测要把 `OUT` 指向新目录）。协议和 unseen14 那一轮逐字节一致（cfg 7.5、fi 3、
explicit tags、sphere solver、planning、factor 1.5、skip-anchor 4、ik-streak 5、seed 42、
res 512），因此试验可以按 `scene_seed` 和 `reports/closedloop_unseen14/` 一一配对。

实测开销：每个任务 18–44 分钟；每个任务约 28 GB 显存（80 GB 的卡能放两个，放不下三个）；每次
试验约 30–50 MB 磁盘。

| 组合 | 任务数 |
|---|---|
| arm 0–6 的 16 个训练过的组合 × 4 个任务 | 64 |
| `arm7_video`、`arm7_depth`、`arm7_normal`、`arm7_segmentation` × 4 个任务 | 16 |
| **合计** | **80** |

arm 7 那一行写进 `ckpts.txt` 后，`CONFIGS="arm7_video arm7_depth arm7_normal arm7_segmentation"`
就只排 arm 7 的任务；launcher 启动前会检查文件是否存在。

专家只需要约 4–8 GB 显存，所以在一张已经跑着两个 28 GB rollout 任务的 80 GB 卡上，可以和队列
**同时**打分：跑完一个就打一个。

### 6.2 直出行

```bash
python scripts/score_sim_replay.py --dump_root <你 rollout 时的 OUT>/percep_dump --rows direct --gpu 0
```

`--dump_root` 指向 rollout 时 `OUT` 目录下的 `percep_dump/`（launcher 默认的 `OUT` 是
`reports/percep_simreplay`）。结果写在 `percep_dump/` 的上一级目录。

对每个 dump 的锚定模态原生产出的内容打分：`depth` → AbsRel，`normal` → 平均余弦，
`segmentation` → mIoU（§7.3），`video` → 对无损模拟器 RGB 的 LPIPS。每次试验约 45 秒。输出：
`scores_direct.json`，外加打印出来的 **Table 2 视图** —— 每个 (行, arm) 一行，每个任务一个
`dyn/floor(n)` 格子，4 个任务齐了之后给出 `RL avg.`。重新打分只读 `.npz` dump：**改指标永远
不需要重跑 rollout。**

### 6.3 专家天花板 —— 在任何级联之前跑

```bash
python scripts/score_sim_replay.py --dump_root ... --rows ceiling --gpu 0
```

**前提：** video 锚定的 dump 必须包含 `sim_rgb`（§3.2）。如果本地的 `rollout.py` 没写这个，就等
队列跑完后装上打包版本，重跑 `armN_video` 这些组合；没有 `sim_rgb`，这一行和 Future RGB 都没法打分。

它让每个装好的专家读**模拟器的无损 RGB**（`sim_rgb`），和同一步模拟器的 depth / 法线 / 角色图
对比 —— 这些帧模型从没碰过，所以这一行和任何 arm 都无关。它读 video 锚定的 dump（和级联打分的
是同一批场景；用 `--ceiling_tags` 可改），每隔 4 个有效步取一步（`--ceiling_stride`）。法线会把
四种坐标轴约定都算一遍（我们的是 x 右、y 下、z 远离相机；常见 benchmark 是 y 上、z 朝向相机
—— 搞错会让余弦悄悄反号）。

默认扫描 §4 表里**所有已接入**的候选（缺包的会被跳过并打印原因）。只想跑其中几个时用
`--experts`，例如 `--experts depth=da2metric+depthpro,normal=marigold,seg=clipseg+groundedsam`；但
写进 Table 2 的 `ceiling.json` 必须来自一次包含全部可用候选的扫描。最慢的是 `samclip`（SAM 自动
mask 要在整张图上撒点），显存紧张时它会逐帧失败 —— 见 §7.2 的 `!`。

也可以一条命令跑完 —— 它会先把天花板完整跑完，然后才让级联去读胜出者：

```bash
python scripts/score_sim_replay.py --dump_root ... --rows direct,ceiling,cascade --gpu 0
```

它会按事先定死的规则，把**每个模态的胜出者**写进 `ceiling.json`：

> 每个模态，对照专家 = **动态区域上合并后的天花板分数**最好的那个候选。法线的坐标轴约定和专家
> 一起选。

这条规则的三个性质，都不能改：

1. **只在天花板行上选，绝不在级联行上选。** 级联行就是要报告的数字，按它选专家等于按自己的结果
   做选择。天花板行不会泄漏，因为它从不看模型输出。
2. **选最强的，不选最弱的。** 选个弱对手是在讨好自己 —— `probe_normal_specialists.py` 里已经写明了
   这条纪律。如果联合生成还能赢过最强的可用专家，这个结论才有分量。
3. **零自由参数。** 只有米制深度模型能当候选；需要对齐的模型要作为单独命名的一行、并声明对齐
   方式，这是 Table 2 caption 的要求。

**为什么不再需要真实帧数据树。** 天花板需要带 ground truth 的真实 RGB。dump 里的 `sim_rgb` +
`sim_depth` 正是这个，而且和被参照的格子是同样的场景、同样的步、同样的相机、同样的干净域 ——
比数据集 episode（那是别的场景）匹配得更紧。HF 数据树（§4）仍然用于附录里的数据集帧探针。

`ceiling.json` 是胜出者的唯一依据。以后重新打分 —— 加了试验、装了新专家 —— 就要先重跑天花板，
再用新文件重跑所有级联，绝不能把按两份不同胜出文件打出来的级联行混在一起。

### 6.4 级联 —— 主表是 arm 7，附录是所有 arm

```bash
python scripts/score_sim_replay.py --dump_root ... --rows cascade --tags 'percep_arm7_video_*' --gpu 0
```

读 `ceiling.json`，**只**用胜出者去读该 arm **想象出来的** RGB，对着同一个模拟器目标打分。缺
`ceiling.json` 时拒绝启动。级联的 floor 是它自己的：专家读模型的锚定 RGB 帧 —— "假设 RGB 不变，
再去感知它"。`--cascade depth=depthpro,...` 可以覆盖胜出者，**只用于附录**。

arm 7 的级联是主表那一行（和直出行同一份权重，所以能单独隔离出"先生成 RGB 再感知"这一顺序
的影响）。对**所有** video 锚定的 dump 跑同一条命令 `--tags 'percep_arm*_video_*'`，就是 Table 2
注释里放到附录的所有 arm 级联扫描；arm 0 那一行是被移到附录的纯 RGB 对照。

### 6.5 真实帧探针 —— 附录

和任何 rollout 无关：每个专家读同样 4 个任务的**数据集**帧，来自 HF 数据树（§4）。包括尺度
不确定的深度模型，它们暂时还不能当级联行。

```bash
printf '%s' "$(for t in close_microwave toilet_seat_down close_box meat_on_grill; do
                 for i in 0 1 2 3 4; do printf '%s/variation0/episodes/episode%s,' $t $i; done
               done)" | sed 's/,$//' > /tmp/eps.txt
export DATA_TREE=$PWD/data/rlbench_unseen_tasks_512_clean EPS_FILE=/tmp/eps.txt SPLIT_TAG=_simreplay5
for m in da2metric depthpro unidepth da3 da2 vggt lotusdepth marigolddepth; do
  python scripts/probe_depth_baselines.py "$m" 20
done
python scripts/probe_normal_specialists.py lotus 20; python scripts/probe_normal_specialists.py marigold 20
python scripts/probe_named_seg_baseline.py; python scripts/probe_sam_vs_ours.py
python scripts/collect_counterparts.py
```

附录探针脚本只覆盖 CLIPSeg（`probe_named_seg_baseline.py`）和纯 SAM 的边界质量
（`probe_sam_vs_ours.py`）；`samclip` 和 `groundedsam` 目前只在 dump 上的天花板 / 级联里有。

这些和 rollout 不是同一批场景，所以永远不会取代 Table 2 里基于 dump 的天花板。如果它们选出的
胜出者和 `ceiling.json` 不同，就在附录里写明 —— 这是有信息量的结果，不是要靠"挑好看的那个"来
解决的冲突。

---

## 7. 怎么读输出

```
percep_arm7_depth_close_microwave  depth:direct  full 0.0475/0.0973  dyn 0.2424/0.6207 (14.8%)  gap 0.00058
                                                 └ 模型/floor ┘      └ 模型/floor ┘    └占比┘   └ 执行误差 ┘
```

### 7.1 `full` 和 `dyn`

一个 episode 里相机是固定的，84–91 % 的像素从头到尾不动，全画面平均的大部分区域里"假设什么都
没变"是**完全正确**的答案。动态区域是**反事实深度**相对于该 chunk 参考帧变化超过 1 cm 的像素
—— 机械臂、它移动的物体、以及它们带来的遮挡变化。**所有模态都用同一个区域**，包括分割和 RGB，
所以同一列的格子覆盖的是同一批像素。**Table 2 报告的是动态区域。**

### 7.2 `floor` 和 `gap`

**`floor` = copy-anchor**：把模型自己的锚定帧复制到每一步，用完全相同的流程打分 —— "假设场景
静止"。它不是竞争模型，而是一条线：低于它的数字没有意义。每一种对照指标都可能被"冻住不动"的
预测骗过，floor 就是用来抓这个的。对完美的世界模型，每个模态在动态区域上 floor 都严格更差
（oracle 测试会检查这一点）。

**`gap` = 执行误差**，指令位姿和**实际到达**位姿之间末端位置距离的中位数。想象的帧对应的是模型
**画出来**的位姿，渲染对应的是控制器**实际到达**的位姿。它随预测距离增大（arm 7 的一次试验：
k<10 时 0.5 mm，k 10–19 时 6 mm，k≥20 时 118 mm；GT 回放全程 0.3 mm）。主结论应取误差还在毫米级
的那几段（json 里的 `by_horizon`）；回放没通过这个门槛的格子留空。

**`!` = 这一格有帧失败了。** 某些帧打分时出错（比如显存不够），剩下的帧仍会给出一个数字，但它和
别的格子不是在同一批帧上算的。打分脚本会在格子后面加 `!`，并在最后列出 `PARTIAL rows`。**带 `!`
的格子不能报告**，释放显存后重跑打分即可（不用重跑 rollout）。

`k` = 想象 chunk 里的第 *k* 帧 = 上次重新规划后执行的第 *k* 步；一个 *k* 是 0.15 秒。丢弃的帧：
**`k = 0`**（锚定帧，是直接给的）和 **`k < 4`**（`is_ramp` —— `skip_anchor_frames` 把这些位姿
换成了过渡插值，不是模型画的）。

### 7.3 分割用的是 `mIoU_merged`

`target` 是唯一一个由**指令**而不是场景决定的角色：训练时会根据每个 episode 的
`seg_targets.json`，把指令提到的 `distractor` 物体提升为 `target`。现场的模拟器没有这个文件，
未见任务数据树里也没有，所以模拟器的 GT **永远不会**出现 `target`，而模型被训练成会画它。
由此带来两个后果，都已处理：

- 打分脚本解码模型输出时**允许出现 `target`**。如果只允许现场出现过的角色，红色的 target 像素会
  被归到剩下颜色里最接近的那个 —— 紫色，也就是 `fixture` —— 一个画对了的 target 就会被判成错的
  类别。（实测：修复前一块纯 target 颜色被解成 `fixture`，修复后是 `target`。）
- 然后在预测和 GT 两边都把 `target` **合并进 `distractor`**，正好去掉这一个依赖指令的区分，别的
  不动。指标名也相应改了。

外部分割专家也按同一规则打分。它们拿到的角色描述不同：CLIPSeg 和 SAM+CLIP 用完整句子
（`ROLE_PROMPT`），Grounded-SAM 用短名词短语（`GDINO_PHRASE`）；`target` 和 `distractor` 在
Grounded-SAM 里共用一个短语 `object`，反正打分时会合并。

mIoU 是**打分区域内** GT 中出现的角色的宏平均；GT 里的 `unknown` 像素（没映射上的 handle）不计；
每次试验的 `seg_unknown_frac` 记录它们的比例。

### 7.4 Future RGB 是对无损帧的 LPIPS

AlexNet LPIPS，空间图在区域上取平均。目标是 `sim_rgb`，每执行一步原样存下。**绝不要**对
`videos/` 下的 mp4 打分 —— H.264 压缩会让 LPIPS 测到的很大一部分是编码器，而不是模型。

---

## 8. 原机器上的参考数字（是另一个 arm 7）

**这些来自 joint2src arm 7，不是表里用的 Hugging Face arm 7。** 它们只说明预期的量级，以及和
floor 的关系；**不是** HF checkpoint 的复现目标，也不能抄进 Table 2。

arm 7 第 4000 步，4 个 RLBench 任务，5 次试验，干净域（`reports/percep_simreplay/`）：

| 路径 | 全画面 模型/floor | 动态区域 模型/floor |
|---|---|---|
| arm 7 深度，直出（4 任务均值） | **0.0598** / 0.0992 | **0.3238** / 0.7051 |
| arm 7 深度，级联 → DA V2 metric（4 任务均值） | 0.3762 / 0.4501 | 0.4192 / 0.6985 |
| DA V2 metric 读真实**数据集**帧 | raw 0.3947 / median 对齐后 0.1109，ρ = +0.980 | — |

`exec_gap` 中位数 0.6–11 mm。你 arm 1 的冒烟测试给出动态区域 0.3199 / 0.5887 —— 量级
一致，**但那是另一个 arm，只能算合理性检查，不是复现。** 复现检查是 `verify.sh` 第 3 步用带过去的
样本做的那个。

---

## 9. 已知的坑

1. **不要单独引用全画面数字**，不要引用没带 floor 的数字，也不要引用没带执行误差的远距离数字。
   有一次不具代表性的试验里，模型是 0.0630、floor 是 0.0644，看起来毫无用处；4 个任务合起来，
   它在动态区域上比 floor 好 54 %。
2. **arm 没训练过的模态就没有对应的行。** arm 0 是 `video+action@1.0`，在 Table 2 里只有一行，
   Future RGB。
3. **级联目前被专家本身卡住了，不是被设计卡住。** DA V2 metric 读真实帧 raw 就有 0.395（结构没问题，
   ρ = +0.980；米制尺度在合成渲染上是坏的）。这就是要扫描天花板、选最强专家的原因。如果天花板
   已经比 arm 7 的直出行还差，级联行就没法体现"先生成 RGB 再感知"这个顺序的影响，必须如实说明，
   不能当成赢了。微调 vs 零样本的差距依然存在：干净的解决办法是在同一棵数据树上训练一个
   `video+depth` 专家，而现有的 arm 里没有这样的模型。
4. **没有可打分帧的试验是一个结果，不是漏跑。** arm 7（joint2src）`close_box` 的第 0 次试验在
   第 7 步连续 5 次 IK 失败，只记下了两帧过渡帧，所以没有贡献任何分数（当前 Table 2 暂定数字里的 `†` 就是它；那些数字来自原机器的 joint2src
   arm 7，会被替换，但这条规则同样适用于你跑出来的结果）。种子是
   固定的，**重跑会复现同样的结果** —— 不要"重跑到凑够 5 次"。每个格子都报告 *有效试验数 / 5*，把
   "规划出了执行不了的动作"当作关于这个模型的结果。在不同的试验子集上比较格子就是选择偏差的陷阱：
   在表旁边报告每个 arm 的可执行比例。
5. **不核对的话，行与行之间差的不只是观测空间。** 所有 arm 必须共用同样的步数、数据树和
   `ACTION_MASK_MIX`。HF arm 7 的 mask 配比没有记录在 checkpoint 里（§4）—— 要确认。原机器上的 arm
   （joint2src arm 7 是 `A1`，arm 0 是 `A0`）是另外的模型，完全不在这张表里。
6. **只有干净域。** rollout 环境用的是固定相机的原版 RLBench；训练时用了 Colosseum 随机化，但
   rollout 里没接上。因此 RGB/LPIPS 有一部分测的是外观域的差异。
7. **分割锚定的 rollout 从一个分布外的锚定帧开始。** 现场 LUT 没有 `target` 提升（§7.3），所以给
   分割策略的锚定帧里，target 显示的是它的基础角色颜色 —— 训练时从没见过这种情况。打分端的 GT 问题
   已经处理了；输入端事后没法修，要写进 caption。
8. **RGB 行和天花板需要 dump 里有 `sim_rgb`。** 它是 2026-09-21 才加的；更早的 dump 没有，这些行
   会被跳过并打印原因。
9. **检查 GT 回放这个门槛。** 在 planning / factor 1.5 / fi 3 下对 `close_microwave` 跑 1 次
   `--gt-replay`，超时、成功率 0 %，而 arm 7 闭环成功率 85 %。还没查原因。它不影响感知打分（反事实
   目标不需要任务成功），但影响对 Table 1 的解读。
10. **`--dump-percep` 每一步都会多花模拟时间**（三种渲染 × 两个视角）—— 跑成功率评测时千万别开。

## 10. 尚未实现

1. **ManiSkill3 `pull_cube`。** `rollout_env_maniskill.py` 已经支持 `extra_renders`，但
   `eval/rollout_maniskill.py` 的循环不写 dump。需要把 `eval/rollout.py` 里写 dump 的那段移植过去。
2. **只拟合锚定帧的仿射变换**，有了它，尺度不确定的深度专家才能当级联行。
3. **Lotus 和 UniDepth** 已经接进打分脚本，但原机器上没装、没测过；请你装上并跑一次 oracle 测试（§4）。
4. **SAM 3 和 §4 里 Vision Banana 的其余专家**还没接入。SAM 3 需要用户先拿到 HF 权限，并且要单独的环境（§4）。
5. **Future RGB 还没有外部对照。** Vision Banana 没有视频任务可参考。能走同一套回放协议的只有能同时
   输出动作的模型 —— 候选是 Action Images 发布的初始化 checkpoint（`step125750`，它自己的协议是
   cfg 10、fi 4、不加 tag，预测窗口 8 秒而不是 6 秒）。是否加进主表由用户决定，launcher 还没有它的配置。
