# DATA —— 数据集现状、variation 划分、扩充流程

> 日期 2026-08-10。配套 `FORK_CHANGES.md`(改了什么)、`TESTING.md`(怎么验证)。

## 0. 为什么不能直接用官方 rlbench

| | 官方 `data/rlbench` | selfgen `data/rlbench_selfgen` |
|---|---|---|
| episodes | 4132 | 1735 |
| 任务 | 5 | 16 |
| 分辨率 | 512×512 | 256×256 |
| `depth.npz` / `mask.npz` | **完全没有** | 有 |
| 任务重叠 | **零** —— close_box / close_laptop_lid / open_box / open_microwave / wipe_desk | PerAct 16 任务 |

我扫过整棵官方树,**一个 `depth.npz` 都不存在**。所以 depth 臂只能用 selfgen 树,不是路径问题,
是深度 GT 根本没被导出过。要达到官方那个规模只能重新渲染。

## 1. 划分:variation0 训练,其余 variation 测试

按 variation 划分 = 留出**未见过的任务配置**(不同目标颜色、物体摆放、按钮顺序),比按 episode
随机划分更严格。

代码里由 `--variations` 强制执行(不是口头约定):

```bash
--variations 0      # 训练集:只有 variation0
--variations '!0'   # 测试集:除 variation0 外的全部
--variations all    # 不划分(上游行为,默认值)
```

实现在 `RLBenchSelfgenDataset._apply_variation_filter`,在 `_load_dataset` 之后过滤
`self.episodes`。**没有这个过滤,一次"训练"会静默地在自己的测试集上训练**,而 loss 和 r_peak
看起来都完全正常——只是测在模型背下来的数据上。`train_arm.sh`/`smoke_arm.sh` 默认传 `0`。

当前划分(2026-08-10 小批扩充后):

```
variations='0'   ->  396 episodes,16 个任务   (扩充前 315)
variations='!0'  -> 1420 episodes,13 个任务
```

### 有 3 个任务没有测试集(已接受)

`slide_block_to_target`、`stack_wine`、`sweep_to_dustpan` 在 RLBench 里**只有 variation0**
(`variation_count()==1`),所以 `!0` 只覆盖 13/16 个任务,这三个任务是**只训不测**的。

这不是 bug,是 RLBench 任务本身的性质。**用户 2026-08-10 决定接受**——报告结果时应说明
留出测试集覆盖 13/16 个任务,这三个任务只贡献训练数据。

## 2. 扩充 variation0 到官方规模

目标:16 任务 × seeds 0-258 = 最多 **4144** 集(≥ 官方 4132)。当前 315,待生成 **~3829**。

```bash
cd /workspace/ttdu/ttd
DRY=1 bash scripts/gen_var0_extend.sh      # 先看分片计划和磁盘预检
bash scripts/gen_var0_extend.sh            # 12 worker 起跑
NW=8 SEED_HI=127 bash scripts/gen_var0_extend.sh   # 想先跑小一点
```

**断点续跑**:`episode_done()` 看 `meta.json`,已完成的直接 skip。重跑同一条命令即可,不会重复生成。

### 这个脚本与既有 `gen_launch.sh` 的两点区别

1. **`--variations 0`**(本次给 `gen_dataset.py` 新加的参数,默认 `all` 保持原行为)。
   不加的话,`seeds 0-258` 会对 `push_buttons` 的 15 个 variation 全部生成 = 3885 集,
   其中 3626 集是**留出集**——算力绝大部分花在测试数据上。
2. **按 seed 分片,不按 task 分片**。`gen_launch.sh` 是 task round-robin,16 任务 12 worker
   分不匀(4 个 worker 拿 2 个任务),且任务耗时差异大(`push_buttons` 远慢于
   `slide_block_to_target`),先做完的 worker 就闲置。按 seed 切片让每个 worker 都跑全部 16
   个任务的一段 seed,负载天然均衡,而且 CoppeliaSim 每 worker 只启动一次。

### 实测数据(2026-08-10)

- **环境可用**:CoppeliaSim 在 `ttd/opt/`,`ttd_eval` 环境有 pyrep+rlbench,xvfb 在。
  实测生成 `slide_block_to_target` 2 集成功(中途一次 demo 采集失败被 `max_attempts=3` 兜住)。
- **新集与旧集兼容**:`depth.npz` dtype 都是 `float16`、256×256,codec 往返 AbsRel
  **0.0649%**(旧集 0.067%),可直接混用。
- **磁盘**:实测 ~11.8 MiB/ep → 3829 集需 **~44 GiB**。脚本内含预检(需求 + 60G 余量),
  不够就拒绝启动(D-038:`/workspace` 是全所共享卷,98% 满,会被外部租户瞬时打满)。
- **耗时(小批实测)**:8 worker、57 分钟产出 78 集 = **81 集/小时**,外推 12 worker
  ≈ **122 集/小时**。剩余 3748 集 ≈ **31 小时**。与历史聚合值(108 集/小时)同量级,可信。

### 小批验收结果(2026-08-10,`NW=8 SEED_HI=24`)

`ok=81 fail=4 skip=315`,variation0 315 → **396** 集,磁盘 +1G。全部 8 个测试文件通过:

| 检查 | 结果 |
|---|---|
| 新集 depth 对齐 + 往返(32 集 × 2 视角) | worst AbsRel **0.1051%** |
| 新集 seg 对齐 + 往返(96 个 instance-view) | worst IoU **1.0000**,另 8 个视角不可见→跳过 |
| seg_targets 覆盖率 | **396/396** |
| 既有 7 个测试 | 全绿(无回归) |

**`sweep_to_dustpan` 失败率高**:4 次失败全是它(seeds 3/12/22/24)。重试会换场景
(`np.random.seed(seed + v*1000003 + attempt*100003)`),所以是连续 3 个不同场景都规划失败,
即这个任务的运动规划失败率本身就高——它原本也只有 15/20 集,失败的 seeds 里 3 和 12 正是
原始缺的那几个。全量跑时它的产出约为请求量的 75-80%,其余任务不受影响。它同时也是那 3 个
没有测试 variation 的任务之一,影响面小,不值得为它改生成器。

## 3. seg 监督:新集也参与(已自动化)

`gen_dataset.py` **不写 `seg_targets.json`**,而缺了它的后果是**静默的**:
`_load_seg_targets` 返回 `None` → `getitem` 把整个样本降级成 video(有意设计,避免造出全空
seg 目标、教出"通常没有东西要分割"的错误先验)。seg 训练不会崩,只是新生成的几千集
**全部不参与 seg 监督**,有效样本还停在 315 —— loss 曲线上完全看不出来。

所以补齐这一步**直接接在 `gen_var0_extend.sh` 的仿真之后**,不是"事后记得跑一下",并且带
覆盖率校验(不足即 `exit 4`)。

它**不需要仿真器**:`seg_targets_gen.py` 从路径里的 variation 号、`meta.json` 的指令文本、
`handles.json`、以及(颜色类任务)`mask.npz`+`rgb` 的像素采样重建目标——正是每个任务
`init_episode()` 当初算的东西。已有文件会 skip。

**实测(2026-08-10)**:
- 对新生成的集有效:2 集全部写入,0 errors。
- 新集能产出**真实 seg 监督**:解码回 mask 与 `handles.json` 比对,两个视角
  **IoU 均为 1.0000**;背景纯黑像素占比 0.996(seg 编码形态正确,不是没被替换的 RGB);
  `action_7d.abs().sum()>0`(co-supervision 成立)。
- 速度:8 个颜色类任务的集 3 秒 → 全树 4144 集约 **25 分钟**。
- 扩充前基线:全树 1735/1735 覆盖率 100%。

手动复查覆盖率:
```bash
OUT=/workspace/ttdu/ttd/data/rlbench_selfgen_v2
find $OUT -mindepth 4 -maxdepth 4 -path '*variation0/episodes/episode*' -type d | wc -l
find $OUT -mindepth 5 -maxdepth 5 -path '*variation0/episodes/episode*/seg_targets.json' | wc -l
# 两个数必须相等
```

## 4. 扩充后要重跑的验证

```bash
bash scripts/run_tests.sh                              # 7 个测试文件
python debug/visualize_pipeline.py --no-vae            # 抽查新集的编码/对齐
```
新数据进来后特别要看 `test_selfgen_alignment.py`——它从磁盘独立重算 camera 参数再比对,
是新集 `camera_params.json` 格式是否与旧集一致的唯一把关。
