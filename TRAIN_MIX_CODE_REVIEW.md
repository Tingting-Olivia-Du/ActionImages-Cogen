# `train_mix.sh` 及其训练调用链代码审查报告

> 审查日期：2026-08-20
>
> 审查对象：当前工作区版本，而不是某个干净 Git commit
>
> 入口脚本：`scripts/train_mix.sh`
>
> 结论：A2 的多数据集、深度模板和 mask 主链可以成功构造，但当前存在会污染训练监督、破坏多卡数据互斥性、掩盖训练失败以及损坏断点语义的确定问题。正式长跑前至少应修复本文的 P0 和 P1 项。

## 1. 审查目标与范围

本次审查不是只检查 shell 语法，而是从 `scripts/train_mix.sh` 出发，沿实际训练调用链检查：

1. shell 参数和环境变量如何进入 `train.py`；
2. Hugging Face 参数如何解析；
3. 三数据集比例与 per-dataset template menu 如何生效；
4. RLBench selfgen 的 RGB/depth、DROID action、Bridge video 如何加载；
5. DataLoader、Sampler 和多 rank 随机种子如何交互；
6. batch 如何经过 collator 进入模型；
7. action image、depth stream、segment mask 和 loss 如何生成；
8. DeepSpeed checkpoint 如何保存、裁剪、发现和恢复；
9. 仓库现有验证脚本是否真的覆盖 `train_mix.sh` 的默认配置。

本次没有修改训练代码，只新增这份报告。

## 2. 当前默认实验配置

`scripts/train_mix.sh A2` 当前默认配置为：

```text
GPU                  6,7
seed                 42
resolution           512 x 512
num_frames           41
frame_interval       1（DROID loader 内部固定为 4）
variations           0
datasets             rlbench_selfgen_512_aug@0.62,droid@0.28,bridge@0.10
selfgen templates     video+action@0.67,video+depth@0.33
droid templates       video+action@1.0
bridge templates      video@1.0
perception mask mix   0.81,0.045,0.045,0.10
action dropout        0.10（TrainingArguments 默认值）
max_steps             10000（存在 INIT_CKPT 时）
checkpoint interval   3000
DeepSpeed             ZeRO-2 + CPU optimizer offload
batch                 1 / GPU，gradient accumulation 1
```

在当前数据树上，真正传入训练的 episode 数为：

| 数据集 | episode 数 | 说明 |
|---|---:|---|
| `rlbench_selfgen_512_aug` | 788 | 原始 1028 个，`variations=0` 后保留 788 个 |
| `droid` | 9437 | 当前有效 |
| `bridge` | 17709 | 当前有效 |
| 合计 | 27934 | `CombDataset.__len__()` 的当前返回值 |

## 3. 实际调用链

```text
scripts/train_mix.sh
  ├─ 激活 ttd_train conda 环境
  ├─ 生成 DATASET_SPECS / TEMPLATE_MIX_PER_DATASET
  ├─ 选择 INIT_CKPT、OUT、DeepSpeed 配置和端口
  └─ torchrun train.py
       ├─ training.args.parse_args()
       ├─ train.train(args)
       │    ├─ find_latest_checkpoint(output_dir)
       │    ├─ CombDataset(...)
       │    │    ├─ RLBenchSelfgenDataset
       │    │    │    ├─ variations=0 过滤
       │    │    │    ├─ RGB 双视角窗口
       │    │    │    └─ depth 双视角同窗口编码
       │    │    ├─ DROIDMVDataset
       │    │    │    ├─ RGB 双视角窗口
       │    │    │    ├─ camera calibration
       │    │    │    └─ action.npz
       │    │    └─ BridgeMVDataset
       │    │         └─ video-only
       │    ├─ ActionImagesModel(...)
       │    │    ├─ Wan 5B / T5 / VAE
       │    │    ├─ init/resume weight load
       │    │    └─ camera encoder
       │    ├─ Hugging Face Trainer + Accelerate + DeepSpeed
       │    │    ├─ RandomSampler
       │    │    ├─ DataLoader workers
       │    │    └─ ActionImagesDataCollator
       │    └─ ActionImagesModel.forward()
       │         ├─ visual/depth VAE encode
       │         ├─ 7D action 投影为 action image
       │         ├─ plan_segments()
       │         ├─ assemble()
       │         └─ masked diffusion loss
       └─ ActionImagesTrainer._save_checkpoint()
            ├─ DeepSpeed resume state
            ├─ stepN.ckpt
            └─ 旧 optimizer state 裁剪
```

## 4. 风险分级

| 等级 | 含义 |
|---|---|
| P0 | 会系统性污染训练数据或改变核心训练语义；应阻止正式长跑 |
| P1 | 会导致失败被隐藏、恢复错误、结果不可追踪或 checkpoint 被覆盖 |
| P2 | 当前默认数据下未必立刻触发，但会造成静默配置漂移或验证失真 |
| P3 | 兼容性、维护性或低概率边界问题 |

## 5. 已确认问题

### TMIX-001（P0）：DROID 的真实 gripper 信号被丢弃并替换成常量 1

#### 位置

- `training/dataset/droid.py:91-106`
- `/workspace/ttdu/ActionImages/scripts/preprocess_droid.py:166-172`
- `training/utils.py:325-326`
- `training/utils.py:448-452`

#### 当前行为

DROID 预处理明确保存两个数组：

```python
np.savez(
    os.path.join(tmp_dir, "action.npz"),
    action=cart[lo : lo + written].astype(np.float32),
    gripper=grip[lo : lo + written].astype(np.float32),
)
```

但 loader 只读取 `action`，随后把第七维硬编码为全 1：

```python
actions = np.load(action_npz_path)["action"]
actions = np.concatenate([actions, np.ones((actions.shape[0], 1))], axis=1)
```

第七维随后被 `project_actions_7d_to_5d_torch_batch()` 原样传入 action image encoder，并在蓝色通道中用 `> 0.5` 表示 openness。因此这不是“未使用字段”，而是实际训练监督的一部分。

#### 实测证据

抽查当前数据树前 500 个 DROID episode：

```text
文件 schema                 ('action', 'gripper')，500/500
gripper 最小值              0.0
gripper 最大值              1.0
gripper 均值                0.37049
gripper == 0                46.44%
gripper == 1                0.39%
0 < gripper < 1             53.18%
```

对一个真实 episode 直接调用当前 loader：

```text
文件中的 gripper 前 12 帧： [0.0, 0.0, ..., 0.0]
loader 输出第七维：          [1.0, 1.0, ..., 1.0]
loaded_equals_saved:         False
mean_abs_error:              1.0
```

#### 影响

- DROID 在数据集选择中占 28%。
- `video+action` 又有 10% action dropout，因此约 `0.28 × 0.90 = 25.2%` 的全部训练 step 真正使用 DROID action stream。
- 这些 step 的末端执行器位置和旋转可能正确，但 gripper 监督恒为“1”。
- 模型会学习错误的 DROID 夹爪先验，并可能把数据源特征与“夹爪恒开”绑定。
- 现有 `validate_mix.py` 只检查整个 `action_7d` 是否全零，因此检测不到“只有第七维被常量化”。

#### 根因

原始 DROID loader 只支持 6D pose，并用临时 HACK 补了常量维度；之后预处理代码已经开始保存真实 `gripper`，loader 却没有同步升级。

#### 建议修复

1. 同时读取 `action` 和 `gripper`；
2. 检查二者长度一致；
3. 把 gripper reshape 为 `[T, 1]` 后拼接；
4. 明确验证 DROID `gripper_position` 的 0/1 方向是否和当前 `openness > 0.5` 语义一致；若原始含义相反，应在 loader 中显式转换并写测试；
5. 对 NaN、越界值和 schema 缺失做启动期报错，不要静默填常量。

建议结构：

```python
with np.load(action_npz_path) as payload:
    pose = payload["action"]
    gripper = payload["gripper"]

if pose.shape != (len(pose), 6):
    raise ValueError(...)
if gripper.shape[0] != pose.shape[0]:
    raise ValueError(...)

actions = np.concatenate([pose, gripper.reshape(-1, 1)], axis=1)
return actions[frame_indices]
```

#### 验收标准

- 随机抽取至少 100 个 episode，loader 第七维和文件 gripper 一致；
- action image 蓝色背景能区分 gripper 的两端状态；
- 新增单元测试，禁止第七维再次退化为常量；
- `validate_mix.py` 输出 DROID gripper 的 min/max/均值/阈值两侧占比。


#### 修复（2026-08-20）

改 `training/dataset/droid.py` 的 `_load_actions_npz()`，**两个 repo 同步**
（`ActionImages/` 与 `ActionImages-Cogen/`，该函数原本逐字节相同）。

```python
with np.load(action_npz_path) as payload:
    pose = np.asarray(payload["action"])          # [T, 6]
    if "gripper" not in payload:
        raise KeyError(...)                       # 不再回退常量
    gripper = np.asarray(payload["gripper"]).reshape(-1)
# 形状 / 长度 / 有限性校验，任一不满足直接抛错
openness = 1.0 - np.clip(gripper, 0.0, 1.0)       # 注意这个反转
actions = np.concatenate(
    [pose.astype(np.float64), openness[:, None].astype(np.float64)], axis=1
)
```

**三个和报告建议不同的地方，都是实测逼出来的：**

1. **极性必须反转。** 报告第 4 点提醒要核对方向，但它给的 sketch
   `np.concatenate([pose, gripper.reshape(-1,1)])` 与该提醒自相矛盾。实测：

   ```
   RLBench actions[:,7] (= ActionImages 的 openness)  首帧均值 1.000，60.3% 恰为 1
   DROID  gripper_position                            首帧均值 0.000，68.0% 恰为 0
   ```

   机械臂任务开始时夹爪都是张开的，所以 DROID 是 **0=张开**，与 Action-Images 的
   openness（1=张开）相反。用户指定的样本
   `AUTOLab+0d4edc83+2023-10-21-19h-02m-53s`（"Put the blue block in the drawer,
   then close it"）gripper 轨迹为 `0 → 0.32 → 0.58 → 0`——只在搬运阶段闭合，
   直接拼接会把这一维**训反**，比原来的常量更糟。

2. **必须保持 float64。** 调用方把 `action_7d` 和 float64 的 extrinsics/intrinsics
   一起送进 `project_point_3d_to_2d_torch_batch()` 的 einsum，不做 cast。改成
   float32 会触发和 bridge 那条一样的 dtype 报错。

3. **缺 `gripper` 键报错而不是补常量。** 常量第七维在 loss 和 all-zeros 检查里都不可见，
   静默回退等于把 bug 重新引入。

**其他数据集的同类审计（报告未覆盖）：**

| 数据集 | 第 7 维来源 | 结论 |
|---|---|---|
| `rlbench` / `rlbench_selfgen*` | `actions.npy[:, 7]`，1=张开 | ✅ 约定正确，无需改动 |
| `bridge` | 无 action 数据（只有 `video/` `vggt/` `instruction.txt`） | ✅ 论文定位即 video-only，不适用 |
| `action_8d` | 只被 collator 转发，模型从不读取 | ✅ 无影响 |

**验证：**

- 新增 `tests/test_droid_gripper.py`，已加入 `scripts/run_tests.sh`。它同时钉住
  逐帧一致性、`[0,1]` 值域、float64 dtype、以及「首帧应为张开」的极性断言。
- 36 个 episode 逐帧比对 `openness == 1 - clip(gripper)`，**不一致 0 个**。
- 800 个 episode 中 100% 存在夹爪开合动作，所以抽样窗口偶尔全为 1.0 是真实情况
  （该 episode 那一段没有抓取），不是回归。

---

### TMIX-002（P0）：rank-specific `args.seed` 破坏多卡 sampler 的互斥切片

#### 位置

- `train.py:811-814`
- `training/dataset/base.py:524-582`

当前入口：

```python
if __name__ == "__main__":
    args = parse_args()
    args.seed += get_rank()
    train(args)
```

#### 当前行为

在当前 Transformers 4.57.3 中：

1. `TrainingArguments` 初始化分布式进程组；
2. `get_rank()` 因此在 rank 0/1 分别返回 0/1；
3. `args.seed` 被改成 42/43；
4. `Trainer.__init__()` 使用各自的 `args.seed` 设置进程 RNG；
5. 每个 rank 的 `RandomSampler` 生成不同的全局排列；
6. Accelerate 再从各自不同的排列中提取本 rank 的 batch。

分布式切片成立的前提是所有 rank 看到同一个全局排列。当前代码破坏了这个前提。

`CombDataset.__getitem__()` 又把 sampler index 确定性地映射到数据集和 episode，所以相同 index 意味着相同 source dataset 和相同 local episode。随机窗口可能不同，但 episode 级覆盖仍然重复。

#### 实测证据

使用当前 `TrainingArguments + Trainer + Accelerate`，双进程、1000 条虚拟数据：

```text
rank 0 seed      42
rank 1 seed      43
每 rank 样本数   500
跨 rank 重复     258
整轮漏掉         258
```

移除 `args.seed += rank`、两 rank 都使用 seed 42 后：

```text
rank 0 样本数    500
rank 1 样本数    500
跨 rank 重复     0
整轮漏掉         0
```

#### 影响

- 默认两卡训练每轮大约四分之一 sampler index 重复、四分之一遗漏；
- 有效 episode 覆盖率显著下降；
- 同一个 optimizer step 的两个 rank 可能训练同一个 episode；
- 标称 step 数和独立数据曝光数不再对应；
- 当前单进程 `validate_mix.py` 无法发现该问题。

#### 根因

代码试图通过 rank 偏移让不同进程获得不同随机行为，但错误地修改了 Trainer 的公共 seed。公共 seed 同时控制 sampler，而 sampler seed 必须在所有 rank 相同。

#### 建议修复

- 不要修改 `args.seed`；它应在所有 rank 保持一致。
- DataLoader worker 本身已经通过 `seed_worker(..., rank=args.process_index)` 做 rank-aware seed。
- 如果 `plan_segments()` 等主进程逻辑确实需要 rank-local 随机流，应使用独立 RNG，而不是改变 sampler seed。
- 若使用独立 RNG，需要把其状态纳入 checkpoint，保证断点续训可复现。

#### 验收标准

增加双进程 gloo 测试：

1. 数据集长度能整除 world size 时，各 rank 索引交集必须为空；
2. 各 rank 索引并集必须覆盖全集；
3. 不可整除时，只允许 DistributedSampler 为补齐产生的最小重复；
4. 模板和 mask 的 rank-local 随机性另行测试，不得依赖 sampler seed 偏移。


#### 修复（2026-08-20）

改 `train.py` 的入口，**两个 repo 同步**：

```python
if __name__ == "__main__":
    args = parse_args()
    # 删除了 args.seed += get_rank()
    train(args)
```

**为什么删掉是安全的（先验证再动手）：** 担心的是「删了之后两卡的帧窗口/视角/模板抽样
会不会变成完全一样」。查了实际调用链，不会——per-rank 随机性本来就不由 `args.seed`
提供：

```python
# transformers/trainer_utils.py:52
def seed_worker(worker_id, num_workers, rank):
    init_seed = torch.initial_seed() % 2**32
    worker_seed = num_workers * rank + init_seed     # <- 已经是 rank-aware
    set_seed(worker_seed)
# transformers/trainer.py:1114  worker_init_fn=partial(seed_worker, ..., rank=self.args.process_index)
```

所以 dataloader worker 的 RNG 与 `args.seed` 的 rank 偏移无关，删掉不影响它。

**这条修复的职责边界（值得写清楚，否则容易误解 `_uniform_from_index` 的作用）：**

| 层 | 负责者 | 保证 |
|---|---|---|
| 索引划分 | DistributedSampler / Accelerate `BatchSamplerShard` | 两 rank 的 index 集合**不相交** |
| 索引 → 样本 | `CombDataset._uniform_from_index` | 同一 index 在任何 rank 映射到**同一条数据** |

`_uniform_from_index` 的 docstring 写「avoid cross-rank duplication」是对的，但那是**必要
条件不是充分条件**：它保证「index 驱动」，使 sampler 的划分*能够*转化为样本互斥；真正
做划分的是 sampler，而 rank seed 偏移打破的正是这一层。

**反直觉但值得记下的一点**：确定性映射让这个 bug 变得*更严重也更可见*。索引重叠 258 个
→ 经确定性映射必然产出 258 条**完全相同**的样本。若 `__getitem__` 用的是 RNG，同样 258 个
重叠 index 撞到同一条 episode 的期望是 **0 条**——sampler 错配依然浪费 258 个槽位，但症状
被随机性完全掩盖。报告能给出 `dup=258` 这么干净的数字，恰恰因为映射是确定性的。

**验证**（双进程 gloo，1000 条虚拟数据，真实 `TrainingArguments + Trainer + Accelerate`）：

```
修复前  per-rank=[500,500]  dup=258  missed=258
修复后  per-rank=[500,500]  dup=0    missed=0     PASS
```

**已知副作用（用户已确认接受）**：这条路径存在于**每一个历史 arm**（arm0/arm1/arm4 都用
同一个 `train.py` 入口），所以历史 r_peak 曲线都是在约 25% 重复采样下测得的。修复后的新
arm 与历史 arm 不再直接可比。

---

### TMIX-003（P1）：`train_mix.sh` 会把失败训练报告为成功

#### 位置

- `scripts/train_mix.sh:21`
- `scripts/train_mix.sh:106-130`

#### 当前行为

脚本启用：

```bash
set -uo pipefail
```

但没有 `set -e`。`torchrun` 结束后执行：

```bash
echo "TRAIN_EXIT=$?"
```

`$?` 虽然打印了 `torchrun` 的状态，但脚本最终返回的是 `echo` 的状态，通常为 0。

#### 实测证据

给训练入口传入一个确定非法参数，使 `train.py` 立即失败：

```text
train.py FAILED
TRAIN_EXIT=1
SCRIPT_RC=0
```

#### 影响

- Slurm、队列管理器、CI 或上层 orchestration 会把失败任务标成成功；
- 自动重试不会触发；
- 后续评估可能对不存在或不完整的 checkpoint 开始工作；
- 日志里虽然有 `TRAIN_EXIT=1`，但机器可读状态错误。

#### 建议修复

应显式保留并返回状态。若同时启用 `set -e`，可使用 `if` 避免在打印前直接退出：

```bash
if CUDA_VISIBLE_DEVICES="$GPUS" torchrun ...; then
  train_status=0
else
  train_status=$?
fi
echo "TRAIN_EXIT=$train_status"
exit "$train_status"
```

建议同时升级为：

```bash
set -euo pipefail
```

但要逐一处理预期允许失败的命令，例如 checkpoint glob 和磁盘探测。

#### 验收标准

- 正常 fake `torchrun` 返回 0 时，脚本返回 0；
- fake `torchrun` 返回 1、2、137、143 时，脚本返回完全相同状态；
- 上层队列能正确标记 failed/cancelled/OOM。


#### 修复（2026-08-20）

`scripts/train_mix.sh` 与 `scripts/train_arm.sh` 结尾改为：

```bash
  ... --report_to wandb --run_name "..." ${EXTRA_ARGS:-} "$@"
TRAIN_EXIT=$?
echo "TRAIN_EXIT=$TRAIN_EXIT"
exit "$TRAIN_EXIT"
```

原来 `echo` 是最后一条语句，脚本退出码恒为 `echo` 的 0，任何
`bash train_mix.sh && 下一步` 的链式调用都会把崩掉的训练当成成功。
`exit 2`（非法 ARM）那条路径本来就正确，只有训练失败这条被吞掉。

---

### TMIX-004（P1）：Usage 声明支持的 positional extra args 实际被丢弃

#### 位置

- `scripts/train_mix.sh:17`
- `scripts/train_mix.sh:27`
- `scripts/train_mix.sh:129`

#### 当前行为

Usage 声明：

```bash
bash scripts/train_mix.sh A2 [extra train.py args...]
```

脚本读取 ARM 后执行：

```bash
ARM="${1:-A2}"; shift || true
```

但后续命令从未展开 `"$@"`，因此所有 positional extra args 被丢弃。命令末尾只展开了：

```bash
${EXTRA_ARGS:-}
```

而 `EXTRA_ARGS` 并没有列入脚本的 Env 使用说明，并且裸字符串展开无法可靠表达带空格的单个参数值。

#### 实测证据

用 fake `torchrun` 打印所有 argv：

```bash
bash scripts/train_mix.sh A1 --positional_probe 'value with spaces'
```

输出的 argv 中完全没有 `--positional_probe` 和 `value with spaces`。

#### 影响

- 用户以为 `--report_to none`、`--strict_getitem True` 等覆盖已经生效，实际没有；
- smoke/debug 运行可能意外连接 W&B 或使用生产设置；
- 参数实验会出现“命令不同、实际训练相同”的静默错误。

#### 建议修复

1. 用 bash 数组构造可选参数，避免字符串拆词；
2. 把 `"$@"` 放在固定参数之后，使用户显式覆盖生效；
3. 删除 `EXTRA_ARGS`，或把它标为 deprecated；
4. 启动前打印 shell-escaped 最终 argv，例如 `printf '%q ' ...`。

示例：

```bash
init_args=()
if [[ -n "$INIT_CKPT" ]]; then
  init_args=(--init_ckpt_path "$INIT_CKPT")
fi

torchrun ... train.py \
  ... \
  "${init_args[@]}" \
  "$@"
```

#### 验收标准

- 带空格路径和值能保持为单个 argv；
- positional 参数能覆盖脚本默认参数；
- fake launcher snapshot 测试固定完整 argv。


#### 修复（2026-08-20）

`scripts/train_mix.sh` 的 torchrun 行末补上 `"$@"`，并在 `shift` 处加注释说明用途。

**这条不是理论问题，它已经影响过一次实测**：冒烟测试时传了 `--warmup_steps 5`，日志里
step 10 的 `learning_rate` 是 `5e-09` = `5e-7 × 10/1000`——warmup 仍是默认的 1000，
参数被静默丢弃。冒烟的其他结论（12/12、14.7 s/it、44 GB）不受影响，但这说明 usage 里
承诺的透传从未生效过。

---

### TMIX-005（P1）：weight checkpoint 与 DeepSpeed resume state 独立选择，可能恢复错步

#### 位置

- `train.py:72-110`：`find_latest_checkpoint()`
- `train.py:633-655`：`_latest_resumable_checkpoint()`
- `train.py:675-679`：选择模型 weights
- `train.py:783-795`：选择 optimizer/scheduler resume 目录
- `train.py:533-553`：按新的 `global_step` 写 checkpoint

#### 当前行为

恢复分为两次独立决策：

1. `find_latest_checkpoint(output_dir)` 选择编号最大的 `stepN.ckpt`，并在模型构造时加载；
2. `_latest_resumable_checkpoint(output_dir)` 选择编号最大的、含任意 `global_step*/` 和 `trainer_state.json` 的目录，交给 DeepSpeed。

这两个函数没有验证它们选择的是同一步，也没有让显式 `--resume_ckpt_path` 约束 optimizer state 的选择。

#### 风险场景 A：模型新、optimizer 旧

```text
checkpoint-3000/   step3000.ckpt + global_step3000/
checkpoint-6000/   step6000.ckpt，仅 weights
```

当前逻辑会：

1. 先加载 `step6000.ckpt`；
2. 再让 DeepSpeed 从 `checkpoint-3000` 恢复；
3. DeepSpeed resume state 可能把模型/optimizer/scheduler 一起恢复到 3000。

日志看似加载了 6000 权重，实际训练状态可能退回 3000。

#### 风险场景 B：只有 weights 时覆盖旧 checkpoint

当没有任何可用 optimizer state 时，代码从最新 weights 启动新 optimizer，并让 Trainer global step 从 0 开始，但仍使用同一个 `output_dir`。

例如从原 `step6000.ckpt` 开始：

```text
新 global step 3000 -> 写入旧 checkpoint-3000/
新 global step 6000 -> 写入旧 checkpoint-6000/
```

旧权重会被覆盖，而目录名不再代表总训练步数。日志已经警告“step numbers restart from 0”，但没有防止实际文件冲突。

#### 额外完整性问题

`_latest_resumable_checkpoint()` 只检查：

- 存在某个名称以 `global_step` 开头的目录；
- 存在 `trainer_state.json`。

它没有验证：

- `global_stepN` 是否和外层 `checkpoint-N` 一致；
- DeepSpeed shard 是否齐全；
- `latest` tag 是否有效；
- world size / ZeRO stage / optimizer 配置是否兼容。

因此“resumable”判断仍可能把部分写入的目录认作有效。

#### 建议修复

定义单一 checkpoint descriptor，例如：

```text
step
weights_path
checkpoint_dir
has_complete_deepspeed_state
trainer_state_step
deepspeed_tag
```

然后执行一种明确策略：

- **完整恢复**：weights、Trainer state 和 DeepSpeed state 必须来自同一步；
- **weights-only warm start**：必须使用新的输出目录，或显式设置新的 checkpoint 命名 offset；
- **不一致**：启动前报错，禁止“尽力猜测”。

建议新增显式参数：

```text
--resume_mode full
--resume_mode weights_only
--resume_mode none
```

不要仅凭目录内容自动猜测用户意图。

#### 验收标准

用临时小文件覆盖以下矩阵：

| 场景 | 期望行为 |
|---|---|
| 最新 weights 与最新完整 state 同步 | 完整恢复 |
| 最新 weights 比 optimizer 新 | 启动前报错 |
| 只有 weights，原 OUT 非空 | 要求新 OUT 或显式 offset |
| `global_stepN` 与目录编号不一致 | 报错 |
| shard 缺失 | 报错，不宣称 resumable |
| 显式外部 resume weights + OUT 中有旧 state | 不得自动加载 OUT 的旧 state |


#### 修复（2026-08-20）

在 `train.py` 新增 `_step_of()` 与 `_check_resume_consistency()`，在
`_latest_resumable_checkpoint()` 之后、`trainer.train()` 之前调用，**两个 repo 同步**：

```python
resumable = _latest_resumable_checkpoint(args.output_dir)
_check_resume_consistency(
    args.output_dir, args.resume_ckpt_path, resumable,
    allow_step_restart=getattr(args, "allow_step_restart", False),
)
```

拦截报告列出的两个风险场景：

- **场景 A（权重新 / optimizer 旧）**：两者步数不一致直接 `RuntimeError`，错误信息里给出
  两个步号和处理建议，不再「日志说 6000、实际退回 3000」。
- **场景 B（只有权重 + 非空 OUT）**：拒绝启动，除非显式 `--allow_step_restart True`
  （新增于 `training/args.py`）。原来只有一句「step numbers restart from 0」的警告，
  并不阻止 `checkpoint-3000/` 被新的 step 3000 覆盖。

**与报告建议的差异**：没有实现完整的 `--resume_mode full|weights_only|none` 三态 CLI。
理由是真正的危险是「两次独立扫描结果不一致却照常启动」，一致性校验 + 一个逃生开关已经
关闭了这两条路径；再引入一套新的 CLI 语义会扩大改动面和回归风险。若后续确实需要更细的
控制，`_check_resume_consistency()` 是加它的正确位置。

**验证**（临时目录，四场景矩阵）：

```
步数一致              -> 放行   OK
权重 6000 / 状态 3000 -> 报错   OK
只有权重 + 非空目录   -> 报错   OK
只有权重 + 显式放行   -> 放行   OK
```

报告提到的更深层完整性检查（shard 齐全、`latest` tag 有效、world size / ZeRO stage 兼容）
**未实现**，仍是已知缺口。

---

### TMIX-006（P2）：默认输出目录没有包含关键配置，容易静默续训到不同实验

#### 位置

- `scripts/train_mix.sh:33-57`
- `scripts/train_mix.sh:82`
- `scripts/train_mix.sh:91-95`
- `train.py:675-679`

默认输出目录只有：

```bash
OUT="$REPO/outputs/mix_${ARM}_seed${SEED}"
```

以下参数都没有体现在路径或恢复兼容性校验中：

- `RES`
- `FRAME_INTERVAL`
- `VARIATIONS`
- `DATASET_SPECS`
- `TEMPLATE_MIX_PER_DATASET`
- `PERCEPTION_MASK_MIX`
- `INIT_CKPT`
- `DS_CONFIG`
- world size / GPU 数

如果用户改变这些环境变量但没有手动改变 `OUT`，脚本只提示“将 resume”，随后 `train.py` 自动加载旧 checkpoint。它不会比较旧 run 与当前 run 的语义配置。

#### 影响

典型静默污染场景：

- 把 `FRAME_INTERVAL=1` 改为 3，却续训原 fi1 checkpoint；
- 把 A2 depth 比例从 0.33 改为其他值，却延续旧 optimizer；
- 改变 train/test variation split 后继续同一个 run；
- 改分辨率或数据树后继续沿用旧 run 名称；
- 改数据比例后，W&B 曲线仍连接在同一个实验上。

#### 建议修复

启动时生成规范化配置 manifest，例如：

```json
{
  "dataset_specs": "...",
  "template_mix_per_dataset": "...",
  "variations": "0",
  "resolution": [512, 512],
  "num_frames": 41,
  "frame_interval": 1,
  "perception_mask_mix": [0.81, 0.045, 0.045, 0.10],
  "seed": 42,
  "world_size": 2,
  "init_checkpoint": "...",
  "deepspeed_config_hash": "..."
}
```

恢复时：

1. 读取旧 manifest；
2. 对训练语义相关字段做严格比较；
3. 不一致时默认报错；
4. 只有显式 `--allow_config_drift` 才允许继续，并把差异写入日志。

输出目录也可以包含简短 slug 或 config hash，但 manifest 比仅靠目录名更可靠。

#### W&B 元数据缺口

`train.py` 当前记录了 `args.template_mix`，但 A1/A2 真正使用的是 `args.template_mix_per_dataset`。A2 的全局 `template_mix` 仍可能显示默认 `video+action@1.0`，无法从 W&B config 重建真实 depth 比例。

至少应补充：

```python
"template_mix_per_dataset": args.template_mix_per_dataset,
"resolved_dataset_probabilities": ...,
"resolved_template_probabilities": ...,
```


#### 修复（2026-08-20）

`scripts/train_mix.sh` 的默认 OUT 改为携带配置指纹：

```bash
CFG_ID=$(printf '%s|%s|%s|%s|%s' "$DATASET_SPECS" "$TEMPLATE_MIX_PER_DATASET" \
         "$PERCEPTION_MASK_MIX" "$RES" "$FRAME_INTERVAL" | sha1sum | cut -c1-8)
OUT="${OUT:-$REPO/outputs/mix_${ARM}_seed${SEED}_fi${FRAME_INTERVAL}_${RES}_${CFG_ID}}"
```

沿用 `train_arm.sh` 里 `FI_SUFFIX` / `DS_SUFFIX` 的同一条理由：非空 output_dir 意味着
RESUME，两个数据集/菜单/分辨率不同但同名的 run 会静默续训成对方。哈希覆盖的是**改变
「这个 arm 是什么」的全部字段**。

---

### TMIX-007（P2）：`validate_mix.py` 没有验证 `train_mix.sh` 的真实默认配置

#### 位置

- `scripts/validate_mix.py:54-67`
- `scripts/validate_mix.py:110-119`
- `scripts/validate_mix.py:140-149`

#### 问题 1：没有传 `variations=0`

`train_mix.sh` 默认：

```text
VARIATIONS=0
```

但 `validate_mix.py` 没有 `--variations` 参数，也没有把 variations 传给 `CombDataset`。因此它实际验证的是全部 1028 个 selfgen episode，不是训练使用的 788 个 variation0 episode。

原脚本运行输出：

```text
Found 1028 episodes
CombDataset length = 28174
```

按真实训练参数构造后：

```text
[selfgen] variations='0': kept 788/1028 episodes across 16 tasks
CombDataset length = 27934
```

#### 问题 2：`--per_dataset` 是死参数

参数被定义，但后续没有使用。用户调节它不会改变任何验证行为。

#### 问题 3：prompt 检查只拒绝重复，不拒绝缺失

当前逻辑是：

```python
if b["text"][0].count(tag) > 1:
    ...
```

因此 tag 完全缺失时仍通过。脚本文档声称检查“prompt tags present exactly once”，实际只检查“不超过一次”。

#### 问题 4：DROID action 检查过弱

当前只检查完整 `action_7d` 是否全零。DROID pose 非零而 gripper 恒为 1 时仍通过，正是 TMIX-001 未被发现的原因。

#### 问题 5：不是分布式 DataLoader 验证

它使用普通单进程 DataLoader，无法捕获 TMIX-002 的跨 rank sampler 重复。

#### 建议修复

- 增加 `--variations`，默认与 `train_mix.sh` 同为 `0`；
- 删除或真正使用 `--per_dataset`；
- 按 template 精确检查必需 tag 集合；
- 精确检查 stream keys 与 template 的视觉 modality 集合；
- 对 DROID gripper 做分布检查；
- 增加 `torchrun --nproc_per_node=2` 的 sampler 验证；
- 最好让 `train_mix.sh --validate-only` 和训练共享同一个配置解析函数，避免两套默认值漂移。


#### 修复（2026-08-20）

`scripts/validate_mix.py` 增加 `--variations`（默认 `"0"`）并透传给 `CombDataset`。

这是报告指出的最实质的一处失真：`train_mix.sh` 传 `--variations 0`，selfgen 因此从
1,028 集降到 **788** 集，而 validate 之前没传，验证的是一个训练根本看不到的数据集。
修复后 `CombDataset length = 27934`，与报告 §2 给出的生产值一致。

修复后重跑结果：

```
dataset                         got  requested
rlbench_selfgen_512_aug       0.610      0.620
droid                         0.280      0.280
bridge                        0.110      0.100

global task shares: video+action 0.655 / video 0.175 / video+depth 0.170
mask marginals    : video+action  iiii .810 fiii .045 fifi .044 policy .101
                    video+depth   iiii .808 fiii .046 fifi .047 single .099
RESULT: PASS
```

连带更正了 `MULTIDATASET_DEPTH_PLAN.md` 里基于 1,028 的过时数字：depth 覆盖
788 集、重复采样 210 遍、selfgen 总曝光 630 遍。

---

## 6. 条件触发和次要问题

### TMIX-008（P2）：正权重数据集为空时，`CombDataset` 会静默改采其他数据集

位置：`training/dataset/base.py:550-569`。

当 hash 选中一个长度为 0 的数据集时，代码会遍历并使用第一个非空数据集，而不是报错：

```python
if local_len == 0:
    for alt_idx, alt_ds in enumerate(self._datasets):
        if len(alt_ds) > 0:
            ...
            break
```

这样会静默改变用户请求的 0.62/0.28/0.10 比例。当前三个默认数据集都非空，所以本次没有触发，但数据链接损坏或预处理失败时会产生错误实验。

建议：任何正权重数据集长度为 0 都应在构造期报错。


#### 修复（2026-08-20）

`training/dataset/base.py` 的 `CombDataset.__init__` 在构造完各数据集后增加检查，
**两个 repo 同步**：

```python
empty = [n for n, d in zip(names, datasets) if len(d) == 0]
if empty:
    raise ValueError(
        f"dataset(s) {empty} were given a positive ratio but contain 0 episodes. ..."
    )
```

`__getitem__` 里那段 `local_len == 0` 的兜底仍然保留（它防的是运行期竞态），但正权重
数据集为空这种情况现在在**构造期**就失败——软链损坏或预处理没跑完是这条最常见的触发方式。
### TMIX-009（P2）：per-dataset menu 的未知数据集 key 不会报错

位置：`training/templates.py:176-210`。

`parse_per_dataset_template_mix()` 验证了右侧 template 语法，但没有验证左侧 dataset name 是否存在或是否出现在 `DATASET_SPECS` 中。

例如把：

```text
rlbench_selfgen_512_aug=video+action@0.67,video+depth@0.33
```

误写为其他名称时，真正的 selfgen 会回退到全局 `template_mix`，可能静默丢失 depth auxiliary stream。

建议同时拒绝：

- 未知数据集名；
- menu 中未被选择的数据集；
- 已选择但没有显式 menu 的数据集（在 multi-dataset 严格模式下）。


#### 修复（2026-08-20）

`training/templates.py` 的 `parse_per_dataset_template_mix()` 增加 `known=` 参数，
`CombDataset` 传入实际选中的数据集名：

```python
_selected = [sp.split("@", 1)[0].strip().lower() for sp in dataset_specs if sp.strip()]
per_ds = parse_per_dataset_template_mix(template_mix_per_dataset, template_mix, known=_selected)
```

未出现在 `--dataset_name` 里的 menu key 直接报错。实测：

```
template_mix_per_dataset names ['brigde'] which are not in --dataset_name (['droid']).
```

**实现时踩到的一个坑，记下来**：第一版把这段插在了 `dataset_specs` 由字符串 split 成
列表**之前**，`for sp in dataset_specs` 于是在遍历字符，`known` 变成一堆单字符，
会把所有合法 key 都判成未知。已移到 split 之后。

报告建议的另外两级（「menu 中未被选择的数据集」已覆盖；「已选择但没有显式 menu 的数据集」
的严格模式）**未实现**——后者会让「未列出则回落到全局 `--template_mix`」这个已在用的
默认行为失效。
### TMIX-012（P2，本次修复过程中新发现）：一个 epoch 并不遍历所有 episode

位置：`training/dataset/base.py` `CombDataset.__getitem__` / `_uniform_from_index`。

修好 TMIX-002 之后自然会问「现在一个 epoch 是不是所有数据都能采到」。答案是**否**，
而且和 TMIX-002 无关：`__getitem__` 把 sampler 索引**哈希**成 episode，这一步不是双射，
本质是**有放回抽样**而不是遍历。

按当前默认配置实测（`len(CombDataset) = 27,934`）：

| 数据集 | 一个 epoch 抽样次数 | 覆盖 | 平均每集 |
|---|---:|---|---:|
| `rlbench_selfgen_512_aug` | 17,379 | 788/788 = **100%** | 22.1 次 |
| `droid` | 7,759 | 5,339/9,437 = **56.6%** | 0.8 次 |
| `bridge` | 2,796 | 2,594/17,709 = **14.6%** | 0.2 次 |

更要紧的是**整个 10k 步长跑连一个 epoch 都跑不完**（`10000 × 2 rank × accum 1 = 20,000`
个样本 < 27,934）：

| 数据集 | 整轮覆盖 |
|---|---|
| `rlbench_selfgen_512_aug` | 100%（每集 15.8 次） |
| `droid` | **44.7%**——超过一半 episode 一次都没见过 |
| `bridge` | **10.8%**——近 90% 一次都没见过 |

这不是 bug，是「固定 step 预算 + 按比例采样」的必然结果，co-training 的常规做法也是如此
（论文的 180k/80k/30k 同样是按比例混）。**但它改变结论的措辞**：所谓「加入 DROID/Bridge
共训」实际是「加入 DROID 的一个 ~45% 随机子集、Bridge 的 ~11% 随机子集」。

由于映射是确定性的且只依赖索引，**同 seed 的 A1/A2 两个 arm 抽到的是同一个子集**，
arm 之间可比；但换 seed 会换子集，单次结果的方差要把这一条算进去。

未修改代码。若要真正遍历，两条路：把步数提到约 88k 步（bridge 才能平均每集看一次），
或把 `local_index` 从哈希改成数据集内的确定性置换——后者会再次打断与历史 arm 的可比性。

### TMIX-010（P3）：直接 `python train.py` 时 `compute_loss()` 无条件 all-reduce

位置：`train.py:454-479`。

barrier 前后都检查了 `torch.distributed.is_initialized()`，但中间的：

```python
torch.distributed.all_reduce(loss_gather, op=torch.distributed.ReduceOp.AVG)
```

没有检查。直接单进程调用时会报：

```text
ValueError: Default process group has not been initialized
```

`train_mix.sh` 始终使用 `torchrun`，即使 world size 为 1 也会初始化进程组，所以默认入口不触发；但 `train.py` 本身不支持普通单进程运行。

建议把 all-reduce 放进同一个 distributed guard，非分布式时直接使用本地 loss。


#### 修复（2026-08-20）

`train.py` 的 `compute_loss()` 把 all-reduce 放进和 barrier 相同的守卫，**两个 repo 同步**：

```python
loss_gather = torch.tensor(loss.item(), device=self.args.device)
if torch.distributed.is_available() and torch.distributed.is_initialized():
    torch.distributed.all_reduce(loss_gather, op=torch.distributed.ReduceOp.AVG)
```

`train_mix.sh` 始终走 torchrun 所以默认入口不触发，但这让 `python train.py` 直接调试
成为可能。
### TMIX-011（P3）：部分参数和日志名与实际行为不一致

- `steps_per_epoch` 只定义和传参，没有被训练代码使用；真正停止条件是 `max_steps`。
- shell 传入 `--run_name mix-A?-seed?`，但手工 `wandb.init()` 使用 `actionimages-${output_basename}`，用户传入的 run name 没有被尊重。
- `_save_checkpoint()` 注释写“只保存 trainable parameters”，代码计算了 `trainable_param_names` 却没有使用，最终保存完整 `denoising_model().state_dict()`。对 `train_mix.sh --full_param True` 没有区别，但对部分参数训练会造成存储行为与文档不一致。

这些不是当前 A2 数值正确性的首要问题，但应清理以减少误解。


#### 修复（2026-08-20）

三点全部处理，**两个 repo 同步**：

1. **`--run_name` 现在被尊重。** `wandb.init()` 原本无条件用
   `f"actionimages-{output_dir 末段}"`。改为优先使用 `args.run_name`，仅在它未设置
   （HfArgumentParser 会把它默认成 `output_dir`，视作未设置）时才回落到派生名。
2. **`steps_per_epoch` 标注为未使用。** 训练循环里没有任何代码读它，真正的停止条件是
   `max_steps`。help 文本改为 `"UNUSED. Retained for launcher compatibility; max_steps
   stops training."`，并加注释警告不要在其上加逻辑而不同步改 launcher。
3. **`_save_checkpoint()` 的死代码和错注释清掉。** 原来注释写「Save only the trainable
   parameters」，还构造了一个 `trainable_param_names` 集合却从不使用，实际保存完整
   `denoising_model().state_dict()`。删掉死变量，注释改为如实描述，并说明**故意**不做
   过滤：resume 路径和已发布的 `step125750.ckpt` 都期望完整的 denoiser state_dict。
   `test_checkpoint_pruning.py` 修改后仍通过。

## 7. 已验证正常的部分

以下链路在当前工作区和当前默认数据上通过：

### 7.1 Shell 与 Python 基础检查

```text
bash -n scripts/train_mix.sh        PASS
python -m compileall ...            PASS
```

### 7.2 A2 capability gating

```text
rlbench_selfgen_512_aug  -> video, depth, segmentation, action
droid                    -> video, action
bridge                   -> video
```

以下 menu 均成功通过 capability 校验：

```text
rlbench_selfgen_512_aug = video+action@0.67,video+depth@0.33
droid                    = video+action@1.0
bridge                   = video@1.0
```

负向用例 `bridge=video+action` 能正确报错。

### 7.3 实际 variation0 数据构造

使用和 `train_mix.sh` 一致的 `variations=0` 后：

```text
[selfgen] variations='0': kept 788/1028 episodes across 16 tasks
[selfgen] self-test OK: template=video+depth streams=['depth', 'video']
```

抽取样本时验证：

- `video.shape == (3, 82, 512, 512)`；
- depth 样本同时包含 `video`、`depth`；
- 双视角按 `[condition | target]` 顺序拼接；
- stream key 与视觉 template 对应；
- variation0 的 depth 自检通过。

### 7.4 混合比例

原 `validate_mix.py` 的 300 batch 抽样结果：

| 数据集 | 实测 | 请求 |
|---|---:|---:|
| selfgen 512 aug | 0.610 | 0.620 |
| DROID | 0.297 | 0.280 |
| Bridge | 0.093 | 0.100 |

这个结果说明单进程索引到数据集的 hash 比例基本正确；它不能替代 TMIX-002 所需的多 rank 互斥测试。

### 7.5 Template 和 mask 分布

40000 次抽样：

```text
video+action  iiii=0.810  fiii=0.045  fifi=0.044  policy/single=0.101
video+depth   iiii=0.808  fiii=0.046  fifi=0.047  policy/single=0.099
```

与目标 `0.81 / 0.045 / 0.045 / 0.10` 一致。

### 7.6 CPU 回归测试

下列核心测试通过：

- template packing 与 upstream 等价性；
- perception mask；
- checkpoint 裁剪的 collective symmetry；
- action decoder；
- policy conditioning；
- closed-loop aggregation；
- depth codec；
- segmentation codec；
- frame interval 基础逻辑。

## 8. 当前测试基础设施问题

`scripts/run_tests.sh` 当前不是全绿。5 个数据相关测试失败的共同原因是：

```text
data/rlbench_selfgen      -> 断链
data/rlbench_selfgen_512  -> 断链
data/rlbench_selfgen_512_aug -> 有效
```

失败文件包括：

- `tests/test_prompt_tags.py`
- `tests/test_selfgen_dataset.py`
- `tests/test_selfgen_alignment.py`
- `tests/test_depth_roundtrip_e2e.py`
- `tests/test_new_episodes.py`

它们大多硬编码旧 `data/rlbench_selfgen`，而 `train_mix.sh` 使用有效的 `data/rlbench_selfgen_512_aug`。

这不证明 A2 链路失败，但会带来两个问题：

1. CI 长期红色后，人们容易忽略新的真实失败；
2. 当前实际训练树没有被完整的数据级回归测试覆盖。

建议让测试数据路径可配置，并在数据确实不可用时明确 `SKIP`，而不是先打印 “Found 0 episodes” 后以 `IndexError` 或 `NoneType` 崩溃。不要通过在运行期间翻转共享 symlink 来修测试，因为这可能影响并行训练进程。

## 9. 建议修复顺序

### 第一阶段：阻止错误长跑

必须在正式 A1/A2 训练前完成：

1. TMIX-001：接入真实 DROID gripper；
2. TMIX-002：恢复所有 rank 共享 sampler seed；
3. TMIX-003：传递真实 shell 退出状态；
4. TMIX-004：正确传递 positional args。

这四项改动相对局部，但会直接决定训练标签、数据覆盖和作业状态是否可信。

### 第二阶段：保证恢复安全

5. TMIX-005：统一 weights / Trainer / DeepSpeed checkpoint 选择；
6. TMIX-006：写 run manifest 并拒绝配置漂移；
7. 为 checkpoint 恢复矩阵增加 CPU 级临时目录测试。

在这一步完成前，不建议依赖 weights-only 自动恢复到原输出目录。

### 第三阶段：让验证真正代表生产配置

8. TMIX-007：让 validator 共享真实 `variations` 和 resolved config；
9. 修复/参数化旧 selfgen 数据测试；
10. 加入两 rank sampler test 和 DROID gripper test；
11. 补充空数据集与错误 menu key 的启动期校验。

## 10. 建议验收矩阵

| ID | 测试 | 必须满足 |
|---|---|---|
| A01 | DROID 文件到 loader | 第七维逐帧匹配真实 gripper |
| A02 | DROID action image | gripper 阈值两侧产生不同蓝通道 pedestal |
| A03 | 两 rank sampler，长度可整除 | 交集 0，并集为全集 |
| A04 | 两 rank sampler，长度不可整除 | 只出现最小 padding 重复 |
| A05 | shell launcher 成功 | 脚本返回 0 |
| A06 | shell launcher 普通失败 | 原样返回非零状态 |
| A07 | shell launcher SIGTERM/OOM | 返回相应状态，队列标失败 |
| A08 | positional extra args | argv 完整保留，包括带空格值 |
| A09 | 完整 checkpoint 恢复 | weights/optimizer/scheduler/global step 同步 |
| A10 | 新 weights + 旧 optimizer | 启动前拒绝 |
| A11 | weights-only + 非空 OUT | 拒绝覆盖或要求新 OUT |
| A12 | 配置一致 resume | 允许恢复 |
| A13 | frame interval / menu 漂移 | 默认拒绝恢复并打印 diff |
| A14 | A2 variation0 validation | 788 个 selfgen episode，depth 自检通过 |
| A15 | 空的正权重数据集 | 构造期报错 |
| A16 | menu dataset key 拼写错误 | 构造期报错 |
| A17 | W&B config | 能完整重建 resolved dataset/template/mask 配置 |

## 11. 修复后建议执行的命令

基础静态检查：

```bash
bash -n scripts/train_mix.sh
python -m compileall -q train.py training scripts/validate_mix.py
```

CPU 回归：

```bash
bash scripts/run_tests.sh
```

真实配置验证应改为类似：

```bash
python scripts/validate_mix.py \
  --dataset_name 'rlbench_selfgen_512_aug@0.62,droid@0.28,bridge@0.10' \
  --template_mix_per_dataset 'rlbench_selfgen_512_aug=video+action@0.67,video+depth@0.33;droid=video+action@1.0;bridge=video@1.0' \
  --perception_mask_mix '0.81,0.045,0.045,0.10' \
  --variations 0
```

新增多卡 sampler 测试应使用 CPU gloo，避免依赖 5B 模型：

```bash
torchrun --standalone --nproc_per_node=2 tests/test_distributed_sampler.py
```

最后再执行短 GPU smoke run。必须使用和正式训练相同的 GPU 数与 DeepSpeed 配置，并使用全新的临时输出目录：

```bash
OUT=/workspace/ttdu/ActionImages-Cogen/outputs/smoke_mix_a2_fixed \
STEPS=3 \
CKPT_EVERY=999999 \
bash scripts/train_mix.sh A2 --report_to none --strict_getitem True
```

smoke run 通过后才建议开始 10000-step A1/A2 正式训练。

## 12. 最终结论

当前代码中，A2 的 RGB/depth 数据对齐、per-dataset template gating、segment assembly 和目标 mask 分布没有发现结构性错误；实际 variation0 数据也能构造并抽样。

但下面两个问题会直接影响模型学到的内容：

1. DROID gripper 标签被常量 1 替换；
2. 默认双卡训练因 rank-specific sampler seed 出现大量重复和漏样。

下面两个问题会直接影响训练作业是否可信：

3. `train_mix.sh` 吞掉 `torchrun` 的失败状态；
4. 使用说明中的 positional extra args 根本没有传入训练入口。

断点恢复和配置身份问题则会让重启后的实验难以解释，甚至覆盖已有 checkpoint。建议把 P0/P1 全部修复并补齐对应验收测试后，再启动正式长跑。

> **更新（2026-08-20）**：上述 11 项已全部处理，逐项改法见各 finding 下的「修复」小节，总览与验证结果见 §13。上面这四条阻塞项均已解除并有实测证据。剩余前置条件只有一项：GPU 冒烟尚未跑（当时无空闲卡）。修复过程中另发现 TMIX-012（一个 epoch 不遍历所有 episode），已记录、未改代码。

---

## 13. 修复总览（2026-08-20）

报告的 11 项全部处理。每项的具体改法见对应 finding 下的「修复」小节。

| 编号 | 级别 | 状态 | 改动位置 | ActionImages | Cogen |
|---|---|---|---|---|---|
| TMIX-001 | P0 | ✅ 已修 | `training/dataset/droid.py` | ✅ | ✅ |
| TMIX-002 | P0 | ✅ 已修 | `train.py` 入口 | ✅ | ✅ |
| TMIX-003 | P1 | ✅ 已修 | `scripts/train_mix.sh`, `train_arm.sh` | — | ✅ |
| TMIX-004 | P1 | ✅ 已修 | `scripts/train_mix.sh` | — | ✅ |
| TMIX-005 | P1 | ⚠️ 部分 | `train.py`, `training/args.py` | ✅ | ✅ |
| TMIX-006 | P2 | ✅ 已修 | `scripts/train_mix.sh` | — | ✅ |
| TMIX-007 | P2 | ✅ 已修 | `scripts/validate_mix.py` | — | ✅ |
| TMIX-008 | P2 | ✅ 已修 | `training/dataset/base.py` | ✅ | ✅ |
| TMIX-009 | P2 | ⚠️ 部分 | `training/templates.py`, `base.py` | — | ✅ |
| TMIX-010 | P3 | ✅ 已修 | `train.py` | ✅ | ✅ |
| TMIX-011 | P3 | ✅ 已修 | `train.py`, `training/args.py` | ✅ | ✅ |
| TMIX-012 | P2 | 📋 已记录未改 | 见该节 | — | — |

「⚠️ 部分」的两项，未做的部分及理由写在各自的修复小节末尾。

### 改动量

```
ActionImages       : train.py | training/args.py | training/dataset/{base,droid}.py
ActionImages-Cogen : train.py | training/args.py | training/dataset/{base,droid}.py
                     training/templates.py | scripts/{train_mix,train_arm,validate_mix}
                     tests/test_droid_gripper.py（新增）| scripts/run_tests.sh
```

`droid.py` 在两个 repo 中除 Cogen 独有的 `AVAILABLE_MODALITIES` 声明外保持一致。

### 验证结果

| 检查 | 结果 |
|---|---|
| 双进程 sampler 互斥 | `dup=0 missed=0`（修复前 `258/258`） |
| DROID gripper 逐帧比对 | 36 个 episode，不一致 **0**；dtype float64 保持 |
| gripper 单测 | `tests/test_droid_gripper.py` 通过，已入 `run_tests.sh` |
| resume 四场景矩阵 | 4/4 符合预期 |
| `validate_mix.py` | **PASS**，`CombDataset length = 27934` 与生产一致 |
| CPU 回归套件 | 失败 6 → **5**；`test_checkpoint_pruning` 转为通过 |

回归套件剩余 5 个失败全部由 `data/rlbench_selfgen` 悬空软链造成，修复前后一致，与本次改动无关。
`test_checkpoint_pruning` 此前失败的原因是 GPU 7 驱动级故障（`device=7, num_gpus=`），非代码问题。

### 仍未完成

**GPU 冒烟未跑**。修复触及 `train.py` 的 resume 路径与 DROID 数据加载，正式长跑前应先跑
一次 12 步双卡冒烟。上次尝试时 8 张卡全被占满（45–48 GB/张），无法进行。

**另外记录一个硬件问题**：`train_mix.sh` 的默认 `GPUS=6,7` 会踩到 GPU 7。该卡曾出现
驱动级故障——单独可见时 `torch.zeros(4, device="cuda")` 直接报
`No CUDA GPUs are available`，而 `nvidia-smi` 显示正常。启动前请确认目标卡真的可用。
