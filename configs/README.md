# configs/ —— 每份配置的来源

**抄配置只从 `ActionImages`（official-repro 分支）抄，不从 `ttd/` 抄。** 两边是不同项目的不同配方，
混用过一次，见下。

**注意：上游只有 `zero.json` 一个文件，`zero2_offload.json` 不是官方的**，是本地为了把 5B full-param
塞进 2×48GB 卡而写的。所以"以 `ActionImages` 为准"是个**约定**（单一来源），不是"更忠于上游"。

| 文件 | 来源 | `optimizer` 块 | offload |
|---|---|---|---|
| `zero.json` | 上游 `UMass-Embodied-AGI/ActionImages@291f71c` 原样，`scripts/train.sh` 用的就是它 | 无 | 无 |
| `zero2_offload.json` | 本地编写；从 `ActionImages/configs/zero2_offload.json` 拷来，逐字节相同 | **有** | 有 |
| `zero2_offload_hfoptim.json` | 本地编写；2026-08-10 从 `ttd/configs/zero2_offload.json` 拷来的旧版 | 无 | 有 |

## `zero2_offload_hfoptim.json` 为什么还留着

2026-08-11 把 `zero2_offload.json` 换成 `ActionImages` 那一版之前，`arm0__seed42` / `arm1__seed42` 已经用旧
配置跑了 4 小时。它们的 checkpoint 里 `global_step*/mp_rank_00_model_states.pt` 的 `lr_scheduler`
键是 `None`（scheduler 由 HF 管，状态在单独的 `scheduler.pt` 里）。换成新配置后 scheduler 改由
DeepSpeed 管，`deepspeed/runtime/engine.py:3094` 会无条件 `lr_scheduler.load_state_dict(None)`，
续跑当场 `AttributeError: 'NoneType' object has no attribute 'pop'`。

所以这两个 arm 中断后**必须**用旧配置续：

```bash
DS_CONFIG=./configs/zero2_offload_hfoptim.json bash scripts/train_arm.sh <arm> --seed 42
```

新开的 run 一律用 `zero2_offload.json`。等 arm0/arm1 跑完，这个文件可以删掉。

## 两者的实际差别：比想象的小

**运行时其实一样，两边最终都跑 `DeepSpeedCPUAdam`，显存也一样。** 没有 `optimizer` 块时 HF 建的是
`torch.optim.AdamW`，但只要开了 `offload_optimizer` 且没设 `zero_force_ds_cpu_optimizer: false`，
`accelerate/accelerator.py:1934-1942` 会调 `map_pytorch_optim_to_deepspeed()` 把它换成
`DeepSpeedCPUAdam`。两个 run 的日志里都有 `Loading extension module cpu_adam`，实测确认。

真正有差别的两处：

1. **checkpoint 布局** —— 有 `optimizer` 块时 scheduler 归 DeepSpeed 管，状态进 `global_step*/`，
   不写 `scheduler.pt`；没有时 scheduler 归 HF 管，写 `scheduler.pt`。这就是上面那个续跑陷阱的根源。
2. **`map_pytorch_optim_to_deepspeed()` 只搬运 `lr` 和 `weight_decay`**
   （`accelerate/utils/deepspeed.py:37`），**betas / eps 被丢掉**，改用 `DeepSpeedCPUAdam` 自己的默认值。
   目前无害（默认值恰好都是 0.9/0.999、1e-8，和 HF `TrainingArguments` 一致），但走这条路径时
   `--adam_beta2`、`--adam_epsilon` 会被**静默忽略**。有 `optimizer` 块的那条路径不存在这个问题
   （`"auto"` 从 `TrainingArguments` 如实解析）。这才是统一到带 `optimizer` 块那版的真正理由——
   跟"是否官方"无关，上游 `zero.json` 同样没有这个块。
