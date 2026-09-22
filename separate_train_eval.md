
# Goal

## 训练
1.我现在需要单独训练6个模型，RLBENCH 和 Maniskill 的分开训练，注意我所有的训练mask 方式都是A1, 每个最后训练到4k 步, 每2k step 可以保存一个checkpoint, 不需要保存优化器状态；优先训练RLBENCH 的模型

  arm1) MIX="video+action@0.4,depth+action@0.6"
        ACTION_MASK_MIX_DEFAULT=A1 ;;

  arm2) MIX="video+action@0.4,segmentation+action@0.6" 
        SEG_MODE_DEFAULT=scene_roles
        ACTION_MASK_MIX_DEFAULT=A1 ;;

  arm3) MIX="video+action@0.4,normal+action@0.6"
        ACTION_MASK_MIX_DEFAULT=A1 ;;


这三种arm, 各自RLBENCH 100% 的数据训练，或者Maniskill的 100% 单独分开训练,wandb offline, global batch size = 4, 你的机器应该可以


## Eval action closed loop

2.然后用每个4k step 的checkpoint, 分开eval closed loop,
RLBENCH 的先eval 这些task
close_box	close_drawer	close_microwave	toilet_seat_down, Meat on Grill
Maniskill 的先eval 2个 unseen tasks，再eval seen tasks 的seed variation
（gt 不行的已经被排除了吧）
每个task 的eval 都20 次


## others

要是代码有bug 或者不完整的地方你就修复补充了继续执行, 

# Details

你要在这台机器上**单独训练 6 个模型**（arm1 / arm2 / arm3 各自只用 RLBench，或各自只用
ManiSkill3），然后**只在它自己训练用的那个仿真器里**做闭环评测。

如果代码有问题或者不完整，你就 debug，把它写完整以后继续执行，并在汇报里说明改了什么（§10）。
这份文档是完整的约定：跑什么、怎么验证、怎么读数字，以及原机器上已经踩过的坑。
**动手之前先读 §9，引用任何数字之前先读 §7。**

---

## 0. 你要跑什么

**这 6 个 run 全部由你来跑：**

| run | 模板菜单 | 训练数据（100%，单一数据源） | 输出目录 |
|---|---|---|---|
| arm1 · RLBench | `video+action@0.4, depth+action@0.6` | `rlbench_selfgen_512_aug_wide` | `outputs/arm1_rlbenchonly_fromofficial_a1_4k` |
| arm2 · RLBench | `video+action@0.4, segmentation+action@0.6` | 同上 | `outputs/arm2_rlbenchonly_fromofficial_a1_4k` |
| arm3 · RLBench | `video+action@0.4, normal+action@0.6` | 同上 | `outputs/arm3_rlbenchonly_fromofficial_a1_4k` |
| arm1 · ManiSkill | 同 arm1 | `maniskill3` | `outputs/arm1_maniskillonly_fromofficial_a1_4k` |
| arm2 · ManiSkill | 同 arm2 | 同上 | `outputs/arm2_maniskillonly_fromofficial_a1_4k` |
| arm3 · ManiSkill | 同 arm3 | 同上 | `outputs/arm3_maniskillonly_fromofficial_a1_4k` |

六个 run 除了**菜单**和**数据源**之外完全相同：都从官方 ActionImages checkpoint warm start，
都是 `ACTION_MASK_MIX=A1`，都训 4000 步、每 2000 步存一个 checkpoint、不存优化器状态。

**这和 `ABLATION_HANDOFF.md` 不是一回事，千万别照那份做。** 那份是**混合训练**：一个 run 同时读
`rlbench@0.7 + maniskill3@0.3`。这里每个 run **只读一棵树**，所以一个 run 的结果只能归因于一个
仿真器的数据。那份文档里的数据配比（0.7/0.3）、"5,710 集 / 23 个任务"的合计、按每集曝光量均衡的
论证、以及 arm0 / arm4–arm8，**在这里全都不适用**。它里面关于环境、下载、坑的部分可以参考，
但凡是涉及数据混合的内容一律忽略。

**评测只在自己的仿真器里做**：RLBench 训出来的模型只评 RLBench，ManiSkill 训出来的只评
ManiSkill。不做跨仿真器评测，除非另行要求。

---

## 1. 把代码带过去，并证明完整到达

你收到的是整个 repo 的 zip。解压后，**先确认下面这些文件都在**——其中好几个是最近才写的，
如果 zip 打包得早，就会缺：

```bash
cd <repo>
for f in scripts/train_arm.sh scripts/launch_single_source.sh scripts/run_single_source_eval.sh \
         scripts/cl_json_complete.py eval/rollout.py eval/rollout_env.py eval/policy.py \
         eval/rollout_maniskill.py eval/rollout_env_maniskill.py scripts/maniskill3_gen.py \
         train.py training/templates.py training/dataset/base.py configs/zero2_offload.json; do
  [ -f "$f" ] || echo "MISSING $f"
done
grep -q 'WANDB_MODE:-}" != offline' scripts/train_arm.sh || echo "STALE train_arm.sh (offline-W&B fix missing)"
grep -q 'refusing to resume' eval/rollout_maniskill.py    || echo "STALE rollout_maniskill.py (resume missing)"
grep -q 'must be 4' scripts/launch_single_source.sh       || echo "STALE launch_single_source.sh"
echo VERIFIED-IF-NOTHING-ABOVE
```

**上面什么都没打印、只打印了最后一行，才算代码完整。** 缺任何一个就停下来汇报，不要自己凭记忆重写。

然后删掉 zip 里可能带过来的**原机器产物**——它们不是代码，而且有两个会造成真实的错误：

```bash
rm -rf wandb/ reports/ logs/          # 原机器的日志和结果，不要和你的混在一起
ls outputs/ 2>/dev/null               # 见下方说明
```

- `wandb/`：原机器的 W&B 日志目录。**它曾经在 wandb 包没装的情况下把 `import wandb` 伪装成成功**
  （Python 把这个目录当成了命名空间包），让预检查通过、训练却在 12.8 GB 权重加载完之后才崩。
- `outputs/`：如果里面有和上表**同名**的目录并且带 `checkpoint-*`，`train_arm.sh` 会**从它续训并
  悄悄忽略 warm start**。启动器会拒绝这种情况（§4），但最好一开始就确认 `outputs/` 里没有这 6 个名字。

---

## 2. 环境

**一个环境同时满足训练和两个仿真器的评测。** 建在持久化磁盘上，不要用 `/opt/conda/envs`——原
机器上那里的环境在一次容器重启里全丢了。

```bash
conda create -p /path/on/persistent/disk/envs/ttd -y python=3.10
PIP=/path/on/persistent/disk/envs/ttd/bin/pip

$PIP install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
$PIP install "numpy==1.26.4" transformers==4.57.3 diffsynth==1.1.9 "imageio[ffmpeg]" \
    safetensors einops sentencepiece protobuf modelscope ftfy trimesh matplotlib \
    opencv-python-headless scikit-image scipy lpips pandas accelerate cffi lxml defusedxml \
    "deepspeed==0.16.9" wandb
# 下面这组版本锁不是可选的，理由见本节末尾
$PIP install "diffusers==0.33.1" "huggingface-hub>=0.34.0,<1.0"
$PIP install -c <(printf 'torch==2.6.0\ntransformers==4.57.3\ndiffusers==0.33.1\nhuggingface-hub==0.36.2\nnumpy==1.26.4\n') xfuser
$PIP install --no-deps https://github.com/facebookresearch/vggt/archive/refs/heads/main.zip

# RLBench 侧：CoppeliaSim 4.1.0（PyRep 4.1 绑定的就是这个版本）
#   下载 CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz（downloads.coppeliarobotics.com/V4_1_0/）并解压
export COPPELIASIM_ROOT=/path/to/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
$PIP install https://github.com/stepjam/PyRep/archive/refs/heads/master.zip
$PIP install --no-deps https://github.com/stepjam/RLBench/archive/refs/heads/master.zip
$PIP install gymnasium pyquaternion natsort

# ManiSkill 侧
$PIP install --no-deps sapien==3.0.3 mplib==0.1.1 toppra h5py pytorch_kinematics \
    fast_kinematics transforms3d dacite tabulate tyro arm_pytorch_utilities pytorch_seed ipython
git clone https://github.com/haosulab/ManiSkill /path/to/ManiSkill && $PIP install --no-deps -e /path/to/ManiSkill

cd <repo> && $PIP install -e .          # 发行版名必须是 "actionimages"，train.py 会检查
```

系统包（原机器上全新容器里都缺）：

```bash
apt-get install -y libglib2.0-0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xkb1 libxcb-shape0 libxcb-xfixes0 \
    libsm6 libice6 libxext6 libxrender1 libfontconfig1 libdbus-1-3 libxi6 libxtst6 libgl1 \
    libglu1-mesa xvfb ffmpeg git
```

把环境变量写进一个 rc 文件，所有脚本都通过 `ENV_RC=` 读它：

```bash
cat > /path/to/env.rc <<'EOF'
export PATH=/path/on/persistent/disk/envs/ttd/bin:$PATH
export COPPELIASIM_ROOT=/path/to/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export PYTHONPATH=<repo>${PYTHONPATH:+:$PYTHONPATH}
unset CUDA_VISIBLE_DEVICES
EOF
```

**为什么锁这几个版本。** `training/models/wan_video_dit.py` 在模块顶层 import `xfuser`，所以
即使不用序列并行它也必须装。`xfuser` 要 `diffusers>=0.33`，而 `diffusers>=0.34` 又要
`huggingface-hub>=1.23`，`transformers==4.57.3` 不接受。`diffusers==0.33.1` 配
`huggingface-hub==0.36.2` 是唯一同时满足三者的点。`inference.py` 也在顶层 import `vggt`。
**`deepspeed` 和 `wandb` 训练必需**——原机器上一次启动就因为评测环境里没装它俩而失败。

**不要装 `flash-attn`。** 启动时的 `Flash Attention library "flash_attn" not found` 看着像警告，
其实不是问题：对这个模型的注意力形状，torch 自己的 SDPA 已经派发到 FlashAttention（实测
54.9 ms 对 memory-efficient 的 88.1 ms）。装它什么都得不到，却要编译很久。

**渲染。** RLBench 闭环在 `xvfb-run -a` 下用软件 GL 就能跑。SAPIEN 需要 Vulkan ICD；
`run_single_source_eval.sh` 会自动指向 SAPIEN 自带的那份，不用你管。

**冒烟测试，不需要 GPU 也不需要 checkpoint：**

```bash
source /path/to/env.rc && cd <repo>
xvfb-run -a python -u eval/rollout.py --gt-replay --tasks close_box --num-trials 1 \
  --variation 0 --res 256 --arm-action-mode planning --max-steps-factor 1.5 --tag smoke --no-record-video
# 期望末行: close_box  1  100.0 ...
python -c "import mani_skill.envs; from mani_skill.utils.registration import REGISTERED_ENVS; \
print('PlaceSphere-v1' in REGISTERED_ENVS)"      # 期望 True
```

---

## 3. 数据与权重

全部从 Hugging Face 下载。如果机器上有了就不用重复下载了。

```bash
cd <repo>/data
# RLBench 训练树：16 个任务，variation0 共 3,960 集，每个任务一个 tar（约 137 GB）
huggingface-cli download TingtingDu/rlbench_selfgen_512_aug_wide --repo-type dataset --local-dir /tmp/rlb_dl
mkdir -p rlbench_selfgen_512_aug_wide && cd rlbench_selfgen_512_aug_wide
for t in /tmp/rlb_dl/archives/*.tar; do tar xf "$t"; done; cd ..
# ManiSkill 训练树：7 个任务，variation0 = 训练用，variation1 = 已训练任务的新 seed（约 32 GB）
huggingface-cli download TingtingDu/maniskill3 --repo-type dataset --local-dir maniskill3
# ManiSkill 未见任务：pull_cube / place_sphere / lift_peg_upright 各 50 集（约 2.1 GB）
huggingface-cli download TingtingDu/maniskill3_heldout --repo-type dataset --local-dir maniskill3_heldout

cd <repo>
huggingface-cli download anyeZHY/ActionImages step125750.ckpt --local-dir checkpoints/official
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir checkpoints/Wan-AI/Wan2.2-TI2V-5B \
    --include "diffusion_pytorch_model*.safetensors" "models_t5_umt5-xxl-enc-bf16*.pth" \
              "Wan*_VAE.pth" "google/*" "*.json"
```

**验证**（数不对就重新下载，不要在残缺的树上训练）：

```bash
find data/rlbench_selfgen_512_aug_wide -name meta.json | wc -l   # 4200（含非 0 variation，训练只用 3960）
find data/maniskill3 -name meta.json | wc -l                     # 1925（1750 var0 + 175 var1）
find data/maniskill3_heldout -name meta.json | wc -l             # 150
ls -la checkpoints/official/step125750.ckpt                     # 12831256021 字节
```

- **RLBench 闭环评测不需要任何数据**：场景是 CoppeliaSim 现场生成的
  （`get_demos(live_demos=True)`）。Hub 上的 `rlbench_unseen_tasks_512_*` 是给离线感知评测用的，
  这里用不到。
- **ManiSkill 评测需要数据树**：rollout 会复现存档 episode 的场景**和相机**，模型的锚定帧就是
  通过这些相机渲出来的。
- `Wan2.2-TI2V-5B` 不是可选的：`.ckpt` 只有去噪器，没有它就没有 VAE 和文本编码器。
  日志里出现 `Loading models from: []` 就是这一步漏了，几分钟后会以
  `'list' object has no attribute 'endswith'` 崩掉。

**磁盘预算**：数据约 171 GB + 底座 32 GB + warm start 13 GB + 6 个 run × 2 个 checkpoint ×
12.8 GB = 154 GB，合计约 370 GB，**再留至少 40 GB 余量**。

---

## 4. 训练

每个 run 一条命令。启动器把原机器踩过的坑都封装了（见脚本头部注释）：

```bash
source /path/to/env.rc && cd <repo>
export ENV_RC=/path/to/env.rc
# 先 DRY_RUN：只做预检并打印将要执行的参数，不启动
DRY_RUN=1 bash scripts/launch_single_source.sh arm1 rlbench 0,1,2,3
# 然后真正启动（8 卡机器：两个 run 并行，各 4 卡）
bash scripts/launch_single_source.sh arm1 rlbench   0,1,2,3
bash scripts/launch_single_source.sh arm1 maniskill 4,5,6,7
```

8 卡机器建议分三轮，每轮同一个 arm 的两个数据源一起跑：arm1 → arm2 → arm3。
每轮约 20 小时（下方实测），三轮约 60 小时。

**参数，以及哪些不是自由选择：**

| 参数 | 值 | 为什么 |
|---|---|---|
| 数据 | RLBench: `rlbench_selfgen_512_aug_wide@1.0`；ManiSkill: `maniskill3@1.0` | **单一数据源 100%**，这正是和混合训练的区别 |
| `ACTION_MASK_MIX` | **A1** = (IIII 0.75, FIII 0, FIFI 0.05, policy 0.20) | 六个 run 统一；policy 模式把 loss 全放在动作段、只给单帧锚点，就是闭环 rollout 提供的条件 |
| `FRAME_INTERVAL` | RLBench **3**；ManiSkill **1** | ManiSkill 的 loader 把步长**写死为 1**（episode 只有 49–103 帧，步长 3 会跑出尾巴）。启动器传 1 只是为了让启动横幅说实话 |
| `SEG_MODE` | arm2 = `scene_roles`（9 角色稠密图） | arm1/arm3 没有分割模板，这个设置对它们不起作用 |
| warm start | 官方 `step125750.ckpt` | 问题是"在一个已经会动作的先验上加感知"，不是从零学 |
| 步数 / 存盘 | 4000 步，每 2000 步，**不存优化器状态** | 每个 checkpoint 12.8 GB；存优化器会变成 120 GB，原机器上两次就是存盘时把盘写满死掉的 |
| batch | **有效 batch = 4** = GPU 数 × 1 × 梯度累积 | 少于 4 卡时启动器自动设累积（2 卡→2），凑不出 4 就拒绝启动。batch 2 的 run 不可比 |
| lr | 5e-7，constant_with_warmup，warmup 1000 | 从别的数据训出的 checkpoint 冷启动，1000 步 warmup 正是为此设计的 |
| 其它 | seed 42，512²，41 帧，全参，bf16，DeepSpeed ZeRO-2 + CPU Adam offload | 与原机器所有 run 一致 |

**W&B 只能离线。** 启动器默认 `WANDB_MODE=offline`，run 写到 `outputs/wandb_offline/wandb/
offline-run-*`，不需要任何 key。训练结束后把这个目录带回有网络的机器，用
`wandb sync outputs/wandb_offline/wandb/offline-run-*` 上传（项目 `actionimages-cogen`）。
run 名是显式的（如 `arm1-rlbenchonly-fromofficial-a1-4k`）——`train_arm.sh` 的默认名会和旧 run
重名，两条曲线同名就是看错 run 的开始。

**健康启动长什么样。** 日志在 `outputs/single_source_logs/<arm>_<src>.log`，要看到全部五样：

```
mask axes: ... action_mask_mix=A1
frame_interval=3 -> window spans 6.0s ...        （ManiSkill 是 frame_interval=1 -> 2.0s）
weights: warm-start from .../checkpoints/official/step125750.ckpt
[selfgen] variations='0': kept 3960/4200 episodes across 16 tasks   （ManiSkill: 1750/1925 across 7 tasks）
{'loss': 0.0x, 'grad_norm': 0.x, ...}             loss 在 0.02–0.06，grad_norm 在 0.1–0.45
```

**在看到第一条 `{'loss': ...}` 之前，不要汇报"训练已启动"。** 原机器上有两次是看到了启动器打印的
PID、甚至看到了正确的启动横幅，结果训练在几十秒后因为环境问题退出（`TRAIN_EXIT=1`）。

**实测速度**（原机器，8 × RTX 6000 Ada 49 GB，两个 4 卡 run 并行）：16.5–18.2 s/步，
4000 步约 18–20 小时；每卡约 32.6 GB。两卡 + 累积 2 的配置实测 32–34 s/步、每卡 46–48 GB
（离 49 GB 很近，长 episode 可能 OOM），能不用就不用。

**续训。** 这几个 run 没有 watchdog。如果中途挂了，用同一条命令加 `RESUME=1` 重新启动：
`train_arm.sh` 会从最新的 `checkpoint-*` 续训。因为不存优化器状态，Adam 的矩会从零开始，
需要的 `--allow_step_restart` 启动器已经带上了。汇报里写清楚在第几步重启过、为什么。

---

## 5. 评测

**先跑 GT-replay 门，再跑模型。** GT replay 用同一个控制器执行每个场景自己的示教动作，不用模型：
一个任务在这里接近 0，就是执行 harness 的上限，**不是**模型的失败。每个模型格子都要对照它读。

```bash
source /path/to/env.rc && cd <repo> && export ENV_RC=/path/to/env.rc
bash scripts/run_single_source_eval.sh gt              # RLBench 用 CPU 并行；ManiSkill 占一点点 GPU
DRY_RUN=1 bash scripts/run_single_source_eval.sh models # 先看队列
setsid nohup bash scripts/run_single_source_eval.sh models > logs/eval.log 2>&1 < /dev/null &
```

默认用 8 张卡、每任务 20 次、评 `checkpoint-4000`。可以用 `GPUS="0 1 2 3"`、`STEP=2000`、
`ARMS=`、`SOURCES=` 覆盖。

**任务清单（写死在 `run_single_source_eval.sh` 里）：**

| 仿真器 | 块 | 任务 | 原机器 GT 天花板 |
|---|---|---|---|
| RLBench（variation0，训练树里没有） | 未见任务 | `close_box` `close_drawer` `close_microwave` `toilet_seat_down` `meat_on_grill` | 全部 20/20 |
| ManiSkill，**先评** | 未见任务（`maniskill3_heldout`） | `pull_cube` `place_sphere` | 20/20，20/20 |
| | | `lift_peg_upright` | **0/20** ← harness 天花板 |
| ManiSkill，**后评** | 已训练任务的新 seed（`maniskill3` variation1） | `pick_cube` `stack_cube` `push_cube` `pull_cube_tool` | 全部 20/20 |

- **`lift_peg_upright` 照跑，但它的模型数字不能当模型能力读。** 这个任务的运动规划解法是让夹爪
  只转约 35°、靠把 peg 抵着桌面翻转立起来，依赖接触力的瞬态；我们的 harness 执行的是笛卡尔末端
  位姿，表达不了这个。原机器上排查过：场景复现精确（首帧与存档视频 MAE 2.4/255）、动作与重算的
  TCP 位姿一致到 0.00 mm、位置跟踪 5 mm、每个路点重复 2/3/4 次都不行。汇报时把它的 GT 0/20
  写在旁边。
- **已训练任务的新 seed 只评这 4 个。** 其余三个训练任务是 harness 天花板：`peg_insertion_side`
  1/20、`plug_charger` 0/20（都是高精度插入），`stack_pyramid` 14/20（可选，要评就同时报 70% 的
  天花板）。
- **variation0 是训练集，永远不要拿来评测**；新 seed 用的是 `data/maniskill3` 的 variation1。

**协议——下面每一项都不是自由选择**（`run_single_source_eval.sh` 里已经写死）：

- `--frame-interval`：RLBench **3**，ManiSkill **1**（ManiSkill 的训练数据就是步长 1，用 3 评等于
  问模型一个它在这个源上从没见过的步长）。
- `--max-ik-fail-streak 5`：这是 harness 的策略而不是模型属性，它单独就能让一个数字变 26 个百分点，
  绝不能跨不同取值比较。
- `--skip-anchor-frames 4`：丢掉 chunk 开头那几帧（它们只是在重建机械臂当前所在的位姿），空出来的
  位置用斜坡补上，chunk 长度不变。
- `--prompt-tag-style explicit`：所有微调过的 arm 训练时 prompt 都带 `<video><action> ` 这类前缀。
- `--record-video`：每个 trial 存两个视频——仿真器实际执行的（`executed/`）和模型想象的
  （`generated/`）。没有它，一次失败就只剩一个标量，分不清是世界模型错了还是动作解码错了。

**场景在模型之间是配对的。** RLBench 用 `blake2b(task|variation|trial)` 播种（只取决于这三个，
和 checkpoint、进程无关）；ManiSkill 回放存档的第 `trial` 个 episode。所以同一个 `(task, trial)`
对每个模型都是同一个场景，模型之间可以做精确的配对 McNemar 检验。

**续跑。** 调度器跳过已经满 20 次的 job，可以反复执行。ManiSkill 的 rollout 会从已有的 JSON 续跑
（并校验协议一致，不一致就拒绝）；**RLBench 的 rollout 不能部分续跑**，被杀掉的 RLBench job 会从
第 0 次重来。

**时间。** 实测每个 trial：RLBench 平均约 400 s（`close_drawer` 306 s 到 `basketball_in_hoop`
575 s），ManiSkill 平均约 570 s，**`pull_cube_tool` 单个 trial 可达 27 分钟**。合计：3 个 RLBench
模型 × 5 任务 × 20 次 ≈ 33 GPU-小时，3 个 ManiSkill 模型 × 7 任务 × 20 次 ≈ 66 GPU-小时，
8 卡约 12–13 小时。一张卡一次只能跑一个 job（每个约 28–30 GB）。

---

## 6. 怎么读输出

结果在 `reports/closedloop_single_source/rollout_<tag>.json`：
- 模型：`ss_<arm>_<rlbench|maniskill>_4k_<task>`；GT 门：`ssgt_<rlbench|maniskill>_<task>`。
- 每个文件自带完整的 `args`，协议事后可查；每个 trial 有 `success / steps / replans /
  ik_fail_rate / stop_reason / scene_seed / prompt`。
- 视频在 `reports/closedloop_single_source/videos/<tag>/{executed,generated}/`，文件名带
  `SUCCESS` / `fail`。

一张总表：

```bash
python - <<'EOF'
import json, glob, os, collections
R = "reports/closedloop_single_source"
cell = {}
for p in glob.glob(f"{R}/rollout_*.json"):
    r = json.load(open(p)).get("results", [])
    if r: cell[os.path.basename(p)[8:-5]] = f"{sum(x['success'] for x in r)}/{len(r)}"
for src, tasks in (("rlbench", "close_box close_drawer close_microwave toilet_seat_down meat_on_grill"),
                   ("maniskill", "pull_cube place_sphere lift_peg_upright pick_cube stack_cube push_cube pull_cube_tool")):
    print(f"\n{src:26s} {'GT':>7s} {'arm1':>7s} {'arm2':>7s} {'arm3':>7s}")
    for t in tasks.split():
        print(f"{t:26s} {cell.get(f'ssgt_{src}_{t}','-'):>7s} " +
              " ".join(f"{cell.get(f'ss_{a}_{src}_4k_{t}','-'):>7s}" for a in ("arm1","arm2","arm3")))
EOF
```

**不同任务集上的聚合成功率不可比，即使 n 一样。** 原机器上中途有一次 arm0 那列显示 50.0%、
arm7 那列 19.6%，纯粹是因为当时 arm0 只跑完了最容易的 6 个任务。永远逐任务比，或者只在两边都
跑过的配对交集上比。

---

## 7. 原机器上的参考数字

**这些都是别的模型的数字**，只用来判断你的结果量级是否合理，**不能**当作你这 6 个 run 的对照组。
GT 天花板例外——它和模型无关，你的 GT 门应该和它基本一致；如果明显更低，先查 harness，别评模型。

RLBench，同 5 个任务，20 次：

| 任务 | GT | 官方 checkpoint（未微调） | arm7 · RLBench+ManiSkill 混合 · 4k |
|---|---|---|---|
| close_box | 20/20 | 5/20 | 6/20 |
| close_drawer | 20/20 | 16/20 | 11/20 |
| close_microwave | 20/20 | 15/20 | 17/20 |
| toilet_seat_down | 20/20 | 3/20 | 7/20 |
| meat_on_grill | 20/20 | 3/20 | 12/20 |

ManiSkill：

| 任务 | GT | arm7 · 混合 · 4k（20 次） | arm7 / arm0 · 纯 ManiSkill · 2k（≤10 次） |
|---|---|---|---|
| pull_cube | 20/20 | 2/20 | 进行中 |
| pick_cube | 20/20 | 0/20 | 0/10 · 0/10 |
| stack_cube | 20/20 | 0/20 | 0/10 · 0/10 |
| push_cube | 20/20 | 4/20 | 1/9 · 1/8 |

**如果你的 ManiSkill 模型在 `pick_cube` / `stack_cube` 上接近 0，这和原机器一致，不说明你的
环境坏了**——前提是你的 GT 门在这些任务上是 20/20。原机器上的诊断表明，模型在自己训练过的场景上
也做不好，失败方式以"连续 5 次 IK 失败"（输出了够不到的位姿）和超时为主，所以问题不在泛化。

---

## 8. 汇报什么

每个训练 run：
1. `checkpoint-2000/step2000.ckpt` 和 `checkpoint-4000/step4000.ckpt`（每个 12.8 GB）。
2. 启动日志，包括 §4 的五行——它们是"这个 run 确实在训它声称的东西"的证据。
3. 离线 W&B 目录 `outputs/wandb_offline/wandb/offline-run-*`（带回来 `wandb sync`）。
4. 任何重启：第几步、为什么。

评测：
1. `reports/closedloop_single_source/` 下全部 JSON（模型 + GT 门）。
2. 视频，至少所有失败的。
3. §6 那张总表。
4. 任何你跳过或改动的任务，以及它的 GT 门数字。

---

## 9. 已知的坑——下面每一条都在原机器上真实发生过

**9.1 `train_arm.sh` 不激活环境。** 它用 PATH 上的 `python`。原机器上第一次启动用到了 base
环境，`ModuleNotFoundError: No module named 'transformers'`；第二次是环境里没装 deepspeed。
启动器现在会在 repo 根目录下先做 import 预检，失败就拒绝启动。

**9.2 预检必须在 repo 根目录下做，而且要按 transformers 自己的方式问。** repo 根目录下的旧
`wandb/` 目录曾让 `import wandb` 在包没装的情况下"成功"，预检通过，训练却在
`WandbCallback requires wandb to be installed` 上崩掉。启动器调用的是
`transformers.integrations.is_wandb_available()`，和框架的判断完全一致。

**9.3 `nohup` 不够，要用 `setsid`。** 原机器上两个 6k 训练在第 ~1480 步因为
`SignalException: got signal: 15` 一起死掉：上层 shell 的进程组被拆掉，SIGTERM 传到了 torchrun。
启动器和调度器都用 `setsid nohup ... < /dev/null &`；启动后确认 `PGID == SID == PID`。

**9.4 输出目录非空 = 静默续训。** `train_arm.sh` 发现 `OUT` 里有 `checkpoint-*` 就会续训，
并忽略 `--init_ckpt_path`。启动器会拒绝，除非你显式 `RESUME=1`。

**9.5 `pgrep -f <脚本名>` 会匹配到只是"提到"这个脚本的 shell。** 很多工具调用都以
`bash -c '<整段命令>'` 运行，而且会作为孤儿进程残留；如果那段命令里写过某个脚本的 heredoc，它的
命令行里就原样带着脚本内容，`pgrep -f` 会一直匹配到它。原机器上两个"等某脚本结束再交接"的守护
脚本因此**永远等不到**，交接根本没发生，也没有任何报错。判断存活时跳过 `argv[1] == -c` 的进程
（读 `/proc/<pid>/cmdline`），并跳过自己的进程组。

**9.6 不要编辑正在运行的 shell 脚本。** bash 按字节偏移续读脚本，改了长度，正在跑的进程会从错位
的地方继续执行。要么先停掉，要么改一个副本再 `mv` 过去（rename 换的是 inode，正在跑的进程继续
读旧的）。

**9.7 两个调度器、两个队列会领同一个 job。** 原机器上在第一个调度器还在排空时手工起了第二个，两边
领了同一个任务、写同一个 JSON，其中一个把另一个已经记下的 16 次结果覆盖成了 0。一次只用一个队列。

**9.8 `nvidia-smi` 显示的 PID 和容器里的 PID 不是同一套。** 容器里 `nvidia-smi` 报的是宿主机 PID，
`ps` / `/proc` 里查不到，很容易被误判为"驱动泄漏了显存"。原机器上那 30 GB 其实是一个还活着的
孤儿 rollout 进程，杀掉就回收了。先按显卡找进程
（`/proc/<pid>/environ` 里的 `CUDA_VISIBLE_DEVICES`），不要急着 `nvidia-smi --gpu-reset`
（容器里通常也没权限）。

**9.9 从 source 过的 env 文件里继承 `CUDA_VISIBLE_DEVICES`。** 一个 env 脚本导出了
`CUDA_VISIBLE_DEVICES=0`，结果每个 worker 都只看到一张卡，绑到 GPU ≥ 1 的 job 报
`ordinal N is not valid`，其余全挤到 GPU 0。rc 里 `unset CUDA_VISIBLE_DEVICES`（上面的模板已经写了）。

**9.10 ManiSkill 的 `lazy_render`。** `ManiSkillRolloutEnv(lazy_render=True)` 让 env 用
`obs_mode="none"` 构建，GT replay 能快几分钟一集——但 `obs_mode` 同时决定相机渲染哪些纹理，于是
取像素时 `KeyError: 'rgb'`。原机器上 18 个模型 job 因此全部空转，因为这个开关只在
`--gt-replay --no-record-video`（唯一一条从不取像素的路径）上测过。现在它由
`(gt_replay and not record_video)` 推导，误用会直接报出原因。**改任何渲染路径之后，都要在真实的
模型 rollout 上验证，不能只在 GT replay 上验证。**

**9.11 日志里的 `Traceback` 不一定是错误。** xfuser 在 import 时会以 `DEBUG ... compat.py`
打印一段 `ImportError: cannot import name 'Flux2KleinPipeline'` 的 traceback，这是它自己捕获了的，
无害。用 grep 找错误时要排除 `compat.py`，否则监视脚本会被它骗到提前退出。

**9.12 磁盘。** 原机器的卷曾经在跑的过程中只剩 632 KB。始终保留 40 GB 以上。

---

## 10. 如果你改了任何东西

这 6 个 run 之间只应该有两个变量：模板菜单和数据源。如果你必须改任何其它东西——不同的 GPU 数、
不同的 batch、任何一个参数——**在汇报里明确写出来**，不要默默吸收，并且**同样地改到全部 6 个 run
上**，而不是只改需要它的那一个。在不同有效 batch 下训出来的 run 不在同一张表里。

如果你修了代码里的 bug，汇报里写清楚：哪个文件、改了什么、为什么、怎么验证的。
