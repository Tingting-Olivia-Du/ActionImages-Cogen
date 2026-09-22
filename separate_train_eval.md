
# Goal

## 训练
1.我现在需要单独训练6个模型，RLBENCH 和 Libero 的分开训练，各自RLBENCH 100% 的数据训练，或者Libero的 100%单独分开训练, 注意我所有的训练mask 方式都是A1; 
不需要保存优化器状态；wandb offline,不需要保存优化器状态；wandb offline,

  arm1) MIX="video+action@0.4,depth+action@0.6"
        ACTION_MASK_MIX_DEFAULT=A1 ;;

  arm2) MIX="video+action@0.4,segmentation+action@0.6" 
        SEG_MODE_DEFAULT=scene_roles
        ACTION_MASK_MIX_DEFAULT=A1 ;;

  arm3) MIX="video+action@0.4,normal+action@0.6"
        ACTION_MASK_MIX_DEFAULT=A1 ;;


### rlbench
rlbench 的最后训练到4k 步, 每2k step 可以保存一个checkpoint, 优先训练RLBENCH 的模型；global batch size = 4

### Libero
4 个 suite 混合,libero 的训练上限是10k 步，每1k 步保存一个模型,这个global batch size 根据你的机器你看着调整

validation 的使用方式你可以test 一下

libero 的数据集还在上传
https://huggingface.co/datasets/TingtingDu/libero
libero 单独训练我之前没有跑过，这部分你有很多需要修改调试的

## Eval action closed loop

2.然后用每个4k step 的checkpoint, 分开eval closed loop,
RLBENCH 的先eval 这些task
close_box	close_drawer	close_microwave	toilet_seat_down, Meat on Grill

Libero 的eval 应该在仿真器里是不是随机的？


## others

要是代码有bug 或者不完整的地方你就修复补充了继续执行, 

# Details

你要在这台机器上**单独训练 6 个模型**：arm1 / arm2 / arm3 各自只用 RLBench，或者各自只用
LIBERO（4 个 suite 混合）。然后**只在它自己训练用的那个仿真器里**做闭环评测。

两个数据源的完成程度不一样，先弄清楚：

- **RLBench 这一半是现成的**：数据、训练启动器、闭环 harness、GT 门都在原机器上跑通过。照做即可。
- **LIBERO 这一半只做到了"数据 + loader + 启动器"**：数据树已经生成好并验证过（§3），loader 已经
  注册、三个 arm 都在上面取样成功，启动器 `DRY_RUN` 通过。但 **LIBERO 从来没有真正训练过，也还
  没有闭环 harness**：`eval/rollout_libero.py` 不存在，要由你按 §5.3 的约定写出来，并先用 GT 门
  证明它是对的，再评模型。这部分预计需要你调试，这是正常的。

如果代码有问题或者不完整，你就 debug，把它写完整以后继续执行，并在汇报里说明改了什么（§10）。
这份文档是完整的约定：跑什么、怎么验证、怎么读数字，以及原机器上已经踩过的坑。
**动手之前先读 §9，引用任何数字之前先读 §7。**

---

## 0. 你要跑什么

**这 6 个 run 全部由你来跑，先跑 RLBench 的 3 个：**

| run | 模板菜单 | 训练数据（100%，单一数据源） | 输出目录 |
|---|---|---|---|
| arm1 · RLBench | `video+action@0.4, depth+action@0.6` | `rlbench_selfgen_512_aug_wide` | `outputs/arm1_rlbenchonly_fromofficial_a1_4k` |
| arm2 · RLBench | `video+action@0.4, segmentation+action@0.6` | 同上 | `outputs/arm2_rlbenchonly_fromofficial_a1_4k` |
| arm3 · RLBench | `video+action@0.4, normal+action@0.6` | 同上 | `outputs/arm3_rlbenchonly_fromofficial_a1_4k` |
| arm1 · LIBERO | 同 arm1 | `libero_spatial / object / goal / 10` 各 0.25 | `outputs/arm1_liberoonly_fromofficial_a1_10k` |
| arm2 · LIBERO | 同 arm2 | 同上 | `outputs/arm2_liberoonly_fromofficial_a1_10k` |
| arm3 · LIBERO | 同 arm3 | 同上 | `outputs/arm3_liberoonly_fromofficial_a1_10k` |

六个 run 都从官方 ActionImages checkpoint warm start，都是 `ACTION_MASK_MIX=A1`，都不存优化器状态。
两个数据源的训练长度不同：

| | 步数 | 存 checkpoint | 有效 batch |
|---|---|---|---|
| RLBench | 4000 | 每 2000 步 | **4**（写死，和原机器所有 run 可比） |
| LIBERO | 上限 10000 | 每 1000 步，**全部保留** | 默认 4，可以按你的机器调（§4.1），三个 arm 必须相同 |

**为什么 LIBERO 要训到 10k，并且要 validation。** LIBERO 从来没训过，4k 步 × batch 4 只相当于每条
demo 被抽到约 9 次，不一定收敛。所以 LIBERO 训到 10k，10 个 checkpoint 全部保留，再用 held-out
validation（§4.5）按一条**事先定好的规则**选**一个**步数，三个 LIBERO arm 都用这个步数做闭环评测。
在同一个数据源内部，3 个 arm 之间只有菜单这一个变量；比较的是同一数据源内的 arm1/2/3。

**LIBERO 是一个数据源，不是四个。** 4 个 suite 混在同一个 run 里训（配比和理由见 §4.2），评测时
按 suite 分开报。不要给每个 suite 单独训一个模型。

**这和 `ABLATION_HANDOFF.md` 不是一回事，千万别照那份做。** 那份是**混合训练**：一个 run 同时读
`rlbench@0.7 + maniskill3@0.3`。这里每个 run **只读一个仿真器的数据**，所以一个 run 的结果只能归因于
一个仿真器。那份文档里的数据配比、"5,710 集 / 23 个任务"的合计、arm0 / arm4–arm8，**在这里全都
不适用**。它里面关于环境、下载、坑的部分可以参考，但凡是涉及数据混合的内容一律忽略。
**这里完全不涉及 ManiSkill**：不用装 SAPIEN，不用下 ManiSkill 数据。

**评测只在自己的仿真器里做**：RLBench 训出来的模型只评 RLBench，LIBERO 训出来的只评 LIBERO。
不做跨仿真器评测，除非另行要求。

---

## 1. 把代码带过去，并证明完整到达

你收到的是整个 repo 的 zip。解压后，**先确认下面这些文件都在**。其中好几个是最近才写的，
如果 zip 打包得早，就会缺：

```bash
cd <repo>
for f in scripts/train_arm.sh scripts/launch_single_source.sh scripts/run_single_source_eval.sh \
         scripts/cl_json_complete.py eval/rollout.py eval/rollout_env.py eval/policy.py \
         eval/rollout_maniskill.py eval/rollout_env_maniskill.py scripts/libero_gen.py \
         scripts/run_libero_val.sh scripts/libero_val_table.py eval/eval_action.py \
         train.py training/templates.py training/dataset/base.py configs/zero2_offload.json; do
  [ -f "$f" ] || echo "MISSING $f"
done
grep -q 'WANDB_MODE:-}" != offline' scripts/train_arm.sh            || echo "STALE train_arm.sh (offline-W&B fix missing)"
grep -q '"libero_spatial", "libero_object"' training/dataset/base.py || echo "STALE base.py (LIBERO not registered)"
grep -q 'LIBERO_SPEC=' scripts/launch_single_source.sh               || echo "STALE launch_single_source.sh"
grep -q 'LIB_STEP' scripts/run_single_source_eval.sh                 || echo "STALE run_single_source_eval.sh"
grep -q 'CKPT_EVERY=1000' scripts/launch_single_source.sh            || echo "STALE launch_single_source.sh (LIBERO 10k)"
grep -q 'Binarise the GT too' eval/eval_action.py                    || echo "STALE eval_action.py (gripper_acc bug)"
echo VERIFIED-IF-NOTHING-ABOVE
```

**上面只打印了最后一行，才算代码完整。** 缺任何一个就停下来汇报，不要自己凭记忆重写。

`eval/rollout_maniskill.py` 不是拿来跑的，它是 §5.3 里你写 LIBERO harness 时的蓝本：同一套
receding-horizon 循环、同一种 JSON、同一种续跑。`scripts/libero_gen.py` 里有 LIBERO 训练数据的
相机和渲染代码，harness 必须复用它，不要重写。

然后删掉 zip 里可能带过来的**原机器产物**。它们不是代码，而且有两个会造成真实的错误：

```bash
rm -rf wandb/ reports/ logs/          # 原机器的日志和结果，不要和你的混在一起
ls outputs/ 2>/dev/null               # 见下方说明
```

- `wandb/`：原机器的 W&B 日志目录。**它曾经在 wandb 包没装的情况下把 `import wandb` 伪装成成功**
  （Python 把这个目录当成了命名空间包），让预检查通过，训练却在 12.8 GB 权重加载完之后才崩。
- `outputs/`：如果里面有和上表**同名**的目录并且带 `checkpoint-*`，`train_arm.sh` 会**从它续训并
  悄悄忽略 warm start**。启动器会拒绝这种情况（§4），但最好一开始就确认 `outputs/` 里没有这 6 个名字。

---

## 2. 环境

**一个环境同时满足训练和两个仿真器的评测**：LIBERO 的闭环 rollout 要在同一个进程里跑模型和
MuJoCo。环境建在持久化磁盘上，不要用 `/opt/conda/envs`，原机器上那里的环境在一次容器重启里全丢了。

```bash
conda create -p /path/on/persistent/disk/envs/ttd -y python=3.10
PIP=/path/on/persistent/disk/envs/ttd/bin/pip

$PIP install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
$PIP install "numpy==1.26.4" transformers==4.57.3 diffsynth==1.1.9 "imageio[ffmpeg]" \
    safetensors einops sentencepiece protobuf modelscope ftfy trimesh matplotlib \
    opencv-python-headless scikit-image scipy lpips pandas accelerate cffi lxml defusedxml \
    "deepspeed==0.16.9" wandb h5py
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

# LIBERO 侧：原机器上生成数据用的就是这组版本
$PIP install "mujoco==2.3.7" "robosuite==1.4.0" "bddl==1.0.1" "gym==0.25.2" \
    easydict future termcolor cloudpickle
git clone https://github.com/Lifelong-Robot-Learning/LIBERO /path/to/LIBERO
$PIP install --no-deps -e /path/to/LIBERO

cd <repo> && $PIP install -e .          # 发行版名必须是 "actionimages"，train.py 会检查
```

**LIBERO 第一次 import 会交互式提问**（"Do you want to specify a custom path for the dataset
folder? (Y/N)"）。放在后台脚本里，它会卡死或者读到 EOF 崩掉。先手工跑一次
`python -c "from libero.libero import benchmark"` 并回答 N，让它写出 `~/.libero/config.yaml`，
然后检查里面的 `bddl_files` / `init_states` / `assets` 都指向 `/path/to/LIBERO/libero/libero/`
下真实存在的目录。**LIBERO 的原始 HDF5 demo 你不需要**：训练数据已经渲染好了（§3），评测用的
init states 在 git 仓库里，不在 HDF5 里。

如果 robosuite / LIBERO 的某个依赖和训练栈冲突（重点看 numpy 和 torch 有没有被改版本），**以训练栈的
版本为准**。LIBERO 那边用 `--no-deps` 装，缺什么补什么，并在汇报里写明。

系统包（原机器上全新容器里都缺）：

```bash
apt-get install -y libglib2.0-0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xkb1 libxcb-shape0 libxcb-xfixes0 \
    libsm6 libice6 libxext6 libxrender1 libfontconfig1 libdbus-1-3 libxi6 libxtst6 libgl1 \
    libglu1-mesa libegl1 xvfb ffmpeg git
```

把环境变量写进一个 rc 文件，所有脚本都通过 `ENV_RC=` 读它：

```bash
cat > /path/to/env.rc <<'RC'
export PATH=/path/on/persistent/disk/envs/ttd/bin:$PATH
export COPPELIASIM_ROOT=/path/to/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export PYTHONPATH=<repo>${PYTHONPATH:+:$PYTHONPATH}
export MUJOCO_GL=egl          # LIBERO 离屏渲染
unset CUDA_VISIBLE_DEVICES
RC
```

**为什么锁这几个版本。** `training/models/wan_video_dit.py` 在模块顶层 import `xfuser`，所以
即使不用序列并行它也必须装。`xfuser` 要 `diffusers>=0.33`，而 `diffusers>=0.34` 又要
`huggingface-hub>=1.23`，`transformers==4.57.3` 不接受。`diffusers==0.33.1` 配
`huggingface-hub==0.36.2` 是唯一同时满足三者的点。`inference.py` 也在顶层 import `vggt`。
**`deepspeed` 和 `wandb` 训练必需**：原机器上一次启动就因为评测环境里没装它俩而失败。

**不要装 `flash-attn`。** 启动时的 `Flash Attention library "flash_attn" not found` 看着像警告，
其实不是问题：对这个模型的注意力形状，torch 自己的 SDPA 已经派发到 FlashAttention（实测
54.9 ms，memory-efficient 是 88.1 ms）。装它什么都得不到，却要编译很久。

**冒烟测试，不需要 GPU 也不需要 checkpoint：**

```bash
source /path/to/env.rc && cd <repo>
xvfb-run -a python -u eval/rollout.py --gt-replay --tasks close_box --num-trials 1 \
  --variation 0 --res 256 --arm-action-mode planning --max-steps-factor 1.5 --tag smoke --no-record-video
# 期望末行: close_box  1  100.0 ...
python - <<'PY'
import os
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
s = benchmark.get_benchmark_dict()["libero_spatial"](); t = s.get_task(0)
env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"),
                         t.problem_folder, t.bddl_file), camera_heights=128, camera_widths=128)
env.seed(0); env.reset(); env.set_init_state(s.get_task_init_states(0)[0])
print("LIBERO OK:", t.language, len(s.get_task_init_states(0)), "init states")   # 期望 50
PY
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
# LIBERO 训练树：4 个 suite，每个 suite 是仓库里的一个顶层目录（约 85 GB）
huggingface-cli download TingtingDu/libero --repo-type dataset --local-dir .
#   -> data/libero_spatial  data/libero_object  data/libero_goal  data/libero_10

cd <repo>
huggingface-cli download anyeZHY/ActionImages step125750.ckpt --local-dir checkpoints/official
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir checkpoints/Wan-AI/Wan2.2-TI2V-5B \
    --include "diffusion_pytorch_model*.safetensors" "models_t5_umt5-xxl-enc-bf16*.pth" \
              "Wan*_VAE.pth" "google/*" "*.json"
```

**LIBERO 数据集可能还没传完。** 写这份文档时 `TingtingDu/libero` 正在上传，`libero_spatial` 和
一部分 `libero_object` 还不在上面。**下载后必须数一遍**。数不对就等一会儿再下一次
（`huggingface-cli download` 会跳过已有文件，只补缺的）。**绝不要在残缺的树上训练**：loader 照样
会启动，只是悄悄少了一个 suite。

**验证：**

```bash
find data/rlbench_selfgen_512_aug_wide -name meta.json | wc -l   # 4200（含非 0 variation，训练只用 3960）
for s in libero_spatial libero_object libero_goal libero_10; do
  printf "%-16s var0=%s var1=%s\n" $s \
    $(find data/$s -path '*variation0*' -name meta.json | wc -l) \
    $(find data/$s -path '*variation1*' -name meta.json | wc -l)
done
# 期望：spatial 450/50，object 450/50，goal 447/50，libero_10 450/50（合计 1997，每个 suite 10 个任务）
ls -la checkpoints/official/step125750.ckpt                     # 12831256021 字节
```

**LIBERO 训练树是什么。** 它由 `scripts/libero_gen.py` 从官方 LIBERO demo 重新渲染而来，不是
官方 HDF5 里那两个固定相机的图：

- **相机是随机的**：每集 4 个视角，分布和 RLBench / ManiSkill 训练树一样（半径 0.55–0.95 m，
  仰角 42–72°，方位 ±150°，FOV 55°，看向场景物体中心），每集重新采样。LIBERO 自带的
  `agentview` 等相机被挪到这些位置上使用，名字改叫 `view1..4`。
- **每个视角有 4 种模态**：rgb、深度、法向、分割。深度单位是**米**，不是 MuJoCo 的归一化 z-buffer。
  分割带有 `scene_roles` 需要的角色表，角色由 BDDL 里的目标物体推出。相机内外参按 RLBench
  约定写在 meta 里。
- **动作标签**是每帧的 TCP 位姿 `[x,y,z, qx,qy,qz,qw, openness]`：世界系，取自
  `robot0_eef_pos/quat`，`openness = (g0−g1)/0.08`，1 表示全开。帧和标签都来自回放的同一个
  `states[t]`，所以严格对齐。
- **过滤**：去掉了 no-op 帧（openvla 的做法），只保留回放后真正成功的 demo（2000 条里丢了 3 条）。
- **划分**：variation0 用于训练（每任务 45 集）；variation1 是每个任务第 10、20… 集留出来的
  5 集，训练不用。这 5 集是给离线感知评测留的，闭环评测不用它们（§5.2）。
- **帧率与长度**：20 Hz。episode 长度（帧，p5 / 中位 / p95）：spatial 91/124/164，
  object 123/146/179，goal 85/110/203，libero_10 185/262/410。

**闭环评测需要哪些数据：**

- **RLBench 不需要任何数据**：场景由 CoppeliaSim 现场生成（`get_demos(live_demos=True)`）。
- **LIBERO 基本也不需要这棵树**：场景来自 benchmark 自带的固定 init states（§5.2）。只有 LIBERO
  的 GT 门要读 variation0 里的 demo（`meta.json`、`actions.npy` 和场景 XML）。
- **`Wan2.2-TI2V-5B` 不是可选的**：`.ckpt` 只有去噪器，没有它就没有 VAE 和文本编码器。
  日志里出现 `Loading models from: []` 就是这一步漏了，几分钟后会以
  `'list' object has no attribute 'endswith'` 崩掉。

**磁盘预算**：数据约 222 GB（RLBench 137 + LIBERO 85）+ 底座 32 GB + warm start 13 GB +
checkpoint：RLBench 3 个 run × 2 个 × 12.8 GB = 77 GB，LIBERO 3 个 run × 10 个 × 12.8 GB = 384 GB。
合计约 730 GB，**再留至少 40 GB 余量**。LIBERO 的 checkpoint 在 validation 选完步数、闭环评测做完
之前一个都不能删。盘不够就先停下来汇报，不要自己改存盘间隔。
RLBench 的 tar 解压完就可以删掉 `/tmp/rlb_dl`。

---

## 4. 训练

### 4.1 启动

每个 run 一条命令。启动器把原机器踩过的坑都封装了（见脚本头部注释）：

```bash
source /path/to/env.rc && cd <repo>
export ENV_RC=/path/to/env.rc
# 先 DRY_RUN：只做预检并打印将要执行的参数，不启动
DRY_RUN=1 bash scripts/launch_single_source.sh arm1 rlbench 0,1,2,3
# 然后真正启动（8 卡机器：两个 run 并行，各 4 卡）
bash scripts/launch_single_source.sh arm1 rlbench 0,1,2,3
bash scripts/launch_single_source.sh arm2 rlbench 4,5,6,7
```

**顺序：RLBench 优先。** 8 卡机器上两个 4 卡 run 并行。RLBench run 约 20 小时（4k 步）。
LIBERO run 在 batch 4 下约 47 小时（10k 步，按同样的 ~17 s/步估算）：

| 时段 | GPU 0–3 | GPU 4–7 |
|---|---|---|
| 0–20 h | arm1 · RLBench | arm2 · RLBench |
| 20–40 h | arm3 · RLBench | arm1 · LIBERO（20–67 h；先过 §4.3 的调试 run） |
| 40–87 h | arm2 · LIBERO | ↑ |
| 67–114 h | ↑ | arm3 · LIBERO |

合计约 4.5–5 天。RLBench 的 3 个模型在第 40 小时左右就全部训完，可以插空先评（§5）。LIBERO 的
GT 门只要 CPU 和 EGL，用前 20 小时写 §5.3 的 harness 并过 GT 门。

**LIBERO 的 batch 你可以调，规则如下。** `BATCH=<n>` 只对 LIBERO 生效，RLBench 永远是 4。有效 batch =
GPU 数 × 1 × 梯度累积，启动器自动算累积，除不尽就拒绝启动。
- 单卡每步只放 1 个样本（每卡约 32.6 GB，49 GB 卡上放不下 2 个），所以加大 batch 只有两条路：
  更多卡，或者梯度累积。梯度累积会让每步时间成倍增加：batch 8 用 4 卡要 ~34 s/步，10k 步约 94 小时。
- 所以**默认就用 4**。只有当你的机器每个 run 能分到 8 张卡（batch 8 不用累积，每步时间基本不变）
  时才考虑 8。
- 学习率不随 batch 自动缩放，保持 5e-7。改了 batch 就在汇报里写明，三个 LIBERO arm 必须一样。
- 调试 run 以及 batch 设置，在 arm1 · LIBERO 正式开始前定下来，之后不再改。

```bash
bash scripts/launch_single_source.sh arm1 libero 4,5,6,7                 # batch 4（默认）
BATCH=8 bash scripts/launch_single_source.sh arm1 libero 0,1,2,3,4,5,6,7 # 只有 8 卡给一个 run 时
```

**参数，以及哪些不是自由选择：**

| 参数 | 值 | 为什么 |
|---|---|---|
| 数据 | RLBench: `rlbench_selfgen_512_aug_wide@1.0`；LIBERO: 四个 suite 各 `@0.25` | **单一仿真器 100%**，这正是和混合训练的区别 |
| `VARIATIONS` | `0` | LIBERO 的 variation1 是留出集，RLBench 的非 0 variation 也不参与训练 |
| `ACTION_MASK_MIX` | **A1** = (IIII 0.75, FIII 0, FIFI 0.05, policy 0.20) | 六个 run 统一；policy 模式把 loss 全放在动作段、只给单帧锚点，这正是闭环 rollout 提供的条件 |
| `FRAME_INTERVAL` | RLBench **3**；LIBERO **1** | 见下方 |
| `SEG_MODE` | arm2 = `scene_roles`（9 角色稠密图） | arm1/arm3 没有分割模板，这个设置对它们不起作用 |
| warm start | 官方 `step125750.ckpt` | 问题是"在一个已经会动作的先验上加感知"，不是从零学 |
| 步数 / 存盘 | RLBench 4000 步、每 2000 步存；LIBERO 10000 步、每 1000 步存；全部保留，**不存优化器状态** | 每个 checkpoint 12.8 GB；存优化器会变成 120 GB，原机器上两次就是存盘时把盘写满死掉的。LIBERO 用哪一步由 §4.5 选 |
| batch | RLBench **有效 batch = 4**；LIBERO 默认 4，可用 `BATCH=` 调 | 有效 batch = GPU 数 × 1 × 梯度累积，启动器自动设累积（2 卡→2），凑不出就拒绝启动 |
| lr | 5e-7，constant_with_warmup，warmup 1000 | 从别的数据训出的 checkpoint 冷启动，1000 步 warmup 正是为此设计的 |
| 其它 | seed 42，512²，41 帧，全参，bf16，DeepSpeed ZeRO-2 + CPU Adam offload | 与原机器所有 run 一致 |

**LIBERO 为什么必须是 `FRAME_INTERVAL=1`。** 一个样本要 41 帧。步长 3 时要 121 帧，而
`libero_spatial` 有 44% 的 episode、`libero_goal` 有 56.5% 的 episode 不到 121 帧，窗口会跑出
episode 末尾。loader 的注册（`training/dataset/base.py`）已经把 LIBERO 写死为 1，启动器传 1
只是为了让启动横幅说实话。步长 1、20 Hz 下，一个窗口覆盖 2 秒。评测也必须用 1（§5）。

### 4.2 LIBERO 四个 suite 怎么配

**用 0.25 / 0.25 / 0.25 / 0.25，已经写在启动器里（`LIBERO_SPEC`）。** loader 的采样方式是：
先按配比选 suite，再在 suite 里均匀选一个 episode，再在 episode 里随机选一个窗口。所以配比决定的是
**每个 suite 得到多少次抽样**。

| suite | 训练集 | 总帧数 | 41 帧窗口数 | 窗口占比 | 0.25 下的抽样数，batch 4：4k 步 / 10k 步 |
|---|---|---|---|---|---|
| libero_spatial | 450 | 62k | 42k | 16% | ≈ 4,000 / 10,000（每集 ≈ 9 / 22 次） |
| libero_object | 450 | 74k | 54k | 21% | ≈ 4,000 / 10,000 |
| libero_goal | 447 | 63k | 43k | 17% | ≈ 4,000 / 10,000 |
| libero_10 | 450 | 138k | 118k | 46% | ≈ 4,000 / 10,000 |

（帧数和窗口数按包含 variation1 的全部 episode 统计，比例和只算 variation0 一样。）

- **等权 = 每条 demo 等权。** 四个 suite 的 episode 数几乎一样（450/450/447/450），0.25 就等于
  在全部 1,797 条 demo 上均匀抽，和 LIBERO 常规的做法一致。
- **不按帧数配。** 按帧数（或窗口数）配会让 `libero_10` 拿到 46%，其余三个 suite 只剩每个
  约 2,400 次抽样。评测按 suite 分开报、四个 suite 同等重要，所以不值得为长 episode 牺牲另外三个。
- **代价要知道。** 等权下，`libero_10` 每条 demo 的窗口覆盖率最低：10k 步时每集约 22 次抽样 × 41 帧，
  而一集中位长度 262 帧。如果最后只有 `libero_10` 明显落后，这是第一个可疑点。但**不要**为此只给
  这一个 run 改配比：六个 run 之间只允许菜单和数据源两个变量（§10）。
- **三个 LIBERO run 用完全相同的配比。**

### 4.3 LIBERO 第一次训练：先跑一个调试 run

LIBERO 从来没有真正训练过，loader 只做过取样测试。**正式启动 10000 步之前**，先跑一个几十步的
调试 run，它写到单独的 `*_debug` 目录和 W&B 名字，不会和正式 run 混：

```bash
DEBUG_STEPS=30 bash scripts/launch_single_source.sh arm2 libero 4,5,6,7
tail -f outputs/single_source_logs/arm2_libero_debug.log
```

用 arm2 调试，因为它依赖的东西最多（`scene_roles` 分割）。要确认：

1. 四个 suite 各打印一行 `[selfgen] variations='0': kept 450/500 episodes across 10 tasks`
   （`libero_goal` 是 `447/497`），而且 arm2 每个 suite 都有 `scene_roles OK`。
2. 出现 `frame_interval=1`、`action_mask_mix=A1`、`warm-start from .../step125750.ckpt`。
3. `{'loss': ...}` 的量级和 RLBench run 相近（0.02–0.06），`grad_norm` 不是 nan/inf，
   每步时间和 RLBench run 相近（约 17 s，同样是 512²、41 帧）。
4. 第 30 步的 checkpoint 能存下来（约 12.8 GB）。
5. 用这个 checkpoint 跑一次 validation 的单个 suite，确认 §4.5 那条链路也是通的：
   `bash scripts/run_libero_val.sh` 只认正式目录，所以直接调用
   `python eval/eval_action.py --ckpt outputs/arm2_liberoonly_fromofficial_a1_10k_debug/checkpoint-30/step30.ckpt --data data/libero_spatial --episodes <一个 variation1 episode> --res 512 --frame_interval 1 --template video+action --protocol i2va --no-video --tag debug`。

过了之后**删掉** `outputs/arm2_liberoonly_fromofficial_a1_10k_debug` 和它对应的离线 W&B run，
再正式启动。loader 在原机器上已经验证到的程度：三个 arm 都能在 4 个 suite 上构建（1,797 集），
取出的样本是 `(3, 82, 512, 512)` 的模态流加 `(41, 7)` 的动作，prompt 形如
`<scene-seg><action> pick up the black bowl ...`。**没有验证过的**：多卡、DeepSpeed、长时间
训练下的吞吐和显存，以及 LIBERO 的动作分布对损失的影响。

### 4.4 W&B、健康启动、续训

**W&B 只能离线。** 启动器默认 `WANDB_MODE=offline`，run 写到 `outputs/wandb_offline/wandb/
offline-run-*`，不需要任何 key。训练结束后把这个目录带回有网络的机器，用
`wandb sync outputs/wandb_offline/wandb/offline-run-*` 上传（项目 `actionimages-cogen`）。
run 名是显式的（如 `arm1-rlbenchonly-fromofficial-a1-4k`、`arm1-liberoonly-fromofficial-a1-10k`）。
`train_arm.sh` 的默认名会和旧 run 重名，两条曲线同名就是看错 run 的开始。

**健康启动长什么样。** 日志在 `outputs/single_source_logs/<arm>_<src>.log`，要看到全部五样：

```
mask axes: ... action_mask_mix=A1
frame_interval=3 -> ...                              （LIBERO 是 frame_interval=1）
weights: warm-start from .../checkpoints/official/step125750.ckpt
[selfgen] variations='0': kept 3960/4200 episodes across 16 tasks   （LIBERO：四行，见 §4.3）
{'loss': 0.0x, 'grad_norm': 0.x, ...}                RLBench 上 loss 0.02–0.06，grad_norm 0.1–0.45
```

**在看到第一条 `{'loss': ...}` 之前，不要汇报"训练已启动"。** 原机器上有两次是看到了启动器打印的
PID、甚至看到了正确的启动横幅，结果训练在几十秒后因为环境问题退出（`TRAIN_EXIT=1`）。

**实测速度**（原机器，RLBench，8 × RTX 6000 Ada 49 GB，两个 4 卡 run 并行）：16.5–18.2 s/步，
4000 步约 18–20 小时，每卡约 32.6 GB。两卡 + 累积 2 的配置实测 32–34 s/步、每卡 46–48 GB
（离 49 GB 很近，可能 OOM），能不用就不用。LIBERO 没有实测数字（batch 4、10k 步按同样速度约 47 小时），把你测到的写进汇报。

**续训。** 这几个 run 没有 watchdog。如果中途挂了，用同一条命令加 `RESUME=1` 重新启动，
`train_arm.sh` 会从最新的 `checkpoint-*` 续训。因为不存优化器状态，Adam 的矩会从零开始，
需要的 `--allow_step_restart` 启动器已经带上了。汇报里写清楚在第几步重启过、为什么。

### 4.5 LIBERO 的 held-out validation 与选步数

**为什么要它。** diffusion loss 太吵，看不出动作收敛没有；闭环评测又太贵（§5.5），不能对 10 个
checkpoint 都跑。所以在 LIBERO 的 **variation1 留出集**上测离线动作误差：给一帧真实锚点，让模型
生成 41 帧的 video+action，解码成 TCP 轨迹，和真值比位置误差、姿态误差、夹爪准确率。

```bash
# "训完" = 这个 run 跑到 10000 步、train_arm.sh 退出、它的 4 张卡空出来。然后把 10 个 checkpoint 分到这 4 张卡上：
bash scripts/run_libero_val.sh arm1 4 1000 2000 3000  &
bash scripts/run_libero_val.sh arm1 5 4000 5000 6000  &
bash scripts/run_libero_val.sh arm1 6 7000 8000       &
bash scripts/run_libero_val.sh arm1 7 9000 10000      &
bash scripts/run_libero_val.sh official 4        # 未微调的基线，只跑一次
python scripts/libero_val_table.py               # 表 + 选出的步数
```

（用 `setsid nohup ... < /dev/null &` 后台跑，§9.3。）

- **固定的 validation 集**：每个 suite 取排序后第 0、2、4、6、8 个任务，每个任务取 variation1 的
  第一条 episode，共 20 条。所有 arm、所有步数都用同一组。
- **固定的设置**：`eval/eval_action.py`，模板 `video+action`（三个 arm 共有的模板），协议 `i2va`
  （一帧真实锚点输入，就是闭环 rollout 的条件），512²，`frame_interval 1`。
- **什么时候跑**：8 张卡都在训练时没有空卡，所以默认是某个 LIBERO run 跑完 10000 步以后，用它空出来的
  4 张卡跑它自己的 10 个 checkpoint（约 2.5 小时）。如果你的机器有空闲的卡，也可以边训边跑：脚本会跳过
  还不存在的 checkpoint 和已经算完的格子，每存一个就再执行一次。**但边训边跑只是提前看，不是提前停**：
  三个 LIBERO run 都训满 10000 步，选步数的规则要在三个 arm 的全部 checkpoint 上用。
- **成本**：每条约 165 s，每个 checkpoint 约 1 小时单卡，一个 arm 的 10 个 checkpoint 约 10 GPU-小时，
  三个 arm 约 30 GPU-小时。
- **选步数的规则（事先定死，不看闭环结果）**：在三个 arm 都有全部 4 个 suite 的步数里，取
  "三个 arm × 四个 suite 的平均位置误差"最低的那一步；和最低值相差 0.5 cm 以内的，取更早的那一步。
  `libero_val_table.py` 最后一行打印 `SELECTED LIB_STEP=...`。**三个 arm 用同一个步数**，不给每个
  arm 单独挑最优，否则比较的就不只是菜单了。
- 如果到 10k 位置误差还在明显下降，照规则选 10k，并在汇报里指出"还没收敛"。**不要自己加训超过
  10k**，先汇报。

**原机器上的基线**（官方 checkpoint，未在 LIBERO 上训练过，同一组 20 条 episode）：

@@OFFICIAL@@

LIBERO 训过的模型应该明显低于这个基线。如果某个 checkpoint 的 `r_peak` 平均值掉到 100 以下，
说明模型已经不画可解码的动作了（坍缩），这比误差本身更要紧。

**`eval/eval_action.py` 里修过一个 bug，你拿到的版本已经修好。** 夹爪准确率原来拿二值化的预测和
原始真值用 `==` 比较；RLBench 的真值刚好是 0/1，所以没问题，但 LIBERO 的 openness 是连续值
（张开约 0.97），于是每一帧都判错、准确率恒为 0。现在真值也在 0.5 处二值化，RLBench 上的结果不变。
§1 的 `grep` 会检查这一点。

---

## 5. 评测

**先跑 GT-replay 门，再跑模型。** GT replay 用同一个控制器执行示教动作，不用模型。一个任务在这里
接近 0，就是执行 harness 的上限，**不是**模型的失败。每个模型格子都要对照它读。

```bash
source /path/to/env.rc && cd <repo> && export ENV_RC=/path/to/env.rc
bash scripts/run_single_source_eval.sh gt               # RLBench 用 CPU 并行；LIBERO 在 harness 写好之前自动跳过
DRY_RUN=1 bash scripts/run_single_source_eval.sh models  # 先看队列
setsid nohup bash scripts/run_single_source_eval.sh models > logs/eval.log 2>&1 < /dev/null &
```

默认用 8 张卡、每任务 20 次。RLBench 评 `checkpoint-4000`（`RLB_STEP`），**LIBERO 评 §4.5 选出的
那一步**：`LIB_STEP=<SELECTED> bash scripts/run_single_source_eval.sh models`。`LIB_STEP` 不给时默认
10000，那只是占位，**选步数之前不要评 LIBERO 模型**。可以用 `GPUS="0 1 2 3"`、`ARMS=`、`SOURCES=`、
`LIB_SPECS=` 覆盖；只评 RLBench 就 `SOURCES=rlbench`。`eval/rollout_libero.py` 不存在时，调度器会打印提示并
**跳过 LIBERO 的 job，RLBench 照跑**。所以 RLBench 的 3 个模型训完就可以先评，不用等 LIBERO。

### 5.1 RLBench

| 任务（variation0，都不在训练树里） | 原机器 GT 天花板 |
|---|---|
| `close_box` `close_drawer` `close_microwave` `toilet_seat_down` `meat_on_grill` | 全部 20/20 |

harness 是现成的 `eval/rollout.py`，协议写死在 `run_single_source_eval.sh` 里（见 §5.4）。

### 5.2 LIBERO：评测不是随机的

**LIBERO 的评测场景是固定的。** 每个任务有 benchmark 自带的 **50 个固定 init state**
（`benchmark.get_task_init_states(task_id)`，存在 LIBERO 仓库的 `.pruned_init` 文件里）。标准协议
是第 `i` 次 trial 用第 `i` 个 init state，而不是随机采样场景。原机器上核对过：这 50 个 init state
**和示教 demo 的初始状态都不一样**（物体位姿的最小距离 0.25–0.98），所以即使评的是训练过的
任务，也是在新摆放下评，不是在背训练集。

我们的协议：

- **任务**：`LIB_SPECS` 里的 `suite:task_index`，默认每个 suite 取第 0 个任务
  （`libero_spatial:0 libero_object:0 libero_goal:0 libero_10:0`，共 4 个任务）。
- **trial**：init state `0..19`（20 次），对每个模型都相同，所以模型之间是配对的。
- **相机**：LIBERO benchmark 自带的是固定相机，**我们不用**，因为训练数据全是随机相机（§3）。
  每个 `(suite, task, init)` 用一个**确定性**的种子，按训练树同样的分布采样 4 个相机（直接调用
  `scripts/libero_gen.py:sample_cameras` / `place_cameras`）。这样同一个 trial 对每个模型的相机
  也完全一样。
- **最大步数**（openvla 惯例，不含开头等待）：spatial 220，object 280，goal 300，libero_10 520。
  开头先执行 10 步"不动 + 夹爪张开"让物体落稳（`num_steps_wait=10`，同样是 openvla 惯例）。
- **成功**：任何一步 `env.check_success()` 为真就算成功并结束。

这 4 个任务是一个起点。完整的 LIBERO 协议是 4 × 10 个任务 × 50 次，按我们 rollout 的速度
（§5.5）根本跑不完。**先跑默认的 4 个**，时间允许再用 `LIB_SPECS` 按 suite 加任务（每个 suite
同时加，保持均衡），并在汇报里写清楚评了哪些。所有 40 个任务的编号和指令：

```bash
python -c "
from libero.libero import benchmark
for s in ['libero_spatial','libero_object','libero_goal','libero_10']:
    b = benchmark.get_benchmark_dict()[s]()
    for i in range(b.n_tasks): print(s, i, b.get_task(i).language)"
```

### 5.3 你要写的 `eval/rollout_libero.py`

**蓝本是 `eval/rollout_maniskill.py` + `eval/rollout_env_maniskill.py`**：同样的 receding-horizon
循环、同样的 policy 调用、同样的 JSON 和视频布局、同样的续跑与协议校验。你要换掉的是环境层。
调度器调用它的 CLI 已经定好（`run_single_source_eval.sh` 的 `lib_cmd`），照着实现：

```
--ckpt X | --gt-replay [--tree data/<suite>]   --tag  --suite  --task-index  --num-trials
--res 512 --cfg 7.5 --steps 50 --frame-interval 1 --prompt-tag-style explicit --axis-solver sphere
--skip-anchor-frames 4 --execution-horizon 41 --max-ik-fail-streak 5 --seed 42
--record-video | --no-record-video   --out DIR
```

**环境层的约定。下面每一条都是原机器上测出来的，不是猜的：**

1. **构建场景**：`OffScreenRenderEnv(bddl_file_name=..., camera_heights=512, camera_widths=512,
   camera_depths=True, camera_segmentations=...)`，然后 `env.seed(...)`、`env.reset()`、
   `env.set_init_state(init_states[trial])`。控制器是默认的 `OSC_POSE`。
2. **相机**：复用 `scripts/libero_gen.py` 的 `sample_cameras` → `place_cameras`。它会把 4 个
   LIBERO 相机挪到采样位姿，并断言它们挂在世界上（`cam_bodyid == 0`）、FOV 55°。场景中心的算法
   和生成器一致：取 BDDL 里物体根 body 的平均位置。**跟着 init state 走，要在
   `set_init_state` 之后算。**
3. **图像**：复用 `render_views`。**robosuite 的图像是 OpenGL 约定，上下颠倒**，rgb / 深度 / 分割
   都要 `[::-1]`，这个函数已经做了。模型的锚点帧必须和训练树完全同一条管线，否则就是在评一个
   分布外的输入。
4. **动作 → 控制器**：模型给出每帧的绝对 TCP 位姿 `[x,y,z,四元数,openness]`（世界系，和训练
   标签相同，§3）。把 `OSC_POSE` 设成**绝对模式**：控制器配置 `control_delta=False`，构建前改
   `controller_configs`，或者构建后对 `env.env.robots[0].controller.use_delta` 赋 False。然后每帧送
   `[x, y, z, 轴角(3), gripper]`。**`robot0_eef_quat` 是 xyzw 顺序**，转轴角之前别搞错。
5. **坐标系要用 GT 门来证明，别假设。** 标签用的是 `robot0_eef_pos/quat`；而 OSC 内部跟踪的是
   grip site 的位姿，姿态可能差一个固定旋转，位置在 LIBERO 的机器人底座上也可能有偏移。GT 门
   位置跟踪到毫米级、成功率接近 20/20，才算约定对了。不对就先查这里。
6. **夹爪**：robosuite 是 **−1 = 张开，+1 = 闭合**，和 ManiSkill（+1 = 张开）**正好相反**。
   `openness > 0.5 → −1`，否则 `+1`。LIBERO 里夹住物体时 openness ≤ 0.33，所以 0.5 是安全的
   阈值。ManiSkill 那边用 0.8，是因为那里夹住的 openness 等于物体宽度；这里不要照抄。
7. **每帧执行多少仿真步**：训练帧率 20 Hz，就是 LIBERO 的控制频率，所以一个模型帧对应一次
   `env.step`。如果 GT 门显示跟踪跟不上，可以对同一个目标重复 2–3 步，但**GT 门和模型必须用同样
   的设置**，并写进 JSON 的 `args`。
8. **`--max-ik-fail-streak`**：OSC 没有 IK 失败，你需要一个对应物。建议：一个路点执行后，位置跟踪
   误差仍大于 2 cm 就记为一次"不可达"，连续 5 次终止 trial（`stop_reason=unreachable_streak`）。
   做了什么就在 JSON 里写清楚。
9. **GT 门（`--gt-replay --tree data/<suite>`）**：LIBERO 的 init states 没有对应的 demo，所以 GT 门
   不在 init states 上跑。它回放**训练树 variation0 里该任务的前 20 条 demo**：用生成器的
   `load_demo_scene` 加载 demo 自己的场景和初始状态，然后把 `actions.npy` 里记录的 TCP 位姿逐帧送进
   **和模型完全相同的**绝对控制器，最后保持末位姿直到步数上限。它验证的是控制器、坐标系和夹爪符号，
   这正是第 4–7 条里容易错的地方。生成器的 `load_demo_scene` 里有自写的资源路径修正，原因是官方的
   `postprocess_model_xml` 处理不了部分 demo XML 里原机器的绝对路径。**GT 门不到约 18/20 就不要评模型。**
10. **JSON**：和 `rollout_maniskill.py` 同样的结构（完整 `args`，每个 trial 有
    `success / steps / replans / ik_fail_rate / stop_reason / init_index / camera_seed / prompt`），
    写到 `--out/rollout_<tag>.json`。续跑时校验协议参数一致，不一致就拒绝。
    `scripts/cl_json_complete.py` 用它判断 job 是否已完成。
11. **EGL 设备**：设 `MUJOCO_EGL_DEVICE_ID`，让 MuJoCo 渲染在 `CUDA_VISIBLE_DEVICES` 指定的那张卡上，
    否则 8 个 job 的渲染会全挤到 0 号卡。
12. **prompt**：`<video><action> ` 前缀加任务指令（`task.language`）。训练树里的指令就是它，
    小写，不带句号。

**写完以后的验证顺序**，每一步都通过了才做下一步：

1. 1 个 trial 的 GT 门，带 `--record-video`，肉眼看一遍 executed 视频。
2. 4 个 suite 各跑 20 次 GT 门。
3. 用**官方 checkpoint** 在一个任务上跑 2 个 trial，只为验证 policy 这一路能跑通。这是管线测试，
   数字没有意义。
4. 然后才评 LIBERO 训出来的模型。

**第 3 步必须看 generated 视频**：模型想象出来的第一帧应该和 executed 的第一帧长得一样。不一样
说明锚点帧的渲染（翻转、相机、分辨率）和训练树对不上。原机器上 ManiSkill harness 出过一次类似的
事：一个只在 GT replay 上测过的渲染开关，让 18 个模型 job 全部空转。

### 5.4 两个仿真器共同的协议

下面每一项都不是自由选择，已经写死在 `run_single_source_eval.sh` 里：

- `--frame-interval`：RLBench **3**，LIBERO **1**，和各自的训练一致。
- `--max-ik-fail-streak 5`：这是 harness 的策略，不是模型属性。它单独就能让一个数字变 26 个
  百分点，绝不能跨不同取值比较。
- `--skip-anchor-frames 4`：丢掉 chunk 开头那几帧（它们只是在重建机械臂当前的位姿），空出来的
  位置用斜坡补上，chunk 长度不变。
- `--prompt-tag-style explicit`：所有微调过的 arm 训练时 prompt 都带 `<video><action> ` 这类前缀。
- `--record-video`：每个 trial 存两个视频，一个是仿真器实际执行的（`executed/`），一个是模型想象的
  （`generated/`）。没有它，一次失败就只剩一个标量，分不清是世界模型错了还是动作解码错了。

**场景在模型之间是配对的。** RLBench 用 `blake2b(task|variation|trial)` 播种，只取决于这三个，
和 checkpoint、进程都无关。LIBERO 的第 `trial` 次用第 `trial` 个 init state 和确定性的相机。
所以同一个 `(task, trial)` 对每个模型都是同一个场景，模型之间可以做精确的配对 McNemar 检验。

**续跑。** 调度器跳过已经满 20 次的 job，可以反复执行。**RLBench 的 rollout 不能部分续跑**，被杀掉的
RLBench job 会从第 0 次重来。LIBERO 的续跑要由你实现（§5.3 第 10 条）。

### 5.5 时间

**RLBench 实测**：每个 trial 平均约 400 s（`close_drawer` 306 s 到 `basketball_in_hoop` 575 s），
3 个模型 × 5 任务 × 20 次 ≈ 33 GPU-小时。一张卡一次只能跑一个 job（每个约 28–30 GB）。

**LIBERO 没有实测**，只能按 chunk 数估算：步长 1 时每个 chunk 推进约 37 帧，最大步数 220–520 的
trial 大约要 6–14 次推理，每次约 1 分钟。所以每个 trial 大约 10–30 分钟，失败的 trial 会跑到上限，
最慢。3 个模型 × 4 任务 × 20 次 ≈ 40–120 GPU-小时。跑完第一个 job 就用实测时间重估，并告诉我。

---

## 6. 怎么读输出

结果在 `reports/closedloop_single_source/rollout_<tag>.json`：
- 模型：RLBench 是 `ss_<arm>_rlbench_4k_<task>`，LIBERO 是 `ss_<arm>_libero_<N>k_<suite>_t<idx>`（`N` = 选出的步数）。
- GT 门：RLBench 是 `ssgt_rlbench_<task>`，LIBERO 是 `ssgt_libero_<suite>_t<idx>`。
- 每个文件自带完整的 `args`，事后可以查协议。
- 视频在 `reports/closedloop_single_source/videos/<tag>/{executed,generated}/`，文件名带
  `SUCCESS` / `fail`。

一张总表：

```bash
LIB_K=10k python - <<'PY'     # LIB_K = §4.5 选出的 LIBERO 步数，如 6k
import json, glob, os
R = "reports/closedloop_single_source"
cell = {}
for p in glob.glob(f"{R}/rollout_*.json"):
    r = json.load(open(p)).get("results", [])
    if r: cell[os.path.basename(p)[8:-5]] = f"{sum(x['success'] for x in r)}/{len(r)}"
rows = [("rlbench", t) for t in "close_box close_drawer close_microwave toilet_seat_down meat_on_grill".split()]
LK = os.environ.get("LIB_K", "10k")        # e.g. LIB_K=6k for the step §4.5 selected
rows += [("libero", k[len(f"ss_arm1_libero_{LK}_"):]) for k in sorted(cell) if k.startswith(f"ss_arm1_libero_{LK}_")]
print(f"{'':34s} {'GT':>7s} {'arm1':>7s} {'arm2':>7s} {'arm3':>7s}")
for src, t in rows:
    print(f"{src + ' ' + t:34s} {cell.get(f'ssgt_{src}_{t}', '-'):>7s} " +
          " ".join(f"{cell.get(f'ss_{a}_{src}_' + ('4k' if src == 'rlbench' else LK) + f'_{t}', '-'):>7s}" for a in ("arm1", "arm2", "arm3")))
PY
```

**不同任务集上的聚合成功率不可比，即使 n 一样。** 原机器上有一次，中途 arm0 那列显示 50.0%、
arm7 那列 19.6%，纯粹是因为当时 arm0 只跑完了最容易的 6 个任务。永远逐任务比，或者只在两边都
跑过的配对交集上比。LIBERO 按 suite 报，不要把 4 个 suite 平均成一个数，除非每个 suite 的任务数相同。

---

## 7. 原机器上的参考数字

**这些都是别的模型的数字**，只用来判断你的结果量级是否合理，**不能**当作你这 6 个 run 的对照组。
GT 天花板例外：它和模型无关，你的 GT 门应该和它基本一致。如果明显更低，先查 harness，别评模型。

RLBench，同 5 个任务，20 次：

| 任务 | GT | 官方 checkpoint（未微调） | arm7 · RLBench+ManiSkill 混合 · 4k |
|---|---|---|---|
| close_box | 20/20 | 5/20 | 6/20 |
| close_drawer | 20/20 | 16/20 | 11/20 |
| close_microwave | 20/20 | 15/20 | 17/20 |
| toilet_seat_down | 20/20 | 3/20 | 7/20 |
| meat_on_grill | 20/20 | 3/20 | 12/20 |

**LIBERO 没有任何参考数字**：没有训练过，也没有 harness。你的 GT 门就是第一个数字。

**模型在自己训练过的仿真器里也可能很低，这不一定说明你的环境坏了。** 在 ManiSkill 上（同样是随机
相机、步长 1、单独训练 4k 步），原机器的模型在**训练过的任务、新 seed** 上几乎都是 0：
pick_cube 0/10、stack_cube 0/10、push_cube 约 1/10，而 GT 门都是 20/20。失败方式以"连续 5 次
不可达"（输出了够不到的位姿）和超时为主。LIBERO 很可能也会低。前提是 GT 门是好的：GT 门好、
模型低，这是一个结果，照实汇报。GT 门低，那是 harness 的 bug，先修。

---

## 8. 汇报什么

每个训练 run：
1. 全部 checkpoint（每个 12.8 GB）：RLBench 的 2k/4k；LIBERO 的 1k–10k。
2. 启动日志，包括 §4.4 的五行。它们是"这个 run 确实在训它声称的东西"的证据。
3. 离线 W&B 目录 `outputs/wandb_offline/wandb/offline-run-*`（带回来 `wandb sync`）。
4. 任何重启：第几步、为什么。LIBERO run 另外报每步时间和每卡显存。

LIBERO validation：
1. `reports/libero_val/` 整个目录，以及 `python scripts/libero_val_table.py` 的输出（含 `SELECTED LIB_STEP`）。
2. 如果到 10k 还在明显下降，明确写出来。

评测：
1. `reports/closedloop_single_source/` 下全部 JSON（模型 + GT 门）。
2. 视频，至少所有失败的，以及 LIBERO GT 门的前几个。
3. §6 那张总表。
4. 任何你跳过或改动的任务，以及它的 GT 门数字。

LIBERO harness：
1. `eval/rollout_libero.py` 本身，以及你对其它文件的任何改动。
2. §5.3 里每一条约定你最终是怎么实现的，特别是第 4、5、7、8 条（绝对控制的设置、坐标系是否需要
   修正、每帧几个仿真步、"不可达"怎么定义），以及 GT 门的跟踪误差。
3. 每个 trial 的实测耗时。

---

## 9. 已知的坑：下面每一条都在原机器上真实发生过

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

**9.10 日志里的 `Traceback` 不一定是错误。** xfuser 在 import 时会以 `DEBUG ... compat.py`
打印一段 `ImportError: cannot import name 'Flux2KleinPipeline'` 的 traceback，这是它自己捕获了的，
无害。用 grep 找错误时要排除 `compat.py`，否则监视脚本会被它骗到提前退出。

**9.11 LIBERO 的图像上下颠倒。** robosuite 的 `IMAGE_CONVENTION` 是 `opengl`，渲出来的 rgb、深度、
分割都是倒的。原机器生成数据时一开始只用一个点验证，结论是"不用翻"，结果是错的：换成多物体
投影测试，翻转后 53/63 个点命中，不翻只有 12/63。harness 里用 `render_views`，不要自己再翻一次，
也不要漏翻。

**9.12 LIBERO 的深度不是米。** MuJoCo 的原始深度是归一化 z-buffer，要用
`robosuite.utils.camera_utils.get_real_depth_map` 转成米。训练树里是米。

**9.13 LIBERO HDF5 里 `obs[t]` 和 `states[t]` 错一位。** `states[t]` 对应的是 `obs[t-1]`
（实测对齐误差 0.32 mm，和 `obs[t]` 对齐是 8 mm）。我们的训练树不受影响，因为帧和标签都从回放的
`states[t]` 得来。但如果你拿官方 HDF5 的 `obs` 或 `actions` 做任何对照，要知道这一点。

**9.14 夹爪符号和 ManiSkill 相反。** robosuite：−1 张开，+1 闭合。原机器上的 ManiSkill harness
是 +1 张开。从 `rollout_env_maniskill.py` 改过来时这里最容易抄错，而抄错的症状（夹爪永远不闭合，
或者一开始就闭合）很容易被误读成"模型不会抓"。

**9.15 LIBERO 首次 import 交互式提问**（§2）。后台进程里它会卡住，没有任何报错。

**9.16 磁盘。** 原机器的卷曾经在跑的过程中只剩 632 KB。始终保留 40 GB 以上。

---

---

## 10. 如果你改了任何东西

同一个数据源的 3 个 run 之间只应该有一个变量：模板菜单（两个数据源之间另外还差数据、步数上限和
LIBERO 可能调过的 batch，这些是设计好的）。如果你必须改任何其它东西，比如不同的 GPU
数、不同的 batch、任何一个参数，**在汇报里明确写出来**，不要默默吸收，并且**同样地改到同一数据源
的全部 3 个 run 上**，而不是只改需要它的那一个。在不同有效 batch 下训出来的 run 不在同一张表里。
LIBERO harness 的任何约定一旦定下（§5.3），GT 门和 3 个模型都必须用同一版；harness 改了，
之前的 JSON 就作废重跑。
