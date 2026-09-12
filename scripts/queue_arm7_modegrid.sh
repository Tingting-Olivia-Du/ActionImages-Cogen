#!/bin/bash
# arm7@10000 的感知流渲染 —— 四条路径 x {IIII, FIFI}，出视频 + 联系表 + 指标。
#
#   nohup setsid bash scripts/queue_arm7_modegrid.sh > logs_queue_modegrid.txt 2>&1 &
#
# 目的：肉眼看 arm7 生成的 rgb / depth / seg / normal 视频有没有大偏差，外加指标。
#
# 【必须先理解的语义，否则数字会被误读】
#
#   video+X   IIII/FIFI  : RGB 给定或部分给定 -> X 预测   = 感知，可与 DA2/Marigold 等比
#   X+action  IIII       : 一帧 X -> 未来 X 视频 + action = X 空间的【世界模型】
#   X+action  FIFI       : X 整段【给定】-> 只预测 action = X 段是 GT 过 VAE，不是生成
#
# 所以：
#   * arm7 的感知流生成质量只在 **IIII** 下可测；
#   * FIFI 下 X 段的 AbsRel/mIoU 量到的是 **VAE 往返地板**（实测 depth 0.36%），不是模型能力；
#     保留 FIFI 是因为它对 **action 段**仍有意义 —— 那是 v2a（完整观测→动作）的上界。
#   * arm7 **没有** video+X 模板，所以与外部深度基线的比较在 arm7 上不成立；
#     那个比较属于 arm6（它训过 video+depth）。
#
# 每张卡跑一个 episode 的全网格；episode 之间用不同的卡并行。
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
source /opt/conda/etc/profile.d/conda.sh; conda activate ttd_train
export PYTHONPATH="$REPO"

CKPT="$REPO/outputs/arm7__seed42_fi3_512_aug_sr/checkpoint-10000/step10000.ckpt"
ONLY="video+action,depth+action,segmentation+action,normal+action"
MODES="iiii,fifi"
NEED="${NEED:-30000}"
EPISODES=("open_drawer/variation0/episodes/episode0"
          "push_buttons/variation0/episodes/episode0"
          "meat_off_grill/variation0/episodes/episode0"
          "close_jar/variation0/episodes/episode0")

say(){ echo "[modegrid $(date -u +%m-%d\ %H:%M)] $*"; }

claim(){   # -> 一张剩余 >= NEED 且不在 $1 里的卡；没有就等
  local used="$1" g
  while true; do
    g=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
        | awk -F', *' -v n="$NEED" -v u="$used" '{f=$3-$2; if(f>=n+2000 && index(","u",", ","$1",")==0) print $1, f}' \
        | sort -k2 -rn | head -1 | cut -d' ' -f1)
    [ -n "$g" ] && { echo "$g"; return; }
    sleep 180
  done
}

USED=""
for i in "${!EPISODES[@]}"; do
  EP="${EPISODES[$i]}"
  G=$(claim "$USED"); USED="$USED,$G"
  TAG="arm7_10000_grid_$(echo "$EP" | cut -d/ -f1)"
  say "episode=$EP -> GPU $G  tag=$TAG"
  CUDA_VISIBLE_DEVICES=$G nohup python -u scripts/modality_mode_grid.py \
    --ckpt "$CKPT" --tag "$TAG" --episode "$EP" \
    --only "$ONLY" --modes "$MODES" \
    --gpu 0 --need_mib "$NEED" \
    --res 512 --num_frames 41 --frame_interval 3 --steps 50 --cfg 7.5 --seed 42 \
    --segmentation_mode scene_roles \
    > "$REPO/logs_modegrid_$i.txt" 2>&1 &
  sleep 60
done

say "四个 episode 已派发，等待完成..."
while [ "$(ps -eo cmd --no-headers | grep -c '[m]odality_mode_grid.py --ckpt')" != "0" ]; do sleep 180; done
say "全部完成"
for i in "${!EPISODES[@]}"; do
  T="arm7_10000_grid_$(echo "${EPISODES[$i]}" | cut -d/ -f1)"
  D="$REPO/reports/modality_mode_grid/$T"
  say "$T: $(ls "$D"/*.mp4 2>/dev/null | wc -l) 个视频, $(ls "$D"/*.png 2>/dev/null | wc -l) 张联系表"
done
say "DONE"
