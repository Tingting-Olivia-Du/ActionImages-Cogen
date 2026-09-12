#!/bin/bash
# 几何中位融合闭环 —— 排队 + OOM 重试，不硬抢卡。
#
#   nohup setsid bash scripts/queue_fusion_closedloop.sh smoke > logs_queue_fusion.txt 2>&1 &
#   nohup setsid bash scripts/queue_fusion_closedloop.sh full  > logs_queue_fusion.txt 2>&1 &
#
# 为什么要排队重试：这盘卡与别的容器共享，实测三次「查到空闲 -> 启动 -> 模型加载完时已被占满」
# 的竞态（rollout_env / heldout_batch_eval 的注释里也记着同一件事）。检查从来不是问题，
# 问题是检查与真正占住显存之间的那十几秒。所以：确认空闲 -> 启动 -> 失败就换卡重来。
#
# 融合模式的代价：每次 replan 跑 4 次扩散采样，~6.8 -> ~27 分钟/rollout。
# smoke = 1 任务 x 3 局（约 1.4h），full = 5 任务各 20 局、按任务切片并行（约 9h）。
#
# 判据：融合是【尾部保护】，不是提升典型情况。离线实测 p90 −10.1% 而 p50 +1.7%。
# 闭环里三条弱路径（17–28% vs RGB 46%）的拖累可能超过这个收益 —— 这正是要测的。
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
source /opt/conda/etc/profile.d/conda.sh
source /workspace/ttdu/ttd/scripts/env_eval.rc 2>/dev/null
conda activate ttd_rollout
export PYTHONPATH="$REPO"

MODE="${1:-smoke}"
CKPT="$REPO/outputs/arm7__seed42_fi3_512_aug_sr/checkpoint-10000/step10000.ckpt"
NEED="${NEED:-34000}"          # 融合要同时装模型 + 四次采样的中间量，留足余量
say(){ echo "[fusion $(date -u +%m-%d\ %H:%M)] $*"; }

launch(){   # $1=tag $2=tasks $3=trials  —— 拿卡、起、失败就换卡重来
  local tag="$1" tasks="$2" trials="$3" attempt=1 g
  while [ "$attempt" -le 8 ]; do
    while true; do
      g=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
          | awk -F', *' -v n="$NEED" '{f=$3-$2; if(f>=n) print $1, f}' | sort -k2 -rn | head -1 | cut -d' ' -f1)
      [ -n "$g" ] && break
      sleep 180
    done
    say "$tag 尝试 $attempt -> GPU $g"
    CUDA_VISIBLE_DEVICES=$g xvfb-run -a python -u eval/rollout.py \
      --ckpt "$CKPT" --tag "$tag" --anchor-modality all --tasks $tasks \
      --variation 0 --num-trials "$trials" \
      --res 512 --cfg 7.5 --steps 50 --frame-interval 3 \
      --prompt-tag-style explicit --axis-solver sphere --arm-action-mode planning \
      --max-steps-factor 1.5 --skip-anchor-frames 4 --execution-horizon 41 --seed 42 \
      > "$REPO/logs_fusion_$tag.txt" 2>&1
    if [ -s "$REPO/reports/closedloop/rollout_$tag.json" ]; then
      say "$tag 完成"; return 0
    fi
    # 报真实错误,不要臆断。上一版写死「多半是 OOM」,结果 8 次全是同一个 ValueError
    # (policy 不认 anchor_modality=all),而那条假设让我先去查抢卡而不是查代码。
    say "$tag 失败: $(grep -oE '^[A-Za-z_.]*(Error|Exception):.*' "$REPO/logs_fusion_$tag.txt" | tail -1 | cut -c1-120)"
    attempt=$((attempt+1)); sleep 120
  done
  say "$tag 连试 8 次仍失败，放弃"; return 1
}

if [ "$MODE" = "smoke" ]; then
  launch "smoke_fusion" "meat_off_grill" 3
else
  # 按任务切片，逐个排队（每片自己找卡，卡不够就等，不会互相踩）
  for t in close_box close_drawer push_buttons meat_off_grill open_drawer; do
    launch "fusion_$t" "$t" 20 &
    sleep 120     # 错开启动，避免同时盯上同一张卡
  done
  wait
fi
say "DONE"
