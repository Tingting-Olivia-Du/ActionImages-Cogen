#!/bin/bash
# arm7u（均匀采样对照）的排队启动器 —— 等当前那批 n=80 评测腾出卡再上。
#
#   nohup setsid bash scripts/queue_arm7_uniform.sh > logs_queue_arm7u.txt 2>&1 &
#
# 为什么要排队而不是直接起:现在 8 张卡全满(0/1/4/5/6 在跑 arm7@4000 的 n=80 对照,
# 2/3 在训 arm7,7 在跑闭环)。硬起会和它们抢显存,而 ZeRO-2 在 512 分辨率下单卡放不下,
# 抢输了就是 OOM。
#
# 为什么要两张卡:zero2_offload 把 fp32 master + Adam 动量放 CPU,但 world_size=1 时
# 没有东西可以分片,整份 fp32 副本要落到单卡上,在优化器步 OOM(见 smoke_arm.sh 头注释)。
#
# 判据:如果 arm7u 的 depth/normal 不再比 RGB 差,那 arm7 观察到的退化就是【更新次数不平衡】,
# 不需要梯度调制或互教那些更贵的机制。反之才轮得到它们。
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
NEED_MIB="${NEED_MIB:-42000}"        # 训练比评测吃得多:arm7 实测每卡 41.7GB
WAIT_PAT="${WAIT_PAT:-arm7_4000n80}" # 先等这批评测结束
OUT="$REPO/outputs/arm7u__seed42_fi3_512_aug_sr"

say(){ echo "[queue-arm7u $(date -u +%m-%d\ %H:%M)] $*"; }

# ---------- 1. 等指定的作业结束 ----------
say "等 $WAIT_PAT 这批评测结束..."
while [ "$(ps -eo cmd --no-headers | grep -c "[h]eldout_batch_eval.py.*$WAIT_PAT")" != "0" ]; do
  sleep 120
done
say "$WAIT_PAT 已结束"

# ---------- 2. 等两张卡同时有 NEED_MIB 空余 ----------
# 两张必须【同时】空,不能一张一张占 —— 半占着一张卡等另一张,会把别人也卡住。
say "等两张卡各有 ${NEED_MIB} MiB 空余..."
while true; do
  FREE=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
         | awk -F', *' -v n="$NEED_MIB" '{f=$3-$2; if(f>=n) printf "%s ",$1}')
  CNT=$(echo $FREE | wc -w)
  if [ "$CNT" -ge 2 ]; then
    G1=$(echo $FREE | cut -d' ' -f1); G2=$(echo $FREE | cut -d' ' -f2)
    say "拿到 GPU $G1,$G2（当前空闲: $FREE）"
    break
  fi
  say "  只有 $CNT 张够用（$FREE），继续等"
  sleep 180
done

# ---------- 3. 起训 ----------
# output_dir 非空会让 train.py RESUME 而不是 warm-start(train.py:490),那会把这个对照臂
# 变成别的东西。宁可不启动也不要静默续跑。
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  say "!! $OUT 非空,拒绝启动(会变成 RESUME 而不是 warm-start)"; exit 3
fi

say "启动 arm7u:GPUS=$G1,$G2  10000 步  CKPT_EVERY=1000"
GPUS="$G1,$G2" SEED=42 STEPS=10000 CKPT_EVERY=1000 SAVE_TOP_K=-1 \
  bash scripts/train_arm.sh arm7u > "$REPO/logs_arm7u.txt" 2>&1
say "train_arm.sh 退出码 $? —— 见 logs_arm7u.txt"
