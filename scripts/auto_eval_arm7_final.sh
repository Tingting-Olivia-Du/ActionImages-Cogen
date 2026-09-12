#!/bin/bash
# arm7 的最终离线判据 —— 等 step10000 落盘后自动跑,无人值守。
#
#   nohup setsid bash scripts/auto_eval_arm7_final.sh > logs_auto_eval_arm7_final.txt 2>&1 &
#
# 做三件事:
#   1. 等 checkpoint-10000 真正写完(等训练日志里的 "Saved checkpoint ... step10000.ckpt",
#      不是等文件出现 —— 文件出现时还在写,12.8GB 要写好几分钟)
#   2. 把 80 个 episode 切 5 片,每片抢一张空卡跑四条路径(action / action_from_{depth,seg,normal})
#   3. 合并五片,给出结论一(四路等价)与结论二(对齐曝光后 vs spec@4000)
#
# 为什么是这 80 个 episode:它们是 reports/heldout_batch_eval/actionspec_4000_matched 用的
# 同一批(8 任务 x 5 episode x 2 variation)。arm7@10000 与 spec@4000 的 `video+action` 曝光
# 恰好匹配(10000 x 0.4 = 4000),所以那份存档就是现成的对照,逐 episode 可配对,不用重跑 spec。
#
# 为什么 n=80 而不是默认的 16:相邻 checkpoint 之间同一条路径的抖动实测 5-20%,而要测的效应
# (比值 1.2-1.3)只有 20-30% —— n=16 分辨不了。80 个把误差棒收窄约 sqrt(5)=2.2 倍。
#
# GPU:每片需要 ~28GB。CUDA_VISIBLE_DEVICES 必须在 python 启动【之前】设 —— heldout_batch_eval.py
# 内部再设已经太晚(import 时 CUDA 已初始化),会让所有片挤到物理 GPU 0 上互相 OOM。
set -uo pipefail
REPO="/workspace/ttdu/ActionImages-Cogen"; cd "$REPO"
source /opt/conda/etc/profile.d/conda.sh
conda activate ttd_train
export PYTHONPATH="$REPO"

CKPT="$REPO/outputs/arm7__seed42_fi3_512_aug_sr/checkpoint-10000/step10000.ckpt"
TRAINLOG="$REPO/logs_arm7.txt"
NEED_MIB="${NEED_MIB:-30000}"
NSHARD="${NSHARD:-5}"
SCRATCH="${SCRATCH:-/tmp/claude-0/-workspace-ttdu/a2319880-c7c3-414f-a5e7-69328956d54e/scratchpad}"
EPS80="$SCRATCH/eps80.txt"

say(){ echo "[auto-eval $(date -u +%m-%d\ %H:%M)] $*"; }

# ---------- 0. 前置检查 ----------
[ -s "$EPS80" ] || { say "!! 缺 $EPS80(那 80 个 episode 的清单),退出"; exit 2; }
say "episode 清单 OK: $(tr ',' '\n' < "$EPS80" | grep -c .) 个"

# ---------- 1. 等 step10000 写完 ----------
# 等日志行而不是等文件:文件一创建就存在,但 12.8GB 要写几分钟,提前读会拿到半截文件。
say "等 checkpoint-10000 落盘(轮询训练日志)..."
while true; do
  if grep -q "Saved checkpoint to .*step10000\.ckpt" "$TRAINLOG" 2>/dev/null; then
    say "训练日志已确认 step10000 保存完成"; break
  fi
  # 训练进程没了但也没写出那一行 = 崩了,别干等
  if ! pgrep -f "train.py --deepspeed" >/dev/null 2>&1; then
    if [ -s "$CKPT" ]; then
      say "训练进程已退出,且 ckpt 存在 —— 按已完成处理"; break
    fi
    say "!! 训练进程消失且没有 step10000.ckpt,退出"; exit 3
  fi
  sleep 120
done
sleep 60   # 给文件系统一点余量
[ -s "$CKPT" ] || { say "!! $CKPT 不存在,退出"; exit 4; }
say "ckpt 大小 $(stat -c%s "$CKPT" | awk '{printf "%.2f GB",$1/1e9}')"

# ---------- 2. 切片 ----------
python3 - "$EPS80" "$SCRATCH" "$NSHARD" <<'PY'
import sys
eps=open(sys.argv[1]).read().split(','); scratch=sys.argv[2]; n=int(sys.argv[3])
for i in range(n):                      # 轮转分片,任务/variation 在各片间均匀
    open(f"{scratch}/final_shard{i}.txt","w").write(','.join(eps[i::n]))
    print(f"  shard{i}: {len(eps[i::n])} eps")
PY

# ---------- 3. 逐片抢卡启动 ----------
# 卡不够就等,不 fail-fast:别的容器随时会占卡,而这活儿跑一次要 3 小时,不值得因为一时没卡就放弃。
claim(){   # -> 一张剩余显存 >= NEED_MIB 且尚未被本脚本占用的物理卡号
  local used="$1"
  while true; do
    local g
    g=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
        | awk -F', *' -v need="$NEED_MIB" -v used="$used" '
            { free=$3-$2; if (free>=need+2000 && index(","used",", ","$1",")==0) print $1, free }' \
        | sort -k2 -rn | head -1 | cut -d' ' -f1)
    [ -n "$g" ] && { echo "$g"; return; }
    sleep 180
  done
}
USED=""
for i in $(seq 0 $((NSHARD-1))); do
  G=$(claim "$USED"); USED="$USED,$G"
  say "shard$i -> 物理 GPU $G"
  CUDA_VISIBLE_DEVICES=$G nohup python -u scripts/heldout_batch_eval.py \
    --ckpt "$CKPT" --tag "arm7_10000_s$i" \
    --modalities action,action_from_depth,action_from_seg,action_from_normal \
    --episodes "$(cat "$SCRATCH/final_shard$i.txt")" \
    --gpu 0 --need_mib "$NEED_MIB" \
    --frame_interval 3 --res 512 --segmentation_mode scene_roles --seed 42 \
    > "$REPO/logs_arm7_10000_s$i.txt" 2>&1 &
  sleep 90     # 让这一片先把显存 reserve 掉,再去挑下一张卡
done

say "五片已启动,等待完成..."
while [ "$(ps -eo cmd --no-headers | grep -c '[h]eldout_batch_eval.py --ckpt.*checkpoint-10000')" != "0" ]; do
  sleep 180
done
say "全部完成,开始合并分析"

# ---------- 4. 合并 + 分析 ----------
python3 - <<'PY'
import json, glob, statistics as st
from math import comb
def sp(k,n):
    if n==0: return 1.0
    lo=sum(comb(n,i) for i in range(0,k+1)); hi=sum(comb(n,i) for i in range(k,n+1))
    return min(1.0, 2*min(lo,hi)/2**n)

A={}
for f in glob.glob('reports/heldout_batch_eval/arm7_10000_s*/per_episode.json'):
    for r in json.load(open(f))['results'].values():
        if r.get("value") is not None: A.setdefault(r.get("modality",""),{})[r["episode"]]=r["value"]
S={}
for r in json.load(open('reports/heldout_batch_eval/actionspec_4000_matched/per_episode.json'))['results'].values():
    if r.get("modality")=="action" and r.get("value") is not None: S[r["episode"]]=r["value"]

base=A.get("action",{})
print("="*72); print(f"arm7@10000 最终判据   n={len(base)} episode"); print("="*72)

print("\n【结论一】四条输入路径是否等价（arm7 内部互比）")
print(f"  {'路径':<8}{'均值':>10}{'比值':>9}{'胜场':>10}{'p':>9}")
print(f"  {'RGB':<8}{st.mean(list(base.values())):>10.4f}{'—':>9}{'—':>10}{'—':>9}")
tot=w=0
for m,n in (("action_from_depth","depth"),("action_from_seg","seg"),("action_from_normal","normal")):
    eps=sorted(set(A.get(m,{}))&set(base))
    if not eps: continue
    mv=st.mean([A[m][e] for e in eps]); r=mv/st.mean([base[e] for e in eps])
    b=sum(1 for e in eps if A[m][e]<base[e]); w+=b; tot+=len(eps)
    print(f"  {n:<8}{mv:>10.4f}{r:>8.2f}x{b:>7}/{len(eps)}{sp(b,len(eps)):>9.3f}")
print(f"\n  汇总（{tot} 次配对，三路共用 RGB 分母故 p 偏乐观）: 非RGB更好 {w}/{tot}  p={sp(w,tot):.4f}")
print(f"  历史 n=16: 4000 24/48 p=1.000 | 6000 18/48 p=0.111 | 7000 17/48 p=0.059")

print("\n【结论二】曝光对齐 arm7@10000 vs spec@4000（video+action 曝光均为 4000）")
eps=sorted(set(base)&set(S))
if eps:
    a=[base[e] for e in eps]; s=[S[e] for e in eps]
    b=sum(1 for e in eps if base[e]<S[e])
    print(f"  arm7 {st.mean(a):.4f}   spec {st.mean(s):.4f}   比值 {st.mean(a)/st.mean(s):.2f}x")
    print(f"  arm7 更好 {b}/{len(eps)}   p={sp(b,len(eps)):.4f}")
    print(f"  历史对齐点: ~1600 曝光 1.08x(10/16,p=0.454) | ~2400 曝光 0.97x(9/16,p=0.804)")
else:
    print("  !! 与 actionspec_4000_matched 没有共同 episode，无法配对")
PY
say "DONE"
