#!/usr/bin/env bash
# Sequential GPU-0-only matched qualitative rollouts after image generation exits.
set -euo pipefail
REPO=/workspace/1228_tingting/ActionImages-Cogen
cd "$REPO"
generation_pid="${1:?pass the running generation process PID}"
while kill -0 "$generation_pid" 2>/dev/null; do sleep 15; done
for sample in figure1_video figure1_depth figure1_normal figure1_segmentation figure2_depth; do
  test -f "reports/paper_figures_arm7_4k/$sample/generation.json"
done
source /workspace/1228_tingting/ttd/scripts/env_eval_ttd_eval.rc
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
OUT="$REPO/reports/paper_figures_arm7_4k/rollouts"
mkdir -p "$OUT"
for model in arm7 arm0; do
  tag="figure_${model}_4k_close_drawer"
  ckpt="$REPO/outputs/${model}_joint2src_seed42_fi3_6k/checkpoint-4000/step4000.ckpt"
  echo "START $tag $(date -u +%FT%TZ)"
  xvfb-run -a python -u scripts/paper_rollout_capture.py \
    --ckpt "$ckpt" --tag "$tag" --tasks close_drawer \
    --variation 0 --num-trials 1 --res 512 --cfg 7.5 --steps 50 \
    --frame-interval 3 --prompt-tag-style explicit --axis-solver sphere \
    --arm-action-mode planning --max-steps-factor 1.5 --skip-anchor-frames 4 \
    --execution-horizon 41 --max-ik-fail-streak 5 --anchor-modality video \
    --seed 42 --record-video --out "$OUT" > "$OUT/$tag.log" 2>&1
  echo "FINISHED $tag $(date -u +%FT%TZ)"
done
echo ALL_PAIRED_ROLLOUTS_COMPLETE
