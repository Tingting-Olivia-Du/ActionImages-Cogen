#!/bin/bash
# Put the three large inputs into /staging ONCE. Run this on the CHTC access point, not locally:
# pulling from HuggingFace at 1 Gb/s inside the cluster beats uploading 170 GB from a lab machine.
#
#   ssh tdu35@ap2002.chtc.wisc.edu
#   export HF_TOKEN=...            # only needed if a repo is private
#   bash chtc/stage_to_chtc.sh
set -euo pipefail
STAGING="${STAGING_DIR:-/staging/$USER}"
HF_USER="${HF_USER:-TingtingDu}"
mkdir -p "$STAGING/fusion"
command -v huggingface-cli >/dev/null || pip install --user -q "huggingface_hub[cli]"

echo "staging into $STAGING (quota: check with  get_quotas )"
df -h "$STAGING" | tail -1

# 1. dataset -- the full variation0 is what TRAINING needs (3,960 episodes, ~137 GB).
#    Not the 14-episode eval subset; that one is for scripts/wait_and_eval_all.sh elsewhere.
if [ ! -f "$STAGING/rlbench_selfgen_512_aug_wide.tar" ]; then
  huggingface-cli download "$HF_USER/rlbench_selfgen_512_aug_wide" --repo-type dataset \
      --local-dir "$STAGING/_dl/rlbench_selfgen_512_aug_wide"
  tar -cf "$STAGING/rlbench_selfgen_512_aug_wide.tar" -C "$STAGING/_dl" rlbench_selfgen_512_aug_wide
  rm -rf "$STAGING/_dl/rlbench_selfgen_512_aug_wide"
fi

# 2. base model -- straight from the Wan org, never re-uploaded.
if [ ! -f "$STAGING/Wan2.2-TI2V-5B.tar" ]; then
  huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir "$STAGING/_dl/Wan2.2-TI2V-5B"
  tar -cf "$STAGING/Wan2.2-TI2V-5B.tar" -C "$STAGING/_dl" Wan2.2-TI2V-5B
  rm -rf "$STAGING/_dl/Wan2.2-TI2V-5B"
fi

# 3. warm start -- weights only, 12 GB. See README: this is a WARM START, not a DeepSpeed resume.
if [ ! -f "$STAGING/step5000.ckpt" ]; then
  huggingface-cli download "$HF_USER/fusion0_wide_F1" step5000.ckpt --local-dir "$STAGING/_dl"
  mv "$STAGING/_dl/step5000.ckpt" "$STAGING/step5000.ckpt"
fi

rmdir "$STAGING/_dl" 2>/dev/null || true
ls -lh "$STAGING" | grep -vE "^total|^d"
