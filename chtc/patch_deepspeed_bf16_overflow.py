#!/usr/bin/env python
"""Make DeepSpeed skip non-finite gradient steps under bf16 (idempotent).

Problem
-------
`DeepSpeedZeroOptimizer.step()` only checks for overflow when the dtype is fp16:

    if self.dtype == torch.float16:
        self.check_overflow()

Under bf16 there is no loss scaler, so a non-finite gradient is never detected
and never skipped. It falls through to `scaled_global_norm()`, and from there
the two grad-norm paths behave differently:

* cpu_offload   -> `complete_grad_norm_calculation_for_cpu_offload` returns the
                   *Python int* `-1` sentinel. `scaled_global_norm` wraps it with
                   `torch.tensor(-1)`, which is int64, and
                   `torch.linalg.vector_norm` raises
                   "Expected a floating point or complex tensor as input. Got Long".
                   -> the run dies.
* non-offload   -> `mask_nan_or_inf_with_val_inplace` uses
                   `torch.tensor(-1.0, dtype=torch.float)`: same sentinel, right
                   dtype, so no crash. But `unscale_and_clip_grads` then computes
                   `clip = (-1 + 1e-6)/clip_grad` and clamps it to 1.0, so the
                   NaN gradient is applied verbatim.
                   -> the weights silently become NaN and stay that way.

Both were reproduced on deepspeed 0.16.9 with a NaN injected into the loss.

Fix
---
Run the existing overflow check for bf16 as well, which makes `step()` take the
branch it already has for fp16: zero the grads, reset the cpu buffers, return.
This is safe because `local_overflow` is already maintained under bf16
(`update_overflow_tracker_for_param_grad` is called unconditionally on the
cpu_offload path) and bf16 uses a static `LossScaler(1.0)` whose
`update_scale()` is a no-op.

Usage
-----
    python scripts/patch_deepspeed_bf16_overflow.py            # apply
    python scripts/patch_deepspeed_bf16_overflow.py --check    # report status
    python scripts/patch_deepspeed_bf16_overflow.py --revert   # restore backup

Re-run this after any `pip install/upgrade deepspeed`, which overwrites it.
"""

import argparse
import shutil
import sys

MARKER = "ActionImages patch (deepspeed bf16 overflow)"

OLD = """        # First compute norm for all group so we know if there is overflow
        if self.dtype == torch.float16:
            self.check_overflow()
"""

NEW = f"""        # First compute norm for all group so we know if there is overflow
        # --- {MARKER} ---
        # Upstream only checks overflow for fp16. Under bf16 there is no loss
        # scaler, so a non-finite gradient is never skipped: it reaches
        # scaled_global_norm(), where the cpu-offload branch returns a Python int
        # -1 sentinel -> torch.tensor(-1) is int64 -> linalg.vector_norm raises
        # "Got Long" (and the non-offload branch instead applies the NaN grad,
        # silently poisoning the weights). Safe because local_overflow is already
        # tracked for bf16 and bf16 uses a static LossScaler(1.0) whose
        # update_scale() is a no-op, so this just skips the step like fp16 does.
        # Managed by scripts/patch_deepspeed_bf16_overflow.py
        if self.dtype in (torch.float16, torch.bfloat16):
            self.check_overflow()
"""


def target_path():
    try:
        import deepspeed.runtime.zero.stage_1_and_2 as m
    except ImportError:
        sys.exit("deepspeed is not importable in this interpreter")
    return m.__file__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    path = target_path()
    backup = path + ".orig"
    src = open(path).read()
    patched = MARKER in src

    if args.check:
        print(f"{path}\n  patched = {patched}\n  backup  = {backup if __import__('os').path.exists(backup) else 'none'}")
        return 0

    if args.revert:
        import os
        if not os.path.exists(backup):
            sys.exit(f"no backup at {backup}")
        shutil.copy(backup, path)
        print(f"reverted {path} from {backup}")
        return 0

    if patched:
        print(f"already patched: {path}")
        return 0
    if src.count(OLD) != 1:
        sys.exit(
            f"expected exactly one overflow-check block in {path}, found {src.count(OLD)}.\n"
            "The upstream source changed -- re-read step() before patching."
        )
    import os
    if not os.path.exists(backup):
        shutil.copy(path, backup)
        print(f"backup written: {backup}")
    open(path, "w").write(src.replace(OLD, NEW))
    print(f"patched: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
