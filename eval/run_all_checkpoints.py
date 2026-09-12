"""Parallel, resumable launcher for the complete 9-checkpoint evaluation matrix."""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OFFICIAL = (
    "official_step125750",
    REPO.parent / "starVLA/playground/Pretrained_models/anyeZHY/ActionImages/step125750.ckpt",
    512,
)
ARMS = [
    (f"{arm}_step{step}", REPO / f"outputs/{arm}__seed42/checkpoint-{step}/step{step}.ckpt", 256)
    for arm in ("arm0", "arm1")
    for step in (1000, 1500, 2000, 2500)
]


def run_job(gpu, job, out, logs):
    tag, ckpt, res = job
    log_path = logs / f"{tag}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(REPO)
    cmd = [
        sys.executable, str(REPO / "eval/eval_all_masks.py"),
        "--ckpt", str(ckpt), "--tag", tag, "--res", str(res),
        "--output", str(out), "--resume",
    ]
    started = time.time()
    with open(log_path, "a", buffering=1) as log:
        log.write(f"\nLAUNCH gpu={gpu} cmd={' '.join(cmd)}\n")
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    return {
        "tag": tag, "gpu": gpu, "returncode": proc.returncode,
        "seconds": round(time.time() - started, 2), "log": str(log_path),
    }


def serial_queue(gpu, jobs, out, logs):
    results = []
    for job in jobs:
        result = run_job(gpu, job, out, logs)
        results.append(result)
        print(json.dumps(result), flush=True)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=str(REPO / "comparisons_all_masks"))
    args = p.parse_args()
    out = Path(args.output)
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    # 512 is isolated on GPU 0; distribute the eight 256 jobs across GPUs 1..3.
    queues = {0: [OFFICIAL], 1: ARMS[0::3], 2: ARMS[1::3], 3: ARMS[2::3]}
    all_results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(serial_queue, gpu, jobs, out, logs) for gpu, jobs in queues.items()]
        for fut in as_completed(futures):
            all_results.extend(fut.result())
    with open(out / "launcher_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    failures = [x for x in all_results if x["returncode"]]
    print(f"ALL_JOBS_FINISHED jobs={len(all_results)} failures={len(failures)}", flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
